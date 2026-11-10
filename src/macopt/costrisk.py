"""Predict drug acquisition-cost spikes and test risk-based pricing buffers.

Data: CMS NADAC price history per NDC (11-digit), as contiguous effective-date
intervals, from github.com/mwine81/nadac (nadac.parquet, commit 3b23f28,
2023-01-10). Coverage is continuous from early 2020 to 2022-10-01.

Question: when a reimbursement rate is set at today's NADAC, which NDCs will
see their acquisition cost rise past that rate within the next 13 weeks?
A rate that falls below cost is a "breach" (the pharmacy goes underwater).

Pipeline:
  1. Weekly price panel (Wednesdays), origins every 4 weeks.
  2. Features from history only (no look-ahead): past returns, volatility,
     change counts, time since last change, sibling-package and labeler moves.
  3. Label: max NADAC over (t, t+13 weeks] >= (1 + 10%) * NADAC at t.
  4. Time split: train origins <= 2021-09-29, test origins >= 2022-01-05.
  5. Decision backtest on test origins: give every NDC the same buffer vs.
     buffers proportional to predicted risk, at equal total buffer.
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "nadac_history.parquet"
SPIKE = 0.10
HORIZON = 13
TRAIN_END = pd.Timestamp("2021-09-29")
TEST_START = pd.Timestamp("2022-01-05")
FEATURES = ["r4", "r13", "r26", "r52", "vol52", "n_chg52", "wks_since_chg",
            "log_price", "sib_r13", "lab_r13", "lab_up_share13", "mkt_r13"]


def weekly_panel(path: Path = RAW) -> pd.DataFrame:
    d = pd.read_parquet(path)
    d["effective_date"] = pd.to_datetime(d.effective_date)
    d["end_date"] = pd.to_datetime(d.end_date)
    weeks = pd.date_range("2020-01-01", "2022-09-28", freq="W-WED")
    d = d[d.end_date >= weeks[0]]
    rows = []
    for ndc, g in d.groupby("ndc", sort=False):
        idx = np.searchsorted(g.effective_date.to_numpy(), weeks.to_numpy(), side="right") - 1
        ok = idx >= 0
        price = np.full(len(weeks), np.nan)
        e = g.end_date.to_numpy()
        valid = ok.copy()
        valid[ok] = e[idx[ok]] >= weeks.to_numpy()[ok]
        price[valid] = g.nadac_per_unit.to_numpy()[idx[valid]]
        rows.append(pd.DataFrame({"ndc": ndc, "week": weeks, "price": price}))
    p = pd.concat(rows, ignore_index=True).dropna()
    return p[p.price > 0]


def build_examples(panel: pd.DataFrame) -> pd.DataFrame:
    w = panel.pivot(index="week", columns="ndc", values="price")
    lw = np.log(w)
    ret1 = lw.diff()
    chg = (ret1.abs() > 1e-9).astype(float).where(ret1.notna())
    feats = {
        "r4": lw - lw.shift(4), "r13": lw - lw.shift(13),
        "r26": lw - lw.shift(26), "r52": lw - lw.shift(52),
        "vol52": ret1.rolling(52, min_periods=52).std(),
        "n_chg52": chg.rolling(52, min_periods=52).sum(),
        "log_price": lw,
    }
    # weeks since last change
    pos = pd.DataFrame(np.arange(len(w))[:, None].repeat(w.shape[1], 1), index=w.index, columns=w.columns)
    last_chg = pos.where(chg == 1).ffill()
    feats["wks_since_chg"] = (pos - last_chg).fillna(104).clip(upper=104)
    fut_max = w[::-1].rolling(HORIZON, min_periods=HORIZON).max()[::-1].shift(-1)
    label = (fut_max >= w * (1 + SPIKE)).astype(float).where(fut_max.notna())
    fut_rise = (fut_max / w - 1)

    origins = w.index[52::4]
    long = []
    for t in origins:
        df = pd.DataFrame({k: v.loc[t] for k, v in feats.items()})
        df["y"] = label.loc[t]
        df["max_rise"] = fut_rise.loc[t]
        df["price"] = w.loc[t]
        df["week"] = t
        long.append(df)
    ex = pd.concat(long).rename_axis("ndc").reset_index()
    ex = ex.dropna(subset=["r52", "vol52", "y"])
    ex["product"] = ex.ndc.str[:9]
    ex["labeler"] = ex.ndc.str[:5]
    # sibling packages / labeler moves at the same origin (known at time t)
    ex["sib_r13"] = ex.groupby(["week", "product"]).r13.transform("mean")
    ex["lab_r13"] = ex.groupby(["week", "labeler"]).r13.transform("mean")
    ex["lab_up_share13"] = ex.assign(up=(ex.r13 > 0.01).astype(float)).groupby(
        ["week", "labeler"]).up.transform("mean")
    ex["mkt_r13"] = ex.groupby("week").r13.transform("mean")
    return ex.reset_index(drop=True)


def top_k_precision(y, s, frac=0.10):
    k = max(1, int(len(s) * frac))
    return float(np.asarray(y)[np.argsort(-np.asarray(s))[:k]].mean())


def fit_and_evaluate(ex: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    tr = ex[ex.week <= TRAIN_END]
    te = ex[ex.week >= TEST_START].copy()
    out = {"train_rows": len(tr), "test_rows": len(te),
           "train_ndcs": int(tr.ndc.nunique()), "test_ndcs": int(te.ndc.nunique()),
           "train_origins": int(tr.week.nunique()), "test_origins": int(te.week.nunique()),
           "train_base_rate": float(tr.y.mean()), "test_base_rate": float(te.y.mean())}

    # Baseline rule: "it went up recently, it will keep going up" (13-week momentum)
    rule = te.r13.to_numpy()
    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    logit.fit(tr[FEATURES], tr.y)
    gbm = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                             min_child_samples=200, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, random_state=7, verbose=-1)
    gbm.fit(tr[FEATURES], tr.y)
    te["p_logit"] = logit.predict_proba(te[FEATURES])[:, 1]
    te["p_gbm"] = gbm.predict_proba(te[FEATURES])[:, 1]

    for name, s in {"momentum_rule": rule, "logistic": te.p_logit, "lightgbm": te.p_gbm}.items():
        out[name] = {"roc_auc": float(roc_auc_score(te.y, s)),
                     "pr_auc": float(average_precision_score(te.y, s)),
                     "precision_top10pct": top_k_precision(te.y, s)}
    imp = pd.Series(gbm.booster_.feature_importance("gain"), index=FEATURES)
    out["lightgbm_gain_share"] = (imp / imp.sum()).sort_values(ascending=False).round(4).to_dict()
    # calibration by predicted-risk decile
    te["decile"] = pd.qcut(te.p_gbm.rank(method="first"), 10, labels=False)
    out["calibration_by_decile"] = te.groupby("decile").agg(
        predicted=("p_gbm", "mean"), actual=("y", "mean")).round(4).to_dict("index")
    return out, te


def risk_buffers(risk: np.ndarray, avg_buffer: float, cap: float = 0.5) -> np.ndarray:
    """Buffers proportional to predicted risk, capped, with the same simple
    (unweighted) mean as a flat buffer. No per-NDC volume is available, so
    every NDC counts equally; a price-weighted mean would let cheap generics
    take large buffers almost for free and overstate the benefit."""
    raw = risk / risk.mean()
    lo, hi = 0.0, 1e3
    for _ in range(80):
        mid = (lo + hi) / 2
        if np.minimum(raw * mid * avg_buffer, cap).mean() < avg_buffer:
            lo = mid
        else:
            hi = mid
    return np.minimum(raw * hi * avg_buffer, cap)


def buffer_backtest(te: pd.DataFrame, avg_buffers=(0.02, 0.05, 0.10)) -> list[dict]:
    """Same average buffer: flat for every NDC vs. scaled by predicted risk.
    Breach = acquisition cost rises above the set rate within 13 weeks."""
    rise = te.max_rise.to_numpy()
    rows = []
    for b in avg_buffers:
        mb = risk_buffers(te.p_gbm.to_numpy(), b)
        flat_rate, risk_rate = float((rise > b).mean()), float((rise > mb).mean())
        rows.append({
            "avg_buffer": b, "n": len(te),
            "flat_breach_rate": flat_rate, "risk_breach_rate": risk_rate,
            "relative_reduction": 1 - risk_rate / flat_rate,
            "mean_buffer_risk": float(mb.mean()),
            "share_near_zero_buffer": float((mb < 0.005).mean()),
        })
    return rows
