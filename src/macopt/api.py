"""HTTP service: run the reimbursement optimizer and score cost-spike risk.

Endpoints
  GET  /health                 liveness + model metadata
  POST /optimize               re-price the drug list under a contract policy
  POST /cost-risk/score        probability each NDC's cost rises >=10% in 13 weeks
"""
from __future__ import annotations

import json
import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Literal

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from macopt.data import load_molecules
from macopt.model import Policy, max_feasible_guarantee, solve_min_touch, solve_minimax, uniform_cut

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("macopt.api")
logging.basicConfig(level=logging.INFO, format="%(message)s")
app = FastAPI(title="Generic drug reimbursement optimizer", version="1.0.0")


@lru_cache
def molecules() -> pd.DataFrame:
    return load_molecules()[0]


@lru_cache
def cost_model() -> tuple[lgb.Booster, dict]:
    meta = json.loads((ROOT / "models" / "cost_risk_meta.json").read_text())
    return lgb.Booster(model_file=str(ROOT / "models" / "cost_risk_lgbm.txt")), meta


class OptimizeRequest(BaseModel):
    guarantee: float = Field(0.20, ge=0, le=0.9)
    max_cut: float = Field(0.60, gt=0, le=1)
    floor_markup: float = Field(0.0, ge=0, le=1)
    mode: Literal["lp", "mip"] = "mip"
    top_n: int = Field(15, ge=1, le=297)


class RiskRecord(BaseModel):
    ndc: str
    r4: float
    r13: float
    r26: float
    r52: float
    vol52: float
    n_chg52: float
    wks_since_chg: float
    log_price: float
    sib_r13: float
    lab_r13: float
    lab_up_share13: float
    mkt_r13: float


class RiskRequest(BaseModel):
    records: list[RiskRecord] = Field(..., min_length=1, max_length=10000)


@app.get("/health")
def health() -> dict:
    _, meta = cost_model()
    return {"status": "ok", "molecules": len(molecules()),
            "cost_model_test_roc_auc": meta["test_roc_auc"]}


@app.post("/optimize")
def optimize(req: OptimizeRequest) -> dict:
    t0 = time.perf_counter()
    mol = molecules()
    pol = Policy(guarantee=req.guarantee, max_cut=req.max_cut, floor_markup=req.floor_markup)
    limit = max_feasible_guarantee(mol, pol)
    if req.guarantee > limit:
        raise HTTPException(422, {"error": "guarantee not reachable under floors and cut cap",
                                  "max_feasible_guarantee": round(limit, 4)})
    r = solve_min_touch(mol, pol) if req.mode == "mip" else solve_minimax(mol, pol)
    b = mol.baseline.to_numpy()
    u = mol.volume.to_numpy()
    change = pd.DataFrame({"molecule": mol.name, "baseline": b, "new_rate": r["prices"],
                           "change_pct": (r["prices"] - b) / b,
                           "annual_savings_usd": u * (b - r["prices"])})
    moved = change[~np.isclose(change.new_rate, change.baseline)]
    out = {
        "mode": req.mode,
        "baseline_spend_usd": float(u @ b),
        "new_spend_usd": r["new_spend_usd"],
        "achieved_reduction": 1 - r["new_spend_usd"] / float(u @ b),
        "molecules_repriced": len(moved),
        "uniform_cut_molecules_below_floor": uniform_cut(mol, pol)["molecules_below_floor"],
        "largest_cut": float(-change.change_pct.min()),
        "top_changes": moved.reindex(moved.annual_savings_usd.abs().sort_values(ascending=False).index)
                            .head(req.top_n).round(6).to_dict("records"),
    }
    if req.mode == "lp":
        out["max_cut_per_extra_guarantee_point"] = r["max_cut_per_guarantee_point"]
    log.info(json.dumps({"event": "optimize", "mode": req.mode, "guarantee": req.guarantee,
                         "repriced": out["molecules_repriced"],
                         "ms": round(1000 * (time.perf_counter() - t0), 1)}))
    return out


@app.post("/cost-risk/score")
def score(req: RiskRequest) -> dict:
    booster, meta = cost_model()
    X = pd.DataFrame([r.model_dump() for r in req.records])
    p = booster.predict(X[meta["features"]])
    log.info(json.dumps({"event": "score", "n": len(X), "mean_p": round(float(p.mean()), 4)}))
    return {"horizon_weeks": meta["horizon_weeks"], "spike_threshold": meta["spike"],
            "scores": [{"ndc": n, "p_spike": round(float(v), 6)} for n, v in zip(X.ndc, p, strict=True)]}
