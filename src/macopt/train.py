"""Train the cost-spike model, evaluate it, and save the serving artifact.

Locally: writes models/ and outputs/cost_risk_results.json.
On Vertex AI: AIP_MODEL_DIR (a gs:// path set by Vertex) receives the model.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import lightgbm as lgb

from macopt.costrisk import (
    FEATURES,
    HORIZON,
    SPIKE,
    TEST_START,
    TRAIN_END,
    buffer_backtest,
    build_examples,
    fit_and_evaluate,
    weekly_panel,
)

ROOT = Path(__file__).resolve().parents[2]


def main() -> dict:
    panel = weekly_panel()
    ex = build_examples(panel)
    res, te = fit_and_evaluate(ex)
    res["panel_rows"] = len(panel)
    res["panel_ndcs"] = int(panel.ndc.nunique())
    res["examples"] = len(ex)
    res["buffer_backtest"] = buffer_backtest(te)
    res["config"] = {"spike": SPIKE, "horizon_weeks": HORIZON,
                     "train_end": str(TRAIN_END.date()), "test_start": str(TEST_START.date()),
                     "features": FEATURES}

    # Final serving model: refit on all labeled examples (train + test periods)
    final = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                               min_child_samples=200, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, random_state=7, verbose=-1)
    final.fit(ex[FEATURES], ex.y)
    mdir = ROOT / "models"
    mdir.mkdir(exist_ok=True)
    final.booster_.save_model(str(mdir / "cost_risk_lgbm.txt"))
    meta = {"features": FEATURES, "spike": SPIKE, "horizon_weeks": HORIZON,
            "trained_on_examples": len(ex), "test_roc_auc": res["lightgbm"]["roc_auc"],
            "test_pr_auc": res["lightgbm"]["pr_auc"]}
    (mdir / "cost_risk_meta.json").write_text(json.dumps(meta, indent=2))

    gcs = os.environ.get("AIP_MODEL_DIR")
    if gcs:  # Vertex AI custom job: Cloud Storage is FUSE-mounted at /gcs/<bucket>
        dest = Path("/gcs") / gcs.removeprefix("gs://")
        dest.mkdir(parents=True, exist_ok=True)
        for f in ("cost_risk_lgbm.txt", "cost_risk_meta.json"):
            shutil.copyfile(mdir / f, dest / f)  # FUSE mount rejects chmod
        res["vertex_model_dir"] = gcs

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    (out / "cost_risk_results.json").write_text(json.dumps(res, indent=2, default=float))
    return res


if __name__ == "__main__":
    r = main()
    print(json.dumps({k: r[k] for k in ("examples", "test_base_rate", "momentum_rule",
                                        "logistic", "lightgbm", "buffer_backtest")}, indent=2))
