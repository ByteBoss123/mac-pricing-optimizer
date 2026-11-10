import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from macopt.api import app
from macopt.costrisk import FEATURES, risk_buffers

ROOT = Path(__file__).resolve().parents[1]
client = TestClient(app)


def test_risk_buffers_keep_average_and_cap():
    rng = np.random.default_rng(0)
    risk = rng.beta(0.5, 5, 5000)
    for b in (0.02, 0.05, 0.10):
        mb = risk_buffers(risk, b)
        assert mb.mean() == pytest.approx(b, rel=1e-6)
        assert mb.max() <= 0.5 + 1e-12
        # higher risk never gets a smaller buffer
        order = np.argsort(risk)
        assert np.all(np.diff(mb[order]) >= -1e-12)


def test_no_lookahead_in_labels():
    """Label at origin t must use only weeks t+1..t+13."""
    from macopt.costrisk import build_examples
    weeks = pd.date_range("2020-01-01", periods=80, freq="W-WED")
    price = np.ones(80)
    price[70:] = 1.5          # jump at week index 70
    panel = pd.DataFrame({"ndc": "00000000001", "week": weeks, "price": price})
    ex = build_examples(panel)
    for _, r in ex.iterrows():
        i = weeks.get_loc(r.week)
        expected = 1.0 if (i + 1 <= 70 <= i + 13) else 0.0
        if i >= 70:
            expected = 0.0    # already at the new level, no further rise
        assert r.y == expected, (i, r.y)
        # features never see the jump before it happens
        if i < 70:
            assert r.r4 == 0 and r.r13 == 0 and r.log_price == 0


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["molecules"] > 250


def test_optimize_mip_matches_offline_result():
    r = client.post("/optimize", json={"guarantee": 0.20, "mode": "mip"})
    body = r.json()
    assert r.status_code == 200
    assert body["achieved_reduction"] >= 0.20 - 1e-9
    assert body["molecules_repriced"] == 46


def test_optimize_rejects_unreachable_guarantee():
    r = client.post("/optimize", json={"guarantee": 0.60})
    assert r.status_code == 422
    assert r.json()["detail"]["max_feasible_guarantee"] < 0.60


def test_score_endpoint():
    rec = {f: 0.0 for f in FEATURES} | {"ndc": "00093000001", "vol52": 0.05, "n_chg52": 10}
    r = client.post("/cost-risk/score", json={"records": [rec, rec | {"vol52": 0.0, "n_chg52": 0}]})
    s = r.json()["scores"]
    assert r.status_code == 200 and len(s) == 2
    assert 0 <= s[1]["p_spike"] <= 1
    meta = json.loads((ROOT / "models" / "cost_risk_meta.json").read_text())
    assert meta["features"] == FEATURES
