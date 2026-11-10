from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from macopt.data import load_molecules
from macopt.model import (Policy, bounds, max_feasible_guarantee,
                          solve_min_touch, solve_minimax, uniform_cut)


@pytest.fixture(scope="module")
def mol():
    return load_molecules()[0]


def toy():
    # 3 molecules: two with headroom, one underwater (baseline below NADAC)
    return pd.DataFrame({
        "molecule": ["A", "B", "C"], "baseline": [1.0, 2.0, 0.5],
        "volume": [100.0, 50.0, 10.0], "nadac_mix": [0.3, 1.5, 0.6],
        "nadac_max": [0.4, 1.8, 0.7]})


def test_data_audit_counts(mol):
    m, a = load_molecules()
    assert a["molecules_used"] == len(m) > 250
    assert (m.baseline > 0).all() and (m.volume > 0).all()
    assert a["molecules"] == a["molecules_used"] + a["dropped_mixed_unit_types"] + a["dropped_ratio_outliers"]


def test_lp_meets_guarantee_and_floors(mol):
    pol = Policy()
    r = solve_minimax(mol, pol)
    b, lo, hi = bounds(mol, pol)
    u = mol.volume.to_numpy()
    assert r["feasible"]
    assert u @ r["prices"] <= (1 - pol.guarantee) * (u @ b) * (1 + 1e-9)
    assert (r["prices"] >= lo - 1e-9).all() and (r["prices"] <= hi + 1e-9).all()
    assert (r["prices"] >= mol.nadac_mix.to_numpy() - 1e-9).all()
    cuts = (b - r["prices"]) / b
    assert cuts.max() == pytest.approx(r["max_cut"], abs=1e-6)


def test_mip_untouched_stay_at_baseline(mol):
    r = solve_min_touch(mol, Policy())
    b = mol.baseline.to_numpy()
    assert r["optimal"]
    untouched = ~r["repriced_mask"]
    assert np.allclose(r["prices"][untouched], b[untouched])
    assert (r["prices"] >= mol.nadac_mix.to_numpy() - 1e-9).all()


def test_mip_touches_fewer_than_lp(mol):
    lp = solve_minimax(mol, Policy())
    mip = solve_min_touch(mol, Policy())
    lp_touched = (~np.isclose(lp["prices"], mol.baseline.to_numpy())).sum()
    assert mip["repriced"] < lp_touched


def test_dual_predicts_resolve(mol):
    base = Policy()
    a = solve_minimax(mol, base)
    b = solve_minimax(mol, replace(base, guarantee=base.guarantee + 0.01))
    assert b["max_cut"] == pytest.approx(a["max_cut"] + a["max_cut_per_guarantee_point"], abs=1e-3)


def test_infeasible_contract_flags_guarantee(mol):
    pol = Policy(guarantee=0.60)
    assert max_feasible_guarantee(mol, pol) < 0.60
    r = solve_minimax(mol, pol)
    assert not r["feasible"] and r["iis_guarantee"]


def test_toy_hand_check():
    t = toy()
    pol = Policy(guarantee=0.20, max_cut=0.60)
    r = solve_min_touch(t, pol)
    # C must be lifted to its floor (0.6); A alone (headroom 0.6*100=60) covers the
    # remaining gap: need 0.2*155=31 + 1 (C lift) = 32 dollars of savings
    assert r["repriced"] == 2 and r["prices"][2] == pytest.approx(0.6)
    assert r["prices"][1] == pytest.approx(2.0)
    u = t.volume.to_numpy()
    assert u @ r["prices"] <= 0.8 * (u @ t.baseline.to_numpy()) + 1e-9


def test_uniform_cut_counts_violations():
    t = toy()
    out = uniform_cut(t, Policy(guarantee=0.30))
    # B: 2.0*0.7=1.4 < 1.5 ; C: 0.35 < 0.6 ; A: 0.7 >= 0.3
    assert out["molecules_below_floor"] == 2
