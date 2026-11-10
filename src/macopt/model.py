"""Gurobi models for setting molecule-level generic reimbursement rates.

Setting: a PBM must re-price a generic drug list so a health plan client's
total generic spend lands at or below a contracted guarantee (D% below the
current baseline), without pushing any pharmacy reimbursement below
acquisition cost, and without moving any one molecule too far in one cycle.

Notation (per molecule m):
  b_m  current reimbursement per unit (Part D gross spend per unit)
  u_m  annual units (Part D dosage units)
  c_m  acquisition cost per unit (NADAC)
  f_m  pharmacy floor = c_m * (1 + floor_markup)
  p_m  new reimbursement per unit (decision)

Contract and policy parameters (guarantee, floor markup, max cut) are
scenario inputs, not data. They are swept rather than fixed.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB


@dataclass
class Policy:
    guarantee: float = 0.20        # client spend must fall >= this share vs baseline
    floor_markup: float = 0.0      # floor = NADAC * (1 + floor_markup)
    max_cut: float = 0.60          # no molecule cut more than this share in one cycle
    floor_col: str = "nadac_mix"   # or "nadac_max" (conservative)
    raise_underwater: bool = True  # lift molecules paid below floor up to the floor


def bounds(mol: pd.DataFrame, pol: Policy) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b = mol.baseline.to_numpy()
    f = mol[pol.floor_col].to_numpy() * (1 + pol.floor_markup)
    lo = np.maximum(f, b * (1 - pol.max_cut))
    hi = np.maximum(b, f) if pol.raise_underwater else b.copy()
    if not pol.raise_underwater:
        lo = np.minimum(lo, b)   # leave underwater molecules where they are
    return b, lo, hi


def _env() -> gp.Env:
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    return env


def max_feasible_guarantee(mol: pd.DataFrame, pol: Policy) -> float:
    """Largest guarantee reachable when every molecule sits at its lower bound."""
    b, lo, _ = bounds(mol, pol)
    u = mol.volume.to_numpy()
    return float(1 - (u * lo).sum() / (u * b).sum())


def uniform_cut(mol: pd.DataFrame, pol: Policy) -> dict:
    """Naive baseline: cut every molecule by the guarantee percentage."""
    b, _, _ = bounds(mol, pol)
    u = mol.volume.to_numpy()
    f = mol[pol.floor_col].to_numpy() * (1 + pol.floor_markup)
    p = b * (1 - pol.guarantee)
    below = p < f - 1e-12
    return {
        "molecules_below_floor": int(below.sum()),
        "spend_below_floor_usd": float((u * b)[below].sum()),
        "pharmacy_shortfall_usd": float((u * (f - p))[below].sum()),
    }


def solve_minimax(mol: pd.DataFrame, pol: Policy, iis_on_fail: bool = True) -> dict:
    """LP: meet the guarantee while minimizing the largest percentage cut."""
    b, lo, hi = bounds(mol, pol)
    u = mol.volume.to_numpy()
    n = len(b)
    with _env() as env, gp.Model("minimax", env=env) as m:
        p = m.addMVar(n, lb=lo, ub=hi, name="p")
        t = m.addVar(lb=0, ub=1, name="max_cut")
        m.addConstrs((p[i] + float(b[i]) * t >= float(b[i]) for i in range(n)), name="cut")
        g = m.addConstr(u @ p <= (1 - pol.guarantee) * float(u @ b), name="guarantee")
        m.setObjective(t, GRB.MINIMIZE)
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            out = {"status": int(m.Status), "feasible": False}
            if iis_on_fail and m.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
                m.computeIIS()
                out["iis_guarantee"] = bool(g.IISConstr)
                out["iis_bound_vars"] = int(sum(p.IISLB))
            return out
        pv = p.X
        return {
            "feasible": True,
            "max_cut": float(t.X),
            # dual of guarantee: change in max cut per $ of extra required savings
            "guarantee_dual_per_usd": float(-g.Pi),
            "max_cut_per_guarantee_point": float(-g.Pi * 0.01 * float(u @ b)),
            "molecules_at_floor": int((np.abs(pv - lo) < 1e-9 * np.maximum(1, lo)).sum()),
            "molecules_raised": int((pv > b + 1e-12).sum()),
            "new_spend_usd": float(u @ pv),
            "prices": pv,
        }


def solve_min_touch(mol: pd.DataFrame, pol: Policy, time_limit: float = 60) -> dict:
    """MIP: meet the guarantee while re-pricing as few molecules as possible."""
    b, lo, hi = bounds(mol, pol)
    u = mol.volume.to_numpy()
    n = len(b)
    forced = lo > b + 1e-12  # underwater molecules that must be lifted
    with _env() as env, gp.Model("min_touch", env=env) as m:
        m.setParam("TimeLimit", time_limit)
        m.setParam("MIPGap", 0)
        p = m.addMVar(n, lb=np.minimum(lo, b), ub=hi, name="p")
        z = m.addMVar(n, vtype=GRB.BINARY, name="repriced")
        # untouched molecules stay at baseline; touched ones move within bounds
        m.addConstr(b - p <= (b - lo).clip(min=0) * z, name="down_link")
        m.addConstr(p - b <= (hi - b) * z, name="up_link")
        m.addConstr(p >= lo - (lo - np.minimum(lo, b)) * (1 - z), name="touched_lo")
        if forced.any():
            m.addConstr(z[np.where(forced)[0]] == 1, name="forced")
        m.addConstr(u @ p <= (1 - pol.guarantee) * float(u @ b), name="guarantee")
        m.setObjective(z.sum(), GRB.MINIMIZE)
        m.optimize()
        if m.SolCount == 0:
            return {"feasible": False, "status": int(m.Status)}
        zv = z.X > 0.5
        pv = p.X
        cuts = (b - pv) / b
        return {
            "feasible": True,
            "optimal": m.Status == GRB.OPTIMAL,
            "mip_gap": float(m.MIPGap),
            "repriced": int(zv.sum()),
            "repriced_forced_raise": int(forced.sum()),
            "repriced_spend_share": float((u * b)[zv].sum() / (u * b).sum()),
            "largest_cut": float(cuts.max()),
            "new_spend_usd": float(u @ pv),
            "num_vars": int(m.NumVars),
            "num_constrs": int(m.NumConstrs),
            "prices": pv,
            "repriced_mask": zv,
        }


def policy_dict(pol: Policy) -> dict:
    return asdict(pol)
