"""Run every scenario and write outputs/ (results.json, CSVs, chart)."""
from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from macopt.data import load_molecules
from macopt.model import (Policy, max_feasible_guarantee, policy_dict,
                          solve_min_touch, solve_minimax, uniform_cut)

OUT = Path(__file__).resolve().parents[2] / "outputs"
GUARANTEES = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
MAX_CUTS = [0.30, 0.40, 0.50, 0.60, 0.80]


def molecule_name(term: str) -> str:
    return re.split(r"\s\d", term, maxsplit=1)[0].strip()


def strip(r: dict) -> dict:
    return {k: v for k, v in r.items() if k not in ("prices", "repriced_mask")}


def main() -> dict:
    OUT.mkdir(exist_ok=True)
    mol, audit = load_molecules()
    mol["name"] = mol.molecule.map(molecule_name)
    base = Policy()
    res: dict = {"data_audit": audit, "base_policy": policy_dict(base)}

    res["max_feasible_guarantee"] = {
        f"max_cut_{s:.2f}": max_feasible_guarantee(mol, replace(base, max_cut=s))
        for s in MAX_CUTS + [1.0]}

    # Base case: naive uniform cut vs. the two optimizers
    res["uniform_cut"] = uniform_cut(mol, base)
    lp = solve_minimax(mol, base)
    mip = solve_min_touch(mol, base)
    res["minimax_lp"] = strip(lp)
    res["min_touch_mip"] = strip(mip)

    # Dual check: re-solve at guarantee + 1 point and compare to the dual forecast
    lp_up = solve_minimax(mol, replace(base, guarantee=base.guarantee + 0.01))
    res["dual_check"] = {"forecast_max_cut": lp["max_cut"] + lp["max_cut_per_guarantee_point"],
                         "resolved_max_cut": lp_up["max_cut"]}

    # Infeasible contract: explain it with an IIS
    res["infeasible_example"] = strip(solve_minimax(mol, replace(base, guarantee=0.60)))

    # Tradeoff sweep: guarantee x per-molecule cut cap
    rows = []
    for g in GUARANTEES:
        for s in MAX_CUTS:
            pol = replace(base, guarantee=g, max_cut=s)
            a = solve_minimax(mol, pol, iis_on_fail=False)
            b = solve_min_touch(mol, pol)
            rows.append({
                "guarantee": g, "max_cut_cap": s,
                "feasible": a["feasible"],
                "lp_max_cut": a.get("max_cut"),
                "mip_repriced": b.get("repriced"),
                "mip_discretionary_cuts": (b["repriced"] - b["repriced_forced_raise"]) if b.get("feasible") else None,
                "mip_optimal": b.get("optimal"),
                "uniform_below_floor": uniform_cut(mol, pol)["molecules_below_floor"],
            })
    sweep = pd.DataFrame(rows)
    sweep.to_csv(OUT / "tradeoff_sweep.csv", index=False)
    res["sweep_all_mip_optimal"] = bool(sweep.mip_optimal.dropna().all())

    # Sensitivity of the base case to the floor definition
    sens = {}
    for label, pol in {
        "floor_nadac_mix": base,
        "floor_nadac_max": replace(base, floor_col="nadac_max"),
        "floor_nadac_mix_plus_10pct": replace(base, floor_markup=0.10),
        "no_underwater_raise": replace(base, raise_underwater=False),
    }.items():
        a, b = solve_minimax(mol, pol), solve_min_touch(mol, pol)
        sens[label] = {"max_feasible_guarantee": max_feasible_guarantee(mol, pol),
                       "lp_max_cut": a.get("max_cut"),
                       "mip_repriced": b.get("repriced"),
                       "mip_forced_raise": b.get("repriced_forced_raise"),
                       "uniform_below_floor": uniform_cut(mol, pol)["molecules_below_floor"]}
    res["floor_sensitivity"] = sens

    # Robustness: drop molecules whose strengths span >5x in NADAC (likely
    # merged formulations from the source's name matching) and re-solve.
    clean, caudit = load_molecules(homogeneous_only=True)
    a, b = solve_minimax(clean, base), solve_min_touch(clean, base)
    res["homogeneous_only"] = {
        "molecules": len(clean),
        "baseline_spend_usd": float(clean.spend.sum()),
        "underwater": int((clean.baseline < clean.nadac_mix).sum()),
        "uniform_below_floor": uniform_cut(clean, base)["molecules_below_floor"],
        "lp_max_cut": a.get("max_cut"),
        "mip_repriced": b.get("repriced"),
        "mip_forced_raise": b.get("repriced_forced_raise"),
        "mip_optimal": b.get("optimal"),
        "max_feasible_guarantee": max_feasible_guarantee(clean, base),
        "max_feasible_guarantee_nadac_max_floor": max_feasible_guarantee(clean, replace(base, floor_col="nadac_max")),
    }

    # Per-molecule base-case prices
    u, bl = mol.volume.to_numpy(), mol.baseline.to_numpy()
    table = mol[["name", "unit", "n_strengths", "volume", "baseline", "nadac_mix", "nadac_max"]].copy()
    table["lp_price"] = lp["prices"]
    table["mip_price"] = mip["prices"]
    table["mip_repriced"] = mip["repriced_mask"]
    table["mip_change_pct"] = (mip["prices"] - bl) / bl
    table["mip_savings_usd"] = u * (bl - mip["prices"])
    table.to_csv(OUT / "molecule_prices.csv", index=False)
    cuts = table[table.mip_repriced & (table.mip_change_pct < 0)].sort_values("mip_savings_usd", ascending=False)
    res["mip_discretionary_cuts"] = cuts[["name", "baseline", "nadac_mix", "mip_price",
                                          "mip_change_pct", "mip_savings_usd"]].to_dict("records")
    raises = table[table.mip_repriced & (table.mip_change_pct > 0)]
    res["underwater_raise_cost_usd"] = float((u * (table.mip_price - bl))[raises.index].sum())

    # Chart: molecules touched vs. deepest allowed cut, per guarantee
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for g, grp in sweep[sweep.feasible].groupby("guarantee"):
        ax[0].plot(grp.max_cut_cap * 100, grp.mip_discretionary_cuts, marker="o", label=f"{g:.0%}")
    ax[0].set(xlabel="Max cut allowed per molecule (%)", ylabel="Molecules cut (MIP minimum)",
              title="Fewer re-priced drugs needs deeper cuts")
    ax[0].legend(title="Guarantee", fontsize=8)
    lp_line = sweep[(sweep.max_cut_cap == 0.6) & sweep.feasible]
    ax[1].plot(lp_line.guarantee * 100, lp_line.lp_max_cut * 100, marker="o", label="Optimized (LP minimax)")
    ax[1].plot(lp_line.guarantee * 100, lp_line.guarantee * 100, ls="--", label="Uniform cut (ignores floors)")
    ax[1].set(xlabel="Client guarantee (% below baseline)", ylabel="Deepest single-drug cut (%)",
              title="Floors force deeper cuts elsewhere")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "tradeoffs.png", dpi=150)

    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=float))
    return res


if __name__ == "__main__":
    r = main()
    print(json.dumps({k: v for k, v in r.items() if k != "mip_discretionary_cuts"}, indent=2, default=float))
    for c in r["mip_discretionary_cuts"]:
        print(c)
