"""Build a molecule-level pricing table from real CMS-derived data.

Source: data/raw/leaderboard.csv from the open Cost Plus Arbitrage dataset
(github.com/AgastyaDhar/costplus-arbitrage, CC-BY-4.0). Its columns used here
come from three CMS releases fetched 2026-07-16:
  * nadac_per_unit  - CMS NADAC (effective 2026-07-15), per strength/NDC group
  * partd_per_unit, Tot_Dsg_Unts - CMS Medicare Part D Spending by Drug,
    generic-name "Overall" rows (molecule level; every strength shares them)
  * medicaid_units  - CMS State Drug Utilization Data units, per strength

Part D figures exist only per molecule, so the decision unit is the molecule.
Per-strength NADAC values are collapsed two ways:
  * nadac_mix: weighted by the real Medicaid strength mix (primary floor)
  * nadac_max: the highest-cost strength (conservative floor, no mix needed)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "leaderboard.csv"

# A Part D / NADAC per-unit ratio outside this band almost always means the
# two sources count different units (e.g. mL vs. bottle). Such rows are
# dropped and counted, not repaired.
RATIO_BAND = (0.5, 50.0)


# The source joins NADAC to Part D by name tokens, which can merge different
# formulations (e.g. carvedilol ER capsules into carvedilol). A molecule whose
# strength-level NADAC spans more than this ratio is flagged as heterogeneous.
HETERO_SPREAD = 5.0


def load_molecules(path: Path = RAW, homogeneous_only: bool = False) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path)
    audit = {"raw_rows": len(raw)}

    df = raw.dropna(subset=["nadac_per_unit", "partd_per_unit", "Tot_Dsg_Unts"])
    df = df[(df.nadac_per_unit > 0) & (df.partd_per_unit > 0) & (df.Tot_Dsg_Unts > 0)]
    audit["rows_with_nadac_and_partd"] = len(df)

    # Rows sharing identical Part D per-unit spend and units are strengths of
    # one Part D molecule (that is how the source joins them).
    df = df.assign(molecule_key=df.partd_per_unit.round(8).astype(str) + "|"
                   + df.Tot_Dsg_Unts.round(1).astype(str))

    def collapse(g: pd.DataFrame) -> pd.Series:
        rep = g[g.is_partd_molecule_row] if g.is_partd_molecule_row.any() else g
        w = g.medicaid_units.fillna(0).to_numpy()
        n = g.nadac_per_unit.to_numpy()
        mix_known = w.sum() > 0
        return pd.Series({
            "molecule": rep.drug_term.iloc[0],
            "unit": g.nadac_pricing_unit.mode().iloc[0],
            "n_units_types": g.nadac_pricing_unit.nunique(),
            "n_strengths": len(g),
            "baseline": g.partd_per_unit.iloc[0],
            "volume": g.Tot_Dsg_Unts.iloc[0],
            "nadac_mix": float((w * n).sum() / w.sum()) if mix_known else float(n.mean()),
            "mix_source": "medicaid_units" if mix_known else "unweighted_mean",
            "nadac_max": float(n.max()),
            "nadac_min": float(n.min()),
        })

    mol = df.groupby("molecule_key").apply(collapse, include_groups=False).reset_index(drop=True)
    audit["molecules"] = len(mol)

    mixed_units = mol.n_units_types > 1
    audit["dropped_mixed_unit_types"] = int(mixed_units.sum())
    mol = mol[~mixed_units]

    ratio = mol.baseline / mol.nadac_mix
    out = (ratio < RATIO_BAND[0]) | (ratio > RATIO_BAND[1])
    audit["dropped_ratio_outliers"] = int(out.sum())
    audit["dropped_ratio_outlier_names"] = sorted(mol.loc[out, "molecule"].tolist())
    mol = mol[~out].copy()

    mol["heterogeneous"] = mol.nadac_max / mol.nadac_min > HETERO_SPREAD
    audit["heterogeneous_molecules"] = int(mol.heterogeneous.sum())
    audit["heterogeneous_underwater"] = int((mol.heterogeneous & (mol.baseline < mol.nadac_mix)).sum())
    if homogeneous_only:
        mol = mol[~mol.heterogeneous].copy()

    mol["spend"] = mol.baseline * mol.volume
    mol = mol.sort_values("spend", ascending=False).reset_index(drop=True)
    audit["molecules_used"] = len(mol)
    audit["mix_from_medicaid"] = int((mol.mix_source == "medicaid_units").sum())
    audit["baseline_spend_usd"] = float(mol.spend.sum())
    audit["baseline_below_nadac_mix"] = int((mol.baseline < mol.nadac_mix).sum())
    return mol, audit


if __name__ == "__main__":
    m, a = load_molecules()
    for k, v in a.items():
        print(k, v)
    print(m.head(15).to_string())
    print((m.baseline / m.nadac_mix).describe())
