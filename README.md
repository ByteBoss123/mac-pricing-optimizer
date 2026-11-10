# Generic Drug Reimbursement Optimization with Gurobi

Linear and mixed-integer programs that re-price a generic drug list so a health plan client's spend meets a contracted discount guarantee, without reimbursing any pharmacy below acquisition cost and without moving any drug too far in one pricing cycle. Built on real CMS data: NADAC acquisition costs, Medicare Part D spending, and Medicaid utilization.

## Problem

A pharmacy benefit manager sets maximum allowable cost (MAC) style reimbursement rates for generics. A client contract guarantees that total generic spend falls at least *D%* below the current baseline. The new rates have to:

* meet the guarantee across the whole drug list,
* stay at or above the pharmacy's acquisition cost (NADAC) for every drug,
* cut no single drug by more than a per-cycle cap, and
* lift any drug currently reimbursed below acquisition cost back up to its floor.

There are many ways to hit the same guarantee. The useful question is what each choice costs: how deep the cuts go, how many drugs get re-priced, and who absorbs the change.

## Data (real, public)

| Field | Source | Level |
|---|---|---|
| Acquisition cost per unit | CMS NADAC, effective 2026-07-15 | strength |
| Current spend per unit, annual units | CMS Medicare Part D Spending by Drug, generic-name "Overall" rows (most recent annual release as of 2026-07-16; the snapshot does not record the data year) | molecule |
| Strength volume mix | CMS State Drug Utilization Data (Medicaid units) | strength |

The CMS hosts are not reachable from the build environment, so the joined values come from the open [Cost Plus Arbitrage dataset](https://github.com/AgastyaDhar/costplus-arbitrage) (CC-BY-4.0). That dataset pulls these exact CMS files and documents its provenance in `dataset/PROVENANCE.md`. The file is committed at `data/raw/leaderboard.csv` with its SHA-256.

**Preparation** (`src/macopt/data.py`): of 2,062 rows, 1,151 have NADAC and Part D values. Part D exists only per molecule, so rows collapse into 351 molecules. Dropped: 51 molecules that mix units (tablets with mL), and 3 whose Part D to NADAC ratio falls outside 0.5x to 50x (likely unit mismatches). That leaves **297 molecules and $12.73B of Part D gross generic spend**. Each molecule's floor is NADAC weighted by its real Medicaid strength mix (Medicaid units cover 295 of the 297).

## Models (`src/macopt/model.py`)

Decision: new reimbursement per unit, *p_m*, for each molecule.

1. **Uniform cut (baseline heuristic).** Cut every drug by *D%*.
2. **Minimax LP.** Meet the guarantee while minimizing the deepest percentage cut on any one drug. The dual value of the guarantee constraint prices each extra point of guarantee.
3. **Min-touch MIP.** Meet the guarantee while re-pricing as few molecules as possible, using binary re-price flags with linking constraints. Untouched drugs stay exactly at baseline.
4. **Feasibility and IIS.** Compute the largest reachable guarantee under the floors and cut caps. For a contract beyond that, Gurobi's irreducible infeasible subsystem (IIS) shows which constraints conflict.

Base policy: 20% guarantee, floor = NADAC, 60% maximum cut per drug per cycle, underwater drugs lifted to the floor. Guarantee and cut cap are swept, since they are contract terms, not data.

## Results (base policy unless noted)

| Approach | Outcome |
|---|---|
| Uniform 20% cut | Pushes **56 molecules below acquisition cost**: a $211.6M pharmacy shortfall on $488.9M of spend |
| Minimax LP | Meets the guarantee with a **deepest cut of 21.7%**; 61 molecules at their floor, 34 lifted |
| Min-touch MIP | Meets the guarantee re-pricing **46 molecules: 34 required lifts plus only 12 discretionary cuts**. Proven optimal (0% gap), 594 variables, 926 constraints |

* **Shadow price.** Each extra point of guarantee deepens the largest required cut by 1.05 points. A fresh re-solve at 21% confirmed this (forecast 22.782%, actual 22.784%).
* **Where the savings are.** The 12 MIP cuts are high-volume staples where Part D spend runs 2.4x to 24x above NADAC: atorvastatin, abiraterone, metoprolol, rosuvastatin, omeprazole, losartan, amlodipine, tamsulosin, duloxetine, lisinopril, ezetimibe and lurasidone. Together they save $2.78B, against $140.1M spent lifting underwater drugs.
* **Breadth vs. depth tradeoff** (`outputs/tradeoff_sweep.csv`, `outputs/tradeoffs.png`). At a 20% guarantee, a 30% per-drug cap needs 53 cuts, while an 80% cap needs 8. All 30 guarantee and cap scenarios solve to proven optimality in about 6 seconds in total.
* **Contract limits.** With a 60% cut cap, the largest reachable guarantee is 53.4% (27.8% at a 30% cap). A 60% guarantee is infeasible, and the IIS isolates the guarantee against the per-drug lower bounds.
* **Floor sensitivity.** Raising the floor to NADAC +10% gives a 22.4% deepest cut and 56 re-priced molecules (44 lifts). Skipping the underwater lifts needs only 11 cuts, with a 20.6% deepest cut.

![Tradeoffs](outputs/tradeoffs.png)

## Robustness check

The source joins NADAC to Part D by drug-name tokens, which merges some formulations. For example, carvedilol ER capsules (NADAC $6.67/unit) land under carvedilol (Part D $0.064/unit). Using each molecule's highest-cost strength as the floor makes even a 0% guarantee infeasible, because of these merges. The volume-weighted floor mostly absorbs them, since the merged formulations carry little volume.

To check, 46 molecules whose strength-level NADAC spans more than 5x were flagged and removed. On the remaining **251 molecules ($9.71B)**, results match: uniform cut breaks 48 floors, deepest LP cut 20.5%, and the MIP re-prices 38 molecules (30 lifts plus 8 cuts, proven optimal). On this subset the max-strength floor also becomes usable, with a largest reachable guarantee of 46.8%.

## Limitations

* Part D "spend per unit" is gross program spending (pre-rebate, including dispensing) used as a proxy for current reimbursement. It is not a commercial plan's actual rates.
* NADAC is from July 2026 while Part D is from an earlier year, so some of the 34 "underwater" molecules may reflect acquisition-cost increases since then rather than current underpayment.
* The Medicaid strength mix stands in for the Medicare mix.
* Rates are set per molecule because Part D is only published per molecule. Real MAC lists price each strength and dosage form.
* Guarantee levels, cut caps and floor markups are scenario inputs, not any real contract's terms.

## Run

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q     # 8 tests
PYTHONPATH=src python -m macopt.run    # writes outputs/
```

Gurobi's pip package includes a size-limited license (2,000 variables and 2,000 constraints). Every model here fits within it, so no license setup is needed. CI runs the tests and the full pipeline on every push (`.github/workflows/ci.yml`).

## Layout

```
data/raw/leaderboard.csv     source snapshot (+ .sha256)
src/macopt/data.py           molecule table, filters, audit counts
src/macopt/model.py          uniform cut, minimax LP, min-touch MIP, IIS
src/macopt/run.py            scenarios, sweep, sensitivity, chart
tests/test_models.py         constraint checks, dual check, hand-solved toy case
outputs/                     results.json, molecule_prices.csv, tradeoff_sweep.csv, tradeoffs.png
```

Data attribution: Dhar, A. *Cost Plus Arbitrage Dataset* (CC-BY-4.0), derived from CMS NADAC, Medicare Part D Spending by Drug, and State Drug Utilization Data (U.S. public domain).
