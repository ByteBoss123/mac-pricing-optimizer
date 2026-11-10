# Decision memo: how to deliver a 20% generic-spend guarantee

**For:** client account and pharmacy network leads. **Basis:** 297 generic molecules, $12.73B of Medicare Part D generic spend, CMS acquisition costs (NADAC). The contract terms below are scenarios, not any real client's terms.

## The decision

The client is owed a 20% reduction in generic spend. There are three ways to deliver it, and they spread the pain differently.

| Option | What happens | Main risk |
|---|---|---|
| **A. Cut everything 20%** | Simple to explain and implement | 56 drugs end up reimbursed below what pharmacies pay for them, a $212M shortfall. Expect network complaints and access problems. |
| **B. Spread the cut as evenly as possible** | Every drug moves; no drug is cut more than 21.7% | Touches nearly the whole list, so it is the most work to implement and explain. |
| **C. Concentrate on a few drugs** | Re-price 46 drugs: lift 34 that pharmacies currently lose money on, and cut 12 high-volume staples (atorvastatin, rosuvastatin, losartan, amlodipine and others) by up to 60% | A handful of very visible changes on common drugs. |

**Recommendation:** start from **C**. It meets the guarantee with the least change to the list. It is also the only option that fixes the 34 drugs pharmacies currently lose money on. Talk to the network about the 12 deep cuts before they go live.

## What changes the answer

* **The guarantee level.** Each extra point of guarantee pushes the deepest required cut up about one point. Contracts above roughly 53% cannot be met at all without breaking the pharmacy floors.
* **How deep one drug may go.** Capping any single drug at 30% instead of 60% raises the number of cut drugs from 12 to 53. Fewer changes means deeper changes; there is no free option.
* **Cost spikes after the rates are set.** Acquisition costs move weekly. In 2022, about 9% of drugs rose 10% or more within a quarter. A forecasting model flags which drugs are likely to spike. Spending the same safety margin on the flagged drugs, instead of spreading it evenly, cut "rate fell below cost" incidents by 22% at a 5% margin and by 38% at a 10% margin (back-tested on 2022 data the model never saw).

## Speed, risk and scale

* **Speed.** Re-pricing the full list takes about 15 milliseconds through the service, so scenarios can be tested live in a client meeting.
* **Risk.** Every rate is checked against the acquisition-cost floor before it ships. If a requested guarantee is impossible, the service says so and returns the highest achievable level instead of a broken price list.
* **Scale.** These models are small. A full strength-level list with tens of thousands of products would need a full Gurobi license, but the same formulation applies.

## Caveats

Medicare gross spend stands in for current client rates, and the Part D year is earlier than the July 2026 cost data. Results are directional until rerun on the client's actual claims and contract terms.
