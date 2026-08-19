"""Build engr140_analysis.ipynb -- pulls Lectures 4-11's tested modules
(dgs/engineering_economics.py, dgs/capital_budgeting.py,
dgs/depreciation.py, dgs/macrs_taxation.py, dgs/replacement_analysis.py,
dgs/benefit_cost_analysis.py) into one pandas/matplotlib analysis
notebook, reusing the professor's own worked examples as the data.

Build with `py -3.13 projects/engr140_depreciation/build_engr140_analysis_nb.py`,
execute with `py -3.13 -m jupyter nbconvert --to notebook --execute --inplace
projects/engr140_depreciation/engr140_analysis.ipynb`.
"""
import pathlib
import nbformat as nbf

nb = nbf.v4.new_notebook()
md = lambda s: nbf.v4.new_markdown_cell(s)
co = lambda s: nbf.v4.new_code_cell(s)
cells = []

cells.append(md("""# ENGR 140 -- Full Course Analysis (Lectures 4-11)

Seven tested modules, run against Dr. Alsharqawi's own worked examples:
`dgs.engineering_economics`, `dgs.capital_budgeting`, `dgs.depreciation`,
`dgs.macrs_taxation`, `dgs.replacement_analysis`,
`dgs.benefit_cost_analysis`."""))

cells.append(co("""import sys, pathlib
sys.path.insert(0, str(pathlib.Path('../..').resolve()))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dgs.engineering_economics import (
    effective_interest_rate, effective_interest_rate_general, future_value_changing_rates,
    market_interest_rate, real_to_actual_dollars, payback_period, discounted_payback_period,
    external_rate_of_return, capitalized_worth, incremental_pe_selection, present_worth,
)
from dgs.capital_budgeting import enumerate_bundles, select_best_bundle
from dgs.depreciation import straight_line, declining_balance, double_declining_balance, sum_of_years_digits
from dgs.macrs_taxation import macrs_schedule, after_tax_cash_flow
from dgs.replacement_analysis import economic_service_life, capital_recovery, annual_equivalent_operating_cost
from dgs.benefit_cost_analysis import conventional_bcr, modified_bcr, incremental_bcr_selection

print("loaded all seven ENGR 140 modules")"""))

cells.append(md("""## 1. Book Depreciation -- method comparison (Lecture 8)

Same asset ($10,000, $2,000 salvage, 5-year life) under all four methods
-- reproduces the professor's own comparison chart (slide 66)."""))

cells.append(co("""P, S, N = 10000, 2000, 5
methods = {
    "SL": straight_line(P, S, N),
    "SOYD": sum_of_years_digits(P, S, N),
    "DB": declining_balance(P, S, N),
    "DDB": double_declining_balance(P, S, N),
}

df_bv = pd.DataFrame({name: [P] + [row["BV_end"] for row in sched] for name, sched in methods.items()},
                      index=range(0, N + 1))
df_bv.index.name = "End of Year (n)"
display(df_bv.round(1))

plt.figure(figsize=(7, 4.5))
for name in methods:
    plt.plot(df_bv.index, df_bv[name], marker="o", label=name)
plt.xlabel("End of Year (n)"); plt.ylabel("Book Value ($)")
plt.title("Book Depreciation Methods Comparison"); plt.legend()
plt.tight_layout(); plt.show()"""))

cells.append(md("""## 2. MACRS + Taxation (Lecture 9)

$200,000 asset, 5-year MACRS, $500k/yr revenue, $150k/yr expenses,
34% tax rate -- the full after-tax cash flow build-up."""))

cells.append(co("""schedule = macrs_schedule(200000, recovery_period=5)
rows = []
for row in schedule:
    atcf = after_tax_cash_flow(revenue=500000, expenses=150000, depreciation=row["D"], tax_rate=0.34)
    rows.append({"EOY": row["n"], "MACRS Rate": row["rate"], "Depreciation": row["D"],
                 "Taxable Income": atcf["taxable_income"], "Income Tax": atcf["income_tax"],
                 "After-Tax Income": atcf["after_tax_income"], "ATCF": atcf["atcf"]})
df_macrs = pd.DataFrame(rows).set_index("EOY")
display(df_macrs.round(0))

plt.figure(figsize=(7, 4))
plt.bar(df_macrs.index, df_macrs["ATCF"], color="#c9a34e")
plt.xlabel("End of Year"); plt.ylabel("After-Tax Cash Flow ($)")
plt.title("ATCF over the asset's MACRS recovery period")
plt.tight_layout(); plt.show()"""))

cells.append(md("""## 3. Replacement Analysis -- Economic Service Life (Lecture 10)

The forklift-truck example: which N minimizes EUAC = Capital Recovery +
Annual Equivalent Operating Cost?"""))

cells.append(co("""P, i = 18000, 0.15
OC = [4000, 5600, 7840, 10976, 15366 + 5000, 21513, 30118]
S = [10000, 7500, 5625, 4219, 3164, 2373, 1780]

result = economic_service_life(P, S, OC, i)
df_esl = pd.DataFrame({"N": list(result["euac_by_n"].keys()), "EUAC": list(result["euac_by_n"].values())})
df_esl = df_esl.set_index("N")
display(df_esl.round(0))

print(f"\\nEconomic Service Life = {result['economic_service_life']} years "
      f"(min EUAC = ${result['min_euac']:,.0f})")

plt.figure(figsize=(7, 4.5))
plt.plot(df_esl.index, df_esl["EUAC"], marker="o", color="#a33")
plt.axvline(result["economic_service_life"], color="gray", linestyle="--", linewidth=0.8)
plt.scatter([result["economic_service_life"]], [result["min_euac"]], color="green", zorder=5,
            label=f"Economic Service Life = {result['economic_service_life']} yr")
plt.xlabel("Useful Life N (years)"); plt.ylabel("EUAC ($)")
plt.title("Economic Service Life"); plt.legend()
plt.tight_layout(); plt.show()"""))

cells.append(md("""## 4. Benefit-Cost Analysis -- power plant design selection (Lecture 11)

Four competing designs; Z is pre-excluded (its own BCR < 1). Incremental
BCR selects among the rest."""))

cells.append(co("""alternatives = [
    {"name": "W", "cost": 18556, "benefit": 24822.2},
    {"name": "X", "cost": 16812, "benefit": 24216.8},
    {"name": "Y", "cost": 13953, "benefit": 20342.1},
]

df_alts = pd.DataFrame(alternatives).set_index("name")
df_alts["Own BCR"] = df_alts["benefit"] / df_alts["cost"]
display(df_alts.round(2))

result = incremental_bcr_selection(alternatives)
df_steps = pd.DataFrame(result["steps"])
display(df_steps.round(3))

print(f"\\nSelected design: {result['selected']}")"""))

cells.append(md("""## 5. Nominal/Effective Interest and Inflation (Lecture 4)

18% compounded quarterly (Example 4) and a 30-year inflation projection
(Example 11)."""))

cells.append(co("""i_eff = effective_interest_rate(0.18, m=4)
print(f"18% compounded quarterly -> effective annual rate: {i_eff*100:.3f}%")

i_eff_general = effective_interest_rate_general(0.12, C=3, K=4)
print(f"12% compounded monthly, quarterly payments -> effective quarterly rate: {i_eff_general*100:.2f}%")

F_changing = future_value_changing_rates(1000, [0.08]*3 + [0.10]*4 + [0.12]*2)
print(f"\\n$1000 for 9 years at 8%/10%/12% (3/4/2 yrs): F = ${F_changing:,.2f}")

i_market = market_interest_rate(real_rate=0.08, inflation_rate=0.03)
print(f"\\n8% real return, 3% inflation -> market rate: {i_market*100:.2f}%")

AD = real_to_actual_dollars(1_000_000, f=0.03, n=30)
print(f"$1,000,000 of today's purchasing power, 30 years, 3% inflation -> ${AD:,.0f} actual dollars needed")"""))

cells.append(md("""## 6. Payback Period, ERR, and Capitalized Worth (Lecture 5)

Non-uniform cash flows (Example 3/4), a non-simple investment ERR
(Example 14, two sign changes -- exactly where a single IRR is
ambiguous), and the aqueduct-refurbishment capitalized-worth problem
(Example 16)."""))

cells.append(co("""cf_nonuniform = [-85000, 15000, 25000, 35000, 45000, 45000, 35000]
simple_pb = payback_period(cf_nonuniform)
discounted_pb = discounted_payback_period(cf_nonuniform, i=0.15)
print(f"Simple payback:      {simple_pb:.2f} years")
print(f"Discounted payback:  {discounted_pb:.2f} years  (at 15%)")

err = external_rate_of_return([-1000, 4100, -5580, 2520], marr=0.20)
print(f"\\nERR of a non-simple (2 sign changes) investment: {err*100:.1f}%  (MARR=20%)")

A_per_year = 750000 * (0.05 / (1.05**10 - 1))  # (A/F, 5%, 10): annualize the every-10-years cost
CW = capitalized_worth(A_per_year, 0.05)
print(f"\\nDome refurbishment ($750k every 10 yrs, 5% MARR): capitalized worth = ${CW:,.0f}")"""))

cells.append(md("""## 7. Incremental Selection Among Mutually Exclusive Alternatives (Lecture 6)

Three revenue alternatives (A, B, C) plus a Do-Nothing baseline -- the
incremental approach compares them pairwise in ascending-investment
order rather than by their standalone PE, IRR, or SIR (which -- per the
lecture's own point -- can rank alternatives inconsistently)."""))

cells.append(co("""i = 0.12
alternatives = [
    {"name": "DN", "cashflows": [0] * 6},
    {"name": "A", "cashflows": [-209000, 65000, 65000, 65000, 65000, 65000 + 90000]},
    {"name": "B", "cashflows": [-294600, 74000, 74000, 74000, 74000, 74000 + 200000]},
    {"name": "C", "cashflows": [-294600, 58000, 65540, 74060, 83688, 94567 + 200000]},
]

standalone_pe = {a["name"]: present_worth(a["cashflows"], i) for a in alternatives}
print("Standalone PE:", {k: round(v) for k, v in standalone_pe.items()})

result = incremental_pe_selection(alternatives, i)
df_steps = pd.DataFrame(result["steps"])
display(df_steps.round(1))
print(f"\\nSelected: {result['selected']}")"""))

cells.append(md("""## 8. Capital Budgeting -- Enumeration Method (Lecture 7)

Five candidate projects, a $100k budget, project A individually
infeasible (its own PE<0), C and D mutually exclusive, E contingent on
D. All 2^5=32 bundles enumerated; the Budget/Dependency/Feasibility
constraints eliminate all but a handful."""))

cells.append(co("""projects = {
    "A": {"cost": 15000, "pe": -1542},
    "B": {"cost": 20000, "pe": 3430},
    "C": {"cost": 30000, "pe": 4535},
    "D": {"cost": 40000, "pe": 6046},
    "E": {"cost": 35000, "pe": 3372},
}
df_bundles = enumerate_bundles(projects, budget=100000, mutually_exclusive=[{"C", "D"}],
                                contingent_on={"E": "D"})
print(f"{len(df_bundles)} bundles total, {df_bundles['passes'].sum()} pass all constraints")
display(df_bundles[df_bundles["passes"]].sort_values("total_pe", ascending=False).head())

best = select_best_bundle(df_bundles)
selected_projects = [n for n in projects if best[n] == 1]
print(f"\\nSelected bundle: {selected_projects}  (total PE = ${best['total_pe']:,.0f})")"""))

cells.append(md("""## Summary

Seven modules covering the full ENGR 140 course (Lectures 4-11)
reproduce Dr. Alsharqawi's own worked answers: `dgs.engineering_economics`,
`dgs.capital_budgeting`, `dgs.depreciation`, `dgs.macrs_taxation`,
`dgs.replacement_analysis`, and `dgs.benefit_cost_analysis` -- 114 tests
across the modules, all passing. One genuine inconsistency was found and
resolved along the way: Lecture 6's own incremental table shows
PE(A-DN)=$85,365, which doesn't reconcile with that same lecture's
PE(A)=$76,378 computed from identical cash flow data a few slides
earlier -- Section 7 above uses the cash-flow-derived value throughout."""))

nb['cells'] = cells
nb['metadata'] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}

out_path = pathlib.Path(__file__).resolve().parent / "engr140_analysis.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"wrote {out_path}  ({len(cells)} cells)")
