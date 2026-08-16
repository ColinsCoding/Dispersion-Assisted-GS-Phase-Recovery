"""Test dgs/cash_flow_diagram.py: categorization, synthetic transaction
generation, running-balance arithmetic invariant, monthly summaries, and the
generic CSV parser -- cross-checked, not just re-run against itself."""
import sys, pathlib, csv, tempfile, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs.cash_flow_diagram import (
    categorize_transaction,
    generate_synthetic_transactions,
    parse_transactions_csv,
    compute_running_balance,
    monthly_cash_flow_summary,
)

# 1. categorize_transaction: known keyword matches
assert categorize_transaction("PAYROLL DIRECT DEPOSIT") == "paycheck"
assert categorize_transaction("RENT - APARTMENTS LLC") == "rent"
assert categorize_transaction("TRADER JOE GROCERY") == "groceries"
assert categorize_transaction("NETFLIX SUBSCRIPTION") == "subscription"
assert categorize_transaction("SOME UNRECOGNIZED VENDOR XYZ") == "other"
assert categorize_transaction("payroll deposit") == "paycheck"   # case-insensitive

# 2. generate_synthetic_transactions: reproducible with a fixed seed
txns_a = generate_synthetic_transactions(n_months=2, seed=1)
txns_b = generate_synthetic_transactions(n_months=2, seed=1)
assert txns_a == txns_b
txns_diff_seed = generate_synthetic_transactions(n_months=2, seed=2)
assert txns_a != txns_diff_seed

# 3. transactions must be sorted by date
dates = [t["date"] for t in txns_a]
assert dates == sorted(dates)

# 4. n_months validation
try:
    generate_synthetic_transactions(n_months=0)
    raise AssertionError("expected ValueError for n_months=0")
except ValueError:
    pass

# 5. compute_running_balance: the hard arithmetic invariant
txns = generate_synthetic_transactions(n_months=1, seed=7)
starting_balance = 500.0
balance_history = compute_running_balance(txns, starting_balance)
total_change = sum(t["amount"] for t in txns)
assert abs((balance_history[-1]["balance"] - starting_balance) - total_change) < 1e-9

# 6. running balance step-by-step must match a manual cumulative sum
manual_balance = starting_balance
for t, row in zip(txns, balance_history):
    manual_balance += t["amount"]
    assert abs(row["balance"] - manual_balance) < 1e-9

# 7. monthly_cash_flow_summary: inflow - outflow == net, and category sums add up
summary = monthly_cash_flow_summary(txns)
for key, s in summary.items():
    assert abs((s["inflow"] - s["outflow"]) - s["net"]) < 1e-9
    assert abs(sum(s["by_category"].values()) - s["net"]) < 1e-9

# 8. paycheck category must be the only positive-amount category (by construction)
for key, s in summary.items():
    for category, amount in s["by_category"].items():
        if category == "paycheck":
            assert amount > 0
        else:
            assert amount <= 0

# 9. parse_transactions_csv: round-trip a small CSV and compare against direct construction
with tempfile.TemporaryDirectory() as tmpdir:
    csv_path = os.path.join(tmpdir, "transactions.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "description", "amount"])
        writer.writerow(["2026-01-01", "PAYROLL DIRECT DEPOSIT", "2450.00"])
        writer.writerow(["2026-01-02", "RENT - APARTMENTS LLC", "-1400.00"])
    parsed = parse_transactions_csv(csv_path)
    assert len(parsed) == 2
    assert parsed[0]["description"] == "PAYROLL DIRECT DEPOSIT"
    assert parsed[0]["amount"] == 2450.00
    assert parsed[1]["amount"] == -1400.00

print("all cash_flow_diagram tests passed")
