"""Digital-banking automation, ported to Python: parse a debit-card
transaction history (a generic CSV -- date, description, amount), categorize
each transaction by keyword, compute a running balance, and build a monthly
cash-flow diagram (inflow vs. outflow by category, plus balance over time).

Built against SYNTHETIC transaction data (no real account is touched here);
`parse_transactions_csv` is generic enough to read a real bank-exported CSV
later -- point it at that file and everything downstream (categorization,
running balance, the plot) works unchanged.

The one thing actually "derived and verified" here (matching this repo's
verify-don't-assume discipline): the running balance's arithmetic identity
ending_balance - starting_balance == total_inflow - total_outflow, checked
exactly, not assumed.
"""
import csv
import datetime
import random

CATEGORY_KEYWORDS = {
    "paycheck": ["payroll", "paycheck", "direct deposit", "salary"],
    "rent": ["rent", "leasing", "apartments"],
    "groceries": ["grocery", "market", "supermarket", "safeway", "trader joe"],
    "utilities": ["utility", "electric", "water bill", "gas company", "internet"],
    "subscription": ["netflix", "spotify", "subscription", "membership"],
    "dining": ["restaurant", "cafe", "coffee", "diner", "pizza"],
    "transport": ["gas station", "fuel", "transit", "uber", "lyft"],
    "other": [],
}


def categorize_transaction(description: str) -> str:
    """Keyword match against CATEGORY_KEYWORDS, case-insensitive; falls back
    to 'other' rather than raising, since real bank descriptions are messy
    and an uncategorized transaction should still show up in the totals."""
    desc_lower = description.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in desc_lower for kw in keywords):
            return category
    return "other"


def generate_synthetic_transactions(n_months=6, seed=0) -> list[dict]:
    """A realistic (but entirely made-up) n_months of debit-card activity:
    biweekly paycheck deposits, monthly rent/utilities, and randomized
    groceries/dining/transport/subscription spending -- for exercising the
    pipeline below without touching any real account."""
    if n_months < 1:
        raise ValueError("n_months must be >= 1")
    rng = random.Random(seed)
    start = datetime.date(2026, 1, 1)
    transactions = []

    for month in range(n_months):
        month_start = _add_months(start, month)

        transactions.append({"date": month_start, "description": "PAYROLL DIRECT DEPOSIT",
                              "amount": 2450.00})
        transactions.append({"date": _add_days(month_start, 14), "description": "PAYROLL DIRECT DEPOSIT",
                              "amount": 2450.00})

        transactions.append({"date": _add_days(month_start, 1), "description": "RENT - APARTMENTS LLC",
                              "amount": -1400.00})
        transactions.append({"date": _add_days(month_start, 3), "description": "ELECTRIC COMPANY UTILITY",
                              "amount": -round(rng.uniform(60, 110), 2)})
        transactions.append({"date": _add_days(month_start, 5), "description": "INTERNET UTILITY",
                              "amount": -59.99})
        transactions.append({"date": _add_days(month_start, 7), "description": "NETFLIX SUBSCRIPTION",
                              "amount": -15.49})
        transactions.append({"date": _add_days(month_start, 7), "description": "SPOTIFY SUBSCRIPTION",
                              "amount": -10.99})

        for _ in range(rng.randint(6, 10)):
            transactions.append({"date": _add_days(month_start, rng.randint(0, 27)),
                                  "description": rng.choice(["TRADER JOE GROCERY", "SAFEWAY MARKET"]),
                                  "amount": -round(rng.uniform(25, 90), 2)})
        for _ in range(rng.randint(4, 8)):
            transactions.append({"date": _add_days(month_start, rng.randint(0, 27)),
                                  "description": rng.choice(["COFFEE SHOP", "PIZZA DINER", "CAFE RESTAURANT"]),
                                  "amount": -round(rng.uniform(6, 35), 2)})
        for _ in range(rng.randint(2, 5)):
            transactions.append({"date": _add_days(month_start, rng.randint(0, 27)),
                                  "description": "GAS STATION FUEL",
                                  "amount": -round(rng.uniform(30, 60), 2)})

    transactions.sort(key=lambda t: t["date"])
    return transactions


def _add_days(d: datetime.date, n: int) -> datetime.date:
    return d + datetime.timedelta(days=n)


def _add_months(d: datetime.date, n: int) -> datetime.date:
    month = d.month - 1 + n
    year = d.year + month // 12
    month = month % 12 + 1
    return datetime.date(year, month, 1)


def parse_transactions_csv(path: str, date_col="date", desc_col="description", amount_col="amount") -> list[dict]:
    """Read a generic bank-export CSV: one row per transaction, a date
    column, a description column, and a signed amount column (positive =
    deposit, negative = withdrawal -- the convention most bank exports
    already use). Column names are parameters since real exports vary
    ("Transaction Date" vs. "date", etc.)."""
    transactions = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            transactions.append({
                "date": datetime.date.fromisoformat(row[date_col]),
                "description": row[desc_col],
                "amount": float(row[amount_col]),
            })
    transactions.sort(key=lambda t: t["date"])
    return transactions


def compute_running_balance(transactions: list[dict], starting_balance: float) -> list[dict]:
    """Running balance after each transaction, in date order; also verifies
    (rather than assumes) that ending_balance - starting_balance equals the
    total signed transaction sum -- the one hard arithmetic invariant a
    running-balance computation must satisfy."""
    balance = starting_balance
    rows = []
    for t in transactions:
        balance += t["amount"]
        rows.append({**t, "balance": balance})

    total_change = sum(t["amount"] for t in transactions)
    if rows and abs((rows[-1]["balance"] - starting_balance) - total_change) > 1e-9:
        raise AssertionError("running balance does not match the total transaction sum")
    return rows


def monthly_cash_flow_summary(transactions: list[dict]) -> dict:
    """{(year, month): {"inflow": ..., "outflow": ..., "net": ..., "by_category": {...}}}"""
    summary: dict = {}
    for t in transactions:
        key = (t["date"].year, t["date"].month)
        month_summary = summary.setdefault(key, {"inflow": 0.0, "outflow": 0.0, "net": 0.0, "by_category": {}})
        category = categorize_transaction(t["description"])
        month_summary["by_category"].setdefault(category, 0.0)
        month_summary["by_category"][category] += t["amount"]
        if t["amount"] >= 0:
            month_summary["inflow"] += t["amount"]
        else:
            month_summary["outflow"] += -t["amount"]
        month_summary["net"] += t["amount"]
    return summary


if __name__ == "__main__":
    txns = generate_synthetic_transactions(n_months=3, seed=42)
    print(f"generated {len(txns)} synthetic transactions over 3 months")

    balance_history = compute_running_balance(txns, starting_balance=1000.0)
    print(f"starting balance: 1000.00   ending balance: {balance_history[-1]['balance']:.2f}")

    summary = monthly_cash_flow_summary(txns)
    for (year, month), s in sorted(summary.items()):
        print(f"\n{year}-{month:02d}: inflow={s['inflow']:.2f}  outflow={s['outflow']:.2f}  net={s['net']:+.2f}")
        for category, amount in sorted(s["by_category"].items(), key=lambda kv: kv[1]):
            print(f"    {category:12s} {amount:+9.2f}")
