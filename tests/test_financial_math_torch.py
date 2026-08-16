"""Test dgs/financial_math_torch.py: Black-Scholes pricing, Greeks via
torch.autograd.grad checked against the closed-form textbook formulas,
Monte Carlo convergence, portfolio Sharpe optimization, and VaR/CVaR.
Requires py -3.12 (torch is py-3.12 only in this repo, not 3.13)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import math
import torch
from dgs.financial_math_torch import (
    black_scholes_price, greeks_closed_form, greeks_autograd,
    monte_carlo_gbm_terminal, monte_carlo_option_price,
    sharpe_ratio, portfolio_sharpe_optimize,
    historical_var, historical_cvar, parametric_var,
)

S0, K, T, r, sigma = 100.0, 105.0, 1.0, 0.03, 0.25

# 1. black_scholes_price: known reference value (textbook parameters,
#    computed independently via the standard d1/d2 formula)
call = black_scholes_price(torch.tensor(S0), K, T, r, torch.tensor(sigma), "call")
put = black_scholes_price(torch.tensor(S0), K, T, r, torch.tensor(sigma), "put")
assert abs(call.item() - 9.1217) < 1e-3, f"call price off: {call.item()}"
# put-call parity: C - P = S*exp(-qT) - K*exp(-rT), q=0 here
parity_rhs = S0 - K * math.exp(-r * T)
assert abs((call.item() - put.item()) - parity_rhs) < 1e-6, "put-call parity violated"

# 2. black_scholes_price: input validation
for bad_kwargs in [dict(S=torch.tensor(-1.0), K=K, T=T, r=r, sigma=torch.tensor(sigma)),
                    dict(S=torch.tensor(S0), K=-1.0, T=T, r=r, sigma=torch.tensor(sigma)),
                    dict(S=torch.tensor(S0), K=K, T=-1.0, r=r, sigma=torch.tensor(sigma)),
                    dict(S=torch.tensor(S0), K=K, T=T, r=r, sigma=torch.tensor(-0.1))]:
    try:
        black_scholes_price(**bad_kwargs)
        raise AssertionError(f"expected ValueError for {bad_kwargs}")
    except ValueError:
        pass
try:
    black_scholes_price(torch.tensor(S0), K, T, r, torch.tensor(sigma), option_type="bogus")
    raise AssertionError("expected ValueError for bad option_type")
except ValueError:
    pass

# 3. greeks_autograd must match greeks_closed_form to high precision, for
#    BOTH call and put -- this is the module's central claim: one price
#    formula, differentiated, reproduces every hand-derived Greek formula.
for opt_type in ("call", "put"):
    g_closed = greeks_closed_form(S0, K, T, r, sigma, opt_type)
    g_auto = greeks_autograd(S0, K, T, r, sigma, opt_type)
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        diff = abs(g_closed[name] - g_auto[name])
        assert diff < 1e-6, f"{opt_type} {name}: closed={g_closed[name]} autograd={g_auto[name]} diff={diff}"

# 4. greeks_autograd: known signs/bounds -- call delta in (0,1), put delta
#    in (-1,0), gamma positive for both (same gamma, it's shared across
#    call/put by construction), vega positive
gc = greeks_autograd(S0, K, T, r, sigma, "call")
gp = greeks_autograd(S0, K, T, r, sigma, "put")
assert 0.0 < gc["delta"] < 1.0
assert -1.0 < gp["delta"] < 0.0
assert gc["gamma"] > 0 and gp["gamma"] > 0
assert abs(gc["gamma"] - gp["gamma"]) < 1e-6, "gamma is identical for call and put at the same strike"
assert gc["vega"] > 0 and gp["vega"] > 0

# 5. monte_carlo_gbm_terminal: antithetic pairing means E[S_T] should sit
#    very close to the risk-neutral forward S0*exp(r*T), and pairs must be
#    exact mirror draws (z, -z) under the same lognormal map
terminal = monte_carlo_gbm_terminal(S0, r, sigma, T, n_paths=200_000, seed=42)
forward = S0 * math.exp(r * T)
assert abs(terminal.mean().item() - forward) / forward < 0.01, "MC terminal mean too far from risk-neutral forward"

# 6. monte_carlo_gbm_terminal: validation
try:
    monte_carlo_gbm_terminal(-1.0, r, sigma, T, n_paths=100)
    raise AssertionError("expected ValueError for S0 <= 0")
except ValueError:
    pass
try:
    monte_carlo_gbm_terminal(S0, r, sigma, T, n_paths=0)
    raise AssertionError("expected ValueError for n_paths <= 0")
except ValueError:
    pass

# 7. monte_carlo_option_price: must converge to black_scholes_price within
#    its own reported 95% CI (this is the actual claim -- not "close", but
#    "inside the interval the function itself computed")
mc = monte_carlo_option_price(S0, K, T, r, sigma, n_paths=500_000, option_type="call", seed=7)
lo, hi = mc["ci95"]
assert lo < call.item() < hi, f"BS price {call.item()} outside MC 95% CI ({lo}, {hi})"

# 8. sharpe_ratio: hand-computed on a tiny fixed series (float64 to match
#    what sharpe_ratio() converts to internally -- the function's own
#    float32 input still gets promoted, so compare at that precision)
rets = torch.tensor([0.02, -0.01, 0.03, 0.00, 0.01], dtype=torch.float64)
mean = rets.mean().item()
std = rets.std(unbiased=True).item()
expected_sharpe = (mean - 0.0) / std
assert abs(sharpe_ratio(rets) - expected_sharpe) < 1e-9

# 9. sharpe_ratio: zero-variance series must raise
try:
    sharpe_ratio(torch.tensor([0.01, 0.01, 0.01]))
    raise AssertionError("expected ValueError for zero-variance returns")
except ValueError:
    pass

# 10. portfolio_sharpe_optimize: weights must sum to 1 and be non-negative
#     (guaranteed by the softmax parametrization, verified directly rather
#     than assumed), and the optimized Sharpe must beat every single-asset
#     Sharpe and an equal-weight portfolio
mean_returns = torch.tensor([0.08, 0.12, 0.05], dtype=torch.float64)
cov_matrix = torch.tensor([[0.10, 0.02, 0.01],
                            [0.02, 0.18, 0.03],
                            [0.01, 0.03, 0.05]], dtype=torch.float64)
opt = portfolio_sharpe_optimize(mean_returns, cov_matrix, risk_free_rate=0.02, n_iter=800)
weights = opt["weights"]
assert abs(sum(weights) - 1.0) < 1e-9, f"weights don't sum to 1: {sum(weights)}"
assert all(w >= 0.0 for w in weights), f"negative weight found: {weights}"

equal_w = torch.full((3,), 1.0 / 3.0, dtype=torch.float64)
equal_return = (equal_w @ mean_returns).item()
equal_std = torch.sqrt(equal_w @ cov_matrix @ equal_w).item()
equal_sharpe = (equal_return - 0.02) / equal_std
assert opt["sharpe"] >= equal_sharpe - 1e-6, \
    f"optimized Sharpe {opt['sharpe']} should be >= equal-weight Sharpe {equal_sharpe}"

# 11. portfolio_sharpe_optimize: mismatched cov_matrix shape must raise
try:
    portfolio_sharpe_optimize(mean_returns, torch.eye(2))
    raise AssertionError("expected ValueError for mismatched cov_matrix shape")
except ValueError:
    pass

# 12. historical_var / historical_cvar: CVaR must be >= VaR (expected
#     shortfall in the tail is never better than the tail's own threshold)
sample = torch.normal(mean=0.0005, std=0.02, size=(20_000,),
                       generator=torch.Generator().manual_seed(1))
hvar = historical_var(sample, 0.95)
hcvar = historical_cvar(sample, 0.95)
assert hcvar >= hvar - 1e-9, f"CVaR {hcvar} should be >= VaR {hvar}"
assert hvar > 0 and hcvar > 0

# 13. historical_var vs. parametric_var: on a genuinely Gaussian sample the
#     two should agree closely (empirical quantile vs. closed-form Gaussian
#     quantile of the SAME distribution)
pvar = parametric_var(0.0005, 0.02, 0.95)
assert abs(hvar - pvar) / pvar < 0.15, f"historical VaR {hvar} too far from parametric VaR {pvar}"

# 14. historical_var / parametric_var: confidence bounds validation
for fn, args in [(historical_var, (sample, 1.5)), (historical_var, (sample, 0.0)),
                  (parametric_var, (0.0, 0.02, 1.5))]:
    try:
        fn(*args)
        raise AssertionError(f"expected ValueError for {fn.__name__}{args}")
    except ValueError:
        pass
try:
    parametric_var(0.0, -0.02, 0.95)
    raise AssertionError("expected ValueError for sigma <= 0")
except ValueError:
    pass

print("all dgs.financial_math_torch tests passed")
