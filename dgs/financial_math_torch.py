"""financial_math_torch.py -- quantitative finance built on the same tool
this repo already uses for physics: PyTorch autograd. Portfolio piece for
job search into quant/financial-engineering roles (not the physics/photonics
track the rest of dgs/ targets), but it reuses the exact machinery
dgs/torch_autograd_dag.py verified: real backward-mode differentiation
through a real computation graph.

WHY AUTOGRAD FOR GREEKS: the textbook approach hand-derives Delta, Gamma,
Vega, Theta, Rho as separate closed-form formulas from the Black-Scholes
PDE. Here Delta/Gamma/Vega/Rho/Theta come from ONE forward computation
(black_scholes_price) differentiated by torch.autograd.grad -- the same
principle as computing gradients of a neural net loss, applied to an
options-pricing formula instead. greeks_closed_form() gives the textbook
formulas back so the tests can verify autograd reproduces them exactly,
not just plausibly.

Requires torch (py 3.12 here, matching this repo's existing convention).
"""

from __future__ import annotations
import math
import torch

_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _validate_option_type(option_type: str) -> None:
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def _validate_positive(**kwargs) -> None:
    for name, value in kwargs.items():
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")


def _norm_cdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF via erf -- avoids a scipy dependency this repo
    doesn't otherwise need, and stays inside torch's autograd graph."""
    return 0.5 * (1.0 + torch.erf(x / _SQRT_2))


def _norm_pdf(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * x * x) / _SQRT_2PI


# ── 1. Black-Scholes price (differentiable) ──────────────────────────────────

def black_scholes_price(S: torch.Tensor, K: float, T: float, r: float,
                         sigma: torch.Tensor, option_type: str = "call",
                         q: float = 0.0) -> torch.Tensor:
    """European option price. S and sigma may be plain floats or tensors
    with requires_grad=True -- the latter is what makes greeks_autograd()
    work. K, T, r, q are treated as fixed (non-differentiated) parameters
    here; pass them as tensors with requires_grad=True yourself if you also
    want Greeks with respect to them beyond the T/r ones greeks_autograd
    already covers.
    """
    _validate_option_type(option_type)
    _validate_positive(K=K, T=T, sigma=float(sigma) if not torch.is_tensor(sigma) else sigma.item())
    if not torch.is_tensor(S):
        S = torch.tensor(float(S))
    if float(S) <= 0:
        raise ValueError(f"S must be > 0, got {float(S)}")
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(float(sigma))

    d1 = (torch.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "call":
        return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)


# ── 2. Greeks: closed-form (textbook) vs. autograd (this module's point) ────

def greeks_closed_form(S: float, K: float, T: float, r: float, sigma: float,
                        option_type: str = "call", q: float = 0.0) -> dict:
    """The textbook analytic formulas -- kept as an independent reference so
    greeks_autograd() can be checked against something derived by hand, not
    just against itself."""
    _validate_option_type(option_type)
    _validate_positive(S=S, K=K, T=T, sigma=sigma)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    Nd1 = 0.5 * (1.0 + math.erf(d1 / _SQRT_2))
    Nd2 = 0.5 * (1.0 + math.erf(d2 / _SQRT_2))
    pdf_d1 = math.exp(-0.5 * d1 * d1) / _SQRT_2PI

    gamma = math.exp(-q * T) * pdf_d1 / (S * sigma * math.sqrt(T))
    vega = S * math.exp(-q * T) * pdf_d1 * math.sqrt(T)

    if option_type == "call":
        delta = math.exp(-q * T) * Nd1
        theta = (-S * math.exp(-q * T) * pdf_d1 * sigma / (2 * math.sqrt(T))
                  - r * K * math.exp(-r * T) * Nd2
                  + q * S * math.exp(-q * T) * Nd1)
        rho = K * T * math.exp(-r * T) * Nd2
    else:
        Nmd1 = 0.5 * (1.0 + math.erf(-d1 / _SQRT_2))
        Nmd2 = 0.5 * (1.0 + math.erf(-d2 / _SQRT_2))
        delta = -math.exp(-q * T) * Nmd1
        theta = (-S * math.exp(-q * T) * pdf_d1 * sigma / (2 * math.sqrt(T))
                  + r * K * math.exp(-r * T) * Nmd2
                  - q * S * math.exp(-q * T) * Nmd1)
        rho = -K * T * math.exp(-r * T) * Nmd2

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def greeks_autograd(S: float, K: float, T: float, r: float, sigma: float,
                     option_type: str = "call", q: float = 0.0) -> dict:
    """Delta, Gamma, Vega, Theta, Rho all fall out of ONE price formula by
    differentiating it -- no separate hand-derived formula per Greek.
    Gamma needs create_graph=True on the first grad() so the graph that
    produced Delta is itself differentiable (a second backward pass)."""
    _validate_option_type(option_type)
    _validate_positive(S=S, K=K, T=T, sigma=sigma)

    S_t = torch.tensor(float(S), dtype=torch.float64, requires_grad=True)
    sigma_t = torch.tensor(float(sigma), dtype=torch.float64, requires_grad=True)
    T_t = torch.tensor(float(T), dtype=torch.float64, requires_grad=True)
    r_t = torch.tensor(float(r), dtype=torch.float64, requires_grad=True)

    d1 = (torch.log(S_t / K) + (r_t - q + 0.5 * sigma_t * sigma_t) * T_t) / (sigma_t * torch.sqrt(T_t))
    d2 = d1 - sigma_t * torch.sqrt(T_t)
    if option_type == "call":
        price = S_t * torch.exp(-q * T_t) * _norm_cdf(d1) - K * torch.exp(-r_t * T_t) * _norm_cdf(d2)
    else:
        price = K * torch.exp(-r_t * T_t) * _norm_cdf(-d2) - S_t * torch.exp(-q * T_t) * _norm_cdf(-d1)

    delta, = torch.autograd.grad(price, S_t, create_graph=True)
    gamma, = torch.autograd.grad(delta, S_t, retain_graph=True)
    vega, = torch.autograd.grad(price, sigma_t, retain_graph=True)
    dprice_dT, = torch.autograd.grad(price, T_t, retain_graph=True)
    rho, = torch.autograd.grad(price, r_t, retain_graph=True)

    return {
        "delta": delta.item(), "gamma": gamma.item(), "vega": vega.item(),
        "theta": -dprice_dT.item(),   # theta = -dPrice/dT (decay as time passes)
        "rho": rho.item(),
    }


# ── 3. Monte Carlo pricing: vectorized GBM paths, verified against §1 ───────

def monte_carlo_gbm_terminal(S0: float, r: float, sigma: float, T: float,
                              n_paths: int, antithetic: bool = True,
                              seed: int | None = None) -> torch.Tensor:
    """Terminal prices under the risk-neutral measure via the EXACT GBM
    solution S_T = S0*exp((r - sigma^2/2)*T + sigma*sqrt(T)*Z) -- no time
    discretization needed for a European payoff, just one draw per path.
    antithetic=True pairs each Z with -Z (variance reduction: halves the
    standard error for the same path count on a symmetric payoff)."""
    _validate_positive(S0=S0, T=T, sigma=sigma)
    if n_paths <= 0:
        raise ValueError(f"n_paths must be > 0, got {n_paths}")
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    half = n_paths // 2 if antithetic else n_paths
    z = torch.randn(half, generator=gen, dtype=torch.float64)
    if antithetic:
        z = torch.cat([z, -z])
    drift = (r - 0.5 * sigma * sigma) * T
    diffusion = sigma * math.sqrt(T) * z
    return S0 * torch.exp(drift + diffusion)


def monte_carlo_option_price(S0: float, K: float, T: float, r: float, sigma: float,
                              n_paths: int = 200_000, option_type: str = "call",
                              antithetic: bool = True, seed: int | None = None) -> dict:
    """Discounted expected payoff under n_paths simulated terminal prices.
    Returns price, standard_error, and a 95% CI so the caller can judge
    convergence against black_scholes_price() rather than trust a point
    estimate blindly."""
    _validate_option_type(option_type)
    S_T = monte_carlo_gbm_terminal(S0, r, sigma, T, n_paths, antithetic, seed)
    if option_type == "call":
        payoff = torch.clamp(S_T - K, min=0.0)
    else:
        payoff = torch.clamp(K - S_T, min=0.0)
    discounted = math.exp(-r * T) * payoff

    price = discounted.mean().item()
    se = (discounted.std(unbiased=True) / math.sqrt(len(discounted))).item()
    return {"price": price, "standard_error": se,
            "ci95": (price - 1.96 * se, price + 1.96 * se), "n_paths": len(discounted)}


# ── 4. Portfolio: Sharpe-ratio optimization by gradient ascent ──────────────

def sharpe_ratio(returns: torch.Tensor, risk_free_rate: float = 0.0) -> float:
    """Per-period Sharpe ratio of a return series (no annualization -- the
    caller scales if the series isn't already at the desired frequency)."""
    returns = torch.as_tensor(returns, dtype=torch.float64)
    excess = returns - risk_free_rate
    std = excess.std(unbiased=True)
    if std.item() == 0:
        raise ValueError("returns have zero variance -- Sharpe ratio is undefined")
    return (excess.mean() / std).item()


def portfolio_sharpe_optimize(mean_returns: torch.Tensor, cov_matrix: torch.Tensor,
                               risk_free_rate: float = 0.0, n_iter: int = 500,
                               lr: float = 0.05) -> dict:
    """Long-only max-Sharpe weights via gradient ASCENT (Adam minimizing
    -Sharpe) instead of the textbook closed-form matrix solve -- weights are
    a softmax of free parameters, which enforces sum(w)=1 and w>=0 by
    construction rather than by a post-hoc projection/clip.
    """
    mean_returns = torch.as_tensor(mean_returns, dtype=torch.float64)
    cov_matrix = torch.as_tensor(cov_matrix, dtype=torch.float64)
    n = mean_returns.shape[0]
    if cov_matrix.shape != (n, n):
        raise ValueError(f"cov_matrix must be ({n}, {n}) to match mean_returns, "
                          f"got {tuple(cov_matrix.shape)}")
    if n_iter <= 0:
        raise ValueError(f"n_iter must be > 0, got {n_iter}")

    theta = torch.zeros(n, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=lr)

    for _ in range(n_iter):
        optimizer.zero_grad()
        w = torch.softmax(theta, dim=0)
        port_return = w @ mean_returns
        port_var = w @ cov_matrix @ w
        port_std = torch.sqrt(port_var)
        neg_sharpe = -(port_return - risk_free_rate) / port_std
        neg_sharpe.backward()
        optimizer.step()

    with torch.no_grad():
        w = torch.softmax(theta, dim=0)
        port_return = (w @ mean_returns).item()
        port_std = torch.sqrt(w @ cov_matrix @ w).item()
        sharpe = (port_return - risk_free_rate) / port_std

    return {"weights": w.detach().numpy().tolist(), "expected_return": port_return,
            "expected_std": port_std, "sharpe": sharpe}


# ── 5. Risk: historical / parametric VaR and CVaR ────────────────────────────

def historical_var(returns: torch.Tensor, confidence: float = 0.95) -> float:
    """Historical (empirical-quantile) Value-at-Risk, returned as a POSITIVE
    number: the loss threshold you're 95% confident you won't exceed."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    returns = torch.as_tensor(returns, dtype=torch.float64)
    q = torch.quantile(returns, 1.0 - confidence)
    return max(-q.item(), 0.0)


def historical_cvar(returns: torch.Tensor, confidence: float = 0.95) -> float:
    """Expected shortfall: mean loss in the tail BEYOND the VaR threshold --
    strictly >= VaR, and sensitive to how bad the tail is, not just where it
    starts."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    returns = torch.as_tensor(returns, dtype=torch.float64)
    var_threshold = torch.quantile(returns, 1.0 - confidence)
    tail = returns[returns <= var_threshold]
    if len(tail) == 0:
        return max(-var_threshold.item(), 0.0)
    return max(-tail.mean().item(), 0.0)


def parametric_var(mu: float, sigma: float, confidence: float = 0.95) -> float:
    """Closed-form Gaussian VaR: -(mu + sigma*z_alpha). Cross-check target
    for historical_var() when the sample really is drawn from N(mu, sigma)."""
    _validate_positive(sigma=sigma)
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    z = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(1.0 - confidence)).item()
    return max(-(mu + sigma * z), 0.0)


if __name__ == "__main__":
    S0, K, T, r, sigma = 100.0, 105.0, 1.0, 0.03, 0.25

    bs_call = black_scholes_price(torch.tensor(S0), K, T, r, torch.tensor(sigma), "call")
    print(f"Black-Scholes call price: {bs_call.item():.4f}")

    g_closed = greeks_closed_form(S0, K, T, r, sigma, "call")
    g_auto = greeks_autograd(S0, K, T, r, sigma, "call")
    print("\nGreeks -- closed-form vs. torch.autograd.grad on the SAME price formula:")
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        print(f"  {name:6s}  closed={g_closed[name]:+.6f}   autograd={g_auto[name]:+.6f}")

    mc = monte_carlo_option_price(S0, K, T, r, sigma, n_paths=500_000, option_type="call", seed=0)
    print(f"\nMonte Carlo call price: {mc['price']:.4f}  (95% CI {mc['ci95'][0]:.4f} - {mc['ci95'][1]:.4f})"
          f"  vs. Black-Scholes {bs_call.item():.4f}")

    mean_returns = torch.tensor([0.08, 0.12, 0.05])
    cov_matrix = torch.tensor([[0.10, 0.02, 0.01],
                                [0.02, 0.18, 0.03],
                                [0.01, 0.03, 0.05]])
    opt = portfolio_sharpe_optimize(mean_returns, cov_matrix, risk_free_rate=0.02)
    print(f"\nMax-Sharpe portfolio weights: {[f'{w:.3f}' for w in opt['weights']]}"
          f"  Sharpe={opt['sharpe']:.4f}")

    sample_returns = torch.normal(mean=0.0005, std=0.02, size=(2000,), generator=torch.Generator().manual_seed(1))
    hvar = historical_var(sample_returns, 0.95)
    hcvar = historical_cvar(sample_returns, 0.95)
    pvar = parametric_var(0.0005, 0.02, 0.95)
    print(f"\n95% VaR: historical={hvar:.4f}  parametric={pvar:.4f}   95% CVaR: {hcvar:.4f}")
