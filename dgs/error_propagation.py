"""Propagation of errors -- how measurement uncertainty flows through a formula.

Every measured quantity has an uncertainty; when you combine measurements in a
formula, those uncertainties combine too. To first order the rule is the
derivative applied to uncertainty -- the Jacobian maps input sigmas to an output
sigma:

    sigma_f^2 = g^T Sigma g ,   g_i = df/dx_i ,   Sigma = input covariance

For INDEPENDENT inputs Sigma is diagonal and this is the familiar
"add-in-quadrature" rule. The partial derivatives here are taken numerically
(central differences, the same engine as dgs.numerical_methods), so propagation
works for ANY f you can evaluate -- no hand-derived formula needed.

Three things, cross-checked against each other:
  * propagate()     -- first-order (linear) propagation via the numerical Jacobian
  * propagate_mc()  -- Monte Carlo: sample the inputs, push through f, take the std
  * closed-form helpers (sum, product, power) -- the textbook rules, for sanity

The first-order and Monte-Carlo answers agree when f is roughly linear over a few
sigma -- and disagree (instructively) when it is not. NumPy only. Education.
"""

import numpy as np


def jacobian(f, values, h=1e-6):
    """Gradient [df/dx_0, df/dx_1, ...] at `values` by central differences (O(h^2)).
    `f` takes a 1-D array of inputs and returns a scalar."""
    values = np.asarray(values, float)
    g = np.zeros_like(values)
    for i in range(len(values)):
        step = np.zeros_like(values)
        step[i] = h * max(1.0, abs(values[i]))      # scale the step to the value
        g[i] = (f(values + step) - f(values - step)) / (2 * step[i])
    return g


def _covariance(sigmas, cov, n):
    if cov is not None:
        return np.asarray(cov, float)
    if sigmas is None:
        raise ValueError("give either sigmas (independent) or a full cov matrix")
    return np.diag(np.asarray(sigmas, float) ** 2)


def propagate(f, values, sigmas=None, cov=None, h=1e-6):
    """First-order error propagation. Returns (f_value, sigma_f).

    sigma_f^2 = g^T Sigma g, with g the numerical gradient of f at `values` and
    Sigma the input covariance (diagonal from `sigmas`, or pass a full `cov`).
    Exact when f is linear; a good approximation when f is smooth over ~1 sigma."""
    values = np.asarray(values, float)
    g = jacobian(f, values, h)
    Sigma = _covariance(sigmas, cov, len(values))
    var = float(g @ Sigma @ g)
    return float(f(values)), np.sqrt(max(var, 0.0))


def propagate_mc(f, values, sigmas=None, cov=None, n=200_000, seed=0):
    """Monte-Carlo propagation: draw n input samples ~ N(values, Sigma), evaluate f
    on each, and return (mean, std). The ground truth the linear formula approximates.
    Captures nonlinearity and skew that the first-order rule misses."""
    values = np.asarray(values, float)
    Sigma = _covariance(sigmas, cov, len(values))
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(values, Sigma, size=n)
    # try a vectorized call f(columns); fall back to a per-sample loop
    try:
        fs = np.asarray(f(samples.T), float)
        if fs.shape != (n,):
            raise ValueError("not vectorized")
    except Exception:
        fs = np.array([f(s) for s in samples])
    return float(np.mean(fs)), float(np.std(fs))


# ── the textbook closed forms (for teaching + sanity-checking) ──────
def add_in_quadrature(*sigmas):
    """Uncertainty of a SUM or DIFFERENCE: absolute sigmas add in quadrature.
    sigma_f = sqrt(sigma_a^2 + sigma_b^2 + ...)  for f = a +/- b +/- ..."""
    return float(np.sqrt(sum(s ** 2 for s in sigmas)))


def product_rule(value, terms):
    """Uncertainty of a PRODUCT/QUOTIENT: RELATIVE sigmas add in quadrature.
    `terms` is a list of (x_i, sigma_i); returns sigma_f for f ~ prod x_i^(+/-1).
    sigma_f/|f| = sqrt( sum (sigma_i/x_i)^2 )."""
    rel = np.sqrt(sum((s / x) ** 2 for x, s in terms))
    return float(abs(value) * rel)


def power_rule(value, x, sigma_x, n):
    """Uncertainty of f = x^n: the relative error is multiplied by |n|.
    sigma_f/|f| = |n| * sigma_x/|x|."""
    return float(abs(value) * abs(n) * sigma_x / abs(x))


# ── object-oriented propagation: a value that carries its own uncertainty ──
class Measurement:
    """A measured value that carries its uncertainty and propagates it through
    arithmetic automatically (operator overloading). `Measurement(2, 0.1) *
    Measurement(5, 0.2)` returns the product with its sigma already combined.

    First-order rules, assuming the two operands are INDEPENDENT:
      +,- : absolute sigmas add in quadrature
      *,/ : relative sigmas add in quadrature
      **n : relative sigma scales by |n|
    (Correlation is not tracked -- e.g. x*x is treated as independent and
    overestimates; use propagate() with a cov matrix for correlated inputs.)"""

    __slots__ = ("value", "sigma")

    def __init__(self, value, sigma=0.0):
        self.value = float(value)
        self.sigma = float(abs(sigma))

    @staticmethod
    def _coerce(o):
        return o if isinstance(o, Measurement) else Measurement(o, 0.0)

    def __add__(self, o):
        o = self._coerce(o)
        return Measurement(self.value + o.value, np.hypot(self.sigma, o.sigma))

    def __sub__(self, o):
        o = self._coerce(o)
        return Measurement(self.value - o.value, np.hypot(self.sigma, o.sigma))

    def __mul__(self, o):
        o = self._coerce(o)
        v = self.value * o.value
        return Measurement(v, abs(v) * _rel_quad(self, o))

    def __truediv__(self, o):
        o = self._coerce(o)
        v = self.value / o.value
        return Measurement(v, abs(v) * _rel_quad(self, o))

    def __pow__(self, n):
        v = self.value ** n
        rel = abs(n) * self.sigma / abs(self.value) if self.value else 0.0
        return Measurement(v, abs(v) * rel)

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, o):
        return self._coerce(o).__sub__(self)

    def __repr__(self):
        return f"{self.value:.4g} +/- {self.sigma:.2g}"


def _rel_quad(a, b):
    ra = a.sigma / abs(a.value) if a.value else 0.0
    rb = b.sigma / abs(b.value) if b.value else 0.0
    return float(np.hypot(ra, rb))


if __name__ == "__main__":
    # the flux-rule emf = B h v (Griffiths 7.13): relative errors add in quadrature
    B, h, v = 0.5, 2.0, 3.0
    sB, sh, sv = 0.01, 0.05, 0.1
    emf = lambda p: p[0] * p[1] * p[2]
    val, sig = propagate(emf, [B, h, v], [sB, sh, sv])
    _, sig_mc = propagate_mc(emf, [B, h, v], [sB, sh, sv])
    closed = product_rule(val, [(B, sB), (h, sh), (v, sv)])
    print(f"emf = B h v = {val:.3f} V")
    print(f"  sigma (linear)     = {sig:.5f}")
    print(f"  sigma (closed-form)= {closed:.5f}")
    print(f"  sigma (Monte Carlo)= {sig_mc:.5f}")
    print(f"  -> dominated by v (10% rel): emf = {val:.2f} +/- {sig:.2f} V")
