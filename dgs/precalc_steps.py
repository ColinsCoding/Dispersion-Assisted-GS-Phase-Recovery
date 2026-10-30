"""
Precalculus steps under Calc 2, checked one step at a time.

In plain words
--------------
Most "hard" Calc 2 problems in Griffiths or modern physics are one calculus idea plus several
precalculus moves: completing the square, a trig identity, exponent and log rules, partial
fractions, a geometric series. Mistakes happen in the precalculus moves.

``Steps`` records a derivation as a list of (skill, expression, why). Every new step is
checked by SymPy to be **equal** to the previous one (their difference simplifies to 0), so a
wrong algebra step raises ``StepError`` instead of being shown as if it were right. Steps that
are not equalities (integrating, taking a limit) are added with ``op=...``, which records the
operation and checks the result by re-doing it in SymPy.

The worked examples (functions ``example_*``) are the precalculus/Calc 2 skills behind:
  * Gaussian integrals (free wave packet, Griffiths 2.4): complete the square
  * <x> in a box (Knight 40, Griffiths 2.2): power-reduction identity + integration by parts
  * Stefan-Boltzmann (Planck spectrum): geometric series + term-by-term integration
  * Radioactive half-life (modern physics): exponent and log rules
  * Relativistic kinetic energy -> (1/2) m v^2: binomial series
  * Field of a finite line charge (Griffiths Ex. 2.2): trig substitution
  * Partial fractions for a two-state decay chain (modern physics)

Sources (IEEE style)
--------------------
[1] J. Stewart, Calculus: Early Transcendentals, 9th ed. Cengage, 2020, chs. 7 and 11.
[2] D. J. Griffiths, Introduction to Electrodynamics, 4th ed. Pearson, 2013, Ex. 2.2.
[3] D. J. Griffiths and D. F. Schroeter, Introduction to Quantum Mechanics, 3rd ed.
    Cambridge Univ. Press, 2018, secs. 2.2 and 2.4.
"""
from dataclasses import dataclass, field

import sympy as sp


class StepError(ValueError):
    """A step that is not equal to the previous one (or an operation that doesn't check out)."""


@dataclass
class Step:
    skill: str
    expr: object
    why: str
    kind: str = "="          # "=" (checked equality) or the name of an operation


@dataclass
class Steps:
    title: str
    unknown: object = None       # for equation steps: equivalent = same solution set in this symbol
    steps: list = field(default_factory=list)

    def start(self, expr, why="given"):
        self.steps = [Step("start", sp.sympify(expr), why)]
        return self

    def eq(self, skill, expr, why=""):
        """Add a step that must EQUAL the previous expression."""
        expr = sp.sympify(expr)
        prev = self.steps[-1].expr
        same = _equivalent(prev, expr, self.unknown) if isinstance(prev, sp.Equality) else _equal(prev, expr)
        if not same:
            raise StepError(f"[{self.title}] step '{skill}' is not equal to the previous step:\n  {prev}\n  != {expr}")
        self.steps.append(Step(skill, expr, why))
        return self

    def op(self, skill, expr, why, check=None, back=None):
        """Add a step produced by an operation (integrate, expand in a series, solve...).

        Verified forward (`check(prev)` must equal expr) and/or backward (`back(expr)` must
        equal prev, e.g. summing a series must return the function it came from).
        """
        if check is None and back is None:
            raise ValueError("an operation step needs a forward check or a backward check")
        expr = sp.sympify(expr)
        prev = self.steps[-1].expr
        if check is not None and not _equal(check(prev), expr):
            raise StepError(f"[{self.title}] operation '{skill}' gives {check(prev)}, not {expr}")
        if back is not None and not _equal(back(expr), prev):
            raise StepError(f"[{self.title}] operation '{skill}': undoing it gives {back(expr)}, not {prev}")
        self.steps.append(Step(skill, expr, why, kind=skill))
        return self

    @property
    def result(self):
        return self.steps[-1].expr

    def table(self):
        import pandas as pd
        return pd.DataFrame([(i, s.skill, sp.sstr(s.expr), s.why, "= checked" if s.kind == "=" else f"{s.kind} checked")
                             for i, s in enumerate(self.steps)],
                            columns=["step", "skill", "expression", "why", "verified"])

    def show(self):
        from IPython.display import Markdown, display
        display(Markdown(f"**{self.title}**"))
        for i, s in enumerate(self.steps):
            tag = "" if s.kind == "=" else f"  [{s.kind}]"
            display(Markdown(f"{i}. *{s.skill}*{tag}: {s.why}"))
            display(s.expr)


def _equivalent(a, b, unknown):
    """Two equations are equivalent if they have the same solutions for `unknown`."""
    if not isinstance(b, sp.Equality):
        return False
    if unknown is None:
        raise StepError("equation steps need Steps(..., unknown=symbol) to compare solution sets")
    sa, sb = sp.solve(a, unknown), sp.solve(b, unknown)
    return len(sa) == len(sb) and all(any(_equal(r, q) for q in sb) for r in sa)


def _equal(a, b):
    if isinstance(a, sp.Equality) or isinstance(b, sp.Equality):
        return False
    d = sp.simplify(sp.expand(a - b))
    if d == 0:
        return True
    d = sp.simplify(sp.trigsimp(sp.expand_trig(sp.powsimp(sp.expand_log(d, force=True), force=True))))
    return d == 0


# --- worked examples ------------------------------------------------------------------------
x, t, v, c, u, th = sp.symbols("x t v c u theta", real=True)
a, b, L, lam, s, z, N0, k1, k2 = sp.symbols("a b L lambda s z N_0 k_1 k_2", positive=True)
n = sp.Symbol("n", integer=True, positive=True)


def example_complete_square():
    """Exponent of a Gaussian wave packet: -a x^2 + b x  ->  -a (x - b/2a)^2 + b^2/4a, then integrate."""
    S = Steps("Gaussian integral: complete the square")
    S.start(-a * x**2 + b * x, "exponent of exp(-a x^2 + b x)")
    S.eq("factor out -a", -a * (x**2 - b / a * x), "pull the x^2 coefficient out")
    S.eq("add and subtract (b/2a)^2", -a * (x**2 - b / a * x + (b / (2 * a))**2 - (b / (2 * a))**2), "half the x-coefficient, squared")
    S.eq("perfect square", -a * (x - b / (2 * a))**2 + b**2 / (4 * a), "x^2 - px + (p/2)^2 = (x - p/2)^2")
    S.op("integrate exp(...) over all x", sp.sqrt(sp.pi / a) * sp.exp(b**2 / (4 * a)),
         "shift x -> x + b/2a, then the standard result sqrt(pi/a)",
         lambda e: sp.simplify(sp.integrate(sp.exp(e), (x, -sp.oo, sp.oo))))
    return S


def example_box_mean_x():
    """<x> for psi_1 in a box: power reduction, then integration by parts."""
    S = Steps("<x> in a box: sin^2 identity + integration by parts")
    S.start(2 / L * x * sp.sin(sp.pi * x / L)**2, "integrand x |psi_1|^2")
    S.eq("power reduction", 2 / L * x * (1 - sp.cos(2 * sp.pi * x / L)) / 2, "sin^2 A = (1 - cos 2A)/2  (precalc identity)")
    S.eq("distribute", x / L - x * sp.cos(2 * sp.pi * x / L) / L, "split into an easy term and a by-parts term")
    S.op("integrate over [0, L]", L / 2, "x/L gives L/2; by parts (u = x, dv = cos) gives 0 because sin(2 pi) = 0",
         lambda e: sp.simplify(sp.integrate(e, (x, 0, L))))
    return S


def example_by_parts_rule():
    """Integration by parts written out for the term x cos(kx), as a checked antiderivative."""
    k = sp.Symbol("k", positive=True)
    S = Steps("integration by parts: u = x, dv = cos(kx) dx")
    S.start(x * sp.sin(k * x) / k - sp.Integral(sp.sin(k * x) / k, x), "uv - integral of v du, with v = sin(kx)/k")
    S.eq("finish the easy integral", x * sp.sin(k * x) / k + sp.cos(k * x) / k**2, "integral of sin(kx)/k = -cos(kx)/k^2")
    S.op("differentiate to check", x * sp.cos(k * x), "d/dx of the antiderivative must give back the integrand",
         lambda e: sp.simplify(sp.diff(e, x)))
    return S


def _sum_geometric_back(series):
    """Sum z^3 * sum_{n>=1} e^{-nz} as a geometric series in r = e^-z, keeping the |r| < 1 branch."""
    r = sp.Symbol("r", positive=True)
    total = sp.summation(r**n, (n, 1, sp.oo))
    if isinstance(total, sp.Piecewise):
        total = total.args[0][0]          # the convergent branch, valid because 0 < e^-z < 1 for z > 0
    return sp.simplify(z**3 * total.subs(r, sp.exp(-z)))


def example_stefan_boltzmann():
    """Integral of z^3/(e^z - 1): geometric series, then term-by-term integration."""
    S = Steps("Stefan-Boltzmann: geometric series inside an integral")
    S.start(z**3 / (sp.exp(z) - 1), "Planck integrand (z = h f / k T)")
    S.eq("multiply top and bottom by e^-z", z**3 * sp.exp(-z) / (1 - sp.exp(-z)), "exponent rules")
    S.op("geometric series, |r| = e^-z < 1", sp.Sum(z**3 * sp.exp(-n * z), (n, 1, sp.oo)),
         "r/(1 - r) = r + r^2 + r^3 + ... with r = e^-z (precalc geometric series); checked by summing it back",
         back=_sum_geometric_back)
    S.op("integrate each term over [0, inf)", sp.Sum(6 / n**4, (n, 1, sp.oo)),
         "every term is positive, so sum and integral may be swapped (Tonelli); integral of z^3 e^{-nz} = 3!/n^4",
         check=lambda e: sp.Sum(sp.simplify(sp.integrate(e.function, (z, 0, sp.oo))), e.limits[0]))
    S.op("zeta(4) = pi^4/90", sp.pi**4 / 15, "6 * pi^4/90", lambda e: sp.simplify(e.doit()))
    return S


def example_half_life():
    """N = N0 e^{-lambda t} = N0/2  ->  t = ln 2 / lambda: exponent and log rules."""
    S = Steps("half-life: exponent and log rules", unknown=t)
    S.start(sp.Eq(N0 * sp.exp(-lam * t), N0 / 2), "decay law at the half-life")
    S.eq("divide both sides by N0", sp.Eq(sp.exp(-lam * t), sp.Rational(1, 2)), "")
    S.eq("take ln of both sides", sp.Eq(-lam * t, sp.log(sp.Rational(1, 2))), "ln(e^A) = A")
    S.eq("log rule ln(1/2) = -ln 2", sp.Eq(-lam * t, -sp.log(2)), "ln(1/q) = -ln q")
    S.op("solve for t", sp.log(2) / lam, "divide by -lambda",
         lambda e: sp.solve(e, t)[0])
    return S


def example_relativistic_ke():
    """K = (gamma - 1) m c^2 ~ (1/2) m v^2: binomial series with beta = v/c."""
    m_ = sp.Symbol("m", positive=True)
    beta = sp.Symbol("beta", positive=True)
    S = Steps("relativistic K -> (1/2) m v^2: binomial series")
    S.start((1 / sp.sqrt(1 - beta**2) - 1) * m_ * c**2, "K = (gamma - 1) m c^2")
    S.eq("negative exponent", ((1 - beta**2)**sp.Rational(-1, 2) - 1) * m_ * c**2, "1/sqrt(q) = q^(-1/2)")
    S.op("binomial series (1 + q)^p = 1 + p q + p(p-1)/2 q^2 + ...",
         (beta**2 / 2 + 3 * beta**4 / 8) * m_ * c**2, "q = -beta^2, p = -1/2, keep two terms",
         lambda e: sp.expand(sp.series(e, beta, 0, 6).removeO()))
    S.op("substitute beta = v/c", m_ * v**2 / 2 + 3 * m_ * v**4 / (8 * c**2),
         "leading term is the Newtonian (1/2) m v^2; the next is the first relativistic correction",
         check=lambda e: sp.expand(e.subs(beta, v / c)))
    return S


def example_trig_sub():
    """Integral of dz/(z^2 + s^2)^{3/2} (finite line charge): substitute z = s tan(theta)."""
    S = Steps("finite line charge: trig substitution z = s tan(theta)")
    S.start(1 / (z**2 + s**2)**sp.Rational(3, 2), "integrand of the field of a line charge (Griffiths Ex. 2.2)")
    S.op("substitute z = s tan(theta), dz = s sec^2(theta) d theta",
         sp.cos(th) / s**2,
         "1 + tan^2 = sec^2 collapses the root to |sec theta|^3; for -pi/2 < theta < pi/2 (the range of arctan) "
         "cos theta > 0, so |cos theta| = cos theta",
         lambda e: sp.simplify(sp.trigsimp(e.subs(z, s * sp.tan(th)) * s / sp.cos(th)**2)).subs(sp.Abs(sp.cos(th)), sp.cos(th)))
    S.op("integrate in theta", sp.sin(th) / s**2, "integral of cos = sin",
         lambda e: sp.integrate(e, th))
    S.op("back-substitute with the triangle: sin(theta) = z / sqrt(z^2 + s^2)",
         z / (s**2 * sp.sqrt(z**2 + s**2)), "opposite / hypotenuse",
         lambda e: sp.simplify(e.subs(sp.sin(th), z / sp.sqrt(z**2 + s**2))))
    S.op("differentiate to check", 1 / (z**2 + s**2)**sp.Rational(3, 2), "gives back the integrand",
         lambda e: sp.simplify(sp.diff(e, z)))
    return S


def example_partial_fractions():
    """Decay chain A -> B -> C: the Laplace-domain factor 1/((s + k1)(s + k2)) splits into two exponentials."""
    p = sp.Symbol("p", positive=True)
    S = Steps("two-step decay chain: partial fractions")
    S.start(1 / ((p + k1) * (p + k2)), "Laplace-domain factor of the A -> B -> C chain")
    S.eq("partial fractions", (1 / (k2 - k1)) * (1 / (p + k1) - 1 / (p + k2)), "A/(p + k1) + B/(p + k2); cover-up gives A = -B = 1/(k2 - k1)")
    S.op("invert each piece: 1/(p + k) <-> e^{-k t}", (sp.exp(-k1 * t) - sp.exp(-k2 * t)) / (k2 - k1),
         "the Bateman shape: rises, then decays",
         lambda e: sp.simplify(sp.inverse_laplace_transform(e, p, t).subs(sp.Heaviside(t), 1)))
    return S


EXAMPLES = [example_complete_square, example_box_mean_x, example_by_parts_rule, example_stefan_boltzmann,
            example_half_life, example_relativistic_ke, example_trig_sub, example_partial_fractions]
