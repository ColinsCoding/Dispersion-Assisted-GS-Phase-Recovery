"""How close can free/open-source tools get to Mathematica-level CAS power?

Mathematica's Wolfram Language is the benchmark commercial CAS: integrated
simplification, ODE/PDE solving, special functions, arbitrary-precision
numerics, and a huge built-in knowledge base, all in one consistent language.

Open-source answers, roughly ranked by how much of that they cover:
  1. SymPy       -- pure Python CAS, what this whole repo already uses.
                    Strong at: algebra, calculus, linear algebra, basic ODEs,
                    Groebner bases, series. Weaker at: PDEs, special-function
                    breadth, performance on large symbolic expressions.
  2. SageMath    -- a "CAS of CAS-es": wraps SymPy + Maxima + Pari/GP + GAP +
                    Singular under one Python-like interface. Closest free
                    equivalent to Mathematica's breadth, but a much bigger
                    install (its own Python distribution).
  3. Maxima      -- older Lisp-based CAS, very strong ODE/PDE solver
                    (better than SymPy's `dsolve` for some nonlinear cases),
                    callable from SymPy via `sympy.maxima` glue in some
                    distros, or standalone.
  4. GNU Octave / Scilab -- numeric, not symbolic; complementary to a CAS,
                    not a replacement.

This module demonstrates SymPy capabilities that ARE Mathematica-competitive
(Groebner bases, ODE solving, series, integral transforms) so you can see
where the open-source floor actually is, rather than assuming it's low.
"""
import sympy as sp


def groebner_basis_demo():
    """Groebner bases -- the algorithm behind Mathematica's `GroebnerBasis`,
    used to solve nonlinear polynomial systems exactly."""
    x, y = sp.symbols('x y')
    # solve x^2 + y^2 = 1 and x = y exactly via Groebner elimination
    polys = [x**2 + y**2 - 1, x - y]
    basis = sp.groebner(polys, x, y, order='lex')
    return {"basis": list(basis.polys), "system": polys}


def ode_solve_demo():
    """SymPy's dsolve on a 2nd-order linear ODE -- same class of problem
    Mathematica's DSolve handles, solved here in pure Python."""
    t = sp.Symbol('t')
    y = sp.Function('y')
    ode = sp.Eq(y(t).diff(t, 2) + 4 * y(t), sp.sin(t))
    sol = sp.dsolve(ode, y(t))
    return {"ode": ode, "solution": sol}


def pde_separation_demo():
    """A separable PDE solved via SymPy's `pde_separate` -- the 1D heat
    equation, showing SymPy CAN do PDEs, just with less automation than
    Mathematica's DSolve[..., PDE]."""
    x, t = sp.symbols('x t')
    alpha = sp.Symbol('alpha', positive=True)
    u = sp.Function('u')
    pde = sp.Eq(sp.diff(u(x, t), t), alpha * sp.diff(u(x, t), x, 2))
    X = sp.Function('X')(x)
    T = sp.Function('T')(t)
    separated = sp.pde_separate(pde, u(x, t), [X, T])
    return {"pde": pde, "separated": separated}


def series_and_special_functions_demo():
    """Series expansion + special functions -- Bessel J0 around x=0, a
    Mathematica-Series[...]-equivalent computation."""
    x = sp.Symbol('x')
    bessel_series = sp.besselj(0, x).series(x, 0, 8)
    return {"bessel_J0_series": bessel_series}


def integral_transform_demo():
    """Fourier transform of a Gaussian -- closed form, symbolic, same class
    of result Mathematica's FourierTransform produces."""
    x, k = sp.symbols('x k', real=True)
    f = sp.exp(-x**2)
    F = sp.fourier_transform(f, x, k)
    return {"f": f, "F_fourier": F}


def simplify_power_demo():
    """Trig/exponential simplification chain -- the kind of multi-step
    simplify Mathematica's Simplify[] does automatically; SymPy needs you to
    pick the right simplification function (trigsimp, radsimp, etc.) more
    often than Mathematica does. That's the real gap: automation, not power."""
    x = sp.Symbol('x', real=True)
    expr = sp.sin(x)**2 + sp.cos(x)**2
    simplified = sp.simplify(expr)
    expr2 = sp.exp(sp.I * x)
    euler = sp.expand(expr2, complex=True)
    return {"trig_identity": (expr, simplified), "euler_expansion": (expr2, euler)}


def cas_capability_table():
    """Qualitative comparison: feature vs Mathematica vs SymPy vs SageMath."""
    return {
        "Polynomial algebra (Groebner, factor, resultants)": {
            "Mathematica": "built-in, fast", "SymPy": "built-in, slower on large systems",
            "SageMath": "built-in via Singular, fast",
        },
        "ODE solving": {
            "Mathematica": "very broad (DSolve)", "SymPy": "good for linear/separable; gaps in nonlinear",
            "SageMath": "wraps Maxima -- broader than SymPy alone",
        },
        "PDE solving": {
            "Mathematica": "broad, automatic", "SymPy": "manual separation-of-variables only",
            "SageMath": "still mostly manual, same gap",
        },
        "Special functions": {
            "Mathematica": "huge built-in library", "SymPy": "strong (Bessel, Legendre, hypergeometric)",
            "SageMath": "matches Mathematica closer (Pari/GP backend)",
        },
        "Arbitrary precision numerics": {
            "Mathematica": "native, seamless", "SymPy": "via mpmath, seamless",
            "SageMath": "native, very strong",
        },
        "common_lesson": "SymPy alone covers most of what a physics/engineering "
                          "workflow needs (this repo proves it). The real "
                          "Mathematica gap is PDE automation and simplification "
                          "heuristics -- SageMath closes most of that gap for free "
                          "by wrapping Maxima/Singular/Pari, at the cost of a much "
                          "bigger install.",
    }


def cas_sympy_5():
    """Five symbolic showcase results bundled as one dict, for a quick demo."""
    return {
        "Groebner": groebner_basis_demo()["basis"],
        "ODE_solution": ode_solve_demo()["solution"],
        "Bessel_series": series_and_special_functions_demo()["bessel_J0_series"],
        "Fourier_of_gaussian": integral_transform_demo()["F_fourier"],
        "Euler_expansion": simplify_power_demo()["euler_expansion"][1],
    }


if __name__ == "__main__":
    print("=== Groebner basis: x^2+y^2=1, x=y ===")
    gb = groebner_basis_demo()
    for p in gb["basis"]:
        print(f"  {p}")

    print("\n=== ODE: y'' + 4y = sin(t) ===")
    ode = ode_solve_demo()
    print(f"  {ode['solution']}")

    print("\n=== PDE separation: 1D heat equation ===")
    pde = pde_separation_demo()
    print(f"  {pde['separated']}")

    print("\n=== Bessel J0 series around x=0 ===")
    series = series_and_special_functions_demo()
    print(f"  {series['bessel_J0_series']}")

    print("\n=== Fourier transform of exp(-x^2) ===")
    ft = integral_transform_demo()
    print(f"  F[exp(-x^2)] = {ft['F_fourier']}")

    print("\n=== Simplification ===")
    sp_demo = simplify_power_demo()
    print(f"  sin^2+cos^2 -> {sp_demo['trig_identity'][1]}")
    print(f"  exp(i*x) expanded -> {sp_demo['euler_expansion'][1]}")

    print("\n=== Capability table (excerpt) ===")
    table = cas_capability_table()
    for feature, comp in table.items():
        if isinstance(comp, dict):
            print(f"  {feature}: SymPy={comp['SymPy']}")
