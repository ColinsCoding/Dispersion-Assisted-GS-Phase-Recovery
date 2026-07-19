"""
SymPy calculus + AI-style problem connector
Topics: powers -5,-3,-1,0,1,3,5; power rule; quotient rule;
chain rule; trig substitution; Dirac delta integrals; intercepts;
while-loop style training/problem engine.

Run:
    python sympy_power_rules_ai_intercepts.py
"""

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import sympy as sp

x = sp.symbols("x", real=True)

POWERS = [-5, -3, -1, 0, 1, 3, 5]


def safe_print(title: str, expr=None):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    if expr is not None:
        print(expr)


def power_rule_table():
    safe_print("POWER RULE LOOP over -5, -3, -1, 0, 1, 3, 5")
    rows = []
    for n in POWERS:
        f = x**n
        d = sp.diff(f, x)
        integ = sp.integrate(f, x)
        rows.append((n, f, d, integ))
        print(f"n={n:>2}: f={sp.sstr(f):>8} | d/dx={sp.sstr(d):>12} | integral={sp.sstr(integ)}")
    return rows


def quotient_rule_demo():
    safe_print("QUOTIENT RULE")
    u = x**3 + 2*x + 1
    v = x**2 + 1
    f = u / v
    quotient_manual = sp.simplify((sp.diff(u, x)*v - u*sp.diff(v, x)) / v**2)
    quotient_sympy = sp.simplify(sp.diff(f, x))
    print("f(x) =", f)
    print("manual quotient rule =", quotient_manual)
    print("SymPy derivative     =", quotient_sympy)
    print("match:", sp.simplify(quotient_manual - quotient_sympy) == 0)


def chain_rule_demo():
    safe_print("CHAIN RULE + GATE/GAIT IDEA")
    inner = x**2 + 3*x
    outer = sp.sin(inner)
    d = sp.diff(outer, x)
    gate = sp.diff(sp.sin(sp.Symbol("u")), sp.Symbol("u"))
    gait = sp.diff(inner, x)
    print("f(x) = sin(x^2 + 3x)")
    print("outer gate derivative cos(u):", gate)
    print("inner gait derivative:", gait)
    print("chain rule result:", d)


def trig_substitution_demo():
    safe_print("TRIG SUBSTITUTION STYLE INTEGRAL")
    expr = sp.sqrt(1 - x**2)
    result = sp.integrate(expr, x)
    print("Integral sqrt(1 - x^2) dx =")
    print(result)
    print("Interpretation: area under a semicircle; common substitution x = sin(theta).")


def dirac_delta_demo():
    safe_print("DIRAC DELTA INTEGRALS")
    a = sp.Rational(3, 2)
    expr = (x**2 + 1) * sp.DiracDelta(x - a)
    val = sp.integrate(expr, (x, -5, 5))
    print("Integral (x^2 + 1) delta(x - 3/2) from -5 to 5 =", val)
    print("Sampling rule: integral f(x) delta(x-a) dx = f(a) when a is inside interval.")


def intercepts_demo():
    safe_print("INTERCEPTS: x-intercepts and y-intercept")
    f = x**3 - 4*x
    roots = sp.solve(sp.Eq(f, 0), x)
    yint = f.subs(x, 0)
    print("f(x) =", f)
    print("x-intercepts:", [(r, 0) for r in roots])
    print("y-intercept:", (0, yint))


def translate_power_to_physics():
    safe_print("TRANSLATE POWER: MATH -> PHYSICS -> COMPUTERS")
    print("Math power x^n: growth/decay shape.")
    print("Electrical power: P = V*I = I^2*R = V^2/R.")
    print("Optical intensity: I(t) = |E(t)|^2.")
    print("AC power feature: mean((signal - mean(signal))^2).")
    print("Computer version: arrays + loops + derivatives + integrals + ML features.")


def ac_power(signal: List[float]) -> float:
    mu = sum(signal) / len(signal)
    return sum((s - mu)**2 for s in signal) / len(signal)


def synthetic_intensity_trace(n: int = 512) -> Tuple[List[float], List[float]]:
    """Make an intensity trace and hidden phase for training demos."""
    intensity = []
    phase = []
    for k in range(n):
        t = 2 * math.pi * k / n
        ph = 0.8 * math.sin(3*t) + 0.25 * math.sin(11*t)
        amp = 1.0 + 0.25 * math.cos(5*t)
        i = amp*amp + 0.05 * math.sin(17*t)
        intensity.append(i)
        phase.append(ph)
    return intensity, phase


def numerical_phase_recovery_like_loop(intensity: List[float], iterations: int = 12, C: float = 0.10) -> List[float]:
    """Toy +C feedback loop: not a full TDGSA, but a safe numerical phase scaffold."""
    n = len(intensity)
    phase = [0.0] * n
    amp = [math.sqrt(max(v, 0.0)) for v in intensity]
    for _ in range(iterations):
        # finite-difference curvature feedback; C controls smoothness correction
        new_phase = phase[:]
        for k in range(1, n - 1):
            curvature = phase[k-1] - 2*phase[k] + phase[k+1]
            gradient_intensity = amp[k+1] - amp[k-1]
            new_phase[k] = phase[k] + C * curvature + 0.01 * gradient_intensity
        phase = new_phase
    return phase


def machine_learning_feature_demo():
    safe_print("AI CONNECTOR: intensity trace -> features -> target")
    intensity, hidden_phase = synthetic_intensity_trace()
    recovered = numerical_phase_recovery_like_loop(intensity)
    features = {
        "mean_intensity": sum(intensity)/len(intensity),
        "ac_power": ac_power(intensity),
        "max_intensity": max(intensity),
        "min_intensity": min(intensity),
        "phase_span_estimate": max(recovered) - min(recovered),
    }
    for k, v in features.items():
        print(f"{k:>20}: {v:.6f}")
    print("Training idea: many traces become rows in a large matrix X; labels/phase metrics become y.")


@dataclass
class Problem:
    subject: str
    prompt: str
    solver: Callable[[], None]


def build_problem_bank() -> List[Problem]:
    return [
        Problem("power rule", "Differentiate x^n for n in [-5,-3,-1,0,1,3,5].", power_rule_table),
        Problem("quotient rule", "Differentiate (x^3+2x+1)/(x^2+1).", quotient_rule_demo),
        Problem("chain rule", "Differentiate sin(x^2+3x).", chain_rule_demo),
        Problem("trig substitution", "Integrate sqrt(1-x^2).", trig_substitution_demo),
        Problem("Dirac delta", "Evaluate integral with delta(x-3/2).", dirac_delta_demo),
        Problem("intercepts", "Find intercepts of x^3-4x.", intercepts_demo),
        Problem("AI physics", "Convert intensity traces to ML features.", machine_learning_feature_demo),
    ]


def ai_problem_loop(max_steps: int = 7):
    safe_print("while (1) AI PROBLEM CONNECTOR, bounded safely in Python")
    bank = build_problem_bank()
    step = 0
    # C-style idea: while (1) { ... if done break; }
    while True:
        problem = bank[step % len(bank)]
        print(f"\nProblem {step+1}: [{problem.subject}] {problem.prompt}")
        problem.solver()
        step += 1
        if step >= max_steps:
            print("\nBreak condition reached. In C this prevents while(1) from running forever.")
            break


def main():
    translate_power_to_physics()
    ai_problem_loop(max_steps=7)


if __name__ == "__main__":
    main()
