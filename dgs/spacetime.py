"""Spacetime, time symmetry, Euler's identity, and quotient rules.

The through-line: e^(i*theta) is the reason ALL of these connect.
  - Quotient rule       -> logarithmic differentiation -> d/dx ln f
  - Euler's identity    -> e^(i*pi) + 1 = 0
  - Phase factor        -> e^(i*omega*t) = time translation operator
  - Lorentz boost       -> mixes x and t the way rotation mixes x and y
  - 4D spacetime        -> Minkowski metric: ds^2 = c^2*dt^2 - dx^2 - dy^2 - dz^2
  - Time symmetry (T)   -> t -> -t reverses momentum, keeps E; e^(i*omega*t) -> e^(-i*omega*t)
  - CPT theorem         -> only CPT combined is always conserved (Lorentz invariance)
  - This repo           -> H(f) = e^(i*pi*D*f^2): D maps to time via f=t/D (Solli 2009)

Run: py -3.13 -c "from dgs.spacetime import demo; demo()"
"""
import numpy as np
import sympy as sp

# ── Quotient rules (three forms) ──────────────────────────────────────────────

def quotient_rules_sympy():
    """Three forms of the quotient rule with SymPy verification.

    Form 1: Standard      (f/g)' = (f'g - fg') / g^2
    Form 2: Log-diff      d/dx[f/g] = (f/g) * (f'/f - g'/g)
    Form 3: Negative exp  (f * g^{-1})' = f' g^{-1} + f * (-1) g^{-2} g'
                           (product rule on f * g^{-1} -- same result)
    """
    x = sp.Symbol("x")
    f, g = sp.Function("f"), sp.Function("g")

    # Form 1: standard
    expr = f(x) / g(x)
    deriv1 = sp.diff(expr, x)
    deriv1_simplified = sp.simplify(deriv1)

    # Form 2: logarithmic differentiation
    # d/dx[f/g] = (f/g) * d/dx[ln(f/g)] = (f/g)*(f'/f - g'/g)
    log_form = (f(x)/g(x)) * (sp.diff(sp.log(f(x)), x) - sp.diff(sp.log(g(x)), x))
    log_simplified = sp.simplify(log_form - deriv1_simplified)  # should be 0

    # Form 3: product rule on f * g^{-1}
    prod_form = sp.diff(f(x) * g(x)**(-1), x)
    prod_simplified = sp.simplify(prod_form - deriv1_simplified)  # should be 0

    return {
        "standard":   deriv1_simplified,
        "log_form_residual": log_simplified,    # 0 if equivalent
        "prod_form_residual": prod_simplified,  # 0 if equivalent
        "x": x, "f": f, "g": g,
    }


def quotient_rule_numeric(f, df, g, dg, x):
    """Numerical quotient rule check: (f'g - fg')/g^2.

    Parameters: f, df, g, dg are functions of x (callables or arrays).
    """
    x = np.asarray(x, float)
    fv, dfv = (f(x) if callable(f) else f), (df(x) if callable(df) else df)
    gv, dgv = (g(x) if callable(g) else g), (dg(x) if callable(dg) else dg)
    if np.any(np.abs(gv) < 1e-12):
        raise ValueError("g(x) too close to zero")
    return (dfv * gv - fv * dgv) / gv**2


# ── Euler's identity ──────────────────────────────────────────────────────────

def eulers_identity_sympy():
    """e^(i*pi) + 1 = 0 -- verified symbolically and numerically.

    Why it matters for this repo:
      H(f) = e^(i*pi*D*f^2)
      At f = 1/sqrt(D): exponent = i*pi  -> H = e^(i*pi) = -1
      The transfer function INVERTS the field at that frequency.
      This is the destructive interference point of the dispersion.
    """
    theta = sp.Symbol("theta", real=True)
    euler = sp.exp(sp.I * theta)
    euler_expanded = sp.cos(theta) + sp.I * sp.sin(theta)
    diff = sp.simplify(euler - euler_expanded)  # should be 0

    # e^(i*pi) + 1
    identity_val = complex(sp.exp(sp.I * sp.pi) + 1)

    # unit circle: |e^(i*theta)|^2 = 1
    magnitude_sq = sp.simplify(sp.Abs(euler)**2)

    # all 5 fundamental constants
    identity_str = "e^(i*pi) + 1 = 0  (Euler: e, i, pi, 1, 0)"

    return {
        "euler_formula": euler_expanded,
        "euler_minus_expanded_residual": diff,
        "identity_value": identity_val,          # ~0+0j
        "magnitude_squared": magnitude_sq,       # 1
        "identity_string": identity_str,
        "repo_connection": "H(f) = e^(i*pi*D*f^2): dispersion IS Euler's formula",
    }


def euler_unit_circle(n_points=256):
    """Points on unit circle: e^(i*theta) for theta in [0, 2*pi]."""
    theta = np.linspace(0, 2 * np.pi, n_points)
    z = np.exp(1j * theta)
    return {"theta": theta, "real": z.real, "imag": z.imag, "abs": np.abs(z)}


# ── Minkowski metric and 4-vectors ────────────────────────────────────────────

C_LIGHT = 2.998e8  # m/s

def minkowski_metric(signature="mostly_minus"):
    """Minkowski metric tensor (4x4).

    signature='mostly_minus': diag(+1, -1, -1, -1)  -- particle physics convention
    signature='mostly_plus':  diag(-1, +1, +1, +1)  -- GR convention

    ds^2 = eta_{mu nu} dx^mu dx^nu
         = c^2 dt^2 - dx^2 - dy^2 - dz^2   (mostly minus)
    """
    if signature == "mostly_minus":
        return np.diag([1.0, -1.0, -1.0, -1.0])
    elif signature == "mostly_plus":
        return np.diag([-1.0, 1.0, 1.0, 1.0])
    else:
        raise ValueError("signature must be 'mostly_minus' or 'mostly_plus'")


def spacetime_interval(event_a, event_b, c=C_LIGHT):
    """Lorentz-invariant spacetime interval between two events.

    event: (t, x, y, z) in SI units (seconds, meters)

    ds^2 = c^2*(dt)^2 - (dx)^2 - (dy)^2 - (dz)^2

    ds^2 > 0: timelike (can be causally connected)
    ds^2 = 0: lightlike (on the light cone)
    ds^2 < 0: spacelike (cannot be causally connected)
    """
    ta, xa, ya, za = event_a
    tb, xb, yb, zb = event_b
    dt, dx, dy, dz = tb - ta, xb - xa, yb - ya, zb - za
    ds2 = c**2 * dt**2 - dx**2 - dy**2 - dz**2
    if ds2 > 1e-30:
        causal = "timelike"
    elif abs(ds2) < 1e-30:
        causal = "lightlike"
    else:
        causal = "spacelike"
    return {"ds2": ds2, "causal": causal, "dt": dt, "dx": dx, "dy": dy, "dz": dz}


def lorentz_boost_x(beta):
    """Lorentz boost matrix along x-axis.

    beta = v/c,  gamma = 1/sqrt(1-beta^2)

    Mixes t and x:
      t' = gamma*(t - beta*x/c)
      x' = gamma*(x - beta*c*t)

    NOTE: same math as a rotation, but with hyperbolic trig (cosh, sinh)
    instead of circular trig (cos, sin). This is WHY time and space
    are 'the same kind of thing' in special relativity.
    """
    if abs(beta) >= 1:
        raise ValueError("|beta| must be < 1 (v < c)")
    gamma = 1.0 / np.sqrt(1 - beta**2)
    # 4x4 boost matrix (ct, x, y, z)
    L = np.eye(4)
    L[0, 0] = gamma
    L[0, 1] = -gamma * beta
    L[1, 0] = -gamma * beta
    L[1, 1] = gamma
    return {"matrix": L, "gamma": gamma, "beta": beta}


def four_velocity(v_x, v_y=0.0, v_z=0.0, c=C_LIGHT):
    """4-velocity U^mu = gamma*(c, vx, vy, vz).

    U^mu * U_mu = c^2 always (Lorentz invariant).
    """
    v2 = v_x**2 + v_y**2 + v_z**2
    if v2 >= c**2:
        raise ValueError("speed must be < c")
    gamma = 1.0 / np.sqrt(1 - v2 / c**2)
    U = np.array([gamma * c, gamma * v_x, gamma * v_y, gamma * v_z])
    eta = minkowski_metric()
    U_lower = eta @ U
    invariant = float(U @ U_lower)
    return {"U": U, "gamma": gamma, "invariant": invariant,
            "invariant_should_be_c2": c**2}


# ── Time symmetry (T), CPT ────────────────────────────────────────────────────

def time_reversal_on_phase(omega, t_max=10.0, N=512):
    """Show that T: t -> -t maps e^(i*omega*t) -> e^(-i*omega*t) = conjugate.

    This is EXACTLY why phase retrieval has the conjugate ambiguity:
      T[E(t)] = E*(-t)
    Both E and E* are related by time reversal -- a physical symmetry.
    The two-dispersion GS cannot distinguish them.
    The NN can, because real signals have a preferred time arrow
    (they start, peak, decay -- not the reverse).
    """
    if omega <= 0:
        raise ValueError("omega must be positive")
    t = np.linspace(-t_max, t_max, N)
    E_forward = np.exp(1j * omega * t)
    E_reversed = np.exp(-1j * omega * t)   # T[E] = E*(- t) at omega>0
    are_conjugates = np.allclose(E_forward, np.conj(E_reversed))
    return {
        "t": t,
        "E_forward": E_forward,
        "E_reversed": E_reversed,
        "are_time_reversal_conjugates": bool(are_conjugates),
        "connection": (
            "E and E* are related by T (time reversal). "
            "GS cannot break this symmetry. NN can because "
            "real signals have an arrow of time (causal envelope)."
        ),
    }


def cpt_summary():
    """C, P, T discrete symmetries and what they do to the fields."""
    return {
        "C (charge conjugation)": {
            "action": "particle <-> antiparticle",
            "E field": "E -> -E",
            "conserved_by": "EM, Strong  (NOT Weak -- CP violation)",
        },
        "P (parity)": {
            "action": "r -> -r  (mirror reflection)",
            "E field": "E -> -E  (polar vector flips)",
            "B field": "B -> +B  (axial vector stays)",
            "conserved_by": "EM, Strong  (NOT Weak -- Wu 1956 experiment)",
        },
        "T (time reversal)": {
            "action": "t -> -t",
            "E field": "E -> +E",
            "B field": "B -> -B  (current reversal)",
            "phase factor": "e^(i*omega*t) -> e^(-i*omega*t) = conjugate",
            "conserved_by": "EM, Strong  (NOT Weak -- K meson oscillations)",
            "BROKEN_by_weak": True,
        },
        "CPT combined": {
            "conserved_by": "ALL interactions (Lorentz invariance requires it)",
            "theorem": "CPT theorem: any Lorentz-invariant QFT conserves CPT",
        },
        "repo_connection": (
            "T-symmetry breaking is why E and E* are physically distinct. "
            "Real optical pulses are causal (E(t)=0 for t<0 approximately). "
            "E*(t) is anti-causal. The NN learns this prior implicitly."
        ),
    }


# ── Modern physics readiness check ───────────────────────────────────────────

READINESS_CHECKLIST = {
    "calculus": {
        "topics": [
            "derivatives (power, chain, product, quotient rules)",
            "integrals (substitution, parts, Gaussian integral)",
            "Taylor series",
            "partial derivatives",
            "gradient, divergence, curl",
        ],
        "test": "Can you compute d/dx[sin(x^2)] and integral of x*e^(-x^2)?",
        "needed_for": ["EM", "QM", "optics", "GS algorithm"],
    },
    "linear_algebra": {
        "topics": [
            "matrix multiply, transpose, inverse",
            "eigenvalues and eigenvectors",
            "dot product, cross product",
            "SVD (PCA uses this)",
            "unitary matrices (MZI mesh, Jones calculus)",
        ],
        "test": "Can you find eigenvalues of a 2x2 matrix by hand?",
        "needed_for": ["QM (Hermitian operators)", "PCA", "photonic_ai.py"],
    },
    "complex_numbers": {
        "topics": [
            "Euler's formula: e^(i*theta) = cos + i*sin",
            "modulus |z|, argument angle(z)",
            "complex conjugate z*",
            "phasors in AC circuits",
        ],
        "test": "What is |e^(i*pi/3)|^2 and angle(e^(i*pi/3))?",
        "needed_for": ["Fourier transform", "GS algorithm", "H(f)=e^(i*pi*D*f^2)"],
        "status": "COVERED (gs_core, dispersive_fourier_teaching, photonic_ai)",
    },
    "classical_mechanics": {
        "topics": [
            "Newton's laws, work-energy theorem",
            "Lagrangian L = T - V",
            "equations of motion from Euler-Lagrange",
            "conservation laws from symmetry (Noether's theorem)",
        ],
        "test": "Can you write the Lagrangian for a pendulum?",
        "needed_for": ["Noether -> symmetry -> CPT theorem"],
        "status": "PARTIALLY COVERED (analytical_mechanics_lagrangian.ipynb)",
    },
    "electromagnetism": {
        "topics": [
            "Coulomb's law, Gauss's law",
            "Faraday's law, Ampere's law",
            "Maxwell's equations (all 4)",
            "plane wave solution E = E0*e^(i*k*x - i*omega*t)",
            "dispersion relation omega(k)",
        ],
        "test": "Can you derive the wave equation from Maxwell's equations?",
        "needed_for": ["GS algorithm", "H(f)", "fiber dispersion"],
        "status": "COVERED (griffiths/, faradays_law.ipynb)",
    },
    "quantum_mechanics": {
        "topics": [
            "Schrodinger equation",
            "wave function, probability density |psi|^2",
            "operators, eigenvalues",
            "uncertainty principle",
            "spin, Bloch sphere",
        ],
        "test": "What does H*psi = E*psi mean physically?",
        "needed_for": ["photonic qubits", "Berry phase", "Project 5 PINN"],
        "status": "COVERED (griffiths/*, quantum_science_jalali.ipynb)",
    },
    "fourier_analysis": {
        "topics": [
            "Fourier series (periodic signals)",
            "Fourier transform (aperiodic)",
            "convolution theorem: FT[f*g] = F*G",
            "DFT, FFT algorithm",
            "transfer function H(f)",
        ],
        "test": "What is FT[e^(-t^2)]?",
        "needed_for": ["GS algorithm", "dispersion", "EVERYTHING in this repo"],
        "status": "COVERED (fourier_series.ipynb, gs_core.py)",
    },
    "special_relativity": {
        "topics": [
            "Lorentz transformation",
            "4-vectors (ct, x, y, z)",
            "Minkowski metric ds^2 = c^2 dt^2 - dx^2",
            "time dilation, length contraction",
            "E = mc^2 (mass-energy)",
        ],
        "test": "What is the spacetime interval and why is it invariant?",
        "needed_for": ["QFT", "CPT theorem", "4D spatial computing"],
        "status": "PARTIALLY COVERED (griffiths/relativity)",
    },
}


def readiness_report():
    """Print modern physics readiness check."""
    print("=" * 65)
    print("  MODERN PHYSICS READINESS ASSESSMENT")
    print("=" * 65)
    covered = [k for k, v in READINESS_CHECKLIST.items() if "COVERED" in v.get("status", "")]
    partial = [k for k, v in READINESS_CHECKLIST.items() if "PARTIALLY" in v.get("status", "")]
    missing = [k for k, v in READINESS_CHECKLIST.items() if "status" not in v]
    print(f"\n  COVERED ({len(covered)}): {', '.join(covered)}")
    print(f"  PARTIAL ({len(partial)}): {', '.join(partial)}")
    print(f"  MISSING ({len(missing)}): {', '.join(missing)}")
    print(f"\n  VERDICT: You have {len(covered)}/{len(READINESS_CHECKLIST)} topics.")
    print(f"  ENOUGH TO PUBLISH: yes -- Project 5 only needs Fourier + EM + QM.")
    print(f"  NEXT GAP: special_relativity -> needed for 4D spacetime module.")
    for topic in missing + partial:
        v = READINESS_CHECKLIST[topic]
        print(f"\n  [{topic}]  test: {v['test']}")
        print(f"    needed for: {', '.join(v['needed_for'])}")


def demo():
    print("=" * 65)
    print("  dgs/spacetime.py  --  demo")
    print("=" * 65)

    # Quotient rules
    print("\n--- Three Quotient Rules (SymPy verified) ---")
    qr = quotient_rules_sympy()
    print("  Form 1 (standard):  (f/g)' =")
    sp.pprint(qr["standard"])
    print(f"  Form 2 (log-diff residual):  {qr['log_form_residual']}  (0 = equivalent)")
    print(f"  Form 3 (product rule residual): {qr['prod_form_residual']}  (0 = equivalent)")

    # Euler's identity
    print("\n--- Euler's Identity ---")
    ei = eulers_identity_sympy()
    print(f"  {ei['identity_string']}")
    print(f"  e^(i*pi) + 1 = {ei['identity_value']:.2e}  (should be 0)")
    print(f"  |e^(i*theta)|^2 = {ei['magnitude_squared']}  (always 1)")
    print(f"  Repo: {ei['repo_connection']}")

    # Spacetime
    print("\n--- Spacetime Interval ---")
    # Two events: light signal from Earth to Moon
    moon_dist = 3.84e8  # meters
    light_travel = moon_dist / C_LIGHT  # seconds
    ev_a = (0, 0, 0, 0)
    ev_b = (light_travel, moon_dist, 0, 0)
    si = spacetime_interval(ev_a, ev_b)
    print(f"  Earth->Moon light signal: ds^2 = {si['ds2']:.3e}  ({si['causal']})")

    # Massive rocket at v=0.5c
    ev_c = (light_travel, 0.5 * C_LIGHT * light_travel, 0, 0)
    si2 = spacetime_interval(ev_a, ev_c)
    print(f"  Rocket at v=0.5c:         ds^2 = {si2['ds2']:.3e}  ({si2['causal']})")

    # Time reversal
    print("\n--- Time Reversal and Phase Conjugation ---")
    tr = time_reversal_on_phase(omega=2.0)
    print(f"  T[e^(i*omega*t)] = e^(-i*omega*t) = conjugate: {tr['are_time_reversal_conjugates']}")
    print(f"  {tr['connection']}")

    # Readiness
    print()
    readiness_report()


if __name__ == "__main__":
    demo()
