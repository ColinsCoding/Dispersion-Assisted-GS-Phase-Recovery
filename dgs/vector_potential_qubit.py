"""How a microwave drives a qubit: the Schrodinger equation with a vector potential, and one table of
numbers that Python, MATLAB and Lean all read from the same file.

In plain words: a charge in an electromagnetic field does not feel "the field" directly in the
Schrodinger equation; it feels the potentials. The rule (minimal coupling) is p -> p - qA:

    i hbar d psi/dt = [ (p - q A(t))^2 / 2m + V(x) ] psi            (velocity gauge)

For a field that is uniform over the atom (wavelength >> atom, the dipole approximation), multiplying
the wavefunction by the phase exp(-i q A(t) x / hbar) turns this into

    i hbar d phi/dt = [ p^2 / 2m + V(x) - q E(t) x ] phi,   E = -dA/dt   (length gauge)

Same physics, different bookkeeping. derive_gauge_transform() checks that identity in SymPy, and
simulate_box() checks it numerically: an electron in a box is driven from level 1 to 2 by a pulse
whose A starts and ends at zero, once in each gauge, and the final populations agree as the basis grows.
The two-level limit gives the Rabi frequency Omega = q E0 |x_12| / hbar (length) =
(q/m) A0 |p_12| / hbar (velocity), and the two are equal because p_12 = i m omega_12 x_12.

The same idea runs a superconducting qubit, with circuit variables in place of fields: the node flux
Phi plays the role of the vector potential (V = -dPhi/dt like E = -dA/dt), and the drive couples to
the qubit's charge, Omega = 2 e beta V0 <0|n|1> / hbar. On the CSUS bench analog (two coupled LC
tanks) the "drive" is the mutual inductance, Omega ~ k omega0, and the flux linkage per ampere M = kL
is the circuit's vector potential.

ONE SOURCE OF NUMBERS. dgs/qubit_platforms.json holds the inputs for three platforms (transmon, CSUS
LC analog, electron-in-a-box). derived() turns them into the table; table_lines() prints it in a
fixed "platform.key value" format; learn/matlab/qubit_platforms_table.m computes the same table
independently from the same JSON; lean_source() writes learn/lean/QubitPlatforms.lean from the JSON
with machine-checked identities. scripts/check_qubit_platforms_everywhere.sh runs all of them, compares
the numbers, and checks every exit code.

Platform values are illustrative orders of magnitude (stated in the JSON), not a specific device.

Connects to: dgs.qubit_microwave (the drive chain that delivers V0), dgs.qubit_analog_bench (the CSUS
bench), dgs.qubit_scope (Rabi/Ramsey), griffiths/ (Griffiths QM ch. 4 / ED ch. 10 potentials).
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
from fractions import Fraction

import numpy as np

JSON_PATH = pathlib.Path(__file__).with_name("qubit_platforms.json")
LEAN_PATH = pathlib.Path(__file__).resolve().parent.parent / "learn" / "lean" / "QubitPlatforms.lean"


# ======================================================================== the shared numbers
def load_platforms(path=None):
    with open(path or JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)
    for key in ("constants", "platforms"):
        if key not in data:
            raise ValueError(f"{path or JSON_PATH}: missing {key!r}")
    return data


def derived(data=None):
    """All derived quantities, {platform: {key: float}}. MATLAB's script uses the SAME formulas."""
    data = data or load_platforms()
    c = data["constants"]
    e, h, kB, me = c["e_C"], c["h_Js"], c["kB_JpK"], c["m_e_kg"]
    hbar = h / (2 * math.pi)
    out = {}

    p = data["platforms"]["transmon"]
    EC = p["EC_over_h_Hz"] * h
    EJ = p["EJ_over_EC"] * EC
    f01 = (math.sqrt(8 * EJ * EC) - EC) / h
    w01 = 2 * math.pi * f01
    n01 = (EJ / (8 * EC)) ** 0.25 / math.sqrt(2)
    V0 = math.sqrt(2 * 1e-3 * 10 ** (p["drive_dBm_at_qubit"] / 10) * p["Z0_ohm"])
    Om = 2 * e * p["beta_Cg_over_Csum"] * V0 * n01 / hbar
    out["transmon"] = {
        "f_transition_Hz": f01, "Csum_F": e ** 2 / (2 * EC), "n01": n01, "V0_V": V0,
        "Omega_rad_s": Om, "f_Rabi_Hz": Om / (2 * math.pi), "t_pi_s": math.pi / Om,
        "vector_potential_amp": V0 / w01,             # drive flux amplitude (Wb): V = -dPhi/dt
        "n_thermal": 1 / math.expm1(h * f01 / (kB * p["T_K"])),
    }

    p = data["platforms"]["csus_lc_analog"]
    Ct = p["C_F"] + p["C_probe_F"]
    w0 = 1 / math.sqrt(p["L_H"] * Ct)
    Om = p["k"] * w0
    out["csus_lc_analog"] = {
        "f_transition_Hz": w0 / (2 * math.pi), "Omega_rad_s": Om, "f_Rabi_Hz": Om / (2 * math.pi),
        "t_pi_s": math.pi / Om,                       # = the energy swap time
        "vector_potential_amp": p["k"] * p["L_H"],    # flux linkage per ampere M = kL (Wb/A)
        "n_thermal": 1 / math.expm1(h * w0 / (2 * math.pi) / (kB * p["T_K"])),
    }

    p = data["platforms"]["box_atom"]
    a, E0 = p["a_m"], p["E0_Vpm"]
    w12 = 3 * math.pi ** 2 * hbar / (2 * me * a ** 2)
    x12 = 16 * a / (9 * math.pi ** 2)
    Om = e * E0 * x12 / hbar
    A0 = E0 / w12
    out["box_atom"] = {
        "f_transition_Hz": w12 / (2 * math.pi), "x12_m": x12, "Omega_rad_s": Om,
        "Omega_velocity_gauge_rad_s": (e / me) * A0 * (me * w12 * x12) / hbar,
        "f_Rabi_Hz": Om / (2 * math.pi), "t_pi_s": math.pi / Om,
        "vector_potential_amp": A0,                   # A0 = E0 / omega (V s / m)
        "n_thermal": 1 / math.expm1(h * w12 / (2 * math.pi) / (kB * p["T_K"])),
    }
    return out


def table_lines(data=None):
    """Fixed-format lines 'platform.key value' (12 significant digits), shared with the MATLAB script."""
    d = derived(data)
    return [f"{plat}.{k} {v:.12e}" for plat in sorted(d) for k, v in sorted(d[plat].items())]


def parse_table(text):
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and "." in parts[0] and not parts[0].startswith(("[", "%")):
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return out


def compare_tables(a, b, rtol=1e-9):
    """Return list of mismatches between two parsed tables (missing keys count)."""
    bad = []
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            bad.append((k, a.get(k), b.get(k)))
        elif not math.isclose(a[k], b[k], rel_tol=rtol, abs_tol=0.0):
            bad.append((k, a[k], b[k]))
    return bad


# ======================================================================== SymPy
def derive_gauge_transform():
    """SymPy: psi = exp(i q A(t) x / hbar) phi turns the velocity-gauge equation into the length-gauge one.

    Returns the residual (should simplify to 0) and the two Hamiltonian actions for display.
    """
    import sympy as sp
    x, t = sp.symbols("x t", real=True)
    q, m, hbar = sp.symbols("q m hbar", positive=True)
    A = sp.Function("A")(t)
    V = sp.Function("V")(x)
    phi = sp.Function("phi")(x, t)
    psi = sp.exp(sp.I * q * A * x / hbar) * phi
    p = lambda f: -sp.I * hbar * sp.diff(f, x)
    Hv_psi = (p(p(psi)) - 2 * q * A * p(psi) + q ** 2 * A ** 2 * psi) / (2 * m) + V * psi   # (p - qA)^2 / 2m, A uniform
    E = -sp.diff(A, t)
    HL_phi = p(p(phi)) / (2 * m) + V * phi - q * E * x * phi
    lhs = sp.I * hbar * sp.diff(psi, t) - Hv_psi
    rhs = sp.exp(sp.I * q * A * x / hbar) * (sp.I * hbar * sp.diff(phi, t) - HL_phi)
    return {"residual": sp.simplify(sp.expand(lhs - rhs)), "H_velocity_psi": Hv_psi, "H_length_phi": HL_phi}


def box_matrix_elements_symbolic():
    """SymPy closed forms for <m|x|n> and <m|d/dx|n> in a box [0, a] (m != n)."""
    import sympy as sp
    x, a = sp.symbols("x a", positive=True)
    m, n = sp.symbols("m n", positive=True, integer=True)
    psi = lambda k: sp.sqrt(2 / a) * sp.sin(k * sp.pi * x / a)
    X = sp.simplify(sp.integrate(psi(m) * x * psi(n), (x, 0, a), conds="none"))
    D = sp.simplify(sp.integrate(psi(m) * sp.diff(psi(n), x), (x, 0, a), conds="none"))
    return {"x_mn": X, "ddx_mn": D}


def sympy_table(data=None):
    """Per-platform rows (quantity, SymPy expression, value) built from the JSON numbers."""
    import sympy as sp
    data = data or load_platforms()
    c = data["constants"]
    e, h, kB, me, hbar, pi = sp.symbols("e h k_B m_e hbar pi", positive=True)
    subs_c = {e: c["e_C"], h: c["h_Js"], kB: c["kB_JpK"], me: c["m_e_kg"], hbar: c["h_Js"] / (2 * math.pi), pi: math.pi}
    rows = {}

    P = data["platforms"]["transmon"]
    EC, r, beta, V0, w = sp.symbols("E_C r beta V_0 omega_01", positive=True)
    f01 = (sp.sqrt(8 * r * EC * EC) - EC) / h
    n01 = (r / 8) ** sp.Rational(1, 4) / sp.sqrt(2)
    Om = 2 * e * beta * V0 * n01 / hbar
    vals = {EC: P["EC_over_h_Hz"] * c["h_Js"], r: P["EJ_over_EC"], beta: P["beta_Cg_over_Csum"],
            V0: math.sqrt(2 * 1e-3 * 10 ** (P["drive_dBm_at_qubit"] / 10) * P["Z0_ohm"])}
    vals[w] = 2 * math.pi * float(f01.subs(vals).subs(subs_c))
    rows["transmon"] = [("f01 (Hz)", f01), ("<0|n|1>", n01), ("Omega (rad/s)", Om), ("t_pi (s)", pi / Om),
                        ("drive flux Phi0 = V0/omega (Wb)", V0 / w)]
    rows["transmon"] = [(q_, ex, float(ex.subs(vals).subs(subs_c))) for q_, ex in rows["transmon"]]

    P = data["platforms"]["csus_lc_analog"]
    L, C, Cp, k = sp.symbols("L C C_p k", positive=True)
    w0 = 1 / sp.sqrt(L * (C + Cp))
    vals = {L: P["L_H"], C: P["C_F"], Cp: P["C_probe_F"], k: P["k"]}
    rows["csus_lc_analog"] = [(q_, ex, float(ex.subs(vals).subs(subs_c))) for q_, ex in [
        ("f0 (Hz)", w0 / (2 * pi)), ("Omega = k omega0 (rad/s)", k * w0), ("swap time pi/Omega (s)", pi / (k * w0)),
        ("M = k L (Wb/A)", k * L)]]

    P = data["platforms"]["box_atom"]
    a, E0 = sp.symbols("a E_0", positive=True)
    w12 = 3 * pi ** 2 * hbar / (2 * me * a ** 2)
    x12 = 16 * a / (9 * pi ** 2)
    vals = {a: P["a_m"], E0: P["E0_Vpm"]}
    rows["box_atom"] = [(q_, ex, float(ex.subs(vals).subs(subs_c))) for q_, ex in [
        ("f12 (Hz)", w12 / (2 * pi)), ("|x12| (m)", x12), ("Omega, length gauge (rad/s)", e * E0 * x12 / hbar),
        ("Omega, velocity gauge (rad/s)", sp.simplify((e / me) * (E0 / w12) * (me * w12 * x12) / hbar)),
        ("A0 = E0/omega (V s/m)", E0 / w12), ("t_pi (s)", pi * hbar / (e * E0 * x12))]]
    return rows


# ======================================================================== numerical gauge check
def box_matrices(N):
    """Energies and x, p matrices for the lowest N box states (hbar = m = a = 1). Closed forms from SymPy."""
    if N < 2:
        raise ValueError("need N >= 2 levels")
    n = np.arange(1, N + 1)
    E = n ** 2 * np.pi ** 2 / 2
    M, Nn = np.meshgrid(n, n, indexing="ij")
    odd = ((M + Nn) % 2 == 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        X = np.where(odd, -8 * M * Nn / (np.pi ** 2 * (M ** 2 - Nn ** 2) ** 2), 0.0)
        D = np.where(odd, 4 * M * Nn / (M ** 2 - Nn ** 2), 0.0)
    X[np.diag_indices(N)] = 0.5
    P = -1j * D
    return E, X, P


def pulse(t, A0, omega, T):
    """A(t) = A0 sin^2(pi t / T) sin(omega t) (zero at both ends) and E = -dA/dt."""
    s = np.sin(np.pi * t / T) ** 2
    ds = np.pi / T * np.sin(2 * np.pi * t / T)
    A = A0 * s * np.sin(omega * t)
    E = -A0 * (ds * np.sin(omega * t) + s * omega * np.cos(omega * t))
    return A, E


def simulate_box(N, gauge, n_periods=40, steps_per_period=200, q=1.0, history=False):
    """Drive level 1 -> 2 with a pi pulse (RWA estimate); return final populations (|c_n|^2).

    history=True returns (t, populations over time). Mid-pulse, energy-basis populations are NOT
    gauge invariant (A != 0 there); only where A = 0, e.g. at the end, must the two gauges agree.
    """
    if gauge not in ("length", "velocity"):
        raise ValueError("gauge must be 'length' or 'velocity'")
    E, X, P = box_matrices(N)
    w = E[1] - E[0]
    T = n_periods * 2 * np.pi / w
    E0 = 2 * np.pi / (q * abs(X[0, 1]) * T)          # pulse area: q E0 |x12| * T/2 = pi
    A0 = E0 / w
    steps = n_periods * steps_per_period
    dt = T / steps
    psi = np.zeros(N, complex); psi[0] = 1
    H0 = np.diag(E)
    hist_t, hist_p = [0.0], [np.abs(psi) ** 2]
    for j in range(steps):
        tm = (j + 0.5) * dt
        A, Ef = pulse(tm, A0, w, T)
        H = H0 - q * Ef * X if gauge == "length" else H0 - q * A * P + 0.5 * q * q * A * A * np.eye(N)
        lam, V = np.linalg.eigh(H)
        psi = V @ (np.exp(-1j * lam * dt) * (V.conj().T @ psi))
        if history:
            hist_t.append((j + 1) * dt); hist_p.append(np.abs(psi) ** 2)
    if history:
        return np.array(hist_t), np.array(hist_p)
    return np.abs(psi) ** 2


def gauge_table(Ns=(2, 3, 4, 8, 16), **kw):
    rows = []
    for N in Ns:
        pl, pv = simulate_box(N, "length", **kw), simulate_box(N, "velocity", **kw)
        rows.append({"N": N, "P2_length": pl[1], "P2_velocity": pv[1], "diff": abs(pl[1] - pv[1]),
                     "leak_length": 1 - pl[0] - pl[1], "leak_velocity": 1 - pv[0] - pv[1]})
    return rows


# ======================================================================== Lean, generated from the JSON
def _q(x):
    """Exact rational Lean literal for a JSON number (via its shortest decimal string)."""
    fr = Fraction(repr(float(x))) if not isinstance(x, int) else Fraction(x)
    return f"({fr.numerator} : ℝ)" if fr.denominator == 1 else f"(({fr.numerator} : ℝ) / {fr.denominator})"


def lean_source(data=None):
    data = data or load_platforms()
    T = data["platforms"]["transmon"]
    A = data["platforms"]["csus_lc_analog"]
    r8 = 8 * T["EJ_over_EC"]
    root = math.isqrt(int(r8)) if float(r8).is_integer() else None
    perfect = root is not None and root * root == r8
    ec = int(T["EC_over_h_Hz"])
    lines = [
        "/- GENERATED by dgs/vector_potential_qubit.py from dgs/qubit_platforms.json. Do not edit by hand:",
        "   change the JSON and regenerate (scripts/check_qubit_platforms_everywhere.sh does both). -/",
        "import Mathlib.Analysis.SpecialFunctions.Sqrt",
        "import Mathlib.Tactic.FieldSimp",
        "import Mathlib.Tactic.Ring",
        "import Mathlib.Tactic.NormNum",
        "import Mathlib.Tactic.Linarith",
        "import Mathlib.Tactic.Positivity",
        "",
        "namespace QubitPlatforms",
        "",
        "noncomputable section",
        "",
        "-- the numbers, copied exactly (as rationals) from the JSON",
        f"def EC_over_h : ℝ := {_q(T['EC_over_h_Hz'])}",
        f"def EJ_over_EC : ℝ := {_q(T['EJ_over_EC'])}",
        f"def L_csus : ℝ := {_q(A['L_H'])}",
        f"def k_csus : ℝ := {_q(A['k'])}",
        "",
        "/-- Gauge equivalence on resonance: the velocity-gauge coupling (q/m) A0 |p12| with A0 = E0/omega",
        "    and p12 = m omega x12 equals the length-gauge coupling q E0 x12. -/",
        "theorem gauge_coupling_equal (q m ω E0 x : ℝ) (hm : m ≠ 0) (hω : ω ≠ 0) :",
        "    (q / m) * (E0 / ω) * (m * ω * x) = q * E0 * x := by",
        "  field_simp",
        "",
        "/-- Box: the commutator identity hbar <m|d/dx|n> = -M omega_mn <m|x|n> holds for the SymPy closed",
        "    forms, for any parity sign s = (-1)^(m+n). This is why p12 = i M omega12 x12. -/",
        "theorem box_commutator (ħ M a p m n s : ℝ) (ha : a ≠ 0) (hM : M ≠ 0) (hp : p ≠ 0)",
        "    (hmn : m ^ 2 - n ^ 2 ≠ 0) :",
        "    ħ * (2 * m * n * (1 - s) / (a * (m ^ 2 - n ^ 2))) =",
        "      -(M * ((m ^ 2 - n ^ 2) * p ^ 2 * ħ / (2 * M * a ^ 2)) *",
        "        (4 * a * m * n * (s - 1) / (p ^ 2 * (m ^ 2 - n ^ 2) ^ 2))) := by",
        "  field_simp",
        "  ring",
        "",
        "/-- Box dipole element 1 -> 2 from the general formula: x12 = -16 a / (9 pi^2). -/",
        "theorem box_x12 (a p : ℝ) (hp : p ≠ 0) :",
        "    4 * a * 1 * 2 * ((-1 : ℝ) - 1) / (p ^ 2 * ((1 : ℝ) ^ 2 - 2 ^ 2) ^ 2) = -16 * a / (9 * p ^ 2) := by",
        "  field_simp",
        "  ring",
        "",
        "/-- Detuned Rabi (and detuned coupled tanks): the transferred fraction never exceeds 1. -/",
        "theorem rabi_contrast_le_one (Ω Δ : ℝ) (h : 0 < Ω ^ 2 + Δ ^ 2) : Ω ^ 2 / (Ω ^ 2 + Δ ^ 2) ≤ 1 := by",
        "  rw [div_le_one h]",
        "  nlinarith [sq_nonneg Δ]",
        "",
        "/-- CSUS bench: series aiding minus opposing is exactly 4M = 4kL (the LCR recipe). -/",
        "theorem csus_series_recipe : (2 * L_csus + 2 * (k_csus * L_csus)) - (2 * L_csus - 2 * (k_csus * L_csus))",
        "    = 4 * (k_csus * L_csus) := by",
        "  ring",
        "",
    ]
    if perfect:
        f01 = (root - 1) * ec
        lines += [
            f"/-- Transmon: with EJ/EC = {T['EJ_over_EC']}, 8 EJ EC = ({root} EC)^2, so f01 = sqrt(8 EJ EC) - EC = {root - 1} EC",
            f"    = {f01} Hz, the value every tool must report. -/",
            "theorem transmon_f01 :",
            f"    Real.sqrt (8 * (EJ_over_EC * EC_over_h) * EC_over_h) - EC_over_h = {f01} := by",
            f"  have h : 8 * (EJ_over_EC * EC_over_h) * EC_over_h = ({root} * EC_over_h) ^ 2 := by",
            "    unfold EJ_over_EC EC_over_h; norm_num",
            "  rw [h, Real.sqrt_sq (by unfold EC_over_h; norm_num)]",
            "  unfold EC_over_h; norm_num",
            "",
        ]
    else:
        lines += ["-- transmon_f01 omitted: 8 * EJ/EC is not a perfect square, so f01 is irrational here.", ""]
    lines += ["end", "", "end QubitPlatforms", ""]
    return "\n".join(lines)


def write_lean(path=None):
    path = pathlib.Path(path or LEAN_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(lean_source(), encoding="utf-8")
    return path


if __name__ == "__main__":
    if "--write-lean" in sys.argv:
        print("wrote", write_lean())
    if "--table" in sys.argv or len(sys.argv) == 1:
        print("\n".join(table_lines()))
