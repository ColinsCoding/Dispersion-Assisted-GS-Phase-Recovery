"""Lagrangian mechanics applied to circuits.

Noether's theorem: every continuous symmetry of the action S = integral[L dt]
has a corresponding conserved quantity.
  Time translation symmetry  ->  energy H = T + V conserved
  Charge translation symmetry -> flux linkage conserved

Circuit Lagrangian analogy (Kirchhoff via Euler-Lagrange):
  Generalized coordinate q = charge on capacitor  (analogous to position x)
  Generalized velocity   q_dot = I = current      (analogous to velocity v)

  T = (1/2) L * q_dot^2          (inductor -- kinetic energy, stores I^2)
  V = (1/2) * q^2 / C            (capacitor -- potential energy, stores V^2)
  D = (1/2) R * q_dot^2          (resistor -- Rayleigh dissipation function)

  Euler-Lagrange with dissipation:
    d/dt(dL/dq_dot) - dL/dq + dD/dq_dot = V_source(t)
    L*q_ddot + R*q_dot + q/C = V_source(t)   <-- KVL!

This module:
  lagrangian_rlc_sympy()      -- symbolic EOM from E-L equation
  noether_energy_sympy()      -- symbolic Hamiltonian + energy conservation
  euler_lagrange_solve()      -- numerical RLC response via Euler-Lagrange
  power_iv()                  -- P = I*V; Thevenin; R -> infinity limit
  circuit_analogy_table()     -- mechanical <-> electrical analogy
  action_integral()           -- S = integral[L dt] numerically
  normal_modes_coupled()      -- two coupled LC oscillators (2 coords)
  lagrangian_transmission_line() -- distributed LC ladder -> wave equation
"""

from __future__ import annotations
import numpy as np
import sympy as sp
from typing import Dict, List, Tuple


# ── Symbolic Lagrangian for RLC ───────────────────────────────────────────────
def lagrangian_rlc_sympy() -> Dict[str, sp.Expr]:
    """Symbolic derivation of KVL from Euler-Lagrange + Rayleigh dissipation.

    Returns dict of symbolic expressions:
      L_circuit, T, V, D, EOM (equation of motion), KVL_check
    """
    t   = sp.Symbol("t")
    L_i, R, C = sp.symbols("L R C", positive=True)   # L_i = inductance
    V_s = sp.Function("V_s")(t)

    q   = sp.Function("q")(t)
    dq  = q.diff(t)     # current I
    ddq = q.diff(t, 2)  # dI/dt

    T_expr = sp.Rational(1, 2) * L_i * dq**2
    V_expr = sp.Rational(1, 2) * q**2 / C
    D_expr = sp.Rational(1, 2) * R * dq**2
    Lag    = T_expr - V_expr

    # Euler-Lagrange: d/dt(dL/dq') - dL/dq + dD/dq' = V_s
    dL_dqdot = sp.diff(Lag, dq)
    dL_dq    = sp.diff(Lag, q)
    dD_dqdot = sp.diff(D_expr, dq)

    EOM_lhs = dL_dqdot.diff(t) - dL_dq + dD_dqdot
    EOM     = sp.Eq(EOM_lhs, V_s)

    # Substitute q''=ddq, q'=dq, check it equals L*q'' + R*q' + q/C = V_s
    EOM_sub  = sp.Eq(L_i * ddq + R * dq + q / C, V_s)
    match    = sp.simplify(EOM_lhs - (L_i * ddq + R * dq + q / C)) == 0

    return {
        "T":          T_expr,
        "V":          V_expr,
        "D_rayleigh": D_expr,
        "L_circuit":  Lag,
        "EOM":        EOM,
        "EOM_standard": EOM_sub,
        "EOM_matches_KVL": match,
    }


# ── Noether's theorem: time symmetry -> energy conservation ──────────────────
def noether_energy_sympy() -> Dict[str, sp.Expr]:
    """Noether's theorem applied to time-translation symmetry.

    If L does not depend explicitly on t, then the Hamiltonian
    H = p*q_dot - L = T + V is conserved (dH/dt = 0).

    p = dL/dq_dot (canonical momentum = L_inductance * I = flux linkage)
    H = p*q_dot - L = T + V = total energy

    Returns symbolic expressions for p, H, and conservation statement.
    """
    L_i, C = sp.symbols("L C", positive=True)
    q, dq  = sp.symbols("q dq", real=True)   # q=charge, dq=current

    T_sym  = sp.Rational(1, 2) * L_i * dq**2
    V_sym  = sp.Rational(1, 2) * q**2 / C
    Lag    = T_sym - V_sym

    p_canon = sp.diff(Lag, dq)              # canonical momentum = L*I = flux
    H_sym   = p_canon * dq - Lag           # Legendre transform -> Hamiltonian
    H_simp  = sp.simplify(H_sym)

    # Verify H = T + V
    H_alt   = T_sym + V_sym
    H_check = sp.simplify(H_simp - H_alt) == 0

    return {
        "canonical_momentum_p": sp.Eq(sp.Symbol("p"), p_canon),
        "Hamiltonian_H":        sp.Eq(sp.Symbol("H"), H_simp),
        "H_equals_TplusV":     H_check,
        "Noether_statement":   sp.Eq(sp.Symbol("dH/dt"), 0),
        "physical_meaning":    "Flux linkage = L*I = canonical momentum; "
                               "H = total electromagnetic energy = conserved "
                               "when L does not depend explicitly on t.",
    }


# ── Numerical RLC response via Euler-Lagrange (state-space) ──────────────────
def euler_lagrange_solve(L_H: float, R_ohm: float, C_F: float,
                          V_source_fn,
                          t_span: Tuple[float, float] = (0, 0.01),
                          q0: float = 0.0, I0: float = 0.0,
                          n_pts: int = 2000) -> Dict:
    """Numerically integrate RLC EOM derived from Euler-Lagrange.

    L*q'' + R*q' + q/C = V_s(t)
    Rewritten as state-space [q, I=q']':
      dq/dt = I
      dI/dt = (V_s(t) - R*I - q/C) / L

    Parameters
    ----------
    V_source_fn : callable
        V_source_fn(t) returns voltage at time t.
    """
    if L_H <= 0 or C_F <= 0:
        raise ValueError("L and C must be positive")
    dt   = (t_span[1] - t_span[0]) / n_pts
    t    = np.linspace(t_span[0], t_span[1], n_pts)
    q    = np.zeros(n_pts)
    I    = np.zeros(n_pts)
    V_c  = np.zeros(n_pts)     # capacitor voltage = q/C
    V_L  = np.zeros(n_pts)     # inductor voltage = L*dI/dt
    P_R  = np.zeros(n_pts)     # power dissipated in R

    q[0], I[0] = q0, I0
    V_c[0]     = q0 / C_F

    for i in range(1, n_pts):
        Vs    = V_source_fn(t[i - 1])
        dI_dt = (Vs - R_ohm * I[i - 1] - q[i - 1] / C_F) / L_H
        I[i]  = I[i - 1] + dI_dt * dt
        q[i]  = q[i - 1] + I[i - 1] * dt
        V_c[i]= q[i] / C_F
        P_R[i]= R_ohm * I[i]**2

    V_L = L_H * np.gradient(I, dt)
    H   = 0.5 * L_H * I**2 + 0.5 * q**2 / C_F   # conserved when R=0
    return {
        "t": t, "q": q, "I": I,
        "V_C": V_c, "V_L": V_L,
        "power_R": P_R,
        "H": H,                                   # Hamiltonian = total energy
        "omega_0": 1 / np.sqrt(L_H * C_F),
        "Q_factor": (1 / R_ohm) * np.sqrt(L_H / C_F) if R_ohm > 0 else np.inf,
        "zeta":     R_ohm / (2 * np.sqrt(L_H / C_F)),
    }


# ── Power P = I*V; Thevenin; R -> infinity ───────────────────────────────────
def power_iv(I: float | np.ndarray,
              V: float | np.ndarray) -> float | np.ndarray:
    """Instantaneous power P = I * V (Watts).

    Sign convention: positive P = power absorbed by element.
    For a resistor: P = I^2 * R = V^2 / R >= 0 always.
    For a source: P < 0 means power delivered to circuit.
    """
    return np.asarray(I) * np.asarray(V)


def thevenin_norton(V_oc: float, I_sc: float) -> Dict:
    """Thevenin and Norton equivalent circuit parameters.

    V_oc : open-circuit voltage (V)
    I_sc : short-circuit current (A)

    Thevenin: V_th = V_oc, R_th = V_oc / I_sc
    Norton:   I_N  = I_sc, R_N  = R_th  (same R)
    """
    if abs(I_sc) < 1e-300:
        R_th = np.inf
    else:
        R_th = V_oc / I_sc
    return {
        "V_oc":  V_oc,
        "I_sc":  I_sc,
        "R_th":  R_th,
        "V_th":  V_oc,       # Thevenin voltage
        "I_N":   I_sc,       # Norton current
        "max_power_transfer": V_oc**2 / (4 * R_th) if np.isfinite(R_th) else 0.0,
        "R_load_for_max_power": R_th,
    }


def open_circuit_limit(V_source: float, R_load: float) -> Dict:
    """I*V as R_load -> infinity (open circuit).

    lim_{R->inf} I = V / R -> 0
    lim_{R->inf} V_load = V_source (all voltage appears across load)
    lim_{R->inf} P = I*V = V^2/R -> 0

    Power goes to zero even though V -> V_source.
    The product I*V has the 0 * inf form resolved by l'Hopital -> 0.
    """
    if R_load <= 0:
        raise ValueError("R_load must be positive")
    I    = V_source / R_load
    V_L  = I * R_load        # = V_source (always, by KVL)
    P    = I * V_L
    eta  = 1.0               # efficiency: V_load / V_source always = 1 (KVL)
    return {"R_load": R_load, "I": I, "V_load": V_L,
            "P_load": P, "eta": eta}


def power_sweep_R(V_source: float, R_int: float,
                   R_load_arr: np.ndarray) -> Dict:
    """Power delivered to R_load vs R_load. Maximum at R_load = R_internal.

    Maximum power transfer theorem: P_max = V^2 / (4*R_int)
    at R_load = R_int.
    """
    I_arr   = V_source / (R_int + R_load_arr)
    P_arr   = I_arr**2 * R_load_arr
    P_max   = V_source**2 / (4 * R_int)
    idx_max = np.argmax(P_arr)
    return {
        "R_load": R_load_arr,
        "I":      I_arr,
        "P_load": P_arr,
        "P_max":  P_max,
        "R_at_Pmax": R_load_arr[idx_max],
        "P_dissipated_Rint": I_arr**2 * R_int,
    }


# ── Mechanical <-> Electrical analogy ────────────────────────────────────────
CIRCUIT_ANALOGY = {
    "position x":              "charge q",
    "velocity v = x_dot":      "current I = q_dot",
    "mass m":                  "inductance L",
    "spring constant k":       "1/C (elastance)",
    "damping coeff b":         "resistance R",
    "force F(t)":              "voltage V_s(t)",
    "kinetic energy (1/2)mv^2": "magnetic energy (1/2)LI^2",
    "potential energy (1/2)kx^2": "electric energy (1/2)q^2/C = (1/2)CV^2",
    "momentum p = mv":         "flux linkage lambda = LI",
    "Lagrangian L=T-V":        "L_circuit = (1/2)LI^2 - q^2/(2C)",
    "Hamiltonian H=T+V":       "H_EM = (1/2)LI^2 + q^2/(2C) = total EM energy",
    "Noether: time symm":      "dH/dt = 0 when no explicit t in L",
    "natural frequency omega": "omega_0 = 1/sqrt(LC)",
    "quality factor Q":        "Q = (1/R)*sqrt(L/C)",
}


def circuit_analogy_table() -> str:
    lines = [f"{'Mechanical':<35} {'Electrical':<35}",
             "=" * 70]
    for mech, elec in CIRCUIT_ANALOGY.items():
        lines.append(f"{mech:<35} {elec:<35}")
    return "\n".join(lines)


# ── Action integral S = integral[L dt] ───────────────────────────────────────
def action_integral(t: np.ndarray, q: np.ndarray, I: np.ndarray,
                     L_H: float, C_F: float) -> float:
    """Numerical action S = integral[T - V] dt.

    T = (1/2) L I^2  (kinetic = magnetic)
    V = (1/2) q^2/C  (potential = electric)

    The physical trajectory minimises S (principle of least action).
    """
    T_arr = 0.5 * L_H * I**2
    V_arr = 0.5 * q**2 / C_F
    L_arr = T_arr - V_arr
    return float(np.trapezoid(L_arr, t))


# ── Two coupled LC oscillators: normal modes ──────────────────────────────────
def normal_modes_coupled(L_H: float, C_F: float, C_couple: float) -> Dict:
    """Two LC oscillators coupled through C_couple.

    Lagrangian: L = (1/2)L(I1^2 + I2^2)
                  - (1/2)(q1^2/C + q2^2/C)
                  - (q1-q2)^2 / (2*C_couple)

    Normal mode frequencies via SymPy eigenvalues of dynamical matrix.
    Mode 1 (symmetric):  both charges in phase
    Mode 2 (antisymmetric): charges out of phase
    """
    omega_0 = 1 / np.sqrt(L_H * C_F)
    omega_c = 1 / np.sqrt(L_H * C_couple)

    # Dynamical matrix K*x = omega^2 * M * x
    # With M = L*I, K entries from dV/dq
    K11 = 1/C_F + 1/C_couple
    K12 = -1/C_couple
    K   = np.array([[K11, K12], [K12, K11]])
    M   = L_H * np.eye(2)

    omega2 = np.linalg.eigvalsh(np.linalg.solve(M, K))
    omega2 = np.sort(omega2)
    omega_modes = np.sqrt(np.clip(omega2, 0, None))

    return {
        "omega_0":     omega_0,
        "omega_symmetric":    omega_modes[0],   # in-phase: lower freq
        "omega_antisymmetric": omega_modes[1],  # out-of-phase: higher freq
        "freq_splitting": omega_modes[1] - omega_modes[0],
        "mode_1_description": "Both charges oscillate in phase (lower omega)",
        "mode_2_description": "Charges oscillate opposite phase (higher omega)",
    }


# ── Transmission line: distributed LC -> wave equation ───────────────────────
def lagrangian_transmission_line(N: int = 20, L_per_m: float = 250e-9,
                                  C_per_m: float = 100e-12,
                                  dx: float = 0.01) -> Dict:
    """Distributed LC ladder -> telegrapher's / wave equation.

    N cells of (L*dx) inductance and (C*dx) capacitance per cell.
    Lagrangian: sum_i [ (1/2)(L*dx)*I_i^2 - (1/2)(q_i^2)/(C*dx) ]

    Continuum limit dx->0: wave equation d^2V/dt^2 = (1/LC) d^2V/dx^2
    Phase velocity v_p = 1/sqrt(LC)

    Returns characteristic impedance, phase velocity, and numerical
    dispersion relation omega(k).
    """
    Z0   = np.sqrt(L_per_m / C_per_m)           # characteristic impedance
    v_p  = 1 / np.sqrt(L_per_m * C_per_m)       # phase velocity

    # Dispersion relation for LC ladder: omega = 2/sqrt(L*C*dx^2) * |sin(k*dx/2)|
    k_arr   = np.linspace(0, np.pi / dx, 500)
    L_cell  = L_per_m * dx
    C_cell  = C_per_m * dx
    omega_arr = (2 / np.sqrt(L_cell * C_cell)) * np.abs(np.sin(k_arr * dx / 2))

    return {
        "Z0":           Z0,
        "v_phase":      v_p,
        "N_cells":      N,
        "L_per_m":      L_per_m,
        "C_per_m":      C_per_m,
        "k_arr":        k_arr,
        "omega_arr":    omega_arr,
        "omega_cutoff": 2 / np.sqrt(L_cell * C_cell),  # Bragg cutoff
        "wave_eq":      "d^2V/dt^2 = (1/LC) * d^2V/dx^2",
        "v_p_formula":  "v_p = 1/sqrt(L*C)  [m/s]",
    }


# ── SymPy: 5 key Lagrangian/Noether equations ────────────────────────────────
def lagrangian_sympy_5() -> Dict[str, sp.Expr]:
    """5 key equations for sp.init_printing."""
    t    = sp.Symbol("t")
    L_i, R, C, q_s, I_s = sp.symbols("L R C q I", positive=True)
    dq   = sp.Symbol("dq", real=True)   # q_dot = I

    T    = sp.Rational(1, 2) * L_i * dq**2
    V    = sp.Rational(1, 2) * q_s**2 / C
    Lag  = T - V
    H    = T + V
    p    = sp.diff(Lag, dq)
    EOM  = sp.Eq(L_i * sp.Symbol("ddq") + R * dq + q_s / C, sp.Symbol("V_s"))

    return {
        "Lagrangian L=T-V":    sp.Eq(sp.Symbol("L_c"), Lag),
        "Hamiltonian H=T+V":   sp.Eq(sp.Symbol("H"), H),
        "canonical_momentum":  sp.Eq(sp.Symbol("p"), p),
        "Euler-Lagrange (KVL)": EOM,
        "omega_0":             sp.Eq(sp.Symbol("omega_0"),
                                     1 / sp.sqrt(L_i * C)),
    }


if __name__ == "__main__":
    print("=== Symbolic RLC from Euler-Lagrange ===")
    rlc = lagrangian_rlc_sympy()
    print(f"  T = {rlc['T']}")
    print(f"  V = {rlc['V']}")
    print(f"  EOM = {rlc['EOM']}")
    print(f"  Matches KVL: {rlc['EOM_matches_KVL']}")

    print("\n=== Noether: time symmetry -> energy ===")
    n = noether_energy_sympy()
    print(f"  p (flux linkage) = {n['canonical_momentum_p']}")
    print(f"  H = {n['Hamiltonian_H']}")
    print(f"  H = T + V: {n['H_equals_TplusV']}")
    print(f"  {n['physical_meaning']}")

    print("\n=== Numerical RLC: step response ===")
    result = euler_lagrange_solve(
        L_H=1e-3, R_ohm=10.0, C_F=1e-6,
        V_source_fn=lambda t: 5.0 * (t > 0),
        t_span=(0, 5e-3), n_pts=2000
    )
    print(f"  omega_0 = {result['omega_0']:.1f} rad/s")
    print(f"  Q = {result['Q_factor']:.3f}")
    print(f"  zeta = {result['zeta']:.3f}")
    print(f"  Max current = {result['I'].max():.4f} A")

    print("\n=== Power P = I*V; Thevenin ===")
    th = thevenin_norton(12.0, 2.0)
    print(f"  R_th = {th['R_th']:.1f} Ohm")
    print(f"  P_max at R_load = R_th: {th['max_power_transfer']:.2f} W")
    for R in [1e3, 1e6, 1e9]:
        oc = open_circuit_limit(12.0, R)
        print(f"  R={R:.0e}: I={oc['I']:.2e} A, P={oc['P_load']:.2e} W")

    print("\n=== Mechanical <-> Electrical analogy ===")
    print(circuit_analogy_table())

    print("\n=== Normal modes: two coupled LC ===")
    nm = normal_modes_coupled(1e-3, 1e-6, 10e-6)
    print(f"  omega_0         = {nm['omega_0']:.1f} rad/s")
    print(f"  omega_symmetric = {nm['omega_symmetric']:.1f} rad/s")
    print(f"  omega_antisym   = {nm['omega_antisymmetric']:.1f} rad/s")

    print("\n=== Transmission line ===")
    tl = lagrangian_transmission_line()
    print(f"  Z0 = {tl['Z0']:.1f} Ohm")
    print(f"  v_p = {tl['v_phase']:.3e} m/s")
    print(f"  {tl['wave_eq']}")

    print("\n=== 5 SymPy equations ===")
    for name, eq in lagrangian_sympy_5().items():
        print(f"  {name}: {eq}")
