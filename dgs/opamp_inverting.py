"""Inverting op-amp amplifier (Nilsson & Riedel, Electric Circuits, Probs. 5.1-5.2),
solved in SymPy, including the step textbooks skip: WHY the ideal rules hold.

Circuit: v_s -> R1 = 4 k -> inverting input (v_n); Rf = 12 k from v_n to v_o;
non-inverting input v_p grounded; supplies +-15 V; load 6 k from v_o to ground.

Ideal constraints:  i_n = i_p = 0  (infinite input resistance)
                    v_p = v_n      (virtual short, valid only WITH negative feedback
                                     and while the output is not saturated)
Finite-gain model:  v_o = A (v_p - v_n); A -> oo recovers v_o = -(Rf/R1) v_s.
Saturation:         v_o = clip(-(Rf/R1) v_s, -15 V, +15 V).
"""
import numpy as np
import sympy as sp

R1_DEFAULT, RF_DEFAULT, RL_DEFAULT, VCC = 4e3, 12e3, 6e3, 15.0
PROBLEM_5_2_SOURCES = [-6.0, -3.5, -1.25, 2.4, 4.5, 5.4]


def ideal_solution():
    """KCL at v_n with i_n = 0 and v_n = v_p = 0 -> v_o in terms of v_s, R1, Rf."""
    vs, vo, vn, R1, Rf = sp.symbols("v_s v_o v_n R_1 R_f", real=True)
    kcl = sp.Eq((vs - vn) / R1 + (vo - vn) / Rf, 0)          # i_n = 0: no current into the op amp
    return sp.solve(kcl.subs(vn, 0), vo)[0], (vs, R1, Rf)


def finite_gain_solution():
    """Same KCL, but v_o = A (v_p - v_n) with v_p = 0; returns v_o(A) and its A -> oo limit."""
    vs, R1, Rf, A = sp.symbols("v_s R_1 R_f A", positive=True)
    vo, vn = sp.symbols("v_o v_n", real=True)        # NOT positive: the inverting output is negative
    sol = sp.solve([sp.Eq((vs - vn) / R1 + (vo - vn) / Rf, 0), sp.Eq(vo, A * (0 - vn))], [vo, vn], dict=True)[0]
    vo_A = sp.simplify(sol[vo])
    return vo_A, sp.limit(vo_A, A, sp.oo), sp.simplify(sol[vn]), (vs, R1, Rf, A)


def output_voltage(vs, R1=R1_DEFAULT, Rf=RF_DEFAULT, vcc=VCC):
    """Ideal op amp with rail saturation at +-vcc."""
    if R1 <= 0 or Rf <= 0 or vcc <= 0:
        raise ValueError("resistances and supply must be positive")
    return np.clip(-Rf / R1 * np.asarray(vs, float), -vcc, vcc)


def is_saturated(vs, R1=R1_DEFAULT, Rf=RF_DEFAULT, vcc=VCC):
    return np.abs(Rf / R1 * np.asarray(vs, float)) > vcc


def linear_input_range(R1=R1_DEFAULT, Rf=RF_DEFAULT, vcc=VCC):
    """|v_s| <= vcc R1 / Rf keeps the output off the rails."""
    v = vcc * R1 / Rf
    return -v, v


def output_current(vs, R1=R1_DEFAULT, Rf=RF_DEFAULT, RL=RL_DEFAULT, vcc=VCC):
    """Current the op amp output must source/sink: into the load plus back through Rf (v_n = 0)."""
    vo = output_voltage(vs, R1, Rf, vcc)
    return vo / RL + vo / Rf
