"""Nuclide / ion notation  ^A X^q  ->  protons, neutrons, electrons -- and a loop
that tests every multiple-choice option (Knight Ch. 37.7 "Into the Nucleus").

Three rules, nothing else:
    protons   Z = atomic number of element X          (the symbol fixes Z)
    neutrons  N = A - Z                               (A = mass number, top-left)
    electrons   = Z - q                               (q = net charge, top-right;
                                                       a NEGATIVE ion has EXTRA electrons)
Companion MATLAB script: learn/matlab/nuclide_options.m (same loop, cross-checked).
"""
import re

import sympy as sp

ATOMIC_NUMBER = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Fe": 26, "Cu": 29, "Ag": 47, "Au": 79, "Pb": 82, "U": 92,
}

_PAT = re.compile(r"^\s*(\d+)\s*([A-Z][a-z]?)\s*(?:(\d*)\s*([+-]))?\s*$")


def parse(notation):
    """'16O2-' -> (A=16, symbol='O', q=-2).  '238U' -> (238, 'U', 0).  '4He2+' -> (4, 'He', +2)."""
    m = _PAT.match(notation)
    if not m:
        raise ValueError(f"cannot parse {notation!r}; use e.g. '16O2-', '23Na+', '238U'")
    A, sym, mag, sign = m.groups()
    if sym not in ATOMIC_NUMBER:
        raise ValueError(f"unknown element {sym!r}")
    q = 0
    if sign:
        q = (int(mag) if mag else 1) * (1 if sign == "+" else -1)
    return int(A), sym, q


def particle_counts(notation):
    """(protons, neutrons, electrons), with physical sanity checks."""
    A, sym, q = parse(notation)
    Z = ATOMIC_NUMBER[sym]
    N, e = A - Z, Z - q
    if N < 0:
        raise ValueError(f"mass number {A} is smaller than Z = {Z}")
    if e < 0:
        raise ValueError(f"charge {q:+d} would need {e} electrons")
    return Z, N, e


def symbolic_rules():
    """The three rules as SymPy equations, solved for (p, n, e)."""
    A, Z, q, p, n, e = sp.symbols("A Z q p n e", integer=True)
    eqs = [sp.Eq(p, Z), sp.Eq(p + n, A), sp.Eq(p - e, q)]
    return sp.solve(eqs, [p, n, e], dict=True)[0], (A, Z, q)


def check_options(notation, options):
    """Loop over (p, n, e) options; return a list of (index, option, correct?, reason)."""
    Z, N, e = particle_counts(notation)
    A, _, q = parse(notation)
    rows = []
    for i, opt in enumerate(options):
        if opt is None:
            rows.append((i, opt, False, "not a numerical option"))
            continue
        p_, n_, e_ = opt
        why = []
        if p_ != Z:
            why.append(f"protons {p_} != Z = {Z}" + (" (used the mass number)" if p_ == A else ""))
        if n_ != N:
            why.append(f"neutrons {n_} != A - Z = {N}" + (" (used A instead of A - Z)" if n_ == A else ""))
        if e_ != e:
            hint = ""
            if e_ == -q:
                hint = " (counted only the extra charge)"
            elif e_ == Z + q:
                hint = " (sign of the charge flipped)"
            elif e_ == A - q:
                hint = " (used the mass number A instead of Z)"
            why.append(f"electrons {e_} != Z - q = {e}" + hint)
        rows.append((i, opt, not why, "; ".join(why) or "all three rules satisfied"))
    return rows


# The options shown in the Mastering question for 16O2-
KNIGHT_OPTIONS = [(8, 8, 10), (8, 8, 2), None, (16, 16, 18), (16, 16, 2), (16, 0, 18)]
