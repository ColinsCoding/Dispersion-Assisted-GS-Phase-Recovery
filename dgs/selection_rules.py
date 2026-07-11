"""Selection rules: which atomic transitions are allowed, which are forbidden, and why it matters.

A hydrogen atom drops from one (n, l, m) state to another by emitting a photon that carries off the
energy difference: energy conservation gives delta_E = h f = h c / lambda. But not every pair of
levels can be connected -- the photon carries one unit of angular momentum and has odd parity, so an
ELECTRIC-DIPOLE transition is only ALLOWED when

    Delta_l = +-1        (the orbital angular momentum must change by exactly one)
    Delta_m = 0, +-1     (set by the photon's polarization: pi for 0, sigma+- for +-1)

with Delta_n unrestricted. Equivalently (the LAPORTE rule) the parity (-1)^l must flip. Anything else
is FORBIDDEN to first order, and these rules are exactly why an atomic spectrum shows some lines
bright and others missing -- they are the grammar of the spectrum.

The rules also set TIMING. A state that has an allowed dipole decay empties in nanoseconds. A state
with NO allowed dipole decay is METASTABLE and lives vastly longer: hydrogen's 2s state can only reach
1s, but that is Delta_l = 0 (forbidden), so 2s survives ~0.12 s -- a hundred million times longer than
2p's ~1.6 ns. Metastable states are the workhorses of lasers (population inversion) and atomic clocks.

Builds full (n,l,m) transition reports, enumerates a state's allowed decays, flags metastability, and
attaches the photon energy/wavelength/frequency (via dgs.hydrogen_atom). Complements
dgs.feynman_atomic_molecular.selection_rule_check (the bare l-only predicate). NumPy/math; py-3.13.
"""

import math
from dgs import hydrogen_atom as H

_H_PLANCK = 6.62607015e-34     # J s
_C = 299792458.0               # m/s
_EV = 1.602176634e-19          # J


def is_dipole_allowed(l_i, m_i, l_f, m_f):
    """True if the electric-dipole rules Delta_l = +-1 and Delta_m in {-1,0,1} are both satisfied."""
    return abs(l_i - l_f) == 1 and abs(m_i - m_f) <= 1


def parity(l):
    """Spatial parity of an orbital: (-1)^l (even for s,d,...; odd for p,f,...)."""
    return (-1) ** l


def parity_flips(l_i, l_f):
    """Laporte rule: an electric-dipole transition must change parity. Equivalent to Delta_l odd."""
    return parity(l_i) != parity(l_f)


def polarization(m_i, m_f):
    """Photon polarization implied by Delta_m: pi (linear) for 0, sigma+ / sigma- for +-1.
    None if |Delta_m| > 1 (no dipole photon carries it)."""
    dm = m_f - m_i
    return {0: "pi (linear)", 1: "sigma- (Delta m=+1)", -1: "sigma+ (Delta m=-1)"}.get(dm, None)


def forbidden_reason(l_i, m_i, l_f, m_f):
    """Plain-language reason a transition is forbidden, or 'allowed'."""
    if abs(l_i - l_f) != 1:
        return f"Delta_l = {l_f - l_i:+d} (electric dipole requires +-1)"
    if abs(m_i - m_f) > 1:
        return f"Delta_m = {m_f - m_i:+d} (electric dipole requires 0, +-1)"
    return "allowed"


def transition_report(state_i, state_f, Z=1):
    """Full report for (n_i,l_i,m_i) -> (n_f,l_f,m_f): allowed?, the delta rules, parity flip,
    polarization, and -- if it emits (E_i > E_f) -- the photon energy/wavelength/frequency."""
    ni, li, mi = state_i
    nf, lf, mf = state_f
    if not (H.valid_state(ni, li, mi) and H.valid_state(nf, lf, mf)):
        raise ValueError("both states must be valid (n>=1, 0<=l<n, |m|<=l)")
    allowed = is_dipole_allowed(li, mi, lf, mf)
    rep = {"initial": state_i, "final": state_f,
           "Delta_n": nf - ni, "Delta_l": lf - li, "Delta_m": mf - mi,
           "allowed": allowed, "reason": forbidden_reason(li, mi, lf, mf),
           "parity_flips": parity_flips(li, lf),
           "polarization": polarization(mi, mf)}
    E_i, E_f = H.energy_level(ni, Z), H.energy_level(nf, Z)
    if E_i > E_f:                                   # emission
        dE = H.transition_energy(ni, nf, Z)
        rep["photon_eV"] = dE
        rep["wavelength_nm"] = H.transition_wavelength_nm(ni, nf, Z)
        rep["frequency_Hz"] = dE * _EV / _H_PLANCK
    return rep


def allowed_decays(n, l, m, Z=1):
    """Every lower-energy state (n_f < n) this state can reach by an allowed electric-dipole
    transition. Empty for the ground state and for metastable states."""
    if not H.valid_state(n, l, m):
        raise ValueError("invalid state")
    out = []
    for nf in range(1, n):                          # lower energy = smaller n
        for lf in H.allowed_l(nf):
            for mf in (m - 1, m, m + 1):
                if H.valid_state(nf, lf, mf) and is_dipole_allowed(l, m, lf, mf):
                    out.append((nf, lf, mf))
    return out


def is_metastable(n, l, m):
    """True if an EXCITED state has no allowed dipole decay (so it lives anomalously long). The
    ground state is stable, not metastable. Hydrogen's 2s (n=2,l=0) is the classic example."""
    if not H.valid_state(n, l, m):
        raise ValueError("invalid state")
    if n == 1:
        return False                                # ground state: stable
    return len(allowed_decays(n, l, m)) == 0


def lifetime_class(n, l, m):
    """Qualitative lifetime from the selection rules: stable ground state, fast allowed dipole
    decay (~ns), or metastable (dipole-forbidden, long-lived)."""
    if (n, l, m) == (1, 0, 0):
        return "stable (ground state)"
    if is_metastable(n, l, m):
        return "metastable (dipole-forbidden, long-lived ~ms-s)"
    return "allowed (electric dipole, ~ns)"


if __name__ == "__main__":
    print("=== allowed vs forbidden (n,l,m) -> (n,l,m) ===")
    cases = [((2,1,0), (1,0,0)), ((2,0,0), (1,0,0)), ((3,2,0), (2,1,0)),
             ((3,2,0), (1,0,0)), ((3,1,1), (2,0,0)), ((3,2,2), (2,1,0))]
    for si, sf in cases:
        r = transition_report(si, sf)
        tag = "ALLOWED " if r["allowed"] else "forbidden"
        extra = (f"  {r.get('wavelength_nm',0):.1f} nm, {r['polarization']}"
                 if r["allowed"] and "wavelength_nm" in r else f"  ({r['reason']})")
        print(f"  {si} -> {sf}: {tag}{extra}")

    print("\n=== timing: the metastable 2s state ===")
    for st in [(2,1,0), (2,0,0), (3,0,0), (1,0,0)]:
        print(f"  n={st[0]} l={st[1]}: decays={allowed_decays(*st)}  -> {lifetime_class(*st)}")

    print("\n=== energy conservation: Lyman-alpha 2p -> 1s ===")
    r = transition_report((2,1,0), (1,0,0))
    print(f"  delta_E = {r['photon_eV']:.3f} eV = h c/lambda,  lambda = {r['wavelength_nm']:.1f} nm,"
          f"  f = {r['frequency_Hz']:.3e} Hz")
