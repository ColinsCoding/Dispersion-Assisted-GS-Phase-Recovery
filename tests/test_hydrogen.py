"""Smoke-test the hydrogen/orbital/spin additions to griffiths.quantum."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import sympy as sp
from griffiths import quantum as q

# energy levels: E_1 = -13.6 eV, E_2 = -3.4 eV
for n in (1, 2, 3):
    print(f"E_{n} = {q.hydrogen_energy(n):.3f} eV")
assert abs(q.hydrogen_energy(2) - (-3.4014)) < 1e-3

# radial wavefunction R_10 = 2 exp(-r)
R10, r = q.hydrogen_radial(1, 0)
print("\nR_10 =", R10, " (expect 2 e^-r)")
print("R_21 =", q.hydrogen_radial(2, 1)[0])

# transitions -> photons: Balmer alpha (3->2) ~ 656 nm, Lyman alpha (2->1) ~ 121.5 nm
print("\nBalmer-alpha (3->2):", round(q.transition_wavelength(3, 2), 1), "nm (expect ~656)")
print("Lyman-alpha  (2->1):", round(q.transition_wavelength(2, 1), 1), "nm (expect ~121.5)")
print("Paschen      (4->3):", round(q.transition_wavelength(4, 3), 1), "nm (IR)")
assert abs(q.transition_wavelength(3, 2) - 656.3) < 1.0

# selection rules: s->p allowed (dl=1), s->d forbidden (dl=2), s->s forbidden
print("\n1s->2p allowed (dl=1):", q.selection_rule_allowed(0, 1))
print("1s->3d allowed (dl=2):", q.selection_rule_allowed(0, 2), "(forbidden)")

# orbital angular factors: pz max along z (theta=0), zero in xy-plane (theta=pi/2)
print("\npz at theta=0:", round(float(q.real_orbital_angular("pz", 0.0, 0.0)), 4),
      " at theta=pi/2:", round(float(q.real_orbital_angular("pz", np.pi/2, 0.0)), 4))

# spin states: theta=0 -> up (1,0); theta=pi -> down (0,1); theta=pi/2 -> equal superposition
print("\nspin up   (theta=0):  ", np.round(q.spin_state(0.0), 3))
print("spin down (theta=pi): ", np.round(np.abs(q.spin_state(np.pi)), 3))
chi = q.spin_state(np.pi/2)
print("spin +x   (theta=pi/2):", np.round(np.abs(chi)**2, 3), "(equal up/down prob)")
assert np.allclose(np.abs(chi)**2, [0.5, 0.5])

# validation
for bad in [lambda: q.hydrogen_energy(0),
            lambda: q.hydrogen_radial(2, 2),
            lambda: q.transition_wavelength(2, 3),
            lambda: q.real_orbital_angular("foo", 0, 0)]:
    try:
        bad()
    except ValueError as e:
        print("err ok:", str(e)[:50])
print("SMOKE PASS")
