"""Test dgs.semi_empirical_mass: the five liquid-drop terms (with the SYMMETRY
term zero at N=Z and growing as (N-Z)^2), binding energies matching measured
values to ~1%, the valley of stability Z*(A) landing on known isotopes with N>Z
for heavy nuclei, and B/A peaking in the iron region."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs import semi_empirical_mass as se

# 1. the individual terms
assert se.volume_term(56) == se.A_VOLUME * 56
assert math.isclose(se.surface_term(56), -se.A_SURFACE * 56 ** (2/3))
assert se.coulomb_term(26, 56) < 0                      # repulsion reduces binding
assert math.isclose(se.coulomb_term(26, 56), -se.A_COULOMB * 26 * 25 / 56 ** (1/3))

# 2. THE SYMMETRY TERM: zero at N=Z, and quadratic in the imbalance N-Z
assert math.isclose(se.symmetry_term(28, 56), 0.0, abs_tol=1e-12)   # N=Z=28 -> 0
assert se.symmetry_term(26, 56) < 0                                 # N != Z penalized
# doubling the imbalance (N-Z: 4 -> 8) quadruples the penalty
assert math.isclose(se.symmetry_term(24, 56) / se.symmetry_term(26, 56), 4.0)
# and it always reduces binding away from N=Z
assert se.symmetry_term(20, 56) < se.symmetry_term(26, 56) < 0

# 3. pairing: even-even bound MORE, odd-odd LESS, odd-A zero
assert se.pairing_term(26, 56) > 0                       # even-even (Fe-56)
assert se.pairing_term(25, 56) < 0                       # odd-odd (Z=25, N=31)... N=31 odd
assert se.pairing_term(26, 55) == 0.0                    # odd A
assert math.isclose(se.pairing_term(26, 56), se.A_PAIRING / math.sqrt(56))

# 4. total binding energy vs measured values, within ~1.5%
measured = {(20, 40): 342.05, (26, 56): 492.26, (50, 120): 1020.5, (92, 238): 1801.69}
for (Z, A), meas in measured.items():
    B = se.binding_energy(Z, A)
    assert abs(B - meas) / meas < 0.015, (Z, A, B, meas)
    assert math.isclose(se.binding_energy_per_nucleon(Z, A), B / A)

# 5. the valley of stability: Z*(A) matches known isotopes, curving to N>Z when heavy
assert se.most_stable_Z(16) == 8                         # O-16, N=Z
assert se.most_stable_Z(238) == 92                       # U-238
assert 24 <= se.most_stable_Z(56) <= 26                  # iron region
# light nuclei sit at N=Z; heavy nuclei have a neutron excess
assert se.most_stable_Z(16) == 16 - se.most_stable_Z(16)         # N == Z
Zh = se.most_stable_Z(238)
assert (238 - Zh) > Zh                                    # N > Z for heavy nuclei

# 6. B/A peaks in the iron-nickel region (most tightly bound nuclei)
assert 52 <= se.peak_mass_number() <= 66

# 7. kwarg bounds
for bad in (lambda: se.binding_energy(30, 20),           # Z > A
            lambda: se.volume_term(0),
            lambda: se.symmetry_term(5, 0),
            lambda: se.most_stable_Z(0)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_semi_empirical_mass: all checks passed")
