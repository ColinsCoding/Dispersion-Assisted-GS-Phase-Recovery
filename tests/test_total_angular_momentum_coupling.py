"""Test the L/S/J commutator verification (rigorous proof that J, not L
or S, is the sharp observable under spin-orbit coupling), term-symbol
construction, and MRI Larmor frequency against real clinical values
(1.5T -> ~63.9 MHz, 3T -> ~127.7 MHz proton resonance)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from fractions import Fraction
from dgs import total_angular_momentum_coupling as tam

# 1. angular_momentum_operators: standard commutation relation [Jx,Jy]=i*Jz
for j in [0.5, 1.0, 1.5]:
    Jx, Jy, Jz, J2 = tam.angular_momentum_operators(j)
    comm = tam.commutator(Jx, Jy)
    assert np.allclose(comm, 1j * Jz, atol=1e-9)
    # J^2 eigenvalue is j(j+1) on every state
    assert np.allclose(J2, j * (j + 1) * np.eye(J2.shape[0]), atol=1e-9)

# 2. verify_good_quantum_numbers: L_z, S_z NOT conserved; J_z, L^2, S^2, J^2 ARE
results = tam.verify_good_quantum_numbers(L=1, S=0.5)
assert not results["L_z"]["conserved"]
assert not results["S_z"]["conserved"]
assert results["J_z"]["conserved"]
assert results["L^2"]["conserved"]
assert results["S^2"]["conserved"]
assert results["J^2"]["conserved"]

# 3. same qualitative result holds for a different (L,S) pair -- not a fluke
#    of one specific case
results2 = tam.verify_good_quantum_numbers(L=2, S=0.5)
assert not results2["L_z"]["conserved"]
assert results2["J_z"]["conserved"]
assert results2["J^2"]["conserved"]

# 4. allowed_J_values: L=1,S=1/2 gives exactly {1/2, 3/2} (Clebsch-Gordan range)
J_values = tam.allowed_J_values(1, 0.5)
assert len(J_values) == 2
assert abs(J_values[0] - 0.5) < 1e-9
assert abs(J_values[1] - 1.5) < 1e-9

# 5. allowed_J_values for L=0 (S-term): only J=S itself
J_values_L0 = tam.allowed_J_values(0, 0.5)
assert len(J_values_L0) == 1
assert abs(J_values_L0[0] - 0.5) < 1e-9

# 6. term_symbol_string matches known hydrogen term symbols exactly
assert tam.term_symbol_string(0, 0.5, 0.5) == "2S1/2"
assert tam.term_symbol_string(1, 0.5, 0.5) == "2P1/2"
assert tam.term_symbol_string(1, 0.5, 1.5) == "2P3/2"

# 7. proton_larmor_frequency_hz matches real clinical MRI values
f_1_5T = tam.proton_larmor_frequency_hz(1.5) / 1e6
f_3T = tam.proton_larmor_frequency_hz(3.0) / 1e6
assert abs(f_1_5T - 63.87) < 0.1
assert abs(f_3T - 127.73) < 0.1
assert abs(f_3T / f_1_5T - 2.0) < 1e-9   # linear in B

# 8. quantum_number_as_fraction is exact, matches Fraction(3,2) for j=1.5
frac = tam.quantum_number_as_fraction(1.5)
assert frac == Fraction(3, 2)
frac_int = tam.quantum_number_as_fraction(1.0)
assert frac_int == Fraction(1, 1)

# 9. input validation
for bad_call in [
    lambda: tam.angular_momentum_operators(-1.0),
    lambda: tam.angular_momentum_operators(0.3),
    lambda: tam.spin_orbit_hamiltonian(1, 0.5, xi=-1.0),
    lambda: tam.allowed_J_values(-1, 0.5),
    lambda: tam.allowed_J_values(1, -0.5),
    lambda: tam.term_symbol_string(-1, 0.5, 0.5),
    lambda: tam.term_symbol_string(6, 0.5, 0.5),   # no letter code beyond H (L=5)
    lambda: tam.proton_larmor_frequency_hz(-1.0),
    lambda: tam.quantum_number_as_fraction(-1.0),
    lambda: tam.quantum_number_as_fraction(0.3),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.total_angular_momentum_coupling tests passed")
