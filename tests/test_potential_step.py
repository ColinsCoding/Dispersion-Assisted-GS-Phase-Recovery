"""Test dgs.potential_step: Fresnel-form reflection, flux conservation R+T=1, total reflection with
an evanescent tail for E<U, the E->U crossover, phase shifts, and quantum reflection off a step down."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from dgs import potential_step as ps

m = hbar = 1.0
U = 1.0

# 1. E > U: closed forms for B/A, R, T and the Fresnel identity
for E in (1.2, 2.0, 5.0, 20.0):
    k = ps.wavevector(E); kp = ps.transmitted_wavevector(E, U)
    assert math.isclose(ps.reflection_coefficient(E, U), ((k-kp)/(k+kp))**2, rel_tol=1e-12)
    assert math.isclose(ps.transmission_coefficient(E, U), 4*k*kp/(k+kp)**2, rel_tol=1e-12)
    # R is exactly the Fresnel reflectance with k in the role of refractive index
    assert math.isclose(ps.reflection_coefficient(E, U), ps.fresnel_reflectance(k, kp), rel_tol=1e-12)

# 2. flux conservation R + T = 1 for E > U
for E in (1.1, 1.5, 3.0, 8.0):
    assert math.isclose(ps.reflection_coefficient(E, U) + ps.transmission_coefficient(E, U),
                        1.0, rel_tol=1e-12)

# 3. classical limit: E >> U gives near-total transmission
assert ps.transmission_coefficient(1000*U, U) > 0.99
assert ps.reflection_coefficient(1000*U, U) < 0.01

# 4. E < U: total reflection, zero transmission, but a nonzero evanescent field
for E in (0.1, 0.5, 0.9):
    assert math.isclose(ps.reflection_coefficient(E, U), 1.0, rel_tol=1e-12)   # R = 1
    assert ps.transmission_coefficient(E, U) == 0.0                            # no propagating flux
    amp = ps.step_amplitudes(E, U)
    assert abs(amp["C_over_A"])**2 > 0                                         # field leaks in
    assert ps.penetration_depth(E, U) > 0

# 5. penetration depth grows as E climbs toward U, and matches hbar/sqrt(2m(U-E))
assert ps.penetration_depth(0.9, U) > ps.penetration_depth(0.1, U)
assert math.isclose(ps.penetration_depth(0.5, U), 1/np.sqrt(2*m*(U-0.5))/hbar, rel_tol=1e-12)

# 6. E -> U crossover is continuous: R -> 1, T -> 0 from above (approach is R ~ 1 - 4 sqrt((E-U)/E))
assert math.isclose(ps.reflection_coefficient(1.000001*U, U), 1.0, rel_tol=2e-2)
assert ps.transmission_coefficient(1.000001*U, U) < 0.02
assert ps.reflection_coefficient(U, U) == 1.0                                 # critical case
# and reflection rises monotonically toward 1 as E descends to U
assert (ps.reflection_coefficient(1.05*U, U) > ps.reflection_coefficient(1.5*U, U)
        > ps.reflection_coefficient(5*U, U))

# 7. reflected phase: 0 for a step up (E>U), pi for a step down, in (-pi,0) for E<U (TIR shift)
assert math.isclose(ps.reflection_phase(4.0, U), 0.0, abs_tol=1e-9)           # step up, k>k' -> B/A>0
assert math.isclose(abs(ps.reflection_phase(1.0, -3.0)), math.pi, abs_tol=1e-9)  # step down -> B/A<0
ph = ps.reflection_phase(0.5, U)
assert -math.pi < ph < 0                                                      # evanescent phase shift

# 8. quantum reflection off a step DOWN (E>U, U<0): R>0 though classically it always transmits
Rdown = ps.reflection_coefficient(1.0, -3.0)
assert 0 < Rdown < 1
assert math.isclose(ps.reflection_coefficient(1.0,-3.0) + ps.transmission_coefficient(1.0,-3.0),
                    1.0, rel_tol=1e-12)

# 9. |B/A| = 1 exactly in the evanescent regime (unit-modulus reflection)
for E in (0.2, 0.6, 0.95):
    assert math.isclose(abs(ps.step_amplitudes(E, U)["B_over_A"]), 1.0, rel_tol=1e-12)

# 10. kwarg bounds
for bad in (lambda: ps.wavevector(0),
            lambda: ps.transmitted_wavevector(0.5, U),      # E<U
            lambda: ps.decay_constant(2.0, U),              # E>U
            lambda: ps.fresnel_reflectance(0, 1)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_potential_step: all checks passed")
