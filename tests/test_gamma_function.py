"""Test dgs.gamma_function: the factorial link, Gamma(1/2)=sqrt(pi), recurrence, reflection and
duplication identities, agreement with math.gamma/lgamma, Beta, Stirling, n-ball volumes, and
Gaussian moments."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dgs import gamma_function as gf

# 1. Gamma(n) = (n-1)! at the positive integers
for n in range(1, 12):
    assert math.isclose(gf.gamma(n), math.factorial(n-1), rel_tol=1e-10)
assert math.isclose(gf.factorial(5), 120, rel_tol=1e-10)          # n! = Gamma(n+1)

# 2. Gamma(1/2) = sqrt(pi), and half-integers
assert math.isclose(gf.gamma(0.5), math.sqrt(math.pi), rel_tol=1e-12)
assert math.isclose(gf.gamma(1.5), math.sqrt(math.pi)/2, rel_tol=1e-12)
assert math.isclose(gf.gamma(2.5), 3*math.sqrt(math.pi)/4, rel_tol=1e-12)

# 3. recurrence Gamma(z+1) = z Gamma(z)
for z in (0.7, 1.3, 2.9, 5.5, 8.2):
    assert math.isclose(gf.gamma(z+1), z*gf.gamma(z), rel_tol=1e-11)

# 4. agreement with math.gamma across positive and negative-non-integer arguments
for z in (0.1, 0.5, 1.0, 2.3, 4.5, 7.7, 10.0, -0.5, -1.5, -2.3):
    assert math.isclose(gf.gamma(z), math.gamma(z), rel_tol=1e-9), z

# 5. reflection formula Gamma(z)Gamma(1-z) = pi/sin(pi z)
for z in (0.2, 0.35, 0.5, 0.8, 1.4):
    assert math.isclose(gf.gamma(z)*gf.gamma(1-z), math.pi/math.sin(math.pi*z), rel_tol=1e-9)

# 6. Legendre duplication Gamma(z)Gamma(z+1/2) = 2^{1-2z} sqrt(pi) Gamma(2z)
for z in (0.6, 1.0, 1.7, 3.0):
    lhs = gf.gamma(z)*gf.gamma(z+0.5)
    rhs = 2.0**(1-2*z)*math.sqrt(math.pi)*gf.gamma(2*z)
    assert math.isclose(lhs, rhs, rel_tol=1e-9), z

# 7. log_gamma matches math.lgamma for z > 0 and is overflow-safe for large z
# (abs_tol needed because lgamma(1)=lgamma(2)=0 exactly -- rel_tol can't compare to zero)
for z in (0.3, 1.0, 5.0, 50.0, 200.0):
    assert math.isclose(gf.log_gamma(z), math.lgamma(z), rel_tol=1e-9, abs_tol=1e-9)
assert math.isclose(gf.log_gamma(171.0), math.lgamma(171.0), rel_tol=1e-9)   # gamma(171) overflows a float

# 8. Beta function B(x,y) = Gamma(x)Gamma(y)/Gamma(x+y), symmetric, B(1,1)=1
assert math.isclose(gf.beta(2, 3), gf.gamma(2)*gf.gamma(3)/gf.gamma(5), rel_tol=1e-10)
assert math.isclose(gf.beta(2.5, 4.1), gf.beta(4.1, 2.5), rel_tol=1e-12)      # symmetric
assert math.isclose(gf.beta(1, 1), 1.0, rel_tol=1e-12)
assert math.isclose(gf.beta(0.5, 0.5), math.pi, rel_tol=1e-9)                 # B(1/2,1/2)=pi

# 9. Stirling approximation approaches Gamma for large z
assert math.isclose(gf.gamma(50)/gf.stirling(50), 1.0, rel_tol=2e-3)
assert abs(gf.gamma(100)/gf.stirling(100) - 1) < abs(gf.gamma(10)/gf.stirling(10) - 1)  # improves

# 10. n-ball volumes: V1=2R, V2=pi R^2, V3=4/3 pi R^3, V4=pi^2/2 R^4
assert math.isclose(gf.n_ball_volume(1, 2.0), 2*2.0, rel_tol=1e-10)
assert math.isclose(gf.n_ball_volume(2, 3.0), math.pi*3.0**2, rel_tol=1e-10)
assert math.isclose(gf.n_ball_volume(3, 1.0), 4/3*math.pi, rel_tol=1e-10)
assert math.isclose(gf.n_ball_volume(4, 1.0), math.pi**2/2, rel_tol=1e-10)

# 11. Gaussian moments int_0^inf x^n e^{-x^2} dx = 1/2 Gamma((n+1)/2)
assert math.isclose(gf.gaussian_moment(0), math.sqrt(math.pi)/2, rel_tol=1e-10)   # 1/2 sqrt(pi)
assert math.isclose(gf.gaussian_moment(1), 0.5, rel_tol=1e-10)
assert math.isclose(gf.gaussian_moment(2), math.sqrt(math.pi)/4, rel_tol=1e-10)
assert math.isclose(gf.gaussian_moment(3), 0.5, rel_tol=1e-10)                    # 1/2 Gamma(2)=1/2

# 12. poles and bounds
for bad in (lambda: gf.gamma(0), lambda: gf.gamma(-1), lambda: gf.gamma(-3),
            lambda: gf.factorial(-2), lambda: gf.beta(0, 1),
            lambda: gf.n_ball_volume(-1), lambda: gf.gaussian_moment(-1)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_gamma_function: all checks passed")
