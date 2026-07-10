"""The Gamma function Gamma(z): the factorial, continued to all real (and complex) numbers.

Of all the "gammas" in physics and math, this is the one that means a *function*. Factorial only
makes sense for whole numbers; Gamma is its smooth completion,
        Gamma(z) = integral_0^inf t^{z-1} e^{-t} dt,
matching Gamma(n) = (n-1)! at the integers but defined everywhere except the non-positive integers
(where it has poles). It is built from a handful of identities:
    * recurrence      Gamma(z+1) = z Gamma(z)          (why (n-1)! shows up, not n!)
    * half-integer    Gamma(1/2) = sqrt(pi)            (the Gaussian integral in disguise)
    * reflection      Gamma(z)Gamma(1-z) = pi/sin(pi z)
    * duplication     Gamma(z)Gamma(z+1/2) = 2^{1-2z} sqrt(pi) Gamma(2z)
    * Stirling        Gamma(z) ~ sqrt(2 pi/z) (z/e)^z  (the ln N! of thermodynamics/entropy)

It runs through physics wherever you count states or normalize a distribution: the VOLUME OF AN
n-BALL V_n = pi^{n/2}/Gamma(n/2+1) R^n (the phase-space volume behind statistical mechanics), the
GAUSSIAN MOMENTS integral_0^inf x^n e^{-x^2} dx = 1/2 Gamma((n+1)/2) (Maxwell-Boltzmann speed
moments, QFT integrals), and the Beta/Gamma/chi-squared distributions. The OTHER gammas are their own
modules -- the Lorentz factor (dgs.special_relativity), the heat-capacity ratio Cp/Cv
(dgs.degrees_of_freedom), the decay rate (griffiths_1_15), the reflection coefficient
(dgs.wave_reflection). This one is the function.

Implemented from scratch with the Lanczos approximation (~15 digits), checked against math.gamma and
the identities above. Pure Python (math/cmath); py-3.13.
"""

import math

# Lanczos approximation coefficients (g = 7, n = 9), good to ~15 significant digits.
_G = 7
_C = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
      771.32342877765313, -176.61502916214059, 12.507343278686905,
      -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]


def _is_nonpositive_integer(z, tol=1e-12):
    return z <= 0 and abs(z - round(z)) < tol


def gamma(z):
    """Gamma(z) for real z, via the Lanczos approximation (reflection for z < 1/2). Poles at the
    non-positive integers raise ValueError."""
    if _is_nonpositive_integer(z):
        raise ValueError("Gamma has poles at 0 and the negative integers")
    if z < 0.5:
        # reflection formula: Gamma(z) = pi / (sin(pi z) Gamma(1-z))
        return math.pi / (math.sin(math.pi * z) * gamma(1 - z))
    z -= 1
    a = _C[0]
    for i in range(1, _G + 2):
        a += _C[i] / (z + i)
    t = z + _G + 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * a


def log_gamma(z):
    """Natural log of |Gamma(z)| -- overflow-safe for large arguments (matches math.lgamma for
    z > 0). Uses the Lanczos series in log form."""
    if _is_nonpositive_integer(z):
        raise ValueError("Gamma has poles at 0 and the negative integers")
    if z < 0.5:
        return math.log(abs(math.pi / math.sin(math.pi * z))) - log_gamma(1 - z)
    z -= 1
    a = _C[0]
    for i in range(1, _G + 2):
        a += _C[i] / (z + i)
    t = z + _G + 0.5
    return 0.5 * math.log(2 * math.pi) + (z + 0.5) * math.log(t) - t + math.log(a)


def factorial(n):
    """n! = Gamma(n+1), valid for any real n > -1 (and the usual integers)."""
    if n <= -1:
        raise ValueError("factorial via Gamma needs n > -1")
    return gamma(n + 1)


def beta(x, y):
    """Beta function B(x,y) = Gamma(x)Gamma(y)/Gamma(x+y), computed in log space for stability."""
    if x <= 0 or y <= 0:
        raise ValueError("beta defined here for x, y > 0")
    return math.exp(log_gamma(x) + log_gamma(y) - log_gamma(x + y))


def stirling(z):
    """Stirling's leading approximation Gamma(z) ~ sqrt(2 pi/z) (z/e)^z."""
    if z <= 0:
        raise ValueError("stirling approximation used here for z > 0")
    return math.sqrt(2 * math.pi / z) * (z / math.e) ** z


def n_ball_volume(n, R=1.0):
    """Volume of an n-dimensional ball of radius R: V_n = pi^{n/2}/Gamma(n/2+1) R^n. Behind the
    phase-space volumes of statistical mechanics."""
    if n < 0 or R < 0:
        raise ValueError("n >= 0 and R >= 0")
    return math.pi ** (n / 2) / gamma(n / 2 + 1) * R ** n


def gaussian_moment(n):
    """The one-sided Gaussian moment integral_0^inf x^n e^{-x^2} dx = 1/2 Gamma((n+1)/2) -- the
    building block of Maxwell-Boltzmann speed averages and many QFT integrals."""
    if n < 0:
        raise ValueError("n must be >= 0")
    return 0.5 * gamma((n + 1) / 2)


if __name__ == "__main__":
    print("=== Gamma completes the factorial ===")
    for n in (1, 2, 3, 4, 5, 6):
        print(f"  Gamma({n}) = {gamma(n):8.3f}   (n-1)! = {math.factorial(n-1)}")
    print(f"  Gamma(1/2) = {gamma(0.5):.10f}  (sqrt(pi) = {math.sqrt(math.pi):.10f})")
    print(f"  Gamma(4.5) = {gamma(4.5):.6f}   (math.gamma = {math.gamma(4.5):.6f})")

    print("\n=== identities ===")
    z = 0.3
    print(f"  reflection  Gamma(z)Gamma(1-z) = {gamma(z)*gamma(1-z):.6f}  "
          f"pi/sin(pi z) = {math.pi/math.sin(math.pi*z):.6f}")
    print(f"  Stirling    Gamma(20)/stirling(20) = {gamma(20)/stirling(20):.6f}  (-> 1)")

    print("\n=== physics: n-ball volumes and Gaussian moments ===")
    for n, exact in [(1, "2R"), (2, "pi R^2"), (3, "4/3 pi R^3"), (4, "pi^2/2 R^4")]:
        print(f"  V_{n}(R=1) = {n_ball_volume(n):.6f}   ({exact})")
    print(f"  int_0^inf e^(-x^2) dx   = {gaussian_moment(0):.6f}  (sqrt(pi)/2 = {math.sqrt(math.pi)/2:.6f})")
    print(f"  int_0^inf x^2 e^(-x^2)  = {gaussian_moment(2):.6f}  (sqrt(pi)/4 = {math.sqrt(math.pi)/4:.6f})")
