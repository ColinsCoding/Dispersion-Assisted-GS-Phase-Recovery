"""Complex analysis you can *compute* -- no symbolic formalization required.

The big theorems of complex analysis are usually taught as pencil-and-paper
identities, but every one of them is just a contour integral you can evaluate
numerically by marching around a circle. If you can write the function as code,
you can get the residue, recover the value, count the zeros -- numerically and
rigorously. That is the same move the whole lab runs on: the field is complex
data, so you compute with it.

  * contour_integral  -- march z = c + R e^{i theta} and sum f(z) dz
  * residue theorem   -- oint f dz = 2 pi i * (sum of residues inside)
  * Cauchy's formula  -- f(z0) = (1/2 pi i) oint f(z)/(z - z0) dz
  * argument principle -- (1/2 pi i) oint f'/f dz = (#zeros - #poles) inside,
    here computed as the winding number of f(z) about the origin.

This is the analyticity that underlies the repo's Kramers-Kronig relation
(causality -> f analytic in the upper half-plane -> absorption and dispersion
are linked). NumPy only. Education.
"""

import numpy as np


def contour_integral(f, center, radius, n=4000):
    """Numerically integrate oint f(z) dz counterclockwise around |z-center|=radius.

    Parametrize z = center + R e^{i theta}, dz = i R e^{i theta} dtheta, and sum.
    """
    if radius <= 0 or n < 16:
        raise ValueError("radius > 0 and n >= 16 required")
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    z = center + radius * np.exp(1j * theta)
    dz = 1j * radius * np.exp(1j * theta) * (2 * np.pi / n)
    return np.sum(f(z) * dz)


def residues_sum(f, center, radius, n=4000):
    """Sum of residues of f inside the contour: (1/2 pi i) oint f dz."""
    return contour_integral(f, center, radius, n) / (2j * np.pi)


def cauchy_value(f, z0, center, radius, n=4000):
    """Recover f(z0) from boundary values: (1/2 pi i) oint f(z)/(z - z0) dz
    (z0 must lie strictly inside the contour). The value in the interior is
    fixed entirely by the values on the rim -- that is analyticity."""
    if abs(z0 - center) >= radius:
        raise ValueError("z0 must be strictly inside the contour")
    return contour_integral(lambda z: f(z) / (z - z0), center, radius, n) / (2j * np.pi)


def winding_number(f, center, radius, n=20000):
    """(#zeros - #poles) of f inside the contour, via the argument principle.

    As z runs once around the loop, f(z) winds around the origin an integer
    number of times; that integer is the count. Computed by unwrapping arg f(z)
    -- purely numerical, no derivative needed.
    """
    theta = np.linspace(0, 2 * np.pi, n, endpoint=True)
    w = f(center + radius * np.exp(1j * theta))
    if np.any(np.abs(w) < 1e-12):
        raise ValueError("f has a zero on the contour; move the contour")
    return int(round((np.unwrap(np.angle(w))[-1] - np.angle(w)[0]) / (2 * np.pi)))


if __name__ == "__main__":
    # oint dz/(z^2+1) around a circle enclosing only z = i  ->  2 pi i * 1/(2i) = pi
    val = contour_integral(lambda z: 1 / (z**2 + 1), center=1j, radius=0.5)
    print(f"oint dz/(z^2+1) around z=i : {val:.6f}   (analytic: pi = {np.pi:.6f})")
    # Cauchy: recover e^z at z0 = 0.5 from a contour of radius 2
    print(f"Cauchy value of e^z at 0.5 : {cauchy_value(np.exp, 0.5, 0, 2):.6f}"
          f"   (e^0.5 = {np.e**0.5:.6f})")
    # argument principle: z^3 has 3 zeros inside the unit circle
    print(f"winding number of z^3      : {winding_number(lambda z: z**3, 0, 1)}")
