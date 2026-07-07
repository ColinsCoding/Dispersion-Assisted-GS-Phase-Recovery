"""Charge configurations: which multipole dominates, and which way the field points.

Put some point charges down and, far away, their potential is a sum of multipole
terms that fall off at different rates -- the first NONZERO one wins:

    monopole  (net charge)   V ~ 1/r     E ~ 1/r^2    parity even  (l=0)
    dipole    (separation)   V ~ 1/r^2   E ~ 1/r^3    parity ODD   (l=1)
    quadrupole               V ~ 1/r^3   E ~ 1/r^4    parity even  (l=2)

A neutral pair of +q and -q has no monopole, so its leading behavior is the DIPOLE,
with moment p = sum q_i r_i -- a VECTOR that points from the negative charge to the
positive one. That direction is the "which way": the dipole field
    E(r) = k [ 3 (p . rhat) rhat - p ] / r^3
points ALONG p on the axis (2kp/r^3) and OPPOSITE to p on the equator (-kp/r^3).

The parity column is the dgs.even_odd story in disguise: under r -> -r the l-th
multipole term is multiplied by (-1)^l, so it is an eigenvector of the parity
operator with eigenvalue +/-1 -- monopole even, dipole odd, quadrupole even. Which
multipole survives is fixed by the SYMMETRY of the charge arrangement.

Complements dgs.electrostatics_multipoles (which has the moments and the potential
expansion) with the field VECTOR, its direction, and the leading-term / parity
classification, verified by shrinking a real two-charge dipole down to the ideal
point-dipole formula. NumPy only; py-3.13.
"""

import numpy as np

K_COULOMB = 8.9875517873681764e9      # 1/(4 pi eps0), N m^2 / C^2


def net_charge(charges):
    """The monopole moment: total charge. Nonzero -> the field is monopole
    (1/r^2) far away, whatever else is going on."""
    return float(np.sum(charges))


def dipole_moment(charges, positions):
    """p = sum q_i r_i, the dipole moment VECTOR. For a neutral configuration it
    is origin-independent and points from the negative charge toward the positive."""
    charges = np.asarray(charges, float)
    positions = np.asarray(positions, float)
    if positions.shape != (len(charges), 3):
        raise ValueError("positions must be (N,3) matching charges")
    return charges @ positions


def dipole_direction(charges, positions, tol=1e-12):
    """Unit vector of p -- literally 'which way' the dipole points (- to +).
    Raises if the configuration has no dipole moment."""
    p = dipole_moment(charges, positions)
    n = np.linalg.norm(p)
    if n < tol:
        raise ValueError("configuration has no dipole moment (|p| ~ 0)")
    return p / n


def coulomb_field(charges, positions, field_point, k=K_COULOMB):
    """The EXACT electric field vector at field_point, by Coulomb superposition
    E = sum k q_i (r - r_i)/|r - r_i|^3. The ground truth every multipole
    approximation is trying to match."""
    charges = np.asarray(charges, float)
    positions = np.asarray(positions, float)
    r = np.asarray(field_point, float)
    E = np.zeros(3)
    for q, ri in zip(charges, positions):
        d = r - ri
        dist = np.linalg.norm(d)
        if dist < 1e-15:
            raise ValueError("field point coincides with a charge")
        E += k * q * d / dist ** 3
    return E


def field_direction(charges, positions, field_point, k=K_COULOMB, tol=1e-30):
    """Unit vector of the exact field at a point -- which way a test charge is
    pushed there."""
    E = coulomb_field(charges, positions, field_point, k)
    n = np.linalg.norm(E)
    if n < tol:
        raise ValueError("field is zero at this point (a null)")
    return E / n


def dipole_field(p, field_point, k=K_COULOMB):
    """The ideal point-dipole field E = k[3(p.rhat)rhat - p]/r^3. On the axis it
    is 2kp/r^3 along p; on the equator it is -kp/r^3, opposite p."""
    p = np.asarray(p, float)
    r = np.asarray(field_point, float)
    dist = np.linalg.norm(r)
    if dist < 1e-15:
        raise ValueError("field point is at the dipole")
    rhat = r / dist
    return k * (3 * np.dot(p, rhat) * rhat - p) / dist ** 3


def multipole_parity(order):
    """Parity of the order-l multipole term under r -> -r: (-1)^l. Returns
    (sign, label). Monopole (l=0) even, dipole (l=1) odd, quadrupole (l=2) even
    -- the eigenvalue of the dgs.even_odd parity operator for that term."""
    if order < 0:
        raise ValueError("multipole order must be >= 0")
    sign = (-1) ** order
    return sign, ("even" if sign > 0 else "odd")


def leading_multipole(charges, positions, tol=1e-9):
    """Classify the first nonzero multipole -- the term that dominates the far
    field. Returns dict with term name, l, potential falloff (1/r^(l+1)), field
    falloff (1/r^(l+2)), and parity. Set by the configuration's symmetry."""
    charges = np.asarray(charges, float)
    if abs(net_charge(charges)) > tol:
        l, term = 0, "monopole"
    elif np.linalg.norm(dipole_moment(charges, positions)) > tol:
        l, term = 1, "dipole"
    else:
        l, term = 2, "quadrupole"      # first neutral, dipole-free nonzero term
    sign, par = multipole_parity(l)
    return {"term": term, "l": l, "potential_falloff": l + 1,
            "field_falloff": l + 2, "parity": par}


if __name__ == "__main__":
    # a physical dipole: +1 C at +z, -1 C at -z, separation a=0.02 -> p = (0,0,0.02)
    a = 0.02
    charges = [1.0, -1.0]
    positions = [[0, 0, a/2], [0, 0, -a/2]]
    p = dipole_moment(charges, positions)
    print("dipole moment p =", np.round(p, 4), " points", np.round(dipole_direction(charges, positions), 3))
    print("net charge:", net_charge(charges), " -> leading term:", leading_multipole(charges, positions))

    # which way does the field point? on the axis (along p) vs the equator
    on_axis = coulomb_field(charges, positions, [0, 0, 5.0])
    equator = coulomb_field(charges, positions, [5.0, 0, 0])
    print("\nfield on axis  (0,0,5):", np.round(field_direction(charges, positions, [0,0,5.0]), 3),
          "  (along +p)")
    print("field on equator(5,0,0):", np.round(field_direction(charges, positions, [5.0,0,0]), 3),
          "  (opposite p, i.e. -z)")

    # the exact field matches the ideal point-dipole formula far away
    print("\nexact vs point-dipole at (0,0,5):")
    print("  exact      :", np.round(on_axis, 8))
    print("  point dipole:", np.round(dipole_field(p, [0, 0, 5.0]), 8))

    print("\nmultipole parity (the even/odd story):")
    for l in (0, 1, 2, 3):
        s, lab = multipole_parity(l)
        print(f"  l={l}: parity {lab} ({s:+d})")
