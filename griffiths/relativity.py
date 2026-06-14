"""Special relativity for Griffiths Ch. 12 (Problems 12.17, 12.19, 12.20, 12.30).

Conventions follow Griffiths: x^0 = ct, metric eta = diag(-1, +1, +1, +1),
so the invariant interval is I = -(c dt)^2 + |dr|^2.
"""

import sympy as sp

ETA = sp.diag(-1, 1, 1, 1)

_AXES = {"x": 1, "y": 2, "z": 3}


def _check_beta(beta):
    beta = sp.sympify(beta)
    if beta.is_number and not abs(beta) < 1:
        raise ValueError(f"|beta| must be < 1, got beta={beta}")
    return beta


def _as_event(e, name="event"):
    v = sp.Matrix(e)
    if v.shape == (1, 4):
        v = v.T
    if v.shape != (4, 1):
        raise ValueError(f"{name} must be 4 components (ct, x, y, z), got shape {v.shape}")
    return v


def boost(beta, axis="x"):
    """4x4 Lorentz boost matrix for velocity v = beta*c along the given axis."""
    beta = _check_beta(beta)
    if axis not in _AXES:
        raise ValueError(f"axis must be one of {sorted(_AXES)}, got {axis!r}")
    g = 1 / sp.sqrt(1 - beta**2)
    L = sp.eye(4)
    i = _AXES[axis]
    L[0, 0] = L[i, i] = g
    L[0, i] = L[i, 0] = -g * beta
    return L


def boost_rapidity(theta, axis="x"):
    """4x4 boost written with the rapidity theta = atanh(beta) -- Problem 12.19(a)."""
    if axis not in _AXES:
        raise ValueError(f"axis must be one of {sorted(_AXES)}, got {axis!r}")
    L = sp.eye(4)
    i = _AXES[axis]
    L[0, 0] = L[i, i] = sp.cosh(theta)
    L[0, i] = L[i, 0] = -sp.sinh(theta)
    return L


def rotation2(phi):
    """Griffiths Eq. 1.29: 2-D rotation block, for comparison with the boost."""
    return sp.Matrix([[sp.cos(phi), sp.sin(phi)], [-sp.sin(phi), sp.cos(phi)]])


def boost2(theta):
    """(ct, x) block of the rapidity boost: cosh/sinh where rotation has cos/sin."""
    return sp.Matrix([[sp.cosh(theta), -sp.sinh(theta)],
                      [-sp.sinh(theta), sp.cosh(theta)]])


def is_lorentz(L):
    """True iff L preserves the Minkowski metric: L^T eta L = eta."""
    L = sp.Matrix(L)
    if L.shape != (4, 4):
        raise ValueError(f"L must be 4x4, got {L.shape}")
    return sp.simplify(L.T * ETA * L - ETA) == sp.zeros(4, 4)


def minkowski(a, b):
    """Minkowski scalar product a.b = -a0 b0 + a1 b1 + a2 b2 + a3 b3."""
    a, b = _as_event(a, "a"), _as_event(b, "b")
    return sp.expand((a.T * ETA * b)[0, 0])


def rapidity(beta):
    """theta = atanh(v/c), Eq. 12.34."""
    return sp.atanh(_check_beta(beta))


def add_velocities(b1, b2):
    """Einstein velocity addition in units of c: (b1+b2)/(1+b1*b2)."""
    b1, b2 = _check_beta(b1), _check_beta(b2)
    return sp.simplify((b1 + b2) / (1 + b1 * b2))


def interval(A, B):
    """Invariant interval I = -(c dt)^2 + |dr|^2 between events (ct, x, y, z)."""
    d = _as_event(B, "B") - _as_event(A, "A")
    return minkowski(d, d)


def classify_interval(I):
    """'timelike' (I<0), 'spacelike' (I>0) or 'lightlike' (I=0), Griffiths signs."""
    I = sp.sympify(I)
    if not I.is_number:
        raise ValueError("interval must be numeric to classify")
    if I < 0:
        return "timelike"
    if I > 0:
        return "spacelike"
    return "lightlike"


def connecting_velocity(A, B):
    """beta vector of the frame in which timelike-separated events A, B co-locate.

    v = (r_B - r_A) / (t_B - t_A); raises unless the separation is timelike
    (otherwise no such frame exists -- you'd need |v| >= c).
    """
    A, B = _as_event(A, "A"), _as_event(B, "B")
    I = interval(A, B)
    if I.is_number and I >= 0:
        raise ValueError(
            f"events are not timelike-separated (I={I}); no co-locating frame"
        )
    d = B - A
    return sp.Matrix(d[1:]) / d[0]


def speed_from_kinetic_ratio(n):
    """Problem 12.30: if E_kin = n * (rest energy), then beta = sqrt(n(n+2))/(n+1)."""
    n = sp.sympify(n)
    if n.is_number and n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    gamma = n + 1                       # (gamma - 1) m c^2 = n m c^2
    return sp.simplify(sp.sqrt(1 - 1 / gamma**2))
