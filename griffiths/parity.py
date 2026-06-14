"""Parity / handedness machinery for Griffiths Problem 1.10.

The "offense" of inversion: it turns a right-handed coordinate frame into a
left-handed one (det = -1), and that single sign is why cross products are
pseudovectors and triple products are pseudoscalars.
"""

import sympy as sp

from .vectors import _as_vec3

# How each object type responds to inversion (x,y,z) -> (-x,-y,-z).
INVERSION_SIGNS = {
    "scalar": 1,        # e.g. mass, charge, A.B
    "pseudoscalar": -1, # e.g. A.(BxC), magnetic helicity
    "vector": -1,       # e.g. r, v, p, E  ("polar" vector)
    "pseudovector": 1,  # e.g. L = rxp, B, torque, omega  ("axial" vector)
}


def parity_matrix():
    """The inversion operator P = -I (improper rotation, det = -1)."""
    return -sp.eye(3)


def handedness(e1, e2, e3):
    """'right' or 'left' according to sign of e1 . (e2 x e3).

    Raises if the three vectors are coplanar (no handedness to speak of).
    """
    e1 = _as_vec3(e1, "e1")
    e2 = _as_vec3(e2, "e2")
    e3 = _as_vec3(e3, "e3")
    triple = sp.simplify(e1.dot(e2.cross(e3)))
    if triple == 0:
        raise ValueError("basis vectors are coplanar; handedness undefined")
    return "right" if triple > 0 else "left"


def invert(obj, kind):
    """Apply spatial inversion to obj, given its transformation type.

    kind: 'scalar' | 'pseudoscalar' | 'vector' | 'pseudovector'.
    Vectors get all components negated; pseudovectors are left alone;
    scalars are unchanged; pseudoscalars flip sign.
    """
    if kind not in INVERSION_SIGNS:
        raise ValueError(
            f"kind must be one of {sorted(INVERSION_SIGNS)}, got {kind!r}"
        )
    sign = INVERSION_SIGNS[kind]
    if kind in ("scalar", "pseudoscalar"):
        return sign * sp.sympify(obj)
    return sign * _as_vec3(obj, "obj")
