"""Griffiths Ch.8 -- Conservation Laws: momentum, not just energy.

vector_identities.py's Poynting theorem already covers ENERGY conservation
(div S + dE/dt = -J.E). This module covers the other conserved quantity
Ch.8 introduces: MOMENTUM. The field itself carries momentum, with density
g = eps0*(E x B), and the force per unit area the field exerts is captured
by the Maxwell stress tensor T -- the same "flux out through a surface"
idea as Poynting's theorem, one rank higher (a tensor instead of a vector,
because momentum is itself a vector, so its flux needs two indices).
"""
import sympy as sp

eps0, mu0, c = sp.symbols('epsilon_0 mu_0 c', positive=True)


def maxwell_stress_tensor(E, B):
    """T_ij = eps0*(E_i*E_j - (1/2)*delta_ij*E^2) + (1/mu0)*(B_i*B_j - (1/2)*delta_ij*B^2).

    E, B: length-3 sympy Matrix/list of field components (numbers or symbols).
    Returns the 3x3 sympy Matrix T. T_ij is the i-th component of the force
    per unit area transmitted across a surface whose normal is the j-th axis
    -- literally a generalization of mechanical stress/pressure to the EM
    field."""
    E = sp.Matrix(E)
    B = sp.Matrix(B)
    if E.shape != (3, 1) or B.shape != (3, 1):
        raise ValueError("E and B must each have 3 components")
    E2 = (E.T * E)[0]
    B2 = (B.T * B)[0]
    T = sp.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            delta_ij = 1 if i == j else 0
            T[i, j] = eps0 * (E[i] * E[j] - sp.Rational(1, 2) * delta_ij * E2) \
                + (1 / mu0) * (B[i] * B[j] - sp.Rational(1, 2) * delta_ij * B2)
    return T


def verify_stress_tensor_symmetric(E, B):
    """T is symmetric (T_ij = T_ji) by construction -- verified directly,
    not assumed. Symmetry is what makes T represent a physical stress
    (no net torque from the tensor's antisymmetric part)."""
    T = maxwell_stress_tensor(E, B)
    return bool(sp.simplify(T - T.T) == sp.zeros(3, 3))


def momentum_density(E, B):
    """g = eps0*mu0*S = eps0*(E x B), where S=(1/mu0)(E x B) is the Poynting
    vector (vector_identities.py). The field carries momentum wherever it
    carries energy flux -- same physical content as S, repackaged as a
    momentum density instead of a power density."""
    E = sp.Matrix(E)
    B = sp.Matrix(B)
    return eps0 * E.cross(B)


def radiation_pressure(intensity, absorbed=True):
    """P = I/c if the surface absorbs the light, 2I/c if it perfectly
    reflects (each photon's momentum is reversed instead of just stopped,
    doubling the momentum transfer). intensity I in W/m^2, c in m/s;
    returns pressure in Pa (N/m^2)."""
    if intensity < 0:
        raise ValueError("intensity must be non-negative")
    c_num = 299792458.0
    return (2.0 if not absorbed else 1.0) * intensity / c_num


if __name__ == "__main__":
    Ex, Ey, Ez = sp.symbols('E_x E_y E_z', real=True)
    Bx, By, Bz = sp.symbols('B_x B_y B_z', real=True)
    E = [Ex, Ey, Ez]
    B = [Bx, By, Bz]

    print("=== Maxwell stress tensor (symbolic) ===")
    T = maxwell_stress_tensor(E, B)
    sp.pprint(T)

    print("\n=== symmetry check ===")
    print("T symmetric:", verify_stress_tensor_symmetric(E, B))

    print("\n=== momentum density g = eps0*(E x B) ===")
    g = momentum_density(E, B)
    sp.pprint(g.T)

    print("\n=== radiation pressure, sunlight at Earth (~1361 W/m^2) ===")
    P_abs = radiation_pressure(1361.0, absorbed=True)
    P_refl = radiation_pressure(1361.0, absorbed=False)
    print(f"absorbing surface: {P_abs:.4e} Pa")
    print(f"reflecting surface: {P_refl:.4e} Pa  (exactly 2x, as expected)")
    print(f"ratio: {P_refl/P_abs:.4f}")
