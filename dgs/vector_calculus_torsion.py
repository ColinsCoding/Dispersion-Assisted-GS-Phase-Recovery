"""Vector calculus and torsion -- Paul's Online Math Notes (Lamar) level,
applied to phase retrieval and quantum/photonic engineering.

The central link between all three topics:

  Phase retrieval (GS) recovers phi(r) such that |E(r)|^2 = I(r).
  phi(r) IS a scalar field. Its gradient grad(phi) is the LOCAL spatial
  frequency (wavevector) of the field. Integrating grad(phi) along a CLOSED
  path should give zero -- UNLESS the path encloses a phase singularity
  (optical vortex), in which case the line integral = 2*pi*N (winding number N).

  That closed-path integral of a gradient is exactly what Stokes' theorem
  controls: line_integral(grad phi) = surface_integral(curl(grad phi)).
  curl(grad phi) = 0 everywhere EXCEPT at vortex cores -- so the
  Stokes theorem detects vortices by their "torsion" of the phase field.

  In quantum computing: the Berry phase gamma = -i * contour_integral
  <psi|d/dR|psi> dR is the same kind of line integral over parameter space.
  Torsion of the parameter-space path determines whether the Berry phase is
  topologically protected (non-zero even under smooth deformations).

Frenet-Serret formulas:
  dT/ds = kappa * N
  dN/ds = -kappa*T + tau*B
  dB/ds = -tau * N
where kappa = curvature (how fast T turns), tau = torsion (how fast the
osculating plane rotates = how much the curve "twists" out of 2D).
"""
import numpy as np
import sympy as sp


# -- Symbolic vector calculus (Cartesian 3D) ----------------------------------

x, y, z = sp.symbols('x y z', real=True)
_coords = (x, y, z)


def gradient(f):
    """grad(f) = [df/dx, df/dy, df/dz] as a SymPy Matrix."""
    return sp.Matrix([sp.diff(f, v) for v in _coords])


def divergence(F):
    """div(F) = dFx/dx + dFy/dy + dFz/dz. F: list/Matrix of 3 expressions."""
    F = list(F)
    return sp.Add(*[sp.diff(F[i], _coords[i]) for i in range(3)])


def curl(F):
    """curl(F) = [dFz/dy - dFy/dz, dFx/dz - dFz/dx, dFy/dx - dFx/dy]."""
    Fx, Fy, Fz = list(F)
    return sp.Matrix([
        sp.diff(Fz, y) - sp.diff(Fy, z),
        sp.diff(Fx, z) - sp.diff(Fz, x),
        sp.diff(Fy, x) - sp.diff(Fx, y),
    ])


def laplacian(f):
    """Laplacian: div(grad(f)) = d^2f/dx^2 + d^2f/dy^2 + d^2f/dz^2."""
    return sp.Add(*[sp.diff(f, v, 2) for v in _coords])


def curl_of_gradient_is_zero(f):
    """Verify curl(grad(f)) = 0 symbolically -- a fundamental identity."""
    g = gradient(f)
    c = sp.simplify(curl(g))
    is_zero = c == sp.zeros(3, 1)
    return {"curl_grad": c, "is_zero": is_zero}


def div_of_curl_is_zero(F):
    """Verify div(curl(F)) = 0 symbolically -- another fundamental identity."""
    c = curl(F)
    d = sp.simplify(divergence(c))
    return {"div_curl": d, "is_zero": d == 0}


# -- Stokes' theorem and divergence theorem (symbolic statement) ---------------

def stokes_theorem_statement():
    """Stokes: closed line integral = surface integral of curl.
    Returns as sp.Eq objects showing both sides."""
    F_sym = sp.Matrix([sp.Function('Fx')(x,y,z),
                       sp.Function('Fy')(x,y,z),
                       sp.Function('Fz')(x,y,z)])
    lhs = sp.Symbol('oint_C_F_dot_dr')
    rhs = sp.Symbol('iint_S_curlF_dot_dS')
    return sp.Eq(lhs, rhs)


def divergence_theorem_statement():
    """Gauss: closed surface integral = volume integral of divergence."""
    lhs = sp.Symbol('oiint_S_F_dot_dS')
    rhs = sp.Symbol('iiint_V_divF_dV')
    return sp.Eq(lhs, rhs)


# -- Phase field gradient and vortex detection --------------------------------

def phase_gradient_2d(phi_func, x_arr, y_arr):
    """Numerically compute grad(phi) on a 2D grid using central differences.
    phi_func(x, y) should return the unwrapped phase at each grid point.
    Returns gx, gy arrays (local wavevectors)."""
    phi = phi_func(x_arr[:, None], y_arr[None, :])
    gx = np.gradient(phi, x_arr, axis=0)
    gy = np.gradient(phi, y_arr, axis=1)
    return gx, gy


def phase_circulation(phi_func, cx, cy, radius, n_pts=360):
    """Line integral of grad(phi) around a circle of given radius centred at
    (cx, cy). A non-zero result (= 2*pi*N) indicates a phase vortex of
    winding number N inside the contour -- the same topology that makes
    phase retrieval hard near vortex cores."""
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    xc = cx + radius * np.cos(theta)
    yc = cy + radius * np.sin(theta)
    # numerical derivative of phi along the contour
    phi_vals = phi_func(xc, yc)
    dphi = np.unwrap(phi_vals)
    total_change = dphi[-1] - dphi[0] + (phi_vals[-1] - phi_vals[0])
    return {"circulation": float(np.sum(np.diff(np.unwrap(phi_vals)))),
            "winding_number": round(float(np.sum(np.diff(np.unwrap(phi_vals)))) / (2 * np.pi)),
            "n_pts": n_pts}


# -- Frenet-Serret: curvature and torsion of a 3D curve ----------------------

def frenet_serret(r_func, t_arr):
    """Compute the Frenet-Serret frame (T, N, B), curvature kappa, and
    torsion tau numerically for a parametric curve r(t) in R^3.
    r_func(t) -> (N,3) array of points."""
    r = np.asarray(r_func(t_arr))           # (N, 3)
    dt = t_arr[1] - t_arr[0]
    dr = np.gradient(r, dt, axis=0)         # dr/dt
    ddr = np.gradient(dr, dt, axis=0)       # d^2r/dt^2
    dddr = np.gradient(ddr, dt, axis=0)     # d^3r/dt^3

    speed = np.linalg.norm(dr, axis=1, keepdims=True)  # |dr/dt|
    T = dr / np.where(speed == 0, 1, speed)            # unit tangent

    cross_dr_ddr = np.cross(dr, ddr)
    kappa_num = np.linalg.norm(cross_dr_ddr, axis=1)   # |dr x d^2r|
    kappa_den = speed[:, 0] ** 3
    kappa = np.where(kappa_den == 0, 0, kappa_num / kappa_den)  # curvature

    # torsion: tau = (dr x d^2r) . d^3r / |dr x d^2r|^2
    tau_num = np.einsum('ij,ij->i', cross_dr_ddr, dddr)
    tau_den = kappa_num ** 2
    tau = np.where(tau_den < 1e-12, 0, tau_num / tau_den)  # torsion

    return {"T": T, "kappa": kappa, "tau": tau, "r": r, "t": t_arr}


def frenet_serret_sympy():
    """Symbolic Frenet-Serret formulas as sp.Eq objects."""
    s = sp.Symbol('s', positive=True)
    T, N_sym, B = sp.symbols('T N B')
    kappa, tau = sp.symbols('kappa tau', positive=True)
    dT_ds = sp.Function('dT_ds')
    dN_ds = sp.Function('dN_ds')
    dB_ds = sp.Function('dB_ds')
    return {
        "dT_ds = kappa*N": sp.Eq(sp.Symbol('dT_ds'), kappa * N_sym),
        "dN_ds = -kappa*T + tau*B": sp.Eq(sp.Symbol('dN_ds'), -kappa * T + tau * B),
        "dB_ds = -tau*N": sp.Eq(sp.Symbol('dB_ds'), -tau * N_sym),
    }


# -- Berry phase (quantum computing / photonics) connection -------------------

def berry_phase_demo(theta_arr, phi_arr):
    """Compute the Berry phase for a two-level quantum system (qubit) along a
    path on the Bloch sphere parametrized by (theta, phi).
    Berry phase = -i * line_integral <psi|d/dR|psi> dR = solid angle / 2.
    For a CLOSED path: gamma = (1/2) * solid_angle_enclosed."""
    dtheta = np.diff(theta_arr)
    dphi   = np.diff(phi_arr)
    # Berry connection A = -cos(theta)/2 * dphi (spin-1/2 on Bloch sphere)
    theta_mid = 0.5 * (theta_arr[:-1] + theta_arr[1:])
    A = -0.5 * np.cos(theta_mid)
    berry_phase = float(np.sum(A * dphi))
    solid_angle = 2 * berry_phase  # gamma = omega/2 for spin-1/2
    return {"berry_phase_rad": berry_phase, "solid_angle": solid_angle,
            "note": "Berry phase = half the solid angle enclosed on Bloch sphere"}


# -- SymPy 5 ------------------------------------------------------------------

def vector_calculus_sympy_5():
    """Five key symbolic results."""
    r = sp.Symbol('r', positive=True)
    f = sp.Function('f')(x, y, z)
    phi = sp.Function('phi')(x, y, z)
    return {
        "Gradient":
            sp.Eq(sp.Symbol('grad_f'), gradient(x**2 * y + sp.sin(z)), evaluate=False),
        "Curl_of_gradient_zero":
            sp.Eq(sp.Symbol("curl_grad_phi"), sp.zeros(3, 1), evaluate=False),
        "Stokes_theorem": stokes_theorem_statement(),
        "Frenet_curvature":
            sp.Eq(sp.Symbol('kappa'), sp.Symbol('|dT_ds|')),
        "Berry_phase":
            sp.Eq(sp.Symbol('gamma_Berry'),
                  -sp.I * sp.Symbol('contour_integral_A_dR')),
    }


if __name__ == "__main__":
    print("=== Gradient of x^2*y + sin(z) ===")
    f_ex = x**2 * y + sp.sin(z)
    print(f"  grad = {gradient(f_ex).T}")

    print("\n=== Divergence of (x^2, y^2, z^2) ===")
    F_ex = [x**2, y**2, z**2]
    print(f"  div = {divergence(F_ex)}")

    print("\n=== Curl(grad(f)) = 0 identity ===")
    res = curl_of_gradient_is_zero(x**2 * y + sp.sin(z))
    print(f"  is zero: {res['is_zero']}")

    print("\n=== div(curl(F)) = 0 identity ===")
    F2 = [x*y, y*z, z*x]
    res2 = div_of_curl_is_zero(F2)
    print(f"  is zero: {res2['is_zero']}")

    print("\n=== Frenet-Serret: helix r(t)=(cos t, sin t, t) ===")
    t_arr = np.linspace(0, 4*np.pi, 4000)
    fs = frenet_serret(lambda t: np.column_stack([np.cos(t), np.sin(t), t]), t_arr)
    kappa_mid = fs["kappa"][len(t_arr)//2]
    tau_mid   = fs["tau"][len(t_arr)//2]
    print(f"  kappa (expect 0.5) = {kappa_mid:.4f}")
    print(f"  tau   (expect 0.5) = {tau_mid:.4f}")

    print("\n=== Phase vortex detection: phi=atan2(y,x) ===")
    circ = phase_circulation(lambda x, y: np.arctan2(y, x), cx=0, cy=0, radius=1.0)
    print(f"  winding number = {circ['winding_number']} (expect 1)")

    print("\n=== Berry phase: closed loop on Bloch sphere ===")
    theta_path = np.full(361, np.pi/3)
    phi_path   = np.linspace(0, 2*np.pi, 361)
    bp = berry_phase_demo(theta_path, phi_path)
    expected_bp = -np.cos(np.pi/3)/2 * 2*np.pi
    print(f"  Berry phase = {bp['berry_phase_rad']:.4f} rad (expect {expected_bp:.4f} = -cos(pi/3)/2 * 2*pi)")

    print("\n=== SymPy 5 ===")
    for k, eq in vector_calculus_sympy_5().items():
        print(f"  {k}: {eq}")
