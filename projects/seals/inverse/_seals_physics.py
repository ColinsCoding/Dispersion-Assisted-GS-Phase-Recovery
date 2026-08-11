"""
_seals_physics.py -- side-effect-free extraction of the pure numerical functions
from ../seals_stable.py, for reuse by the inverse/ package.

Why this file exists, instead of `import seals_stable`:
seals_stable.py is a research-notebook-style script (VSCode/Jupyter "# %%" cells)
with executable top-level code -- it loads default parameters, prints summaries,
generates several matplotlib figures, and also runs an unrelated poker-hand Monte
Carlo simulation and a Phong-BRDF card render (its own Sections 9-10). Importing
it directly would trigger all of that as an import side effect, which is unsafe
and has nothing to do with the SEALS physics this package needs.

Every function below is copied VERBATIM from seals_stable.py -- no numerical
change of any kind. Only the pure physics functions are here (seals_displacement,
seals_theta_map, seals, form_factor_P, rayleigh_debye, the Mie pipeline, and the
P_DEFAULT parameter dict); the poker/BRDF sections were not needed and are not
duplicated. seals_stable.py itself is untouched by this change.
"""
import numpy as np
from scipy.special import spherical_jn, spherical_yn

# Default SEALS parameters (verbatim from seals_stable.py P_DEFAULT)
P_DEFAULT = dict(
    dia    = 9940e-9,      # particle diameter [m]
    npar   = 1.39,         # particle n
    nmed   = 1.00,         # medium n
    r      = 0.10,         # detector distance [m]
    d      = 9.0909e-7,    # grating groove spacing [m]
    D      = 0.065,        # inter-grating distance [m]
    a      = 0.9023,       # grating tilt [rad]
    dcorr  = -4.2448e-4,   # lens correction [m]
    P      = 0.0058,       # lens diameter [m]
    NA     = 0.70,         # numerical aperture
    mangle = 20.0,         # measurement angle offset [deg]
    lam1   = 1580e-9,      # min wavelength [m]
    lam2   = 1600e-9,      # max wavelength [m]
    N_lam  = 500,          # wavelength samples
)


def seals_displacement(lam, d, D, a):
    """
    Beam vertical displacement y(lambda).
    y = (D/6) . tan(Delta) / (1 + tan(Delta).tan(alpha))
    where Delta = alpha - arcsin(lambda/d - sin alpha)   [grating diffraction angle]

    Returns y (m), valid_mask (bool array).
    Clips the arcsin argument to avoid domain errors.
    """
    arg = lam / d - np.sin(a)
    valid = np.abs(arg) < 1.0
    arg_safe = np.clip(arg, -0.9999, 0.9999)
    theta_d = np.arcsin(arg_safe)          # diffracted angle
    Delta = a - theta_d                    # angular deviation
    tan_D = np.tan(Delta)
    tan_a = np.tan(a)
    denom = 1.0 + tan_D * tan_a
    y = (D / 6.0) * tan_D / denom
    y = y - y[-1]                          # reference to last point
    return y, valid


def seals_theta_map(y, P, NA, dcorr):
    """
    Map beam displacement -> scattering angle (degrees).
    theta = arctan[2/P . (y - y_c + d_corr) . tan(arcsin(NA))]
    """
    ycenter = (y[0] - y[-1]) / 2.0
    theta = np.degrees(np.arctan(
        (2.0 / P) * (y - ycenter + dcorr) * np.tan(np.arcsin(NA))
    ))
    return theta


def seals(d, D, a, dcorr, P, NA, lamvec):
    """Full SEALS pipeline: lamvec -> (y, theta_scat_deg)."""
    y, valid = seals_displacement(lamvec, d, D, a)
    theta = seals_theta_map(y, P, NA, dcorr)
    return y, theta, valid


def form_factor_P(u):
    """
    Rayleigh-Debye form factor P(u) = 3(sin u - u cos u)/u^3
    Numerically stable: Taylor expansion for |u| < 0.1
      P(u) = 1 - u^2/10 + u^4/280 - u^6/15120 + ...
    """
    out = np.empty_like(u, dtype=float)
    small = np.abs(u) < 0.1
    u_s = u[small]
    u2 = u_s**2
    out[small] = 1.0 - u2/10.0 + u2**2/280.0 - u2**3/15120.0
    u_l = u[~small]
    out[~small] = 3.0*(np.sin(u_l) - u_l*np.cos(u_l)) / u_l**3
    return out


def rayleigh_debye(dia, lam, n_bg, n_sp, theta_rad, R):
    """
    Rayleigh-Debye-Gans scattering intensity I(theta).
    theta_rad: array of scattering angles [rad]
    Returns I [W/m^2] assuming unit incident intensity.

    Returns intensity ONLY -- this model has no complex-field / phase
    representation in the original MATLAB (rayleighdebye.m) or in this port.
    """
    a      = dia / 2.0
    k      = 2*np.pi*n_bg / lam
    n_rel  = n_sp / n_bg
    u      = 2*k*a*np.sin(theta_rad/2.0)
    f_th   = 1.0 + np.cos(theta_rad)**2          # Rayleigh dipole factor
    P_th   = form_factor_P(u)                     # RDG structure factor
    prefac = ((n_rel**2-1)/(n_rel**2+2))**2
    I = np.abs(P_th*f_th)**2 * prefac / (2*R**2) * (2*np.pi/lam)**4 * a**6
    return I


def _mie_bessel(n_arr, x):
    """
    Spherical Bessel j_n(x), y_n(x) and their 'derivative' combinations
    needed for Mie coefficients, using scipy for stability.
    """
    jx = np.array([spherical_jn(int(n), x) for n in n_arr])
    yx = np.array([spherical_yn(int(n), x) for n in n_arr])
    j1x = np.empty_like(jx)
    j1x[0] = np.sin(x)/x if x > 1e-30 else 1.0
    j1x[1:] = jx[:-1]
    y1x = np.empty_like(yx)
    y1x[0] = -np.cos(x)/x if x > 1e-30 else 0.0
    y1x[1:] = yx[:-1]
    return jx, yx, j1x, y1x


def mie_coefficients(npar, nmed, dia, lam):
    """
    Compute Mie coefficients a_n, b_n.
    Returns: n_arr, an, bn, x (size parameter), nmax
    """
    x    = np.pi * dia / (lam / nmed)          # size parameter
    m    = npar / nmed                           # relative index
    nmax = int(np.round(2 + x + 4*x**(1/3)))   # Wiscombe truncation
    nmax = max(nmax, 5)

    n_arr = np.arange(1, nmax+1, dtype=float)
    z     = m * x

    jx, yx, j1x, y1x = _mie_bessel(n_arr, x)
    jz, _,  j1z, _   = _mie_bessel(n_arr, z)

    hx  = jx + 1j*yx
    h1x = j1x + 1j*y1x

    ax  = x*j1x  - n_arr*jx
    az  = z*j1z  - n_arr*jz
    ahx = x*h1x  - n_arr*hx
    m2  = m*m

    an = (m2*jz*ax  - jx*az) / (m2*jz*ahx - hx*az)
    bn = (   jz*ax  - jx*az) / (   jz*ahx - hx*az)

    return n_arr, an, bn, x, nmax


def _angular_functions(nmax, u_arr):
    """
    Compute pi_n(cos theta), tau_n(cos theta) for all n=1..nmax, all angles at once.
    u_arr = cos(theta), shape (N_ang,)
    Returns pi_mat, tau_mat, shape (nmax, N_ang).
    """
    N = len(u_arr)
    pi_m  = np.zeros((nmax, N))
    tau_m = np.zeros((nmax, N))
    pi_m[0] = 1.0
    tau_m[0] = u_arr
    if nmax >= 2:
        pi_m[1] = 3.0*u_arr
        tau_m[1] = 6.0*u_arr**2 - 3.0  # tau_2 = 3cos(2theta) = 3(2u^2-1)
    for j in range(2, nmax):           # j is 0-indexed -> MATLAB order n = j+1
        n = j + 1
        pi_m[j] = ((2*n-1)/(n-1))*u_arr*pi_m[j-1] - (n/(n-1))*pi_m[j-2]
        tau_m[j] = n*u_arr*pi_m[j] - (n+1)*pi_m[j-1]
    return pi_m, tau_m


def mie(npar, nmed, dia, lam, angles_rad, r):
    """
    Full Mie scattering at given angles.
    Returns: sigma_s, I_p, I_s, an, bn, T_p, T_s

    E_theta, E_phi (complex far fields) are computed internally but not
    returned by this function, matching seals_stable.py's signature exactly
    (established name/shape preserved -- see inverse/measurement.py for how
    this package reconstructs the complex fields from I_p/I_s/T_p/T_s without
    duplicating the Bessel-function internals).
    """
    n_arr, an, bn, x, nmax = mie_coefficients(npar, nmed, dia, lam)
    k   = 2*np.pi*nmed / lam
    phi = np.pi  # scattering plane (phi=pi same as MATLAB)

    u = np.cos(angles_rad)                # (N_ang,)
    pi_m, tau_m = _angular_functions(nmax, u)   # (nmax, N_ang)

    wt = (2*n_arr+1) / (n_arr*(n_arr+1))  # (nmax,)
    pin = wt[:, None] * pi_m              # (nmax, N_ang)
    tin = wt[:, None] * tau_m             # (nmax, N_ang)

    S1 = np.sum(an[:, None]*pin + bn[:, None]*tin, axis=0)
    S2 = np.sum(an[:, None]*tin + bn[:, None]*pin, axis=0)

    phase = np.exp(1j*k*r)
    E_theta = phase / (-1j*k*r) * np.cos(phi) * S2   # complex (N_ang,)
    E_phi   = phase / ( 1j*k*r) * np.cos(phi) * S1   # complex (N_ang,)

    T_p = np.angle(E_theta)
    T_s = np.angle(E_phi)
    I_p = np.abs(E_theta)**2
    I_s = np.abs(E_phi)**2

    x2    = x*x
    en    = (2*n_arr+1)*(np.abs(an)**2 + np.abs(bn)**2)
    qsca  = 2*np.sum(en)/x2
    A     = np.pi*(dia/2)**2
    sigma_s = qsca * A

    return sigma_s, I_p, I_s, an, bn, T_p, T_s
