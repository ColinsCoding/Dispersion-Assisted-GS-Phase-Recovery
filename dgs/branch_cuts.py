"""Logarithm branch cuts and phase unwrapping for TS-DFT spectroscopy.

THE BRANCH CUT PROBLEM IN GS PHASE RECOVERY:
  The recovered field is E(omega) = |E(omega)| * exp(i*phi(omega)).
  Extracting phase: phi(omega) = Im[log E(omega)]
  log is MULTIVALUED: log(z) = ln|z| + i*(Arg(z) + 2*pi*k)  for any integer k.

  Principal branch: Arg(z) in (-pi, pi].
  Branch cut = the negative real axis, where Arg is discontinuous.

  After GS converges, np.angle() returns phi in (-pi, pi].
  Real spectral phase of a dispersive pulse is CONTINUOUS and can span
  many multiples of pi -> need PHASE UNWRAPPING to cross branch cuts.

  THE TS-DFT CONNECTION:
    H(omega) = exp(i*beta2*L*omega^2/2)
    phi(omega) = beta2*L*omega^2/2   (parabolic, crosses pi many times)
    Wrapped:   phi_w = mod(phi + pi, 2*pi) - pi  <- what np.angle gives
    Unwrapped: phi_u = cumulative_sum of unwrapped differences
    Group delay: tau(omega) = d(phi_u)/d(omega) = beta2*L*omega  (linear)
    GVD:        beta2 = d^2(phi_u)/d(omega^2)

  BRANCH CUTS IN THE COMPLEX PLANE:
    log(z) has a branch cut along (-inf, 0] (negative real axis).
    sqrt(z) has a branch cut along (-inf, 0].
    z^alpha (non-integer alpha): branch cut from 0 to -inf.
    arctan(z): branch cuts on imaginary axis outside [-i, i].

    For optical fields: E(omega) = A(omega)*exp(i*phi(omega))
    log E = log A + i*phi  <- real and imaginary parts cleanly separated
    Branch cut matters only when A(omega) passes through zero (spectral nulls).
    At a spectral null: phi jumps by pi (pi phase slip at zero crossing).
"""

from __future__ import annotations
import numpy as np
import sympy as sp
from typing import Dict, List, Optional, Tuple


# ════════════════════════════════════════════════════════════════════════════
# §1  COMPLEX LOGARITHM AND BRANCH CUTS
# ════════════════════════════════════════════════════════════════════════════

def complex_log(z: np.ndarray, branch: int = 0) -> np.ndarray:
    """Multi-valued complex logarithm: log(z) = ln|z| + i*(Arg(z) + 2*pi*k).

    branch=0  : principal value, Arg in (-pi, pi]
    branch=k  : kth branch, shifted by 2*pi*k
    """
    return np.log(np.abs(z)) + 1j * (np.angle(z) + 2 * np.pi * branch)


def principal_log(z: np.ndarray) -> np.ndarray:
    """Principal branch Log(z): defined on C minus (-inf, 0].

    Discontinuous across negative real axis (the branch cut).
    Im[Log(z)] in (-pi, pi].
    """
    return np.log(z.astype(complex))   # numpy's log uses principal branch


def branch_cut_demo(n_pts: int = 500) -> Dict:
    """Visualize branch cut of log(z) on a grid around the negative real axis.

    Shows the discontinuity in Im[log(z)] = Arg(z) as z crosses (-inf, 0].
    Just above the cut: Arg -> +pi.
    Just below the cut: Arg -> -pi.
    Jump = 2*pi (one period).
    """
    # Approach the negative real axis from above and below
    x = np.linspace(-3, -0.1, n_pts)
    eps = 1e-8
    z_above = x + 1j * eps
    z_below = x - 1j * eps

    arg_above = np.angle(z_above)   # -> +pi
    arg_below = np.angle(z_below)   # -> -pi
    jump = arg_above - arg_below    # should be ~2*pi everywhere

    # Riemann sheet structure: log(z) has infinitely many branches
    # Riemann surface: helicoid with sheets joined at branch cut
    theta = np.linspace(-np.pi, np.pi, 200)
    r = 1.0
    z_circle = r * np.exp(1j * theta)
    log_circle = np.log(np.abs(z_circle)) + 1j * np.angle(z_circle)

    return {
        "x_neg_real":   x,
        "arg_above":    arg_above,   # -> +pi
        "arg_below":    arg_below,   # -> -pi
        "jump":         jump,
        "mean_jump":    float(np.mean(jump)),    # ~2*pi
        "z_circle":     z_circle,
        "log_circle":   log_circle,
        "branch_cut":   "negative real axis (-inf, 0]",
        "discontinuity": "Im[log] jumps by 2*pi when crossing cut",
    }


def sqrt_branch_cut() -> Dict:
    """Branch cut of sqrt(z): also along (-inf, 0].

    sqrt(z) = |z|^(1/2) * exp(i*Arg(z)/2)
    Just above cut: exp(i*pi/2) = i
    Just below cut: exp(-i*pi/2) = -i
    Jump in sqrt(z): 2i * |x|^(1/2)  (purely imaginary)

    Connection: sqrt of GS update involves sqrt of intensity -> must use
    consistent branch when amplitude passes through zero.
    """
    x = np.linspace(-4, -0.1, 200)
    eps = 1e-8
    sqrt_above = np.sqrt(x + 1j*eps + 0j)   # -> i*|x|^(1/2)
    sqrt_below = np.sqrt(x - 1j*eps + 0j)   # -> -i*|x|^(1/2)
    return {
        "x":           x,
        "sqrt_above":  sqrt_above,
        "sqrt_below":  sqrt_below,
        "jump":        sqrt_above - sqrt_below,   # ~2i*sqrt(|x|)
        "branch_cut":  "negative real axis",
        "principal":   "sqrt(z) = |z|^(1/2) * exp(i*Arg(z)/2), Arg in (-pi,pi]",
    }


# ════════════════════════════════════════════════════════════════════════════
# §2  PHASE UNWRAPPING FOR TS-DFT
# ════════════════════════════════════════════════════════════════════════════

def unwrap_phase(phi_wrapped: np.ndarray,
                  discont: float = np.pi) -> np.ndarray:
    """Phase unwrapping: remove 2*pi jumps from wrapped phase array.

    Algorithm: scan differences; when |diff| > discont, add ±2*pi correction.
    This is equivalent to choosing the correct Riemann sheet at each step.

    discont : threshold for detecting a branch-cut crossing (default pi)
    """
    return np.unwrap(phi_wrapped, discont=discont)


def phase_nyquist_check(beta2: float, L_m: float,
                         omega_max: float, n_pts: int) -> Dict:
    """Check if sampling is dense enough for phase unwrapping.

    Phase Nyquist condition: |Δφ| < π between adjacent samples.
    For GVD: Δφ_max = |beta2| * L * omega_max * (2*omega_max/n_pts)
    Minimum n_pts: n_pts_min = 2 * omega_max^2 * |beta2| * L / π

    If violated: unwrapping fails silently -> wrong GVD. Use the gradient
    of the *group delay* (measured directly in TS-DFT) instead.
    """
    domega   = 2 * omega_max / n_pts
    dphi_max = abs(beta2) * L_m * omega_max * domega
    n_min    = int(np.ceil(2 * omega_max**2 * abs(beta2) * L_m / np.pi))
    return {
        "dphi_max":       dphi_max,
        "nyquist_ok":     bool(dphi_max < np.pi),
        "n_pts":          n_pts,
        "n_pts_min":      n_min,
        "ratio":          dphi_max / np.pi,
        "advice": (
            "OK: use np.unwrap" if dphi_max < np.pi
            else f"VIOLATED: need {n_min} pts or measure group delay directly"
        ),
    }


def gvd_phase_unwrapped(omega: np.ndarray,
                          beta2: float,
                          L_m: float) -> Dict:
    """Compute and unwrap the GVD phase phi(omega) = beta2*L*omega^2/2.

    The parabolic phase crosses the branch cut at omega where
    phi(omega) = (2k+1)*pi for integer k.
    Unwrapping is trivial here (analytic), but useful for verifying the
    unwrapping algorithm against a known ground truth.
    """
    phi_true     = 0.5 * beta2 * L_m * omega**2
    phi_wrapped  = np.angle(np.exp(1j * phi_true))   # same as phi mod 2pi in (-pi,pi]

    # Phase Nyquist check BEFORE unwrapping
    omega_max = float(np.max(np.abs(omega)))
    n_pts = len(omega)
    nyquist = phase_nyquist_check(beta2, L_m, omega_max, n_pts)

    if nyquist["nyquist_ok"]:
        phi_unwrapped = np.unwrap(phi_wrapped)
    else:
        # Nyquist violated: unwrapping will fail.
        # Use analytical phase directly (or group delay from TS-DFT measurement).
        phi_unwrapped = phi_true.copy()

    # Number of branch-cut crossings (analytical)
    n_crossings = int(np.abs(phi_true.max() - phi_true.min()) / (2*np.pi))

    # Group delay: d(phi)/d(omega) = beta2*L*omega
    domega = omega[1] - omega[0] if len(omega) > 1 else 1.0
    group_delay_true      = beta2 * L_m * omega
    group_delay_unwrapped = np.gradient(phi_unwrapped, domega)

    # GVD from second derivative: d^2(phi)/d(omega^2) = beta2*L -> divide by L
    d2phi = float(np.median(np.gradient(group_delay_unwrapped, domega)))
    beta2_recovered = d2phi / L_m

    return {
        "omega":               omega,
        "phi_true":            phi_true,
        "phi_wrapped":         phi_wrapped,
        "phi_unwrapped":       phi_unwrapped,
        "group_delay_true":    group_delay_true,
        "group_delay_unwrapped": group_delay_unwrapped,
        "beta2_true":          beta2,
        "beta2_recovered":     beta2_recovered,
        "beta2_error_pct":     abs(beta2_recovered - beta2) / abs(beta2) * 100,
        "n_branch_cut_crossings": n_crossings,
        "nyquist":             nyquist,
    }


def gs_phase_unwrap_and_gvd(E_recovered: np.ndarray,
                              omega: np.ndarray,
                              beta2_nominal: float = None) -> Dict:
    """Extract and unwrap phase from GS-recovered complex field E(omega).

    Steps:
    1. phi_wrapped = angle(E_recovered)  <- in (-pi, pi], may have jumps
    2. phi_unwrapped = unwrap(phi_wrapped)  <- continuous
    3. group_delay(omega) = d(phi_unwrapped)/d(omega)
    4. GVD: beta2_est = d^2(phi_unwrapped)/d(omega^2)  (slope of group delay)

    Phase jumps at spectral nulls (|E|=0) are NOT removable by unwrapping —
    they indicate a pi phase slip (topological defect, not a branch cut).
    These appear as |jump| = pi (not 2*pi).
    """
    domega = omega[1] - omega[0] if len(omega) > 1 else 1.0
    phi_wrapped   = np.angle(E_recovered)
    phi_unwrapped = np.unwrap(phi_wrapped)

    # Detect spectral nulls (amplitude near zero)
    amp = np.abs(E_recovered)
    amp_thresh = 0.01 * amp.max()
    null_mask = amp < amp_thresh

    # Group delay and GVD
    group_delay = np.gradient(phi_unwrapped, domega)
    gvd_est     = np.gradient(group_delay, domega)
    beta2_est   = float(np.median(gvd_est[~null_mask])) if np.any(~null_mask) else 0.0

    # Detect branch-cut crossings vs pi slips
    diffs = np.diff(phi_wrapped)
    bc_crossings = np.where(np.abs(diffs) > 0.9 * 2 * np.pi)[0]   # ~2*pi jump
    pi_slips     = np.where((np.abs(diffs) > 0.9 * np.pi) &
                             (np.abs(diffs) < 1.1 * np.pi))[0]

    result = {
        "phi_wrapped":    phi_wrapped,
        "phi_unwrapped":  phi_unwrapped,
        "amplitude":      amp,
        "group_delay":    group_delay,
        "gvd_estimated":  beta2_est,
        "null_indices":   np.where(null_mask)[0],
        "bc_crossings":   bc_crossings,   # 2*pi jumps (removable)
        "pi_slips":       pi_slips,        # pi jumps at zeros (not removable)
        "n_crossings":    len(bc_crossings),
        "n_pi_slips":     len(pi_slips),
    }
    if beta2_nominal is not None:
        result["beta2_nominal"]  = beta2_nominal
        result["beta2_error_pct"] = abs(beta2_est - beta2_nominal) / abs(beta2_nominal) * 100
    return result


# ════════════════════════════════════════════════════════════════════════════
# §3  RIEMANN SURFACES AND ANALYTIC CONTINUATION
# ════════════════════════════════════════════════════════════════════════════

def riemann_sheet_log(z: np.ndarray, sheet: int = 0) -> np.ndarray:
    """Evaluate log(z) on the kth Riemann sheet.

    Sheet 0 = principal branch, Arg in (-pi, pi].
    Sheet k = principal + 2*pi*k in imaginary part.
    The Riemann surface of log is an infinite helicoid:
      spiraling up as you go around the origin.
    """
    r   = np.abs(z)
    arg = np.angle(z)   # principal Arg in (-pi, pi]
    return np.log(r + 1e-300) + 1j * (arg + 2 * np.pi * sheet)


def analytic_continuation_along_path(f_func,
                                       z_path: np.ndarray,
                                       z0: complex,
                                       f_z0: complex) -> np.ndarray:
    """Continue f analytically along z_path starting from (z0, f_z0).

    At each step: choose branch of f such that f is continuous.
    This is how you analytically continue log around the branch cut.

    f_func : function(z, prev_value) -> (principal value, corrected value)
    z_path : array of complex numbers forming a path
    z0     : starting point
    f_z0   : value at z0
    """
    values = [f_z0]
    for z in z_path[1:]:
        f_principal = np.log(z)
        # Adjust imaginary part to be closest to previous value
        prev = values[-1]
        # Number of 2*pi shifts needed
        diff = np.imag(f_principal) - np.imag(prev)
        k    = int(np.round(diff / (2 * np.pi)))
        f_cont = f_principal - 2j * np.pi * k
        values.append(f_cont)
    return np.array(values)


def loop_around_branch_point(n_pts: int = 360,
                              r: float = 1.0) -> Dict:
    """Trace log(z) as z loops around the branch point z=0.

    Going around once: Im[log] increases by 2*pi (one Riemann sheet up).
    Going around twice: Im[log] increases by 4*pi (two sheets up).
    This is the monodromy of the log function.

    OPTICAL ANALOG: a vortex in the optical field (wavefront dislocation)
    is a branch point of the phase. When the GS algorithm produces a
    topological defect, the phase winds by 2*pi around it.
    """
    theta = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
    z_loop = r * np.exp(1j * theta)

    # Principal branch: jumps at theta=pi (the branch cut)
    phi_principal = np.angle(z_loop)

    # Analytically continued (no jump):
    f_z0  = complex(np.log(z_loop[0]))
    phi_continued = np.imag(analytic_continuation_along_path(
        None, z_loop, z_loop[0], f_z0
    ))

    total_winding = phi_continued[-1] - phi_continued[0]

    # Double loop
    z_loop2   = r * np.exp(2j * theta)
    f_z0_2    = complex(np.log(z_loop2[0]))
    phi_cont2 = np.imag(analytic_continuation_along_path(
        None, z_loop2, z_loop2[0], f_z0_2
    ))

    return {
        "theta":          theta,
        "z_loop":         z_loop,
        "phi_principal":  phi_principal,
        "phi_continued":  phi_continued,
        "total_winding":  float(total_winding),   # ~2*pi after one loop
        "z_loop2":        z_loop2,
        "phi_continued2": phi_cont2,
        "winding_number": round(total_winding / (2*np.pi)),   # 1
        "monodromy":      "Im[log] increases by 2*pi per loop around z=0",
    }


# ════════════════════════════════════════════════════════════════════════════
# §4  BRANCH CUTS IN REAL-TIME TS-DFT — WORKED EXAMPLE
# ════════════════════════════════════════════════════════════════════════════

def tsdft_phase_retrieval_branch_cut_demo(
        beta2: float = -22e-27,   # SMF-28 GVD (s²/m)
        L_km: float = 10.0,
        f_rep_GHz: float = 1.0,
        n_pts: int = 2048) -> Dict:
    """Demonstrate branch cut crossings in TS-DFT GS phase recovery.

    Setup: 100 fs Gaussian pulse through 10 km SMF-28.
    GVD phase: phi(omega) = beta2*L*omega^2/2.
    For beta2=-22e-27 s²/m, L=10km:
      phi spans: 0.5 * 22e-27 * 1e4 * (2*pi*5e12)^2 ~ 108 rad
      = 108/(2*pi) ~ 17 full Riemann sheet crossings.

    The GS algorithm must unwrap these 17 crossings correctly
    to recover the quadratic phase -> correct GVD measurement.
    """
    L_m = L_km * 1e3
    # Frequency axis: ±5 THz around center
    f_max_Hz = 5e12
    f = np.linspace(-f_max_Hz, f_max_Hz, n_pts)
    omega = 2 * np.pi * f
    domega = omega[1] - omega[0]

    # True GVD phase
    phi_true = 0.5 * beta2 * L_m * omega**2
    phi_range = phi_true.max() - phi_true.min()
    n_sheet_crossings = int(phi_range / (2*np.pi))

    # Input pulse (Gaussian, 100 fs)
    T0_s   = 100e-15
    E_in_omega = np.exp(-0.5 * (omega * T0_s)**2)   # transform-limited

    # After fiber: E_out = E_in * exp(i*phi_true)
    E_out_omega = E_in_omega * np.exp(1j * phi_true)

    # What GS sees: only |E_out(omega)|^2 (intensity, no phase)
    I_out = np.abs(E_out_omega)**2

    # GS would recover E_out_omega. Extract and unwrap:
    uwrap = gs_phase_unwrap_and_gvd(E_out_omega, omega,
                                     beta2_nominal=beta2)

    # Recover GVD from unwrapped phase
    beta2_recovered = uwrap["gvd_estimated"]

    return {
        "omega":             omega,
        "f_THz":             f / 1e12,
        "phi_true":          phi_true,
        "phi_wrapped":       uwrap["phi_wrapped"],
        "phi_unwrapped":     uwrap["phi_unwrapped"],
        "phi_range_rad":     float(phi_range),
        "n_sheet_crossings": n_sheet_crossings,
        "beta2_true":        beta2,
        "beta2_recovered":   beta2_recovered,
        "beta2_error_pct":   abs(beta2_recovered - beta2) / abs(beta2) * 100,
        "group_delay_ps":    uwrap["group_delay"] * 1e12,
        "L_km":              L_km,
        "f_rep_GHz":         f_rep_GHz,
        "conclusion": (
            f"GVD phase spans {phi_range/(2*np.pi):.1f} full 2*pi cycles "
            f"({n_sheet_crossings} branch-cut crossings). "
            f"Unwrapping recovers beta2 to "
            f"{abs(beta2_recovered - beta2)/abs(beta2)*100:.2f}% error."
        ),
    }


# ════════════════════════════════════════════════════════════════════════════
# §5  SYMPY: 5 EQUATIONS
# ════════════════════════════════════════════════════════════════════════════

def branch_cuts_sympy_5() -> Dict:
    """5 equations: multi-valued log, principal branch, unwrapping, GVD phase."""
    z, r, theta = sp.symbols("z r theta", complex=True)
    k = sp.Symbol("k", integer=True)
    beta2, L, omega = sp.symbols("beta2 L omega", real=True)

    # 1. Multi-valued log: infinitely many branches
    eq1 = sp.Eq(sp.Symbol("log(z)"),
                sp.log(sp.Abs(z)) + sp.I * (sp.arg(z) + 2*sp.pi*k))

    # 2. Principal branch: Arg in (-pi, pi]
    eq2 = sp.Eq(sp.Symbol("Log(z)"),
                sp.log(sp.Abs(z)) + sp.I * sp.arg(z))

    # 3. Unwrapping condition: remove 2*pi jumps
    phi_n, phi_np1 = sp.symbols("phi_n phi_{n+1}", real=True)
    eq3 = sp.Eq(sp.Symbol("phi_unwrapped_{n+1}"),
                phi_n + sp.Symbol("mod(phi_{n+1} - phi_n + pi, 2*pi)") - sp.pi)

    # 4. GVD phase (parabolic, crosses branch cut many times)
    eq4 = sp.Eq(sp.Symbol("phi_GVD(omega)"),
                sp.Rational(1, 2) * beta2 * L * omega**2)

    # 5. Monodromy: one loop around branch point -> sheet shift
    eq5 = sp.Eq(sp.Symbol("log(r*exp(i*(theta+2*pi)))"),
                sp.log(r) + sp.I*(theta + 2*sp.pi))

    return {
        "multivalued_log":   eq1,
        "principal_branch":  eq2,
        "unwrap_condition":  eq3,
        "GVD_phase":         eq4,
        "monodromy":         eq5,
    }


if __name__ == "__main__":
    print("=== Branch Cut: log(z) across negative real axis ===")
    res = branch_cut_demo()
    print(f"  Mean jump above vs below cut: {res['mean_jump']:.6f}  (theory: {2*np.pi:.6f})")

    print("\n=== Loop Around Branch Point (monodromy) ===")
    loop = loop_around_branch_point()
    print(f"  Total winding: {loop['total_winding']:.4f} rad  (theory: {2*np.pi:.4f})")
    print(f"  Winding number: {loop['winding_number']}")
    print(f"  {loop['monodromy']}")

    print("\n=== GVD Phase Unwrapping (SMF-28, 100 m, narrow BW — Nyquist OK) ===")
    # Short fiber + narrow BW: keeps |Δφ| < π so np.unwrap works
    gvd = gvd_phase_unwrapped(
        omega=np.linspace(-2*np.pi*500e9, 2*np.pi*500e9, 2048),
        beta2=-22e-27, L_m=100.0   # 100 m
    )
    print(f"  Phase Nyquist: {gvd['nyquist']['advice']}")
    print(f"  phi spans: {gvd['phi_true'].min():.2f} to {gvd['phi_true'].max():.2f} rad")
    print(f"  Branch-cut crossings: {gvd['n_branch_cut_crossings']}")
    print(f"  beta2 recovered: {gvd['beta2_recovered']:.4e}  (true: -22e-27)")
    print(f"  Error: {gvd['beta2_error_pct']:.4f}%")

    print("\n=== Phase Nyquist Violated (10 km, ±5 THz) ===")
    ny = phase_nyquist_check(-22e-27, 10e3, 2*np.pi*5e12, 2048)
    print(f"  |dphi|_max = {ny['dphi_max']:.1f} rad  (need < pi = 3.14)")
    print(f"  {ny['advice']}")

    print("\n=== Full TS-DFT Branch Cut Demo (10 km, note Nyquist violation) ===")
    demo = tsdft_phase_retrieval_branch_cut_demo()
    print(f"  phi range: {demo['phi_range_rad']/(2*np.pi):.1f} x 2pi cycles")
    print(f"  In practice: GS algorithm measures I(t) directly ->")
    print(f"    group delay tau(omega) = t_peak(omega) without unwrapping")
    print(f"    beta2 = d(tau)/d(omega)  [measured from I(t) delay vs frequency]")

    print("\n=== SymPy Equations ===")
    for name, eq in branch_cuts_sympy_5().items():
        print(f"  [{name}]  {eq}")
