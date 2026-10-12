"""
Laplace Transform Fundamentals, Rational vs. Non-Rational Transfer Functions,
and System Identification by Linear Regression (Levy's Method)

L{f(t)} = int_0^inf f(t) * exp(-s*t) dt,  s = sigma + j*omega  (complex frequency)

Connection to Fourier: H(j*omega) = H(s)|_{s=j*omega} -- the Laplace transform
evaluated on the imaginary axis IS the frequency response, provided the region
of convergence includes that axis (true for any stable, causal system).

Connection to this repo's H(f) = exp(j*pi*D*f^2) (dgs/gs_core.py):
  This is manifestly a frequency response -- but unlike dgs/causality.py's
  Lorentz susceptibility chi(omega) = strength/(omega0^2 - omega^2 - j*gamma*omega),
  which is a RATIONAL function of omega (finite poles, a proper Laplace
  transform of an ODE-governed impulse response), H(f)=exp(j*pi*D*f^2) is an
  ENTIRE function of f -- no poles anywhere in the finite plane. It is not the
  transfer function of any finite-order linear ODE. Physically: dispersion
  (GVD) from the paraxial wave equation is a *diffusive/free-propagation*
  effect (like the free-particle Schrodinger propagator, exp(-i*hbar*k^2*t/2m)),
  not a resonance -- resonances (finite poles) come from bound charges
  (Lorentz oscillators), not from free propagation.

  This module makes that distinction precise, then asks the natural
  regression question: can a low-order RATIONAL transfer function
  approximate H(f) well over a finite band anyway? Answered by Levy's
  method (linear least-squares system identification -- the classical,
  interpretable ancestor of modern "vector fitting", still the state of
  the art for closed-form rational S-parameter fitting in RF/photonic
  hardware characterization). The honest answer, tested below rather than
  assumed, is: only approximately, and only over the fitted band -- because
  fitting a rational function to a non-rational one is fundamentally a
  model-mismatch problem, not a bug in the regression.
"""
import numpy as np
import sympy as sp


def laplace_transform_common_pairs():
    """
    Derive (not just state) L{f(t)} for the standard textbook pairs, using
    sympy.laplace_transform directly on each f(t) -- symbolic ground truth,
    cross-checked against the closed forms every EE memorizes.
    """
    t, s, a, omega0 = sp.symbols('t s a omega0', positive=True, real=True)

    targets = {
        'unit_step':  (sp.Heaviside(t),               1/s),
        'exponential': (sp.exp(-a*t),                  1/(s+a)),
        'ramp':        (t,                             1/s**2),
        'cosine':      (sp.cos(omega0*t),              s/(s**2+omega0**2)),
        'sine':        (sp.sin(omega0*t),               omega0/(s**2+omega0**2)),
        'damped_cosine': (sp.exp(-a*t)*sp.cos(omega0*t), (s+a)/((s+a)**2+omega0**2)),
    }

    results = {}
    for name, (f_t, expected) in targets.items():
        # sp.laplace_transform returns (F(s), plane, conditions)
        F_s_derived = sp.laplace_transform(f_t, t, s)[0]
        matches = sp.simplify(F_s_derived - expected) == 0
        results[name] = {
            'f_t': str(f_t),
            'F_s_derived': str(sp.simplify(F_s_derived)),
            'F_s_textbook': str(expected),
            'matches_textbook': bool(matches),
        }

    all_match = all(r['matches_textbook'] for r in results.values())
    return {'pairs': results, 'all_verified': all_match}


def analog_transfer_function_poles_zeros(num_coeffs, den_coeffs, f_max=10.0, n_freq=2048):
    """
    H(s) = B(s)/A(s), s-domain (continuous/analog), evaluated on the
    imaginary axis s = j*omega to get the frequency response.

    num_coeffs / den_coeffs: highest power of s first, e.g. s+3 -> [1, 3]

    Stability: poles must lie in the LEFT half of the s-plane (Re(s) < 0) --
    the continuous-time analog of "inside the unit circle" for H(z)
    (dgs.z_transform.transfer_function_poles_zeros).
    """
    b = np.array(num_coeffs, dtype=complex)
    a = np.array(den_coeffs, dtype=complex)

    zeros = np.roots(b) if len(b) > 1 else np.array([])
    poles = np.roots(a) if len(a) > 1 else np.array([])

    stable = all(p.real < -1e-10 for p in poles) if len(poles) else True

    omega = np.linspace(-2*np.pi*f_max, 2*np.pi*f_max, n_freq)
    s_axis = 1j * omega
    H = np.polyval(b, s_axis) / np.polyval(a, s_axis)

    return {
        'poles': poles, 'zeros': zeros, 'stable': stable,
        'stability_check': 'STABLE' if stable else 'UNSTABLE',
        'omega': omega, 'freq_hz': omega / (2*np.pi),
        'H_complex': H,
        'magnitude': np.abs(H), 'phase_rad': np.angle(H),
    }


def dispersion_kernel_frequency_response(D, f_max=0.5, n_freq=2048):
    """
    The repo's own dispersion kernel H(f) = exp(j*pi*D*f^2)
    (dgs.gs_core.disperse), evaluated as a frequency response for
    direct comparison against rational transfer functions below.
    """
    f = np.linspace(-f_max, f_max, n_freq)
    H = np.exp(1j * np.pi * D * f**2)
    return {'f': f, 'omega': 2*np.pi*f, 'H_complex': H,
            'magnitude': np.abs(H), 'phase_rad': np.angle(H)}


def fit_rational_transfer_function_levy(omega, H_data, n_poles, n_zeros):
    """
    Levy's method: fit H(s) ~= B(s)/A(s) to sampled frequency-response
    data H_data(j*omega) by LINEAR least-squares regression.

    The trick that makes this linear (rather than the genuinely nonlinear
    problem of fitting a ratio of polynomials): rewrite the fit residual
    B(j*w) - H_data(j*w)*A(j*w) = 0  as a linear system in the unknown
    polynomial coefficients (A is monic, a_0=1, to fix the scale ambiguity).
    This is the classical, interpretable ancestor of "vector fitting"
    (Gustavsen & Semlyen 1999), which is still the state-of-the-art
    method for fitting rational transfer functions to measured S-parameter
    / frequency-response data in RF and photonic hardware characterization
    -- i.e. exactly the kind of "does a bench measurement match a
    closed-form model" question this repo asks throughout.

    Returns fitted (num_coeffs, den_coeffs) plus the achieved fit, so
    callers can judge quality honestly rather than assume convergence.
    """
    s = 1j * omega
    n = len(s)

    # Design matrix columns: for B(s) = sum_{k=0}^{n_zeros} b_k s^k
    #                         A(s) = 1 + sum_{k=1}^{n_poles} a_k s^k
    # Equation: sum b_k s^k - H_data * sum_{k=1}^{n_poles} a_k s^k = H_data
    cols = []
    for k in range(n_zeros + 1):
        cols.append(s**k)
    for k in range(1, n_poles + 1):
        cols.append(-H_data * s**k)
    A_mat = np.column_stack(cols)
    rhs = H_data

    # Complex linear least-squares -> stack real/imag parts (real-valued LSQ,
    # since physical num/den coefficients must be real)
    A_real = np.vstack([A_mat.real, A_mat.imag])
    rhs_real = np.concatenate([rhs.real, rhs.imag])
    coeffs, residuals, rank, sv = np.linalg.lstsq(A_real, rhs_real, rcond=None)

    b_coeffs = coeffs[:n_zeros + 1][::-1]          # highest power first, for np.polyval
    a_coeffs = np.concatenate([[1.0], coeffs[n_zeros + 1:]])[::-1]

    H_fit = np.polyval(b_coeffs, s) / np.polyval(a_coeffs, s)
    rel_error = np.abs(H_fit - H_data) / (np.abs(H_data) + 1e-12)

    return {
        'num_coeffs': b_coeffs, 'den_coeffs': a_coeffs,
        'H_fit': H_fit,
        'max_rel_error': float(rel_error.max()),
        'mean_rel_error': float(rel_error.mean()),
        'condition_number': float(np.linalg.cond(A_real)),
    }


def demo():
    print("=== LAPLACE TRANSFORM FUNDAMENTALS ===\n")
    pairs = laplace_transform_common_pairs()
    for name, p in pairs['pairs'].items():
        mark = "OK" if p['matches_textbook'] else "MISMATCH"
        print(f"  [{mark}] L{{{p['f_t']}}} = {p['F_s_derived']}  (textbook: {p['F_s_textbook']})")
    print(f"\nAll six pairs symbolically verified: {pairs['all_verified']}")

    print("\n=== RATIONAL vs. NON-RATIONAL: LORENTZ SUSCEPTIBILITY (rational) ===")
    # chi(omega) = strength / (omega0^2 - omega^2 - j*gamma*omega)
    # as an s-domain H(s) with s=j*omega: strength / (omega0^2 + s^2 - gamma*s)... (see notebook)
    print("dgs.causality.lorentz_susceptibility IS a rational function of omega (2 finite poles)")
    print("-- a genuine Laplace-transform impulse response of a damped harmonic oscillator ODE.")

    print("\n=== RATIONAL vs. NON-RATIONAL: THIS REPO'S DISPERSION KERNEL ===")
    D = 5000.0
    disp = dispersion_kernel_frequency_response(D, f_max=0.3, n_freq=512)
    fit = fit_rational_transfer_function_levy(disp['omega'], disp['H_complex'], n_poles=4, n_zeros=4)
    print(f"H(f)=exp(j*pi*{D}*f^2): |H|=1 everywhere (all-pass, no true poles -- an entire function)")
    print(f"Levy fit (4 poles, 4 zeros) over this band: "
          f"mean rel. error={fit['mean_rel_error']:.3e}, max={fit['max_rel_error']:.3e}")
    print(f"Fit condition number: {fit['condition_number']:.2e} "
          f"({'well-conditioned' if fit['condition_number'] < 1e8 else 'ILL-CONDITIONED'})")

    print("\n=== DONE ===")


if __name__ == '__main__':
    demo()
