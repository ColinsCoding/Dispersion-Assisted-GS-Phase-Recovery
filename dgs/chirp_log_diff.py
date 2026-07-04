"""Three things that turn out to be one idea: the chain rule applied to a
chirp's phase, logarithmic differentiation applied to a complex analytic
signal, and the matrix exponential/logarithm in linear algebra.

QUADRATIC CHIRP + CHAIN RULE: a linear-FM chirp has phase phi(t) =
2*pi*(f0*t + 0.5*chirp_rate*t^2) -- quadratic in t. Its instantaneous
frequency is (1/2pi)*dphi/dt = f0 + chirp_rate*t, a direct chain-rule
application (d/dt[cos(phi(t))] = -sin(phi(t))*phi'(t)). This is exactly
the physics behind this repo's dispersion operator H(f)=exp(j*pi*D*f^2)
-- a quadratic spectral PHASE, which is precisely what CHIRPS a pulse in
time (fiber dispersion, the Coppinger/Jalali time-stretch system).

LOGARITHMIC DIFFERENTIATION, done on a complex analytic signal instead of
a real algebraic expression: for z(t)=A(t)*exp(i*phi(t)), the ordinary
log-differentiation identity d/dt[ln(u)] = u'/u gives
    d/dt[ln z] = (dA/dt)/A(t)  +  i * dphi/dt
Real part = fractional amplitude modulation rate; imaginary part = the
instantaneous ANGULAR frequency, divided by 2*pi to get Hz. This is a
genuine, standard signal-processing technique (instantaneous frequency
estimation from an analytic signal), not just a calculus-class trick --
and it recovers the SAME chain-rule answer as differentiating the phase
directly, verified below.

MATRIX EXPONENTIAL/LOGARITHM: exp(A) for a diagonalizable matrix
A=P*D*P^-1 is P*exp(D)*P^-1 -- literally applying exp() to just the
eigenvalues (the elementwise operation the earlier chirp used pointwise
now happens pointwise on eigenvalues instead of on time samples). This is
what solves the linear ODE dx/dt=Ax exactly: x(t)=exp(A*t)*x0 -- verified
against independent RK4 numerical integration.
"""

import numpy as np
from scipy.linalg import expm

from dgs.causality import hilbert_transform


def quadratic_chirp(t, f0, chirp_rate):
    """A linear-FM ('quadratic phase') chirp: cos(2*pi*(f0*t + 0.5*chirp_rate*t^2))."""
    return np.cos(quadratic_chirp_phase(t, f0, chirp_rate))


def quadratic_chirp_phase(t, f0, chirp_rate):
    """The phase itself: phi(t) = 2*pi*(f0*t + 0.5*chirp_rate*t^2)."""
    t = np.asarray(t, dtype=float)
    return 2 * np.pi * (f0 * t + 0.5 * chirp_rate * t ** 2)


def instantaneous_frequency_analytic(t, f0, chirp_rate):
    """The EXACT chain-rule answer: f_inst(t) = (1/2pi) dphi/dt = f0 + chirp_rate*t."""
    t = np.asarray(t, dtype=float)
    return f0 + chirp_rate * t


def analytic_signal(x):
    """z(t) = x(t) + i*H[x](t), built from dgs.causality.hilbert_transform
    (not reimplemented)."""
    x = np.asarray(x, dtype=float)
    return x + 1j * hilbert_transform(x)


def log_derivative_signal(z, t):
    """The logarithmic-differentiation quantity d/dt[ln z] = z'/z, computed
    directly (numerical derivative of z divided by z) -- exactly the
    "differentiate the log" identity, applied to a complex signal instead
    of a real algebraic expression."""
    t = np.asarray(t, dtype=float)
    z = np.asarray(z, dtype=complex)
    dz_dt = np.gradient(z, t)
    return dz_dt / z


def instantaneous_frequency_from_log_derivative(z, t):
    """Im(d/dt[ln z]) / (2*pi) -- the instantaneous frequency in Hz,
    recovered via logarithmic differentiation of the analytic signal."""
    return np.imag(log_derivative_signal(z, t)) / (2 * np.pi)


def amplitude_modulation_rate_from_log_derivative(z, t):
    """Re(d/dt[ln z]) = (dA/dt)/A -- the fractional amplitude modulation
    rate. Should be ~0 for a constant-envelope chirp."""
    return np.real(log_derivative_signal(z, t))


def matrix_exp_via_eigendecomposition(A):
    """exp(A) = P * exp(D) * P^-1, applying exp() ELEMENTWISE to just the
    eigenvalues -- the linear-algebra analog of applying a scalar function
    pointwise, now pointwise on eigenvalues instead of samples."""
    A = np.asarray(A, dtype=complex)
    eigvals, P = np.linalg.eig(A)
    exp_D = np.diag(np.exp(eigvals))
    result = P @ exp_D @ np.linalg.inv(P)
    return np.real_if_close(result, tol=1000)


def matrix_log_via_eigendecomposition(A):
    """log(A) = P * log(D) * P^-1 -- the inverse operation, same
    eigendecomposition trick. Requires A's eigenvalues to avoid the branch
    cut (real, positive, for the simple np.log used here)."""
    A = np.asarray(A, dtype=complex)
    eigvals, P = np.linalg.eig(A)
    log_D = np.diag(np.log(eigvals))
    result = P @ log_D @ np.linalg.inv(P)
    return np.real_if_close(result, tol=1000)


def solve_linear_ode_via_matrix_exp(A, x0, t):
    """x(t) = exp(A*t) @ x0 for each t in the array -- the matrix
    exponential SOLVING dx/dt=Ax exactly, one exp(A*t) evaluation per
    requested time (via scipy.linalg.expm, an independent, trusted
    implementation, not the eigendecomposition above)."""
    x0 = np.asarray(x0, dtype=float)
    t = np.asarray(t, dtype=float)
    return np.array([expm(A * ti) @ x0 for ti in t])


def solve_linear_ode_via_rk4(A, x0, t):
    """The SAME dx/dt=Ax, solved by an independent method (RK4 time-
    stepping, no matrix exponential anywhere) -- to cross-check the
    matrix-exponential solution against."""
    A = np.asarray(A, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    t = np.asarray(t, dtype=float)
    xs = np.zeros((len(t), len(x0)))
    xs[0] = x0
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        x = xs[i]
        k1 = A @ x
        k2 = A @ (x + 0.5 * dt * k1)
        k3 = A @ (x + 0.5 * dt * k2)
        k4 = A @ (x + dt * k3)
        xs[i + 1] = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return xs


if __name__ == "__main__":
    print("=== Quadratic chirp: chain rule vs. logarithmic differentiation ===")
    f0, chirp_rate = 5.0, 3.0
    t = np.linspace(0, 2, 4000)
    x = quadratic_chirp(t, f0, chirp_rate)
    z = analytic_signal(x)

    f_inst_exact = instantaneous_frequency_analytic(t, f0, chirp_rate)
    f_inst_logdiff = instantaneous_frequency_from_log_derivative(z, t)
    # exclude ~15% from each edge: the FFT-based Hilbert transform assumes
    # periodicity, and a chirp is not periodic on [0,2], so the boundaries
    # (not the interior) carry the artifact -- shown explicitly, not just
    # trimmed away silently
    margin = int(0.15 * len(t))
    interior = slice(margin, -margin)
    edge_err = np.max(np.abs(f_inst_exact[:margin] - f_inst_logdiff[:margin]))
    max_err = np.max(np.abs(f_inst_exact[interior] - f_inst_logdiff[interior]))
    print(f"f_inst(t) = f0 + chirp_rate*t = {f0} + {chirp_rate}*t  (exact, chain rule)")
    print(f"recovered via d/dt[ln(analytic signal)], INTERIOR max error: {max_err:.4f} Hz")
    print(f"  (edge region max error: {edge_err:.4f} Hz -- FFT-Hilbert periodicity artifact,")
    print(f"   not a chirp-tracking failure; a chirp is not periodic on a finite window)")

    amp_rate = amplitude_modulation_rate_from_log_derivative(z, t)
    print(f"amplitude modulation rate (should be ~0, constant envelope), interior: "
          f"max |Re(d/dt ln z)| = {np.max(np.abs(amp_rate[interior])):.4f}")

    print("\n=== Matrix exp/log: linear algebra's version of the same elementwise trick ===")
    A = np.array([[-1.0, 2.0], [0.5, -3.0]])
    expA_eig = matrix_exp_via_eigendecomposition(A)
    expA_scipy = expm(A)
    print(f"exp(A) via eigendecomposition vs scipy.linalg.expm, max error: "
          f"{np.max(np.abs(expA_eig - expA_scipy)):.2e}")

    logA = matrix_log_via_eigendecomposition(A + 5 * np.eye(2))   # shift to avoid negative eigenvalues
    roundtrip = matrix_exp_via_eigendecomposition(logA)
    print(f"exp(log(A+5I)) round-trip error: {np.max(np.abs(roundtrip - (A + 5*np.eye(2)))):.2e}")

    x0 = np.array([1.0, 0.0])
    t_ode = np.linspace(0, 1, 6)
    x_expm = solve_linear_ode_via_matrix_exp(A, x0, t_ode)
    x_rk4 = solve_linear_ode_via_rk4(A, x0, t_ode)
    print(f"dx/dt=Ax: matrix-exponential solution vs independent RK4, max error: "
          f"{np.max(np.abs(x_expm - x_rk4)):.2e}")
