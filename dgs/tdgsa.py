"""Time-domain Gerchberg-Saxton with two dispersive fibres (physical units).

Faithful to the ECE 279AS / Jalali-Lab setup: a pulse is split down two fibres of
different group-velocity dispersion (D1, D2 in ps/nm); each arm is detected by a
photodiode + oscilloscope, giving two intensity-only measurements I1, I2; the GS
loop alternates between the two dispersed planes to recover the phase. Civilian
optical metrology / single-shot spectroscopy. Standalone.
"""

import numpy as np

C_LIGHT = 299_792_458.0          # m/s


def gdd_ps2(D_ps_per_nm, lambda_nm=1550.0):
    """Group-delay dispersion beta2*L (ps^2) from total fibre dispersion D (ps/nm):
    beta2 L = -D lambda^2 / (2 pi c)."""
    lam_m = lambda_nm * 1e-9
    D_s_per_m = D_ps_per_nm * 1e-3                       # ps/nm -> s/m
    beta2L_s2 = -D_s_per_m * lam_m**2 / (2 * np.pi * C_LIGHT)
    return beta2L_s2 * 1e24                              # s^2 -> ps^2


def disperse(field, D_ps_per_nm, dt_ps, lambda_nm=1550.0):
    """Propagate a field through dispersion D (ps/nm); dt_ps = sample spacing (ps).
    H(f) = exp(i * 0.5 * beta2L * (2 pi f)^2), f in 1/ps."""
    N = len(field)
    f = np.fft.fftfreq(N, dt_ps)
    H = np.exp(1j * 0.5 * gdd_ps2(D_ps_per_nm, lambda_nm) * (2 * np.pi * f)**2)
    return np.fft.ifft(np.fft.fft(field) * H)


def make_pulse(t_ps, T0_ps=2000.0, envelope_chirp=1.0, phase_kind="none", phase_coef=1.0):
    """A chirped Gaussian E(t) = exp(-(1+iC)/2 (t/T0)^2) with an extra engineered
    phase: 'none', 'quadratic' (even degree), or 'cubic' (odd degree).

    Returns (E, amplitude, phase).
    """
    tau = t_ps / T0_ps
    env = np.exp(-(1 + 1j * envelope_chirp) / 2 * tau**2)
    if phase_kind == "none":
        extra = np.zeros_like(t_ps)
    elif phase_kind == "quadratic":
        extra = phase_coef * tau**2
    elif phase_kind == "cubic":
        extra = phase_coef * tau**3
    else:
        raise ValueError("phase_kind must be none/quadratic/cubic")
    E = env * np.exp(1j * extra)
    return E, np.abs(E), np.angle(E)


def tdgsa(I1, I2, D1, D2, dt_ps, n_iter=100, init="warm", seed=0, lambda_nm=1550.0):
    """Recover the object field from intensities at two dispersions D1, D2.

    init='warm' starts from sqrt(I1) back-propagated with zero phase (the
    no-chirp guess Gabriel used); init='random' uses a random phase (which tends
    to stall). Returns (x_recovered, errors).
    """
    if init not in ("warm", "random"):
        raise ValueError("init must be 'warm' or 'random'")
    A1, A2 = np.sqrt(np.maximum(I1, 0)), np.sqrt(np.maximum(I2, 0))
    if init == "warm":
        x = disperse(A1.astype(complex), -D1, dt_ps, lambda_nm)      # zero-phase guess
    else:
        rng = np.random.default_rng(seed)
        x = disperse(A1 * np.exp(1j * rng.uniform(-np.pi, np.pi, len(A1))), -D1, dt_ps, lambda_nm)
    errors = []
    for _ in range(n_iter):
        d1 = disperse(x, D1, dt_ps, lambda_nm)
        d1 = A1 * np.exp(1j * np.angle(d1))                          # enforce I1
        x = disperse(d1, -D1, dt_ps, lambda_nm)
        d2 = disperse(x, D2, dt_ps, lambda_nm)
        d2 = A2 * np.exp(1j * np.angle(d2))                          # enforce I2
        x = disperse(d2, -D2, dt_ps, lambda_nm)
        errors.append(float(np.linalg.norm(np.abs(disperse(x, D2, dt_ps, lambda_nm)) - A2)
                            / (np.linalg.norm(A2) + 1e-12)))
    return x, np.array(errors)


def align_phase(phi_rec, phi_true, weight):
    """Amplitude-weighted RMS phase error, modulo global offset and twin."""
    best = None
    for sign in (+1, -1):
        d = phi_true - sign * phi_rec
        offset = np.angle(np.sum(weight * np.exp(1j * d)))
        aligned = sign * phi_rec + offset
        err = np.sqrt(np.sum(weight * np.angle(np.exp(1j * (phi_true - aligned)))**2)
                      / np.sum(weight))
        if best is None or err < best[0]:
            best = (err, aligned)
    return best
