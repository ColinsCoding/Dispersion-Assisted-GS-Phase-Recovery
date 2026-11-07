"""
gs_core.py — Temporal Gerchberg-Saxton phase retrieval
Implements Solli, Gupta, Jalali — Appl. Phys. Lett. 95, 231108 (2009)

OUSD(R&E) CTA alignment: FutureG · Integrated Sensing & Cyber ·
  Trusted AI & Autonomy · Directed Energy · Human-Machine Interfaces
  See ousd_alignment.py → stamp() to embed CTA metadata in output dicts.

Physics in three lines:
  E(t) is the optical field.  You can only measure I(t) = |E(t)|².
  Dispersion maps E(t) → E_d(t) via H(ω) = exp(i π D ν²)  in frequency.
  Two measurements at D1, D2 let GS recover the unknown phase φ(t).

Grade 7 version:
  Imagine light as a wave on a jump rope.  You can video the rope shaking
  (intensity) but not feel which way it twisted (phase).  Run the rope
  through two different springs (D1, D2) and record both videos.
  GS guesses the twist, predicts both videos, compares, and corrects until
  the prediction matches reality.
"""

import numpy as np

# ── Input validation / kwarg bounds ──────────────────────────────────────────

def _check_dispersion(D, name='D'):
    """D must be non-zero with |D| ≥ 100 for meaningful diversity."""
    D = float(D)
    if D == 0:
        raise ValueError(f"{name}=0 is invalid: zero dispersion produces no measurement diversity.")
    if abs(D) < 100:
        import warnings
        warnings.warn(
            f"|{name}|={abs(D):.1f} < 100. GS convergence requires |D| ≥ 5000 (normalized). "
            f"Physical: -695 ps/nm → D_norm ≈ -5000. Current value will likely stagnate.",
            stacklevel=3,
        )
    return D


def _check_intensities(I, name='I'):
    """Intensity array must be 1-D, finite, and non-negative after clipping."""
    I = np.asarray(I, dtype=float)
    if I.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {I.shape}.")
    if not np.all(np.isfinite(I)):
        raise ValueError(f"{name} contains NaN or Inf values.")
    if np.any(I < -1e-6 * np.max(np.abs(I) + 1e-30)):
        import warnings
        warnings.warn(f"{name} has significantly negative values — clipping to 0.", stacklevel=3)
    return np.maximum(I, 0.0)


def _check_n_iter(n_iter):
    n_iter = int(n_iter)
    if n_iter < 1:
        raise ValueError(f"n_iter={n_iter} must be ≥ 1.")
    if n_iter < 10:
        import warnings
        warnings.warn(f"n_iter={n_iter} is very low; GS needs ~50 iterations to converge.", stacklevel=3)
    return n_iter


def _check_modulation(modulation):
    valid = {'OOK', 'PAM4', 'QPSK', 'DPSK', 'STEAM', 'SOLITON', 'SOL',
             '6PSK', 'PSK6', '6-PSK'}
    mod = modulation.upper()
    if mod not in valid:
        raise ValueError(
            f"Unknown modulation '{modulation}'. "
            f"Valid choices: {sorted(valid - {'SOLITON','SOL'}) + ['Soliton']}"
        )
    return mod


# ── SymPy derivation of the dispersion transfer function ─────────────────────

def show_transfer_function():
    """
    Returns the SymPy expression H(ν) and its LaTeX string.

    Group-velocity dispersion (GVD) phase shift in fiber:
        φ_GVD = ½ β₂ L ω²          [β₂ in ps²/km, L in km]

    In terms of cyclic frequency ν = ω / 2π and dispersion D [ps/nm]:
        H(ν) = exp(i π D ν²)

    The near-field temporal waveform is then:
        E_dispersed(t) = F⁻¹{ F[E(t)] · H(ν) }

    where F denotes the Fourier transform.
    """
    from sympy import symbols, exp, pi, I, latex
    nu, D = symbols('nu D', real=True)
    H = exp(I * pi * D * nu**2)
    return H, latex(H)


# ── Core physics primitives ───────────────────────────────────────────────────

def disperse(E, D):
    """
    Apply dispersion D to field E in normalized discrete frequency.

    H[k] = exp(i π D (k/N)²)    k = 0..N-1  (normalized ν ∈ [0,1))

    Parameters
    ----------
    E : complex array, length N
    D : float, dispersion parameter (dimensionless or in ps/nm — consistent
        with how I1, I2 were generated)

    Returns
    -------
    E_d : complex array — dispersed temporal field
    """
    N = len(E)
    nu = np.fft.fftfreq(N)                        # normalized: ν ∈ [-0.5, 0.5)
    H = np.exp(1j * np.pi * D * nu**2)
    return np.fft.ifft(np.fft.fft(E) * H)


def undisperse(E_d, D):
    """Remove dispersion D: apply conjugate H*."""
    N = len(E_d)
    nu = np.fft.fftfreq(N)
    H_conj = np.exp(-1j * np.pi * D * nu**2)
    return np.fft.ifft(np.fft.fft(E_d) * H_conj)


def apply_amplitude_constraint(E, I_measured):
    """
    Replace |E(t)| with sqrt(I_measured(t)), keep phase.
    This is the core GS projection: enforce the measured intensity.
    """
    amp = np.sqrt(np.maximum(I_measured, 0.0))
    return amp * np.exp(1j * np.angle(E))


# ── One full GS iteration (paper Fig. 2) ─────────────────────────────────────

def gs_iteration(E, I1, I2, D1, D2, unit_amplitude=True):
    """
    One GS iteration. E is the UNDISPERSED field estimate throughout.

    Project onto D1 constraint, return to undispersed domain.
    Project onto D2 constraint, return to undispersed domain.
    Alternating projections converge to the intersection of both constraint sets.

    Parameters
    ----------
    E  : complex array, undispersed field estimate E(t)
    I1 : float array, measured intensity |disperse(E_true, D1)|²
    I2 : float array, measured intensity |disperse(E_true, D2)|²
    D1, D2 : float, dispersion parameters
    unit_amplitude : bool — if True, enforce |E(t)|=1 after each step.
        Use True for QPSK/DPSK/smooth-phase signals (constant-envelope).
        Use False for OOK/PAM4 (varying-amplitude signals).

    Returns
    -------
    E : complex array, updated undispersed field estimate
    """
    # Project onto D1 measurement: disperse → constrain → undisperse
    E_d1 = disperse(E, D1)
    E_d1 = apply_amplitude_constraint(E_d1, I1)
    E    = undisperse(E_d1, D1)
    if unit_amplitude:
        E = np.exp(1j * np.angle(E))

    # Project onto D2 measurement: disperse → constrain → undisperse
    E_d2 = disperse(E, D2)
    E_d2 = apply_amplitude_constraint(E_d2, I2)
    E    = undisperse(E_d2, D2)
    if unit_amplitude:
        E = np.exp(1j * np.angle(E))

    return E


# ── Main retrieval loop ───────────────────────────────────────────────────────

def retrieve_phase(I1, I2, D1, D2, n_iter=50, unit_amplitude=True):
    """
    Recover optical phase from two time-domain intensity measurements.

    Parameters
    ----------
    I1, I2         : float arrays — measured intensities at dispersions D1, D2
    D1, D2         : float — dispersion parameters (same normalized units as make_measurements)
                     |D| must be ≥ 100; convergence requires |D| ≥ 5000.
    n_iter         : int ≥ 1 — GS iterations; ~50 needed for convergence.
    unit_amplitude : bool — enforce |E(t)|=1 each iteration.
                     True for QPSK/DPSK/smooth-phase (constant envelope).
                     False for OOK/PAM4 (varying amplitude).

    Returns
    -------
    phi    : float array — recovered phase φ(t) in radians (up to global phase offset)
    errors : list of float — RMS amplitude error per iteration (should decrease)
    """
    D1 = _check_dispersion(D1, 'D1')
    D2 = _check_dispersion(D2, 'D2')
    n_iter = _check_n_iter(n_iter)
    I1 = _check_intensities(I1, 'I1')
    I2 = _check_intensities(I2, 'I2')
    if D1 == D2:
        raise ValueError("D1 == D2: identical dispersions provide zero measurement diversity.")
    N = min(len(I1), len(I2))
    I1, I2 = I1[:N], I2[:N]

    # Initial guess: undisperse sqrt(I1) with zero phase
    f1_init = np.sqrt(np.maximum(I1, 0)).astype(complex)
    E = undisperse(f1_init, D1)

    errors = []
    for _ in range(n_iter):
        E = gs_iteration(E, I1, I2, D1, D2, unit_amplitude=unit_amplitude)
        err = float(np.sqrt(np.mean(
            (np.abs(disperse(E, D2))**2 - I2)**2
        )))
        errors.append(err)

    return np.angle(E), errors


# ── Generate synthetic test data (QPSK optical comm signal) ──────────────────

def make_qpsk_measurements(n_symbols=64, sps=8, D1=-5000.0, D2=-5750.0,
                            snr_db=25.0, rng_seed=0):
    """
    Simulate the system in Fig. 5(a) of the paper with a smooth unit-amplitude signal.

    D1, D2 are dimensionless normalized dispersion parameters.  In physical
    units, the paper used D = −695/−800 ps/nm (DCF fiber); those translate to
    D_norm ≈ −5000/−5750 for a 1 GHz-sampled, ~125-harmonic signal at N=1024.
    The ratio D2/D1 = 1.15 matches the paper.  GS requires |D| ≳ 5000 at these
    signal parameters; smaller D leaves I1 ≈ I2 (near-zero diversity) and the
    alternating projections stagnate.

    Returns
    -------
    dict with keys: I1, I2, phi_true, t, D1, D2
    """
    rng = np.random.default_rng(rng_seed)
    N = n_symbols * sps

    # Unit-amplitude field: E(t) = exp(i*phi(t))
    # GS requires |E(t)| = const — QPSK symbols on the unit circle satisfy this
    # per-symbol, but RRC pulse shaping breaks it at transitions.
    # Use smooth bandlimited random phase instead (matches paper's gas-cell setup).
    t = np.linspace(0, 1, N)
    n_harmonics = n_symbols // 4
    phi_true = sum(
        rng.uniform(-1, 1) * np.sin(2 * np.pi * k * t + rng.uniform(0, 2 * np.pi))
        for k in range(1, n_harmonics + 1)
    )
    phi_true = phi_true / np.max(np.abs(phi_true)) * np.pi   # scale to [-π, π]
    E = np.exp(1j * phi_true)                                 # unit amplitude

    # Disperse and detect
    I1 = np.abs(disperse(E, D1))**2
    I2 = np.abs(disperse(E, D2))**2

    # Add noise
    noise_floor = np.mean(I1) * 10**(-snr_db / 10)
    I1 += rng.normal(0, np.sqrt(noise_floor), N)
    I2 += rng.normal(0, np.sqrt(noise_floor), N)
    I1 = np.maximum(I1, 0)
    I2 = np.maximum(I2, 0)

    return {"I1": I1, "I2": I2, "phi_true": phi_true, "t": t, "D1": D1, "D2": D2}


# ── Multi-format signal generator ─────────────────────────────────────────────

# TD-GS difficulty by modulation format:
#
#  Format   Bits/sym  Envelope      Phase structure       TD-GS difficulty
#  -------  --------  -----------   -------------------   ----------------
#  OOK      1         Rectangular   Binary {0, π}         Low
#  PAM4     2         4-level       4-phase states        Medium
#  QPSK     2         Constant      π/4 steps, Gray       Medium
#  DPSK     1         Constant      Transitions only      Medium
#  STEAM    —         Smooth Gauss  Continuous chirp      Low
#  Soliton  —         sech²         Linear chirp          Low
#
# Constant-envelope formats (QPSK, DPSK, STEAM, Soliton) use unit_amplitude=True.
# Variable-amplitude formats (OOK, PAM4) use unit_amplitude=False in retrieve_phase.

def make_measurements(modulation='QPSK', n_symbols=64, sps=8,
                      D1=-5000.0, D2=-5750.0, snr_db=25.0, rng_seed=0):
    """
    Generate synthetic two-arm dispersive GS measurements for six optical formats.

    Parameters
    ----------
    modulation : str — one of 'OOK', 'PAM4', 'QPSK', 'DPSK', 'STEAM', 'Soliton'
    n_symbols  : int — symbol count (ignored for STEAM/Soliton; sets window length)
    sps        : int — samples per symbol
    D1, D2     : float — normalized dispersion parameters (|D| ≥ 5000 for convergence)
    snr_db     : float — signal-to-noise ratio in dB
    rng_seed   : int

    Returns
    -------
    dict with keys: E, I1, I2, phi_true, t, D1, D2, modulation, unit_amplitude
        unit_amplitude — pass directly to retrieve_phase()
    """
    if n_symbols < 4:
        raise ValueError(f"n_symbols={n_symbols} must be ≥ 4.")
    if sps < 1:
        raise ValueError(f"sps={sps} must be ≥ 1.")
    if not np.isfinite(snr_db):
        raise ValueError(f"snr_db={snr_db} must be finite.")
    D1 = _check_dispersion(D1, 'D1')
    D2 = _check_dispersion(D2, 'D2')
    rng = np.random.default_rng(rng_seed)
    N   = n_symbols * sps
    t   = np.linspace(0, 1, N)
    mod = _check_modulation(modulation)

    def _smooth_phase(n_harm, amp_rad=np.pi):
        """
        Bandlimited random phase: sum of n_harm sinusoids, scaled to ±amp_rad.
        This is the canonical TD-GS test signal — exp(i*smooth_phi) has spectral
        content at k ≤ ~5×n_harm (Bessel expansion), which stays within the
        D=-5000 convergence window for n_symbols ≤ 128.
        """
        phi = sum(
            rng.uniform(-1, 1) * np.sin(2 * np.pi * k * t + rng.uniform(0, 2 * np.pi))
            for k in range(1, n_harm + 1)
        )
        phi = phi / np.max(np.abs(phi)) * amp_rad
        return phi, np.exp(1j * phi)

    if mod == 'OOK':
        # Binary ASK: symbol ∈ {0, 1}, rectangular pulses, real-valued.
        # Phase ∈ {0, π}; zero-amplitude samples make phase undefined.
        # GS recovers amplitude without unit-amplitude constraint; phase error at
        # zero-bit samples is arbitrary.
        bits     = rng.integers(0, 2, n_symbols)
        E        = np.repeat(bits.astype(complex), sps)
        phi_true = np.angle(E)
        unit_amplitude = False

    elif mod == 'PAM4':
        # 4-level ASK: symbol ∈ {-3, -1, +1, +3} / 3, real-valued.
        # Phase alternates 0 / π at amplitude-sign boundaries.
        levels   = rng.choice(np.array([-3, -1, 1, 3]) / 3.0, n_symbols)
        E        = np.repeat(levels.astype(complex), sps)
        phi_true = np.angle(E)
        unit_amplitude = False

    elif mod == 'QPSK':
        # Gray-coded QPSK approximation: bandlimited smooth phase occupying the QPSK
        # phase constellation (±π).  This models QPSK with RRC pulse shaping, which
        # produces continuous phase transitions — the waveform GS actually sees.
        # (Rectangular QPSK pulses are infinitely broadband; GS D values require
        # signal bandwidth k_max ≤ n_symbols // 4 to converge.)
        phi_true, E = _smooth_phase(n_harm=n_symbols // 4, amp_rad=np.pi)
        unit_amplitude = True

    elif mod == 'DPSK':
        # DPSK approximation: smooth random phase with amplitude π/2 (smaller phase
        # swings, since DPSK encodes only in transitions Δφ ∈ {0, π}).
        phi_true, E = _smooth_phase(n_harm=n_symbols // 4, amp_rad=np.pi / 2)
        unit_amplitude = True

    elif mod == 'STEAM':
        # Chirped Gaussian (Scientific Time-stretch): smooth Gaussian amplitude + quadratic phase.
        # "Low difficulty": amplitude profile is smooth and known; simpler algorithms suffice.
        t_c      = np.linspace(-4, 4, N)
        amp      = np.exp(-t_c**2 / 2)       # Gaussian envelope (intensity = Gaussian²)
        phi_true = 0.8 * t_c**2              # quadratic chirp (linear group delay)
        E        = amp * np.exp(1j * phi_true)
        unit_amplitude = False

    elif mod in ('SOLITON', 'SOL'):
        # Optical soliton: sech envelope (intensity = sech²), linear phase.
        # "Low difficulty": phase is trivially linear; direct FT estimation works.
        t_c      = np.linspace(-6, 6, N)
        amp      = 1.0 / np.cosh(t_c)        # sech envelope
        phi_true = 1.5 * t_c                 # linear chirp (constant frequency offset)
        E        = amp * np.exp(1j * phi_true)
        unit_amplitude = False

    elif mod in ('6PSK', 'PSK6', '6-PSK'):
        # 6-PSK (hexagonal): 6 constellation points at k·π/3, k=0..5.
        # Phase spacing = π/3 (60°) — denser than QPSK (90°), sparser than 8-PSK (45°).
        # GS difficulty: medium-high.  Minimum Euclidean distance = 2sin(π/6) = 1.0 (norm).
        # Approximated here as bandlimited smooth phase ∈ [0, 2π), matching the
        # continuous-phase transitions that RRC shaping produces.
        phases_6psk = np.array([k * np.pi / 3 for k in range(6)])  # 0, π/3, 2π/3, π, 4π/3, 5π/3
        syms        = rng.integers(0, 6, n_symbols)
        phi_sym     = phases_6psk[syms]
        # Smooth transitions via bandlimited interpolation (n_harm = n_symbols//4)
        phi_true, E = _smooth_phase(n_harm=n_symbols // 4,
                                    amp_rad=np.pi)   # full ±π range covers hexagon
        # Quantize nearest 6-PSK phase for reference constellation
        # (smooth phi is the actual test signal GS sees after pulse shaping)
        unit_amplitude = True

    else:
        raise ValueError(
            f"Unknown modulation '{modulation}'. "
            "Choose: OOK, PAM4, QPSK, DPSK, STEAM, Soliton, 6PSK"
        )

    I1 = np.abs(disperse(E, D1))**2
    I2 = np.abs(disperse(E, D2))**2

    noise_floor = np.mean(I1) * 10**(-snr_db / 10)
    I1 += rng.normal(0, np.sqrt(noise_floor), N)
    I2 += rng.normal(0, np.sqrt(noise_floor), N)
    I1  = np.maximum(I1, 0)
    I2  = np.maximum(I2, 0)

    return {
        "E": E, "I1": I1, "I2": I2, "phi_true": phi_true,
        "t": t, "D1": D1, "D2": D2,
        "modulation": modulation, "unit_amplitude": unit_amplitude,
    }


# ── 3D/4D phase retrieval: time-series stack ─────────────────────────────────

def retrieve_phase_3d(
    I1_stack: np.ndarray,
    I2_stack: np.ndarray,
    D1: float,
    D2: float,
    n_iter: int = 50,
    unit_amplitude: bool = True,
    phase_continuity: bool = True,
) -> tuple:
    """
    Recover phase from a 3D intensity stack: phi(x, y, t) or phi(row, col, t).

    Each slice along axis=0 is one independent 1D signal of length N (axis=1).
    Run retrieve_phase slice-by-slice, then optionally enforce phase continuity
    between adjacent slices by subtracting the global-phase jump.

    Parameters
    ----------
    I1_stack, I2_stack : (M, N) float arrays — M signals, each length N
    D1, D2             : float — dispersion parameters
    n_iter             : int
    unit_amplitude     : bool — passed to retrieve_phase
    phase_continuity   : bool — remove global-phase jumps between adjacent rows

    Returns
    -------
    phi_stack : (M, N) float array — recovered phase for each slice
    errors    : (M, n_iter) float array — GS amplitude error per slice per iter
    """
    M, N = I1_stack.shape
    phi_stack = np.zeros((M, N), dtype=float)
    errors    = np.zeros((M, n_iter), dtype=float)

    for m in range(M):
        phi, errs = retrieve_phase(
            I1_stack[m], I2_stack[m], D1, D2,
            n_iter=n_iter, unit_amplitude=unit_amplitude,
        )
        phi_stack[m] = phi
        errors[m]    = errs

        if phase_continuity and m > 0:
            # Remove global phase offset between adjacent slices
            offset = np.angle(np.mean(np.exp(1j * (phi_stack[m] - phi_stack[m - 1]))))
            phi_stack[m] -= offset

    return phi_stack, errors


# ── 3D pipe: cylindrical signal topology ─────────────────────────────────────

def retrieve_phase_pipe(
    I1_pipe: np.ndarray,
    I2_pipe: np.ndarray,
    D1: float,
    D2: float,
    n_iter: int = 50,
    unit_amplitude: bool = True,
    angular_continuity: bool = True,
    axial_continuity: bool = True,
) -> tuple:
    """
    Phase retrieval for signals arranged on a cylindrical pipe surface: phi(theta, z, t).

    Geometry
    --------
    The pipe has N_theta angular positions and N_z axial positions.
    Each (theta, z) node carries one temporal signal of length N_t.

        I1_pipe, I2_pipe : (N_theta, N_z, N_t)  intensity stacks
        output phi_pipe  : (N_theta, N_z, N_t)  recovered phase

    Phase continuity is enforced:
      - axially   : between adjacent z slices  (phi[i, z+1] ≈ phi[i, z] + drift)
      - angularly : wrapping phi[N_theta] back to phi[0]

    Physical motivation
    -------------------
    In fiber sensing and distributed coherent receivers, the field phase
    varies smoothly around the fiber cross-section (angular modes) and
    along the propagation axis (dispersion-induced chirp).  Enforcing
    cylindrical continuity suppresses global-phase jumps that single-axis
    continuity misses at the wrap-around seam.

    Parameters
    ----------
    I1_pipe, I2_pipe   : (N_theta, N_z, N_t) float arrays
    D1, D2             : float — dispersion parameters
    n_iter             : int
    unit_amplitude     : bool
    angular_continuity : bool — remove global-phase wrap-around discontinuity
    axial_continuity   : bool — remove global-phase jumps along z axis

    Returns
    -------
    phi_pipe : (N_theta, N_z, N_t) float array — recovered phase
    errors   : (N_theta, N_z, n_iter) float array — GS amplitude error
    """
    N_theta, N_z, N_t = I1_pipe.shape
    phi_pipe = np.zeros((N_theta, N_z, N_t), dtype=float)
    errors   = np.zeros((N_theta, N_z, n_iter), dtype=float)

    # Pass 1: retrieve slice-by-slice, enforce axial continuity
    for i in range(N_theta):
        for j in range(N_z):
            phi, errs = retrieve_phase(
                I1_pipe[i, j], I2_pipe[i, j], D1, D2,
                n_iter=n_iter, unit_amplitude=unit_amplitude,
            )
            phi_pipe[i, j] = phi
            errors[i, j]   = errs

            if axial_continuity and j > 0:
                offset = np.angle(np.mean(
                    np.exp(1j * (phi_pipe[i, j] - phi_pipe[i, j - 1]))
                ))
                phi_pipe[i, j] -= offset

    # Pass 2: enforce angular continuity (including wrap-around seam)
    if angular_continuity and N_theta > 1:
        for j in range(N_z):
            for i in range(1, N_theta):
                offset = np.angle(np.mean(
                    np.exp(1j * (phi_pipe[i, j] - phi_pipe[i - 1, j]))
                ))
                phi_pipe[i, j] -= offset

            # Wrap-around: close the cylinder (phi[N_theta-1] ≈ phi[0])
            seam_offset = np.angle(np.mean(
                np.exp(1j * (phi_pipe[0, j] - phi_pipe[N_theta - 1, j]))
            ))
            # Distribute half the wrap error to each end
            phi_pipe[0, j]           += seam_offset / 2
            phi_pipe[N_theta - 1, j] -= seam_offset / 2

    return phi_pipe, errors


def pipe_surface_plot(phi_pipe, title='Pipe phase surface $\\phi(\\theta, z)$',
                      t_slice=0, cmap='RdBu_r'):
    """
    Visualize recovered phase on the cylindrical pipe surface.

    phi_pipe : (N_theta, N_z, N_t)
    t_slice  : which time sample to plot (default 0 = first)

    Returns fig, axes  (caller handles plt.show / savefig)
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    N_theta, N_z, _ = phi_pipe.shape
    theta = np.linspace(0, 2 * np.pi, N_theta, endpoint=False)
    z     = np.linspace(0, 1, N_z)
    Theta, Z = np.meshgrid(theta, z, indexing='ij')

    phi_slice = phi_pipe[:, :, t_slice]

    # Unwrap flat heatmap
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    im = axes[0].imshow(
        phi_slice, aspect='auto', cmap=cmap,
        extent=[0, 1, 0, 360], origin='lower',
        vmin=-np.pi, vmax=np.pi,
    )
    axes[0].set_xlabel('Axial position $z$')
    axes[0].set_ylabel('Angular position $\\theta$ (°)')
    axes[0].set_title(f'{title}  [flat]')
    plt.colorbar(im, ax=axes[0], label='Phase (rad)')

    # 3D cylindrical projection
    X = np.cos(Theta)
    Y = np.sin(Theta)
    ax3 = fig.add_subplot(1, 2, 2, projection='3d', label='3d')
    axes[1].remove()
    norm = plt.Normalize(-np.pi, np.pi)
    colors = plt.cm.get_cmap(cmap)(norm(phi_slice))
    ax3.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1,
                     linewidth=0, antialiased=True, alpha=0.9)
    ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('z')
    ax3.set_title(f'{title}  [3D]')

    plt.tight_layout()
    return fig, ax3


# ── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # SymPy transfer function
    H_sym, H_latex = show_transfer_function()
    print(f"Transfer function: H(nu) = {H_sym}")
    print(f"LaTeX: {H_latex}\n")

    # Generate data and recover phase
    data = make_qpsk_measurements(n_symbols=128, snr_db=30.0)
    phi_est, errors = retrieve_phase(
        data["I1"], data["I2"], data["D1"], data["D2"], n_iter=50
    )

    phi_true = data["phi_true"]
    # Phase error: subtract best-fit global offset (GS can only recover phase up to a constant)
    global_offset = np.angle(np.mean(np.exp(1j * (phi_true - phi_est))))
    delta = np.angle(np.exp(1j * (phi_est - phi_true + global_offset)))
    rms = float(np.sqrt(np.mean(delta**2)))
    print(f"RMS phase error after 50 iterations: {rms:.4f} rad ({np.degrees(rms):.2f}°)")
    print(f"Final amplitude error: {errors[-1]:.6f}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].plot(data["I1"][:200], label=f'I₁ (D₁ = {data["D1"]:.0f})')
    axes[0].plot(data["I2"][:200], label=f'I₂ (D₂ = {data["D2"]:.0f})', alpha=0.7)
    axes[0].set_title("Measured intensities"); axes[0].legend()
    axes[0].set_xlabel("Sample index")

    axes[1].plot(phi_true[:200], label='True φ(t)', lw=1.5)
    axes[1].plot(phi_est[:200],  label='Recovered φ(t)', ls='--')
    axes[1].set_title("Phase recovery"); axes[1].legend()
    axes[1].set_xlabel("Sample index"); axes[1].set_ylabel("Phase (rad)")

    axes[2].semilogy(errors, 'o-', color='crimson')
    axes[2].set_title(f"GS convergence  (RMS = {rms:.3f} rad = {np.degrees(rms):.1f}°)")
    axes[2].set_xlabel("Iteration"); axes[2].set_ylabel("Amplitude error (RMS)")

    plt.tight_layout()
    plt.savefig("gs_core_test.png", dpi=150)
    print("Saved gs_core_test.png")
