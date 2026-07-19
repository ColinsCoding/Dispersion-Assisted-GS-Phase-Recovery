"""1D FDTD -- the Yee-grid engine that lives underneath Ansys Lumerical/FDTD Solutions.

Commercial photonic FDTD tools do exactly this, just in 3D with a GUI on top:
march E and H fields forward in time on an interleaved (staggered) grid, inject
a source, absorb outgoing waves at the boundary, and run a running Fourier
transform ("frequency-domain monitor") at fixed points to turn the time-domain
result into a transmission/reflection spectrum.

Yee grid (1D, TEz: Ez and Hy only) update equations from Maxwell's curl equations:
    dHy/dt = -(1/mu0)   dEz/dx
    dEz/dt = -(1/eps0 eps_r)  dHy/dx
finite-differenced on a grid staggered by half a cell in space and half a step
in time -- this module IS dgs.em_dispersion's H(omega) = exp(i beta_2 omega^2 L/2)
derived the other way: instead of assuming a dispersion relation and propagating
a spectrum, FDTD propagates the raw fields and lets the dispersion/reflection
physics emerge from the material profile eps_r(x). The Fabry-Perot slab test
below is the direct time-domain analog of dgs.photonic_circuits.ring_transfer_function.

NumPy only. Education -- a from-scratch look at what Ansys/Lumerical automate.
"""

import numpy as np

from dgs.em_dispersion import C

EPS0 = 8.8541878128e-12   # vacuum permittivity [F/m]
MU0 = 4e-7 * np.pi        # vacuum permeability [H/m]


def courant_dt(dx, S=0.99, c=C):
    """Time step from the 1D Courant stability limit dt = S*dx/c, S in (0,1].
    S=1 is the exact 1D magic-time-step; S<1 leaves margin for the numerics."""
    if dx <= 0:
        raise ValueError(f"dx must be positive, got {dx}")
    if not (0 < S <= 1):
        raise ValueError(f"Courant number S must be in (0, 1], got {S}")
    return S * dx / c


def gaussian_pulse(t, t0, spread):
    """Gaussian envelope source exp(-((t-t0)/spread)^2) -- a broadband pulse
    whose Fourier transform is itself Gaussian, wide enough to probe a spectrum
    of frequencies in one FDTD run (same trick as the ultrashort pulse in
    Coppinger/Jalali's photonic time-stretch: one shot excites many frequencies)."""
    if spread <= 0:
        raise ValueError(f"spread must be positive, got {spread}")
    return np.exp(-((t - t0) / spread) ** 2)


def slab_eps_profile(Nx, slab_start, slab_cells, n_slab, n_bg=1.0):
    """Relative permittivity array eps_r = n^2 for a single dielectric slab of
    index n_slab embedded in background index n_bg -- the FDTD equivalent of
    specifying a material region in Ansys/Lumerical's structure editor."""
    if not (0 <= slab_start < slab_start + slab_cells <= Nx):
        raise ValueError("slab region must lie inside the grid [0, Nx)")
    if n_slab <= 0 or n_bg <= 0:
        raise ValueError("refractive indices must be positive")
    eps_r = np.full(Nx, n_bg ** 2, dtype=float)
    eps_r[slab_start:slab_start + slab_cells] = n_slab ** 2
    return eps_r


def run_fdtd_1d(eps_r, dx, n_steps, source_index, source_waveform,
                 probe_index, freqs=None, S=0.99):
    """March the 1D Yee grid n_steps times with a soft source at source_index
    and first-order Mur absorbing boundaries at both ends (so outgoing waves
    leave instead of reflecting off the domain edge -- the FDTD stand-in for
    Ansys's PML). Records the Ez time series at probe_index and, if freqs is
    given, accumulates a running discrete Fourier transform there (a
    frequency-domain monitor): spectrum(f) = sum_n Ez[probe,n] exp(-i 2 pi f t_n) dt.

    source_waveform(t) -> scalar, added directly to Ez[source_index] each step
    (a soft source: it superposes with whatever is already there, so reflected
    light passes through it undisturbed, same as leaving a Lumerical source
    "on" throughout the run)."""
    eps_r = np.asarray(eps_r, dtype=float)
    Nx = len(eps_r)
    if np.any(eps_r < 1.0):
        raise ValueError("eps_r must be >= 1 everywhere (no gain/vacuum-violating media)")
    if not (0 <= source_index < Nx) or not (0 <= probe_index < Nx):
        raise ValueError("source_index/probe_index must lie inside the grid")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")

    dt = courant_dt(dx, S)
    Ez = np.zeros(Nx)
    Hy = np.zeros(Nx - 1)

    Ez_probe = np.zeros(n_steps)
    do_dft = freqs is not None
    if do_dft:
        freqs = np.asarray(freqs, dtype=float)
        spectrum = np.zeros(len(freqs), dtype=complex)

    coeff_abc = (C * dt - dx) / (C * dt + dx)

    for n in range(n_steps):
        t = n * dt

        # save pre-update boundary + interior-neighbor values for Mur ABC
        e0_old, e1_old = Ez[0], Ez[1]
        eN_old, eNm1_old = Ez[-1], Ez[-2]

        # H update (curl of E)
        Hy += (dt / (MU0 * dx)) * (Ez[1:] - Ez[:-1])
        # E update (curl of H, scaled by local eps_r)
        Ez[1:-1] += (dt / (EPS0 * eps_r[1:-1] * dx)) * (Hy[1:] - Hy[:-1])

        # soft source injection
        Ez[source_index] += source_waveform(t)

        # first-order Mur ABC (absorbs outgoing waves at both ends)
        Ez[0] = e1_old + coeff_abc * (Ez[1] - e0_old)
        Ez[-1] = eNm1_old + coeff_abc * (Ez[-2] - eN_old)

        Ez_probe[n] = Ez[probe_index]
        if do_dft:
            spectrum += Ez[probe_index] * np.exp(-1j * 2 * np.pi * freqs * t) * dt

    result = {"Ez_probe": Ez_probe, "dt": dt}
    if do_dft:
        result["spectrum"] = spectrum
        result["freqs"] = freqs
    return result


def fabry_perot_power_reflectivity(n_slab, n_bg=1.0):
    """Single-interface normal-incidence power reflectivity R = ((n_slab-n_bg)/(n_slab+n_bg))^2
    (Fresnel). Same role as the field self-coupling r in dgs.photonic_circuits'
    ring resonator -- how much amplitude bounces at one "mirror"."""
    return ((n_slab - n_bg) / (n_slab + n_bg)) ** 2


def fabry_perot_transmission(freqs, n_slab, L, n_bg=1.0):
    """Analytic (transfer-matrix/Airy) power transmission spectrum of a lossless
    dielectric slab of index n_slab, thickness L, sandwiched in background n_bg:
        T(delta) = 1 / (1 + F sin^2(delta)),   F = 4R / (1-R)^2
    with single-pass phase delta = n_slab*k0*L, k0 = 2*pi*f/c, R the single-face
    Fresnel reflectivity above. This is the exact same Airy lineshape as
    dgs.photonic_circuits.finesse's ring -- a slab is a straightened-out ring
    resonator (a Fabry-Perot etalon), so this function is the ground truth the
    time-domain FDTD run below must reproduce."""
    freqs = np.asarray(freqs, dtype=float)
    if L <= 0:
        raise ValueError(f"L must be positive, got {L}")
    R = fabry_perot_power_reflectivity(n_slab, n_bg)
    F = 4 * R / (1 - R) ** 2
    k0 = 2 * np.pi * freqs / C
    delta = n_slab * k0 * L
    return 1.0 / (1 + F * np.sin(delta) ** 2)


def slab_transmission_fdtd(n_slab, L, freqs, dx, pad_cells=60, pulse_spread=None, S=0.99):
    """Run the FDTD twice (reference: vacuum everywhere, structure: slab present)
    and take the ratio of frequency-domain monitors at the same far probe point --
    exactly the normalization workflow Lumerical/Ansys use to turn a raw field
    recording into a transmission spectrum, T(f) = |E_structure(f)/E_reference(f)|^2.
    Returns (T_fdtd, T_analytic) so the numerics can be checked against
    fabry_perot_transmission.

    pulse_spread defaults to 1/(pi*max(freqs)) -- short enough in time that the
    (unmodulated, baseband) Gaussian's spectrum still has appreciable content out
    at the requested optical frequencies; n_steps is sized generously (6 domain
    transit times) so the slab's ring-down fully leaves before the DFT window ends,
    which matters far more for accuracy here than grid resolution does."""
    slab_cells = int(round(L / dx))
    if slab_cells < 1:
        raise ValueError("dx too coarse to resolve slab thickness L")
    Nx = 2 * pad_cells + slab_cells
    source_index = pad_cells // 2
    slab_start = pad_cells
    probe_index = Nx - pad_cells // 2 - 1

    if pulse_spread is None:
        pulse_spread = 1.0 / (np.pi * np.max(freqs))
    t0 = 8 * pulse_spread
    n_steps = int(round((t0 + 6 * Nx * dx / C) / courant_dt(dx, S)))

    def source(t):
        return gaussian_pulse(t, t0, pulse_spread)

    eps_ref = np.ones(Nx)
    ref = run_fdtd_1d(eps_ref, dx, n_steps, source_index, source,
                       probe_index, freqs=freqs, S=S)

    eps_slab = slab_eps_profile(Nx, slab_start, slab_cells, n_slab)
    struct = run_fdtd_1d(eps_slab, dx, n_steps, source_index, source,
                          probe_index, freqs=freqs, S=S)

    T_fdtd = np.abs(struct["spectrum"] / ref["spectrum"]) ** 2
    T_analytic = fabry_perot_transmission(freqs, n_slab, L)
    return T_fdtd, T_analytic


if __name__ == "__main__":
    dx = 2e-8  # 20 nm cells -- resolves near-IR wavelengths ~50x/wavelength
    n_slab, L = 2.0, 2e-6  # silicon-ish index, 2 um slab
    freqs = np.linspace(150e12, 250e12, 9)  # near-IR, telecom-adjacent band

    T_fdtd, T_analytic = slab_transmission_fdtd(n_slab, L, freqs, dx)
    print("f (THz)   T_fdtd   T_analytic")
    for f, tf, ta in zip(freqs, T_fdtd, T_analytic):
        print(f"{f/1e12:7.1f}   {tf:.4f}    {ta:.4f}")
    print(f"max |T_fdtd - T_analytic| = {np.max(np.abs(T_fdtd - T_analytic)):.4f}")
