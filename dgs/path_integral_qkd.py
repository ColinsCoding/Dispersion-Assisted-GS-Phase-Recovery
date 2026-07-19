"""Monte Carlo path integrals for the Schrodinger equation + a QKD eavesdropping
estimator -- two places randomness, not determinism, is the right computational
tool.

PART A: free-particle propagator (closed form, used as a sanity check) and a
Euclidean (imaginary-time) Path Integral Monte Carlo (PIMC) for the harmonic
oscillator -- the textbook bridge from Feynman's real-time path integral
sum-over-histories to a Metropolis random walk over discretized paths. PIMC
estimates <x^2> and <H> against the *exact* analytic thermal values for a QHO,
which is the standard way the method is validated in research use.

PART B: a BB84 intercept-resend Monte Carlo, estimating the eavesdropper-induced
quantum bit error rate (QBER) against the analytic 25% prediction. Same
methodology, different physics -- sampling noise to estimate a probability you
could also derive in closed form, which is the whole point of Monte Carlo: use
it where the closed form gets hard, check it where it doesn't.

NumPy only; an optional torch-accelerated batch path sampler is provided for
PIMC when torch is available (py-3.12 env), purely as a speed-up -- the physics
and the numpy reference path are authoritative.
"""

import numpy as np

# -- A1. Free-particle propagator: analytic + a deterministic lattice check ------

def free_particle_propagator_analytic(x, x0, t, m=1.0, hbar=1.0):
    """K(x,t; x0,0) = sqrt(m/(2 pi i hbar t)) exp(i m (x-x0)^2 / (2 hbar t))."""
    if t <= 0:
        raise ValueError("t must be > 0")
    pref = np.sqrt(m / (2j * np.pi * hbar * t))
    return pref * np.exp(1j * m * (x - x0) ** 2 / (2 * hbar * t))


def free_particle_propagator_euclidean_analytic(x, x0, tau, m=1.0, hbar=1.0):
    """Imaginary-time (Euclidean) free-particle kernel K_E(x,tau;x0,0) =
    sqrt(m/(2 pi hbar tau)) exp(-m(x-x0)^2/(2 hbar tau)) -- a real, decaying
    Gaussian (the Wick rotation t -> -i*tau of the oscillatory real-time
    kernel). This is the object Euclidean PIMC actually samples."""
    if tau <= 0:
        raise ValueError("tau must be > 0")
    pref = np.sqrt(m / (2 * np.pi * hbar * tau))
    return pref * np.exp(-m * (x - x0) ** 2 / (2 * hbar * tau))


def free_particle_propagator_two_slice_check(x, x0, tau, m=1.0, hbar=1.0, n_grid=4000, span=12.0):
    """Verify the path-integral composition law in imaginary time: splitting
    [0,tau] into two slices and integrating out the intermediate point x1,

        K_E(x,tau; x0,0) =? Integral dx1  K_E(x,tau/2; x1,0) * K_E(x1,tau/2; x0,0),

    should reproduce the one-slice analytic kernel. (The real-time version of
    this identity is true too, but its kernel is a pure, non-decaying phase --
    see `free_particle_propagator_analytic` -- so a plain trapezoid grid over
    a finite domain converges poorly; that numerical headache is exactly why
    Monte Carlo path integrals are formulated in imaginary time, as the PIMC
    harmonic oscillator below does.)"""
    half = tau / 2.0
    spread = span * np.sqrt(hbar * tau / m)
    x1 = np.linspace(min(x, x0) - spread, max(x, x0) + spread, n_grid)
    K_first = free_particle_propagator_euclidean_analytic(x1, x0, half, m, hbar)
    K_second = free_particle_propagator_euclidean_analytic(x, x1, half, m, hbar)
    K_two_slice = np.trapezoid(K_second * K_first, x1)
    K_analytic = free_particle_propagator_euclidean_analytic(x, x0, tau, m, hbar)
    return K_two_slice, K_analytic


# -- A2. Euclidean Path Integral Monte Carlo for the harmonic oscillator ---------

def pimc_harmonic_oscillator(beta, n_slices=64, n_sweeps=20000, m=1.0, omega=1.0,
                              step_size=0.5, seed=0, burn_in=2000):
    """Metropolis PIMC for V(x) = 1/2 m omega^2 x^2 at inverse temperature beta.

    A closed ring of n_slices imaginary-time beads x_0..x_{n_slices-1} (periodic)
    with discretized Euclidean action

        S_E = sum_i [ m/(2 eps) (x_{i+1}-x_i)^2 + eps * V(x_i) ],   eps = beta/n_slices

    Metropolis moves on individual beads sample the path-integral measure
    exp(-S_E/hbar). Returns estimators for <x^2> and <H> (virial estimator),
    plus their analytic QHO thermal values for comparison.
    """
    rng = np.random.default_rng(seed)
    eps = beta / n_slices
    x = np.zeros(n_slices)

    def local_action_delta(x, i, x_new):
        ip, im = (i + 1) % n_slices, (i - 1) % n_slices
        old = (m / (2 * eps)) * ((x[ip] - x[i]) ** 2 + (x[i] - x[im]) ** 2) + eps * 0.5 * m * omega ** 2 * x[i] ** 2
        new = (m / (2 * eps)) * ((x[ip] - x_new) ** 2 + (x_new - x[im]) ** 2) + eps * 0.5 * m * omega ** 2 * x_new ** 2
        return new - old

    x2_samples, H_samples = [], []
    n_total = n_sweeps + burn_in
    for sweep in range(n_total):
        for i in range(n_slices):
            x_new = x[i] + step_size * rng.standard_normal()
            dS = local_action_delta(x, i, x_new)
            if dS <= 0 or rng.random() < np.exp(-dS):
                x[i] = x_new
        if sweep >= burn_in:
            x2_samples.append(np.mean(x ** 2))
            # virial/thermodynamic estimator for <H> on a ring polymer
            kinetic = n_slices / (2 * beta) - (m / (2 * eps ** 2 * n_slices)) * np.sum((np.roll(x, -1) - x) ** 2)
            potential = np.mean(0.5 * m * omega ** 2 * x ** 2)
            H_samples.append(kinetic + potential)

    x2_samples, H_samples = np.array(x2_samples), np.array(H_samples)
    return {
        "x2_mc": float(np.mean(x2_samples)),
        "x2_mc_err": float(np.std(x2_samples) / np.sqrt(len(x2_samples))),
        "H_mc": float(np.mean(H_samples)),
        "H_mc_err": float(np.std(H_samples) / np.sqrt(len(H_samples))),
        "x2_analytic": qho_thermal_x2(beta, m, omega),
        "H_analytic": qho_thermal_energy(beta, omega),
    }


def qho_thermal_energy(beta, omega, hbar=1.0):
    """<H> = (hbar omega/2) coth(hbar omega beta / 2) -- exact QHO thermal energy."""
    return (hbar * omega / 2) / np.tanh(hbar * omega * beta / 2)


def qho_thermal_x2(beta, m=1.0, omega=1.0, hbar=1.0):
    """<x^2> = (hbar / (2 m omega)) coth(hbar omega beta / 2) -- exact QHO thermal <x^2>."""
    return (hbar / (2 * m * omega)) / np.tanh(hbar * omega * beta / 2)


def pimc_harmonic_oscillator_torch(beta, n_slices=64, n_sweeps=20000, m=1.0,
                                    omega=1.0, step_size=0.5, seed=0, burn_in=2000,
                                    device=None):
    """Same Metropolis PIMC as `pimc_harmonic_oscillator`, but using torch
    tensors so the per-bead Metropolis sweep can run on GPU. Falls back to
    raising ImportError if torch isn't installed -- the numpy version above is
    the reference implementation; this is purely a speed-up."""
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator(device=device).manual_seed(seed)
    eps = beta / n_slices
    x = torch.zeros(n_slices, device=device)

    x2_samples, H_samples = [], []
    n_total = n_sweeps + burn_in
    for sweep in range(n_total):
        for i in range(n_slices):
            ip, im = (i + 1) % n_slices, (i - 1) % n_slices
            x_new = x[i] + step_size * torch.randn((), device=device, generator=g)
            old = (m / (2 * eps)) * ((x[ip] - x[i]) ** 2 + (x[i] - x[im]) ** 2) + eps * 0.5 * m * omega ** 2 * x[i] ** 2
            new = (m / (2 * eps)) * ((x[ip] - x_new) ** 2 + (x_new - x[im]) ** 2) + eps * 0.5 * m * omega ** 2 * x_new ** 2
            dS = new - old
            if dS <= 0 or torch.rand((), device=device, generator=g) < torch.exp(-dS):
                x[i] = x_new
        if sweep >= burn_in:
            x2_samples.append(torch.mean(x ** 2).item())
            kinetic = n_slices / (2 * beta) - (m / (2 * eps ** 2 * n_slices)) * torch.sum((torch.roll(x, -1) - x) ** 2).item()
            potential = torch.mean(0.5 * m * omega ** 2 * x ** 2).item()
            H_samples.append(kinetic + potential)

    x2_samples, H_samples = np.array(x2_samples), np.array(H_samples)
    return {
        "x2_mc": float(np.mean(x2_samples)),
        "H_mc": float(np.mean(H_samples)),
        "x2_analytic": qho_thermal_x2(beta, m, omega),
        "H_analytic": qho_thermal_energy(beta, omega),
        "device": device,
    }


# -- B. BB84 intercept-resend Monte Carlo QBER estimator -------------------------

def bb84_intercept_resend_qber(n_bits=100_000, eavesdrop=True, seed=0):
    """Monte Carlo simulation of BB84 with an optional intercept-resend
    eavesdropper. Alice sends random bits in random bases; Bob measures in a
    random basis; if `eavesdrop`, Eve intercepts, measures in a random basis,
    and resends her (possibly wrong) result before Bob sees it. Returns the
    Monte Carlo QBER on the sifted key (matching bases only) plus the analytic
    prediction (0 with no eavesdropper, 25% with intercept-resend)."""
    rng = np.random.default_rng(seed)
    alice_bits = rng.integers(0, 2, n_bits)
    alice_bases = rng.integers(0, 2, n_bits)   # 0 = Z basis, 1 = X basis

    if eavesdrop:
        eve_bases = rng.integers(0, 2, n_bits)
        # Eve measures in her basis: if it matches Alice's, she gets the bit
        # right; if not, she gets a uniformly random bit (50/50 collapse).
        eve_bits = np.where(
            eve_bases == alice_bases,
            alice_bits,
            rng.integers(0, 2, n_bits),
        )
        # Eve resends in her own basis with her measured bit -- this is what
        # Bob actually receives.
        channel_bits, channel_bases = eve_bits, eve_bases
    else:
        channel_bits, channel_bases = alice_bits, alice_bases

    bob_bases = rng.integers(0, 2, n_bits)
    # Bob's result: correct if his basis matches the basis the photon was
    # actually prepared/resent in; otherwise a fresh random 50/50 collapse.
    bob_bits = np.where(
        bob_bases == channel_bases,
        channel_bits,
        rng.integers(0, 2, n_bits),
    )

    sifted = alice_bases == bob_bases     # public basis-reconciliation step
    n_sifted = int(np.sum(sifted))
    errors = int(np.sum(alice_bits[sifted] != bob_bits[sifted]))
    qber_mc = errors / n_sifted if n_sifted else float("nan")

    return {
        "n_bits": n_bits,
        "n_sifted": n_sifted,
        "qber_mc": qber_mc,
        "qber_analytic": 0.25 if eavesdrop else 0.0,
    }
