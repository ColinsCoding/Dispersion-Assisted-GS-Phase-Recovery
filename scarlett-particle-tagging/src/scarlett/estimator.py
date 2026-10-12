"""Particle estimator: diameter, refractive index, concentration and motion.

Floating-point REFERENCE implementation. The fixed-point version an ASIC
would run is in scarlett.asic, and a C version in c/scarlett_estimator.c.

Angular pattern (diameter, refractive index, concentration)
    The measured voltage per angle channel is modelled as
        v_j = N * T_j(dia, npar) + noise,
    where T_j is the Mie prediction for one particle (a "template") and N is
    the number of particles in the probe volume. For every candidate
    (dia, npar) in a lookup table the best N has a closed form,
        N_k = sum(w v T_k) / sum(w T_k^2),   w = 1 / sigma^2,
    and the candidate with the smallest chi^2 = sum(w (v - N_k T_k)^2) wins.
    The diameter is then refined continuously between grid points.
    Confidence interval: all diameters with chi^2 - chi^2_min below the
    chi-square quantile for one parameter (3.84 at 95 %).

Motion (Brownian diffusion)
    Intensity autocorrelation g2(tau) = 1 + beta exp(-2 Gamma tau), with
    Gamma = D q^2, gives the diffusion coefficient D and, through
    Stokes-Einstein, an independent hydrodynamic diameter.

The confidence intervals describe NOISE only, under the assumption that the
model (a homogeneous sphere, known angles, known index when it is not
fitted) is exactly right. Model error is not included unless you put it in
`sigma` (see combined_sigma).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

from . import physics as ph
from .sensor import OpticalBench, SensorConfig, scattered_power_per_particle


# ----------------------------------------------------------------------
# Templates and lookup table
# ----------------------------------------------------------------------
def particle_template(dia, npar, bench: OpticalBench, sensor: SensorConfig):
    """Expected TIA voltage per channel from ONE particle, dark removed [V]."""
    return sensor.responsivity * sensor.tia_gain * scattered_power_per_particle(dia, npar, bench)


def combined_sigma(volts, sigma_noise, rel_uncertainty=0.02):
    """Noise plus a proportional model/calibration uncertainty, in quadrature.

        sigma_j^2 = sigma_noise_j^2 + (rel_uncertainty * v_j)^2

    The second term is a deliberate allowance for everything the model gets
    slightly wrong (filter ripple, angle calibration, gain drift). Without
    it the fit claims a precision no real instrument has.
    """
    if rel_uncertainty < 0:
        raise ValueError("rel_uncertainty must be non-negative")
    return np.sqrt(np.asarray(sigma_noise, dtype=float) ** 2
                   + (rel_uncertainty * np.asarray(volts, dtype=float)) ** 2)


@dataclass(frozen=True)
class LookupTable:
    dia: np.ndarray          # (K,) candidate diameters [m]
    npar: np.ndarray         # (K,) candidate particle indices [-]
    templates: np.ndarray    # (K, J) volts at the TIA output per particle, dark removed
    mask: np.ndarray         # (J,) channels used in the fit

    def __post_init__(self):
        k, j = self.templates.shape
        if self.dia.shape != (k,) or self.npar.shape != (k,) or self.mask.shape != (j,):
            raise ValueError("inconsistent lookup-table shapes")
        if self.mask.sum() < 3:
            raise ValueError("fewer than 3 usable channels")


def build_lookup_table(diameters, npars, bench: OpticalBench, sensor: SensorConfig):
    """Mie templates for every (diameter, index) pair, index-major order.
    Cost: one Mie evaluation per candidate per channel."""
    diameters = np.atleast_1d(np.asarray(diameters, dtype=float))
    npars = np.atleast_1d(np.asarray(npars, dtype=float))
    if np.any(diameters <= 0) or np.any(npars <= 0):
        raise ValueError("diameters and refractive indices must be positive")
    dia, npar, rows = [], [], []
    for n in npars:
        for d in diameters:
            dia.append(d); npar.append(n)
            rows.append(particle_template(d, n, bench, sensor))
    return LookupTable(np.array(dia), np.array(npar), np.array(rows), bench.usable_channels())


# ----------------------------------------------------------------------
# Angular-pattern estimator
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Estimate:
    dia: float                 # best-fit diameter [m]
    dia_ci: tuple              # confidence interval [m]
    npar: float
    npar_ci: tuple
    n_particles: float         # fitted number of particles in the probe volume
    n_particles_sigma: float
    concentration: float       # [particles/m^3]
    concentration_ci: tuple
    chi2: float
    dof: int
    fit_probability: float     # P(chi^2 >= observed | model correct); small = poor fit
    ambiguous: bool            # a separate, distant diameter fits about as well
    refined: bool              # True if the diameter was refined between grid points
    confidence_level: float
    index: int                 # best row of the lookup table (grid stage)
    chi2_grid: np.ndarray      # (K,) chi^2 of every grid candidate


def _grid_step(values):
    grid = np.unique(values)
    return float(np.min(np.diff(grid))) if grid.size > 1 else 0.0


def estimate_particle(volts, sigma, lut: LookupTable, bench: OpticalBench,
                      sensor: SensorConfig | None = None, confidence=0.95,
                      channel_ok=None, refine=True):
    """Fit one angular measurement.

    volts       (J,) dark-subtracted TIA voltage per channel
    sigma       scalar or (J,) RMS uncertainty of `volts` (positive)
    sensor      needed only for refinement between grid points
    channel_ok  optional (J,) bool; False drops a channel (e.g. ADC clipped)
    """
    volts = np.asarray(volts, dtype=float)
    if volts.shape != lut.mask.shape:
        raise ValueError(f"expected {lut.mask.size} channels, got shape {volts.shape}")
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), volts.shape)
    if np.any(sigma <= 0) or not np.all(np.isfinite(volts)):
        raise ValueError("sigma must be positive and volts finite")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    m = lut.mask if channel_ok is None else lut.mask & np.asarray(channel_ok, dtype=bool)
    if m.sum() < 3:
        raise ValueError("fewer than 3 channels left to fit")

    w = 1.0 / sigma[m] ** 2
    v = volts[m]

    def fit(template):
        """Best particle count for one template: (chi2, N, sum(w T^2))."""
        t = template[m]
        den = float(np.sum(w * t * t))
        scale = max(float(np.sum(w * v * t)) / den, 0.0)   # a count cannot be negative
        return float(np.sum(w * (v - scale * t) ** 2)), scale, den

    grid = np.array([fit(t) for t in lut.templates])
    chi2 = grid[:, 0]
    best = int(np.argmin(chi2))
    delta = stats.chi2.ppf(confidence, df=1)
    dia_step, npar_step = _grid_step(lut.dia), _grid_step(lut.npar)

    dia_hat, npar_hat = float(lut.dia[best]), float(lut.npar[best])
    chi2_min, n_hat, den = grid[best]
    refined = False
    if refine and sensor is not None and dia_step > 0:
        cost = lambda d: fit(particle_template(d, npar_hat, bench, sensor))[0]
        res = optimize.minimize_scalar(cost, bounds=(dia_hat - dia_step, dia_hat + dia_step),
                                       method="bounded", options={"xatol": dia_step * 1e-3})
        if res.fun <= chi2_min:
            dia_hat = float(res.x)
            chi2_min, n_hat, den = fit(particle_template(dia_hat, npar_hat, bench, sensor))
            refined = True

        def edge(direction):
            """Walk outward until chi2 has risen by `delta`, then solve for the crossing."""
            g = lambda d: cost(d) - chi2_min - delta
            span, limit = dia_step / 16, lut.dia.max() - lut.dia.min()
            while span < limit:
                far = dia_hat + direction * span
                if far <= 0:
                    break
                if g(far) > 0:
                    lo, hi = sorted((dia_hat, far))
                    return float(optimize.brentq(g, lo, hi, xtol=dia_step * 1e-3))
                span *= 2
            return float(dia_hat + direction * limit)       # never rose: unconstrained on this side
        dia_ci = (edge(-1), edge(+1))
    else:
        accepted_dia = lut.dia[chi2 - chi2_min <= delta]
        dia_ci = (float(accepted_dia.min() - dia_step / 2), float(accepted_dia.max() + dia_step / 2))

    accepted = chi2 - chi2_min <= delta
    accepted[best] = True
    npar_ci = (float(lut.npar[accepted].min() - npar_step / 2),
               float(lut.npar[accepted].max() + npar_step / 2))
    outside = (lut.dia[accepted] < dia_ci[0] - dia_step) | (lut.dia[accepted] > dia_ci[1] + dia_step)

    n_params = 1 + (dia_step > 0) + (npar_step > 0)
    dof = max(int(m.sum()) - n_params, 1)
    n_sigma = float(1.0 / np.sqrt(den))
    z = stats.norm.ppf(0.5 + confidence / 2)
    return Estimate(
        dia=dia_hat, dia_ci=dia_ci, npar=npar_hat, npar_ci=npar_ci,
        n_particles=float(n_hat), n_particles_sigma=n_sigma,
        concentration=float(n_hat) / bench.probe_volume,
        concentration_ci=(max(n_hat - z * n_sigma, 0.0) / bench.probe_volume,
                          (n_hat + z * n_sigma) / bench.probe_volume),
        chi2=float(chi2_min), dof=dof, fit_probability=float(stats.chi2.sf(chi2_min, dof)),
        ambiguous=bool(np.any(outside)), refined=refined, confidence_level=confidence,
        index=best, chi2_grid=chi2)


def diameter_cramer_rao(dia, npar, n_particles, sigma, bench: OpticalBench,
                        sensor: SensorConfig, rel_step=1e-4):
    """Best achievable 1-sigma diameter precision for this noise level [m].

    Cramer-Rao bound for diameter with the particle count N as a nuisance
    parameter (refractive index assumed known):
        s = N dT/d(dia),  t = T,
        sigma_dia >= 1 / sqrt( s.s - (s.t)^2 / (t.t) )   (dot products weighted by 1/sigma^2)
    The part of the signal change that looks like "more particles" carries no
    size information and is projected out.
    """
    m = bench.usable_channels()
    h = dia * rel_step
    t_hi = particle_template(dia + h, npar, bench, sensor)[m]
    t_lo = particle_template(dia - h, npar, bench, sensor)[m]
    t0 = particle_template(dia, npar, bench, sensor)[m]
    s = n_particles * (t_hi - t_lo) / (2 * h)
    w = 1.0 / np.broadcast_to(np.asarray(sigma, dtype=float), m.shape)[m] ** 2
    info = np.sum(w * s * s) - np.sum(w * s * t0) ** 2 / np.sum(w * t0 * t0)
    if info <= 0:
        raise ValueError("the signal carries no diameter information at this point")
    return float(1.0 / np.sqrt(info))


# ----------------------------------------------------------------------
# Motion estimator (dynamic light scattering)
# ----------------------------------------------------------------------
def intensity_autocorrelation(intensity, max_lag):
    """Normalized g2(tau) = <I(t) I(t+tau)> / <I>^2 for lags 0 .. max_lag."""
    x = np.asarray(intensity, dtype=float)
    if x.ndim != 1 or x.size < 4:
        raise ValueError("intensity must be a 1-D array with at least 4 samples")
    if not 1 <= max_lag < x.size:
        raise ValueError("max_lag must be between 1 and len(intensity) - 1")
    mean = x.mean()
    if mean <= 0:
        raise ValueError("mean intensity must be positive")
    return np.array([np.mean(x[: x.size - lag] * x[lag:]) for lag in range(max_lag + 1)]) / mean ** 2


def fit_decay_rate(g2, dt, floor=0.5):
    """Fit g2 - 1 = beta exp(-2 Gamma tau) by a straight line in log space.

    Uses lags 1 .. L where g2 - 1 stays above `floor` times its lag-1 value
    (lag 0 is excluded: it contains uncorrelated detector noise). Keeping
    only the well-measured early part of the decay avoids taking the log of
    values that are mostly noise.
    Returns (Gamma [1/s], beta).
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    y = np.asarray(g2, dtype=float)[1:] - 1.0
    if y.size < 3 or y[0] <= 0:
        raise ValueError("no positive correlation at lag 1: nothing to fit")
    below = np.flatnonzero(y < floor * y[0])
    stop = below[0] if below.size else y.size
    if stop < 3:
        raise ValueError("correlation decays within 3 samples: sample faster")
    tau = dt * np.arange(1, stop + 1)
    slope, intercept = np.polyfit(tau, np.log(y[:stop]), 1)
    return float(-slope / 2), float(np.exp(intercept))


def estimate_motion(intensity, dt, lam, nmed, theta, max_lag, n_segments=8,
                    temperature=293.15, viscosity=1.0016e-3, confidence=0.95):
    """Diffusion coefficient and hydrodynamic diameter from an intensity trace.

    The trace is split into `n_segments`; each gives its own decay rate, and
    their scatter gives the confidence interval (Student t). Returns a dict.

    Expect a coarse answer. A single-speckle intensity fluctuates by 100 %,
    so the decay rate converges slowly: roughly 5-10 % scatter after a few
    thousand correlation times, plus a few percent upward bias from
    normalizing short segments by their own mean.
    """
    x = np.asarray(intensity, dtype=float)
    if n_segments < 3:
        raise ValueError("n_segments must be at least 3 to estimate a spread")
    seg = x.size // n_segments
    if seg <= max_lag + 1:
        raise ValueError("trace too short for this max_lag and n_segments")
    gammas = np.array([fit_decay_rate(intensity_autocorrelation(x[i * seg:(i + 1) * seg], max_lag), dt)[0]
                       for i in range(n_segments)])
    gamma = float(gammas.mean())
    half = float(stats.t.ppf(0.5 + confidence / 2, n_segments - 1) * gammas.std(ddof=1)
                 / np.sqrt(n_segments))
    if gamma - half <= 0:
        raise ValueError("decay rate not resolved from zero: record a longer trace")
    q2 = float(ph.scattering_vector(lam, nmed, theta)) ** 2
    to_dia = lambda g: float(ph.diameter_from_diffusion(g / q2, temperature, viscosity))
    return {
        "decay_rate": gamma, "decay_rate_ci": (gamma - half, gamma + half),
        "diffusion": gamma / q2, "diffusion_ci": ((gamma - half) / q2, (gamma + half) / q2),
        "hydrodynamic_dia": to_dia(gamma),
        "hydrodynamic_dia_ci": (to_dia(gamma + half), to_dia(gamma - half)),
        "n_segments": n_segments, "confidence_level": confidence,
    }
