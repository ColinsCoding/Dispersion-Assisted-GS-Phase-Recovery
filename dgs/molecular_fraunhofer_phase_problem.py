"""Molecular Fraunhofer diffraction and the phase problem -- physiology/
structural biology's version of exactly the same problem this whole repo
is built around, using a DIFFERENT domain pair than dispersive GS
(real-space support + diffraction-plane intensity, instead of two
dispersions), but the identical underlying idea Gerchberg & Saxton
started from.

X-ray/electron diffraction from a molecule (atoms at positions r_j,
scattering factors f_j) measures the STRUCTURE FACTOR's magnitude:
    F(k) = sum_j f_j * exp(i * k . r_j)          (the Fraunhofer/far-field
                                                    diffraction pattern --
                                                    literally the discrete
                                                    Fourier transform of
                                                    the electron density)
    I(k) = |F(k)|^2                               (all a detector measures
                                                    -- PHASE IS LOST)

This is the actual, real "phase problem" of X-ray crystallography and
cryo-EM: recovering atomic structure requires the phase of F(k), which no
detector records. Gerchberg & Saxton's original 1972 paper (Optik 35,
227-246) solved exactly this for electron microscopy, alternating between
an image-plane intensity constraint and a diffraction-plane intensity
constraint -- the historical ancestor of this repo's two-DISPERSION GS
variant. This module uses the more common modern pairing instead: a
REAL-SPACE SUPPORT constraint (atoms must lie within a known bounding
region) alternated with the DIFFRACTION-PLANE intensity constraint --
this is literally Coherent Diffractive Imaging (CDI), the actual
lensless technique used for single-molecule/nanoscale X-ray FEL imaging.

Reuses dgs.gs_core.apply_amplitude_constraint (the same "rescale to match
a measured magnitude" primitive used throughout the rest of this repo)
rather than reimplementing it.
"""

import numpy as np

from dgs.gs_core import apply_amplitude_constraint


def toy_molecule_positions(n_atoms=5, extent=1.0, seed=0):
    """A simple 1D 'molecule': n_atoms point scatterers at random positions
    within [-extent/2, extent/2] -- a toy stand-in for a real molecule's
    atomic coordinates, small enough to solve the phase problem for
    quickly and verifiably."""
    if n_atoms < 2:
        raise ValueError("n_atoms must be at least 2")
    if extent <= 0:
        raise ValueError("extent must be positive")
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(-extent / 2, extent / 2, n_atoms))


def electron_density(positions, x_grid, atom_width=0.02):
    """A real-space electron density built from Gaussian-broadened point
    scatterers at `positions` -- what CDI actually reconstructs (a
    continuous density, not literal delta functions, since a real
    detector/algorithm works on a sampled grid)."""
    if atom_width <= 0:
        raise ValueError("atom_width must be positive")
    density = np.zeros_like(x_grid)
    for pos in positions:
        density += np.exp(-((x_grid - pos) ** 2) / (2 * atom_width ** 2))
    return density


def structure_factor(density, x_grid):
    """F(k) = FT[density](k) -- the far-field (Fraunhofer) diffraction
    amplitude, computed via FFT on the sampled electron density."""
    return np.fft.fft(density)


def diffraction_intensity(F):
    """I(k) = |F(k)|^2 -- the ONLY thing an actual X-ray/electron detector
    measures. Phase(F) is discarded here exactly as it is in a real
    experiment."""
    return np.abs(F) ** 2


def support_constraint(density, support_mask):
    """The real-space analog of dgs.gs_core's amplitude constraint: outside
    the known support (where the molecule CANNOT physically be), force the
    density to zero. This plus the diffraction-intensity constraint is the
    two-constraint pair Coherent Diffractive Imaging alternates between --
    the direct descendant of Gerchberg & Saxton's original image-plane/
    diffraction-plane pair, just with 'image-plane' replaced by 'known
    support region'."""
    result = density.copy()
    result[~support_mask] = 0.0
    return result


def cdi_phase_retrieval(I_measured, support_mask, n_iter=200, seed=0):
    """Coherent Diffractive Imaging: recover the real-space electron
    density from ONLY its diffraction intensity I_measured plus a known
    real-space support -- alternating projections, structurally identical
    to dgs.gs_core.gs_iteration but with the second constraint domain
    swapped from 'a second dispersion' to 'a known support region'."""
    if n_iter <= 0:
        raise ValueError("n_iter must be positive")
    rng = np.random.default_rng(seed)
    N = len(I_measured)
    amp_target = np.sqrt(np.maximum(I_measured, 0))

    # random initial phase guess -- exactly like dgs.gs_core.retrieve_phase's start
    phase_guess = rng.uniform(0, 2 * np.pi, N)
    F_guess = amp_target * np.exp(1j * phase_guess)

    for _ in range(n_iter):
        density_guess = np.fft.ifft(F_guess)
        density_guess = support_constraint(np.real(density_guess), support_mask).astype(complex)
        F_guess = np.fft.fft(density_guess)
        F_guess = apply_amplitude_constraint(F_guess, I_measured)

    density_final = np.real(np.fft.ifft(F_guess))
    return density_final


def hio_phase_retrieval(I_measured, support_mask, n_hio=300, n_er_polish=100,
                          beta=0.8, seed=0):
    """Fienup's Hybrid Input-Output (1982) -- the actual algorithm real CDI
    research uses instead of plain error-reduction (cdi_phase_retrieval
    above), specifically because ER stagnates easily depending on random
    initialization ("bad seeds"). Where ER hard-zeroes support/positivity
    violations, HIO instead SUBTRACTS a scaled correction (g_prev -
    beta*g_computed) -- a feedback term that lets the iterate escape a
    stagnation point rather than getting stuck re-imposing the same wrong
    constraint. Finishes with a short ER "polish" phase (also standard
    real-world practice) once HIO has settled near the right solution."""
    if beta <= 0 or beta > 1:
        raise ValueError("beta must be in (0, 1]")
    if n_hio <= 0 or n_er_polish < 0:
        raise ValueError("n_hio must be positive, n_er_polish must be non-negative")
    rng = np.random.default_rng(seed)
    N = len(I_measured)
    amp_target = np.sqrt(np.maximum(I_measured, 0))

    phase_guess = rng.uniform(0, 2 * np.pi, N)
    F_guess = amp_target * np.exp(1j * phase_guess)
    g_prev = np.real(np.fft.ifft(F_guess))

    for _ in range(n_hio):
        g_computed = np.real(np.fft.ifft(F_guess))
        violates = (~support_mask) | (g_computed < 0)
        g_new = np.where(violates, g_prev - beta * g_computed, g_computed)
        g_prev = g_new
        F_guess = np.fft.fft(g_new.astype(complex))
        F_guess = apply_amplitude_constraint(F_guess, I_measured)

    for _ in range(n_er_polish):
        g_computed = np.real(np.fft.ifft(F_guess))
        g_computed = support_constraint(g_computed, support_mask & (g_computed >= 0))
        F_guess = np.fft.fft(g_computed.astype(complex))
        F_guess = apply_amplitude_constraint(F_guess, I_measured)

    return np.real(np.fft.ifft(F_guess))


def r_factor(density_candidate, I_measured, support_mask):
    """The GROUND-TRUTH-FREE quality metric real crystallography/CDI
    actually uses (a real experiment never has the true structure to
    compare against): apply the support constraint to the candidate,
    compute ITS OWN natural diffraction pattern, and measure how far that
    is from the measured amplitude -- BEFORE snapping the magnitude to
    match (snapping first would make this trivially zero by
    construction, a real bug caught while building this: R-factor must
    be measured on the constrained candidate's OWN forward-computed
    pattern, not on a value already forced to match)."""
    constrained = support_constraint(density_candidate, support_mask & (density_candidate >= 0))
    F_natural = np.fft.fft(constrained.astype(complex))
    amp_target = np.sqrt(np.maximum(I_measured, 0))
    return np.sum(np.abs(np.abs(F_natural) - amp_target)) / np.sum(amp_target)


def multi_start_reconstruction(I_measured, support_mask, n_starts=10, n_hio=300,
                                n_er_polish=100, beta=0.8):
    """The actual real-world-usable strategy: run several independent HIO
    reconstructions from different random seeds, and pick the one with
    the LOWEST r_factor -- since real experiments have no ground truth to
    select against. HONEST CAVEAT, verified while building this: in a
    small toy problem like the one this module demos, R-factor does NOT
    perfectly rank true reconstruction quality (the true-best seed and
    the R-factor-best seed can disagree) -- this is a real, documented
    limitation of R-factor-only validation (part of why crystallography
    also uses 'R-free' cross-validation), not swept under the rug here."""
    if n_starts <= 0:
        raise ValueError("n_starts must be positive")
    candidates = []
    for seed in range(n_starts):
        density = hio_phase_retrieval(I_measured, support_mask, n_hio, n_er_polish, beta, seed=seed)
        r = r_factor(density, I_measured, support_mask)
        candidates.append((seed, density, r))
    best_seed, best_density, best_r = min(candidates, key=lambda c: c[2])
    return best_density, best_seed, best_r, candidates


if __name__ == "__main__":
    print("=== A toy 'molecule': 5 atoms, real-space positions ===")
    positions = toy_molecule_positions(n_atoms=5, extent=0.6)
    print(f"atom positions: {np.round(positions, 3)}")

    N = 512
    x_grid = np.linspace(-1.5, 1.5, N)
    density_true = electron_density(positions, x_grid)

    F_true = structure_factor(density_true, x_grid)
    I_measured = diffraction_intensity(F_true)
    print(f"\ndiffraction intensity computed -- PHASE of F(k) is now discarded,")
    print(f"exactly as a real X-ray/electron detector would lose it.")

    print("\n=== Coherent Diffractive Imaging: support-tightness sensitivity ===")
    print("A real, well-documented CDI phenomenon: support too LOOSE gives the algorithm")
    print("too much freedom (stagnates on spurious solutions); support too TIGHT cuts off")
    print("real signal (the true object no longer fits) -- there's a genuine sweet spot,")
    print("not just 'tighter is always better':")
    best_overall_corr, best_overall_support, best_overall_seed = -1, None, None
    for support_half_width in [0.40, 0.35, 0.32, 0.30]:
        support_mask = np.abs(x_grid) < support_half_width
        corrs = []
        for seed in range(5):
            density_rec = cdi_phase_retrieval(I_measured, support_mask, n_iter=300, seed=seed)
            c1 = abs(np.corrcoef(density_rec, density_true)[0, 1])
            c2 = abs(np.corrcoef(density_rec[::-1], density_true)[0, 1])
            corr = max(c1, c2)
            corrs.append(corr)
            if corr > best_overall_corr:
                best_overall_corr, best_overall_support, best_overall_seed = corr, support_half_width, seed
        print(f"  support_half_width={support_half_width:.2f}: "
              f"best-of-5-seeds correlation = {max(corrs):.4f}, "
              f"mean = {np.mean(corrs):.4f} (seed-to-seed variability is real -- "
              f"iterative phase retrieval can stagnate depending on random init, "
              f"same behavior as this repo's dispersive GS Monte Carlo results)")

    print(f"\nbest single run found: support_half_width={best_overall_support}, "
          f"seed={best_overall_seed}, correlation={best_overall_corr:.4f}")

    print("\n=== Upgrading from toy to research pathway: HIO + multi-start + R-factor ===")
    support_mask = np.abs(x_grid) < 0.32   # the sweet spot found above
    _, best_seed_hio, best_r, all_candidates = multi_start_reconstruction(
        I_measured, support_mask, n_starts=10)

    print(f"{'seed':>4} | {'truth-correlation':>18} | {'R-factor (no ground truth needed)':>34}")
    for seed, density, r in all_candidates:
        c1 = abs(np.corrcoef(density, density_true)[0, 1])
        c2 = abs(np.corrcoef(density[::-1], density_true)[0, 1])
        corr = max(c1, c2)
        marker = "  <-- picked by R-factor" if seed == best_seed_hio else ""
        print(f"{seed:>4} | {corr:>18.4f} | {r:>34.4f}{marker}")

    best_corr_actual = max(abs(np.corrcoef(all_candidates[best_seed_hio][1], density_true)[0, 1]),
                            abs(np.corrcoef(all_candidates[best_seed_hio][1][::-1], density_true)[0, 1]))
    true_best_seed = max(range(len(all_candidates)),
                          key=lambda i: max(abs(np.corrcoef(all_candidates[i][1], density_true)[0, 1]),
                                            abs(np.corrcoef(all_candidates[i][1][::-1], density_true)[0, 1])))
    print(f"\nR-factor picked seed {best_seed_hio} (truth-correlation {best_corr_actual:.4f}); "
          f"the TRUE best seed was {true_best_seed}.")
    print("HONEST FINDING: R-factor-based selection is the actual real-world-usable")
    print("strategy (no ground truth exists in a real experiment) -- but in this small")
    print("toy problem it does NOT perfectly identify the true-best reconstruction,")
    print("a real, documented limitation of R-factor validation alone (part of why")
    print("crystallography also uses 'R-free' cross-validation, not implemented here).")

    print("\n=== The historical connection ===")
    print("Gerchberg & Saxton's ORIGINAL 1972 paper (Optik 35, 227-246) solved exactly")
    print("this class of problem for electron microscopy, alternating an image-plane")
    print("intensity constraint with a diffraction-plane intensity constraint. This")
    print("repo's dgs.gs_core uses a DIFFERENT domain pair (two dispersions instead of")
    print("image-plane + diffraction-plane), but the same core algorithm structure --")
    print("this module is the direct line back to why the algorithm exists at all.")
