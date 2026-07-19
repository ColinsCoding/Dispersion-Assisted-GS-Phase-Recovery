"""Cell biology, the Jalali-lab way: diffusion, heredity, and photosynthesis are
each a physics/CS topic already in this repo wearing a biology costume.

  * MEMBRANE TRANSPORT (Fick's law, osmosis) -- a diffusion equation, the same
    object as the imaginary-time free-particle spread in
    dgs.path_integral_qkd (<x^2> grows linearly in time there too).
  * HEREDITY (Mendelian crosses) -- a discrete probability distribution
    (binomial/multinomial), the same machinery as dgs.bayes_inference.
  * PHOTOSYNTHESIS (light reactions) -- Beer-Lambert absorption and a
    wavelength-dependent action spectrum, literally optics: an absorption
    spectrometer measurement, just done by a leaf instead of a photodiode.

The "yeast -> plant cell" level-up: `yeast_cell_water_balance` has osmosis
only (no light-harvesting machinery); `plant_cell_water_balance` adds the
rigid cell wall (turgor pressure) AND a chloroplast absorbing light --
strictly more physics layered on the same diffusion core.
"""

import math
import numpy as np

_R = 8.314462618   # gas constant [J/(mol K)]


# -- Membrane transport: Fick's law and osmosis ----------------------------------

def fick_flux(D, dC_dx, A=1.0):
    """Fick's first law: J = -D * A * dC/dx [mol/s] (diffusive flux down a
    concentration gradient; D is the diffusion coefficient, A the membrane area)."""
    return -D * A * dC_dx


def diffusion_time_scale(L, D):
    """Characteristic diffusion time t ~ L^2 / (2D) for a concentration front to
    cross distance L -- the same x^2 ~ (diffusion coeff) * t scaling as the
    imaginary-time random walk in dgs.path_integral_qkd.pimc_harmonic_oscillator
    (there hbar/m plays the role of D); diffusion is slow precisely because
    distance only grows as sqrt(time), not time."""
    return L ** 2 / (2 * D)


def osmotic_pressure(total_solute_molarity, T=298.0):
    """van't Hoff osmotic pressure Pi = i*M*R*T [Pa] for total solute molarity M
    [mol/m^3] (i, the van't Hoff factor for dissociation, folded into M here)."""
    return total_solute_molarity * _R * T


def osmotic_water_flux(Pi_in, Pi_out, Lp, A=1.0):
    """Water volume flux J_v = Lp * A * (Pi_in - Pi_out) [m^3/s] across a membrane
    of hydraulic permeability Lp; positive = net water influx (hypotonic outside)."""
    return Lp * A * (Pi_in - Pi_out)


# -- Heredity: Mendelian crosses as a discrete probability distribution ----------

def punnett_cross(allele_pair_1, allele_pair_2):
    """All four gametic combinations of a monohybrid cross, e.g.
    punnett_cross('Aa', 'Aa') -> {'AA': 1, 'Aa': 2, 'aa': 1} (counts out of 4),
    the textbook Punnett square as a frequency table instead of a grid."""
    from collections import Counter
    counts = Counter()
    for a in allele_pair_1:
        for b in allele_pair_2:
            genotype = "".join(sorted(a + b, key=str.islower))  # dominant (uppercase) first
            counts[genotype] += 1
    return dict(counts)


def phenotype_ratio(genotype_counts, dominant_allele):
    """Collapse genotype counts into a dominant/recessive phenotype ratio.
    A phenotype is recessive only if it has zero copies of `dominant_allele`."""
    dominant = sum(c for g, c in genotype_counts.items() if dominant_allele in g)
    recessive = sum(genotype_counts.values()) - dominant
    return {"dominant": dominant, "recessive": recessive}


def chi_square_heredity_test(observed, expected):
    """Pearson chi-square statistic sum((O-E)^2/E) for testing observed offspring
    counts against a Mendelian-ratio null hypothesis (e.g. expected 3:1)."""
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    return float(np.sum((observed - expected) ** 2 / expected))


# -- Photosynthesis: light absorption is an optics measurement ------------------

def beer_lambert_absorption(I0, epsilon, concentration, path_length):
    """Beer-Lambert law: I = I0 * exp(-epsilon * c * L) -- the SAME exponential
    attenuation law as fiber/atmospheric optical loss, just with a pigment
    (chlorophyll) instead of a fiber's intrinsic loss coefficient."""
    return I0 * np.exp(-epsilon * concentration * path_length)


def chlorophyll_action_spectrum(wavelength_nm):
    """A simplified two-pigment action spectrum: chlorophyll a (peaks ~430, 662 nm)
    + chlorophyll b (peaks ~453, 642 nm), each a sum of Gaussian absorption bands.
    Returns relative absorption in [0,1] -- not a real spectrophotometer trace,
    but the right qualitative shape (blue and red peaks, a green "gap" near
    550 nm, which is *why* leaves look green: that band is reflected, not absorbed)."""
    wl = np.asarray(wavelength_nm, dtype=float)

    def gaussian_band(center, width, height):
        return height * np.exp(-((wl - center) / width) ** 2)

    chl_a = gaussian_band(430, 20, 1.0) + gaussian_band(662, 18, 0.85)
    chl_b = gaussian_band(453, 22, 0.75) + gaussian_band(642, 20, 0.55)
    total = chl_a + chl_b
    return total / np.max(total)


def quantum_yield(electrons_transferred, photons_absorbed):
    """Quantum yield phi = electrons transferred / photons absorbed. Photosystem
    II in vivo runs near phi ~ 0.85-1.0 under low light -- a near-unity quantum
    efficiency that solid-state photodetectors are explicitly benchmarked against."""
    if photons_absorbed <= 0:
        raise ValueError("photons_absorbed must be > 0")
    return electrons_transferred / photons_absorbed


# -- Level-up: yeast cell (diffusion+osmosis only) -> plant cell (+ wall + light) -

def yeast_cell_water_balance(C_in, C_out, Lp=1e-12, A=50e-12, T=298.0):
    """A free-floating yeast cell: water flux is set by osmosis alone (no rigid
    wall to push back). Returns the osmotic pressures and the resulting water
    flux; a real yeast cell in a hypotonic bath would lyse without a wall --
    which is exactly why plants (below) evolved one."""
    Pi_in = osmotic_pressure(C_in, T)
    Pi_out = osmotic_pressure(C_out, T)
    J_v = osmotic_water_flux(Pi_in, Pi_out, Lp, A)
    return {"Pi_in": Pi_in, "Pi_out": Pi_out, "water_flux": J_v}


def plant_cell_water_balance(C_in, C_out, Lp=1e-12, A=50e-12, T=298.0,
                              wall_elastic_modulus=1e7, wall_strain=0.0,
                              light_intensity=1.0, wavelength_nm=662.0):
    """A plant cell: same osmotic core as the yeast model, PLUS (1) a rigid cell
    wall that builds back-pressure (turgor) as it stretches, which caps net
    water influx instead of letting the cell lyse, and (2) a chloroplast that
    absorbs `light_intensity` according to the chlorophyll action spectrum at
    `wavelength_nm`, producing photosynthetic quantum yield. Two new physics
    layers on the identical diffusion/osmosis core -- the "level up"."""
    Pi_in = osmotic_pressure(C_in, T)
    Pi_out = osmotic_pressure(C_out, T)

    # turgor pressure resists influx like a spring: P_wall = E * strain
    P_wall = wall_elastic_modulus * wall_strain
    net_driving_pressure = (Pi_in - Pi_out) - P_wall
    J_v = Lp * A * net_driving_pressure

    absorption = float(chlorophyll_action_spectrum(wavelength_nm))
    absorbed_light = light_intensity * absorption

    return {
        "Pi_in": Pi_in, "Pi_out": Pi_out, "P_wall": P_wall,
        "water_flux": J_v, "absorbed_light_fraction": absorption,
        "absorbed_light": absorbed_light,
    }
