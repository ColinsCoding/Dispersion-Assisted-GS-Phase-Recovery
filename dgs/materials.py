"""
Material library: refractive index and dispersion of optical materials used in imaging.

In plain words
--------------
Every lens, window, fiber and crystal in an imaging setup bends and delays light by an
amount that depends on the color. One equation, the Sellmeier equation, describes that for
most transparent materials:

    n^2(lambda) = 1 + sum_i  B_i lambda^2 / (lambda^2 - C_i)        (lambda in micrometers)

Store three B's and three C's per material and everything else follows by calculus:

* ``n``       refractive index: how much the material slows light (phase velocity c/n)
* ``n_group`` group index n - lambda dn/dlambda: how much it slows a *pulse*
* ``beta2``   group-velocity dispersion: how fast a pulse spreads in time (ps^2/km)
* ``D``       the same thing in fiber units (ps/(nm km)); D > 0 means blue arrives first
* ``abbe``    Abbe number V_d: low V = strong color fringing in a lens
* ``fresnel_R`` reflection loss at one uncoated surface, normal incidence

For time-stretch imaging (the Jalali lab's STEAM camera and photonic time-stretch ADC), the
dispersion D is what maps wavelength to time: a bandwidth d_lambda spreads over
D * L * d_lambda. ``stretch_ps`` gives that for a length of *bulk* material. Real dispersive
fiber adds waveguide dispersion on top, so fiber datasheet D values differ from bulk silica.

The same table and formulas are in ``dgs/c/materials/materials.h`` (C, gcc) and
``dgs/cuda/materials.cu`` (CUDA, nvcc); ``tests/test_materials.py`` checks all three agree.

Sources (IEEE style)
--------------------
[1] I. H. Malitson, "Interspecimen comparison of the refractive index of fused silica,"
    J. Opt. Soc. Am., vol. 55, no. 10, pp. 1205-1209, 1965.
[2] SCHOTT AG, "Optical glass data sheets: N-BK7."
[3] I. H. Malitson and M. J. Dodge, "Refractive index and birefringence of synthetic
    sapphire," J. Opt. Soc. Am., vol. 62, p. 1405, 1972.
[4] I. H. Malitson, "A redetermination of some optical properties of calcium fluoride,"
    Appl. Opt., vol. 2, no. 11, pp. 1103-1107, 1963.
[5] D. E. Zelmon, D. L. Small, and D. Jundt, "Infrared corrected Sellmeier coefficients for
    congruently grown lithium niobate and 5 mol.% magnesium oxide-doped lithium niobate,"
    J. Opt. Soc. Am. B, vol. 14, no. 12, pp. 3319-3322, 1997.

The coefficients were entered from memory of these papers and are checked in the tests
against well-known index values (n_d of N-BK7 = 1.5168, fused silica n(1550 nm) = 1.4440,
and so on). Check against the original paper before using them for a design.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C0 = 299_792_458.0                 # speed of light, m/s
FRAUNHOFER_UM = {'F': 0.4861327, 'd': 0.5875618, 'C': 0.6562725}   # H-beta, He-d, H-alpha lines


@dataclass(frozen=True)
class Material:
    """One Sellmeier material: n^2 = 1 + sum B_i lam^2 / (lam^2 - C_i), lam in micrometers."""
    name: str
    B: tuple[float, float, float]
    C: tuple[float, float, float]          # um^2
    lam_min_um: float
    lam_max_um: float
    ref: str
    use: str                                # where it shows up in an imaging setup


LIBRARY: dict[str, Material] = {m.name: m for m in [
    Material('fused_silica', (0.6961663, 0.4079426, 0.8974794),
             (0.0684043**2, 0.1162414**2, 9.896161**2), 0.21, 6.7, '[1] Malitson 1965',
             'optical fiber, UV/IR windows, objective lenses'),
    Material('n_bk7', (1.03961212, 0.231792344, 1.01046945),
             (0.00600069867, 0.0200179144, 103.560653), 0.3, 2.5, '[2] SCHOTT N-BK7',
             'the default lens and prism glass'),
    Material('sapphire_o', (1.4313493, 0.65054713, 5.3414021),
             (0.0726631**2, 0.1193242**2, 18.028251**2), 0.2, 5.5, '[3] Malitson & Dodge 1972 (ordinary ray)',
             'scratch-proof windows, Ti:sapphire laser crystals'),
    Material('caf2', (0.5675888, 0.4710914, 3.8484723),
             (0.050263605**2, 0.1003909**2, 34.649040**2), 0.23, 9.7, '[4] Malitson 1963',
             'low-dispersion (apochromatic) lenses, UV and IR optics'),
    Material('linbo3_e', (2.9804, 0.5981, 8.9543), (0.02047, 0.0666, 416.08), 0.4, 5.0,
             '[5] Zelmon 1997 (congruent, extraordinary)', 'electro-optic modulators (z-cut MZM, TE mode)'),
    Material('linbo3_o', (2.6734, 1.2290, 12.614), (0.01764, 0.05914, 474.6), 0.4, 5.0,
             '[5] Zelmon 1997 (congruent, ordinary)', 'electro-optic modulators (ordinary polarization)'),
]}

ALIASES = {'silica': 'fused_silica', 'sio2': 'fused_silica', 'bk7': 'n_bk7', 'sapphire': 'sapphire_o',
           'al2o3': 'sapphire_o', 'calcium_fluoride': 'caf2', 'lithium_niobate': 'linbo3_e', 'ln': 'linbo3_e'}


def get(name: str) -> Material:
    """Look a material up by name or alias, ignoring case, spaces and hyphens ('N-BK7', 'Fused Silica')."""
    key = name.strip().lower().replace('-', '_').replace(' ', '_')
    key = ALIASES.get(key, key)
    if key not in LIBRARY:
        raise KeyError(f"unknown material {name!r}; known: {sorted(LIBRARY)} (aliases: {sorted(ALIASES)})")
    return LIBRARY[key]


def _mat(m) -> Material:
    return m if isinstance(m, Material) else get(m)


def _check_range(m: Material, lam):
    lam = np.asarray(lam, dtype=float)
    if np.any(lam < m.lam_min_um) or np.any(lam > m.lam_max_um):
        raise ValueError(f'{m.name}: Sellmeier fit is valid for {m.lam_min_um}-{m.lam_max_um} um only')
    return lam


def _f_derivs(m: Material, lam):
    """n^2 and its first two derivatives with respect to lambda (um), in closed form."""
    lam2 = lam*lam
    f, f1, f2 = np.ones_like(lam), np.zeros_like(lam), np.zeros_like(lam)
    for B, C in zip(m.B, m.C):
        d = lam2 - C
        f = f + B*lam2/d
        f1 = f1 - 2*B*C*lam/d**2
        f2 = f2 + 2*B*C*(3*lam2 + C)/d**3
    return f, f1, f2


def n(material, lam_um):
    """Refractive index at wavelength lam_um (micrometers)."""
    m = _mat(material)
    f, _, _ = _f_derivs(m, _check_range(m, lam_um))
    return np.sqrt(f)


def dn_dlam(material, lam_um):
    """First and second derivatives of n with respect to wavelength, in 1/um and 1/um^2."""
    m = _mat(material)
    f, f1, f2 = _f_derivs(m, _check_range(m, lam_um))
    nn = np.sqrt(f)
    n1 = f1/(2*nn)
    n2 = (f2 - 2*n1*n1)/(2*nn)
    return n1, n2


def n_group(material, lam_um):
    """Group index n_g = n - lambda dn/dlambda (a pulse travels at c/n_g)."""
    lam = np.asarray(lam_um, dtype=float)
    n1, _ = dn_dlam(material, lam)
    return n(material, lam) - lam*n1


def beta2(material, lam_um):
    """Group-velocity dispersion beta2 = lambda^3 / (2 pi c^2) d^2n/dlambda^2, in ps^2/km."""
    lam = np.asarray(lam_um, dtype=float)
    _, n2 = dn_dlam(material, lam)
    b2_si = (lam*1e-6)**3/(2*np.pi*C0**2)*(n2*1e12)          # s^2/m
    return b2_si*1e24*1e3                                     # ps^2/km


def D(material, lam_um):
    """Dispersion parameter D = -(lambda/c) d^2n/dlambda^2, in ps/(nm km)."""
    lam = np.asarray(lam_um, dtype=float)
    _, n2 = dn_dlam(material, lam)
    d_si = -(lam*1e-6)/C0*(n2*1e12)                           # s/m^2
    return d_si*1e12*1e-9*1e3                                 # ps/(nm km)


def zero_dispersion_um(material, lo_um=None, hi_um=None):
    """Wavelength where beta2 = 0 (bisection); None if beta2 does not change sign in [lo, hi]."""
    m = _mat(material)
    lo = lo_um or m.lam_min_um*1.05
    hi = hi_um or m.lam_max_um*0.95
    b_lo, b_hi = beta2(m, lo), beta2(m, hi)
    if np.sign(b_lo) == np.sign(b_hi):
        return None
    for _ in range(80):
        mid = 0.5*(lo + hi)
        if np.sign(beta2(m, mid)) == np.sign(b_lo):
            lo = mid
        else:
            hi = mid
    return 0.5*(lo + hi)


def abbe(material):
    """Abbe number V_d = (n_d - 1)/(n_F - n_C). Low V: strong chromatic aberration in a lens."""
    m = _mat(material)
    nF, nd, nC = (n(m, FRAUNHOFER_UM[k]) for k in 'FdC')
    return float((nd - 1)/(nF - nC))


def fresnel_R(material, lam_um, n_outside=1.0):
    """Reflectance of one uncoated surface at normal incidence, ((n1 - n2)/(n1 + n2))^2."""
    nn = n(material, lam_um)
    return ((nn - n_outside)/(nn + n_outside))**2


def stretch_ps(material, length_km, lam_center_um, bandwidth_nm):
    """Time spread of a bandwidth after a length of BULK material: |D| L d_lambda, in ps.

    This is the wavelength-to-time map of time-stretch imaging, for material dispersion only;
    a dispersive fiber's own D (material + waveguide) should be used for a real system.
    """
    return float(abs(D(material, lam_center_um))*length_km*bandwidth_nm)


def table(lam_um=1.55):
    """Rows (name, n, n_g, beta2, D, Abbe, R) at one wavelength, for every material valid there."""
    rows = []
    for m in LIBRARY.values():
        if m.lam_min_um <= lam_um <= m.lam_max_um:
            rows.append((m.name, float(n(m, lam_um)), float(n_group(m, lam_um)), float(beta2(m, lam_um)),
                         float(D(m, lam_um)), abbe(m) if m.lam_min_um <= FRAUNHOFER_UM['F'] else float('nan'),
                         float(fresnel_R(m, lam_um))))
    return rows


if __name__ == '__main__':
    print(f"{'material':14s} {'n':>8s} {'n_g':>8s} {'beta2':>10s} {'D':>9s} {'Abbe':>7s} {'R':>7s}   at 1550 nm")
    for r in table(1.55):
        print(f'{r[0]:14s} {r[1]:8.5f} {r[2]:8.5f} {r[3]:10.2f} {r[4]:9.2f} {r[5]:7.2f} {r[6]:7.4f}')
