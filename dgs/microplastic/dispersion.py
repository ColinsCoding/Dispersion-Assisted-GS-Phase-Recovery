"""Month 2: real wavelength-dependent complex refractive index -- dispersion,
absorption, and material transfer functions -- built on the causal Lorentz-
oscillator machinery already in dgs/causality.py.

Month 1 (physics.py) treated the refractive index as a single fixed number.
Real materials don't work that way: n(omega) varies with frequency
(dispersion), and by the Kramers-Kronig relations already proved in
dgs/causality.py, any medium with dispersion MUST also absorb somewhere --
Re(chi) and Im(chi) are Hilbert-transform pairs, not independent choices.
This module puts a single UV resonance on each polymer from materials.py,
calibrates its strength so the model matches that polymer's tabulated
visible-band index, and builds the resulting material transfer function
H(omega) = exp(i*n_tilde(omega)*omega*L/c) for a real slab of material.

Calibrated single-Lorentz-oscillator model, not measured multi-wavelength
dispersion data -- see materials.py's own uncertainty note. Swapping in real
Sellmeier-fit coefficients per polymer would replace only the calibration
step; the causal structure (KK-consistent, dispersion implies absorption)
is the part that's exactly right regardless.

NumPy + SciPy. Education / forward-model, not measured dispersion data.
"""

import numpy as np
from scipy.optimize import brentq

from dgs import causality as ca
from dgs.microplastic import physics as phy
from dgs.microplastic import materials as mat

C = phy.C
LAMBDA_D = 589.3e-9          # sodium D-line, the wavelength materials.py's n is tabulated at
OMEGA_D = 2 * np.pi * C / LAMBDA_D


def n_tilde_lorentz(omega, omega0, gamma, strength):
    """Complex index from a single Lorentz oscillator: n~ = sqrt(1 + chi(omega)),
    the nonmagnetic Maxwell relation (month 1's eps_r = n~^2, here eps_r = 1+chi
    for a dilute-oscillator susceptibility) applied to dgs.causality's causal
    chi(omega). Re(n~) is the dispersion curve; Im(n~) is absorption -- both
    come from the SAME chi, so they can't be set independently."""
    chi = ca.lorentz_susceptibility(omega, omega0, gamma, strength)
    return np.sqrt(1 + chi)


def calibrate_strength(polymer, omega0, gamma, omega_ref=OMEGA_D,
                        bracket=(1e20, 1e34)):
    """Solve for the oscillator strength so Re(n_tilde(omega_ref)) matches
    `polymer`'s tabulated refractive index from materials.py at omega_ref
    (default: the sodium D-line, matching how that table was tabulated).
    Re(n~) grows monotonically with strength here, so a bracketed root-find
    (brentq) is reliable rather than an unconstrained solver."""
    n_target = mat.refractive_index(polymer)

    def residual(strength):
        return n_tilde_lorentz(omega_ref, omega0, gamma, strength).real - n_target

    lo, hi = bracket
    if residual(lo) > 0 or residual(hi) < 0:
        raise ValueError(f"target n={n_target} not bracketed by strength in {bracket}; widen it")
    return brentq(residual, lo, hi)


def polymer_dispersion_model(polymer, lambda0_uv_nm=200.0, gamma_frac=0.002):
    """Build a calibrated Lorentz-oscillator dispersion model for a polymer
    from materials.py. lambda0_uv_nm places the resonance (typical polymer
    electronic transitions sit ~150-250 nm, in the UV, which is exactly why
    bulk commodity plastics look transparent and non-absorbing across the
    visible band). gamma_frac sets the damping as a fraction of omega0; since
    Re(chi) and Im(chi) share the same strength for fixed omega0/gamma, a
    broad gamma_frac (e.g. 0.05) reproduces the target n but drags kappa up
    with it to the point that a bulk sample would be opaque within microns
    -- clearly wrong for materials known to be visibly transparent for
    millimeters to centimeters. gamma_frac=0.002 (a narrower UV line) keeps
    kappa in the range where a mm-scale slab (§6-7 of the month-2 notebook,
    or a single microplastic particle's own path length) shows real but
    partial absorption instead of either nothing or total extinction.
    Returns (omega0, gamma, strength) ready for n_tilde_lorentz."""
    omega0 = 2 * np.pi * C / (lambda0_uv_nm * 1e-9)
    gamma = gamma_frac * omega0
    strength = calibrate_strength(polymer, omega0, gamma)
    return omega0, gamma, strength


def transfer_function(omega, n_tilde_fn, L):
    """H(omega) = exp(i n~(omega) omega L / c): the material transfer function
    for a slab of thickness L, using a wavelength-dependent n~ instead of
    month 1's single fixed value. Structurally identical to the
    H(nu)=exp(i*pi*D*nu^2) kernel this whole repo's phase-retrieval work
    inverts -- but built from a real causal medium, not a pure quadratic-
    phase toy dispersion."""
    n_tilde = n_tilde_fn(omega)
    return np.exp(1j * n_tilde * omega * L / C)


def apply_slab(t, Et, n_tilde_fn, L):
    """Propagate a time-domain pulse E(t) through a slab of thickness L with
    dispersion/absorption model n_tilde_fn(omega), via FFT -> multiply by
    transfer_function -> IFFT. Returns (t_out, E_out) on the same grid physics.py
    uses (inverse_fourier_transform's t0 bookkeeping)."""
    omega, Ef = phy.fourier_transform(t, Et)
    H = transfer_function(omega, n_tilde_fn, L)
    t_out, Et_out = phy.inverse_fourier_transform(omega, Ef * H, t0=t[0])
    return t_out, Et_out


if __name__ == "__main__":
    omega0, gamma, strength = polymer_dispersion_model("PET")
    n_at_D = n_tilde_lorentz(OMEGA_D, omega0, gamma, strength)
    print(f"PET calibrated model: omega0={omega0:.3e}, gamma={gamma:.3e}, strength={strength:.3e}")
    print(f"n_tilde(sodium D)     = {n_at_D}  (target n={mat.refractive_index('PET')})")

    lam_blue = 450e-9
    omega_blue = 2 * np.pi * C / lam_blue
    n_blue = n_tilde_lorentz(omega_blue, omega0, gamma, strength)
    print(f"n_tilde(450 nm, blue) = {n_blue}  (normal dispersion: n rises toward the UV resonance)")
