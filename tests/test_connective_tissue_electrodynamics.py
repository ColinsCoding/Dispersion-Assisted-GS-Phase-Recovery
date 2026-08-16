"""Test dgs/connective_tissue_electrodynamics.py's two regimes of Maxwell's
equations in matter applied to connective tissue: optical collagen
form-birefringence (Wiener mixing) and electrical Cole-Cole dielectric
dispersion, tied together by a direct time-domain causality check.
Locks in a real sign-convention bug found this session: a Lorentz-oscillator
susceptibility written with -i*gamma*omega (a commonly seen but, under
numpy's FFT convention, WRONG-half-plane form here) fails the causality
check; +i*gamma*omega is the form consistent with this module's Cole-Cole
convention."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.connective_tissue_electrodynamics import (
    complex_refractive_index, absorption_coefficient,
    wiener_parallel_permittivity, wiener_perpendicular_permittivity,
    form_birefringence, verify_form_birefringence_limits,
    cole_cole_permittivity, causality_fraction_energy_at_negative_time,
    fftfreq_omega_grid,
)

# 1. Complex refractive index: lossless case must give purely real n
n_lossless = complex_refractive_index(2.25 + 0.0j)
assert abs(n_lossless.imag) < 1e-12 and abs(n_lossless.real - 1.5) < 1e-12

# 2. Absorption coefficient must be positive for a lossy (Im(eps)>0) medium
n_lossy = complex_refractive_index(2.25 + 0.01j)
alpha_abs = absorption_coefficient(n_lossy, omega=2 * np.pi * 3e8 / 800e-9)
assert alpha_abs > 0, "a lossy medium (Im(eps)>0) must give positive absorption"

# 3. Wiener perpendicular permittivity must never exceed parallel (classical
#    Wiener bound ordering -- the mechanism that makes the medium birefringent)
for f in [0.1, 0.3, 0.5, 0.7, 0.9]:
    eps_par = wiener_parallel_permittivity(f, eps_fibril=1.47**2, eps_ground=1.35**2)
    eps_perp = wiener_perpendicular_permittivity(f, eps_fibril=1.47**2, eps_ground=1.35**2)
    assert eps_perp <= eps_par, f"f={f}: Wiener perpendicular must not exceed parallel"

# 4. Form birefringence vanishes at f=0 and f=1, positive in between
assert verify_form_birefringence_limits()
dn_mid = form_birefringence(0.5)
assert dn_mid > 0, "form birefringence should be positive for an intermediate fibril fraction"

# 5. Bounds: fibril_fraction outside [0,1] must raise
for bad_f in [-0.1, 1.1]:
    try:
        form_birefringence(bad_f)
        raise AssertionError(f"expected ValueError for fibril_fraction={bad_f}")
    except ValueError:
        pass

# 6. Cole-Cole reduces to Debye at alpha=0 and matches the closed-form value at omega=0
eps_at_zero = cole_cole_permittivity(np.array([0.0]), eps_static=80.0, eps_inf=4.0, tau=1.0, alpha=0.0)
assert abs(eps_at_zero[0] - 80.0) < 1e-9, "Cole-Cole/Debye must equal eps_static at omega=0"

# 7. Cole-Cole bounds: tau<=0, alpha out of [0,1), eps_static<=eps_inf must all raise
for bad_kwargs in [
    dict(omega=np.array([1.0]), eps_static=80.0, eps_inf=4.0, tau=0.0, alpha=0.0),
    dict(omega=np.array([1.0]), eps_static=80.0, eps_inf=4.0, tau=1.0, alpha=1.0),
    dict(omega=np.array([1.0]), eps_static=4.0, eps_inf=80.0, tau=1.0, alpha=0.0),
]:
    try:
        cole_cole_permittivity(**bad_kwargs)
        raise AssertionError(f"expected ValueError for {bad_kwargs}")
    except ValueError:
        pass

# 8. Causality check: Cole-Cole/Debye susceptibility must be overwhelmingly
#    concentrated at t>=0 (small residual is expected numerical truncation)
omega_grid = fftfreq_omega_grid(n=8192, domega=0.02)
for alpha in [0.0, 0.2, 0.4]:
    eps = cole_cole_permittivity(omega_grid, eps_static=80.0, eps_inf=4.0, tau=1.0, alpha=alpha)
    frac_neg = causality_fraction_energy_at_negative_time(eps - 4.0)
    assert frac_neg < 0.05, f"alpha={alpha}: Cole-Cole should be clearly causal, got frac={frac_neg:.4f}"

# 9. The sign-convention regression: a Lorentz-oscillator susceptibility with
#    the correct (+i*gamma*omega) sign must pass causality; the other common
#    sign (-i*gamma*omega) must clearly fail it -- both checked, not assumed
omega_optical = fftfreq_omega_grid(n=8192, domega=1e12)
omega0, gamma = 3e15, 5e14
chi_correct = 1.0 / (omega0**2 - omega_optical**2 + 1j * gamma * omega_optical)
chi_wrong = 1.0 / (omega0**2 - omega_optical**2 - 1j * gamma * omega_optical)
frac_correct = causality_fraction_energy_at_negative_time(chi_correct)
frac_wrong = causality_fraction_energy_at_negative_time(chi_wrong)
assert frac_correct < 0.01, f"correct-sign Lorentz susceptibility should be causal, got {frac_correct:.4f}"
assert frac_wrong > 0.9, f"wrong-sign Lorentz susceptibility should clearly fail causality, got {frac_wrong:.4f}"

# 10. causality_fraction_energy_at_negative_time bounds: too-short array, all-zero input
try:
    causality_fraction_energy_at_negative_time(np.zeros(4))
    raise AssertionError("expected ValueError for n<8")
except ValueError:
    pass
try:
    causality_fraction_energy_at_negative_time(np.zeros(16))
    raise AssertionError("expected ValueError for identically-zero input")
except ValueError:
    pass

print("all dgs.connective_tissue_electrodynamics tests passed")
