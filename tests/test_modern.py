"""Test Compton scattering and bremsstrahlung against known values + conservation."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from griffiths import modern as mp

# 1. Compton wavelength ~ 2.426 pm
assert abs(mp.compton_wavelength() - 2.42631e-12) < 1e-16

# 2. wavelength shift: 0 forward, lambda_C at 90 deg, 2 lambda_C at backscatter
lamC = mp.compton_wavelength()
assert abs(mp.compton_shift(0.0)) < 1e-20
assert abs(mp.compton_shift(np.pi/2) - lamC) < 1e-18
assert abs(mp.compton_shift(np.pi) - 2*lamC) < 1e-18           # maximum at theta=pi
# shift is independent of incident wavelength
assert abs(mp.compton_scattered_wavelength(1e-10, np.pi/2) - (1e-10 + lamC)) < 1e-18

# 3. scattered energy: unchanged forward, minimum (Compton edge) at backscatter
E = 100e3 * 1.602176634e-19                                    # 100 keV photon
assert abs(mp.compton_scattered_energy(E, 0.0) - E) < 1e-25    # no shift forward
assert mp.compton_scattered_energy(E, np.pi) < E              # loses energy
# monotone: more angle -> less energy
angs = np.linspace(0, np.pi, 20)
Eout = mp.compton_scattered_energy(E, angs)
assert np.all(np.diff(Eout) < 0)

# 4. the formula satisfies relativistic energy-momentum conservation
c = 299792458.0; mec2 = mp.electron_rest_energy()
for theta in (0.5, 1.5, np.pi):
    Ep = mp.compton_scattered_energy(E, theta)
    Ee = E + mec2 - Ep                                         # electron energy after
    pxc = E - Ep*np.cos(theta)                                 # (p_e c) components
    pyc = -Ep*np.sin(theta)
    pe2c2 = pxc**2 + pyc**2
    assert abs(Ee**2 - (pe2c2 + mec2**2)) / Ee**2 < 1e-12      # E_e^2 = (p c)^2 + (m c^2)^2

# 5. wavelength <-> energy consistency: E' = hc/lambda'  matches the energy formula
h, lam_in = 6.62607015e-34, 1e-11
E_in = h*c/lam_in
lam_out = mp.compton_scattered_wavelength(lam_in, 1.2)
assert abs(h*c/lam_out - mp.compton_scattered_energy(E_in, 1.2)) / (h*c/lam_out) < 1e-10

# 6. bremsstrahlung: Duane-Hunt cutoff and max photon energy
assert abs(mp.bremsstrahlung_cutoff_wavelength(50e3) - 2.4797e-11) < 1e-14   # ~24.8 pm
assert abs(mp.bremsstrahlung_max_photon_energy(50e3)/1.602176634e-19/1e3 - 50.0) < 1e-6
# higher voltage -> shorter cutoff (1/V)
assert abs(mp.bremsstrahlung_cutoff_wavelength(100e3) - mp.bremsstrahlung_cutoff_wavelength(50e3)/2) < 1e-16

# 7. spectrum: zero below cutoff, positive above, with a peak
lam = np.linspace(10e-12, 200e-12, 4000)
I = mp.bremsstrahlung_spectrum(lam, 50e3)
lmin = mp.bremsstrahlung_cutoff_wavelength(50e3)
assert np.all(I[lam < lmin] == 0) and np.any(I > 0)
assert lam[np.argmax(I)] > lmin                               # peak is above the cutoff

print(f"TEST PASS  (Compton lambda_C={lamC*1e12:.4f} pm; shift 0/lamC/2lamC; energy "
      f"conserves momentum (E_e^2=(pc)^2+(mc^2)^2); Duane-Hunt cutoff {lmin*1e12:.2f} pm "
      f"= hc/eV; spectrum edge + peak)")
