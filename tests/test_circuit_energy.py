"""Test dgs.circuit_energy: 1/2 C v^2 and 1/2 L i^2 tanks, RC discharge
dissipates EXACTLY the stored energy (numeric == closed form == 1/2 C v0^2),
lossless LC conserves E_C+E_L while it sloshes, a real RLC natural response
conserves E_C+E_L+E_R to O(h^2), the resistor's heat only ever increases,
and the damped ring frequency sits just below f0 and vanishes past critical
damping."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import circuit_energy as ce
from dgs import spice

R, C, v0, L = 1e3, 1e-6, 5.0, 1e-3

# 1. the energy-tank formulas
assert np.isclose(ce.capacitor_energy(C, v0), 0.5 * C * v0**2)
assert np.isclose(ce.inductor_energy(L, 0.2), 0.5 * L * 0.2**2)

# 2. RC discharge: the numerically integrated heat equals the closed form
#    AND both equal the full stored energy 1/2 C v0^2 (energy is conserved)
t = np.linspace(0, 8 * R * C, 6000)          # ~8 tau: essentially fully discharged
v, i = ce.rc_discharge(R, C, v0, t)
assert np.isclose(v[0], v0) and np.isclose(i[0], v0 / R)
E_stored = 0.5 * C * v0**2
E_num = ce.dissipated_energy(i, R, t)[-1]
E_ana = ce.rc_energy_released(R, C, v0, t)[-1]
assert np.isclose(E_num, E_ana, rtol=1e-4)
assert np.isclose(E_ana, E_stored, rtol=1e-3)      # all the joules end up as heat
# and the released-energy formula is R-independent: 10x R, same total
E_ana_bigR = ce.rc_energy_released(10 * R, C, v0, t * 10)[-1]
assert np.isclose(E_ana_bigR, E_stored, rtol=1e-3)

# 3. dissipated energy is monotonically non-decreasing (a resistor only heats)
E_R_cum = ce.dissipated_energy(i, R, t)
assert np.all(np.diff(E_R_cum) >= -1e-18)

# 4. lossless LC (R=0): energy sloshes between the fields but total is fixed
t2 = np.linspace(0, 2e-3, 20000)
vc, iL = spice.rlc_step_response(0.0, L, C, t2, V=0.0, vc0=v0, il0=0.0)
audit = ce.energy_audit(0.0, L, C, t2, vc, iL)
assert audit["max_drift"] < 1e-3                    # conserved to O(h^2)
# it really does slosh: both tanks fill and empty
assert audit["E_C"].min() < 0.1 * audit["E0"]       # cap empties...
assert audit["E_L"].max() > 0.8 * audit["E0"]       # ...into the coil

# 5. real RLC natural response: E_C + E_L + E_R stays flat = E_0
vc, iL = spice.rlc_step_response(20.0, L, C, t2, V=0.0, vc0=v0, il0=0.0)
audit = ce.energy_audit(20.0, L, C, t2, vc, iL)
assert np.isclose(audit["E0"], E_stored)
assert audit["max_drift"] < 1e-3
# energy actually leaves the fields: final stored energy is a fraction of E0
assert (audit["E_C"][-1] + audit["E_L"][-1]) < 0.1 * audit["E0"]
assert audit["E_R"][-1] > 0.5 * audit["E0"]         # most of it is now heat

# 6. damped ring frequency: below the undamped f0, and zero once over-damped
f0 = spice.resonant_frequency(L, C)
fd = ce.damped_ring_frequency(20.0, L, C)
assert 0 < fd < f0
Rc = spice.critical_resistance(L, C)                # 2 sqrt(L/C)
assert ce.damped_ring_frequency(Rc, L, C) == 0.0    # critically damped: no ring
assert ce.damped_ring_frequency(2 * Rc, L, C) == 0.0  # over-damped: no ring
# undamped limit R->0 recovers f0
assert np.isclose(ce.damped_ring_frequency(0.0, L, C), f0)

# 7. the FFT of the decaying ring peaks at f_d (ties to the DFT work)
dt = t2[1] - t2[0]
nfft = 8 * len(vc)                                   # zero-pad for fine bins
spec = np.abs(np.fft.rfft(vc - vc.mean(), n=nfft))
freqs = np.fft.rfftfreq(nfft, dt)
f_peak = freqs[np.argmax(spec)]
assert abs(f_peak - fd) < 0.03 * fd                 # peak lands on f_d...
assert abs(f_peak - fd) < abs(f_peak - f0)          # ...nearer f_d than the undamped f0

# 8. kwarg bounds
for bad in (lambda: ce.capacitor_energy(0, 1),
            lambda: ce.inductor_energy(-1, 1),
            lambda: ce.dissipated_energy(iL, -1, t2),
            lambda: ce.rc_discharge(0, C, v0, t),
            lambda: ce.damped_ring_frequency(20.0, 0, C)):
    try:
        bad()
        assert False, "expected ValueError"
    except ValueError:
        pass

print("test_circuit_energy: all checks passed")
