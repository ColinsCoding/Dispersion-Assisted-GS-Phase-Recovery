"""Test griffiths/quantum.py's split_step_driven: matches split_step at zero
drive, conserves norm, and actually transfers population when resonantly
driven between two bound states of a boundary-value problem."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from griffiths import quantum as qm

# 1. zero drive must exactly reproduce split_step
x = np.linspace(-10, 10, 400)
V0 = 0.5 * x**2   # harmonic well
psi0 = qm.gaussian_packet(x, x0=1.0, k0=0.0, sigma=1.0)
dt, steps = 0.01, 50

frames_plain = qm.split_step(psi0, x, V0, dt, steps)
frames_driven_zero = qm.split_step_driven(psi0, x, V0, lambda t: 0 * x, dt, steps)
assert np.allclose(frames_plain, frames_driven_zero, atol=1e-10)

# 2. norm is conserved under the driven propagator (unitarity), even with a
# nonzero, time-dependent drive
drive = lambda t: 0.3 * np.cos(2.0 * t) * x
frames = qm.split_step_driven(psi0, x, V0, drive, dt, steps)
dx = x[1] - x[0]
norms = np.sum(np.abs(frames) ** 2, axis=1) * dx
assert np.allclose(norms, 1.0, atol=1e-6)

# 3. resonant driving between the two lowest bound states of a FINITE square
# well (a real boundary-value problem via solve_tise) transfers population
# from the ground state toward the first excited state. A finite well is
# used rather than the infinite well or a harmonic well deliberately: the
# infinite well's hard Dirichlet cutoff conflicts with split_step_driven's
# FFT kinetic step (which assumes periodic boundaries), producing spurious
# wraparound leakage independent of drive strength -- caught by running
# this at several field strengths and noticing the "leakage" didn't scale
# down as E0 did (a numerical artifact, not physics). A harmonic well
# avoids that artifact (its eigenstates decay smoothly to the grid edges)
# but its EQUALLY SPACED levels mean a drive resonant with the 0-1
# transition is simultaneously resonant with 1-2, 2-3, ... -- population
# climbs the ladder instead of Rabi-flopping between two levels, which is
# exactly why real qubits are built from ANHARMONIC wells. A finite square
# well has genuinely non-uniform spacing and smooth (non-hard-wall) decay,
# so it is both numerically consistent with the propagator and a legitimate
# two-level qubit approximation.
a_half, V0_well = 3.0, 8.0
x_well = np.linspace(-12, 12, 900)
V_well = np.where(np.abs(x_well) < a_half, 0.0, V0_well)
E, psi_states = qm.solve_tise(x_well, V_well, n_states=3)
psi1, psi2 = psi_states[:, 0], psi_states[:, 1]
dx_well = x_well[1] - x_well[0]

omega_21 = E[1] - E[0]
omega_32 = E[2] - E[1]
assert abs(omega_21 - omega_32) > 0.1 * omega_21    # genuinely anharmonic spacing

E0_field = 0.05
drive_well = lambda t: -E0_field * np.cos(omega_21 * t) * x_well

steps_well = 6000
dt_well = 0.01
frames_well = qm.split_step_driven(psi1.astype(complex), x_well, V_well, drive_well,
                                    dt_well, steps_well, store_every=30)

# excited-state population |<psi2|psi(t)>|^2 should rise well above its
# t=0 value (0, since we start exactly in the ground state)
overlaps = frames_well @ (np.conj(psi2) * dx_well)
P_e = np.abs(overlaps) ** 2
assert P_e[0] < 1e-6                          # starts in the ground state
assert P_e.max() > 0.9                        # resonant drive nearly fully populates state 2

# norm conservation holds throughout this real drive too
norms = np.sum(np.abs(frames_well) ** 2, axis=1) * dx_well
assert np.allclose(norms, 1.0, atol=1e-4)

print("TEST PASS  (split_step_driven matches split_step at zero drive; norm conserved "
      "under a time-dependent drive; resonant driving of an anharmonic finite-well "
      "boundary-value problem drives near-complete Rabi population transfer)")
