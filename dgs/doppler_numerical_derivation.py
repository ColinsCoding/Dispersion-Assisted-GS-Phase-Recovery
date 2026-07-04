"""Derives the relativistic Doppler shift NUMERICALLY, from a simulated
emitter/receiver couple, rather than evaluating the closed-form formula
dgs.special_relativity.relativistic_doppler already provides. The two
physical ingredients used are each already-tested primitives from that
module (lorentz_factor for time dilation) -- nothing here assumes the
Doppler formula itself:

  1. TIME DILATION: a source emitting pulses at proper frequency f0 (in its
     own rest frame) ticks, AS SEEN FROM THE LAB, at the dilated rate
     dt_lab = gamma * (1/f0) between successive emission events.
  2. CLASSICAL LIGHT TRAVEL TIME: each pulse, once emitted at some lab
     position, takes |distance|/c to reach a fixed receiver.

Combining just these two (no relativistic Doppler formula assumed) and
measuring the actual spacing between arrival times at the receiver
reproduces the standard closed-form result -- confirming the formula is a
consequence of time dilation + light travel time, not an independent
postulate.
"""

import numpy as np

from dgs.special_relativity import C_SI, lorentz_factor, relativistic_doppler


def simulate_pulse_train_arrival_times(f0_proper, v, n_pulses, x0, x_receiver, c=C_SI):
    """Simulate n_pulses emitted at proper frequency f0_proper by a source
    moving at constant velocity v along x, starting at position x0 at lab
    time t=0. Returns (t_emit, x_emit, t_arrival) at a receiver fixed at
    x_receiver, derived from time dilation (for the emission schedule) plus
    ordinary light travel time (for arrival) -- no Doppler formula used."""
    if abs(v) >= c:
        raise ValueError(f"|v| must be < c; got v={v:.3e}")
    if f0_proper <= 0:
        raise ValueError("f0_proper must be positive")
    if n_pulses < 2:
        raise ValueError("need at least 2 pulses to measure a period")

    gamma = lorentz_factor(v, c)["gamma"]
    tau0 = 1.0 / f0_proper
    dt_lab_emit = gamma * tau0              # time dilation: lab sees the source tick slower

    n = np.arange(n_pulses)
    t_emit = n * dt_lab_emit
    x_emit = x0 + v * t_emit

    travel_dist = np.abs(x_receiver - x_emit)
    t_arrival = t_emit + travel_dist / c
    return t_emit, x_emit, t_arrival


def numerical_doppler_frequency(f0_proper, v, n_pulses, x0, x_receiver, c=C_SI):
    """Observed frequency measured purely from the spacing between
    consecutive pulse-arrival times at the receiver -- the numerical
    derivation's actual result, with no closed-form Doppler formula
    anywhere in this function."""
    _, _, t_arrival = simulate_pulse_train_arrival_times(
        f0_proper, v, n_pulses, x0, x_receiver, c)
    periods_obs = np.diff(t_arrival)
    period_obs = float(periods_obs.mean())
    return {
        "f_obs": 1.0 / period_obs,
        "period_obs": period_obs,
        "period_std": float(periods_obs.std()),   # should be ~0: constant v -> constant observed period
        "t_arrival": t_arrival,
    }


def compare_numerical_vs_analytic_doppler(f0_proper, v, approaching, n_pulses=20, c=C_SI):
    """Set up an emitter/receiver couple geometrically so the emitter is
    either approaching or receding from the receiver for the ENTIRE
    simulated pulse train, run the numerical derivation, and compare its
    result against dgs.special_relativity.relativistic_doppler's
    closed-form prediction for the same (f0, v, approaching)."""
    if v <= 0:
        raise ValueError("v must be positive (this convention moves the source in +x; "
                          "use approaching=True/False to choose the receiver's side)")

    gamma = lorentz_factor(v, c)["gamma"]
    dt_lab_emit = gamma / f0_proper
    total_lab_time = n_pulses * dt_lab_emit
    # margin just needs to keep the receiver beyond every emission position;
    # note a buffer many orders of magnitude bigger than this (e.g. adding a
    # full light-second) would blow up t_arrival's absolute scale relative to
    # the ~dt_lab_emit-sized period differences being measured, silently
    # losing float64 precision in the subtraction -- keep it proportional.
    buffer = abs(v) * total_lab_time * 2
    x0 = 0.0
    x_receiver = x0 + buffer if approaching else x0 - buffer

    numeric = numerical_doppler_frequency(f0_proper, v, n_pulses, x0, x_receiver, c)
    analytic = relativistic_doppler(f0_proper, v, c, approaching=approaching)
    return numeric, analytic


if __name__ == "__main__":
    f0 = 1.0e14   # Hz, an arbitrary optical-ish reference frequency
    v = 0.6 * C_SI

    for approaching in (True, False):
        numeric, analytic = compare_numerical_vs_analytic_doppler(f0, v, approaching, n_pulses=20)
        label = "approaching" if approaching else "receding"
        rel_err = abs(numeric["f_obs"] - analytic["f_obs"]) / analytic["f_obs"]
        print(f"[{label}] v={v/C_SI:.1f}c")
        print(f"  numerical (simulated pulse train):  f_obs = {numeric['f_obs']:.6e} Hz  "
              f"(period std = {numeric['period_std']:.3e}, should be ~0)")
        print(f"  analytic (closed-form formula):     f_obs = {analytic['f_obs']:.6e} Hz")
        print(f"  relative error: {rel_err:.2e}\n")
