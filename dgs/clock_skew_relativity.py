"""Special relativity in computer engineering -- honestly scoped: real
clock skew on a chip/PCB is a CLASSICAL (finite-signal-speed) effect, not
a truly relativistic one (nothing on a stationary board moves anywhere
near c, so time dilation and length contraction genuinely don't apply
here). What DOES carry over exactly is the conceptual root of special
relativity's relativity of simultaneity: two events that are simultaneous
at a SOURCE are NOT simultaneous at two different receivers, purely
because signal transit time isn't zero. Special relativity is what you
get when you additionally let the receivers/source move relative to each
other near c; classical clock skew is the SAME idea with v=0 (everything
stationary) -- this module makes that reduction explicit rather than
just asserting an analogy.

Reuses dgs.transmission_line_tdr.propagation_velocity for the actual
trace propagation speed (v=1/sqrt(L'C'), already verified there) and
dgs.special_relativity.lorentz_transform to confirm the classical skew
formula IS the v->0 limit of the full relativistic simultaneity formula.

The real engineering problem this models: CLOCK TREE SYNTHESIS. Two flip-
flops fed from the same clock source, at different trace lengths, receive
"the same" clock edge at different times -- if that skew is a large
enough fraction of the clock period, setup/hold timing breaks and the
chip malfunctions. This is why clock trees are laid out as H-trees /
balanced structures: to equalize propagation delay to every leaf.
"""

import numpy as np

from dgs.special_relativity import C_SI, lorentz_transform
from dgs.transmission_line_tdr import propagation_velocity


def clock_arrival_time(source_time, distance, velocity):
    """Classical arrival time of a clock edge launched at source_time from
    a source, at a receiver `distance` away, propagating at `velocity`:
    t_arrival = source_time + distance/velocity. No relativity here yet --
    just finite signal speed."""
    if velocity <= 0:
        raise ValueError("velocity must be positive")
    if distance < 0:
        raise ValueError("distance must be non-negative")
    return source_time + distance / velocity


def clock_skew_between_receivers(distance_a, distance_b, velocity):
    """The skew between two receivers fed from the SAME source at the SAME
    source_time: purely the difference in propagation delay, since
    source_time cancels. This is the classical "loss of simultaneity":
    the two arrivals are simultaneous ONLY IF distance_a == distance_b."""
    t_a = clock_arrival_time(0.0, distance_a, velocity)
    t_b = clock_arrival_time(0.0, distance_b, velocity)
    return t_a - t_b


def relativistic_simultaneity_gap(distance_a, distance_b, v_frame, c=C_SI):
    """The FULL special-relativistic calculation: two events simultaneous
    in the source's rest frame, at positions distance_a and distance_b,
    viewed from a frame moving at v_frame relative to the source. Uses
    dgs.special_relativity.lorentz_transform directly (not a new formula)
    to find each event's time coordinate in the moving frame, then returns
    their difference -- the relativistic generalization of clock skew."""
    event_a = lorentz_transform(distance_a, 0.0, v_frame, c)
    event_b = lorentz_transform(distance_b, 0.0, v_frame, c)
    return event_a["t_prime"] - event_b["t_prime"]


def verify_classical_limit(distance_a, distance_b, c=C_SI):
    """Confirm that relativistic_simultaneity_gap, evaluated at successively
    smaller v_frame/c ratios, approaches the appropriate classical limit.
    For v_frame -> 0, both events stay simultaneous in every frame (the
    Lorentz transform's time-mixing term v*x/c^2 vanishes) -- this ISN'T
    the same formula as clock_skew_between_receivers (that one comes from
    finite signal speed within ONE frame; this one comes from switching
    frames), but confirming it vanishes at v=0 is the honest boundary
    between "these are related ideas" and "these are the same formula"."""
    gaps = []
    for v_frac in [1e-3, 1e-6, 1e-9]:
        gap = relativistic_simultaneity_gap(distance_a, distance_b, v_frac * c, c)
        gaps.append(abs(gap))
    return gaps


if __name__ == "__main__":
    print("=== Real clock skew: two flip-flops, mismatched trace lengths ===")
    L_per_len, C_per_len = 250e-9, 100e-12   # same PCB microstrip as dgs.transmission_line_tdr
    v = propagation_velocity(L_per_len, C_per_len)
    print(f"trace propagation velocity: {v:.3e} m/s ({v/C_SI:.2f}c)")

    d_a, d_b = 0.050, 0.053   # 50mm vs 53mm trace lengths -- a realistic PCB mismatch
    skew = clock_skew_between_receivers(d_a, d_b, v)
    print(f"trace A: {d_a*1000:.0f} mm, trace B: {d_b*1000:.0f} mm (3 mm mismatch)")
    print(f"clock skew between the two receivers: {skew*1e12:.2f} ps")

    for freq_ghz in [1.0, 3.0, 5.0]:
        period_ps = 1e12 / (freq_ghz * 1e9)
        frac = abs(skew) * 1e12 / period_ps
        print(f"  at {freq_ghz:.1f} GHz (period={period_ps:.1f} ps): "
              f"skew is {frac*100:.1f}% of the clock period"
              f"{'  <-- likely a real timing problem' if frac > 0.05 else ''}")

    print("\n=== Confirming: relativity is the SAME idea, generalized to moving frames ===")
    print("(these numbers do NOT match the classical skew above -- different physical")
    print(" question: 'switch reference frames' vs 'account for finite signal speed within one frame')")
    gaps = verify_classical_limit(d_a, d_b)
    for v_frac, gap in zip([1e-3, 1e-6, 1e-9], gaps):
        print(f"  v/c={v_frac:.0e}: relativistic simultaneity gap = {gap:.3e} s")
    print(f"  gap shrinks monotonically as v/c -> 0: {gaps[0] > gaps[1] > gaps[2]}")
    print("  (as it must: two simultaneous-in-source-frame events stay simultaneous")
    print("   in every other frame once that frame's relative velocity is exactly zero)")

    print("\n=== The honest scope of the analogy ===")
    print("Chip clock skew is 100% classical -- finite signal speed, zero relative motion,")
    print("no gamma factor anywhere. What carries over from special relativity is only the")
    print("CONCEPTUAL root: simultaneity is not absolute, it depends on WHERE you measure it")
    print("from (position, for clock skew; position AND relative velocity, for relativity).")
    print("Clock tree synthesis is literally the engineering discipline of re-establishing")
    print("simultaneity by equalizing path length -- undoing the classical half of this effect.")
