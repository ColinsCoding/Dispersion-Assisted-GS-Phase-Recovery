"""Test dgs.microfluidic_circuits: Hagen-Poiseuille resistance and its r^4 law, the
fluidic Ohm's law (Q=dP/R), series/parallel channels behaving like resistors, a
parallel flow divider, laminar Re and high Pe for a typical chip, and a gcc C
cross-check of the resistance and flow."""
import sys, pathlib, math, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs import microfluidic_circuits as mf

mu = mf.MU_WATER
r, L = 50e-6, 0.01

# 1. Hagen-Poiseuille resistance and the r^4 law
R = mf.hydraulic_resistance_circular(r, L)
assert math.isclose(R, 8 * mu * L / (math.pi * r**4))
assert math.isclose(mf.hydraulic_resistance_circular(r/2, L) / R, 16.0)   # r^4: half r -> 16x
assert math.isclose(mf.hydraulic_resistance_circular(r, 2*L) / R, 2.0)    # linear in length

# 2. rectangular channel: shallower (small h) is much more resistive, and h/w symmetric
Rrect = mf.hydraulic_resistance_rectangular(100e-6, 20e-6, L)
assert Rrect > 0
assert mf.hydraulic_resistance_rectangular(100e-6, 10e-6, L) > Rrect       # h^3: shallower worse
assert math.isclose(mf.hydraulic_resistance_rectangular(100e-6, 20e-6, L),
                    mf.hydraulic_resistance_rectangular(20e-6, 100e-6, L))  # w,h swap invariant

# 3. fluidic Ohm's law round trip, and Q = pi r^4 dP / (8 mu L)
dP = 1000.0
Q = mf.flow_rate(dP, R)
assert math.isclose(mf.pressure_drop(Q, R), dP)                            # round trip
assert math.isclose(Q, math.pi * r**4 * dP / (8 * mu * L))                # Hagen-Poiseuille

# 4. series adds resistance, parallel adds conductance (like resistors)
R1, R2 = 2e12, 6e12
assert math.isclose(mf.series_resistance(R1, R2), 8e12)
assert math.isclose(mf.parallel_resistance(R1, R2), 1 / (1/R1 + 1/R2))     # = 1.5e12
assert mf.parallel_resistance(R1, R2) < min(R1, R2)                        # parallel is smaller

# 5. parallel flow divider: a wider (16x lower R) channel hogs 16/17 of the flow
Rw = mf.hydraulic_resistance_circular(2*r, L)                              # 2x radius -> R/16
assert math.isclose(R / Rw, 16.0)
flows, total = mf.parallel_channel_flows(dP, [R, Rw])
assert math.isclose(total, dP / mf.parallel_resistance(R, Rw))            # total = dP/R_par
assert math.isclose(flows[1] / total, 16 / 17, rel_tol=1e-6)             # wide bore hogs it
assert math.isclose(flows[0] / total, 1 / 17, rel_tol=1e-6)

# 6. flow regime: laminar Re, and high Pe (diffusion-limited mixing)
v_slow = 1e-3                                                              # 1 mm/s, typical
assert mf.reynolds_number(v_slow, 2*r) < 1.0                              # deeply laminar
assert mf.reynolds_number(30e-3, 2*r) < 2000                             # still laminar when fast
assert mf.peclet_number(v_slow, 2*r) > 10                                # advection dominates
assert math.isclose(mf.mean_velocity(Q, r), Q / (math.pi * r**2))

# 7. C cross-check (gcc-guarded): the 50 um / 1 cm / 1000 Pa channel
if mf.gcc_available():
    with tempfile.TemporaryDirectory() as d:
        c = mf.compile_and_run_c(d)
    assert math.isclose(c["R_hyd"], R, rel_tol=1e-9)
    assert math.isclose(c["Q"], Q, rel_tol=1e-9)
    print("test_microfluidic_circuits: all checks passed (incl. C cross-check)")
else:
    print("test_microfluidic_circuits: all checks passed (C cross-check skipped: no gcc)")

# 8. kwarg bounds
for bad in (lambda: mf.hydraulic_resistance_circular(0, L),
            lambda: mf.flow_rate(dP, 0),
            lambda: mf.parallel_resistance(1e12, -1),
            lambda: mf.reynolds_number(1e-3, 0),
            lambda: mf.mean_velocity(Q, 0)):
    try:
        bad(); assert False
    except ValueError:
        pass
