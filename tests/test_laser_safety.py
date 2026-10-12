"""Test dgs.laser_safety: irradiance/beam-area geometry, that NOHD is exactly the
distance where E(z) hits the supplied MPE, the OD<->attenuation-factor identity,
and the 0.25s blink/aversion boundary Class 2 safety is actually built on."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs import laser_safety as ls

# 1. beam area and irradiance: basic geometry, and irradiance scales as 1/d^2
assert math.isclose(ls.beam_area(2.0), math.pi)                      # d=2 -> A=pi*1^2
assert math.isclose(ls.beam_irradiance(1.0, 2.0), 1.0 / math.pi)
assert ls.beam_irradiance(1.0, 1.0) == 4 * ls.beam_irradiance(1.0, 2.0)   # halve d -> 4x irradiance

# 2. beam growth with distance: pure geometry, theta=0 -> no growth
assert ls.beam_diameter_at_distance(1e-3, 1e-3, 100.0) == 1e-3 + 100.0 * 1e-3
assert ls.beam_diameter_at_distance(1e-3, 0.0, 100.0) == 1e-3

# 3. NOHD: by construction, irradiance AT the computed NOHD must equal the MPE used
power_W, divergence, mpe, d0 = 5e-3, 1e-3, 25.0, 1e-3
nohd = ls.nominal_ocular_hazard_distance(power_W, divergence, mpe, d0)
d_at_nohd = ls.beam_diameter_at_distance(d0, divergence, nohd)
E_at_nohd = ls.beam_irradiance(power_W, d_at_nohd)
assert math.isclose(E_at_nohd, mpe, rel_tol=1e-9)

# more power -> larger hazard distance; more divergence -> smaller (spreads out faster)
assert ls.nominal_ocular_hazard_distance(2*power_W, divergence, mpe, d0) > nohd
assert ls.nominal_ocular_hazard_distance(power_W, 2*divergence, mpe, d0) < nohd

# collimated beam (theta=0) has no finite NOHD under this model -- rejected, not silently wrong
try:
    ls.nominal_ocular_hazard_distance(power_W, 0.0, mpe, d0); assert False
except ValueError:
    pass

# 4. optical density <-> attenuation factor: OD is exactly log10(E/MPE), and the
# attenuation factor it buys is exactly 10^OD by definition (round-trip identity)
E, mpe2 = 1000.0, 25.0
od = ls.optical_density_required(E, mpe2)
assert math.isclose(od, math.log10(E / mpe2))
assert math.isclose(ls.attenuation_factor(od), E / mpe2)
assert math.isclose(E / ls.attenuation_factor(od), mpe2)   # attenuated irradiance lands exactly at MPE

# already-below-MPE beams need no attenuation (clamped at 0, not negative)
assert ls.optical_density_required(1.0, 25.0) == 0.0

# 5. the one universal numeric constant in visible-laser classification: the
# 0.25s aversion/blink response time Class 2 safety is built on
assert ls.is_extended_source_safe_via_blink(0.1)
assert ls.is_extended_source_safe_via_blink(0.25)
assert not ls.is_extended_source_safe_via_blink(0.26)
assert not ls.is_extended_source_safe_via_blink(10.0)

# 6. all seven IEC/ANSI hazard classes are described
for cls in ('1', '1M', '2', '2M', '3R', '3B', '4'):
    assert cls in ls.CLASS_DESCRIPTIONS
    assert len(ls.CLASS_DESCRIPTIONS[cls]) > 10

# 7. kwarg bounds
for bad in (lambda: ls.beam_area(0),
            lambda: ls.beam_irradiance(-1.0, 1.0),
            lambda: ls.beam_diameter_at_distance(-1.0, 1e-3, 1.0),
            lambda: ls.nominal_ocular_hazard_distance(0, 1e-3, 25.0),
            lambda: ls.nominal_ocular_hazard_distance(1e-3, -1e-3, 25.0),
            lambda: ls.optical_density_required(0, 25.0),
            lambda: ls.attenuation_factor(-1.0),
            lambda: ls.is_extended_source_safe_via_blink(0)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_laser_safety: all checks passed")
