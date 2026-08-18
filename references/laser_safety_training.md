# Laser safety — reference notes

A written primer covering what a standard institutional laser-safety
online training (the kind required before bench access to a Class 3B/4
system in any university photonics lab, Jalali-lab included) actually
teaches, plus the quantitative side (`dgs/laser_safety.py`). **This is
study material, not a substitute for your institution's actual
Laser Safety Officer (LSO) sign-off** — complete the real course and
keep the certificate on file before touching a live beam.

## 1. Why laser light is hazardous — the quantum-optics reason

An incandescent bulb and a laser can emit the same total power and still
pose wildly different eye hazards, because the hazard depends on
**irradiance** (power per unit area) at the retina, not power alone.
Three properties of laser light, all consequences of stimulated emission
into a single (or few) electromagnetic mode, conspire to make that
irradiance dangerous:

- **Monochromaticity** — energy concentrated in a narrow $\Delta\lambda$
  (or even one photon energy $E=hc/\lambda$, see `dgs/blackbody.py`'s
  `photon_energy_eV`), so it isn't spread across the visible spectrum.
- **Collimation** — low divergence keeps the beam narrow over distance
  (see `beam_diameter_at_distance_cm` below), unlike a thermal source's
  $1/r^2$ spread in all directions.
- **Coherence** — the eye's lens can focus a collimated coherent beam
  down to a near-diffraction-limited spot on the retina, concentrating
  power into an area ~10,000x smaller than the input beam. This is also
  why laser light is well-described by **Poissonian (coherent-state)
  photon statistics** (Mandel Q=0) rather than the super-Poissonian
  Bose-Einstein statistics of thermal light — see `dgs/quantum_statistics.py`
  for the actual occupation-number math behind that distinction.

The retina has no pain receptors and a blink reflex too slow (~0.25s) to
protect against many exposures — you can sustain permanent retinal
damage from a visible or near-IR beam with no warning sensation at all.
Near-UV and far-IR wavelengths threaten the cornea/lens instead (the eye
doesn't focus those wavelengths onto the retina, but they're absorbed
before getting there).

## 2. ANSI Z136.1 classification (what the label on the laser means)

| Class | Meaning |
|---|---|
| 1 | Safe under all normal operating conditions (fully enclosed systems) |
| 1M | Safe unless viewed with magnifying optics |
| 2 | Visible, low power (<1 mW) — blink reflex is assumed protective |
| 2M | Class 2, but hazardous with magnifying optics |
| 3R | Low-to-moderate risk, direct beam viewing hazardous, diffuse-reflection generally safe |
| 3B | Direct beam viewing ALWAYS hazardous; diffuse reflection usually safe. Typical bench laser diode / HeNe range |
| 4 | Direct AND diffuse-reflection viewing hazardous; fire/skin hazard. Typical amplified/pulsed research laser range |

Most research photonics benches (amplifiers, pulsed sources, anything
feeding a supercontinuum stage like this repo's
`dgs/dispersive_barcode_correlator.py`'s block 60) run Class 3B or 4.

## 3. Control measures (the actual training content, in order of preference)

1. **Engineering controls** (preferred — don't rely on behavior):
   interlocked enclosures, beam shutters, beam dumps/blocks at the end of
   every optical path, key control, remote interlock connector.
2. **Administrative controls**: written Standard Operating Procedures
   (SOPs), posted warning signage at the Nominal Hazard Zone (NHZ)
   boundary, restricted/logged access, this training itself.
3. **Personal protective equipment (PPE)**: laser safety eyewear rated
   for the SPECIFIC wavelength and optical density (OD) in use — eyewear
   rated for one wavelength range provides **no protection** at another
   (this is the single most common real-world laser safety failure).

## 4. The quantitative side — `dgs/laser_safety.py`

Four tested functions (`tests/test_laser_safety.py`, 13 tests) implementing
the standard ANSI Z136.1 non-focusing-beam model:

```python
from dgs.laser_safety import (
    irradiance_W_cm2, beam_diameter_at_distance_cm,
    nominal_hazard_distance_cm, required_optical_density,
)

# worked example: 1W CW laser, 2mm exit beam, 1mrad divergence,
# MPE (maximum permissible exposure) = 2.5e-3 W/cm^2 (a representative
# visible-CW retinal MPE order of magnitude -- look up the real value
# for your actual wavelength/exposure duration before relying on this)
power_W, d0_cm, divergence_rad, mpe = 1.0, 0.2, 0.001, 2.5e-3

nhz_cm = nominal_hazard_distance_cm(power_W, mpe, d0_cm, divergence_rad)
print(f"Nominal Hazard Distance: {nhz_cm/100:.1f} m")  # ~223.7 m

incident = irradiance_W_cm2(power_W, d0_cm)
od_needed = required_optical_density(incident, mpe)
print(f"Eyewear OD required at the source: {od_needed:.2f}")  # ~4.10
```

(a 223 m NHZ for a 1 W beam at 1 mrad divergence looks large because this
is an UNFOCUSED, unobstructed-line-of-sight model — real lab NHZs are
almost always cut short by walls/enclosures long before this distance;
the number is exactly why unenclosed 3B/4 beams need engineering
controls, not a claim that the room needs to be 223 m across.)

**MPE values are wavelength-, exposure-duration-, and pulse-regime-
dependent** (a pulsed source needs a completely different MPE table than
CW) — always pull the real number from ANSI Z136.1 Table 5 (or your
institution's LSO) for the actual laser in use. The functions above take
MPE as an input precisely so a wrong assumed MPE doesn't get silently
baked into the code.

## 5. What the real online training covers (so you know what to expect)

Typical modules, in the order most EHS/LSO courses present them:
1. Characteristics of laser light (§1 above)
2. Biological effects on eye and skin
3. Classification system (§2 above)
4. Control measures — engineering, administrative, PPE (§3 above)
5. Alignment procedures (the highest-incident-rate activity — most real
   laser eye injuries happen during alignment with the beam unblocked
   below eye level)
6. Non-beam hazards: electrical (high-voltage power supplies), fire,
   chemical (dye lasers), collateral radiation
7. Emergency procedures and incident reporting

Complete your institution's actual course for the certificate — this
document is meant to make that course easier to follow, not to replace it.
