"""Build notebooks/quantum_spin_zeeman_mri.ipynb -- ties together this
session's 4 new modules: electron_spin_resonance, thomas_precession_
gyromagnetic_anomaly, torch.stern_gerlach_zeeman_hydrogen, and
total_angular_momentum_coupling. Uses the py312 (torch) kernel since one
section needs torch."""
import pathlib
import nbformat as nbf

md = lambda s: nbf.v4.new_markdown_cell(s)
co = lambda s: nbf.v4.new_code_cell(s)
nb = nbf.v4.new_notebook()

nb.cells = [
md("""# Electron spin, from quantum numbers to an MRI scanner
### ESR -> Stern-Gerlach -> the gyromagnetic anomaly -> total angular momentum -> MRI

A single thread runs through all of this: an electron (or proton) has a magnetic
moment tied to its angular momentum, and that magnetic moment does specific,
checkable things in a field -- it resonates at a precise frequency (ESR/NMR/MRI),
it splits a beam into a discrete, EVEN number of spots (Stern-Gerlach), it comes out
almost exactly double what classical E&M predicts (the gyromagnetic anomaly), and
once spin-orbit coupling enters the picture, only the TOTAL angular momentum
J = L + S stays sharp. Every number below is checked against a real, known
reference value -- not just internally self-consistent."""),

co("""import numpy as np
import matplotlib.pyplot as plt
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))

from dgs import electron_spin_resonance as esr
from dgs import thomas_precession_gyromagnetic_anomaly as tp
from dgs import total_angular_momentum_coupling as tam

print(f"electron gyromagnetic ratio: {esr.gyromagnetic_ratio_hz_per_tesla()/1e9:.3f} GHz/T "
      f"(real value: ~28.025 GHz/T)")"""),

md("""## 1. Electron Spin Resonance: the Zeeman resonance condition

The resonance condition is $h f = g\\mu_B B$. Real X-band EPR spectrometers run at
9.5 GHz -- inverting the formula predicts the field they must use."""),
co("""f_xband = 9.5e9
B_needed = esr.field_for_resonance_tesla(f_xband)
print(f"X-band EPR at {f_xband/1e9:.1f} GHz requires B = {B_needed:.3f} T "
      f"(real X-band spectrometers: ~0.34 T)")

m_s = esr.spin_quantum_numbers(s=0.5)
print(f"space quantization for spin-1/2: allowed m_s = {m_s} -- exactly 2 states")

fields = np.linspace(0.05, 1.0, 200)
freqs_ghz = [esr.larmor_frequency_hz(B) / 1e9 for B in fields]
plt.figure(figsize=(6, 4))
plt.plot(fields, freqs_ghz)
plt.axvline(B_needed, color="C3", ls="--", label=f"X-band: {B_needed:.3f} T")
plt.axhline(9.5, color="C3", ls=":")
plt.xlabel("field B (T)"); plt.ylabel("resonance frequency (GHz)")
plt.title("Electron spin resonance condition: f = g*mu_B*B/h")
plt.legend(); plt.tight_layout(); plt.show()"""),

md("""## 2. Real application: tooth-enamel EPR dosimetry

Radiation creates stable free radicals in tooth enamel (hydroxyapatite); their EPR
signal grows linearly with accumulated dose and persists for years. This is a real
retrospective radiation dosimetry technique (used in Chernobyl/Fukushima dose
reconstruction) -- an ordinary extracted tooth becomes a dosimeter."""),
co("""sensitivity = 50.0
noise_floor = 5.0
doses = np.linspace(0, 5, 50)
signal = [esr.enamel_epr_signal(d, sensitivity) for d in doses]
lod = esr.minimum_detectable_dose_gray(noise_floor, sensitivity)

plt.figure(figsize=(6, 4))
plt.plot(doses, signal)
plt.axvline(lod, color="C3", ls="--", label=f"LOD = {lod:.2f} Gy")
plt.xlabel("accumulated dose (Gy)"); plt.ylabel("EPR signal (a.u.)")
plt.title("Tooth-enamel EPR dosimetry: linear dose-response")
plt.legend(); plt.tight_layout(); plt.show()
print(f"limit of detection: {lod:.3f} Gy (real reported range: tens of mGy to ~0.1 Gy)")"""),

md("""## 3. Can spin be a classically spinning charged sphere? No.

Using the electron's classical radius and moment of inertia $I=\\frac{2}{5}mr^2$,
matching its REAL spin angular momentum $\\hbar/2$ requires an equatorial surface
speed hundreds of times the speed of light -- the standard textbook argument that
spin is intrinsically quantum."""),
co("""r_e = tp.classical_electron_radius_m()
v, v_over_c = tp.required_surface_speed_m_s(9.1093837015e-31, r_e)
print(f"classical electron radius: {r_e:.3e} m (real: ~2.818e-15 m)")
print(f"required surface speed for spin hbar/2: {v:.3e} m/s = {v_over_c:.1f}x the speed of light")

anomaly = tp.gyromagnetic_anomaly()
print(f"\\ngyromagnetic anomaly: g_classical = {anomaly['g_classical']}, "
      f"g_quantum = {anomaly['g_quantum']}")
print(f"QED anomalous moment a_e = (g-2)/2 = {anomaly['qed_anomaly_a_e']:.4e} "
      f"(real measured value: ~1.16e-3)")"""),

md("""## 4. Stern-Gerlach: an EVEN split proves spin is real (torch)

Silver's outer 5s electron has L=0 -- the ONLY angular momentum available is spin
(S=1/2), giving 2S+1 = 2 beam spots: an EVEN number, with no undeflected center spot.
That is impossible for pure orbital angular momentum (always ODD, 2l+1). This cell
needs the **py312 (torch)** kernel."""),
co("""from dgs.torch import stern_gerlach_zeeman_hydrogen as sgz

J_silver = 0.5
print(f"silver valence electron: L=0, S=1/2, J=1/2 -> "
      f"{int(round(2*J_silver+1))} beam spots, even split: {sgz.is_even_split(J_silver)}")

m_Ag_kg = 107.87 * sgz.AMU
v_beam = sgz.most_probable_beam_speed_m_s(1200.0, m_Ag_kg)
z_one = sgz.stern_gerlach_deflection_m(1000.0, 0.035, 0.035, m_Ag_kg, v_beam)
print(f"beam speed at 1200K oven: {v_beam:.1f} m/s")
print(f"total beam split: {2*z_one*1e3:.3f} mm (real 1922 experiment: sub-mm scale)")

for name, L, S, J in [("2S_1/2", 0, 0.5, 0.5), ("2P_1/2", 1, 0.5, 0.5), ("2P_3/2", 1, 0.5, 1.5)]:
    g = sgz.lande_g_factor(L, S, J)
    print(f"  {name}: Lande g_J = {g:.4f}")"""),

md("""## 5. The anomalous Zeeman spectrum of hydrogen (torch)

$2P_{3/2} \\to 2S_{1/2}$ has DIFFERENT g-factors on each side, so the transition
splits into more than the naive 3-line "normal" Zeeman pattern."""),
co("""lines, g_upper, g_lower = sgz.zeeman_transition_lines(
    L_upper=1, S_upper=0.5, J_upper=1.5,
    L_lower=0, S_lower=0.5, J_lower=0.5,
    B_tesla=0.5, E0_joules=1.634e-18)
print(f"g_upper (2P_3/2) = {g_upper:.4f}, g_lower (2S_1/2) = {g_lower:.4f}")
print(f"number of distinct anomalous Zeeman lines: {len(lines)} (normal Zeeman: 3)")

energies_ev = np.array([l["energy_j"] for l in lines]) / 1.602e-19
plt.figure(figsize=(7, 3))
for e in energies_ev:
    plt.axvline(e, ymin=0.1, ymax=0.9, lw=1.5)
plt.xlabel("transition energy (eV, relative)")
plt.title(f"Anomalous Zeeman spectrum: 2P_3/2 -> 2S_1/2 ({len(lines)} lines)")
plt.yticks([]); plt.tight_layout(); plt.show()"""),

md("""## 6. Why J, not L or S, is the sharp observable

With spin-orbit coupling $H_{SO}=\\xi\\,\\mathbf{L}\\cdot\\mathbf{S}$, explicit matrix
commutators settle it directly: $[H_{SO}, L_z]$ and $[H_{SO}, S_z]$ are nonzero, but
$[H_{SO}, J_z]$, $[H_{SO}, L^2]$, $[H_{SO}, S^2]$, $[H_{SO}, J^2]$ all vanish."""),
co("""results = tam.verify_good_quantum_numbers(L=1, S=0.5)
for name, r in results.items():
    status = "CONSERVED" if r["conserved"] else "NOT conserved"
    print(f"  [H_SO, {name}]  norm = {r['commutator_norm']:.2e}  ->  {status}")

print()
for J in tam.allowed_J_values(1, 0.5):
    print(f"  J={tam.quantum_number_as_fraction(J)}  ->  term symbol {tam.term_symbol_string(1, 0.5, J)}")"""),

md("""## 7. MRI: the exact same Larmor physics, a different particle

MRI uses proton (not electron) spin precession -- same resonance condition, the
proton's real gyromagnetic ratio (42.577 MHz/T) instead of the electron's."""),
co("""for B in [1.5, 3.0]:
    f = tam.proton_larmor_frequency_hz(B) / 1e6
    ref = 63.9 if B == 1.5 else 127.7
    print(f"  B = {B} T  ->  proton Larmor frequency = {f:.2f} MHz (real clinical scanner: ~{ref} MHz)")

fields = np.linspace(0.1, 3.0, 100)
freqs = [tam.proton_larmor_frequency_hz(B) / 1e6 for B in fields]
plt.figure(figsize=(6, 4))
plt.plot(fields, freqs)
for B, ref in [(1.5, 63.9), (3.0, 127.7)]:
    plt.plot(B, ref, "C3o", ms=8)
plt.xlabel("field B (T)"); plt.ylabel("proton Larmor frequency (MHz)")
plt.title("MRI resonance condition (real clinical field strengths marked)")
plt.tight_layout(); plt.show()"""),

md("""## Summary

| Section | Real-world check | Result |
|---|---|---|
| ESR resonance | X-band field | 0.339 T (real: ~0.34 T) |
| Tooth-enamel dosimetry | LOD | ~0.1 Gy (real: tens of mGy-0.1 Gy) |
| Classical spin sphere | surface speed | ~171x c (impossible) |
| Gyromagnetic anomaly | QED a_e | 1.15e-3 (real: ~1.16e-3) |
| Stern-Gerlach | beam split | ~1 mm (real 1922: sub-mm) |
| Anomalous Zeeman | g-factors | exact match to textbook values |
| J vs L,S conservation | matrix commutators | J conserved, L/S not (verified) |
| MRI | proton Larmor f | 63.87/127.73 MHz (real: 63.9/127.7 MHz) |

Every number here traces back to a tested module in `dgs/`:
`electron_spin_resonance.py`, `thomas_precession_gyromagnetic_anomaly.py`,
`torch/stern_gerlach_zeeman_hydrogen.py`, `total_angular_momentum_coupling.py`.""")
]

nb.metadata.kernelspec = {
    "display_name": "Python 3.12 (torch)",
    "language": "python",
    "name": "py312",
}

out = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "quantum_spin_zeeman_mri.ipynb"
nbf.write(nb, out)
print(f"wrote {out}")
