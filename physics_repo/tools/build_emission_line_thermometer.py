"""Generate notebooks/emission_line_thermometer.ipynb."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from nbkit import md, co, setup_cell, write

cells = [
md(r"""# Emission-line ratios as a stellar thermometer: the calculus

The Maxwell-Boltzmann notebook gave the *forward* law: at temperature $T$ the ratio of populations of
two hydrogen levels is $R(T)=\dfrac{g_j}{g_i}\,e^{-\Delta E/k_BT}$, and (with equal transition
probabilities) the emission-line strengths follow the same ratio. This notebook does the *inverse*
problem an astronomer actually solves: **measure the line ratio, compute the temperature.** That is a
short exercise in calculus:

1. **Invert** $R(T)$ algebraically to get $T(R)$.
2. **Differentiate** to get the sensitivity $dR/dT$ -- how sharply a line ratio pins the temperature.
3. **Propagate error**: a ratio measured from a finite number of detected photons carries Poisson
   noise, and $\delta T=\delta R/|dR/dT|$ turns that into a temperature uncertainty.
4. Connect to the **detector**: photons focused by a lens onto a $32\times32$ pixel sensor, its field
   of view, and how the pixel counts become $R$.

Self-contained: NumPy, SymPy, Pandas, Matplotlib only."""),
setup_cell(),

md(r"""## Forward law and a worked temperature

Hydrogen $n=3$ over $n=2$: $g_3=18$, $g_2=8$, $\Delta E=E_3-E_2=1.89\ \mathrm{eV}$. In electron-volts,
$k_B=8.617\times10^{-5}\ \mathrm{eV/K}$, so $k_BT=1.72\ \mathrm{eV}$ at $20{,}000\ \mathrm{K}$ gives the
textbook $R=0.75$."""),
co("""kB_eV = C.K_B / C.E                     # Boltzmann constant in eV/K
g_ratio = 18/8                          # g3/g2
dE = 1.89                               # E3 - E2 in eV
def R_of_T(T):
    return g_ratio * np.exp(-dE / (kB_eV * T))
print("R(20000 K) =", round(R_of_T(20000.0), 4), " (textbook 0.75)")
assert abs(R_of_T(20000.0) - 0.75) < 0.02"""),

md(r"""## 1. Invert $R(T)$ to get the temperature (SymPy)

Solving $R=\dfrac{g_j}{g_i}e^{-\Delta E/k_BT}$ for $T$ gives
$$T = \frac{\Delta E}{k_B\big[\ln(g_j/g_i)-\ln R\big]}.$$
SymPy performs the inversion; the numeric round-trip confirms a measured ratio recovers the
temperature that produced it."""),
co("""T, R, dEs, kBs, gs = sp.symbols('T R Delta_E k_B g', positive=True)
sol = sp.solve(sp.Eq(R, gs*sp.exp(-dEs/(kBs*T))), T)[0]
print("T(R) =", sol)
assert sp.simplify(sol - dEs/(kBs*sp.log(gs/R))) == 0

def T_of_R(Rmeas):
    return dE / (kB_eV * (np.log(g_ratio) - np.log(Rmeas)))
print("measured R = 0.75  ->  T =", round(T_of_R(0.75)), "K  (round-trips to 20000 K)")
assert abs(T_of_R(R_of_T(20000.0)) - 20000.0) < 1.0"""),

md(r"""## 2. Sensitivity: $dR/dT$

Differentiating $R(T)$,
$$\frac{dR}{dT}=R\,\frac{\Delta E}{k_BT^2}.$$
The fractional sensitivity $\dfrac{d\ln R}{d\ln T}=\dfrac{\Delta E}{k_BT}$ shows the diagnostic is
strongest when the level spacing is comparable to the thermal energy, $\Delta E\sim k_BT$: too cold
and the upper level is empty (tiny signal), too hot and the ratio saturates (flat, insensitive)."""),
co("""Tsym = sp.symbols('T', positive=True)
Rexpr = sp.Rational(18,8) * sp.exp(-sp.Float(1.89)/(sp.Float(kB_eV)*Tsym))
dRdT = sp.diff(Rexpr, Tsym)
# check dR/dT = R * dE/(kB T^2)
assert sp.simplify(dRdT - Rexpr*sp.Float(1.89)/(sp.Float(kB_eV)*Tsym**2)) == 0
print("dR/dT = R * dE/(kB T^2)   [verified symbolically]")
# fractional sensitivity dlnR/dlnT = dE/(kB T)
for Tk in (5000, 10000, 20000, 40000):
    print(f"T={Tk:6d} K:  dE/(kB T) = {dE/(kB_eV*Tk):.2f}  (fractional sensitivity of R to T)")"""),

md(r"""## 3. Error propagation: photons -> ratio -> temperature

The line ratio is measured as a ratio of detected photon counts, $R=N_j/N_i$. Photon counting is
Poisson, so $\delta N=\sqrt N$ and the fractional error on the ratio is
$$\frac{\delta R}{R}=\sqrt{\frac{1}{N_j}+\frac{1}{N_i}}.$$
Propagating through $T(R)$ with $\delta T=\delta R/|dR/dT|$ and $dR/dT=R\,\Delta E/k_BT^2$ gives
$$\frac{\delta T}{T}=\frac{k_BT}{\Delta E}\,\frac{\delta R}{R}.$$
More photons (a bigger telescope or longer exposure) shrink the temperature error as $1/\sqrt{N}$."""),
co("""def temperature_uncertainty(T_true, N_i):
    # assume the weaker upper line carries N_j = R * N_i counts
    R = R_of_T(T_true)
    N_j = R * N_i
    dR_over_R = np.sqrt(1/N_j + 1/N_i)              # Poisson error on the ratio
    dT_over_T = (kB_eV * T_true / dE) * dR_over_R   # propagated to temperature
    return dT_over_T * T_true

rows = []
for N_i in (1e3, 1e4, 1e5, 1e6):
    dT = temperature_uncertainty(20000.0, N_i)
    rows.append({"N_i (photons)": f"{N_i:.0e}", "delta_T (K)": round(dT, 1),
                 "delta_T / T": f"{dT/20000:.2%}"})
df = pd.DataFrame(rows)
print(df.to_string(index=False))
# 100x more photons -> 10x smaller temperature error (1/sqrt(N))
assert abs(temperature_uncertainty(20000.,1e3)/temperature_uncertainty(20000.,1e5) - 10) < 0.5"""),

md(r"""## 4. The detector: a $32\times32$ pixel sensor and its field of view

A lens of focal length $f$ images the star onto a square sensor of side $d$. The full field of view is
$\theta_{\rm FOV}=2\arctan\!\big(d/2f\big)$, and each of the $32\times32=1024$ pixels subtends
$\theta_{\rm FOV}/32$. A spectrometer disperses the light so that different columns record different
wavelengths; summing the photon counts under each emission line gives $N_i$ and $N_j$, and their ratio
is the thermometer above. The sensor's job is to turn photons into the integers whose Poisson
statistics set the temperature precision."""),
co("""d = 10e-3          # sensor side, 10 mm
f = 50e-3          # focal length, 50 mm
fov = 2*np.arctan(d/(2*f))
print(f"field of view      = {np.degrees(fov):.2f} deg")
print(f"per-pixel (32x32)  = {np.degrees(fov/32)*3600:.1f} arcsec/pixel")
print(f"total pixels       = {32*32}")
# a 32x32 frame with two emission lines (as columns) plus Poisson noise
rng = np.random.default_rng(0)
frame = rng.poisson(2.0, size=(32, 32)).astype(float)     # background
frame[:, 10] += rng.poisson(200.0, size=32)               # bright line i (n=2)
frame[:, 22] += rng.poisson(150.0, size=32)               # weaker line j (n=3)
N_i = frame[:, 10].sum() - 2.0*32                          # background-subtracted line counts
N_j = frame[:, 22].sum() - 2.0*32
print(f"line counts: N_i = {N_i:.0f}, N_j = {N_j:.0f}, measured R = {N_j/N_i:.3f} "
      f"-> T = {T_of_R(N_j/N_i):.0f} K")"""),

md(r"""## Plots"""),
co(r"""fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
Ts = np.linspace(3000, 50000, 400)
ax[0].plot(Ts, R_of_T(Ts), color="#4C78A8")
ax[0].set_xlabel("temperature (K)"); ax[0].set_ylabel("line ratio R")
ax[0].set_title("forward law R(T)")
ax[1].plot(Ts, R_of_T(Ts)*dE/(kB_eV*Ts**2)*1e4, color="#E45756")
ax[1].set_xlabel("temperature (K)"); ax[1].set_ylabel(r"dR/dT  ($10^{-4}$/K)")
ax[1].set_title("sensitivity peaks near dE ~ kT")
Ns = np.logspace(3, 7, 50)
ax[2].loglog(Ns, [temperature_uncertainty(20000., n) for n in Ns], color="#54A24B")
ax[2].set_xlabel("photons N_i"); ax[2].set_ylabel("temperature error (K)")
ax[2].set_title(r"$\delta T \propto 1/\sqrt{N}$")
plt.tight_layout(); plt.show()"""),

md(r"""## Summary

- **Forward:** $R(T)=(g_j/g_i)e^{-\Delta E/k_BT}$ (from the Boltzmann populations).
- **Inverse (calculus):** $T=\Delta E/\{k_B[\ln(g_j/g_i)-\ln R]\}$ turns a measured ratio into a
  temperature; the round-trip is exact.
- **Sensitivity:** $dR/dT=R\,\Delta E/k_BT^2$; the fractional sensitivity is $\Delta E/k_BT$, best when
  the line spacing matches the thermal energy.
- **Precision:** Poisson photon statistics give $\delta R/R=\sqrt{1/N_j+1/N_i}$ and hence
  $\delta T/T=(k_BT/\Delta E)\,\delta R/R\propto 1/\sqrt N$ -- more light, sharper temperature.
- **Detector:** a lens sets the field of view $2\arctan(d/2f)$; a $32\times32$ sensor integrates line
  photons into the counts whose statistics set the temperature error.

Subject-verb-object: the star emits lines; the lens forms an image; the sensor counts photons; the
ratio measures temperature; calculus propagates the error."""),
]

write("emission_line", "thermometer", cells)
