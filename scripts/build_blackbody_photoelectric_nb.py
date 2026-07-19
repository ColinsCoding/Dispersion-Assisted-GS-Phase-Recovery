"""Build notebooks/blackbody_photoelectric_photon.ipynb -- Planck->Stefan + photoelectric."""
import pathlib
import nbformat as nbf

md = lambda s: nbf.v4.new_markdown_cell(s)
co = lambda s: nbf.v4.new_code_cell(s)
nb = nbf.v4.new_notebook()

nb.cells = [
md("""# From Planck's spectrum to Stefan's law, and why the photoelectric effect ignores intensity
### two experiments that forced the photon E = h*nu = hbar*omega

Two facts broke classical physics at the turn of the 20th century, and both are fixed
by the **same** idea -- that light comes in quanta of energy $E=h\\nu=\\hbar\\omega$:

1. **Blackbody radiation.** Planck's *spectral distribution* $B(\\lambda,T)$, integrated
   over **all wavelengths**, gives the total radiated power -- and it comes out exactly
   the experimentally observed **Stefan-Boltzmann law** $M=\\sigma T^4$, with $\\sigma$
   built from $h$, $c$, $k_B$ alone. Classical (Rayleigh-Jeans) theory instead diverges
   -- the *ultraviolet catastrophe*.
2. **The photoelectric effect.** The maximum kinetic energy of ejected electrons
   depends **only on frequency**, $K_\\max=h\\nu-W$, and **not at all on intensity**.
   Brighter light ejects *more* electrons, not *faster* ones -- impossible for a
   classical wave, natural for photons.

This notebook derives Stefan's law from Planck's spectrum, checks $\\sigma$ against
fundamental constants, and shows the intensity-independence of the photoelectric
stopping voltage -- using `dgs/blackbody.py` and `dgs/photoelectric.py`. The quantum
$\\hbar\\omega$ here is the very one whose ladder we built in
`dgs/quantum_oscillator.py`. Civilian education."""),

co("""import numpy as np, matplotlib.pyplot as plt
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))
from dgs import blackbody as bb, photoelectric as pe
h, c, k, sigma = bb.H_PLANCK, bb.C_LIGHT, bb.K_BOLTZ, bb.SIGMA_SB
print("ready -- h=%.3e J s, c=%.3e m/s, k=%.3e J/K, sigma=%.4e W/m^2/K^4" % (h,c,k,sigma))"""),

md("""## 1. Planck's spectral distribution

$$B(\\lambda,T)=\\frac{2hc^2}{\\lambda^5}\\,\\frac{1}{e^{hc/\\lambda k T}-1}\\quad[\\text{W m}^{-2}\\,\\text{sr}^{-1}\\,\\text{nm}^{-1}]$$

Each curve is the light a body at temperature $T$ radiates at each wavelength. As $T$
rises the whole curve lifts **and** its peak slides to shorter wavelength (hotter = bluer)
-- Wien's displacement law, the dashed locus. The area under each curve (over all
wavelengths) is the *total* power -- that integral is Stefan's law, next."""),

co("""lam = np.linspace(50, 3000, 1500)   # nm
plt.figure(figsize=(8.5, 4.6))
peaks = []
for T in (3000, 4000, 5000, 5778):
    B = bb.planck_radiance(lam, T)
    plt.plot(lam, B/1e9, lw=2, label=f"T={T} K")
    lp = bb.wien_peak_nm(T); peaks.append((lp, bb.planck_radiance(lp, T)/1e9))
peaks = np.array(peaks)
plt.plot(peaks[:,0], peaks[:,1], "k--o", ms=4, label="Wien peak $\\\\lambda_{max}T=b$")
plt.xlabel("wavelength [nm]"); plt.ylabel("spectral radiance [GW m$^{-2}$ sr$^{-1}$ nm$^{-1}$]")
plt.title("Planck's law: the spectral distribution of blackbody radiation")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
print("the Sun (T~5778 K) peaks at %.0f nm -- green, mid-visible" % bb.wien_peak_nm(5778))"""),

md("""## 2. Integrate over all wavelengths $\\to$ the Stefan-Boltzmann law

Add up Planck over every wavelength (equivalently every frequency) and the messy
integral collapses to a clean $T^4$. In frequency, the radiated **exitance** is
$$M=\\int_0^\\infty \\frac{2\\pi h\\nu^3}{c^2}\\frac{d\\nu}{e^{h\\nu/kT}-1}
   \\;\\overset{x=h\\nu/kT}{=}\\; \\frac{2\\pi k^4 T^4}{c^2h^3}\\int_0^\\infty\\frac{x^3\\,dx}{e^x-1}.$$
The remaining pure number is $\\displaystyle\\int_0^\\infty\\frac{x^3}{e^x-1}dx=\\frac{\\pi^4}{15}$,
so
$$\\boxed{\\,M=\\sigma T^4,\\qquad \\sigma=\\frac{2\\pi^5k^4}{15\\,c^2h^3}\\,}$$
-- the Stefan-Boltzmann constant, out of $h$, $c$, $k$ with **no free parameters**."""),

co("""# the dimensionless integral, numerically vs the exact pi^4/15
x = np.linspace(1e-6, 60, 400000)
I_num = np.trapezoid(x**3/np.expm1(x), x)
print("integral x^3/(e^x-1): numeric %.6f  vs  pi^4/15 = %.6f" % (I_num, np.pi**4/15))

# sigma from fundamental constants
sigma_derived = 2*np.pi**5 * k**4 / (15 * c**2 * h**3)
print("sigma derived  = %.6e W/m^2/K^4" % sigma_derived)
print("sigma (CODATA) = %.6e  -> match: %s" % (sigma, np.isclose(sigma_derived, sigma, rtol=1e-4)))

# integrate Planck in frequency at a few T and confirm M = sigma T^4
def exitance_numeric(T):
    nu = np.linspace(1e11, 5e15, 400000)
    M_nu = 2*np.pi*h*nu**3/c**2 / np.expm1(h*nu/(k*T))
    return np.trapezoid(M_nu, nu)
for T in (300, 1000, 5778):
    print(f"T={T:5d} K: integral of Planck = {exitance_numeric(T):.4e},  "
          f"sigma T^4 = {sigma*T**4:.4e} W/m^2")"""),

co("""# the signature: total power scales as T^4 (slope 4 on log-log)
T = np.linspace(300, 6000, 200)
P = bb.stefan_boltzmann_power(T)     # sigma T^4
plt.figure(figsize=(7.5, 4.3))
plt.loglog(T, P, lw=2, label="$M=\\\\sigma T^4$")
plt.loglog(T, sigma*(T/1000.0)**4*1e12, ":", color="gray", alpha=0)  # keep scale
slope = np.polyfit(np.log(T), np.log(P), 1)[0]
plt.xlabel("temperature [K]"); plt.ylabel("radiated power [W m$^{-2}$]")
plt.title(f"Stefan-Boltzmann: total radiation $\\\\propto T^4$ (fitted slope {slope:.3f})")
plt.legend(); plt.grid(alpha=0.3, which="both"); plt.tight_layout(); plt.show()
print("doubling T multiplies radiated power by 2^4 =", 2**4)"""),

md("""## 3. Wien's displacement law -- where the peak sits

Setting $\\partial B/\\partial\\lambda=0$ gives a transcendental equation whose root is
$x=hc/\\lambda_{max}kT=4.965$, so $\\lambda_{max}T=b=hc/(4.965\\,k)=2.898\\times10^{-3}$ m K.
The peak wavelength is *inversely* proportional to temperature -- the quantitative form
of "hotter is bluer.\""""),

co("""# solve x = 5(1 - e^-x) by iteration -> 4.965
x = 5.0
for _ in range(100):
    x = 5*(1 - np.exp(-x))
b = h*c/(x*k)
print("Wien root x = %.5f" % x)
print("Wien b = hc/(x k) = %.6e m K  (accepted 2.898e-3)" % b)
print("check via dgs.blackbody at 5778 K: lambda_max = %.1f nm  (b/T = %.1f nm)"
      % (bb.wien_peak_nm(5778), b/5778*1e9))"""),

md("""## 4. The photoelectric effect: $K_{\\max}=h\\nu-W$, independent of intensity

Einstein: one photon of energy $h\\nu$ ejects one electron, spending the work function
$W$ to free it and leaving the rest as kinetic energy,
$$K_{\\max}=h\\nu-W.$$
Measure it with a *stopping voltage* $V_0=K_{\\max}/e$. Two experimental fingerprints,
both fatal to classical wave theory:

* $V_0$ vs $\\nu$ is a **straight line of slope $h/e$**, the same slope for every metal;
  the $x$-intercept is the threshold $\\nu_0=W/h$ (below it, no emission at any brightness).
* $V_0$ (hence $K_{\\max}$) **does not depend on intensity** -- only the *number* of
  electrons (the photocurrent) does. A brighter wave should give more energetic
  electrons; it does not."""),

co("""# stopping voltage vs frequency for three metals -> parallel lines of slope h/e
metals = {"cesium": 2.1, "sodium": 2.28, "zinc": 4.3}
plt.figure(figsize=(8.5, 4.4))
nu = np.linspace(3e14, 1.5e15, 200)
for name, phi in metals.items():
    nu0 = pe.threshold_frequency(phi)
    V = np.where(nu > nu0, [pe.stopping_voltage(n, phi) if n > nu0 else 0 for n in nu], 0)
    plt.plot(nu/1e15, V, lw=2, label=f"{name} ($W$={phi} eV, $\\\\nu_0$={nu0/1e15:.2f}$\\\\times10^{{15}}$)")
plt.axhline(0, color="gray", lw=0.6)
plt.xlabel("frequency [$10^{15}$ Hz]"); plt.ylabel("stopping voltage $V_0$ [V]")
plt.title("Photoelectric: $V_0=(h/e)\\\\,\\\\nu-W/e$ -- slope $h/e$, threshold at $W/h$")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

# the slope IS h/e, recovered from the line
phi = 2.28; nu_fit = np.linspace(pe.threshold_frequency(phi)*1.1, 1.5e15, 50)
Vf = np.array([pe.stopping_voltage(n, phi) for n in nu_fit])
slope = np.polyfit(nu_fit, Vf, 1)[0]
print("fitted slope dV0/dnu = %.4e V s   vs   h/e = %.4e V s" % (slope, h/1.602176634e-19))"""),

co("""# intensity independence: same frequency, 1x vs 3x brightness
phi, nu = 2.28, 1.2e15
V0 = pe.stopping_voltage(nu, phi)          # eV / e  = K_max in volts
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(["dim (1x)", "bright (3x)"], [V0, V0], color=["#4C78A8", "#F58518"])
ax[0].set(ylabel="stopping voltage $V_0$ [V]",
          title="$K_{max}$ is the SAME at both intensities")
ax[0].text(0.5, V0*0.5, f"$V_0$={V0:.2f} V\\n(same $\\\\nu$)", ha="center")
ax[1].bar(["dim (1x)", "bright (3x)"], [1, 3], color=["#4C78A8", "#F58518"])
ax[1].set(ylabel="photocurrent (rel.)",
          title="only the NUMBER of electrons scales with intensity")
plt.tight_layout(); plt.show()
print(f"K_max = h*nu - W = {V0:.3f} eV at nu={nu:.1e} Hz -- unchanged if we brighten the lamp;")
print("classical wave theory predicts K_max should grow with intensity. It does not.")"""),

md("""## 5. One quantum fixes both: $E=h\\nu=\\hbar\\omega$

Classical physics fails **both** experiments in the same way -- it treats light as a
continuous wave whose energy is set by *amplitude* (intensity). Quantizing the energy
into photons of $E=h\\nu=\\hbar\\omega$ fixes both at once:

* the blackbody integral converges (no ultraviolet catastrophe) and yields $\\sigma T^4$;
* the photoelectric energy is set by $\\nu$, not brightness, giving $K_{\\max}=h\\nu-W$.

That $\\hbar\\omega$ is exactly the quantum of energy whose ladder
$E_n=(n+\\tfrac12)\\hbar\\omega$ we built in `dgs/quantum_oscillator.py`: a blackbody
cavity is a bath of such oscillators (one per mode), and Planck's spectrum is their
thermal $\\langle E\\rangle=\\hbar\\omega/(e^{\\hbar\\omega/kT}-1)$ summed over modes."""),

co("""# Planck per mode IS the oscillator's thermal energy (minus zero point)
from dgs import quantum_oscillator as qo
omega = 2*np.pi*3e14          # a mid-visible mode
T = 5778.0
n_bar = qo.mean_occupation(omega, T, hbar=bb.HBAR if hasattr(bb,'HBAR') else h/(2*np.pi), kB=k)
hbar = h/(2*np.pi)
E_mode = hbar*omega*n_bar     # thermal energy above zero-point = Planck per mode
print("mode hbar*omega = %.3e J = %.2f eV" % (hbar*omega, hbar*omega/1.602e-19))
print("mean quanta <n> at %.0f K = %.4f ; thermal energy/mode = %.3e J" % (T, n_bar, E_mode))
print("this <n>*hbar*omega, summed over all cavity modes, IS Planck's spectrum -> Stefan's T^4.")"""),

md("""## What we showed

* **Planck's spectral distribution, integrated over all wavelengths, is Stefan's law**
  $M=\\sigma T^4$, and $\\sigma=2\\pi^5k^4/15c^2h^3$ comes out of $h,c,k$ with the pure
  number $\\int x^3/(e^x-1)\\,dx=\\pi^4/15$ -- verified numerically.
* **Wien's law** $\\lambda_{max}T=b$ from the transcendental root $x=4.965$.
* **The photoelectric effect** gives $K_{\\max}=h\\nu-W$: stopping voltage linear in
  $\\nu$ with slope $h/e$, and **independent of intensity** -- intensity moves only the
  electron count.
* Both are the same quantum $E=h\\nu=\\hbar\\omega$ -- the oscillator ladder of
  `dgs/quantum_oscillator.py`, now radiating."""),
]

nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "blackbody_photoelectric_photon.ipynb"
nbf.write(nb, str(out))
print("wrote", out)
