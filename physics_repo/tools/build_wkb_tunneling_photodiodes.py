"""Generate notebooks/wkb_tunneling_photodiodes.ipynb."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from nbkit import md, co, setup_cell, write

cells = [
md(r"""# The WKB approximation and tunneling in photodiodes

The WKB (semiclassical) approximation solves the Schrodinger equation where the potential varies slowly on
the scale of the de Broglie wavelength (Griffiths Ch. 9). Its most useful product is the **tunneling
probability** through a barrier a classical particle could never cross:
$$T\approx e^{-2\gamma},\qquad \gamma=\frac1\hbar\int_a^b\sqrt{2m\,[V(x)-E]}\;dx,$$
the integral running between the classical turning points. This one formula runs from alpha decay (Gamow)
to the scanning tunneling microscope -- and to the **dark current of a photodiode**.

In a reverse-biased photodiode the electric field tilts the bands into a **triangular barrier**, and WKB
through it gives the **Fowler-Nordheim / Zener** law $T\propto e^{-4\sqrt{2m}\,\Phi^{3/2}/(3\hbar eE)}$: a
field-driven **band-to-band tunneling** current that adds to the dark current and, at high field, drives
avalanche/Zener breakdown. (The same triangular barrier appeared as the confining well in the laser-gain
notebook -- tilt it the other way and it is a tunneling barrier.)

We verify WKB against the **exact** rectangular-barrier transmission, derive the triangular-barrier
(photodiode) case in SymPy, and compute Zener tunneling for a real semiconductor. Self-contained: NumPy,
SymPy, Pandas, Matplotlib. Units $\hbar=m=1$ for the model barriers, SI for the device."""),
setup_cell(),

md(r"""## The tunneling integral, tested on the rectangular barrier

For a rectangular barrier of height $V_0$ and width $L$ with $E<V_0$, the forbidden-region momentum is
imaginary, $\kappa=\sqrt{2m(V_0-E)}/\hbar$, and WKB gives simply $T_{\text{WKB}}=e^{-2\kappa L}$. The
**exact** transmission is
$$T=\left[1+\frac{V_0^2\sinh^2(\kappa L)}{4E(V_0-E)}\right]^{-1}.$$
For a thick/high barrier the exact result becomes $\dfrac{16E(V_0-E)}{V_0^2}\,e^{-2\kappa L}$: WKB captures
the **exponent exactly** and misses only the constant prefactor. We confirm the ratio approaches that
prefactor and that $\ln T$ has slope $-2\kappa$ in both."""),
co("""V0, E = 4.0, 1.0                                              # hbar = m = 1
kappa = np.sqrt(2*(V0 - E))
T_exact = lambda L: 1/(1 + V0**2*np.sinh(kappa*L)**2/(4*E*(V0 - E)))
T_wkb   = lambda L: np.exp(-2*kappa*L)
prefactor = 16*E*(V0 - E)/V0**2                              # thick-barrier prefactor WKB omits

rows = []
for L in (2.0, 3.0, 4.0, 5.0):
    te, tw = T_exact(L), T_wkb(L)
    rows.append({"L": L, "exact T": f"{te:.3e}", "WKB T": f"{tw:.3e}", "ratio": round(te/tw, 4)})
print(pd.DataFrame(rows).to_string(index=False))
print(f"\\nexact/WKB ratio -> 16 E (V0-E)/V0^2 = {prefactor}  (WKB gets the exponent, misses the prefactor)")
assert abs(T_exact(5.0)/T_wkb(5.0) - prefactor) < 1e-3
# log-slope is -2 kappa for both
slope_exact = (np.log(T_exact(5.0)) - np.log(T_exact(4.0)))/1.0
assert abs(slope_exact - (-2*kappa)) < 1e-3
print(f"d(ln T)/dL = {slope_exact:.4f}  =  -2 kappa = {-2*kappa:.4f}  (WKB exponent is exact)")"""),

md(r"""## The triangular barrier: Fowler-Nordheim / Zener tunneling (SymPy)

A uniform field tilts a barrier to $V(x)=\Phi-Fx$ ($F=eE$ the electric force). Tunneling runs from $x=0$
to the turning point $x_t=\Phi/F$. SymPy evaluates the WKB integral:
$$\gamma=\frac1\hbar\int_0^{\Phi/F}\sqrt{2m(\Phi-Fx)}\,dx=\frac{2\sqrt{2m}\,\Phi^{3/2}}{3\hbar F},\qquad
T=e^{-2\gamma}=\exp\!\left[-\frac{4\sqrt{2m}\,\Phi^{3/2}}{3\hbar eE}\right].$$
This is the **Fowler-Nordheim law** (field emission) and, for band-to-band tunneling across a reverse-biased
junction with $\Phi\sim E_g$, the **Zener** tunneling that feeds a photodiode's dark current. Note the
signature $\ln T\propto-1/E$: the current is exponentially sensitive to the field."""),
co("""m, hbar, Phi, F, x = sp.symbols('m hbar Phi F x', positive=True)
gamma = sp.simplify(sp.integrate(sp.sqrt(2*m*(Phi - F*x)), (x, 0, Phi/F))/hbar)
assert sp.simplify(gamma - 2*sp.sqrt(2*m)*Phi**sp.Rational(3,2)/(3*hbar*F)) == 0
T_tri = sp.exp(-2*gamma)
print("WKB exponent gamma =", gamma)
print("triangular-barrier transmission  T = exp(-2 gamma) = exp[-4 sqrt(2m) Phi^{3/2} / (3 hbar e E)]")
print("  -> Fowler-Nordheim / Zener band-to-band tunneling; ln T is proportional to -1/E")"""),

md(r"""## Zener tunneling in a photodiode: dark current vs reverse-bias field

Put in real numbers for a silicon-like junction: barrier $\Phi\approx E_g=1.12$ eV, tunneling effective
mass $m^*\approx0.2\,m_e$. Sweep the depletion-region field $E$. The transmission is negligible until the
field reaches $\sim10^8\,\mathrm{V/m}$, then rises by tens of orders of magnitude over a narrow range --
exactly the steep dark-current turn-on that marks the onset of **Zener/avalanche breakdown** and limits how
far a photodiode can be reverse-biased."""),
co("""Eg = 1.12*C.E                                                # barrier height (J)
mstar = 0.2*C.M_E                                            # tunneling effective mass
def zener_T(field):                                         # field in V/m
    expo = 4*np.sqrt(2*mstar)*Eg**1.5/(3*C.HBAR*C.E*field)
    return np.exp(-expo), expo

rows = []
for field in (3e7, 1e8, 2e8, 3e8, 5e8):
    T, expo = zener_T(field)
    rows.append({"field [V/m]": f"{field:.0e}", "FN exponent": round(expo, 2), "tunneling T": f"{T:.2e}"})
print(pd.DataFrame(rows).to_string(index=False))
print("\\nT jumps from ~1e-53 to ~1e-3 between 3e7 and 5e8 V/m -> steep dark-current / breakdown onset")
assert zener_T(3e8)[0] > zener_T(3e7)[0]*1e40                # tunneling rises enormously with field"""),

md(r"""## Plots"""),
co(r"""fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.2))
# (1) rectangular barrier: exact vs WKB transmission
Ls = np.linspace(1, 6, 100)
ax[0].semilogy(Ls, [T_exact(L) for L in Ls], color="#4C78A8", label="exact")
ax[0].semilogy(Ls, [T_wkb(L) for L in Ls], "--", color="#E45756", label="WKB $e^{-2\\kappa L}$")
ax[0].set_xlabel("barrier width L"); ax[0].set_ylabel("transmission T")
ax[0].set_title("WKB vs exact: same slope, const. offset"); ax[0].legend(fontsize=8)
# (2) the triangular barrier picture
xg = np.linspace(-0.5, 2.2, 200); Phi_v, F_v, E_v = 2.0, 1.0, 0.0
Vg = np.where(xg < 0, 0.0, np.where(xg < Phi_v/F_v, Phi_v - F_v*xg, 0.0))
ax[1].plot(xg, Vg, color="#333"); ax[1].axhline(E_v, ls=":", color="#4C78A8", label="E")
ax[1].fill_between(xg, E_v, Vg, where=(Vg > E_v), color="#E45756", alpha=0.2)
ax[1].annotate("tunnel", xy=(0.9, 0.6), color="#E45756")
ax[1].set_xlabel("position x"); ax[1].set_ylabel("V(x)")
ax[1].set_title("reverse bias -> triangular barrier"); ax[1].legend(fontsize=8)
# (3) Zener transmission vs reverse-bias field (dark-current turn-on)
fields = np.logspace(7.3, 8.9, 100)
Ts = np.array([zener_T(f)[0] for f in fields])
ax[2].semilogy(fields/1e8, Ts, color="#E45756")
ax[2].set_xlabel("field [10$^8$ V/m]"); ax[2].set_ylabel("tunneling T")
ax[2].set_title("Zener tunneling: steep dark-current onset")
plt.tight_layout(); plt.show()"""),

md(r"""## Exercises

1. **Gamow / alpha decay.** For the Coulomb barrier $V(r)=\tfrac{2Ze^2}{4\pi\varepsilon_0 r}$ outside the
   nucleus, evaluate the WKB integral and reproduce the Geiger-Nuttall relation between half-life and
   decay energy -- the original triumph of tunneling theory.
2. **STM.** With a vacuum barrier of work function $\Phi\approx4$ eV, show the tunneling current changes by
   a decade for every ~1 Angstrom change in tip-sample gap -- the atomic resolution of the microscope.
3. **Resonant-tunneling diode.** Put two barriers in series with a well between; find the energies where
   $T\to1$ (resonances) and relate them to the well's quasi-bound states.
4. **Temperature vs field.** Compare Zener (field) tunneling current with thermally activated dark current
   $\propto e^{-E_g/2k_BT}$; find the field/temperature boundary where tunneling dominates.

## Summary

- WKB gives the tunneling probability $T\approx e^{-2\gamma}$, $\gamma=\tfrac1\hbar\int\sqrt{2m(V-E)}\,dx$
  between turning points. Against the **exact** rectangular barrier it reproduces the exponent exactly and
  misses only the prefactor $16E(V_0-E)/V_0^2$ (verified).
- The **triangular barrier** (reverse-biased junction) gives, in SymPy, the **Fowler-Nordheim / Zener** law
  $T=\exp[-4\sqrt{2m}\,\Phi^{3/2}/(3\hbar eE)]$ with $\ln T\propto-1/E$.
- For a silicon-like photodiode this **band-to-band tunneling** is negligible until $E\sim10^8\,$V/m, then
  rises by tens of orders of magnitude -- the dark-current turn-on and the onset of Zener/avalanche
  breakdown that bounds the usable reverse bias.

Subject-verb-object: the field tilts the barrier; the WKB integral sets the exponent; the tunneling current
turns on steeply; the dark current bounds the bias."""),
]

write("wkb_tunneling", "photodiodes", cells)
