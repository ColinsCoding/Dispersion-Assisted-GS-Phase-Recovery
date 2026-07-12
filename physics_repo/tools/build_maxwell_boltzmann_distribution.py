"""Generate notebooks/maxwell_boltzmann_distribution.ipynb (Serway Ch. 10 notation)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from nbkit import md, co, setup_cell, write

cells = [
md(r"""# The Maxwell-Boltzmann distribution

Continuing directly from the microstate count $W=N!/(n_0!\,n_1!\cdots)$: maximizing $W$ subject to
fixed particle number and fixed total energy gives an exponential population law. Following Serway,
*Modern Physics*, Chapter 10, the probability that a particle occupies a state of energy $E_i$ at
absolute temperature $T$ is

$$f_{\rm MB} = A\,e^{-E_i/k_BT}, \tag{10.3}$$

and if the level has degeneracy (statistical weight) $g_i$, the number of particles in it is

$$n_i = g_i\,f_{\rm MB} = g_i A\,e^{-E_i/k_BT}, \tag{10.4}
\qquad N = \sum_i n_i. \tag{10.5}$$

The normalization $A$ is fixed by $N=\sum_i n_i$. Two consequences carry this notebook:

1. **Level populations** fall exponentially with energy. Ratios remove $A$:
   $\dfrac{n_j}{n_i}=\dfrac{g_j}{g_i}\,e^{-(E_j-E_i)/k_BT}$. We reproduce Serway's Example 10.1
   (emission lines from stellar hydrogen).
2. In the continuum, the density of states $g(E)\propto v^2$ turns (10.4) into **Maxwell's speed
   distribution** (10.8), whose moments give the mean, rms, and most-probable speeds.

The notebook is self-contained: only NumPy, SymPy, Pandas, and Matplotlib are required."""),
setup_cell(),

md(r"""## The Boltzmann factor and its dimension

The argument $E_i/k_BT$ must be dimensionless: $[k_B]=\mathrm{J/K}$, so $k_BT$ is an energy and the
ratio is a pure number. At room temperature $k_BT\approx 0.0259\ \mathrm{eV}$; at a stellar
$20{,}000\ \mathrm{K}$ it is $\approx 1.72\ \mathrm{eV}$. These two scales decide whether excited
atomic states (electron-volts above the ground state) are populated at all."""),
co("""kT = U.Quantity(C.K_B, U.ENERGY/U.TEMPERATURE) * U.Quantity(300.0, U.TEMPERATURE)
print("k_B T dimension:", kT.dim, "-> energy:", kT.dim == U.ENERGY)
def kT_eV(T):
    return C.K_B * T / C.E            # thermal energy in electron-volts
for T in (300.0, 20000.0):
    print(f"T = {T:8.0f} K   k_B T = {kT_eV(T):.4f} eV")
assert abs(kT_eV(300) - 0.02585) < 1e-4"""),

md(r"""## Example 10.1 -- Emission lines from stellar hydrogen

Hydrogen levels are $E_n=-13.6/n^2\ \mathrm{eV}$ with degeneracy $g_n=2n^2$. Populations relative to
the ground state are $\dfrac{n_j}{n_1}=\dfrac{g_j}{g_1}\,e^{-(E_j-E_1)/k_BT}$. At $300\ \mathrm{K}$ the
excited states are essentially empty; at $20{,}000\ \mathrm{K}$ they are measurably populated. The
emission-line strength is proportional to the upper-state population (equal transition
probabilities assumed), giving $\dfrac{S(3\to1)}{S(2\to1)}=\dfrac{n_3}{n_2}=0.75$."""),
co("""levels = {1: (-13.6, 2), 2: (-3.40, 8), 3: (-1.51, 18)}    # n: (E_n [eV], g_n = 2n^2)
def pop_ratio(j, i, T):
    (Ej, gj), (Ei, gi) = levels[j], levels[i]
    return (gj/gi) * np.exp(-(Ej - Ei) / kT_eV(T))

rows = []
for T in (300.0, 20000.0):
    rows.append({"T (K)": T, "n2/n1": pop_ratio(2,1,T), "n3/n1": pop_ratio(3,1,T),
                 "n3/n2 = S(3->1)/S(2->1)": pop_ratio(3,2,T)})
df = pd.DataFrame(rows)
print(df.to_string(index=False))
# textbook checks: ground state dominates at 300 K; the emission-strength ratio is 0.75 at 20000 K
assert pop_ratio(2,1,300) < 1e-100                       # 4 * e^{-395} ~ 0
assert abs(pop_ratio(2,1,20000) - 0.0107) < 5e-4         # n2/n1 = 0.0107
assert abs(pop_ratio(3,2,20000) - 0.75) < 0.02           # S(3->1)/S(2->1) = 0.75"""),

md(r"""## From populations to Maxwell's speed distribution

For a monatomic ideal gas each particle has $E=\tfrac12 mv^2$. The number of velocity states between
$v$ and $v+dv$ is the volume of a spherical shell in velocity space, $g(E)\,dE\propto 4\pi v^2\,dv$.
Multiplying the density of states by the Boltzmann factor $e^{-mv^2/2k_BT}$ and normalizing to the
number density $N/V$ gives Serway's Eq. (10.8):

$$n(v)\,dv=\frac{4\pi N}{V}\left(\frac{m}{2\pi k_BT}\right)^{3/2}v^2\,e^{-mv^2/2k_BT}\,dv. \tag{10.8}$$

SymPy confirms that the coefficient $A=(m/2\pi k_BT)^{3/2}$ is exactly what makes
$\int_0^\infty n(v)\,dv=N/V$."""),
co("""m, kB, T, v = sp.symbols('m k_B T v', positive=True)
a = m/(2*kB*T)                                            # so the exponent is -a v^2
shape = v**2 * sp.exp(-a*v**2)
I2 = sp.integrate(shape, (v, 0, sp.oo))                   # normalization integral
print("int_0^inf v^2 e^{-a v^2} dv =", sp.simplify(I2))
# require 4 pi A I2 = 1  ->  A = 1/(4 pi I2); check it equals (m/2 pi kB T)^{3/2}
A = sp.simplify(1/(4*sp.pi*I2))
assert sp.simplify(A - (m/(2*sp.pi*kB*T))**sp.Rational(3,2)) == 0
print("normalization A =", A, " = (m/2 pi kB T)^(3/2)  [verified]")"""),

md(r"""## Characteristic speeds from the moments

Three speeds summarize the distribution, each a different moment integral:
- **most probable** $v_{\rm mp}$: the maximum of $n(v)$, from $\frac{d}{dv}\big(v^2e^{-av^2}\big)=0$;
- **mean** $\bar v=\langle v\rangle$, the first moment;
- **root-mean-square** $v_{\rm rms}=\sqrt{\langle v^2\rangle}$, from the second moment.

SymPy evaluates the Gaussian integrals and returns the textbook closed forms
$v_{\rm mp}=\sqrt{2k_BT/m}$, $\bar v=\sqrt{8k_BT/\pi m}$ (Eq. 10.12), $v_{\rm rms}=\sqrt{3k_BT/m}$."""),
co("""v_mp = sp.solve(sp.diff(shape, v), v)[0]              # maximum of the distribution
mean_v = sp.simplify(sp.integrate(v*shape, (v,0,sp.oo)) / I2)
v_rms = sp.sqrt(sp.simplify(sp.integrate(v**2*shape, (v,0,sp.oo)) / I2))
print("v_mp  =", v_mp,  " = sqrt(2 kB T/m)")
print("v_bar =", mean_v, " = sqrt(8 kB T/ pi m)")
print("v_rms =", v_rms,  " = sqrt(3 kB T/m)")
assert sp.simplify(v_mp  - sp.sqrt(2*kB*T/m)) == 0
assert sp.simplify(mean_v - sp.sqrt(8*kB*T/(sp.pi*m))) == 0
assert sp.simplify(v_rms  - sp.sqrt(3*kB*T/m)) == 0
# ordering v_mp < v_bar < v_rms, independent of m, kB, T:
print("ratios  v_mp : v_bar : v_rms = 1 : %.3f : %.3f"
      % (float(sp.sqrt(4/sp.pi)), float(sp.sqrt(sp.Rational(3, 2)))))"""),

co("""# numbers for nitrogen (N2) gas at 300 K
m_N2 = 28 * 1.66053906660e-27           # kg
def speeds(T, mass):
    return dict(v_mp=np.sqrt(2*C.K_B*T/mass), v_bar=np.sqrt(8*C.K_B*T/(np.pi*mass)),
                v_rms=np.sqrt(3*C.K_B*T/mass))
s = speeds(300.0, m_N2)
print("N2 at 300 K:  v_mp = %.0f m/s,  v_bar = %.0f m/s,  v_rms = %.0f m/s"
      % (s['v_mp'], s['v_bar'], s['v_rms']))
assert s['v_mp'] < s['v_bar'] < s['v_rms']
assert abs(s['v_rms'] - 517) < 5                          # textbook ~ 517 m/s"""),

md(r"""## The Gaussian integral formula

Serway quotes $\displaystyle\int_0^\infty z^{2j}e^{-a z^2}\,dz=\frac{1\cdot3\cdot5\cdots(2j-1)}
{2^{\,j+1}a^{\,j}}\sqrt{\frac{\pi}{a}}$, the workhorse behind every moment above. SymPy verifies it for
$j=1,2,3$."""),
co("""z, aa = sp.symbols('z a', positive=True)
def rhs(j):
    num = sp.prod([2*t-1 for t in range(1, j+1)]) if j >= 1 else 1
    return sp.Rational(num) / (2**(j+1) * aa**j) * sp.sqrt(sp.pi/aa)
for j in (1, 2, 3):
    lhs = sp.integrate(z**(2*j) * sp.exp(-aa*z**2), (z, 0, sp.oo))
    assert sp.simplify(lhs - rhs(j)) == 0
    print(f"j={j}:  int z^{2*j} e^(-a z^2) dz =", sp.simplify(lhs), " [matches the formula]")"""),

md(r"""## Plots"""),
co("""fig, ax = plt.subplots(1, 2, figsize=(11.5, 4))
# (a) speed distribution at several temperatures (Fig. 10.4)
vv = np.linspace(0, 2500, 600)
for T, c in [(200, "#4C78A8"), (300, "#54A24B"), (600, "#E45756")]:
    A = (m_N2/(2*np.pi*C.K_B*T))**1.5
    nv = 4*np.pi*A * vv**2 * np.exp(-m_N2*vv**2/(2*C.K_B*T))
    ax[0].plot(vv, nv, color=c, label=f"T = {T} K")
    ax[0].axvline(np.sqrt(2*C.K_B*T/m_N2), color=c, ls=":", alpha=0.6)
ax[0].set_xlabel("speed v (m/s)"); ax[0].set_ylabel("n(v)  (arb.)")
ax[0].set_title("Maxwell speed distribution for N2 (dotted = v_mp)"); ax[0].legend()

# (b) hydrogen level populations vs temperature (Example 10.1)
Ts = np.linspace(2000, 30000, 300)
for j, c in [(2, "#4C78A8"), (3, "#E45756")]:
    ax[1].plot(Ts, [pop_ratio(j,1,T) for T in Ts], color=c, label=f"n{j}/n1")
ax[1].set_xlabel("temperature (K)"); ax[1].set_ylabel("population ratio")
ax[1].set_title("hydrogen excited-state populations (MB)"); ax[1].legend()
plt.tight_layout(); plt.show()"""),

md(r"""## Applied physics and computer engineering

The Boltzmann factor $e^{-E/k_BT}$ is the single most reused exponential in device physics. It sets
the **thermal (Johnson) noise** floor $k_BT$ that an ADC front end fights; the **dark-count** and
leakage rates of a photodiode, which rise as $e^{-E_g/k_BT}$; the carrier populations that the
Fermi-Dirac distribution reduces to in the Boltzmann limit; and the Arrhenius $e^{-E_a/k_BT}$ rate of
any thermally activated process. Numerically, evaluating these distributions is a vectorized
`exp` over an energy or velocity array -- the same array-and-loop pattern as every chapter here, and
the reason a spectrometer's firmware carries a temperature reading alongside every intensity.

Subject-verb-object: temperature populates the levels; the Boltzmann factor weights the energies; the
detector counts the populated states; the software computes the distribution."""),

md(r"""## Summary

- The maximized microstate count yields $n_i=g_iA\,e^{-E_i/k_BT}$ (Eqs. 10.3-10.5); ratios cancel
  $A$ and give the exponential population law.
- **Example 10.1** reproduced: at $300\ \mathrm{K}$ hydrogen is entirely in the ground state; at
  $20{,}000\ \mathrm{K}$, $n_2/n_1=0.0107$ and the emission ratio $S(3\to1)/S(2\to1)=0.75$.
- The density of states $g(E)\propto v^2$ turns the Boltzmann factor into Maxwell's speed
  distribution (Eq. 10.8), with $A=(m/2\pi k_BT)^{3/2}$ verified symbolically.
- Its moments give $v_{\rm mp}=\sqrt{2k_BT/m}<\bar v=\sqrt{8k_BT/\pi m}<v_{\rm rms}=\sqrt{3k_BT/m}$
  (517 m/s rms for N2 at 300 K), all from the one Gaussian-integral formula.

This is the statistical bridge from the microstate counting of the previous notebook to the thermal
physics that governs spectra, detectors, and noise."""),
]

write("maxwell_boltzmann", "distribution", cells)
