"""Generate notebooks/cell_throughput_budget.ipynb."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from nbkit import md, co, setup_cell, write

cells = [
md(r"""# A photon and throughput budget for 100,000 cells/second

The instrument goal is to classify $R=10^5$ cells per second. That fixes a hard **time budget** of
$t_{\rm cell}=1/R=10\ \mu\mathrm{s}$ per cell, into which four things must fit: illuminate the cell,
collect its photons, digitize the signal, and run a classifier. This notebook builds the quantitative
budget and answers which stage is the real bottleneck.

The physics thread is the same Poisson photon statistics used to turn emission lines into a
temperature: a cell scatters or fluoresces a finite number of photons $N$, and the signal-to-noise
ratio is $\sqrt N$. The question "can we classify a cell in $10\ \mu\mathrm{s}$?" is really "can we
collect enough photons in $10\ \mu\mathrm{s}$?" -- a statistics problem before it is a compute problem.

Self-contained: NumPy, SymPy, Pandas, Matplotlib, and `math.erfc` from the standard library."""),
setup_cell(),

md(r"""## 1. The time budget

$R=10^5\ \mathrm{s^{-1}}$ gives $t_{\rm cell}=10\ \mu\mathrm{s}$. If the transit waveform is sampled at
$M$ points per cell, the analog-to-digital converter must run at $f_s = M R$."""),
co("""import math
R = 1e5                                  # cells per second (the goal)
t_cell = 1.0 / R
M = 128                                  # samples digitized per cell transit
f_s = M * R                              # required ADC sample rate
print(f"time per cell      t_cell = {t_cell*1e6:.1f} us")
print(f"samples per cell   M      = {M}")
print(f"required ADC rate  f_s    = {f_s/1e6:.1f} MS/s")
assert abs(t_cell - 1e-5) < 1e-12"""),

md(r"""## 2. Photon budget and the Poisson limit

A cell emits photons at rate $\Phi$; the optics collect a fraction $\eta$; in the window $t_{\rm cell}$
the detector registers $N=\Phi\,\eta\,t_{\rm cell}$ photons. Photon counting is Poisson, so the shot
noise is $\sqrt N$ and $\mathrm{SNR}=N/\sqrt N=\sqrt N$. Reaching a useful SNR in only $10\ \mu\mathrm{s}$
is the central constraint."""),
co("""def photons_per_cell(flux, eta, t=t_cell):
    return flux * eta * t
for flux, eta in [(1e10, 0.05), (1e9, 0.10), (1e8, 0.20)]:
    N = photons_per_cell(flux, eta)
    print(f"flux={flux:.0e}/s, eta={eta:.2f}:  N = {N:6.0f} photons/cell,  SNR = sqrt(N) = {np.sqrt(N):5.1f}")
# SNR is set by the collected photon count alone
assert np.isclose(np.sqrt(photons_per_cell(1e9, 0.10)), np.sqrt(1000.0))"""),

md(r"""## 3. How many photons does a classification need? (SymPy + statistics)

Two cell types produce mean photon counts $\mu_0$ and $\mu_1=(1+c)\mu_0$, where $c$ is the contrast.
With Poisson noise $\sigma\approx\sqrt{\bar\mu}$, the separability is the discriminability index
$d'=\dfrac{\mu_1-\mu_0}{\sqrt{\bar\mu}}=c\sqrt{\mu_0}$ (to leading order), and the minimum
classification error of an ideal threshold is $\varepsilon=Q(d'/2)$ with
$Q(x)=\tfrac12\mathrm{erfc}(x/\sqrt2)$. Inverting for the required photon count,
$$\mu_0=\left(\frac{2\,Q^{-1}(\varepsilon)}{c}\right)^2.$$
SymPy confirms the leading-order $d'=c\sqrt{\mu_0}$, and we evaluate the requirement numerically."""),
co("""mu0, c = sp.symbols('mu0 c', positive=True)
mu1 = (1+c)*mu0
d_prime = sp.simplify((mu1 - mu0)/sp.sqrt((mu0+mu1)/2))
d_prime_lead = sp.series(d_prime, c, 0, 2).removeO()      # leading order in small contrast
print("d' =", d_prime, "  ~", d_prime_lead, "(small c)")
assert sp.simplify(d_prime_lead - c*sp.sqrt(mu0)) == 0

def Q(x):        return 0.5*math.erfc(x/math.sqrt(2))     # Gaussian tail
def Qinv(p):                                              # invert Q by bisection
    lo, hi = 0.0, 12.0
    for _ in range(100):
        mid = 0.5*(lo+hi); (lo, hi) = (mid, hi) if Q(mid) > p else (lo, mid)
    return 0.5*(lo+hi)
def photons_required(contrast, err):
    return (2*Qinv(err)/contrast)**2

rows = []
for contrast in (0.5, 0.2, 0.1):
    for err in (1e-2, 1e-3):
        mu = photons_required(contrast, err)
        rows.append({"contrast c": contrast, "error eps": err, "photons mu0 needed": round(mu),
                     "collect in 10us at rate": f"{mu/t_cell:.1e} /s"})
print(pd.DataFrame(rows).to_string(index=False))"""),

md(r"""## 4. Compute budget -- and why photons, not FLOPs, are the wall

Per cell the pipeline runs an FFT of the $M$-sample waveform ($\approx 5M\log_2 M$ FLOPs) plus a small
classifier (a $M\times H$ then $H\times2$ matrix-vector product). Multiplied by $R$ cells/s this is the
sustained throughput a CPU or GPU must hold. It comes out in the low GFLOP/s range -- trivial for
modern hardware -- so the binding constraint is collecting enough photons in $10\ \mu\mathrm{s}$, not
arithmetic."""),
co("""H = 32                                          # hidden units in a tiny classifier
fft_flops = 5*M*np.log2(M)
clf_flops = 2*M*H + 2*H*2                        # two matrix-vector products
per_cell = fft_flops + clf_flops
throughput = per_cell * R
print(f"FFT       : {fft_flops:8.0f} FLOP/cell")
print(f"classifier: {clf_flops:8.0f} FLOP/cell")
print(f"total     : {per_cell:8.0f} FLOP/cell")
print(f"sustained : {throughput/1e9:.3f} GFLOP/s at {R:.0e} cells/s  (a modern CPU exceeds this)")
# ADC data rate for context
print(f"ADC data  : {f_s*12/8/1e6:.1f} MB/s at 12-bit, {M} samples/cell")
assert throughput < 1e11                         # comfortably below CPU/GPU peak"""),

md(r"""## 5. Plots"""),
co("""fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
# SNR vs photons collected
Ns = np.logspace(1, 5, 100)
ax[0].loglog(Ns, np.sqrt(Ns), color="#4C78A8")
ax[0].set_xlabel("photons/cell N"); ax[0].set_ylabel("SNR = sqrt(N)")
ax[0].set_title("Poisson-limited SNR")
# classification error vs photons for several contrasts
mu = np.logspace(1, 4, 100)
for contrast, cc in [(0.5, "#4C78A8"), (0.2, "#54A24B"), (0.1, "#E45756")]:
    err = [Q(contrast*np.sqrt(m)/2) for m in mu]
    ax[1].semilogy(mu, err, color=cc, label=f"c={contrast}")
ax[1].set_xlabel("photons/cell mu0"); ax[1].set_ylabel("classification error")
ax[1].set_title("more photons -> lower error"); ax[1].legend()
# required collected-photon rate vs throughput goal
Rs = np.logspace(3, 6, 100)
ax[2].loglog(Rs, photons_required(0.2, 1e-2)*Rs, color="#F58518")
ax[2].axvline(1e5, ls=":", color="gray"); ax[2].text(1.1e5, 1e9, "goal 1e5/s", color="gray")
ax[2].set_xlabel("cells/s"); ax[2].set_ylabel("photons/s to collect (c=0.2, eps=1%)")
ax[2].set_title("faster sorting needs proportionally more light")
plt.tight_layout(); plt.show()"""),

md(r"""## Summary

- The $10^5$ cells/s goal fixes $t_{\rm cell}=10\ \mu\mathrm{s}$; at $M=128$ samples/cell the ADC runs
  at $12.8\ \mathrm{MS/s}$ ($\sim19\ \mathrm{MB/s}$ at 12 bit).
- Photon counting is Poisson, so $\mathrm{SNR}=\sqrt N$; classifying two types at contrast $c$ with
  error $\varepsilon$ needs $\mu_0=(2Q^{-1}(\varepsilon)/c)^2$ photons -- e.g. $c=0.2$, $\varepsilon=1\%$
  needs $\sim540$ photons, i.e. a collected rate of $\sim5\times10^7$ photons/s.
- The compute is only a few GFLOP/s -- far below hardware limits -- so the instrument is
  **photon-limited, not compute-limited**: throughput scales with how much light you collect per cell.

Subject-verb-object: the laser illuminates the cell; the optics collect photons; the ADC digitizes the
transit; the classifier decides; Poisson statistics set the floor. This is the statistical-physics
photon budget behind the sorter's throughput target."""),
]

write("cell", "throughput_budget", cells)
