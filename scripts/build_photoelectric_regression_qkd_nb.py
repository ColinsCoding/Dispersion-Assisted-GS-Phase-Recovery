"""Build notebooks/photoelectric_regression_qkd.ipynb -- "The Photoelectric
Effect Is y = mx + b: From Planck's Constant to Quantum Key Distribution".

The photoelectric equation eV_stop = hf - phi is a straight line, so
measuring Planck's constant is linear regression. The notebook fits
simulated mercury-lamp data three ways (SymPy closed form, PyTorch
gradient descent on CUDA when available, standard errors), contrasts the
two-parameter physics model with small neural networks (18-24 hidden
units), and then follows the photon idea into quantum communication: a
Born-rule BB84 simulation with fibre loss, noise and an eavesdropper.

Needs: sympy, numpy, scipy, matplotlib, torch.

Build:   py -3.13 scripts/build_photoelectric_regression_qkd_nb.py
Execute: py -3.13 -m jupyter nbconvert --to notebook --execute --inplace
         notebooks/photoelectric_regression_qkd.ipynb
(or: make notebook-photoelectric-qkd)
"""
import pathlib
import nbformat as nbf

nb = nbf.v4.new_notebook()
md = lambda s: nbf.v4.new_markdown_cell(s)
co = lambda s: nbf.v4.new_code_cell(s)
cells = []

cells.append(md(r"""# The Photoelectric Effect Is $y = mx + b$

Light shining on a metal knocks electrons out of it. Einstein's 1905
explanation — light arrives in packets (photons) of energy $hf$ — gives

$$eV_\text{stop} = K_\text{max} = hf - \phi
\qquad\Longleftrightarrow\qquad
\underbrace{V_\text{stop}}_{y} = \underbrace{\tfrac{h}{e}}_{m}\,\underbrace{f}_{x}\;\underbrace{-\,\tfrac{\phi}{e}}_{b}.$$

So measuring Planck's constant is a **linear regression**: the slope is
$h/e$ and the intercept is minus the work function. This notebook:

| Part | What | Tools |
|---|---|---|
| 1 | The physics and a simulated mercury-lamp experiment | NumPy |
| 2 | Least squares: derive it, then solve it | SymPy |
| 3 | The same fit by gradient descent — machine learning's way | PyTorch (CUDA if available) |
| 4 | Error bars on $h$ and $\phi$ | statistics |
| 5 | A neural network with 18–24 hidden units, for contrast | PyTorch |
| 6 | One photon at a time: quantum key distribution (BB84) | PyTorch |

This previews the start of Chapter 38 (quantization of light). The
measurements are **simulated**, with stated noise; the constants are
CODATA values."""))

cells.append(co(r"""import sys

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import torch
from scipy.constants import c, e, h
from scipy.optimize import brentq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("python:", sys.executable)
print("torch :", torch.__version__, "| device:", device,
      "-", torch.cuda.get_device_name(0) if device.type == "cuda" else "no CUDA in this kernel")
torch.manual_seed(0)
rng = np.random.default_rng(38)
eV = e                                         # 1 eV in joules
print(f"h = {h:.6e} J s,  h/e = {h / e:.6e} V s")"""))

# ================================================================ PART 1
cells.append(md(r"""# Part 1 — The physics, and a simulated experiment

Classical waves predicted that brighter light should eject faster
electrons, and that any colour would work if you waited long enough.
Experiments said otherwise:

| Observation | Photon explanation |
|---|---|
| Below a **threshold frequency** $f_0$, no electrons at all, however bright | one photon must carry at least $\phi$; $f_0 = \phi/h$ |
| The **maximum** electron energy depends on frequency, not brightness | each electron absorbs one photon: $K_\text{max} = hf - \phi$ |
| Brighter light gives **more** electrons, not faster ones | more photons per second |
| Emission starts instantly | no energy needs to accumulate |

The stopping voltage $V_\text{stop}$ is the reverse (DC) voltage that
just stops the fastest electrons: $eV_\text{stop} = K_\text{max}$.

**Simulated experiment.** A mercury lamp supplies five well-known
spectral lines. The cathode has an illustrative work function of
2.00 eV. Each line is measured three times with 20 mV of noise."""))

cells.append(co(r"""phi_true_eV = 2.00
lines_nm = np.array([365.0, 404.7, 435.8, 546.1, 577.0])          # mercury lines
line_freq = c / (lines_nm * 1e-9)
photon_eV = h * line_freq / eV

print(" line (nm)   f (1e14 Hz)   photon energy (eV)   electrons?")
for wavelength, frequency, energy in zip(lines_nm, line_freq, photon_eV):
    print(f"  {wavelength:6.1f}      {frequency / 1e14:6.3f}          {energy:5.3f}             "
          f"{'yes' if energy > phi_true_eV else 'no'}")
hene = h * c / 632.8e-9 / eV
print(f"  632.8 (HeNe laser)          {hene:5.3f}             {'yes' if hene > phi_true_eV else 'no -- below threshold'}")

repeats, noise_V = 3, 0.020
f_data = np.repeat(line_freq, repeats)
V_data = h * f_data / e - phi_true_eV + rng.normal(0, noise_V, f_data.size)
print(f"\n{f_data.size} measurements of the stopping voltage, from {V_data.min():.3f} V to {V_data.max():.3f} V")
assert hene < phi_true_eV and np.all(photon_eV > phi_true_eV)"""))

# ================================================================ PART 2
cells.append(md(r"""# Part 2 — Least squares, derived

"Best line" means the $m$ and $b$ that minimise the sum of squared
misses,

$$S(m, b) = \sum_i \big(y_i - m x_i - b\big)^2 .$$

$S$ is a bowl-shaped (quadratic) function of $m$ and $b$, so its
minimum is where both partial derivatives vanish. Expanding $S$, it
depends on the data only through five sums: $n$, $\sum x$, $\sum y$,
$\sum x^2$, $\sum xy$ (and $\sum y^2$, which drops out)."""))

cells.append(co(r"""m, b = sp.symbols("m b", real=True)
n, Sx, Sy, Sxx, Sxy, Syy = sp.symbols("n S_x S_y S_xx S_xy S_yy", real=True)

# S(m, b) written with the data sums
S = Syy - 2 * m * Sxy - 2 * b * Sy + m**2 * Sxx + 2 * m * b * Sx + n * b**2
best = sp.solve([sp.diff(S, m), sp.diff(S, b)], [m, b], dict=True)[0]
print("m ="); sp.pprint(sp.factor(best[m]))
print("\nb ="); sp.pprint(sp.factor(best[b]))

sums = {n: f_data.size, Sx: f_data.sum(), Sy: V_data.sum(), Sxx: (f_data**2).sum(), Sxy: (f_data * V_data).sum()}
slope = float(best[m].subs(sums))
intercept = float(best[b].subs(sums))
print(f"\nslope     m = {slope:.5e} V s   ->  h = m e = {slope * e:.4e} J s")
print(f"intercept b = {intercept:.4f} V     ->  phi = -b = {-intercept:.3f} eV")
assert np.allclose([slope, intercept], np.polyfit(f_data, V_data, 1), rtol=1e-9)"""))

# ================================================================ PART 3
cells.append(md(r"""# Part 3 — The same fit by gradient descent

Machine learning rarely solves for the minimum directly. It starts
somewhere, measures the slope of the error surface (the gradient), and
steps downhill — the method that scales to models with millions of
parameters, where no formula exists.

One practical detail matters a lot: frequencies are around $10^{15}$
and voltages around 1. Gradient descent on such mismatched scales
crawls. **Standardising** the input (subtract the mean, divide by the
standard deviation) makes the bowl round, and a few hundred steps
suffice. The answer is converted back afterwards."""))

cells.append(co(r"""x = torch.tensor(f_data, dtype=torch.float64, device=device)
y = torch.tensor(V_data, dtype=torch.float64, device=device)
x_mean, x_std = x.mean(), x.std()
x_scaled = (x - x_mean) / x_std

w = torch.zeros(1, dtype=torch.float64, device=device, requires_grad=True)     # slope on the scaled axis
w0 = torch.zeros(1, dtype=torch.float64, device=device, requires_grad=True)    # intercept on the scaled axis
learning_rate, history = 0.1, []
for step in range(500):
    loss = torch.mean((w * x_scaled + w0 - y) ** 2)
    loss.backward()
    with torch.no_grad():
        w -= learning_rate * w.grad
        w0 -= learning_rate * w0.grad
        w.grad.zero_(); w0.grad.zero_()
    history.append(loss.item())

slope_gd = (w / x_std).item()
intercept_gd = (w0 - w * x_mean / x_std).item()
print(f"gradient descent on {device}:  m = {slope_gd:.5e} V s,  b = {intercept_gd:.4f} V")
print(f"closed form             :  m = {slope:.5e} V s,  b = {intercept:.4f} V")
assert abs(slope_gd / slope - 1) < 1e-8 and abs(intercept_gd - intercept) < 1e-8

plt.figure(figsize=(6, 3.2))
plt.semilogy(np.array(history) - min(history) + 1e-16)
plt.xlabel("gradient-descent step"); plt.ylabel("loss above its minimum (V$^2$)")
plt.title("Standardised inputs: the loss drops geometrically"); plt.tight_layout(); plt.show()"""))

# ================================================================ PART 4
cells.append(md(r"""# Part 4 — Error bars: how well do we know $h$?

The scatter of the points about the line estimates the noise,
$s^2 = \sum r_i^2/(n-2)$ (two parameters were fitted, so two degrees of
freedom are used up). Standard least-squares results then give

$$\sigma_m^2 = \frac{n\,s^2}{n\sum x^2 - (\sum x)^2}, \qquad
\sigma_b^2 = \frac{s^2 \sum x^2}{n\sum x^2 - (\sum x)^2}.$$"""))

cells.append(co(r"""residuals = V_data - (slope * f_data + intercept)
n_pts = f_data.size
s2 = np.sum(residuals**2) / (n_pts - 2)
denominator = n_pts * np.sum(f_data**2) - np.sum(f_data)**2
sigma_m = np.sqrt(n_pts * s2 / denominator)
sigma_b = np.sqrt(s2 * np.sum(f_data**2) / denominator)

h_fit, h_sigma = slope * e, sigma_m * e
f0_fit = -intercept / slope
print(f"noise estimated from residuals: {np.sqrt(s2) * 1e3:.1f} mV  (simulated with {noise_V * 1e3:.0f} mV)")
print(f"h   = ({h_fit:.4e} +/- {h_sigma:.1e}) J s    CODATA {h:.4e}   ->  {(h_fit - h) / h_sigma:+.2f} sigma")
print(f"phi = ({-intercept:.3f} +/- {sigma_b:.3f}) eV       true {phi_true_eV:.3f}")
print(f"threshold f0 = -b/m = {f0_fit / 1e14:.3f} x 1e14 Hz  ->  {c / f0_fit * 1e9:.0f} nm")
assert abs(h_fit - h) < 3 * h_sigma and abs(-intercept - phi_true_eV) < 3 * sigma_b"""))

# ================================================================ PART 5
cells.append(md(r"""# Part 5 — A neural network for contrast

A small neural network — one hidden layer of 18 to 24 tanh units — can
also fit fifteen points. It has 55 to 73 adjustable numbers instead of
two. Fit each size to the same data and compare:

- how well it matches the measurements (training error);
- what it predicts at 250 nm, an ultraviolet frequency *outside* the
  measured range, where the photon model gives $hf/e - \phi/e$;
- whether any of its parameters is Planck's constant."""))

cells.append(co(r"""def train_network(hidden, seed):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(1, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 1))
    net = net.to(device=device, dtype=torch.float64)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    inputs, targets = x_scaled[:, None], y[:, None]
    for _ in range(3000):
        optimizer.zero_grad()
        loss = torch.mean((net(inputs) - targets) ** 2)
        loss.backward()
        optimizer.step()
    return net


def predict_net(net, frequencies):
    with torch.no_grad():
        tensor = torch.tensor(frequencies, dtype=torch.float64, device=device)
        return net(((tensor - x_mean) / x_std)[:, None]).squeeze(1).cpu().numpy()


f_uv = c / 250e-9
truth_uv = h * f_uv / e - phi_true_eV
line_uv = slope * f_uv + intercept
line_rms = np.sqrt(np.mean(residuals**2))
print(f"model                parameters   training RMS (mV)   V_stop at 250 nm (truth {truth_uv:.3f} V)")
print(f"straight line              2          {line_rms * 1e3:6.1f}            {line_uv:.3f} V")
networks = {}
for hidden in (18, 20, 22, 24):
    net = train_network(hidden, seed=hidden)
    networks[hidden] = net
    n_params = sum(p.numel() for p in net.parameters())
    rms = np.sqrt(np.mean((predict_net(net, f_data) - V_data) ** 2))
    print(f"network, {hidden} hidden        {n_params:3d}          {rms * 1e3:6.1f}            {predict_net(net, [f_uv])[0]:.3f} V")
assert abs(line_uv - truth_uv) < 0.05"""))

cells.append(co(r"""f_axis = np.linspace(4.0e14, 12.5e14, 400)
plt.figure(figsize=(8, 4))
plt.plot(f_data / 1e14, V_data, "ko", ms=4, label="simulated measurements")
plt.plot(f_axis / 1e14, h * f_axis / e - phi_true_eV, color="0.6", lw=4, alpha=0.5, label="true photon model")
plt.plot(f_axis / 1e14, slope * f_axis + intercept, "C0", label="fitted line (2 parameters)")
for hidden, net in networks.items():
    plt.plot(f_axis / 1e14, predict_net(net, f_axis), "--", lw=1, label=f"network, {hidden} hidden")
plt.axvspan(f_data.min() / 1e14, f_data.max() / 1e14, color="C2", alpha=0.08, label="measured range")
plt.axhline(0, color="0.8", lw=0.8)
plt.xlabel("frequency (1e14 Hz)"); plt.ylabel("stopping voltage (V)")
plt.title("Inside the data every model agrees; outside it only the physics does")
plt.legend(fontsize=7, loc="upper left"); plt.tight_layout(); plt.show()"""))

cells.append(md(r"""Read the table and the plot together. Inside the measured range the
networks match the data about as well as the line, often slightly
better — they are flexible enough to follow the noise. Outside it they
have no reason to stay straight, and they don't. And none of their
dozens of numbers is $h$ or $\phi$: they *describe* these fifteen
points; the line *explains* them.

That is the general lesson for machine learning in physics: a model
that builds in the known structure needs less data, extrapolates
correctly, and returns parameters with physical meaning. A flexible
model is the right tool when the structure is unknown — and the wrong
one when a two-parameter law is sitting in the textbook.

# Part 6 — One photon at a time: quantum communication

The photoelectric effect says light is delivered in indivisible
quanta. Quantum communication uses exactly that: information carried
by *single* photons, which cannot be split or copied.

**How few photons is "single"?** Telecom fibre carries light at
1550 nm."""))

cells.append(co(r"""photon_1550_eV = h * c / 1550e-9 / eV
rate = 1e-9 / (photon_1550_eV * eV)
print(f"one 1550 nm photon carries {photon_1550_eV:.3f} eV")
print(f"a 1 nW beam is {rate:.2e} photons every second")
print(f"that photon cannot free an electron from the {phi_true_eV:.2f} eV cathode of Part 1 --")
print("which is why telecom single-photon detectors use narrow-gap semiconductors such as InGaAs (gap about 0.75 eV)")
assert photon_1550_eV < phi_true_eV"""))

cells.append(md(r"""## BB84: a key that reveals eavesdroppers

Alice wants to share a secret random key with Bob. For each bit she
sends one photon polarised in one of two **bases**:

| Basis | bit 0 | bit 1 |
|---|---|---|
| rectilinear (+) | 0° | 90° |
| diagonal (×) | 45° | 135° |

Bob measures each photon in a randomly chosen basis. The quantum rule
for a single photon meeting a polariser is the **Born rule** — the
one-photon version of Malus's law: a photon polarised at angle
$\theta$ passes a polariser at angle $\alpha$ with probability
$\cos^2(\theta - \alpha)$.

- Same basis → Bob gets Alice's bit with certainty ($\cos^2 0 = 1$).
- Different basis → Bob gets a coin flip ($\cos^2 45^\circ = 1/2$).

Afterwards they compare *bases* publicly (not bits) and keep only the
matching ones: the **sifted key**. They sacrifice a sample to measure
the **quantum bit error rate** (QBER).

An eavesdropper, Eve, cannot copy an unknown photon. If she measures
each one and sends a replacement, half the time she guesses the wrong
basis, scrambles the photon, and causes an error with probability 1/2:
QBER $= \tfrac12 \times \tfrac12 = 25\%$. **Listening leaves
fingerprints.**

The simulation below uses the Born rule directly, with 100,000 photons,
a 50 km fibre (0.2 dB/km loss, so 10 % of photons arrive), and optional
detector noise."""))

cells.append(co(r"""def measure(angle_deg, basis, generator):
    # Born rule: a photon at angle_deg meets a polariser at basis*45 deg; outcome 0 with prob cos^2(difference)
    p_zero = torch.cos(torch.deg2rad(angle_deg - 45.0 * basis)) ** 2
    return torch.rand(angle_deg.shape, generator=generator, device=angle_deg.device, dtype=angle_deg.dtype) >= p_zero


def bb84(n_photons, eve_present, detector_error, distance_km, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    coin = lambda: torch.rand(n_photons, generator=generator, device=device) < 0.5
    alice_bit, alice_basis = coin(), coin()
    angle = 45.0 * alice_basis.double() + 90.0 * alice_bit.double()
    if eve_present:                                              # intercept and resend
        eve_basis = coin()
        eve_bit = measure(angle, eve_basis.double(), generator)
        angle = 45.0 * eve_basis.double() + 90.0 * eve_bit.double()
    transmittance = 10 ** (-0.2 * distance_km / 10)
    arrived = torch.rand(n_photons, generator=generator, device=device) < transmittance
    bob_basis = coin()
    bob_bit = measure(angle, bob_basis.double(), generator)
    bob_bit ^= torch.rand(n_photons, generator=generator, device=device) < detector_error
    sifted = arrived & (alice_basis == bob_basis)
    errors = (alice_bit[sifted] != bob_bit[sifted]).double()
    return sifted.sum().item(), errors.mean().item()


N_PHOTONS = 100_000
print(f"{N_PHOTONS:,} photons over 50 km of fibre\n")
print("scenario                         sifted bits    QBER")
qber = {}
for label, eve_present, noise in [("quiet line", False, 0.0), ("2 % detector noise", False, 0.02),
                                  ("Eve intercepts everything", True, 0.0)]:
    kept, error_rate = bb84(N_PHOTONS, eve_present, noise, 50.0, seed=7)
    qber[label] = (kept, error_rate)
    print(f"{label:30s}  {kept:8,d}      {error_rate * 100:5.2f} %")

kept_eve, q_eve = qber["Eve intercepts everything"]
assert qber["quiet line"][1] == 0.0
assert abs(q_eve - 0.25) < 4 * np.sqrt(0.25 * 0.75 / kept_eve)
assert abs(qber["2 % detector noise"][1] - 0.02) < 0.01"""))

cells.append(md(r"""About 5 % of the photons survive both the fibre loss (10 %) and the
basis comparison (50 %). The quiet line has no errors; detector noise
gives a small QBER; Eve gives about 25 %.

**How much error is tolerable?** Errors are also what an eavesdropper
would cause, so Alice and Bob must assume the worst and spend part of
the key to correct errors and erase whatever Eve might know. For BB84
the secure fraction that remains is (Shor–Preskill)

$$r = 1 - 2H_2(Q), \qquad H_2(Q) = -Q\log_2 Q - (1-Q)\log_2(1-Q),$$

which reaches zero at a QBER of about 11 %."""))

cells.append(co(r"""H2 = lambda q: -q * np.log2(q) - (1 - q) * np.log2(1 - q)
secure_fraction = lambda q: 1 - 2 * H2(q)
threshold = brentq(secure_fraction, 0.01, 0.2)
print(f"secure key fraction falls to zero at QBER = {threshold * 100:.1f} %")
for label, (kept, error_rate) in qber.items():
    fraction = secure_fraction(error_rate) if 0 < error_rate < 0.5 else (1.0 if error_rate == 0 else 0.0)
    print(f"  {label:28s} QBER {error_rate * 100:5.2f} %  ->  "
          f"{max(fraction, 0) * kept:7,.0f} secret bits from {kept:,} sifted")
assert 0.105 < threshold < 0.115

q_axis = np.linspace(0.001, 0.2, 400)
plt.figure(figsize=(6.5, 3.4))
plt.plot(q_axis * 100, np.maximum(secure_fraction(q_axis), 0))
plt.axvline(threshold * 100, color="C3", ls="--", label=f"limit {threshold * 100:.1f} %")
plt.axvline(25, color="0.5", ls=":", label="intercept-resend Eve (25 %)")
plt.xlabel("QBER (%)"); plt.ylabel("secret bits per sifted bit")
plt.title("BB84: tolerate noise, detect eavesdropping"); plt.legend(); plt.tight_layout(); plt.show()"""))

cells.append(md(r"""# Summary

| Idea | Equation | What it gave here |
|---|---|---|
| Photoelectric effect | $eV_\text{stop} = hf - \phi$ | a straight line in $f$ |
| Least squares | $\partial S/\partial m = \partial S/\partial b = 0$ | $h$ to within its error bar |
| Gradient descent | step against the gradient | the same answer, on the GPU, after standardising |
| Error bars | $\sigma_m^2 = ns^2/(n\sum x^2 - (\sum x)^2)$ | $h$ with a stated uncertainty |
| Neural network | 55–73 parameters | fits the points, but neither extrapolates reliably nor yields $h$ |
| Born rule | $P = \cos^2(\theta - \alpha)$ | the single-photon Malus's law |
| BB84 | QBER, $r = 1 - 2H_2(Q)$ | 0 % quiet, about 25 % with an eavesdropper, secure below 11 % |

**What is simulated.** The stopping voltages, the photons and the
eavesdropper are generated by the code. The BB84 model is the textbook
protocol with ideal single photons; real systems use attenuated laser
pulses and need extra countermeasures (decoy states) that are not
modelled here."""))

nb["cells"] = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "photoelectric_regression_qkd.ipynb"
out.write_text(nbf.writes(nb), encoding="utf-8")
print(f"wrote {out}, {len(cells)} cells")
