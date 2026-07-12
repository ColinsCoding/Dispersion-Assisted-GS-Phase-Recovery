"""Generate notebooks/04_precalculus_review.ipynb."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from nbkit import md, co, frontmatter, setup_cell, section, write

cells = [
frontmatter("04", "Precalculus review: sinusoids, exponentials, logarithms", {
    "physics": "Oscillation and decay are the two universal time behaviours.",
    "mathematics": "Trigonometric identities, the exponential, the logarithm.",
    "algorithm": "Rewrite a phase-shifted sinusoid as a sum of sine and cosine (phasor coefficients).",
    "software": "SymPy for identities; NumPy for sampled waveforms.",
    "experiment": "A signal generator emits a sinusoid; a decade counter spans logarithmic frequency.",
    "engineering": "A direct-digital synthesizer generates a sinusoid from a phase accumulator.",
}),
setup_cell(),

section("English explanation"),
md("""Two functions describe almost all elementary physics in time: the **sinusoid**, for anything
that oscillates, and the **exponential**, for anything that grows or decays. Their inverse, the
**logarithm**, turns multiplicative spans (decades of frequency, orders of magnitude of intensity)
into additive ones, which is why spectra and gains are plotted in decibels and log scales. This
chapter fixes the identities the calculus and Fourier chapters will lean on.

Subject-verb-object: the oscillator repeats; the exponential decays; the logarithm linearizes."""),

section("Mathematical derivation"),
md("""The angle-addition identity $\\cos(\\omega t+\\phi)=\\cos\\phi\\cos\\omega t-\\sin\\phi\\sin\\omega t$
shows any phase-shifted sinusoid is a linear combination
$A\\cos(\\omega t+\\phi)=a\\cos\\omega t+b\\sin\\omega t$ with $a=A\\cos\\phi$, $b=-A\\sin\\phi$; the pair
$(a,b)$ is the phasor of chapter 03. The exponential obeys $e^{x+y}=e^x e^y$ and
$\\frac{d}{dx}e^{x}=e^{x}$; the logarithm inverts it, $\\ln(xy)=\\ln x+\\ln y$."""),

section("Dimensions and SI units"),
md("""The argument of any sinusoid, exponential, or logarithm must be **dimensionless**. In
$\\cos(\\omega t)$, the angular frequency $\\omega$ has units $\\mathrm{rad/s}$ so that $\\omega t$ is a
pure number; in $e^{-t/\\tau}$, the time constant $\\tau$ carries seconds. `physkit.units` confirms
$\\omega t$ is dimensionless."""),
co("""omega = U.Quantity(2*3.141592653589793*1e3, U.FREQUENCY)   # 1 kHz -> rad/s magnitude
t = U.Quantity(1e-3, U.TIME)
phase = omega * t
print("omega*t dimension:", phase.dim, "(must be dimensionless):", phase.dim.is_dimensionless)
assert phase.dim.is_dimensionless"""),

section("SymPy derivation"),
co("""t, w, phi, A = sp.symbols('t omega phi A', real=True)
expanded = sp.expand(sp.expand_trig(A*sp.cos(w*t + phi)))    # distribute A so coeff() extracts cleanly
a = expanded.coeff(sp.cos(w*t)); b = expanded.coeff(sp.sin(w*t))
print("A cos(w t + phi) = (%s) cos(w t) + (%s) sin(w t)" % (a, b))
assert sp.simplify(a - A*sp.cos(phi)) == 0 and sp.simplify(b + A*sp.sin(phi)) == 0
# amplitude recovered from the coefficients: a^2 + b^2 = A^2
assert sp.simplify(a**2 + b**2 - A**2) == 0
print("a^2 + b^2 = A^2  (phasor magnitude) verified")"""),

section("NumPy implementation"),
co("""fs, f0, tau = 2000.0, 50.0, 0.05
t = np.arange(0, 0.1, 1/fs)
osc = np.cos(2*np.pi*f0*t)
decay = np.exp(-t/tau)
damped = decay * osc
# recover A from a cos + b sin decomposition of a shifted sinusoid
A, phi = 2.0, np.deg2rad(30)
a, b = A*np.cos(phi), -A*np.sin(phi)
print(f"reconstructed amplitude sqrt(a^2+b^2) = {np.hypot(a,b):.4f} (A = {A})")"""),

section("Pandas tables"),
co("""decades = np.array([10, 100, 1000, 10000, 100000.0])
df = pd.DataFrame({"frequency_Hz": decades, "log10": np.log10(decades),
                   "period_ms": np.round(1e3/decades, 4)})
print(df.to_string(index=False))"""),

section("Matplotlib plots"),
co("""fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot(t*1e3, osc, label="cos"); ax[0].plot(t*1e3, decay, label="exp decay")
ax[0].plot(t*1e3, damped, label="damped osc")
ax[0].set_xlabel("time (ms)"); ax[0].set_title("oscillation and decay"); ax[0].legend()
ax[1].semilogx(decades, 20*np.log10(decades/decades[0]))
ax[1].set_xlabel("frequency (Hz, log)"); ax[1].set_ylabel("dB")
ax[1].set_title("logarithm linearizes decades")
plt.tight_layout(); plt.show()"""),

section("PyTorch (optional)"),
co("""if torch is not None:
    tt = torch.tensor(t)
    d = torch.exp(-tt/tau) * torch.cos(2*np.pi*f0*tt)
    print("torch damped max:", float(d.max()), "| numpy:", float(damped.max()))
else:
    print("PyTorch absent -- NumPy waveforms above are authoritative.")"""),

section("Exercises"),
md("""1. Derive the sum-to-product identity $\\cos\\alpha+\\cos\\beta=2\\cos\\frac{\\alpha+\\beta}{2}
   \\cos\\frac{\\alpha-\\beta}{2}$ and use it to explain acoustic beats.
2. Fit $a\\cos\\omega t+b\\sin\\omega t$ to a noisy sampled sinusoid by least squares and recover
   $A,\\phi$.
3. Show that a signal decaying as $e^{-t/\\tau}$ falls by $8.69\\,\\mathrm{dB}$ per time constant."""),

section("Engineering applications"),
md("""A **direct-digital synthesizer** builds a sinusoid by incrementing a phase accumulator and
reading a cosine lookup table -- the discrete form of $\\cos(\\omega t)$, with $\\omega$ set by the
increment. **Firmware** stores the table; a **DSP** demodulator projects a signal onto $\\cos\\omega t$
and $\\sin\\omega t$ (the $a,b$ above) to recover amplitude and phase. Logarithmic axes let a single
plot span the decades of frequency an instrument covers.

Summary (subject-verb-object): the accumulator advances the phase; the table yields the sinusoid; the
demodulator projects onto sine and cosine; the log scale compresses the decades."""),
]

write("04", "precalculus_review", cells)
