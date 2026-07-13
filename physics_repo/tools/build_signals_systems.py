"""Generate notebooks/signals_systems.ipynb."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from nbkit import md, co, setup_cell, write

cells = [
md(r"""# Signals and systems: delta, impulse response, filters, and time-stretch

A linear time-invariant (LTI) system is completely described by one signal: its **impulse response**
$h(t)$, the output when the input is a Dirac delta $\delta(t)$. Every other output follows by
**convolution**, $y=x*h$, and in the frequency domain that becomes a product,
$$Y(f)=X(f)\,H(f),\qquad H(f)=\mathcal F\{h\}.$$
$H(f)$ is the **transfer function**; its magnitude shapes the spectrum (a filter) and its phase delays
the components (dispersion). This notebook builds the chain -- delta, impulse response, convolution,
transfer function, filter -- and ends on the system this repository is built around: **photonic
time-stretch**, an all-pass LTI system with quadratic phase $H(f)=e^{\,j\pi D f^2}$ that leaves the
spectrum untouched but maps frequency to time.

Self-contained: NumPy, SymPy, Pandas, Matplotlib."""),
setup_cell(),

md(r"""## The Dirac delta and its sifting property

The delta $\delta(t)$ is the identity of convolution: $x*\delta=x$. Formally it is defined by the
sifting property $\int\delta(t-a)\,f(t)\,dt=f(a)$, which SymPy evaluates directly. Numerically it is a
limit of unit-area pulses that grow narrow and tall."""),
co("""t, a = sp.symbols('t a', real=True)
sift = sp.integrate(sp.DiracDelta(t - a) * sp.cos(t), (t, -sp.oo, sp.oo))
print("int delta(t-a) cos(t) dt =", sift, " (sifts out cos(a))")
assert sp.simplify(sift - sp.cos(a)) == 0
# numerical delta: a narrow Gaussian of unit area
def delta_approx(tg, width):
    d = np.exp(-tg**2/(2*width**2)) / (width*np.sqrt(2*np.pi))
    return d
tg = np.linspace(-5, 5, 4001); dt = tg[1]-tg[0]
for w in (1.0, 0.3, 0.1):
    print(f"width={w}: area = {np.trapezoid(delta_approx(tg,w), tg):.4f} (-> 1),"
          f" peak = {delta_approx(np.array([0.0]),w)[0]:.2f}")
assert np.isclose(np.trapezoid(delta_approx(tg,0.1), tg), 1.0, atol=1e-3)"""),

md(r"""## Impulse response and convolution: an RC low-pass filter

A first-order RC low-pass has impulse response $h(t)=\dfrac{1}{RC}e^{-t/RC}u(t)$ (causal). Its response
to any input is $y=x*h$. Feeding a unit step reproduces the familiar charging curve
$1-e^{-t/RC}$ -- computed here by numerical convolution and checked against the analytic result."""),
co("""RC = 1.0
tg = np.linspace(0, 10, 2000); dt = tg[1]-tg[0]
h = np.exp(-tg/RC)/RC                              # causal impulse response
step = np.ones_like(tg)                            # unit step input
y = np.convolve(step, h)[:len(tg)] * dt            # discrete convolution (scaled by dt)
analytic = 1 - np.exp(-tg/RC)                       # step response of an RC low-pass
print("max |numeric - analytic| step response =", np.max(np.abs(y - analytic)))
assert np.max(np.abs(y - analytic)) < 2e-2
# impulse response integrates to 1 (DC gain = 1)
assert np.isclose(np.trapezoid(h, tg), 1.0, atol=1e-3)"""),

md(r"""## Transfer function and frequency response

The transfer function is the Fourier transform of the impulse response,
$H(f)=\dfrac{1}{1+j2\pi fRC}$, a low-pass with $-3\ \mathrm{dB}$ corner at $f_c=1/(2\pi RC)$. The FFT of
the sampled $h(t)$ reproduces it."""),
co("""N = 4096; T = 40.0; dt = T/N
tg = np.arange(N)*dt
h = np.exp(-tg/RC)/RC
f = np.fft.rfftfreq(N, d=dt)
H_fft = np.fft.rfft(h)*dt                           # numerical transfer function
H_ana = 1/(1 + 1j*2*np.pi*f*RC)                     # analytic
err = np.max(np.abs(H_fft[:200] - H_ana[:200]))
print(f"corner frequency f_c = 1/(2 pi RC) = {1/(2*np.pi*RC):.4f} Hz")
print(f"max |H_fft - H_analytic| (low band) = {err:.3e}")
assert err < 2e-2
# at f_c the magnitude is 1/sqrt(2) (-3 dB)
fc = 1/(2*np.pi*RC); Hc = 1/(1+1j*2*np.pi*fc*RC)
assert np.isclose(abs(Hc), 1/np.sqrt(2), atol=1e-6)"""),

md(r"""## Filtering: removing a tone by its spectrum

A signal made of a low tone plus a high tone is passed through the low-pass. In the spectrum the high
tone sits above the corner and is attenuated; in time the fast wiggle disappears, leaving the slow
component. Multiplication in frequency ($Y=XH$) is the efficient way to filter."""),
co("""N = 4096; fs = 200.0; dt = 1/fs
tg = np.arange(N)*dt
x = np.sin(2*np.pi*2*tg) + 0.7*np.sin(2*np.pi*40*tg)   # 2 Hz + 40 Hz
f = np.fft.rfftfreq(N, d=dt)
RCf = 1/(2*np.pi*8)                                    # corner at 8 Hz
H = 1/(1 + 1j*2*np.pi*f*RCf)
y = np.fft.irfft(np.fft.rfft(x)*H, n=N)
# the 40 Hz component is strongly attenuated; the 2 Hz survives
Xmag = np.abs(np.fft.rfft(x)); Ymag = np.abs(np.fft.rfft(y))
i2, i40 = np.argmin(abs(f-2)), np.argmin(abs(f-40))
print(f"2 Hz amplitude:  in {Xmag[i2]:.0f} -> out {Ymag[i2]:.0f}  (kept)")
print(f"40 Hz amplitude: in {Xmag[i40]:.0f} -> out {Ymag[i40]:.0f}  (removed)")
assert Ymag[i40] < 0.3*Xmag[i40] and Ymag[i2] > 0.6*Xmag[i2]"""),

md(r"""## The repository's system: time-stretch as all-pass quadratic phase

Group-velocity dispersion is an LTI system with $H(f)=e^{\,j\pi D f^2}$. Its magnitude is **one**
(all-pass, so energy and spectrum are untouched), but its phase is quadratic, giving a group delay
$\tau_g(f)=-\frac{1}{2\pi}\frac{d\phi}{df}=-D f$ that is **linear in frequency**. A short pulse is
therefore spread in time, each frequency arriving at its own moment -- the frequency-to-time mapping
behind the dispersive Fourier transform and the photonic time-stretch ADC. In the strong-dispersion
limit the output envelope $|y(t)|^2$ is a scaled copy of the input **spectrum** $|X(f)|^2$."""),
co("""N = 8192; T = 200.0; dt = T/N
tg = (np.arange(N) - N//2)*dt
f = np.fft.fftfreq(N, d=dt)
tau0 = 1.5
x = np.exp(-tg**2/(2*tau0**2)).astype(complex)     # short Gaussian pulse
for D in (0.0, 20.0, 200.0):
    H = np.exp(1j*np.pi*D*f**2)                     # all-pass, quadratic phase
    y = np.fft.ifft(np.fft.fft(x)*H)
    width = 2*np.sqrt(np.sum(tg**2*np.abs(y)**2)/np.sum(np.abs(y)**2))   # rms-based width
    energy = np.sum(np.abs(y)**2)*dt
    print(f"D={D:6.1f}:  |H|=1? {np.allclose(np.abs(H),1)},  output width = {width:6.2f},"
          f"  energy = {energy:.3f} (conserved)")
# all-pass conserves energy for every D (Parseval, |H|=1)
E0 = np.sum(np.abs(x)**2)
for D in (20.0, 200.0):
    y = np.fft.ifft(np.fft.fft(x)*np.exp(1j*np.pi*D*f**2))
    assert np.isclose(np.sum(np.abs(y)**2), E0)     # energy conserved
    assert np.allclose(np.abs(np.exp(1j*np.pi*D*f**2)), 1.0)   # magnitude flat"""),

md(r"""## Plots"""),
co(r"""fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
# RC low-pass frequency response
f = np.fft.rfftfreq(4096, d=40.0/4096)
H = 1/(1 + 1j*2*np.pi*f*1.0)
ax[0].semilogx(f[1:], 20*np.log10(np.abs(H[1:])), color="#4C78A8")
ax[0].axvline(1/(2*np.pi), ls=":", color="gray"); ax[0].axhline(-3, ls=":", color="gray")
ax[0].set_xlabel("frequency (Hz)"); ax[0].set_ylabel("|H| (dB)")
ax[0].set_title("RC low-pass: -3 dB at f_c")
# filtering: spectrum before/after
N=4096; fs=200.0; tg=np.arange(N)/fs
x = np.sin(2*np.pi*2*tg)+0.7*np.sin(2*np.pi*40*tg)
fr = np.fft.rfftfreq(N, d=1/fs); Hf = 1/(1+1j*2*np.pi*fr*(1/(2*np.pi*8)))
y = np.fft.irfft(np.fft.rfft(x)*Hf, n=N)
ax[1].plot(fr, np.abs(np.fft.rfft(x)), color="#E45756", label="input")
ax[1].plot(fr, np.abs(np.fft.rfft(y)), color="#4C78A8", label="low-passed")
ax[1].set_xlim(0, 60); ax[1].set_xlabel("frequency (Hz)"); ax[1].set_ylabel("|X(f)|")
ax[1].set_title("filter removes the 40 Hz tone"); ax[1].legend()
# time-stretch: a pulse broadening with dispersion
Nt=8192; Tt=200.0; dt=Tt/Nt; tt=(np.arange(Nt)-Nt//2)*dt; ff=np.fft.fftfreq(Nt, d=dt)
xp = np.exp(-tt**2/(2*1.5**2)).astype(complex)
for D,c in [(0.0,"#4C78A8"),(60.0,"#54A24B"),(200.0,"#E45756")]:
    yp = np.fft.ifft(np.fft.fft(xp)*np.exp(1j*np.pi*D*ff**2))
    ax[2].plot(tt, np.abs(yp)**2, color=c, label=f"D={D:.0f}")
ax[2].set_xlim(-60, 60); ax[2].set_xlabel("time"); ax[2].set_ylabel("|y(t)|^2")
ax[2].set_title("dispersion stretches the pulse (energy conserved)"); ax[2].legend()
plt.tight_layout(); plt.show()""" ),

md(r"""## Summary

- An LTI system is one signal, the **impulse response** $h(t)$: $y=x*h$, and $Y(f)=X(f)H(f)$.
- The **Dirac delta** is the convolution identity (sifting $\int\delta(t-a)f\,dt=f(a)$); a numerical
  delta is a unit-area pulse taken narrow.
- A **filter** shapes $|H(f)|$: the RC low-pass has $H=1/(1+j2\pi fRC)$, $-3\ \mathrm{dB}$ at
  $f_c=1/(2\pi RC)$, and removes tones above the corner.
- **Time-stretch/dispersion** is the all-pass $H=e^{\,j\pi Df^2}$: magnitude one (spectrum and energy
  preserved), quadratic phase, linear group delay $-Df$ -- the frequency-to-time map at the heart of
  the dispersive Fourier transform.

Subject-verb-object: the delta probes the system; the impulse response defines it; convolution
produces the output; the transfer function filters the spectrum; dispersion stretches the pulse. This
is the signals-and-systems language every later instrument chapter speaks."""),
]

write("signals", "systems", cells)
