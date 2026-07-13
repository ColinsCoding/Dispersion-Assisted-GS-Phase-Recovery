"""Generate notebooks/dispersion_assisted_phase_recovery.ipynb."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from nbkit import md, co, setup_cell, write

cells = [
md(r"""# Dispersion-assisted Gerchberg-Saxton phase recovery

A detector measures **intensity**, $|y(t)|^2$; the phase of the field is thrown away. Recovering that
phase from magnitude alone is *phase retrieval*, and it is ill-posed: many different signals share the
same magnitude (global phase, conjugate/twin, and translation ambiguities). The idea this repository
is named for is to **assist** the recovery with known **dispersion**.

Dispersion is the all-pass LTI system $H_D(f)=e^{\,j\pi D f^2}$ from the signals-and-systems notebook:
it does not change the spectrum, only reshuffles the phase, so each amount of dispersion $D$ produces a
*different* intensity pattern $I_D(t)=\big|\mathcal F^{-1}\{X(f)\,H_D(f)\}\big|^2$ of the *same* signal.
Measuring $I_D$ at several $D$ gives **diversity**: a set of magnitude constraints that, together,
pin down the phase. A Gerchberg-Saxton alternating projection enforces all of them at once and
reconstructs the field.

Self-contained: NumPy, SymPy, Pandas, Matplotlib."""),
setup_cell(),

md(r"""## The forward model

Take a complex test field $x(t)$ with non-trivial amplitude *and* phase. For a chosen set of
dispersions $\{D_k\}$, the measured intensities are $I_k=|\mathcal F^{-1}\{X H_{D_k}\}|^2$ with
$X=\mathcal F\{x\}$ and $H_{D_k}(f)=e^{\,j\pi D_k f^2}$. Because $|H_{D_k}|=1$, every measurement carries
the same energy -- dispersion only redistributes it in time."""),
co("""N = 256
t = (np.arange(N) - N//2).astype(float)
f = np.fft.fftfreq(N)
rng = np.random.default_rng(0)
# test field: two Gaussian lobes with a quadratic (chirp) phase -- amplitude AND phase structure
amp = np.exp(-(t+18)**2/(2*7**2)) + 0.8*np.exp(-(t-14)**2/(2*5**2))
x_true = amp * np.exp(1j*(0.03*t + 0.0008*t**2))
X_true = np.fft.fft(x_true)

def disperse(X, D):
    return np.fft.ifft(X * np.exp(1j*np.pi*D*f**2))

# Both-sign dispersion: because H_D(f)=exp(j pi D f^2) is even in f, using only D>=0 leaves a
# conjugate/twin symmetry that phase retrieval cannot break; negative D removes it.
D_set = [-40.0, 0.0, 20.0, 60.0]                      # dispersion diversity (both signs)
support = np.abs(t) < 40                              # known time window holding the field
I = [np.abs(disperse(X_true, D))**2 for D in D_set]   # the measurements
for D, Ik in zip(D_set, I):
    print(f"D={D:6.1f}:  measured energy = {Ik.sum():.4f}")
assert np.allclose([Ik.sum() for Ik in I], I[0].sum())   # all-pass conserves energy"""),

md(r"""## Why one measurement is not enough

A single intensity fixes only the magnitude in that plane; the phase is free. We quantify "recovered
up to global phase" with the phase-invariant distance
$d(a,b)=\sqrt{\|a\|^2+\|b\|^2-2|\langle a,b\rangle|}$ (zero iff $a,b$ differ only by a global phase --
the gauge freedom from the ambiguities notebook)."""),
co("""def phase_invariant_distance(a, b):
    aa = np.vdot(a, a).real; bb = np.vdot(b, b).real
    return np.sqrt(max(aa + bb - 2*abs(np.vdot(a, b)), 0.0))

# a field with the correct D=0 magnitude but random phase reproduces I[0] yet is wrong
bad = np.sqrt(I[0]) * np.exp(1j*rng.uniform(0, 2*np.pi, N))
print("reproduces I[0]? ", np.allclose(np.abs(bad)**2, I[0]))
print("but distance to the true field (mod global phase) =",
      round(phase_invariant_distance(bad, x_true), 3), " (large -> wrong)")
assert phase_invariant_distance(bad, x_true) > 1.0"""),

md(r"""## Gerchberg-Saxton with dispersion diversity

Alternating projection: hold an estimate of the spectrum $X$. For each plane $k$, propagate to that
dispersion, **replace the magnitude** with the measured $\sqrt{I_k}$ while keeping the current phase,
propagate back (multiply by $H_{D_k}^{*}$ since $|H|=1$), and average the resulting spectra over all
planes. Iterating drives the estimate toward a field consistent with *every* measurement at once."""),
co("""def gs_once(I_list, D_list, n_iter, seed):
    r = np.random.default_rng(seed)
    X = np.fft.fft(np.sqrt(I_list[0]) * np.exp(1j*r.uniform(0, 2*np.pi, N)))   # random phase start
    hist = []
    for _ in range(n_iter):
        for Ik, D in zip(I_list, D_list):                     # serial magnitude projections
            Hk = np.exp(1j*np.pi*D*f**2)
            y = np.fft.ifft(X * Hk)
            y = np.sqrt(Ik) * np.exp(1j*np.angle(y))          # enforce measured magnitude
            X = np.fft.fft(y) * np.conj(Hk)                   # back-propagate (|H|=1)
        xo = np.fft.ifft(X); xo[~support] = 0.0; X = np.fft.fft(xo)   # support projection
        resid = sum(np.sum((np.abs(np.fft.ifft(X*np.exp(1j*np.pi*D*f**2))) - np.sqrt(Ik))**2)
                    for Ik, D in zip(I_list, D_list))
        hist.append(np.sqrt(resid))
    return np.fft.ifft(X), hist

def gs_recover(I_list, D_list, n_iter=600, restarts=24):
    # phase retrieval is non-convex: take the best of several random restarts
    best_x, best_hist, best_r = None, None, np.inf
    for s in range(restarts):
        xr, h = gs_once(I_list, D_list, n_iter, seed=s)
        if h[-1] < best_r:
            best_r, best_x, best_hist = h[-1], xr, h
    return best_x, best_hist

x_rec, hist = gs_recover(I, D_set)
d = phase_invariant_distance(x_rec, x_true)
print(f"final measurement residual = {hist[-1]:.2e}")
print(f"phase-invariant distance to true field = {d:.2e}  (recovered up to global phase)")
assert d < 0.05                                          # ~1% of ||x||: the field is recovered"""),

md(r"""## Diversity is what breaks the ambiguity

Repeat the recovery with a growing number of dispersion planes. One plane leaves the phase
under-determined; adding dispersed measurements collapses the reconstruction error toward zero. The
dispersion is the *assist*: it manufactures the extra magnitude constraints that a single detector
cannot provide."""),
co("""rows = []; dist = {}
for k in (1, 2, 3, 4):
    xr, h = gs_recover(I[:k], D_set[:k], n_iter=400, restarts=8)
    dist[k] = phase_invariant_distance(xr, x_true)
    rows.append({"# dispersion planes": k, "planes D": D_set[:k],
                 "final residual": f"{h[-1]:.2e}", "distance to true": f"{dist[k]:.2e}"})
df = pd.DataFrame(rows)
print(df.to_string(index=False))
# adding dispersion planes improves the recovery
assert dist[4] < dist[1]"""),

md(r"""## Plots"""),
co(r"""# align the recovered field to the true one's global phase for display
ph = np.angle(np.vdot(x_rec, x_true)); x_disp = x_rec*np.exp(1j*ph)
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
ax[0].plot(t, np.abs(x_true), color="#4C78A8", lw=2, label="true |x|")
ax[0].plot(t, np.abs(x_disp), "--", color="#E45756", label="recovered")
ax[0].set_xlabel("t"); ax[0].set_ylabel("amplitude"); ax[0].legend(); ax[0].set_title("amplitude")
m = np.abs(x_true) > 0.05                        # show phase only where there is signal
ax[1].plot(t[m], np.unwrap(np.angle(x_true))[m], color="#4C78A8", lw=2, label="true phase")
ax[1].plot(t[m], np.unwrap(np.angle(x_disp))[m], "--", color="#E45756", label="recovered")
ax[1].set_xlabel("t"); ax[1].set_ylabel("phase (rad)"); ax[1].legend(); ax[1].set_title("phase (the hard part)")
ax[2].semilogy(hist, color="#54A24B")
ax[2].set_xlabel("Gerchberg-Saxton iteration"); ax[2].set_ylabel("measurement residual")
ax[2].set_title("convergence with 4 dispersion planes")
plt.tight_layout(); plt.show()""" ),

md(r"""## Summary

- A detector measures $|y|^2$ and discards phase; recovering it from one magnitude is ambiguous
  (global phase, twin, shift).
- **Dispersion** $H_D(f)=e^{\,j\pi D f^2}$ is all-pass: it preserves spectrum and energy but produces a
  different intensity pattern for each $D$ -- **measurement diversity** at no cost in light.
- A multi-plane **Gerchberg-Saxton** loop enforces every dispersed magnitude at once and reconstructs
  the field to the global-phase gauge; the reconstruction error falls as more dispersion planes are
  added.
- The instrument realizes this directly: a dispersive fiber (or the time-stretch stage) applies the
  known $D$, the ADC records $I_D$, and GPU FFTs run the GS iteration -- the dispersion-assisted
  phase-recovery pipeline this repository is built around.

Subject-verb-object: dispersion diversifies the measurement; the detector records intensities;
Gerchberg-Saxton enforces the magnitudes; the algorithm recovers the phase."""),
]

write("dispersion_assisted", "phase_recovery", cells)
