"""Generate notebooks/spectral_initialization.ipynb."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from nbkit import md, co, setup_cell, write

cells = [
md(r"""# Spectral initialization for global phase recovery

The gradient-descent notebook ended on a warning: from a *random* start, gradient descent on the
non-convex phase-retrieval loss can settle in a **spurious** minimum -- low loss, wrong field. The fix
is **spectral initialization**: instead of starting from noise, start from an estimate built directly
from the measurements that already lands in the correct basin. Gradient descent from there converges
to the true field -- this is the **Wirtinger Flow** algorithm (Candes-Li-Soltanolkotabi, 2015), the
method that made non-convex phase retrieval reliable.

The estimate is an eigenvector. For measurements $y_i=|\langle a_i,x\rangle|^2$, form the data matrix
$$Y=\frac1m\sum_i y_i\,a_i a_i^{H}.$$
Its expectation is $\mathbb E[Y]=\lVert x\rVert^2 I + x x^{H}$, whose top eigenvector is $x/\lVert x
\rVert$. So the **leading eigenvector of $Y$**, scaled by $\lambda=\sqrt{\mathrm{mean}(y)}\approx\lVert
x\rVert$, points at the true field. Spectral init needs *incoherent* (random-like) measurements; we
demonstrate it on the standard random-Gaussian model, then connect back to the dispersion instrument.

Self-contained: NumPy, SymPy, Pandas, Matplotlib."""),
setup_cell(),

md(r"""## The measurement model and the data matrix

Random complex-Gaussian sensing vectors $a_i$ (rows of $A$), $m$ intensity measurements
$y=|Ax|^2$. Near the information limit $m\approx4N$, the difference between a good and a bad start is
the difference between success and failure."""),
co("""rng = np.random.default_rng(0)
N = 64; m = 4*N                                      # near the m ~ 4N information limit
x_true = rng.standard_normal(N) + 1j*rng.standard_normal(N)
A = (rng.standard_normal((m, N)) + 1j*rng.standard_normal((m, N)))/np.sqrt(2)
y = np.abs(A @ x_true)**2                            # intensity-only measurements
def phase_invariant_distance(a, b):
    return np.sqrt(max(np.vdot(a,a).real+np.vdot(b,b).real-2*abs(np.vdot(a,b)), 0.0))
print(f"N={N}, m={m} (= {m//N}N),  ||x|| = {np.linalg.norm(x_true):.3f}")"""),

md(r"""## Spectral initialization: the leading eigenvector of $Y$

Build $Y=\tfrac1m A^{H}\,\mathrm{diag}(y)\,A$ and take its top eigenvector, scaled by $\sqrt{\mathrm
{mean}(y)}$. It is only moderately aligned with $x$ (correlation well below 1), but -- crucially -- it
sits in the **basin of attraction** of the true solution, which a random guess does not."""),
co("""Y = (A.conj().T @ (y[:, None]*A)) / m               # data matrix
w, V = np.linalg.eigh(Y)                             # eigh: ascending eigenvalues, unit eigenvectors
x0_spec = np.sqrt(np.mean(y)) * V[:, -1]             # top eigenvector, scaled to ||x||
x0_rand = (rng.standard_normal(N)+1j*rng.standard_normal(N))*np.sqrt(np.mean(y))/np.sqrt(N)

align = abs(np.vdot(V[:,-1], x_true))/np.linalg.norm(x_true)     # |<v, x/||x|| >|
print(f"spectral init: |correlation with x| = {align:.3f},  distance = {phase_invariant_distance(x0_spec, x_true):.2f}")
print(f"random   init: distance = {phase_invariant_distance(x0_rand, x_true):.2f}")
assert phase_invariant_distance(x0_spec, x_true) < phase_invariant_distance(x0_rand, x_true)"""),

md(r"""## Why the eigenvector points at $x$ (SymPy / expectation)

For a complex-Gaussian $a$, $\mathbb E\big[\,|\langle a,x\rangle|^2\,a a^{H}\big]=\lVert x\rVert^2 I +
c\,x x^{H}$ for a positive constant $c$ (its value depends on the Gaussian normalization). The extra
rank-one term $c\,xx^{H}$ raises the eigenvalue **along $x$** above the rest, so $x$ is the unique top
eigenvector of $\mathbb E[Y]$ -- averaging many measurements, the leading eigenvector *is* the signal
direction. A Monte-Carlo estimate confirms the alignment and the spectral gap."""),
co("""# Monte-Carlo E[Y] with x fixed: check top eigenvector aligns with x, eigenvalue ratio ~ 2:1
xu = x_true/np.linalg.norm(x_true)
Ymc = np.zeros((N, N), complex)
for _ in range(4000):
    a = (rng.standard_normal(N)+1j*rng.standard_normal(N))/np.sqrt(2)
    Ymc += abs(np.vdot(a, xu))**2 * np.outer(a, a.conj())
Ymc /= 4000
wl, Vl = np.linalg.eigh(Ymc)
print(f"top eigenvector alignment with x: {abs(np.vdot(Vl[:,-1], xu)):.3f}  (-> 1)")
print(f"spectral gap (top eigenvalue / next): {wl[-1]/wl[-2]:.2f}  (> 1 -> x separated)")
assert abs(np.vdot(Vl[:,-1], xu)) > 0.9 and wl[-1]/wl[-2] > 1.1"""),

md(r"""## Wirtinger Flow: gradient descent from each start

The Wirtinger Flow gradient is $\nabla f(z)=\tfrac1m A^{H}\big[(|Az|^2-y)\odot(Az)\big]$ (the same
backprop gradient as the previous notebook). Descend from the spectral init and, for contrast, from
the random init. At $m=4N$ the spectral start drives the error to zero; the random start stalls in a
spurious minimum."""),
co("""def wf_grad(z):
    return (A.conj().T @ ((np.abs(A@z)**2 - y)*(A@z))) / m
def wirtinger_flow(z0, iters=4000, mu=0.15):
    z = z0.copy(); step = mu/np.mean(y); hist = []
    for _ in range(iters):
        z = z - step*wf_grad(z); hist.append(phase_invariant_distance(z, x_true))
    return z, hist

z_spec, h_spec = wirtinger_flow(x0_spec)
z_rand, h_rand = wirtinger_flow(x0_rand)
print(f"WF from spectral init: final distance = {h_spec[-1]:.2e}  (recovered)")
print(f"WF from random   init: final distance = {h_rand[-1]:.2e}  (spurious minimum)")
assert h_spec[-1] < 1e-3                              # global recovery
assert h_rand[-1] > 1.0                               # random init fails at m = 4N"""),

md(r"""## Plots"""),
co(r"""fig, ax = plt.subplots(1, 2, figsize=(11.5, 4))
ax[0].semilogy(h_spec, color="#4C78A8", label="from spectral init")
ax[0].semilogy(h_rand, color="#E45756", label="from random init")
ax[0].set_xlabel("Wirtinger-Flow iteration"); ax[0].set_ylabel("distance to true field")
ax[0].set_title("spectral init converges; random init stalls"); ax[0].legend()
zc = z_spec*np.exp(1j*np.angle(np.vdot(z_spec, x_true)))    # align global phase
ax[1].plot(x_true.real, color="#4C78A8", lw=2, label="Re(x) true")
ax[1].plot(zc.real, "--", color="#E45756", label="Re(x) recovered")
ax[1].set_xlabel("index"); ax[1].set_title("field recovered from intensity only"); ax[1].legend()
plt.tight_layout(); plt.show()""" ),

md(r"""## Connecting to the dispersion instrument

Spectral initialization *requires incoherent* measurements -- the eigenvector argument relies on the
sensing vectors being random-like. Dispersion alone, $A_D=\mathcal F^{-1}e^{\,j\pi Df^2}\mathcal F$, is
a **unitary, coherent** operator: its rows are not incoherent, so the plain spectral estimate does not
align with $x$ (verified separately), and the Gerchberg-Saxton projections remain the better solver
for pure dispersion diversity. To bring spectral initialization -- and its global-convergence
guarantee -- to the instrument, add **random coded masks** (multiply by random phase screens before
dispersing, i.e. coded diffraction patterns). That makes the effective sensing incoherent, and the
spectral-init + Wirtinger-Flow pipeline shown here applies directly. This is the design lesson: choose
the measurement so the reconstruction is well-posed."""),

md(r"""## Summary

- Gradient descent on the phase-retrieval loss is non-convex; from a random start it can fail.
  **Spectral initialization** starts from the leading eigenvector of $Y=\tfrac1m\sum y_i a_ia_i^H$,
  which sits in the correct basin because $\mathbb E[Y]=\lVert x\rVert^2I+xx^H$ has top eigenvector $x$.
- **Wirtinger Flow** = spectral init + gradient descent recovers the field exactly at $m\approx4N$,
  while the same descent from a random start stalls in a spurious minimum.
- The method needs **incoherent** measurements; pure dispersion is coherent (use GS), but adding
  **random coded masks** makes spectral init + WF applicable to the dispersion instrument.

Subject-verb-object: the measurements build the data matrix; the eigenvector estimates the field;
Wirtinger Flow descends to the truth; incoherent sensing makes it global. This completes the
phase-recovery trilogy -- projections (GS), gradient descent, and spectral-initialized gradient flow."""),
]

write("spectral", "initialization", cells)
