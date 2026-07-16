"""Generate notebooks/gradient_descent_phase_retrieval.ipynb."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from nbkit import md, co, setup_cell, write

cells = [
md(r"""# Gradient descent and backpropagation for phase retrieval and dispersion calibration

The Gerchberg-Saxton notebook recovered phase by *alternating projections*. The modern alternative is
to treat the instrument as a **differentiable function** and minimize a loss with **gradient descent**,
where the gradient is computed by **backpropagation** (the chain rule through the forward model). The
same machinery does two jobs at once:

1. **recover the field** $x$ from dispersed intensity measurements, and
2. **find the dispersion parameters** $D$ themselves -- calibrating the instrument from data.

The forward model is $I_D(x)=\big|\,\mathcal F^{-1}\{e^{\,j\pi D f^2}\,\mathcal F\{x\}\}\big|^2$, and the
loss is the intensity mismatch $L=\sum_k\sum_n\big(I_{D_k}(x)_n-\hat I_{k,n}\big)^2$. We derive its
gradient in closed form (Wirtinger calculus), verify it equals a finite-difference gradient (and Torch
autograd when present) -- so "backprop" is not a black box -- then descend to recover $x$ and to
estimate $D$.

Self-contained: NumPy, SymPy, Pandas, Matplotlib; optional PyTorch."""),
setup_cell(),

md(r"""## The differentiable forward model

$A_D x=\mathcal F^{-1}\{H_D\,\mathcal F\{x\}\}$ with $H_D=e^{\,j\pi D f^2}$ is the dispersion operator
(all-pass, $|H_D|=1$, so $A_D$ is unitary and its adjoint is $A_D^{H}z=\mathcal F^{-1}\{H_D^{*}\,
\mathcal F\{z\}\}$). The measured intensity is $I_D=|A_D x|^2$."""),
co("""N = 128
f = np.fft.fftfreq(N)
def H(D): return np.exp(1j*np.pi*D*f**2)
def A(x, D):  return np.fft.ifft(H(D)*np.fft.fft(x))         # forward (disperse)
def AH(z, D): return np.fft.ifft(np.conj(H(D))*np.fft.fft(z))  # adjoint (unitary: = A(., -D))
# adjoint check: <A x, z> == <x, A^H z>
rng = np.random.default_rng(0)
x = rng.standard_normal(N)+1j*rng.standard_normal(N); z = rng.standard_normal(N)+1j*rng.standard_normal(N)
assert np.allclose(np.vdot(A(x,7.0), z), np.vdot(x, AH(z,7.0)))
print("adjoint identity holds: <A x, z> = <x, A^H z>")"""),

md(r"""## The gradient is backpropagation

For $L=\sum_n(|y_n|^2-\hat I_n)^2$ with $y=A_Dx$, Wirtinger calculus gives the descent gradient
$$\nabla_x L \;=\; 4\,A_D^{H}\!\big[(|y|^2-\hat I)\odot y\big].$$
This is exactly what an autograd engine computes by the chain rule through $|A_Dx|^2$. We prove it by
matching a **finite-difference** directional derivative $\frac{L(x+\epsilon v)-L(x-\epsilon v)}{2\epsilon}$
to $\mathrm{Re}\langle\nabla_xL,v\rangle$, and (if PyTorch is present) to autograd."""),
co("""D_meas = [-30.0, 0.0, 25.0]
x_true = (np.exp(-((np.arange(N)-54)**2)/(2*6**2)) + 0.7*np.exp(-((np.arange(N)-82)**2)/(2*4**2))) \
         * np.exp(1j*(0.05*np.arange(N)))
I_meas = [np.abs(A(x_true, D))**2 for D in D_meas]

def loss(x):
    return sum(np.sum((np.abs(A(x,D))**2 - Ik)**2) for D, Ik in zip(D_meas, I_meas))
def grad(x):
    g = np.zeros(N, complex)
    for D, Ik in zip(D_meas, I_meas):
        y = A(x, D); r = np.abs(y)**2 - Ik
        g += 4*AH(r*y, D)
    return g

# finite-difference check along a random complex direction
xp = rng.standard_normal(N)+1j*rng.standard_normal(N)
v  = rng.standard_normal(N)+1j*rng.standard_normal(N)
eps = 1e-6
fd = (loss(xp+eps*v) - loss(xp-eps*v))/(2*eps)
an = np.real(np.vdot(grad(xp), v))                          # Re<grad, v>
print(f"finite-difference derivative = {fd:.5e}")
print(f"analytic Re<grad, v>        = {an:.5e}")
assert abs(fd-an)/abs(fd) < 1e-4
print("analytic gradient == finite-difference gradient  (this IS backprop)")
if torch is not None:
    xt = torch.tensor(xp, requires_grad=True)
    Lt = sum(((torch.fft.ifft(torch.tensor(H(D))*torch.fft.fft(xt))).abs()**2 -
              torch.tensor(Ik)).pow(2).sum() for D, Ik in zip(D_meas, I_meas))
    Lt.backward()
    print("torch autograd matches analytic grad:",
          np.allclose(2*np.conj(xt.grad.numpy()), grad(xp), rtol=1e-4))"""),

md(r"""## Gradient descent: descending the loss, and refining an estimate

Descend $x\leftarrow x-\mu\,\nabla_xL$ with momentum and a support projection. The gradient (backprop)
drives the loss down by orders of magnitude. But phase retrieval is **non-convex**: from a random
start, gradient descent can settle in a *spurious* minimum -- a low-loss field that is not the true one.
This is why gradient methods are paired with a good initialization (spectral estimate, or the GS
result). Started near a good estimate, the same descent **refines** it toward the truth -- the role
gradient descent plays inside learned/unrolled reconstructors."""),
co("""support = np.abs(np.arange(N)-N//2) < 45
def phase_invariant_distance(a, b):
    return np.sqrt(max(np.vdot(a,a).real+np.vdot(b,b).real-2*abs(np.vdot(a,b)), 0.0))

def descend(x0, mu=3e-3, iters=1500):
    x = x0.copy(); vel = np.zeros(N, complex); h = []
    for _ in range(iters):
        vel = 0.9*vel - mu*grad(x); x = x + vel
        x[~support] = 0                                          # support projection
        h.append(loss(x))
    return x, h

# (a) random start: the loss plunges, but the solution may be spurious
r = np.random.default_rng(1)
x_rand, hist = descend(np.sqrt(I_meas[1])*np.exp(1j*r.uniform(0,2*np.pi,N)))
print(f"random start:  loss {hist[0]:.2e} -> {hist[-1]:.2e}  (descends {hist[0]/hist[-1]:.0e}x),"
      f" distance {phase_invariant_distance(x_rand, x_true):.2f}")
assert hist[-1] < 1e-3*hist[0]                                  # gradient descent reduces the loss

# (b) warm start (as from a spectral/GS init): descent refines it toward the truth
x0 = x_true + 0.4*np.linalg.norm(x_true)/np.sqrt(N)*(r.standard_normal(N)+1j*r.standard_normal(N))
x_ref, _ = descend(x0)
d0, d1 = phase_invariant_distance(x0, x_true), phase_invariant_distance(x_ref, x_true)
print(f"warm start:    distance {d0:.3f} -> {d1:.3f}  (refined)")
assert d1 < d0"""),

md(r"""## Finding the dispersion parameter

Now flip the unknown: the field $x$ is a known calibration pulse, but the dispersion $D$ the instrument
applied is unknown. Minimize $L(D)=\sum_n(|A_Dx|^2-\hat I)^2$ over $D$. This loss is *multimodal*, so we
follow the standard calibration recipe: a coarse **grid scan** brackets the global minimum, then a
**gradient-based** step (the analytic $dL/dD$, verified against finite differences) refines $D$ to
sub-grid precision -- instrument self-calibration."""),
co("""D_true = 41.0
I_D = np.abs(A(x_true, D_true))**2                            # measurement at unknown dispersion
def lossD(D):  return np.sum((np.abs(A(x_true, D))**2 - I_D)**2)
def gradD(D):                                                # analytic dL/dD (backprop through |A_D x|^2)
    y = A(x_true, D)
    yD = np.fft.ifft((1j*np.pi*f**2)*H(D)*np.fft.fft(x_true)) # dy/dD
    return 4*np.sum((np.abs(y)**2 - I_D)*np.real(np.conj(y)*yD))

# the analytic derivative matches finite differences (backprop for a physical parameter)
Dt = 50.0; fd = (lossD(Dt+0.5) - lossD(Dt-0.5))/1.0
print(f"dL/dD at D={Dt}: analytic {gradD(Dt):.3f} vs finite-diff {fd:.3f}")
assert abs(gradD(Dt) - fd)/abs(fd) < 1e-2

# 1) coarse grid scan brackets the global minimum
Ds = np.linspace(0, 80, 200)
D_grid = Ds[np.argmin([lossD(D) for D in Ds])]
print(f"grid scan minimum at D = {D_grid:.3f}  (true {D_true})")
assert abs(D_grid - D_true) < 0.5
# 2) gradient-based (Newton) refinement to sub-grid precision
D_est = D_grid
for _ in range(15):
    hess = (gradD(D_est+0.3) - gradD(D_est-0.3))/0.6
    D_est -= gradD(D_est)/hess
print(f"gradient-refined dispersion D = {D_est:.4f}  (true {D_true})")
assert abs(D_est - D_true) < 0.05"""),

md(r"""## Plots"""),
co(r"""fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
ax[0].semilogy(hist, color="#4C78A8")
ax[0].set_xlabel("gradient-descent step"); ax[0].set_ylabel("loss")
ax[0].set_title("gradient descent: loss descends")
ph = np.angle(np.vdot(x_ref, x_true)); xd = x_ref*np.exp(1j*ph)   # align global phase
ax[1].plot(np.abs(x_true), color="#4C78A8", lw=2, label="true |x|")
ax[1].plot(np.abs(xd), "--", color="#E45756", label="refined")
ax[1].set_title("field refined by descent"); ax[1].legend()
Ds = np.linspace(0, 80, 300)
ax[2].plot(Ds, [lossD(D) for D in Ds], color="#54A24B")
ax[2].axvline(D_true, ls=":", color="#E45756"); ax[2].axvline(D_est, ls="--", color="#4C78A8")
ax[2].set_xlabel("dispersion D"); ax[2].set_ylabel("loss L(D)")
ax[2].set_title("finding D: minimum at the true value")
plt.tight_layout(); plt.show()""" ),

md(r"""## Summary

- The instrument is a **differentiable function** $I_D(x)=|A_Dx|^2$; phase retrieval and dispersion
  calibration are both minimizations of the intensity-mismatch loss $L$.
- The descent gradient $\nabla_xL=4A_D^{H}[(|A_Dx|^2-\hat I)\odot A_Dx]$ is derived in closed form and
  shown to equal a finite-difference gradient (and Torch autograd) -- **backpropagation is the chain
  rule, nothing hidden**.
- **Gradient descent** drives the loss down and **refines** a good estimate toward the truth; from a
  random start the non-convex loss can trap it in a spurious minimum, so it is paired with a
  spectral/GS initialization. The **same gradient in $D$** recovers the dispersion parameter from a
  calibration measurement -- one differentiable pipeline, two unknowns.
- This differentiable-optics view is what makes learned / unrolled phase-retrieval networks possible:
  the forward physics is the network, autograd supplies the gradients, and one loss connects the field
  and the instrument parameters.

Subject-verb-object: the loss measures the mismatch; backpropagation computes the gradient; gradient
descent recovers the field; the same gradient calibrates the dispersion."""),
]

write("gradient_descent", "phase_retrieval", cells)
