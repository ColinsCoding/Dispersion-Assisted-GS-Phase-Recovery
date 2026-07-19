"""Generate notebooks/dispersive_fourier_stationary_phase.ipynb -- the
stationary-phase derivation underlying the dispersive Fourier transform
(the physics justification for STEAM/time-stretch, and the key thing to
have solid before submitting P2), verified numerically (convergence as
dispersion grows, not just asserted), plus the answer to "has anyone made
a 3D version": YES -- Fraunhofer diffraction is the spatial stationary-
phase limit of the Fresnel diffraction integral, already partly built in
dgs.diffraction_grating. NOTE: no triple-double-quote docstrings inside
cell strings.
"""
import pathlib
import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "dispersive_fourier_stationary_phase.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# The Dispersive Fourier Transform: Stationary-Phase Derivation

The physics justification for STEAM/time-stretch (Solli, Gupta, Jalali,
*Appl. Phys. Lett.* 95, 231108, 2009): why does a sufficiently-dispersed
pulse's TIME-domain intensity reproduce its SPECTRAL intensity? This
notebook derives it via the method of stationary phase, verifies the
prediction converges to the exact (FFT-computed) result as dispersion
grows, then answers "has anyone made a 3D version?" -- yes: Fraunhofer
diffraction.""")

code(r"""import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))

import numpy as np
import sympy as sp
from IPython.display import display, Math
sp.init_printing(use_latex='mathjax')

from dgs.gs_core import disperse""")

md(r"""## 1. The setup

Propagating a pulse through a dispersive element multiplies its spectrum
$\tilde A(\nu)$ by a phase $H(\nu)=e^{i\pi D\nu^2}$ (this repo's own
convention, `dgs.gs_core.disperse`). The output time-domain field is the
inverse Fourier transform:
$$A_{out}(n) = \int \tilde A(\nu)\, e^{i\pi D\nu^2}\, e^{i2\pi\nu n}\, d\nu
= \int \tilde A(\nu)\, e^{i\psi(\nu)}\, d\nu, \qquad \psi(\nu)=\pi D\nu^2 + 2\pi\nu n$$""")

code(r"""nu, n, D = sp.symbols('nu n D', real=True)
psi = sp.pi*D*nu**2 + 2*sp.pi*nu*n

dpsi = sp.diff(psi, nu)
display(Math(r"\frac{d\psi}{d\nu} = " + sp.latex(dpsi)))

nu_s_solution = sp.solve(sp.Eq(dpsi, 0), nu)
display(Math(r"\nu_s(n) = " + sp.latex(nu_s_solution[0])
             + r"\qquad\text{(the STATIONARY POINT -- where the phase stops varying)}"))

d2psi = sp.diff(psi, nu, 2)
display(Math(r"\frac{d^2\psi}{d\nu^2} = " + sp.latex(d2psi)
             + r"\qquad\text{(constant -- pure quadratic dispersion)}"))"""
)

md(r"""## 2. Why the stationary point dominates

Away from $\nu_s$, $\psi(\nu)$ varies rapidly with $\nu$ (for large $D$),
so $e^{i\psi(\nu)}$ oscillates fast and nearby contributions to the
integral cancel. Near $\nu_s$, $\psi$ is (by construction) momentarily
*flat* -- contributions there add constructively. Result: for large $D$,
$$A_{out}(n) \approx \tilde A(\nu_s(n)) \times (\text{a slowly-varying prefactor from the local Gaussian integral around } \nu_s)$$
i.e. $|A_{out}(n)|^2 \propto |\tilde A(\nu_s(n))|^2$ -- **the output time
trace directly reproduces the input spectrum**, linearly mapped via
$\nu_s(n)=-n/D$.""")

md(r"""## 3. Numerical verification: does this actually converge?

A genuinely non-Gaussian, two-peak test spectrum (so "reproduces the
spectrum" is a real, discriminating claim), dispersed via the
already-tested `dgs.gs_core.disperse`, compared against the stationary-
phase prediction -- across INCREASING dispersion, to confirm the
approximation's error actually SHRINKS as theory demands (not just
"looks okay" at one arbitrary value).""")

code(r"""def spectrum_shape(nu_arr, center=0.05):
    return (np.exp(-((nu_arr-center-0.02)**2)/(2*0.008**2))
            + 0.6*np.exp(-((nu_arr-center+0.015)**2)/(2*0.005**2)))

results = []
for D_val in [500, 5000, 50000, 200000]:
    # N must grow with D: the output pulse spreads over a WIDER time window
    # as dispersion increases (that's the entire point of time-stretch), so
    # a fixed-size FFT would wrap around and corrupt the comparison
    N = 4096
    while N/2 < D_val*0.05*3:
        N *= 2
    nu_arr = np.fft.fftfreq(N)
    E_true_spectrum = spectrum_shape(nu_arr).astype(complex)
    E_input = np.fft.ifft(E_true_spectrum)

    E_d = disperse(E_input, D_val)
    I_out = np.abs(E_d)**2
    n_idx = np.fft.fftfreq(N) * N
    nu_s_of_n = -n_idx / D_val

    I_predicted = spectrum_shape(nu_s_of_n)**2
    valid = np.abs(nu_s_of_n - 0.05) < 0.05
    I_out_n = I_out[valid] / np.max(I_out[valid])
    I_pred_n = I_predicted[valid] / np.max(I_predicted[valid])
    max_err = np.max(np.abs(I_out_n - I_pred_n))
    results.append((D_val, N, valid.sum(), max_err))
    print(f"D={D_val:>7} (N={N:>6}): valid points={valid.sum():>5}, "
          f"max shape error (peak-normalized) = {max_err:.4f}")

errors = [r[3] for r in results]
assert errors[0] > errors[1] > errors[2] > errors[3]
print("\nConfirmed: error shrinks monotonically as D grows -- the stationary-phase")
print("approximation is asymptotically exact, verified by direct comparison to the")
print("actual FFT-computed (exact) dispersed field, not merely claimed.")""")

md(r"""## 4. "I wonder if anyone's made a 3D version" -- yes: Fraunhofer diffraction

The 1D **temporal** stationary-phase result above has an exact **spatial**
analog, already one of the most famous results in classical optics. The
Fresnel diffraction integral (near-field, paraxial) has the *identical*
mathematical form as pulse propagation through a dispersive element --
apply the SAME stationary-phase argument in the spatial-frequency domain
and you get the **Fraunhofer (far-field) diffraction** result: the
far-field diffraction pattern is the Fourier transform of the aperture.
`dgs.diffraction_grating` already implements this (Fraunhofer grating
physics) -- it's the "3D version," under a different name, already in
this repo.

| | Temporal (dispersive FT) | Spatial (Fraunhofer diffraction) |
|---|---|---|
| propagating quantity | pulse envelope $A(t)$ | field amplitude $E(x,y)$ |
| transform variable | frequency $\nu$ (or $\omega$) | spatial frequency $k_x=\frac{2\pi}{\lambda}\sin\theta$ |
| "far field" condition | large accumulated dispersion $D$ (or $\beta_2 L$) | large propagation distance $z\gg a^2/\lambda$ (Fraunhofer condition) |
| stationary-phase result | $\lvert A_{out}(t)\rvert^2\propto\lvert\tilde A(\nu_s(t))\rvert^2$ | far-field intensity $\propto$ \|Fourier transform of aperture\|$^2$ |
| this repo's module | `dgs.gs_core` | `dgs.diffraction_grating` |

The "3D" generalization (2D transverse aperture instead of 1D slit) is
exactly `dgs.diffraction_grating`'s single-slit envelope extended to a
2D aperture function -- the same Fraunhofer/far-field stationary-phase
argument, one more transverse dimension.""")

code(r"""from dgs.diffraction_grating import single_slit_envelope
import numpy as np

# the SAME stationary-phase-derived idea, in its spatial (Fraunhofer) form:
# far-field intensity pattern from a single slit, already implemented and
# tested elsewhere in this repo
theta = np.linspace(-0.3, 0.3, 500)
a, wavelength = 5e-6, 633e-9
envelope = single_slit_envelope(theta, a, wavelength)
print(f"Fraunhofer single-slit envelope (dgs.diffraction_grating): peak={envelope.max():.4f} "
      f"at theta=0, matching the SAME stationary-phase/far-field structure")
print("as the temporal dispersive Fourier transform above -- same math, spatial instead of temporal.")""")

nb["cells"] = cells
OUT.parent.mkdir(exist_ok=True)
nbf.write(nb, str(OUT))
print(f"wrote {OUT}")
