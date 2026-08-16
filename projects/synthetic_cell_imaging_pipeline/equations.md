# Equations

All equations below are public-domain textbook mathematics (see `README.md`'s
disclaimer). Each entry notes what is numerically/symbolically **verified** by the
corresponding MATLAB/notebook code, not just stated.

## Part 1 — Field theory

$$\mathbf{E} = -\nabla V, \qquad \nabla\cdot\mathbf{E}, \qquad \nabla\times\mathbf{E}, \qquad \nabla^2 V$$

Point charge: $V = \dfrac{kq}{r}$, $r=\sqrt{x^2+y^2+z^2}$.

Physical dipole: $V = \dfrac{kq}{r_+} - \dfrac{kq}{r_-}$, $r_\pm = \sqrt{x^2+y^2+(z\mp a/2)^2}$.

**Verified:** $\nabla\cdot\mathbf{E}=0$ and $\nabla^2 V = 0$ away from the source (symbolic,
`simplify` reduces both to exactly `0`); $\nabla\times\mathbf{E}=\mathbf{0}$ identically (any
gradient field is curl-free); far-field falloff exponents measured directly from the
symbolic field: point charge $\propto 1/r^{2.000}$, dipole (on-axis) $\propto 1/r^{3.000}$.

## Part 2 — Bessel's equation from cylindrical symmetry

Helmholtz equation, field with no $\phi$ or $z$ dependence:

$$\frac{1}{\rho}\frac{d}{d\rho}\!\left(\rho\frac{dR}{d\rho}\right) + k^2 R = 0
\;\;\Longleftrightarrow\;\;
\rho^2 R'' + \rho R' + k^2\rho^2 R = 0 \quad\text{(Bessel's equation, order 0)}$$

Solutions: $R(\rho) = J_0(k\rho)$ (regular at $\rho=0$), and $J_1(k\rho)$ for the order-1 case.

**Verified:** symbolic residual of $J_0(k\rho)$ substituted into Bessel's equation is exactly
`0`; first four zeros of $J_0$ match Abramowitz & Stegun reference values
$(2.4048, 5.5201, 8.6537, 11.7915)$ to $<10^{-3}$; zeros of $J_1$:
$(3.8317, 7.0156, 10.1735, 13.3237)$.

## Part 3 — Even/odd symmetry

$$f(x) = f_e(x) + f_o(x), \qquad f_e(x)=\frac{f(x)+f(-x)}{2},\quad f_o(x)=\frac{f(x)-f(-x)}{2}$$

$$f_e(-x)=f_e(x)\ \text{(even)}, \qquad f_o(-x)=-f_o(x)\ \text{(odd)}$$

FFT correspondence: a real **even** signal has a real spectrum; a real **odd** signal has
a purely imaginary spectrum.

**Verified:** decomposition reconstructs the original signal to machine precision
($\max|f_e+f_o-f|<10^{-15}$); on an exactly-centered (odd-length) sample grid, a real even
test signal's FFT has $\max|\mathrm{Im}|/\max|\mathrm{Re}| \sim 10^{-17}$, and a real odd
test signal's FFT has $\max|\mathrm{Re}|/\max|\mathrm{Im}| \sim 10^{-17}$.

## Part 4 — Forward model

$$\mathbf{y} = H\mathbf{x} + \mathbf{n}$$

$x$: synthetic cell object (vectorized $N\times N$ image). $H$: exact separable-Gaussian
blur matrix, $H = H_1 \otimes H_1$ (Kronecker product of a 1D circulant Gaussian blur
matrix with itself). $n$: additive detector noise.

**Verified:** the dense-matrix blur $H\mathbf{x}$ agrees with an independent FFT-based
circular convolution to $\max|\Delta| \sim 10^{-16}$ (machine precision) — the matrix
construction is exactly right, not just "close."

## Part 5 — Matrix analysis

$$\mathrm{rank}(H), \qquad H = U\Sigma V^T \ (\text{SVD}), \qquad \mathrm{cond}(H)=\frac{\sigma_{\max}}{\sigma_{\min}}$$

$$\hat{\mathbf{x}}_{\text{naive}} = H^{+}\mathbf{y} \quad(\text{pseudoinverse}),\qquad
\hat{\mathbf{x}}_{\lambda} = (H^TH+\lambda I)^{-1}H^T\mathbf{y} \quad(\text{Tikhonov})$$

**Verified (last MATLAB run, $N=24$):** $\mathrm{rank}(H)=576=N^2$ (full rank — every
direction has *some* response) but $\mathrm{cond}(H)\approx 1.10\times10^{9}$ (some
directions are still ~$10^{-9}\times$ weaker than the strongest); naive pseudoinverse
reconstruction error $\approx 1.48\times10^7$ vs. best-$\lambda$ Tikhonov error
$\approx 5.21$ — a $\sim2.8\times10^{6}\times$ improvement from regularization alone.

## Part 6 — Kinetics

$$C(t) = C_0 e^{-kt}, \qquad I(t) = \alpha C(t) + \text{noise}\quad(\text{synthetic optical measurement})$$

Analytic: $\ln I(t) = \ln(\alpha C_0) - kt$ (linear in $t$, slope $=-k$).

**Verified (last MATLAB run, true $k=0.35$):** analytic log-linear fit $k=0.3416$ (2.41%
error); `lsqcurvefit` nonlinear fit on raw data $k=0.3473$ (0.76% error); the Python
notebook's `scipy.optimize.curve_fit` and PyTorch autograd fit, run on the **exact same
exported dataset**, both independently recover $k=0.3473$ as well.

## Part 7 — Subsystem gains

$$I_{pd} = R_\lambda P_{opt}\ (\text{A/W}), \qquad V_{amp}=R_f I_{pd}\ (\text{V/A}),
\qquad \text{count} = \mathrm{round}\!\left(\frac{V_{amp}}{V_{ref}}(2^{n}-1)\right)$$

**Verified:** composing the individual stage gains ($R_\lambda \times R_f$) reproduces the
directly-computed end-to-end gain $P_{opt}\to V_{amp}$ to $<10^{-9}$ relative error;
round-tripping a $50\,\mu\text{W}$ input through photodiode → TIA → 12-bit ADC → inverse
chain recovers $50.014\times10^{-6}$ W (0.03% error, within one ADC quantization step).

## Part 8 — Boolean decision logic

$$\text{detected} = (\text{signal} > \text{threshold}_{\text{signal}}), \quad
\text{focused} = (\text{focus\_metric} > \text{threshold}_{\text{focus}}), \quad
\text{stable} = (|\text{error}| < \text{tolerance})$$

$$\text{accept} = \text{detected} \;\wedge\; \text{focused} \;\wedge\; \text{stable}$$

**Verified:** a sharp reference image (focus metric $72.5$) passes the focus gate; the
same object deliberately blurred (focus metric $5.3$) fails it against the same
threshold ($20$) — the logic actually discriminates real cases, not just runs without
error.

## Part 9 — PID feedback

$$e[n] = \text{target} - \text{measured}[n], \qquad
u[n] = K_p e[n] + K_i\!\sum_{i=0}^n e[i] + K_d\big(e[n]-e[n-1]\big)$$

**Verified:** starting from error $3.5$, a $K_p=0.6,K_i=0.15,K_d=0.05$ loop reduces the
error to $0.0224$ within 60 steps, settling to within 5% of the initial error by step 12.

## Part 11 — Pipeline

$$\text{acquire}\to\text{preprocess}\to\text{calibrate}\to\text{reconstruct}\to\text{classify}\to\text{store}$$

Calibration: $y_{\text{cal}} = y_{\text{proc}} / g$, $g$ = per-pixel gain map.

**Verified:** dividing out a synthetic $\pm5\%$ gain map recovers the un-gained
measurement to $0.97\%$ relative error; the regularized reconstruction ($\lambda=3.16\times10^{-3}$)
PASSES the pipeline's own focus gate (metric $40.3$ vs. threshold $30$), in contrast with
Part 8's raw (unreconstructed) blurred case, which fails; `checkpoint.mat` round-trips
through `save`/`load` exactly.

## Part 12 — PyTorch autograd (notebook only)

$$\theta = (\log(\alpha C_0),\ k), \qquad \mathcal{L}(\theta) = \big\lVert y_{\text{model}}(\theta) - y_{\text{measured}}\big\rVert_2^2$$

**Verified:** `torch.autograd`'s $\partial\mathcal{L}/\partial k$ matches a central
finite-difference gradient on the identical loss function to $3.3\times10^{-11}$ relative
error.
