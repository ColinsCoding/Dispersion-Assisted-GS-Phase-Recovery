# Solutions

Every answer below was computed, not estimated — see the command noted for each.

**P1.** Solving $|E_{\text{point}}(r_1{=}5)| = |E_{\text{dipole}}(r_2)|$ exactly (not the far-field
approximation) with the project's own symbolic fields, $a=0.2$: $E_{\text{point}}(5) = 0.04$.
The dipole field is a much weaker source than an isolated point charge at equal distance (its
near-equal-and-opposite charges mostly cancel), so it must be measured closer in to reach the
same magnitude — scanning $E_{\text{dipole}}(r)$ shows it crosses $0.04$ between $r=2$ and
$r=3$; root-finding gives $r_2 = 2.1575$, **ratio $r_2/r_1 = 0.4315$**.

**P2.** $\lambda_c = 2\pi a / j_{0,1} \Rightarrow a = \lambda_c j_{0,1}/(2\pi) = 1.0\times2.4048/(2\pi)$
= **0.383 cm**.

**P3.** Predicted: $f_e(x) = 3\cos(2x)-1$ (cosine + constant are even), $f_o(x)=5\sin(3x)$ (sine
is odd). Verified numerically: $\max|f_{e,\text{computed}} - f_{e,\text{predicted}}| = 1.15\times10^{-14}$,
$\max|f_{o,\text{computed}} - f_{o,\text{predicted}}| = 1.13\times10^{-14}$ — matches to
floating-point noise.

**P4.** `part4_forward_model(20, 2.0, 0.02, 0).H` gives $\mathrm{cond}(H) = 2.4713\times10^{12}$,
vs. the default $N{=}24,\sigma{=}1.5$ case's $1.1019\times10^9$ — **larger**, confirming more
blur worsens conditioning (a wider PSF suppresses high spatial frequencies more severely,
pushing more singular values toward zero).

**P5.** Best achievable error was $5.2102$; target $=10\times = 52.102$. Scanning the existing
`lambdas`/`errs` arrays, the first $\lambda$ with `err < 52.102` is
$\boxed{\lambda \approx 3.8\times10^{-6}}$ (error $44.82$ there).

**P6.** Prediction: $k$ is a decay-RATE parameter, independent of the amplitude scale
$\alpha C_0$ — changing $\alpha C_0$ should barely move the fitted $k$ (it changes the
absolute noise-to-signal ratio slightly, since noise is added at a fixed absolute
`noise_std`, not scaled). Verified: $\alpha C_0=2.4 \to k_{\text{fit}}=0.3473$ (0.76% error);
$\alpha C_0=5.0 \to k_{\text{fit}}=0.3487$ (0.36% error) — both close to true $k=0.35$, the
higher-amplitude case fitting *slightly* better purely from improved relative SNR, exactly as
predicted.

**P7.** $I_{pd}=0.5\times50\mu\text{W}=25\mu\text{A}$, $V_{amp}=2\times10^4\times25\mu\text{A}=0.5\text{V}$,
ADC count $=\mathrm{round}(0.5/3.3\times4095)=620$. Recovered power $=49.96\,\mu\text{W}$
(**0.073% error**) — yes, round-trip recovery still holds comfortably within a few
quantization steps.

**P8.** Bisecting `part4_forward_model`'s `sigma_blur` between $0.3$ and $1.5$ for the focus
metric to cross $20.0$ (via `fzero`): $\sigma_{\text{blur}} = \boxed{0.849}$.

**P9.** With $K_p{=}0.3,K_i{=}0.05,K_d{=}0.02$: the loop still converges within 60 steps
(final error $0.0071$, even smaller than the default gains' $0.0224$) but settles later —
**step 21** vs. the default gains' step 12 (9 steps slower), consistent with weaker gains
producing a slower but still-stable response.

**P10.** $\lambda/4 = 550\,\text{nm}/4 = 137.5\,\text{nm}$ — a physical length tolerance on
wavefront/focus error. Part 4's $\sigma_{\text{blur}}=1.5$ is a **dimensionless pixel-count**
in the synthetic $24\times24$ image grid, with no assigned physical pixel pitch. These are
**not directly comparable** as given — converting $\sigma_{\text{blur}}$ to a physical length
would require specifying the imaging system's magnification/pixel pitch; the comparison
illustrates that "blur in pixels" and "wavefront tolerance in nm" are different *kinds* of
quantities that only become comparable once a physical scale is attached.

**P11.** `run_pipeline(24, 1.5, 0.02, 7)` (seed=7): focus metric $41.66$ (vs. seed=0's $40.31$),
decision = **PASS**, unchanged from the default seed's PASS — the gain-map/noise realization
differs, but not enough to cross the classification threshold either way.

**P12.** At the PyTorch optimum ($k_{\text{fit}}=0.34733$), re-evaluating $d\mathcal{L}/dk$
gives $5.55\times10^{-16}$ — **essentially exactly zero**, because a gradient-descent optimizer
stops (in the limit) exactly where the loss gradient vanishes; that IS the definition of
a stationary point/optimum, unlike the notebook's gradient-comparison cell which deliberately
evaluates the gradient at a non-optimal point ($k=0.2$) specifically so the gradient is
nonzero and worth comparing against a finite difference.

**P14.** Editing `part9_pid_feedback.m`'s internal `rng(2)` to `rng(7)` and re-running
`part14_integrated_project()`: **all cross-consistency checks still pass** — `pid_final_error`
changes to $0.0104$ (vs. the original seed's $0.0224$, both comfortably under the
$15\%$-of-initial-error convergence threshold), and every other stage's numbers
(cond(H), rank(H), reconstruction error, both Boolean decisions, kinetics fit) are
unaffected, since Part 9's own RNG seed is independent of Parts 4-8's data pipeline.
(File restored to its original `rng(2)` after this check — see git history/diff.)
