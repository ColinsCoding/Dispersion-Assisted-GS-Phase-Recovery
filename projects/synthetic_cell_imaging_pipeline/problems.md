# Problems

Original problems, one or more per Part, with fresh numbers not drawn from any specific
textbook or patent exercise. Solutions are in `solutions.md`.

**P1.** A point charge sits at the origin with $V(r) = kq/r$. A second charge configuration
is a physical dipole with charges at $z=\pm a/2$. At what ratio of distances $r_{\text{dipole}}/r_{\text{point}}$
do the point-charge field and the on-axis dipole field have the *same magnitude*, given
$a = 0.2$ m and both configurations start at $r_{\text{point}} = r_{\text{dipole}} = 5$ m?
(Hint: use the measured falloff exponents from Part 1's code, not a fresh derivation.)

**P2.** A circular waveguide of radius $a=1.2$ cm is used to guide a wave whose lowest-order
azimuthally-symmetric ($m=0$) TM mode must have a cutoff wavelength (not frequency) of
exactly $\lambda_c = 1.0$ cm. Using $J_0$'s first zero, find the required radius $a$ to
3 significant figures. (Reframe `part2_bessel_coordinates.m`'s cutoff relation in terms of
$\lambda_c = 2\pi a / j_{0,1}$.)

**P3.** A signal $f(x) = 3\cos(2x) + 5\sin(3x) - 1$ is sampled on $x\in[-5,5]$. Without
computing anything, state which terms belong to $f_e(x)$ and which to $f_o(x)$, then verify
numerically using the Part 3 decomposition code.

**P4.** Using `build_cell_object`/`build_blur_matrix` with $N=20$ instead of the default 24,
and $\sigma_{\text{blur}}=2.0$, report the new $\mathrm{cond}(H)$. Is it larger or smaller
than the $N=24,\sigma=1.5$ case, and does that match the expectation that MORE blur -> WORSE
conditioning?

**P5.** For the $N=24$, $\sigma_{\text{blur}}=1.5$ system, find the smallest $\lambda$ (to
one significant figure, from the existing `lambdas` sweep) at which the Tikhonov
reconstruction error first drops below $10\times$ the best achievable error.

**P6.** A hypothetical second detector has responsivity-times-C0 product $\alpha C_0 = 5.0$
(vs. this project's default $\alpha C_0 \approx 2.4$) and the same true $k=0.35$. Predict
(without running code) whether `lsqcurvefit`'s fitted $k$ should be affected by this change,
then verify.

**P7.** A different photodetector has responsivity $0.5$ A/W (vs. this project's $0.8$ A/W)
and everything else in Part 7's chain unchanged. For the same $50\,\mu\text{W}$ input, what
ADC count results, and does the round-trip recovery still hold to within a few quantization
steps?

**P8.** Using Part 8's calibrated focus metric, at what blur $\sigma$ (search between $0.3$
and $1.5$) does the focus metric first drop below the $20.0$ threshold used to gate the raw
(unreconstructed) acquisition? Bisect numerically using `part4_forward_model`'s `sigma_blur`
argument.

**P9.** Retune the PID gains to $K_p=0.3, K_i=0.05, K_d=0.02$ (weaker than the defaults) and
report whether the loop still converges within 60 steps, and if so, how much later it
settles compared to the default gains.

**P10.** For the "camera manufacturing" row of the Part 10 tolerance table, express the
"focus shift < 1/4 wave" tolerance in physical units (nm) for a system operating at
$\lambda = 550$ nm (visible light), and compare it to Part 4's blur kernel's
$\sigma_{\text{blur}}=1.5$ pixel scale — are they the same *kind* of tolerance, or
fundamentally different quantities being compared?

**P11.** Using the Part 11 pipeline functions individually (not `run_pipeline`), acquire a
frame with `seed=7` instead of the default `seed=0`, and report whether the final
`classify()` decision changes.

**P12.** At the PyTorch autograd fit's optimum (not the initial guess used in the notebook's
gradient-comparison cell), evaluate $d\mathcal{L}/dk$ via autograd. What value should it be
close to, and why?

**P14.** Run `part14_integrated_project()` with a different random seed baked into `part9_pid_feedback`
(edit its internal `rng(2)` to `rng(7)`) and report whether the integrated summary's
cross-consistency checks still all pass.
