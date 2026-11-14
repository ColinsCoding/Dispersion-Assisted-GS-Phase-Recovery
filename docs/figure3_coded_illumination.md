# Figure 3 read as a phase-retrieval design loop

Source: B. Jalali, A. Kadambi and V. Roychowdhury, "Physics-AI Symbiosis," arXiv:2109.05959, Fig. 3
(which in turn summarizes an LED-array microscope design method, their ref. [25]). Everything below is my
own paraphrase and extension; the figure itself is not reproduced here.

## 1. What the three panels say

The figure is a loop, not a pipeline:

| Panel | Role | Phase-retrieval name |
|---|---|---|
| (a) coded-illumination microscope | LEDs light the sample one pattern at a time; the camera records only intensity | **forward model** `I_k = |A_k x|^2` |
| (b) computational reconstruction | many intensity frames in, a phase image out | **phase retrieval** (GS, ptychography, gradient descent) |
| (c) physics-based learned design | the reconstruction error is differentiated back to the LED pattern | **design update** `s <- s - eta * dL/ds` |

The new idea is panel (c): the *illumination* is a trainable parameter, and the reconstruction algorithm is
inside the gradient path.

## 2. Phase-retrieval definitions, applied

Let `x` be the unknown complex sample (amplitude and phase). Illumination pattern `k` applies a known linear
operator `A_k` (for an LED at angle `k`, a tilt `exp(i k.r)` followed by the lens pupil). The camera sees

```math
y_k = |A_k x|^2 = (A_k x)^{*} \odot (A_k x), \qquad k = 1,\dots,K
```

The star is the complex conjugate: intensity is the field times its own conjugate, i.e. the bra-ket product
`<psi|psi>` pixel by pixel. The conjugate is exactly what throws the phase away, so one frame cannot be inverted.
Phase retrieval is the search for `x` consistent with all `K` magnitudes.

Equivalent "lifted" form, useful for design: with `a` one row of `A_k`,

```math
|a^{H} x|^2 = \mathrm{tr}\!\left( (a a^{H})\,(x x^{H}) \right)
```

so every measurement is a *linear* measurement of the rank-one matrix `x x^H`. Whether `x` is recoverable is then a
question about how well the set `{a a^H}` spans matrix space.

**Diversity is the design target.** In this repo the diversity comes from dispersion `D`; the GS note on
`D = -600` failing because the diversity was nearly zero is the same failure as two LED patterns that
light almost the same angles. In the microscope the designer controls diversity through the LED brightness vector
`s`:

```math
y_k(s) = \sum_{j} s_{kj}\, |A_j x|^2 \quad (\text{incoherent LEDs add in intensity})
```

and the loss `L(s) = || x_hat(y(s)) - x_true ||^2` is minimized over `s`, with `x_hat` produced by an
*unrolled* reconstruction (a fixed number of GS or gradient steps) so autograd can pass through it.

## 3. Moving the same loop from visible light to microwave and THz

Nothing in section 2 mentions a wavelength. What changes is the hardware that plays each role:

| Optical figure | Microwave to THz version |
|---|---|
| LED array | antenna or source array; `s` is the amplitude (and, if coherent, phase) weight per element |
| sample `x` | scatterer, metasurface, or the transfer function of a device under test |
| lens + camera | receive array with square-law (power) detectors, which is why only `|.|^2` survives |
| gradient on `s` | gradient on source weights, or on bias voltages that set them |

**When a lumped circuit is legitimate.** Treat a part as a lumped element only if its size is well below the
wavelength; a common rule is `size < lambda / 10`:

| Frequency | lambda | lumped up to about |
|---|---|---|
| 1 GHz | 30 cm | 3 cm |
| 10 GHz | 3 cm | 3 mm |
| 100 GHz | 3 mm | 300 um |
| 1 THz | 300 um | 30 um |

So microwave source networks can be designed with R, L, C and SPICE, but toward THz the elements become
comparable to the wavelength and you need distributed models. The transfer-matrix solver in
`dgs/metasurface_surrogate.py` is that next step: it is the distributed, exact version of the same linear
time-invariant block.

**Why time invariance matters.** If the device is linear and time-invariant (LTI), everything about it is its
frequency response `H(w)`, and the measurement per frequency is `|H(w)|^2`. Two consequences:

1. Each frequency is an independent phase-retrieval problem, so the illumination diversity can be spent across
   frequency, which is the time-stretch trick (wavelength to time through dispersion).
2. If the LTI system is also causal and minimum-phase, the Kramers-Kronig relations tie `arg H` to `ln|H|`, so
   magnitude alone *does* determine phase. Non-minimum-phase systems (extra delay, all-pass sections) break
   this, and then the designed diversity is what rescues recovery.

## 4. "Focus attention" as a design lever (my interpretation)

If `K` patterns are available but only a few are informative for a given sample, a learned weight per pattern,
`w = softmax(theta)`, lets the optimizer concentrate measurement time on the informative ones. That is the same
form as attention weights, `sum_k w_k y_k`, here applied over illumination patterns instead of tokens. It is a
natural parameterization of the design vector `s` in section 2, with the bonus that the weights are non-negative
and sum to one, matching a fixed optical power budget.

## 5. What is and is not established

- Panels (a) to (c) and the gradient-through-reconstruction idea come from the cited paper.
- The lifted-measurement form and the lumped-versus-distributed table are standard results, not claims from
  the paper.
- Sections 3 and 4 are my extensions. The section 2 loop is simulated in
  `dgs/coded_illumination_design.py`; the microwave/THz mapping and the softmax-attention weighting are not.

## Result of the section 2 experiment

1-D complex sample (32 points), coded masks with a fixed power budget, 20-step unrolled GS, masks trained by
autograd on 64 samples and scored on 128 held-out ones (phase-invariant fidelity, 1.0 = perfect):

| K masks | random masks | learned masks (3 seeds) |
|---|---|---|
| 4 | 0.10 - 0.12 | 0.30 - 0.34 |
| 8 | 0.12 - 0.15 | 0.58 - 0.63 |

Learned illumination wins in every run, but absolute fidelity is still modest: with only 20 GS iterations and few
masks the problem is hard, and I have not tuned beyond one setting.

Important caveat: the reconstructor is not broken. With 8 random masks it reaches 0.73 after 200 iterations
and 0.98 after 1000 (plain GS is slow). So the table above, scored at 20 iterations, largely measures
**convergence speed under a small iteration budget**: learned illumination makes GS converge faster. It does not
yet show that learned masks make otherwise-unrecoverable samples recoverable. Testing that needs the same
comparison at 1000+ iterations, which I have not run.
