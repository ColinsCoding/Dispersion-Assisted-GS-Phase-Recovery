# Importing a SEALS intensity trace into TD-GSA — engineering report

**Audience:** written for an electrical-engineering reader (RF/microwave, optoelectronics,
signal-chain background — e.g. EC ENGR 279AS *Special Topics in Physical and Wave
Electronics: RF and Microwave Transmitter Design*) rather than a pure-physics reader.
Verified two independent ways (see §4) per this project's own testing standard.

## 1. What exists on each side

**SEALS** (`matlab/{main,SEALS,mie-2,rayleighdebye}.m`, ported to
`inverse/_seals_physics.py`) is a dispersive-grating spectrometer: a broadband source
illuminates a scattering particle, and a grating pair maps each wavelength to a unique
beam displacement / scattering angle (`SEALS.m`'s `y(lambda)`, `theta(lambda)`). At each
mapped angle, the Mie (or Rayleigh–Debye–Gans) model predicts a scattered intensity
`I_p(theta)`, `I_s(theta)` — the **intensity trail**: literally what a photodiode array
or spectrometer CCD would read out as one scan, `I_p(lambda)` over the ~500-point
wavelength sweep `SEALS.m` uses by default.

**TD-GSA** (`dgs/gs_core.py`, this repo's canonical implementation) is a **two-measurement**
alternating-projections phase-retrieval algorithm: given `I1 = |E|^2` and
`I2 = |disperse(E, D)|^2` (the same hidden complex field `E`, measured before and after a
known dispersion `H(nu) = exp(i*pi*D*nu^2)`), it recovers the phase of `E`. It requires
`D1 != D2`, both nonzero, and — per `gs_core.py`'s own kwarg-bounds warning — `|D| >= 5000`
(normalized units) for reliable convergence.

**Already built, before this report:** `projects/seals/inverse/dispersion.py` implements
`dispersive_operator()` in the *exact same* transfer-function convention as
`dgs.gs_core.disperse` (cross-checked in `tests/test_seals_dispersion.py`), and
`inverse/phase_retrieval.py` implements a generic, PyTorch-autograd multi-measurement
phase-retrieval routine architecturally parallel to GS. What was **not** built: an actual
call from SEALS-derived data into `dgs.gs_core` itself, and a check that the classical
(GS) and autograd paths agree with each other and with ground truth on that data. That
gap is what `inverse/seals_to_tdgsa.py` (new, this report) closes.

## 2. The one-measurement problem (read this before wiring anything up)

A real SEALS spectrometer, as built, records **one** intensity trace per scan. TD-GSA
needs **two**, at two different known dispersions, for the measurement diversity that
makes the inverse problem well-posed at all — a single square-law measurement cannot
determine phase (global-phase ambiguity alone rules that out, and `phase_retrieval.py`
already emits a runtime warning if called with only one measurement/operator pair).

So "importing the SEALS intensity trail into TD-GSA" cannot mean *feed the one existing
trace in and get phase out*. It means: **the native SEALS trace becomes one of the two
measurement planes, and a second plane is produced by passing the same hidden field
through a second, known dispersion.** `dispersion.py`'s own docstring already flags this
exact idea as a *future* architectural direction (a second, dispersive-fiber measurement
arm) rather than something the single-shot instrument in `SEALS_paper.pdf` implements —
this report keeps that framing explicit. In EE terms: this is proposing a **second
diversity branch** on the existing single-channel receiver, the same way a coherent
receiver adds a second (quadrature) detection path to break an ambiguity a single
detector can't resolve.

## 3. The bridge (`projects/seals/inverse/seals_to_tdgsa.py`)

```python
lamvec, theta_deg, mie_fields = seals_intensity_trace()      # SEALS's native I_p(lambda)
I1, I2 = build_gs_measurements(mie_fields, D1=6000.0, D2=-7000.0)  # two dispersed planes,
                                                                     # via dgs.gs_core.disperse
phi_gs, errors, _ = gs_core.retrieve_phase_with_history(I1, I2, D1, D2, n_iter=150,
                                                          unit_amplitude=False)
```

`unit_amplitude=False` because Mie-scattered intensity genuinely varies with angle — this
is not a constant-envelope (QPSK-like) signal, which matters for §4 below.
`build_gs_measurements` calls `dgs.gs_core.disperse` directly (via the already-verified
`dispersive_operator` bridge), so the dispersion applied is provably identical to what the
rest of this repo's TD-GSA pipeline uses, not a reimplementation.

## 4. Two independent verifications (as requested — a dual check, not a single pass)

`run_bridge_demo()` runs **both** available algorithms on the **identical** `(I1, I2)`
pair and compares each against Mie's own known phase (`T_p`, available here only because
this is a validation pass against a model with a known answer — a real instrument
would not have this):

1. **Classical GS** — `dgs.gs_core.retrieve_phase_with_history` (this repo's canonical
   TD-GSA implementation).
2. **Autograd** — `projects/seals/inverse/phase_retrieval.retrieve_phase` (PyTorch,
   already built in this package), given the same `I1`, `I2`.

Measured result (`tests/test_seals_to_tdgsa.py`, `D1=6000, D2=-7000, N=500`):

| Check | Result |
|---|---|
| GS fits its own measurements (self-consistency) | residual `2.5e-23` — essentially exact |
| GS phase vs. Mie ground truth | **RMS ≈ 0.50 rad** |
| Autograd phase vs. Mie ground truth | **RMS ≈ 0.28 rad** |
| GS vs. autograd (do the two methods agree with each other?) | **RMS ≈ 0.50 rad** — no |

> **Correction (later session):** the autograd number above used to also read `~0.50 rad`,
> reported as independent confirmation of GS's ~0.5 rad wall. It was wrong: Mie-scattered field
> amplitudes are physically tiny (`~1e-5` to `5e-4`), so the raw, un-normalized intensity loss
> Adam was minimizing was `~1e-16`-scale — small enough that Adam's default `eps=1e-8` silently
> stalled every update, and the optimizer never moved from its all-zero initial phase guess. The
> "converged" `0.50 rad` was literally an untouched initial guess's score, coincidentally close
> to GS's real answer. Fixed in `run_bridge_demo` by normalizing amplitude/intensities to `O(1)`
> before optimization (phase is scale-invariant, so this doesn't change what's being solved) —
> the same class of scale-sensitivity `tests/test_seals_phase_retrieval.py` had already flagged
> for a milder case, but that safeguard was never carried over to this specific caller until now.
> With the bug fixed, autograd genuinely optimizes (loss drops `~200x`) and lands on `0.28 rad`
> — notably *better* than GS, but still not the true phase, and still `0.50 rad` from GS's
> answer. The qualitative conclusion below (2-plane blind phase retrieval is genuinely
> underdetermined here) survives the correction; the specific numbers and the "both algorithms
> land in the same ~0.5 rad band" framing do not.

Swept across four `(D1, D2)` pairs spanning `-5000..-23000`, GS's error stayed in the same
`~0.4–0.5 rad` band every time (this part of the sweep used only `rms_gs_vs_truth`, unaffected
by the autograd bug above) — this is not a tuning artifact of one bad dispersion choice.

## 5. Honest conclusion: this is a real, diagnosed limitation, not a bug

GS converges to a solution that fits `(I1, I2)` essentially exactly (residual `~1e-23`)
but is **not** the true Mie phase, and a second, independently-implemented algorithm
(autograd — now genuinely optimizing, see the correction above) converges to a *different*
wrong answer on the same data (`0.28 rad` vs. GS's `0.50 rad`, `0.50 rad` apart from each
other). Both facts together say the two-measurement inverse problem is **genuinely
underdetermined** for this signal, not that either implementation has a bug (a *shared* bug
would tend to produce the *same* wrong answer, not two different ones — checked explicitly in
`test_bridge_demo_known_limitation_blind_gs_does_not_cleanly_recover_mie_phase`). Autograd's
gradient-based fit handles the weak-signal region somewhat better than GS's hard projection
(hence the lower RMS), but neither reaches anywhere near the `0.0014 rad` that a 3rd
measurement plane achieves (§6) — extra measurement diversity fixes this problem; a better
2-plane algorithm only partially does.

**The specific mechanism, diagnosed directly rather than left as "varying amplitude is
hard":** GS's recovered phase error correlates strongly with signal amplitude
(Pearson `r = -0.55` between per-sample phase error and `log(amplitude)`). Binning by
amplitude quartile: the top quartile (strong signal, near the forward-scattering peak
around `theta ~ 0`) recovers to a mean error of `0.33 rad`; the bottom quartile (weak
signal, the wide-angle tail, where Mie-scattered intensity has dropped off) has a mean
error of `1.54 rad` — the amplitude spans a `~20x` dynamic range across the trace
(`2.5e-5` to `5.2e-4`). This is visible directly in `seals_to_tdgsa_bridge.ipynb`'s Step 3
plot: GS tracks the true phase closely near `theta=0` and only degrades to noise at wide
angles.

The mechanism is the textbook failure mode of intensity-only phase retrieval, not a
mysterious ambiguity: GS's amplitude-constraint step each iteration enforces
`|E| = sqrt(I)` — wherever the measured intensity is near zero, that constraint carries
almost no information, so the phase there is free to wander while the algorithm still
converges (fits its own near-zero measurements exactly). It is not that GS failed to
converge; it is that a photon-starved sample has essentially nothing to converge *to*.
This also explains why a second, differently-implemented algorithm (autograd, now properly
normalized — see the correction above) lands on a *different* wrong answer in the same
weak-signal region — both are correctly minimizing their loss function, and that loss function
is nearly flat with respect to phase wherever amplitude is near zero. Checked directly for the
fixed autograd path too, not assumed by analogy: `Pearson r = -0.47` between its per-sample
phase error and `log(amplitude)` (`p=1.8e-28`) — the same mechanism, somewhat less severe than
GS's `r=-0.55`, consistent with autograd's lower overall RMS.

**Why blind TD-GSA is the wrong tool here, and what already is the right one:**
`projects/seals/inverse/inverse_scattering.py` (already in this package, see
`../README.md`) takes a **model-based** approach: it fits particle diameter directly
against the *known* Mie functional form (a derivative-free search), rather than
reconstructing phase blind. Because it uses the physical model as a strong prior, it is a
far better-conditioned inverse problem than 2-plane blind phase retrieval — and is the
already-validated path this project recommends for SEALS-specific inversion.

## 6. Next steps — now implemented and measured (`projects/seals/inverse/gs_multiplane.py`)

Both ideas below were prototypes in the original version of this section. They have
since been implemented (`gs_multiplane.py`) and measured (`tests/test_seals_to_tdgsa.py`,
`tests/test_gs_multiplane.py`) — one confirmed the prediction decisively, the other did
not, and both results are reported honestly per this project's own testing standard.

1. **More measurement diversity — confirmed, and it essentially solves the problem.**
   `gs_multiplane.retrieve_phase_n_plane` generalizes `gs_core`'s fixed 2-plane loop to
   N ≥ 2 dispersion planes (`build_gs_measurements_n`), reusing `gs_core.disperse` /
   `undisperse` / `apply_amplitude_constraint` directly. Measured on the same
   `(D1=6000, D2=-7000)` case as §4, adding one 3rd plane (`D3=12000`) drops the RMS
   phase error from **0.50 rad → 0.0014 rad** — a ~350× improvement, not incremental.
   Swept N=2..5: N=2 reproduces the known ~0.5 rad ambiguity; N=3, 4, 5 all land below
   0.002 rad. The general fix for underdetermined phase retrieval — a receiver with more
   independent observations resolving more ambiguity — is exactly what happened here.

2. **Hybrid physical-model regularization — implemented, honest null result.**
   `gs_multiplane.apply_prior_regularized_amplitude` blends the undispersed field's
   amplitude toward a Mie-fitted envelope (from `inverse_scattering.estimate_diameter`,
   fit on the native trace alone, unaware of the true diameter beyond a search bracket)
   wherever the native trace intensity is below a floor. Measured effect on the 2-plane
   case: **0.5044 rad → 0.5141 rad — no improvement** (`test_amplitude_prior_regularization_
   honest_null_result`). Diagnosed mechanism: the blend happens once per iteration in the
   undispersed domain, but the very next iteration's first per-plane projection
   re-imposes `|E_d| = sqrt(I_j)` exactly in the *dispersed* domain — the hard per-plane
   constraint overwrites whatever the prior contributed before that iteration completes.
   Fixing this would require blending the prior *inside* each per-plane projection (where
   dispersion has already mixed which native-domain sample maps to which dispersed-domain
   sample, so "where is the native trace weak" is no longer a simple per-index mask) — out
   of scope here since recommendation 1 above already resolves the ambiguity this was
   meant to address; not pursued further.
3. **Prefer `inverse_scattering.py` for SEALS-specific work.** Still true for recovering a
   physical parameter (particle diameter) rather than the raw field phase — but with N=3+
   planes now solving 2-plane blind GS's original problem almost completely, the choice
   between the two approaches is no longer forced by the phase-retrieval ambiguity that
   originally motivated this recommendation.

## 7. Cross-check against a historical, independent finding (even-degree phase ambiguity)

`notebooks/ece279_tdgsa_recreation.ipynb` reproduces a separate result from this project's
predecessor work (Yiming's MATLAB TDGSA code, Jalali Lab / ECE 279AS slide 23): blind TDGSA
fails on EVEN-degree phase polynomials (e.g. a quadratic chirp) because the intensity
constraint alone cannot distinguish `φ` from `-φ` (Hermitian symmetry); ODD-degree
polynomials (cubic) converge cleanly. That is a *different* candidate mechanism from §5's
amplitude-weakness diagnosis, worth checking against directly rather than assuming §5 is the
whole story.

`dgs.dispersion_gs_prototype.compare_phase` — used throughout this bridge to score every RMS
number above — already searches both signs (`φ` vs `-φ`) when aligning GS's recovered phase
against ground truth, so that specific ambiguity is already corrected for in every figure in
§4-§6. `diagnose_even_odd_ambiguity` (`seals_to_tdgsa.py`) checks the complementary question:
after that correction, is the *residual* error concentrated in the trace's even component
(reflected about its midpoint index) — the signature the historical failure mode predicts?

Measured (`test_even_odd_ambiguity_is_not_the_driver_here`): residual error even-part RMS
`1.102 rad` vs. odd-part RMS `1.017 rad` — a ratio of `1.08`, roughly balanced. **The
historical even-degree-phase mechanism is NOT what's driving the SEALS/Mie residual error** —
the true Mie phase itself isn't predominantly even either (even-part RMS `0.987` vs. odd-part
RMS `1.442`). §5's amplitude-weakness diagnosis remains the identified mechanism; this section
exists so that conclusion is checked against the alternative explanation directly, not just
assumed by default.

## 8. Does the (now-fixed) autograd path overfit noisy measurements?

A natural follow-up once §4's autograd bug was fixed: with a genuinely-working gradient-based
optimizer and no explicit regularization, does it eventually start fitting measurement noise
instead of the true phase — the classic "loss keeps dropping, but the answer gets worse"
overfitting curve familiar from ML training? Neither of the algorithms used elsewhere in this
report is susceptible by construction: classical GS is a hard alternating-projection method
(no notion of "training longer"), and the *noiseless* autograd runs in §4-§7 converge to a
fixed point almost immediately and stay there. Noisy data plus enough optimizer steps is a
different, genuinely new test.

`demonstrate_autograd_overfitting` (`seals_to_tdgsa.py`) adds multiplicative Gaussian noise
(`add_measurement_noise`) to the 2-plane SEALS measurements at a chosen `noise_std`, then runs
`phase_retrieval.retrieve_phase_with_history` — a new checkpointed variant of the autograd
optimizer — recording RMS-vs-*true* (noiseless) Mie phase at each checkpoint, separately from
the training loss it's actually minimizing.

**Measured, `tests/test_seals_to_tdgsa.py`:**

| `noise_std` | best RMS (checkpoint) | final (converged) RMS | overfitting gap |
|---|---|---|---|
| `0.05` (realistic, matches `inverse_scattering`'s convention) | `0.076 rad` @ step 10000 | `0.076 rad` @ step 10000 | `0.000 rad` — none |
| `0.3` | `0.424 rad` @ step 1500 | `0.427 rad` @ step 10000 | `+0.003 rad` |
| `0.6` | `0.481 rad` @ step 35 | `0.506 rad` @ step 10000 | `+0.025 rad` |
| `1.5` | `0.497 rad` @ step 3 | `0.563 rad` @ step 10000 | `+0.066 rad` |
| `3.0` | `0.498 rad` @ step 3 | `0.604 rad` @ step 10000 | `+0.105 rad` |

**Real overfitting, confirmed — but a secondary effect, not a dominant one.** At realistic
measurement noise (~5%), there is no overfitting: RMS-vs-truth improves monotonically to its
plateau. Above roughly 30% noise, a genuine gap opens and grows with noise level — training
longer (lower loss) makes the true-phase match measurably *worse*, the textbook overfitting
signature. The gap stays modest in absolute terms (a few hundredths of a radian even at 60%
noise) because the underlying 2-plane problem is already badly underdetermined (~0.5 rad
baseline error from §4-§5); overfitting compounds that structural limitation rather than being
the main story. Practical implication: at high measurement noise, early stopping (or the
autograd equivalent of a convergence criterion) is a real, cheap improvement — unlike §6's
amplitude-prior regularization, which measurably did *not* help.

## 9. Does the 3-plane fix (§6) survive measurement noise?

§6 fixed the *structural* 2-plane ambiguity (0.50 → 0.0014 rad) — but that number is noiseless
synthetic data. A natural, separate question: does 3-plane accuracy hold once the measurements
themselves are noisy? This is **not** the same question as §8's overfitting result — §8 asks
whether the *gradient-based* autograd path, given noisy *2-plane* (already broken) data, gets
worse the longer it trains; this asks whether the *projection-based* classical GS path, given
noisy *3-plane* (already fixed) data, loses accuracy simply from noisy inputs. Classical GS has
no "trained too long" regime at all — its hard `|E|=sqrt(I)` constraint reaches a fixed point
almost immediately regardless of noise or `n_iter` (self-consistency stays at `~1e-23`
throughout, confirmed in `projects/seals/inverse/noise_robustness.py` and
`tdgsa_noise_robustness.ipynb`), so there's nothing to overfit *to* — noise is injected
directly into the reconstruction instead, with no denoising step in between.

**Measured (`tests/test_noise_robustness.py`, same `D1=6000, D2=-7000, D3=12000`):**

| noise level | 3-plane RMS vs. true phase |
|---|---|
| 0% (noiseless) | `0.0014 rad` — the §6 result, unchanged |
| 5% (realistic photodiode) | `0.032 rad` — still good, not perfect |
| 15% | `0.154 rad` |
| 30% | `0.296 rad` |
| 60% | `0.439 rad` |
| 150% | `0.996 rad` — back to badly wrong |

**Conclusion: measurement diversity and measurement noise are independent axes, both now
characterized.** More dispersion planes fixed the structural ambiguity permanently — that
result stands regardless of noise. But it does not buy any noise immunity: accuracy degrades
roughly with noise level because classical GS has no built-in denoising. At realistic noise
(~5%), the 3-plane fix is still dramatically better than the 2-plane floor (`0.032` vs.
`~0.50` rad) — the fix remains genuinely useful, just not perfect. What would help further
(not implemented here): repeated-scan averaging before GS, a soft/noise-aware amplitude
constraint, or — consistent with §5's standing recommendation — `inverse_scattering.py`'s
model-based fit, which should be inherently more noise-robust since it uses the known Mie
functional form as a strong prior rather than reconstructing an unconstrained field blind.

## 10. Verification commands

```bash
py -3.12 -m pytest tests/test_seals_to_tdgsa.py tests/test_noise_robustness.py -v   # 20 tests, all passing (§4-§9 numbers above)
py -3.12 -m projects.seals.inverse.seals_to_tdgsa        # prints the live demo numbers
py -3.12 -m projects.seals.inverse.noise_robustness      # prints the §9 noise-sweep numbers
make seals-tdgsa                                          # same, via the Makefile (see below)
```

## 11. OUSD(R&E) CTA alignment

This bridge touches **Integrated Sensing and Cyber** (a passive optical particle-sizing
sensor) and **Trusted AI and Autonomy** (the autograd path is the same differentiable-
optics machinery `dgs.differentiable_optics_tutorial` uses elsewhere in this repo) —
consistent with `dgs/ousd_alignment.py`'s existing CTA table.
