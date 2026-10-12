/-
Undergraduate research workbook (time-stretch / photonics themes), Batch 1:
Lean 4 + Mathlib formalizations of representative claims.
Each theorem is a mathematical abstraction of a problem; the physics assumptions are
stated as hypotheses. Problem IDs refer to the workbook (JW-001 ... JW-050).
-/
import Mathlib.Analysis.SpecialFunctions.Log.Base
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

open Real
-- some hypotheses are physical assumptions kept for clarity even when a proof does not use them
set_option linter.unusedVariables false

namespace JW

/-! JW-001: wavelength and frequency, c = λ f. If λ doubles (same medium), f halves. -/
theorem freq_of_wavelength {c lam f : ℝ} (hlam : lam ≠ 0) (h : c = lam * f) : f = c / lam := by
  field_simp; linarith [h]

theorem double_wavelength_halves_freq {c lam : ℝ} (hlam : lam ≠ 0) :
    c / (2 * lam) = (c / lam) / 2 := by
  field_simp

/-! JW-008: decibels add when gains multiply (both gains positive). -/
theorem dB_mul {g1 g2 : ℝ} (h1 : 0 < g1) (h2 : 0 < g2) :
    10 * logb 10 (g1 * g2) = 10 * logb 10 g1 + 10 * logb 10 g2 := by
  rw [logb_mul h1.ne' h2.ne']; ring

/-! JW-020: Gaussian pulse broadening T(z) = T0 * sqrt(1 + (z/LD)^2).
    The pulse never gets shorter than T0 (unchirped input), and it doubles at z = sqrt 3 * LD. -/
theorem broadening_ge {T0 z LD : ℝ} (hT0 : 0 ≤ T0) : T0 ≤ T0 * sqrt (1 + (z / LD) ^ 2) := by
  have h1 : 1 ≤ sqrt (1 + (z / LD) ^ 2) := by
    have h := sqrt_le_sqrt (show (1 : ℝ) ≤ 1 + (z / LD) ^ 2 by nlinarith [sq_nonneg (z / LD)])
    rwa [sqrt_one] at h
  nlinarith

theorem broadening_doubles {T0 LD : ℝ} (hLD : LD ≠ 0) :
    T0 * sqrt (1 + (sqrt 3 * LD / LD) ^ 2) = 2 * T0 := by
  have h3 : (sqrt 3 * LD / LD) ^ 2 = 3 := by
    rw [mul_div_assoc, div_self hLD, mul_one, sq_sqrt (by norm_num)]
  rw [h3, show (1 : ℝ) + 3 = 2 ^ 2 by norm_num, sqrt_sq (by norm_num)]; ring

/-! JW-024: attenuation ODE dP/dz = -α P is solved by P(z) = P0 exp(-α z). -/
theorem attenuation_solves_ode (P0 α z : ℝ) :
    HasDerivAt (fun z => P0 * exp (-α * z)) (-α * (P0 * exp (-α * z))) z := by
  have h : HasDerivAt (fun x => -α * x) (-α) z := by
    simpa using (hasDerivAt_id z).const_mul (-α)
  have h2 := h.exp.const_mul P0
  exact h2.congr_deriv (by ring)

/-! JW-032: time-stretch factor M = 1 + (D2 L2)/(D1 L1) is at least 1
    when both dispersion-length products are positive (same sign). -/
theorem stretch_factor_ge_one {d1 d2 : ℝ} (h1 : 0 < d1) (h2 : 0 ≤ d2) : 1 ≤ 1 + d2 / d1 := by
  have : 0 ≤ d2 / d1 := div_nonneg h2 h1.le
  linarith

/-! JW-033: Bayes' rule. The positive predictive value of a screen is at most 1, and
    it equals sens*prev / (sens*prev + (1-spec)*(1-prev)). -/
theorem ppv_le_one {s p q : ℝ} (hs : 0 ≤ s) (hp : 0 ≤ p) (hp1 : p ≤ 1) (hq1 : q ≤ 1)
    (hpos : 0 < s * p + (1 - q) * (1 - p)) :
    s * p / (s * p + (1 - q) * (1 - p)) ≤ 1 := by
  rw [div_le_one hpos]
  nlinarith [mul_nonneg (sub_nonneg.mpr hq1) (sub_nonneg.mpr hp1)]

/-! JW-034: rare-cell sampling. Probability of at least one rare cell in n independent draws,
    1 - (1-p)^n, is at most n p (union bound, via Bernoulli's inequality). -/
theorem at_least_one_le_np {p : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (n : ℕ) :
    1 - (1 - p) ^ n ≤ n * p := by
  have hb := one_add_mul_le_pow (show (-2 : ℝ) ≤ -p by linarith) n
  have : (1 : ℝ) + n * (-p) = 1 - n * p := by ring
  rw [this, show (1 : ℝ) + -p = 1 - p by ring] at hb
  linarith

/-! JW-038: the exponential waiting time is memoryless:
    P(T > s + t) / P(T > s) = P(T > t) with survival function exp(-r t). -/
theorem exp_memoryless (r s t : ℝ) : exp (-r * (s + t)) / exp (-r * s) = exp (-r * t) := by
  rw [← exp_sub]; congr 1; ring

/-! JW-043: exponential intensity statistics: P(I > 8 <I>) = exp(-8) < 1/256 (since e ≥ 2). -/
theorem exceed_8_mean_small : exp (-8 : ℝ) ≤ 1 / 256 := by
  have he : (2 : ℝ) ≤ exp 1 := by linarith [add_one_le_exp (1 : ℝ)]
  have h8 : (2 : ℝ) ^ 8 ≤ exp 1 ^ 8 := pow_le_pow_left₀ (by norm_num) he 8
  rw [← exp_nat_mul] at h8
  rw [exp_neg, one_div]
  apply inv_anti₀ (by norm_num)
  norm_num at h8 ⊢; linarith

/-! JW-047: numerical aperture NA = sqrt(n1^2 - n2^2) is positive exactly when the core index
    exceeds the cladding index (both positive). -/
theorem na_pos {n1 n2 : ℝ} (h2 : 0 < n2) (h12 : n2 < n1) : 0 < sqrt (n1 ^ 2 - n2 ^ 2) := by
  apply sqrt_pos.mpr; nlinarith

end JW
