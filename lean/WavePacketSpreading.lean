/-
How a free Gaussian wave packet spreads, with the derivatives written out. Lean 4 + Mathlib.
With hm = ħ/m and starting width σ₀, the position variance is
    var(t) = σ₀² + (hm·t / (2σ₀))²        (equivalently σ₀²·(1 + (hm·t/(2σ₀²))²)).
-/
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Tactic.FieldSimp

set_option linter.unusedVariables false

namespace WavePacket

noncomputable def var (σ0 hm t : ℝ) : ℝ := σ0 ^ 2 + (hm * t / (2 * σ0)) ^ 2

/-- the two common ways of writing the width agree -/
theorem var_other_form (σ0 hm t : ℝ) (h : σ0 ≠ 0) :
    var σ0 hm t = σ0 ^ 2 * (1 + (hm * t / (2 * σ0 ^ 2)) ^ 2) := by
  unfold var; field_simp

/-- var is a parabola in t: var(t) = σ₀² + a·t², with a = (hm/(2σ₀))² -/
theorem var_parabola (σ0 hm : ℝ) :
    (fun t => var σ0 hm t) = fun t => σ0 ^ 2 + (hm / (2 * σ0)) ^ 2 * t ^ 2 := by
  funext t; unfold var; ring

/-- first derivative, written out: d/dt [σ₀² + a·t²] = 0 + a·(2t) = hm²·t / (2σ₀²) -/
theorem var_deriv (σ0 hm t : ℝ) (h : σ0 ≠ 0) :
    HasDerivAt (fun t => var σ0 hm t) (hm ^ 2 * t / (2 * σ0 ^ 2)) t := by
  rw [var_parabola]
  have hp : HasDerivAt (fun t => (hm / (2 * σ0)) ^ 2 * t ^ 2) ((hm / (2 * σ0)) ^ 2 * (2 * t)) t := by
    have := (hasDerivAt_pow 2 t).const_mul ((hm / (2 * σ0)) ^ 2)
    simpa using this
  have hc := HasDerivAt.add (hasDerivAt_const t (σ0 ^ 2)) hp
  -- convert leaves side goals: the two ways of writing the function agree (rfl), and the slope arithmetic
  convert hc using 1 <;> first | rfl | (field_simp; ring)

/-- second derivative: the slope itself grows at the constant rate hm² / (2σ₀²), so var is a parabola in t -/
theorem var_second_deriv (σ0 hm t : ℝ) :
    HasDerivAt (fun t => hm ^ 2 * t / (2 * σ0 ^ 2)) (hm ^ 2 / (2 * σ0 ^ 2)) t := by
  simpa using ((hasDerivAt_id t).const_mul (hm ^ 2)).div_const (2 * σ0 ^ 2)

/-- the packet is narrowest at t = 0 (where the derivative is zero) ... -/
theorem narrowest_at_zero (σ0 hm t : ℝ) : var σ0 hm 0 ≤ var σ0 hm t := by
  unfold var; simp; positivity

theorem slope_zero_at_zero (σ0 hm : ℝ) : hm ^ 2 * 0 / (2 * σ0 ^ 2) = 0 := by simp

/-- ... and it spreads the same way forward and backward in time -/
theorem time_symmetric (σ0 hm t : ℝ) : var σ0 hm (-t) = var σ0 hm t := by
  unfold var; ring

/-- at long times the width grows linearly ("ballistic" spreading): σ(t) ≥ |hm·t / (2σ₀)| -/
theorem width_at_least_linear (σ0 hm t : ℝ) : |hm * t / (2 * σ0)| ≤ Real.sqrt (var σ0 hm t) := by
  rw [← Real.sqrt_sq_eq_abs]
  apply Real.sqrt_le_sqrt
  unfold var; nlinarith [sq_nonneg σ0]

/-- a narrower start spreads faster: for t ≥ 0, a smaller σ₀ has a larger growth rate of the variance
    (the uncertainty principle: tight in position means wide in momentum) -/
theorem narrower_spreads_faster (σ0 σ1 hm t : ℝ) (h0 : 0 < σ0) (h01 : σ0 < σ1) (ht : 0 ≤ t) :
    hm ^ 2 * t / (2 * σ1 ^ 2) ≤ hm ^ 2 * t / (2 * σ0 ^ 2) := by
  have hnum : 0 ≤ hm ^ 2 * t := mul_nonneg (sq_nonneg hm) ht
  apply div_le_div_of_nonneg_left hnum (by positivity)
  nlinarith

end WavePacket
