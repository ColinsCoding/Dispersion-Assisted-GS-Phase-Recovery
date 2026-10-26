import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum

open Filter Topology

/-- The forward difference quotient of x^2 is exactly 2x + h. -/
theorem diff_quot_sq (x h : ℝ) (hh : h ≠ 0) : ((x + h) ^ 2 - x ^ 2) / h = 2 * x + h := by
  field_simp
  ring

/-- Calculus limit: the difference quotient of x^2 tends to 2x as h → 0 (h ≠ 0). -/
theorem limit_diff_quot_sq (x : ℝ) :
    Tendsto (fun h : ℝ => ((x + h) ^ 2 - x ^ 2) / h) (𝓝[≠] 0) (𝓝 (2 * x)) := by
  have hc : Tendsto (fun h : ℝ => 2 * x + h) (𝓝[≠] 0) (𝓝 (2 * x)) := by
    have : Tendsto (fun h : ℝ => 2 * x + h) (𝓝 0) (𝓝 (2 * x + 0)) :=
      (continuous_const.add continuous_id).tendsto 0
    simpa using this.mono_left nhdsWithin_le_nhds
  refine hc.congr' ?_
  filter_upwards [self_mem_nhdsWithin] with h hh
  exact (diff_quot_sq x h hh).symm

/-- The 3-point second difference (the finite-difference kinetic term) is exact on quadratics. -/
theorem second_diff_quadratic (a b c x h : ℝ) (hh : h ≠ 0) :
    ((a * (x + h) ^ 2 + b * (x + h) + c) - 2 * (a * x ^ 2 + b * x + c)
      + (a * (x - h) ^ 2 + b * (x - h) + c)) / h ^ 2 = 2 * a := by
  field_simp
  ring

/-- Infinite-well levels: E_n = n^2 E_1, so the ratio E_2 / E_1 = 4 (the classifier's shape feature). -/
theorem box_ratio (E1 : ℝ) (h1 : E1 ≠ 0) : (2 : ℝ) ^ 2 * E1 / ((1 : ℝ) ^ 2 * E1) = 4 := by
  field_simp
  norm_num

/-- Harmonic levels E_n = (n + 1/2) ħω: the ratio E_1 / E_0 = 3. -/
theorem qho_ratio (hw : ℝ) (h : hw ≠ 0) : ((1 : ℝ) + 1 / 2) * hw / (((0 : ℝ) + 1 / 2) * hw) = 3 := by
  field_simp
  norm_num

/-- Interferometric height: h = λ φ / (4π) is linear in φ, so a phase error δφ gives height error λ δφ / (4π). -/
theorem height_error_linear (lam phi dphi : ℝ) :
    lam * (phi + dphi) / (4 * Real.pi) - lam * phi / (4 * Real.pi) = lam * dphi / (4 * Real.pi) := by
  ring
