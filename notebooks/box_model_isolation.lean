import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Linarith

open Filter Topology

/-- Box levels E_n = n^2 E_1: neighbouring levels differ by (2n+1) E_1 (every gap is different). -/
theorem box_gap (E1 : ℝ) (n : ℕ) : ((n:ℝ) + 1) ^ 2 * E1 - (n:ℝ) ^ 2 * E1 = (2 * n + 1) * E1 := by
  ring

/-- Box transitions n -> n+1 have distinct frequencies: (2n+1) = (2m+1) forces n = m (addressable qudit). -/
theorem box_gaps_distinct (n m : ℕ) (h : 2 * n + 1 = 2 * m + 1) : n = m := by
  omega

/-- Correspondence principle: relative level spacing (2n+1)/n^2 -> 0 as n -> infinity. -/
theorem correspondence_spacing :
    Tendsto (fun n : ℕ => (2 * (n:ℝ) + 1) / (n:ℝ) ^ 2) atTop (𝓝 0) := by
  have h1 : Tendsto (fun n : ℕ => (1:ℝ) / (n:ℝ)) atTop (𝓝 0) := tendsto_one_div_atTop_nhds_zero_nat
  have h2 : Tendsto (fun n : ℕ => 2 * ((1:ℝ) / n) + ((1:ℝ) / n) * ((1:ℝ) / n)) atTop (𝓝 (2 * 0 + 0 * 0)) :=
    (h1.const_mul 2).add (h1.mul h1)
  simp only [mul_zero, add_zero] at h2
  refine h2.congr' ?_
  filter_upwards [eventually_ne_atTop 0] with n hn
  have : (n:ℝ) ≠ 0 := by exact_mod_cast hn
  field_simp

/-- Undamped isolator: transmissibility |1/(1 - r^2)| < 1 exactly when r^2 > 2 (drive above sqrt 2 x resonance). -/
theorem isolation_above_sqrt2 (r2 : ℝ) (h : 2 < r2) : |1 / (1 - r2)| < 1 := by
  have hneg : 1 - r2 < 0 := by linarith
  rw [abs_div, abs_one, abs_of_neg hneg, div_lt_one (by linarith)]
  linarith

/-- Harmonic levels are equally spaced: every n -> n+1 gap is the same hbar*omega (not addressable). -/
theorem harmonic_gap (hw : ℝ) (n : ℕ) : ((n:ℝ) + 1 + 1/2) * hw - ((n:ℝ) + 1/2) * hw = hw := by
  ring
