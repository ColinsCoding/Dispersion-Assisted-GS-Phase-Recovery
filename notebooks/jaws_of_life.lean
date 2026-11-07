import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Tactic

/-!
# Jaws of life: four facts about the rescue-tool linkage (companion to dgs/jaws_of_life.py)

Slider-crank: pivot O, ear E at radius r and angle psi = theta + delta, crosshead C on the axis
at distance c(psi) = r cos psi + sqrt(l^2 - r^2 sin^2 psi), link length l.
-/

/-- The crosshead distance used by the module. -/
noncomputable def crosshead (r l ψ : ℝ) : ℝ :=
  r * Real.cos ψ + Real.sqrt (l ^ 2 - r ^ 2 * Real.sin ψ ^ 2)

/-- Loop closure: c(psi) satisfies the law of cosines l^2 = c^2 + r^2 - 2 c r cos psi
(whenever the link can reach, l^2 >= r^2 sin^2 psi). -/
theorem crosshead_law_of_cosines (r l ψ : ℝ) (h : r ^ 2 * Real.sin ψ ^ 2 ≤ l ^ 2) :
    crosshead r l ψ ^ 2 + r ^ 2 - 2 * crosshead r l ψ * r * Real.cos ψ = l ^ 2 := by
  unfold crosshead
  have hs : Real.sqrt (l ^ 2 - r ^ 2 * Real.sin ψ ^ 2) ^ 2 = l ^ 2 - r ^ 2 * Real.sin ψ ^ 2 :=
    Real.sq_sqrt (by linarith)
  have hp := Real.sin_sq_add_cos_sq ψ
  linear_combination hs - r ^ 2 * hp

/-- Monotonic piston extension: over the working range 0 <= psi1 <= psi2 <= pi/2 the crosshead
moves toward the pivot as the jaws open, so x_p = c(0) - c(theta) never decreases. -/
theorem crosshead_antitone (r l ψ₁ ψ₂ : ℝ) (hr : 0 ≤ r)
    (h0 : 0 ≤ ψ₁) (h12 : ψ₁ ≤ ψ₂) (h2 : ψ₂ ≤ Real.pi / 2) :
    crosshead r l ψ₂ ≤ crosshead r l ψ₁ := by
  unfold crosshead
  have hpi : ψ₂ ≤ Real.pi := by linarith [Real.pi_pos]
  have hcos : Real.cos ψ₂ ≤ Real.cos ψ₁ := Real.cos_le_cos_of_nonneg_of_le_pi h0 hpi h12
  have hsin : Real.sin ψ₁ ≤ Real.sin ψ₂ :=
    Real.sin_le_sin_of_le_of_le_pi_div_two (by linarith [Real.pi_pos]) h2 h12
  have hs1 : 0 ≤ Real.sin ψ₁ := Real.sin_nonneg_of_nonneg_of_le_pi h0 (by linarith)
  have hsq : Real.sin ψ₁ ^ 2 ≤ Real.sin ψ₂ ^ 2 := pow_le_pow_left₀ hs1 hsin 2
  have hroot : Real.sqrt (l ^ 2 - r ^ 2 * Real.sin ψ₂ ^ 2)
      ≤ Real.sqrt (l ^ 2 - r ^ 2 * Real.sin ψ₁ ^ 2) := by
    apply Real.sqrt_le_sqrt
    nlinarith [sq_nonneg r]
  nlinarith [mul_le_mul_of_nonneg_left hcos hr]

/-- Virtual work trades force for travel: if the piston's work equals the tips' work and the tips
travel farther than the piston, the tip force is smaller than the piston force. -/
theorem lever_law (Fp Fs dx dw : ℝ) (hFp : 0 < Fp) (hdx : 0 < dx) (hlong : dx < dw)
    (hwork : Fp * dx = Fs * dw) : Fs < Fp := by
  have hdw : 0 < dw := by linarith
  by_contra hcon
  rw [not_lt] at hcon
  have : Fp * dw ≤ Fs * dw := mul_le_mul_of_nonneg_right hcon hdw.le
  nlinarith

/-- Cutter: the same blade torque gives more force closer to the pivot (F = T / d). -/
theorem cut_near_pivot (T d₁ d₂ : ℝ) (hT : 0 < T) (hd : 0 < d₁) (h : d₁ < d₂) :
    T / d₂ < T / d₁ :=
  div_lt_div_of_pos_left hT hd h
