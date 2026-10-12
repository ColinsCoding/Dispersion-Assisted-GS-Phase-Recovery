/-
Phase retrieval (dispersion-assisted Gerchberg-Saxton), Lean 4 + Mathlib.
Why intensity loses the phase, why the GS magnitude step is the best possible correction,
and why the GS error can never increase.
-/
import Mathlib.Analysis.SpecialFunctions.Complex.Circle
import Mathlib.Analysis.Normed.Group.Basic
import Mathlib.Order.Monotone.Basic

open Complex

namespace PhaseRetrieval

/-! 1. The phase problem: a detector cannot see phase. -/

/-- e^{iφ} has length 1: a phase changes direction, never size -/
theorem phase_unit (φ : ℝ) : ‖exp (φ * I)‖ = 1 := norm_exp_ofReal_mul_I φ

/-- so multiplying the whole field by e^{iφ} leaves every intensity |E|² unchanged
    (the global-phase ambiguity: no algorithm can recover a constant phase offset) -/
theorem intensity_blind (E : ℂ) (φ : ℝ) : ‖exp (φ * I) * E‖ ^ 2 = ‖E‖ ^ 2 := by
  rw [norm_mul, phase_unit, one_mul]

/-- complex conjugation (time reversal) also leaves intensity unchanged: the sign-flip ambiguity -/
theorem intensity_blind_conj (E : ℂ) : ‖(starRingEnd ℂ) E‖ = ‖E‖ := Complex.norm_conj E

/-! 2. Dispersion is all-pass: its transfer function exp(iπDν²) has unit modulus,
    so it moves energy around in time but never adds or removes any. -/

theorem dispersion_all_pass (D ν : ℝ) : ‖exp (((Real.pi * D * ν ^ 2 : ℝ) : ℂ) * I)‖ = 1 :=
  norm_exp_ofReal_mul_I _

/-! 3. The GS magnitude step is the nearest point.
    GS replaces the current estimate z by a·z/|z|: the measured magnitude a, with z's phase kept.
    No other complex number w with |w| = a is closer to z. -/

/-- the correction moves z by exactly | |z| − a | -/
theorem projection_distance (z : ℂ) (a : ℝ) (hz : z ≠ 0) :
    ‖z - ((a / ‖z‖ : ℝ) : ℂ) * z‖ = |‖z‖ - a| := by
  have hn : 0 < ‖z‖ := norm_pos_iff.mpr hz
  have : z - ((a / ‖z‖ : ℝ) : ℂ) * z = (((1 - a / ‖z‖) : ℝ) : ℂ) * z := by push_cast; ring
  rw [this, norm_mul, Complex.norm_real, Real.norm_eq_abs, ← abs_of_pos hn, ← abs_mul, abs_of_pos hn]
  congr 1; field_simp

/-- and every other point with the measured magnitude is at least that far away -/
theorem projection_optimal (z w : ℂ) (a : ℝ) (hz : z ≠ 0) (hw : ‖w‖ = a) :
    ‖z - ((a / ‖z‖ : ℝ) : ℂ) * z‖ ≤ ‖z - w‖ := by
  rw [projection_distance z a hz, ← hw]
  exact abs_norm_sub_norm_le z w

/-! 4. The error never increases (the error-reduction property of Gerchberg-Saxton / Fienup).
    One GS half-step goes from an estimate x (which already satisfies constraint B) to its
    nearest point p in constraint set A, then to the nearest point q of p in B. Because x was a
    candidate in B, the new error ‖p − q‖ is at most the old one ‖x − p‖. -/

theorem error_reduction_step {V : Type*} [NormedAddCommGroup V] (B : Set V) (x p q : V)
    (hx : x ∈ B) (hq : ∀ b ∈ B, ‖p - q‖ ≤ ‖p - b‖) : ‖p - q‖ ≤ ‖x - p‖ := by
  rw [← norm_neg (x - p), neg_sub]
  exact hq x hx

/-- chaining the steps: if each iteration's error is at most the previous one, the error after
    n iterations is at most the starting error (it can stall, but it can never get worse) -/
theorem error_never_increases (e : ℕ → ℝ) (h : ∀ k, e (k + 1) ≤ e k) (n : ℕ) : e n ≤ e 0 := by
  have : Antitone e := antitone_nat_of_succ_le h
  exact this (Nat.zero_le n)

end PhaseRetrieval
