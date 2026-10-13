/-
The math behind the three optical receivers, Lean 4 + Mathlib.
-/
import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.SpecialFunctions.Complex.Circle

open Complex

namespace Receivers

/-! Coherent receiver: a balanced photodiode pair measures the difference of two intensities.
    Mixing the signal E with a strong local oscillator LO, |LO + E|² − |LO − E|² = 4 Re(conj(LO)·E):
    the field itself (in phase), not just its intensity. -/
theorem balanced_detection (LO E : ℂ) :
    normSq (LO + E) - normSq (LO - E) = 4 * (starRingEnd ℂ LO * E).re := by
  simp [normSq_apply, mul_re, conj_re, conj_im]; ring

/-- the second balanced pair uses the LO shifted by 90° (multiplied by i) and measures the other half:
    4 Im(conj(LO)·E). Together the two pairs give the full complex field. -/
theorem quadrature_detection (LO E : ℂ) :
    normSq (I * LO + E) - normSq (I * LO - E) = 4 * (starRingEnd ℂ LO * E).im := by
  simp [normSq_apply, mul_re, mul_im, conj_re, conj_im]; ring

/-! Kramers-Kronig receiver: a carrier C is sent with the signal s. If the carrier is stronger than
    the signal, |s| < C, the received field C + s never passes through zero, so log(C + s) is well
    defined and its real and imaginary parts (log-amplitude and phase) form a Hilbert-transform pair.
    That is the "minimum-phase" condition the KK receiver needs. -/
theorem kk_never_zero (C : ℝ) (s : ℂ) (h : ‖s‖ < C) : (C : ℂ) + s ≠ 0 := by
  intro h0
  have : s = -(C : ℂ) := by linear_combination h0
  rw [this, norm_neg, Complex.norm_real, Real.norm_eq_abs] at h
  linarith [le_abs_self C]

/-- and it stays at least C − |s| away from zero, so the logarithm never blows up -/
theorem kk_margin (C : ℝ) (s : ℂ) : C - ‖s‖ ≤ ‖(C : ℂ) + s‖ := by
  have h := norm_sub_norm_le (C : ℂ) (-s)
  rw [sub_neg_eq_add, norm_neg, Complex.norm_real, Real.norm_eq_abs] at h
  linarith [le_abs_self C]

/-! Dispersion-assisted GS receiver: splitting the light into n branches gives each photodiode 1/n
    of the power; nothing is lost, but each detector sees 1/n of the signal (a 10·log₁₀ n dB penalty
    per detector, 7 dB for five branches). -/
theorem split_conserves (P : ℝ) (n : ℕ) (hn : n ≠ 0) : (n : ℝ) * (P / n) = P := by
  field_simp

/-- every receiver ends by undoing the fiber's dispersion digitally: applying exp(iθ) then exp(−iθ)
    returns the original field (the inverse filter is exact, with no loss) -/
theorem dispersion_undo (θ : ℝ) (E : ℂ) : exp (-(θ : ℂ) * I) * (exp ((θ : ℂ) * I) * E) = E := by
  rw [← mul_assoc, ← Complex.exp_add]; simp

end Receivers
