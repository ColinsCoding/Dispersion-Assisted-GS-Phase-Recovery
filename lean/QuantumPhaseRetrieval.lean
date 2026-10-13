/-
Phase retrieval for a quantum wavefunction (Pauli's question, 1933): can ψ be recovered from
probability densities |ψ(x, t)|² measured at several times? Lean 4 + Mathlib.

Free evolution (Schrödinger, 1926) multiplies each momentum component by
  U(k, t) = exp(−i (ħ/m) k² t / 2),
the same operator as fiber dispersion exp(i β₂L ω²/2) with β₂L ↔ −(ħ/m) t.
-/
import Mathlib.Analysis.SpecialFunctions.Complex.Circle

open Complex

namespace QuantumPR

/-- the free propagator for one momentum component -/
noncomputable def U (hm k t : ℝ) : ℂ := exp (((-(hm * k ^ 2 * t / 2)) : ℝ) * I)

/-- it has length 1: probability is conserved (unitary evolution) -/
theorem U_unit (hm k t : ℝ) : ‖U hm k t‖ = 1 := norm_exp_ofReal_mul_I _

/-- evolving for t₁ then t₂ is evolving for t₁ + t₂ (the time-evolution group) -/
theorem U_compose (hm k t1 t2 : ℝ) : U hm k t1 * U hm k t2 = U hm k (t1 + t2) := by
  unfold U; rw [← Complex.exp_add]; congr 1; push_cast; ring

/-- evolving backwards undoes it: this is the "go back to t = 0" step inside Gerchberg-Saxton -/
theorem U_undo (hm k t : ℝ) : U hm k (-t) * U hm k t = 1 := by
  rw [U_compose, neg_add_cancel]; simp [U]

/-- complex conjugation reverses time: conj(U(t)) = U(−t). So the conjugate wavefunction's densities
    at times t equal ψ's densities at −t; measuring at t = 0 and at positive times only breaks the tie. -/
theorem conj_reverses_time (hm k t : ℝ) : (starRingEnd ℂ) (U hm k t) = U hm k (-t) := by
  unfold U; rw [← Complex.exp_conj]; congr 1
  simp only [map_mul, Complex.conj_ofReal, Complex.conj_I]; push_cast; ring

/-- a global phase is invisible in every density: |e^{iθ} ψ|² = |ψ|². In quantum mechanics it is
    also physically meaningless, so recovering ψ "up to a global phase" is recovering all of it. -/
theorem global_phase_invisible (ψ : ℂ) (θ : ℝ) : ‖exp ((θ : ℂ) * I) * ψ‖ ^ 2 = ‖ψ‖ ^ 2 := by
  rw [norm_mul, norm_exp_ofReal_mul_I, one_mul]

/-- the dictionary between quantum free evolution and fiber dispersion: the two phases agree when
    β₂L = −(ħ/m) t, with momentum k playing the role of angular frequency ω -/
theorem schrodinger_is_dispersion (hm k t : ℝ) :
    (-(hm * t)) * k ^ 2 / 2 = -(hm * k ^ 2 * t / 2) := by ring

/-- a Gaussian packet spreads: σ(t)² = σ₀²(1 + (ħt/(2mσ₀²))²) is never smaller than σ₀² -/
theorem gaussian_spreads (σ0 hm t : ℝ) : σ0 ^ 2 ≤ σ0 ^ 2 * (1 + (hm * t / (2 * σ0 ^ 2)) ^ 2) := by
  nlinarith [sq_nonneg (hm * t / (2 * σ0 ^ 2)), sq_nonneg σ0, mul_nonneg (sq_nonneg σ0) (sq_nonneg (hm * t / (2 * σ0 ^ 2)))]

end QuantumPR
