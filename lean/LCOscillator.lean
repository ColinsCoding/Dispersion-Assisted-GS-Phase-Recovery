/-
The LC circuit as a harmonic oscillator: the machine-checkable pieces. Lean 4 + Mathlib.
Companion to notebooks/lc_quantum_oscillator.ipynb.

What is proved here:
  1. The mapping: the mass-spring energy with m = L, k = 1/C, x = Q, p = Phi is the LC energy
  2. Energy conservation: if Q' = I and L I' = -Q/C then dE/dt = L I I' + Q Q'/C = 0
  3. The solution: Q(t) = Q0 cos(w t) has Q'' = -w^2 Q, so it solves L Q'' + Q/C = 0 when w^2 L C = 1
  4. Quantum ladder: E_(n+1) - E_n = hbar w, and E_0 = hbar w / 2 > 0
  5. Zero-point fluctuations: dQ^2 = hbar/(2Z), dPhi^2 = hbar Z/2  =>  (dQ dPhi)^2 = (hbar/2)^2
  6. Terahertz: the free-space wavelength at 1 THz is under 1 mm; time stretch M > 1 raises bandwidth

Check with (from a Lake project that depends on Mathlib):
    lake env lean LCOscillator.lean
-/
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv
import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.FieldSimp

namespace LCOsc

/-! ## 1. The mapping mass-spring -> LC -/

theorem energy_mapping (L C Q Φ : ℝ) :
    Φ ^ 2 / (2 * L) + (1 / C) * Q ^ 2 / 2 = Φ ^ 2 / (2 * L) + Q ^ 2 / (2 * C) := by
  ring

/-! ## 2. Energy conservation (time-translation symmetry) -/

/-- Qd stands for dQ/dt and Id for dI/dt. -/
theorem energy_rate_zero (L C Q I Qd Id : ℝ) (h1 : Qd = I) (h2 : L * Id = -Q / C) :
    L * I * Id + Q * Qd / C = 0 := by
  rw [h1]
  linear_combination I * h2

/-! ## 3. The cosine solves the LC equation -/

theorem hasDerivAt_cos_mul (w Q0 t : ℝ) :
    HasDerivAt (fun t => Q0 * Real.cos (w * t)) (Q0 * (-Real.sin (w * t) * w)) t := by
  have h := ((hasDerivAt_id t).const_mul w).cos
  simpa using h.const_mul Q0

theorem hasDerivAt_sin_mul (w Q0 t : ℝ) :
    HasDerivAt (fun t => Q0 * (-Real.sin (w * t) * w)) (Q0 * (-(Real.cos (w * t) * w) * w)) t := by
  have h := ((hasDerivAt_id t).const_mul w).sin
  have h2 := ((h.neg).mul_const w).const_mul Q0
  simpa using h2

theorem cos_second_deriv (w Q0 t : ℝ) :
    deriv (deriv (fun t => Q0 * Real.cos (w * t))) t = -(w ^ 2) * (Q0 * Real.cos (w * t)) := by
  have hd : deriv (fun t => Q0 * Real.cos (w * t)) = fun t => Q0 * (-Real.sin (w * t) * w) := by
    funext s; exact (hasDerivAt_cos_mul w Q0 s).deriv
  rw [hd, (hasDerivAt_sin_mul w Q0 t).deriv]
  ring

/-- With Q'' = -w^2 Q and w^2 L C = 1, the LC equation L Q'' + Q/C = 0 holds. -/
theorem lc_equation (L C w Q : ℝ) (hC : C ≠ 0) (hw : w ^ 2 * (L * C) = 1) :
    L * (-(w ^ 2) * Q) + Q / C = 0 := by
  have : L * (-(w ^ 2) * Q) + Q / C = Q / C * (1 - w ^ 2 * (L * C)) := by
    field_simp
    ring
  rw [this, hw]
  ring

/-! ## 4. The quantum ladder -/

theorem ladder_spacing (hb w : ℝ) (n : ℕ) :
    hb * w * ((n + 1 : ℕ) + 1 / 2) - hb * w * (n + 1 / 2) = hb * w := by
  push_cast
  ring

theorem zero_point_positive (hb w : ℝ) (hhb : 0 < hb) (hw : 0 < w) :
    0 < hb * w * ((0 : ℕ) + 1 / 2) := by
  push_cast
  positivity

/-! ## 5. Zero-point fluctuations sit at the uncertainty limit -/

theorem zero_point_product (hb Z dQ dP : ℝ) (hZ : Z ≠ 0)
    (hQ : dQ ^ 2 = hb / (2 * Z)) (hP : dP ^ 2 = hb * Z / 2) :
    (dQ * dP) ^ 2 = (hb / 2) ^ 2 := by
  rw [mul_pow, hQ, hP]
  field_simp

/-! ## 6. Terahertz numbers -/

/-- c / (1 THz) < 1 mm: a millimetre-sized circuit is not lumped at 1 THz. -/
theorem thz_wavelength_sub_mm : (299792458 : ℝ) / 10 ^ 12 < 1 / 1000 := by
  norm_num

theorem stretch_raises_bandwidth (B M : ℝ) (hB : 0 < B) (hM : 1 < M) : B < M * B := by
  nlinarith

end LCOsc
