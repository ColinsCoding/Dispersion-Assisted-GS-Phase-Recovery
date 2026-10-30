/-
The 3D particle in a box, Psi(x, y, z, t): the machine-checkable pieces. Lean 4 + Mathlib.
Companion to notebooks/wavefunction_4d_rigorous.ipynb.

What is proved here:
  1. Calculus: d^2/dx^2 sin(k x) = -k^2 sin(k x)          (the spatial factor solves X'' = -k^2 X)
  2. Separation of variables: if X'' = -k^2 X, T' = -i w T and hbar w = hbar^2 k^2 / 2m, then
     X(x) T(t) satisfies i hbar d/dt (X T) = -(hbar^2/2m) d^2/dx^2 (X T) at each point (algebra over C)
  3. Energies E = E1 (n1^2 + n2^2 + n3^2): the 3-fold degeneracy and the beat gap E211 - E111 = 3 E1
  4. The beat repeats: cos(w (t + 2 pi / w)) = cos(w t) for w ≠ 0
  5. Normalization constant: sqrt(2/L)^2 = 2/L for L > 0
  6. Ground state obeys the uncertainty bound: pi^2/3 - 2 >= 1, so dx dp >= hbar/2

What is NOT proved here (SymPy shows it symbolically in the notebook): orthogonality integrals,
the continuity equation, and Ehrenfest's theorem for the full 3D superposition.

Check with (from a Lake project that depends on Mathlib):
    lake env lean Box3DWaveFunction.lean
-/
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.Real.Pi.Bounds
import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.FieldSimp

namespace Box3D

/-! ## 1. The spatial factor solves X'' = -k^2 X -/

theorem hasDerivAt_sin_mul (k x : ℝ) :
    HasDerivAt (fun x => Real.sin (k * x)) (Real.cos (k * x) * k) x := by
  simpa using ((hasDerivAt_id x).const_mul k).sin

theorem hasDerivAt_cos_mul (k x : ℝ) :
    HasDerivAt (fun x => Real.cos (k * x)) (-Real.sin (k * x) * k) x := by
  simpa using ((hasDerivAt_id x).const_mul k).cos

theorem deriv_sin_mul (k : ℝ) :
    deriv (fun x => Real.sin (k * x)) = fun x => Real.cos (k * x) * k := by
  funext x; exact (hasDerivAt_sin_mul k x).deriv

theorem sin_mul_second_deriv (k x : ℝ) :
    deriv (deriv (fun x => Real.sin (k * x))) x = -(k ^ 2) * Real.sin (k * x) := by
  rw [deriv_sin_mul]
  have h : HasDerivAt (fun x => Real.cos (k * x) * k) (-Real.sin (k * x) * k * k) x :=
    (hasDerivAt_cos_mul k x).mul_const k
  rw [h.deriv]; ring

/-- The wall conditions: sin(n pi x / L) vanishes at x = 0 and x = L for every natural n. -/
theorem box_walls (n : ℕ) (L : ℝ) (hL : L ≠ 0) :
    Real.sin (n * Real.pi / L * 0) = 0 ∧ Real.sin (n * Real.pi / L * L) = 0 := by
  refine ⟨by simp, ?_⟩
  rw [div_mul_cancel₀ _ hL]
  exact Real.sin_nat_mul_pi n

/-! ## 2. Separation of variables -/

/-- Pointwise TDSE for a separated solution X(x) T(t). X2 stands for X'', T1 for T'. -/
theorem separated_tdse (X X2 T T1 k w hb m : ℂ)
    (hX : X2 = -(k ^ 2) * X) (hT : T1 = -(Complex.I * w) * T)
    (hE : hb * w = hb ^ 2 * k ^ 2 / (2 * m)) :
    Complex.I * hb * (X * T1) = -(hb ^ 2 / (2 * m)) * (X2 * T) := by
  rw [hX, hT]
  linear_combination (X * T) * hE + (-(hb * w * X * T)) * Complex.I_sq

/-! ## 3. Energies -/

def energy (E1 : ℝ) (n1 n2 n3 : ℕ) : ℝ := E1 * ((n1 : ℝ) ^ 2 + (n2 : ℝ) ^ 2 + (n3 : ℝ) ^ 2)

theorem degenerate_211 (E1 : ℝ) : energy E1 2 1 1 = energy E1 1 2 1 ∧ energy E1 1 2 1 = energy E1 1 1 2 := by
  unfold energy; constructor <;> norm_num

theorem beat_gap (E1 : ℝ) : energy E1 2 1 1 - energy E1 1 1 1 = 3 * E1 := by
  unfold energy; norm_num; ring

/-! ## 4. The beat is periodic -/

theorem beat_periodic (w t : ℝ) (hw : w ≠ 0) :
    Real.cos (w * (t + 2 * Real.pi / w)) = Real.cos (w * t) := by
  rw [mul_add, mul_div_cancel₀ _ hw, Real.cos_add_two_pi]

/-! ## 5. Normalization constant -/

theorem norm_const_sq (L : ℝ) (hL : 0 < L) : Real.sqrt (2 / L) ^ 2 = 2 / L :=
  Real.sq_sqrt (div_pos two_pos hL).le

/-- In 3D the constant is (2/L)^(3/2): its square is (2/L)^3. -/
theorem norm_const_3d_sq (L : ℝ) (hL : 0 < L) : (Real.sqrt (2 / L) ^ 3) ^ 2 = (2 / L) ^ 3 := by
  rw [← pow_mul, show 3 * 2 = 2 * 3 by norm_num, pow_mul, norm_const_sq L hL]

/-! ## 6. Uncertainty for the ground state -/

/-- For psi_1 of a box, (dx dp / hbar)^2 = (pi^2/3 - 2) / 4. It is at least 1/4 because pi > 3. -/
theorem ground_state_uncertainty : (Real.pi ^ 2 / 3 - 2) / 4 ≥ 1 / 4 := by
  have h := Real.pi_gt_three
  have h9 : Real.pi ^ 2 > 9 := by nlinarith
  linarith

end Box3D
