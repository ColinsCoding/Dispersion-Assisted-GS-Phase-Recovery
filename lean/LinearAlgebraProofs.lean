/-
Linear algebra proofs for engineering: matrices, a market model, and attention. Lean 4 + Mathlib.
Companion to notebooks/linear_algebra_proofs.ipynb and dgs/linear_algebra_proofs.py.

What is proved here:
  1. Matrix rules: det(AB) = det A det B, (AB)^T = B^T A^T, tr(AB) = tr(BA), the 2x2 det formula
  2. A kernel (null space) is a subspace: the midpoint of two solutions of A v = 0 is a solution
     (compare DispersionPhaseRetrieval.lean: the magnitude set |z| = 1 is NOT closed under midpoints)
  3. Cauchy-Schwarz in R^2, from the Lagrange identity
  4. Projections: if P P = P then (1 - P)(1 - P) = 1 - P (the complementary projection)
  5. Supply and demand: linear demand Q = a - b P and supply Q = c + d P meet at
     P* = (a - c)/(b + d); a demand shift up by delta raises P* by delta/(b + d) > 0
  6. Softmax (the attention weights): the weights are positive, sum to 1, and adding a constant
     to every score changes nothing (why torch subtracts the max before exp)

Check with (from a Lake project that depends on Mathlib):
    lake env lean LinearAlgebraProofs.lean
-/
import Mathlib.LinearAlgebra.Matrix.Determinant.Basic
import Mathlib.LinearAlgebra.Matrix.Trace
import Mathlib.LinearAlgebra.Matrix.Notation
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Positivity

open Matrix

namespace LinAlg

/-! ## 1. Matrix rules -/

theorem det_product {n : Type*} [Fintype n] [DecidableEq n] (A B : Matrix n n ℝ) :
    (A * B).det = A.det * B.det :=
  Matrix.det_mul A B

theorem transpose_product {n : Type*} [Fintype n] (A B : Matrix n n ℝ) :
    (A * B)ᵀ = Bᵀ * Aᵀ :=
  Matrix.transpose_mul A B

theorem trace_cyclic {n : Type*} [Fintype n] (A B : Matrix n n ℝ) :
    (A * B).trace = (B * A).trace :=
  Matrix.trace_mul_comm A B

theorem det_two_by_two (a b c d : ℝ) : !![a, b; c, d].det = a * d - b * c :=
  Matrix.det_fin_two_of a b c d

/-! ## 2. The null space is closed under midpoints -/

theorem kernel_midpoint {m n : Type*} [Fintype n] (A : Matrix m n ℝ) (v w : n → ℝ)
    (hv : A *ᵥ v = 0) (hw : A *ᵥ w = 0) : A *ᵥ ((1 / 2 : ℝ) • (v + w)) = 0 := by
  rw [Matrix.mulVec_smul, Matrix.mulVec_add, hv, hw]
  simp

/-! ## 3. Cauchy-Schwarz in the plane -/

theorem cauchy_schwarz_2d (a b c d : ℝ) : (a * c + b * d) ^ 2 ≤ (a ^ 2 + b ^ 2) * (c ^ 2 + d ^ 2) := by
  nlinarith [sq_nonneg (a * d - b * c)]

/-! ## 4. Complementary projection -/

theorem complement_projection {R : Type*} [Ring R] (P : R) (h : P * P = P) :
    (1 - P) * (1 - P) = 1 - P := by
  simp only [sub_mul, mul_sub, one_mul, mul_one, h]
  abel

/-! ## 5. Supply and demand -/

theorem equilibrium_price (a b c d P Q : ℝ) (hbd : b + d ≠ 0)
    (hD : Q = a - b * P) (hS : Q = c + d * P) : P = (a - c) / (b + d) := by
  rw [eq_div_iff hbd]
  linear_combination hD - hS

theorem demand_shift_raises_price (a b c d δ : ℝ) (hb : 0 < b) (hd : 0 < d) (hδ : 0 < δ) :
    (a + δ - c) / (b + d) - (a - c) / (b + d) = δ / (b + d) ∧ 0 < δ / (b + d) := by
  refine ⟨by ring, by positivity⟩

/-! ## 6. Softmax: the attention weights -/

theorem softmax_pos {n : ℕ} [NeZero n] (x : Fin n → ℝ) (i : Fin n) :
    0 < Real.exp (x i) / ∑ j, Real.exp (x j) :=
  div_pos (Real.exp_pos _) (Finset.sum_pos (fun j _ => Real.exp_pos (x j)) Finset.univ_nonempty)

theorem softmax_sum_one {n : ℕ} [NeZero n] (x : Fin n → ℝ) :
    ∑ i, Real.exp (x i) / ∑ j, Real.exp (x j) = 1 := by
  rw [← Finset.sum_div]
  exact div_self (Finset.sum_pos (fun j _ => Real.exp_pos (x j)) Finset.univ_nonempty).ne'

theorem softmax_shift {n : ℕ} (x : Fin n → ℝ) (c : ℝ) (i : Fin n) :
    Real.exp (x i + c) / ∑ j, Real.exp (x j + c) = Real.exp (x i) / ∑ j, Real.exp (x j) := by
  simp only [Real.exp_add]
  rw [← Finset.sum_mul, mul_div_mul_right _ _ (Real.exp_pos c).ne']

end LinAlg
