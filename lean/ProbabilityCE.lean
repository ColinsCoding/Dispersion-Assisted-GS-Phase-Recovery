/-
Probability for computer engineering, from the physics of the hardware. Lean 4 + Mathlib.
Companion to notebooks/probability_computer_engineering.ipynb and dgs/probability_ce.py.

What is proved here:
  1. Binomial bit errors: the probabilities C(n,k) p^k (1-p)^(n-k) add up to 1
  2. Union bound for a packet: 1 - (1-p)^n <= n p (Bernoulli's inequality)
  3. Variance identity: sum w (x - mu)^2 = sum w x^2 - mu^2 (what np.var computes)
  4. Markov's inequality for a finite distribution: a P(X >= a) <= E[X]
  5. Exponential arrivals are memoryless: e^(-l(s+t)) / e^(-l s) = e^(-l t)
  6. Fermi-Dirac symmetry: f(x) + f(-x) = 1 with f(x) = 1 / (1 + e^x)

Check with (from a Lake project that depends on Mathlib):
    lake env lean ProbabilityCE.lean
-/
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.Order.Chebyshev
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

namespace ProbCE

/-! ## 1. Binomial probabilities sum to 1 -/

theorem binomial_sum_one (p : ℝ) (n : ℕ) :
    ∑ k ∈ Finset.range (n + 1), p ^ k * (1 - p) ^ (n - k) * (n.choose k : ℝ) = 1 := by
  rw [← add_pow]
  simp

/-! ## 2. Union bound for a packet of n bits -/

theorem packet_union_bound (p : ℝ) (n : ℕ) (hp1 : p ≤ 1) :
    1 - (1 - p) ^ n ≤ n * p := by
  have h := one_add_mul_le_pow (show (-2 : ℝ) ≤ -p by linarith) n
  rw [← sub_eq_add_neg] at h
  linarith

/-! ## 3. Variance identity -/

theorem variance_identity {n : ℕ} (w x : Fin n → ℝ) (hw : ∑ i, w i = 1) (μ : ℝ)
    (hμ : μ = ∑ i, w i * x i) :
    ∑ i, w i * (x i - μ) ^ 2 = ∑ i, w i * x i ^ 2 - μ ^ 2 := by
  have h : ∀ i, w i * (x i - μ) ^ 2 = w i * x i ^ 2 - 2 * μ * (w i * x i) + μ ^ 2 * w i :=
    fun i => by ring
  simp only [h, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, hw, ← hμ]
  ring

/-! ## 4. Markov's inequality -/

theorem markov_finite {n : ℕ} (w X : Fin n → ℝ) (a : ℝ) (hw : ∀ i, 0 ≤ w i) (hX : ∀ i, 0 ≤ X i) :
    a * ∑ i, (if a ≤ X i then w i else 0) ≤ ∑ i, w i * X i := by
  rw [Finset.mul_sum]
  apply Finset.sum_le_sum
  intro i _
  split_ifs with h
  · nlinarith [hw i]
  · simp only [mul_zero]
    exact mul_nonneg (hw i) (hX i)

/-! ## 5. Exponential waiting times are memoryless -/

theorem memoryless (l s t : ℝ) :
    Real.exp (-l * (s + t)) / Real.exp (-l * s) = Real.exp (-l * t) := by
  rw [div_eq_iff (Real.exp_pos _).ne', ← Real.exp_add]
  ring_nf

/-! ## 6. Fermi-Dirac symmetry -/

theorem fermi_symmetry (x : ℝ) : 1 / (1 + Real.exp x) + 1 / (1 + Real.exp (-x)) = 1 := by
  have h1 : 0 < 1 + Real.exp x := by positivity
  have h2 : Real.exp (-x) = 1 / Real.exp x := by rw [Real.exp_neg, one_div]
  rw [h2]
  have hx : 0 < Real.exp x := Real.exp_pos x
  field_simp
  ring

end ProbCE
