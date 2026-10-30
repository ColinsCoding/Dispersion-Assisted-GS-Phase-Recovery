/-
Dispersion-assisted Gerchberg-Saxton phase retrieval: the machine-checkable pieces. Lean 4 + Mathlib.
Companion to notebooks/dispersion_phase_retrieval_lean.ipynb and dgs/gs_core.py.

The forward model (Jalali's time stretch): at each frequency nu the field is multiplied by
H_D(nu) = exp(i pi D nu^2). GS alternates between two measured intensities |H_D1 E|^2 and |H_D2 E|^2.

What is proved here:
  1. The dispersion phase pi D nu^2 has second derivative 2 pi D: D is the group-delay dispersion
  2. |H_D(nu)| = 1: dispersion changes only phase, so it conserves energy at every frequency
  3. Dispersion is a group: H_D1 H_D2 = H_(D1+D2) and H_D H_(-D) = 1 (undisperse inverts disperse)
  4. Unitary => distances are preserved: |H x - H y| = |x - y|
  5. The amplitude projection z -> a z/|z| is the NEAREST point with magnitude a
  6. The magnitude set {|z| = a} is not convex (midpoint of 1 and -1 is 0): why GS can stagnate
  7. Alternating nearest-point projections never increase the gap between the two sets
     (Fienup's error-reduction argument), stated for any metric space

What is NOT proved here: that GS converges to the TRUE phase (uniqueness), or anything about
noise. Item 7 assumes each step is an exact nearest-point projection; the extra unit-amplitude
step in gs_core.gs_iteration(unit_amplitude=True) adds a third projection that item 7 does not cover.

Check with (from a Lake project that depends on Mathlib):
    lake env lean DispersionPhaseRetrieval.lean
-/
import Mathlib.Analysis.SpecialFunctions.Complex.Circle
import Mathlib.Analysis.SpecialFunctions.Pow.Deriv
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Topology.MetricSpace.Basic
import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.FieldSimp

open Complex

namespace DAPR

/-- The dispersion transfer function H_D(nu) = exp(i pi D nu^2). -/
noncomputable def H (D ν : ℝ) : ℂ := exp (((Real.pi * D * ν ^ 2 : ℝ) : ℂ) * I)

/-! ## 1. The phase is quadratic: its second derivative is 2 pi D -/

theorem phase_hasDerivAt (D ν : ℝ) :
    HasDerivAt (fun ν => Real.pi * D * ν ^ 2) (2 * Real.pi * D * ν) ν := by
  have h := (hasDerivAt_pow 2 ν).const_mul (Real.pi * D)
  refine h.congr_deriv ?_
  norm_num
  ring

theorem phase_second_deriv (D ν : ℝ) :
    deriv (deriv (fun ν => Real.pi * D * ν ^ 2)) ν = 2 * Real.pi * D := by
  have hd : deriv (fun ν => Real.pi * D * ν ^ 2) = fun ν => 2 * Real.pi * D * ν := by
    funext x; exact (phase_hasDerivAt D x).deriv
  rw [hd]
  have h := (hasDerivAt_id ν).const_mul (2 * Real.pi * D)
  simp

/-! ## 2. Unit modulus: energy is conserved at every frequency -/

theorem H_norm (D ν : ℝ) : ‖H D ν‖ = 1 := by
  unfold H
  exact norm_exp_ofReal_mul_I _

theorem energy_conserved (D ν : ℝ) (X : ℂ) : ‖H D ν * X‖ = ‖X‖ := by
  rw [norm_mul, H_norm, one_mul]

/-! ## 3. Dispersion composes and inverts -/

theorem H_add (D1 D2 ν : ℝ) : H D1 ν * H D2 ν = H (D1 + D2) ν := by
  unfold H
  rw [← exp_add]
  congr 1
  push_cast
  ring

theorem H_inverse (D ν : ℝ) : H D ν * H (-D) ν = 1 := by
  rw [H_add, add_neg_cancel]
  unfold H
  simp

/-! ## 4. Unitary steps preserve distances -/

theorem dist_preserved (D ν : ℝ) (x y : ℂ) : ‖H D ν * x - H D ν * y‖ = ‖x - y‖ := by
  rw [← mul_sub, energy_conserved]

/-! ## 5. The amplitude projection is the nearest point on the circle |w| = a -/

theorem amplitude_projection_nearest (z w : ℂ) (a : ℝ) (hz0 : z ≠ 0) (hw : ‖w‖ = a) :
    ‖z - ((a / ‖z‖ : ℝ) : ℂ) * z‖ ≤ ‖z - w‖ := by
  have hz : 0 < ‖z‖ := norm_pos_iff.mpr hz0
  have h1 : z - ((a / ‖z‖ : ℝ) : ℂ) * z = ((1 - a / ‖z‖ : ℝ) : ℂ) * z := by
    push_cast; ring
  have h2 : |1 - a / ‖z‖| * ‖z‖ = |‖z‖ - a| := by
    rw [← abs_of_pos hz, ← abs_mul, abs_of_pos hz, sub_mul, div_mul_cancel₀ _ hz.ne', one_mul]
  rw [h1, norm_mul, Complex.norm_real, Real.norm_eq_abs, h2, ← hw]
  exact abs_norm_sub_norm_le z w

theorem projection_has_magnitude (z : ℂ) (a : ℝ) (hz0 : z ≠ 0) (ha : 0 ≤ a) :
    ‖((a / ‖z‖ : ℝ) : ℂ) * z‖ = a := by
  have hz : 0 < ‖z‖ := norm_pos_iff.mpr hz0
  rw [norm_mul, Complex.norm_real, Real.norm_eq_abs, abs_of_nonneg (div_nonneg ha hz.le),
    div_mul_cancel₀ _ hz.ne']

/-! ## 6. The magnitude constraint set is not convex (midpoint in linear algebra) -/

theorem magnitude_set_not_convex :
    ∃ x y : ℂ, ‖x‖ = 1 ∧ ‖y‖ = 1 ∧ ‖(x + y) / 2‖ ≠ 1 := by
  refine ⟨1, -1, by simp, by simp, ?_⟩
  simp

/-! ## 7. Alternating projections: the gap never grows (error reduction) -/

/-- x ∈ B, a ∈ A; x' is a nearest point of B to a, and a' is a nearest point of A to x'.
Then the new gap dist x' a' is at most the old gap dist x a. -/
theorem error_nonincreasing {X : Type*} [MetricSpace X] (A B : Set X) (x a x' a' : X)
    (hx : x ∈ B) (ha : a ∈ A)
    (hB : ∀ y ∈ B, dist x' a ≤ dist y a) (hA : ∀ z ∈ A, dist x' a' ≤ dist x' z) :
    dist x' a' ≤ dist x a :=
  calc dist x' a' ≤ dist x' a := hA a ha
    _ ≤ dist x a := hB x hx

end DAPR
