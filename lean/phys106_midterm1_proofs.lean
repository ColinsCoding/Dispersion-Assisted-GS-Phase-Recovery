/-
PHYS 106 (Modern Physics) Midterm 1 -- proof obligations behind the corrections.
Lean 4 + Mathlib. Companion to notebooks/phys106_midterm1_corrections_sympy_lean.ipynb.

What these theorems are (and are not):
  * They prove ALGEBRA: isolations, residuals, and exact rational arithmetic.
  * They do NOT prove physics. "L = L0/gamma" is assumed as a model; Lean only checks
    that the rearrangements and numbers that follow from that model are right.
  * Relativity numbers use c = 1 units (velocities in units of c, distances in km,
    times as c*t in km), so every constant is an exact rational.

Check with (from a Lake project that depends on Mathlib):
    lake env lean phys106_midterm1_proofs.lean
-/
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

namespace Phys106

/-! ## 1. Special relativity (Q1.2 - Q1.4) -/

/-- The square matters: with beta = 4/5, 1 - beta^2 = 9/25. -/
theorem one_sub_beta_sq : (1 : ℝ) - (4/5)^2 = 9/25 := by norm_num

/-- The exam error dropped the square: 1 - beta = 1/5, which is NOT 9/25. -/
theorem dropped_square_is_wrong : (1 : ℝ) - 4/5 ≠ 9/25 := by norm_num

theorem sqrt_one_sub_beta_sq : Real.sqrt (1 - (4/5 : ℝ)^2) = 3/5 := by
  rw [one_sub_beta_sq, show (9/25 : ℝ) = (3/5)^2 by norm_num]
  exact Real.sqrt_sq (by norm_num)

/-- gamma(0.8 c) = 5/3 exactly. -/
theorem gamma_four_fifths : 1 / Real.sqrt (1 - (4/5 : ℝ)^2) = 5/3 := by
  rw [sqrt_one_sub_beta_sq]; norm_num

/-- Length-contraction isolation: L * gamma = L0 with gamma ≠ 0 gives L = L0 / gamma. -/
theorem length_contraction_isolate (L L0 γ : ℝ) (hγ : γ ≠ 0) (h : L * γ = L0) :
    L = L0 / γ :=
  eq_div_of_mul_eq hγ h

/-- Q1.2: a 600 km proper length seen at 0.8 c is 360 km. -/
theorem length_contraction_600km :
    (600 : ℝ) / (1 / Real.sqrt (1 - (4/5 : ℝ)^2)) = 360 := by
  rw [gamma_four_fifths]; norm_num

/-- Q1.3 (general): events simultaneous in one frame (dt = 0) and spatially separated
(dx ≠ 0) are NOT simultaneous in a frame moving at v ≠ 0. -/
theorem not_simultaneous (γ v c dx : ℝ) (hγ : γ ≠ 0) (hv : v ≠ 0) (hc : c ≠ 0)
    (hx : dx ≠ 0) : γ * (0 - v * dx / c^2) ≠ 0 := by
  rw [zero_sub, mul_neg, neg_ne_zero]
  exact mul_ne_zero hγ (div_ne_zero (mul_ne_zero hv hx) (pow_ne_zero 2 hc))

/-- Q1.3 numbers (c = 1, km): c*dt' = gamma*(c*dt - beta*dx) = -800 km, i.e. about -2.67 ms. -/
theorem simultaneity_numbers : (5/3 : ℝ) * (0 - 4/5 * 600) = -800 := by norm_num

/-- Q1.3: dx' = gamma*(dx - beta*c*dt) = 1000 km. -/
theorem dx_prime_numbers : (5/3 : ℝ) * (600 - 4/5 * 0) = 1000 := by norm_num

/-- The interval is invariant under a boost (c = 1), given gamma^2 (1 - beta^2) = 1. -/
theorem interval_invariant (γ β t x : ℝ) (h : γ^2 * (1 - β^2) = 1) :
    (γ * (t - β * x))^2 - (γ * (x - β * t))^2 = t^2 - x^2 := by
  have e : (γ * (t - β * x))^2 - (γ * (x - β * t))^2 = γ^2 * (1 - β^2) * (t^2 - x^2) := by
    ring
  rw [e, h, one_mul]

/-- Q1.3: both frames agree s^2 = -(600 km)^2 < 0 (spacelike), so s^2 cannot decide
simultaneity -- it is the same in every frame. -/
theorem interval_numbers : (-800 : ℝ)^2 - 1000^2 = 0^2 - 600^2 := by norm_num
theorem interval_spacelike : (0 : ℝ)^2 - 600^2 < 0 := by norm_num

/-- Q1.4: u = -0.4, v = +0.8 (units of c) gives u' = -10/11 ≈ -0.909. -/
theorem velocity_addition_numbers : ((-2/5 : ℝ) - 4/5) / (1 - (-2/5) * (4/5)) = -10/11 := by
  norm_num

/-- The exam error used u = +0.4 (wrong sign), which gives -10/17 ≈ -0.588 instead. -/
theorem velocity_addition_sign_error : ((2/5 : ℝ) - 4/5) / (1 - (2/5) * (4/5)) = -10/17 := by
  norm_num

/-- The corrected speed is still below c. -/
theorem velocity_addition_subluminal : (-1 : ℝ) < -10/11 ∧ (-10/11 : ℝ) < 1 := by norm_num

/-! ## 2. Photoelectric effect (Q2.1 - Q2.2) -/

/-- Required residual form of Einstein's equation K = h f - phi. -/
theorem photoelectric_residual (K h f φ : ℝ) (hK : K = h * f - φ) : K - h * f + φ = 0 := by
  rw [hK]; ring

/-- Work function isolation from K = h f - phi and K = e Vs. -/
theorem work_function_isolate (K h f φ e Vs : ℝ) (hK : K = h * f - φ) (hs : K = e * Vs) :
    φ = h * f - e * Vs := by
  linarith

/-- A physical work function is positive whenever the photon energy exceeds e*Vs. The exam
value phi = -9.05e-19 J violates this, which flags an error before any grading. -/
theorem work_function_pos (hf eVs : ℝ) (h : eVs < hf) : 0 < hf - eVs := by linarith

/-- Speed isolation: K = (1/2) m v^2, m > 0, v ≥ 0 gives v = sqrt(2K/m). -/
theorem vmax_isolate (K m v : ℝ) (hm : 0 < m) (hv : 0 ≤ v) (hK : K = 1/2 * m * v^2) :
    v = Real.sqrt (2 * K / m) := by
  have e : 2 * K / m = v^2 := by
    rw [hK, div_eq_iff hm.ne']; ring
  rw [e, Real.sqrt_sq hv]

/-! ## 3. Electromagnetism and photons (Q3.3 - Q3.4) -/

/-- Velocity-selector isolation (required form): E = v B with B ≠ 0 gives v = E / B. -/
theorem velocity_selector_isolate (E v B : ℝ) (hB : B ≠ 0) (h : E = v * B) : v = E / B :=
  eq_div_of_mul_eq hB h.symm

/-- Starting from the force balance q E = q v B (q ≠ 0, B ≠ 0). -/
theorem velocity_selector (q E v B : ℝ) (hq : q ≠ 0) (hB : B ≠ 0) (h : q * E = q * v * B) :
    v = E / B := by
  have h1 : E = v * B := mul_left_cancel₀ hq (by rw [h]; ring)
  exact velocity_selector_isolate E v B hB h1

/-- Q3.3 numbers: 200000 V/m / 0.030 T = 2e7/3 m/s ≈ 6.67e6 m/s. -/
theorem velocity_selector_numbers : (200000 : ℝ) / (3/100) = 20000000/3 := by norm_num

/-- Required residual form of the photon energy E = hbar * omega. -/
theorem photon_energy_residual (E ħ ω : ℝ) (h : E = ħ * ω) : E - ħ * ω = 0 := by
  rw [h]; ring

/-- Photon wavelength isolation: E = h c / lam with lam ≠ 0 and E ≠ 0 gives lam = h c / E. -/
theorem photon_wavelength_isolate (E h c lam : ℝ) (hlam : lam ≠ 0) (hE0 : E ≠ 0)
    (hE : E = h * c / lam) : lam = h * c / E := by
  have h1 : E * lam = h * c := by rw [hE]; field_simp
  rw [eq_div_iff hE0, mul_comm]; exact h1

/-- Hydrogen analogue (NOT the exam's diagram): a 10 eV electron cannot lift ground-state
hydrogen to n = 2, since that needs 13.6 (1 - 1/4) = 10.2 eV. -/
theorem hydrogen_10eV_below_first_excitation : (10 : ℝ) < 13.6 * (1 - 1/2^2) := by norm_num

end Phys106
