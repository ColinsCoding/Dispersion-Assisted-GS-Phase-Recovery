/-
EEE 117 resistive circuits: the machine-checkable algebra. Lean 4 + Mathlib.
Companion to notebooks/eee117_circuits_sympy_lean.ipynb and dgs/eee117_circuits.py.

What is proved here (resistances are positive reals):
  1. Ohm's law and power: P = V I with V = I R gives P = I^2 R and P = V^2 / R
  2. Series and parallel: two resistors in parallel equal R1 R2 / (R1 + R2), less than either one
  3. Voltage divider: the two drops add to Vs; current divider: branch currents add to Is
  4. Wheatstone bridge: R1 R4 = R2 R3 implies the bridge output is zero
  5. Delta-to-wye: the wye seen between terminals 1 and 2 equals R12 parallel (R23 + R31)
  6. Ideal op-amps: inverting gain -Rf/Rin, non-inverting gain 1 + Rf/Rg
  7. Energy: in a series circuit the source power equals the resistor powers (Tellegen)
  8. Binary: a 3-bit R-2R ladder output is Vref (4 b2 + 2 b1 + b0) / 8

Check with (from a Lake project that depends on Mathlib):
    lake env lean EEE117Circuits.lean
-/
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Positivity

namespace EEE117

/-! ## 1. Ohm's law and power -/

theorem power_I2R (V I R : ℝ) (h : V = I * R) : V * I = I ^ 2 * R := by
  rw [h]; ring

theorem power_V2R (V I R : ℝ) (hR : R ≠ 0) (h : V = I * R) : V * I = V ^ 2 / R := by
  rw [h]; field_simp

/-! ## 2. Series and parallel -/

theorem parallel_two (R1 R2 : ℝ) (h1 : 0 < R1) (h2 : 0 < R2) :
    1 / (1 / R1 + 1 / R2) = R1 * R2 / (R1 + R2) := by
  field_simp
  ring

theorem parallel_less (R1 R2 : ℝ) (h1 : 0 < R1) (h2 : 0 < R2) :
    R1 * R2 / (R1 + R2) < R1 := by
  rw [div_lt_iff₀ (by positivity)]
  nlinarith

/-! ## 3. Dividers -/

theorem voltage_divider_sum (Vs R1 R2 : ℝ) (h1 : 0 < R1) (h2 : 0 < R2) :
    Vs * R1 / (R1 + R2) + Vs * R2 / (R1 + R2) = Vs := by
  field_simp

theorem current_divider_sum (Is R1 R2 : ℝ) (h1 : 0 < R1) (h2 : 0 < R2) :
    Is * R2 / (R1 + R2) + Is * R1 / (R1 + R2) = Is := by
  field_simp; ring

/-- Both branches of a current divider see the same voltage. -/
theorem current_divider_same_voltage (Is R1 R2 : ℝ) :
    (Is * R2 / (R1 + R2)) * R1 = (Is * R1 / (R1 + R2)) * R2 := by
  ring

/-! ## 4. Wheatstone bridge -/

theorem wheatstone_balance (Vs R1 R2 R3 R4 : ℝ) (h1 : 0 < R1) (h2 : 0 < R2) (h3 : 0 < R3)
    (h4 : 0 < R4) (hbal : R1 * R4 = R2 * R3) :
    Vs * R2 / (R1 + R2) - Vs * R4 / (R3 + R4) = 0 := by
  have hA : R1 + R2 ≠ 0 := by positivity
  have hB : R3 + R4 ≠ 0 := by positivity
  rw [sub_eq_zero, div_eq_div_iff hA hB]
  linear_combination Vs * hbal * (-1)

/-! ## 5. Delta-to-wye -/

theorem delta_wye_terminals_12 (R12 R23 R31 : ℝ) (h1 : 0 < R12) (h2 : 0 < R23) (h3 : 0 < R31) :
    R12 * R31 / (R12 + R23 + R31) + R12 * R23 / (R12 + R23 + R31)
      = R12 * (R23 + R31) / (R12 + (R23 + R31)) := by
  rw [← add_assoc]
  field_simp
  ring

/-! ## 6. Ideal op-amps -/

/-- Inverting: V- = 0 (virtual ground), KCL at V-: Vin/Rin + Vout/Rf = 0. -/
theorem inverting_gain (Vin Vout Rin Rf : ℝ) (hin : Rin ≠ 0) (hf : Rf ≠ 0)
    (kcl : Vin / Rin + Vout / Rf = 0) : Vout = -(Rf / Rin) * Vin := by
  field_simp at kcl ⊢
  linear_combination kcl

/-- Non-inverting: V- = Vin, KCL at V-: Vin/Rg + (Vin - Vout)/Rf = 0. -/
theorem noninverting_gain (Vin Vout Rg Rf : ℝ) (hg : Rg ≠ 0) (hf : Rf ≠ 0)
    (kcl : Vin / Rg + (Vin - Vout) / Rf = 0) : Vout = (1 + Rf / Rg) * Vin := by
  field_simp at kcl ⊢
  linear_combination -kcl

/-! ## 7. Energy conservation -/

theorem series_power_balance (Vs I R1 R2 : ℝ) (h : Vs = I * (R1 + R2)) :
    Vs * I = I ^ 2 * R1 + I ^ 2 * R2 := by
  rw [h]; ring

/-! ## 8. Binary R-2R ladder -/

theorem r2r_three_bit (Vref b2 b1 b0 : ℝ) :
    Vref * (b2 / 2 + b1 / 4 + b0 / 8) = Vref * (4 * b2 + 2 * b1 + b0) / 8 := by
  ring

end EEE117
