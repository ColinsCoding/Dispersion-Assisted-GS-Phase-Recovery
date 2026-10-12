-- 2 Γ₂ = Γ₁ + 2 Γ_φ with non-negative rates (natural numbers in some unit) implies 2 Γ₂ ≥ Γ₁, i.e. T₂ ≤ 2 T₁
theorem t2_le_two_t1 (g1 gphi g2 : Nat) (h : 2 * g2 = g1 + 2 * gphi) : g1 ≤ 2 * g2 := by omega
-- equality T₂ = 2 T₁ happens exactly when there is no pure dephasing
theorem t2_eq_two_t1_iff (g1 gphi g2 : Nat) (h : 2 * g2 = g1 + 2 * gphi) : (2 * g2 = g1 ↔ gphi = 0) := by omega
-- example from section 5: T₁ = 40, T_φ = 60 -> rates in units of 1/120: Γ₁ = 3, Γ_φ = 2, so 2 Γ₂ = 3 + 4 = 7
theorem example_rates : 2 * 7 = 2 * 3 + 2 * 4 ∧ 3 ≤ 7 := by decide
