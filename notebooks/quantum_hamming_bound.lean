def hammingOK (n : Nat) : Bool := 2 * (1 + 3 * n) ≤ 2 ^ n
theorem no_code_below_five : [1, 2, 3, 4].all (fun n => !hammingOK n) = true := by decide
theorem bound_from_five : (List.range 8).all (fun i => hammingOK (i + 5)) = true := by decide
theorem five_qubit_code_is_perfect : 2 * (1 + 3 * 5) = 2 ^ 5 := by decide
theorem shor_code_fits : 2 * (1 + 3 * 9) ≤ 2 ^ 9 := by decide
theorem classical_hamming_7_4_perfect : 1 + 7 = 2 ^ 3 := by decide
