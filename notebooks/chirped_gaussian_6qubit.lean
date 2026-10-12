-- six qubits: 2^6 basis states
theorem six_qubits : 2 ^ 6 = 64 := by decide
-- np.roll by 32 on a 64-site ring, applied twice, is the identity (rolling is undone by rolling back)
theorem roll_twice : ∀ j : Fin 64, ((j.val + 32) % 64 + 32) % 64 = j.val := by decide +kernel
-- the centered position x = j - 32 (as an integer) covers -32 .. 31 exactly once
-- the rolled index j holds position x = j for j < 32 and x = j - 64 otherwise: 6-bit two's complement
theorem twos_complement : ∀ j : Fin 64, (if j.val < 32 then (j.val : Int) else (j.val : Int) - 64) =
    ((j.val % 32 : Nat) : Int) - 32 * ((j.val / 32 : Nat) : Int) := by decide +kernel
def positions : List Int := (List.range 64).map (fun (j : Nat) => (j : Int) - 32)
theorem positions_cover : positions.length = 64 ∧ positions.head? = some (-32) ∧ positions.getLast? = some 31 ∧
    positions.Nodup := by decide +kernel
