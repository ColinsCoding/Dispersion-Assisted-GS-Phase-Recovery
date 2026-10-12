-- Shor code facts, checked by the Lean kernel (core Lean 4, no Mathlib)
structure Pauli where
  x : Nat
  z : Nat

def popc (n : Nat) : Nat := (List.range 9).foldl (fun acc i => acc + (n >>> i) % 2) 0

-- symplectic product: Paulis commute iff it is even
def commutes (a b : Pauli) : Bool := popc ((a.x &&& b.z) ^^^ (a.z &&& b.x)) % 2 == 0

def zz (i j : Nat) : Pauli := ⟨0, (1 <<< i) ||| (1 <<< j)⟩
def stabilizers : List Pauli :=
  [zz 0 1, zz 1 2, zz 3 4, zz 4 5, zz 6 7, zz 7 8, ⟨0b000111111, 0⟩, ⟨0b111111000, 0⟩]
def Xbar : Pauli := ⟨0, 0b111111111⟩     -- Z on all nine qubits
def Zbar : Pauli := ⟨0b111111111, 0⟩     -- X on all nine qubits

theorem stabilizers_commute :
    stabilizers.all (fun a => stabilizers.all (fun b => commutes a b)) = true := by decide +kernel
theorem logicals_commute_with_stabilizers :
    stabilizers.all (fun s => commutes s Xbar && commutes s Zbar) = true := by decide +kernel
theorem logicals_anticommute : commutes Xbar Zbar = false := by decide +kernel

def singleErrors : List Pauli :=
  (List.range 9).flatMap (fun j => [⟨1 <<< j, 0⟩, ⟨1 <<< j, 1 <<< j⟩, ⟨0, 1 <<< j⟩])   -- X, Y, Z on qubit j
theorem every_single_error_detected :
    singleErrors.all (fun e => stabilizers.any (fun s => !commutes s e)) = true := by decide +kernel

-- the Pauli-frame decoder model from Section 5
def grp (n b : Nat) : Nat := popc ((n >>> (3 * b)) &&& 7)
def fails (e : Pauli) : Bool :=
  let zpar := (List.range 3).map (fun b => grp e.z b % 2)
  let resid := (List.range 3).map (fun b => if grp e.x b ≥ 2 then 1 else 0)
  (zpar.foldl (· + ·) 0 ≥ 2) || (resid.foldl (· + ·) 0 % 2 == 1)

def combine (a b : Pauli) : Pauli := ⟨a.x ^^^ b.x, a.z ^^^ b.z⟩
def onQubit (j : Nat) : List Pauli := [⟨1 <<< j, 0⟩, ⟨1 <<< j, 1 <<< j⟩, ⟨0, 1 <<< j⟩]
def doubleErrors : List Pauli :=
  (List.range 9).flatMap (fun i => (List.range 9).flatMap (fun j =>
    if i < j then (onQubit i).flatMap (fun a => (onQubit j).map (fun b => combine a b)) else []))

theorem no_single_error_fails : (singleErrors.filter fails).length = 0 := by decide +kernel
theorem double_error_count : doubleErrors.length = 324 := by decide +kernel
theorem failing_double_errors : (doubleErrors.filter fails).length = 144 := by decide +kernel
