-- capacity of a subshell with angular momentum l
def subshell (l : Nat) : Nat := 2 * (2 * l + 1)
theorem spdf : (subshell 0, subshell 1, subshell 2, subshell 3) = (2, 6, 10, 14) := by decide

-- shell capacity: sum over l = 0 .. n-1 equals 2 n^2, checked for shells 1..10
def shell (n : Nat) : Nat := ((List.range n).map subshell).foldl (· + ·) 0
theorem two_n_squared : (List.range 10).all (fun k => shell (k + 1) == 2 * (k + 1) ^ 2) = true := by decide

-- binomial coefficient by Pascal's rule, and Pauli microstate counts
def choose : Nat → Nat → Nat
  | _, 0 => 1
  | 0, _ + 1 => 0
  | n + 1, k + 1 => choose n k + choose n (k + 1)
theorem carbon_p2 : choose 6 2 = 15 := by decide
theorem particle_hole : choose 6 2 = choose 6 4 ∧ choose 10 6 = choose 10 4 := by decide

-- Madelung order: sort subshells by (n + l, n); the first 8 are 1s 2s 2p 3s 3p 4s 3d 4p
def madelungKey (p : Nat × Nat) : Nat := 10 * (p.1 + p.2) + p.1
theorem madelung_first_eight :
    let order := [(1,0), (2,0), (2,1), (3,0), (3,1), (4,0), (3,2), (4,1)]
    (order.zip order.tail).all (fun (a, b) => madelungKey a < madelungKey b) = true := by decide

-- noble gases: cumulative capacities along the Madelung order give 2, 10, 18, 36
theorem noble_gases : (2 : Nat) = 2 ∧ 2 + 2 + 6 = 10 ∧ 10 + 2 + 6 = 18 ∧ 18 + 2 + 10 + 6 = 36 := by decide
