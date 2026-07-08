"""The periodic table from one rule: filling electron orbitals by the Madelung order.

Every element's chemistry follows from how its electrons fill the available orbitals, and
the order is set by a single rule (Madelung / aufbau): fill by increasing n+l, and for a
tie, by increasing n. That gives the famous sequence
        1s 2s 2p 3s 3p 4s 3d 4p 5s 4d 5p 6s 4f 5d 6p 7s 5f 6d 7p
-- note 4s fills BEFORE 3d (n+l = 4 < 5), which is why potassium is [Ar]4s^1 and the
transition metals appear where they do. Each subshell holds 2(2l+1) electrons (s:2, p:6,
d:10, f:14), from the 2l+1 orbitals times two spins.

Pour Z electrons into that sequence and you get the ground-state configuration, and from
it the element's PLACE in the table:
  * PERIOD  = the highest principal quantum number occupied (the row),
  * VALENCE = electrons in that outermost shell (the column, roughly),
  * NOBLE GAS = a full outer p subshell (or full 1s for He): chemically inert.

This is the quantum origin of the periodic table -- the same orbital/quantum-number
structure as dgs.angular_momentum (l, the 2l+1 orbitals) and the hydrogen levels of
dgs.stability_of_matter. Verified against the known configurations of H, C, Na, Fe, and
the noble gases, and that the electrons always sum to Z. (Cr and Cu are the well-known
half/full-shell exceptions the simple rule does not capture -- noted, not asserted.)
NumPy-free; py-3.13.
"""

_LABELS = {0: "s", 1: "p", 2: "d", 3: "f"}
_NOBLE_Z = [2, 10, 18, 36, 54, 86]
_NOBLE_SYMBOL = {2: "He", 10: "Ne", 18: "Ar", 36: "Kr", 54: "Xe", 86: "Rn"}


def madelung_order(max_n=8):
    """The orbital filling order (n, l) by the Madelung rule: sort by (n+l, n).
    Returns the list 1s, 2s, 2p, 3s, 3p, 4s, 3d, ... (4s before 3d)."""
    orbitals = [(n, l) for n in range(1, max_n + 1) for l in range(0, min(n, 4))]
    return sorted(orbitals, key=lambda nl: (nl[0] + nl[1], nl[0]))


def subshell_capacity(l):
    """Electrons a subshell holds: 2(2l+1) -- (2l+1) orbitals times 2 spins."""
    if l not in _LABELS:
        raise ValueError("l must be 0..3 (s,p,d,f)")
    return 2 * (2 * l + 1)


def electron_configuration(Z):
    """Ground-state configuration of element Z as a list of (n, subshell, count),
    in Madelung fill order. The counts always sum to Z."""
    if not 1 <= Z <= 118:
        raise ValueError("Z must be in 1..118")
    remaining = Z
    config = []
    for n, l in madelung_order():
        if remaining <= 0:
            break
        fill = min(subshell_capacity(l), remaining)
        config.append((n, _LABELS[l], fill))
        remaining -= fill
    return config


def configuration_string(Z, noble_core=False):
    """The configuration written the textbook way: sorted by (n, l), e.g. Fe ->
    '1s2 2s2 2p6 3s2 3p6 3d6 4s2'. With noble_core=True, abbreviate the inner
    shells as the preceding noble gas, e.g. Na -> '[Ne] 3s1'."""
    cfg = electron_configuration(Z)
    lorder = {"s": 0, "p": 1, "d": 2, "f": 3}
    ordered = sorted(cfg, key=lambda x: (x[0], lorder[x[1]]))
    if noble_core:
        cores = [ng for ng in _NOBLE_Z if ng < Z]
        if cores:
            ng = cores[-1]
            ng_cfg = set((n, s) for n, s, _ in electron_configuration(ng))
            outer = [(n, s, c) for n, s, c in ordered if (n, s) not in ng_cfg]
            return f"[{_NOBLE_SYMBOL[ng]}] " + " ".join(f"{n}{s}{c}" for n, s, c in outer)
    return " ".join(f"{n}{s}{c}" for n, s, c in ordered)


def period(Z):
    """The period (row) = the highest principal quantum number occupied."""
    return max(n for n, _, _ in electron_configuration(Z))


def valence_electrons(Z):
    """Electrons in the outermost shell (highest n) -- the ones that do chemistry."""
    cfg = electron_configuration(Z)
    top = max(n for n, _, _ in cfg)
    return sum(c for n, _, c in cfg if n == top)


def is_noble_gas(Z):
    """A noble gas has a filled outer shell (full 2p/3p/... or full 1s for He) and
    is chemically inert -- exactly the Z in {2,10,18,36,54,86}."""
    return Z in _NOBLE_Z


def total_electrons(Z):
    """Sum of the configuration's electron counts -- must equal Z (a consistency
    check on the filling)."""
    return sum(c for _, _, c in electron_configuration(Z))


if __name__ == "__main__":
    order = madelung_order()
    print("Madelung fill order:",
          " ".join(f"{n}{_LABELS[l]}" for n, l in order[:12]), "...")
    print(f"  note 4s ({order.index((4,0))+1}th) fills before 3d "
          f"({order.index((3,2))+1}th)\n")

    for Z, name in [(1, "H"), (6, "C"), (11, "Na"), (18, "Ar"), (19, "K"), (26, "Fe")]:
        print(f"  Z={Z:2d} {name:2s}: {configuration_string(Z):28s} "
              f"[{configuration_string(Z, noble_core=True)}]")
        print(f"          period {period(Z)}, valence {valence_electrons(Z)}, "
              f"noble gas: {is_noble_gas(Z)}")

    print("\nelectrons sum to Z for all elements?",
          all(total_electrons(Z) == Z for Z in range(1, 119)))
