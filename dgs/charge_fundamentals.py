"""Griffiths' three opening facts about electric charge (Introduction to
Electrodynamics, Ch. 1 preamble) -- formalized and verified, not just quoted:

  1. Charge comes in two varieties (positive and negative); like charges
     repel, opposite charges attract.
  2. Charge is conserved: the total charge of an isolated system never
     changes -- the actual physical principle underneath every KCL node
     equation in dgs.spice.dc_nodal_analysis (a circuit node's "sum of
     currents in = sum of currents out" IS charge conservation applied to
     a point, dq/dt=0 at steady state).
  3. Charge is quantized: q = n*e for integer n -- no fraction of an
     electron's charge has ever been observed in isolation.

CHARGE'S TWO-VALUED NATURE, AS A TRUTH TABLE: Coulomb's law's sign behavior
(same-sign charges repel, opposite-sign charges attract) is exactly an XNOR
gate once charge sign is mapped to a boolean -- verified below by comparing
the two truth tables element-by-element, not asserted by analogy.

NumPy only. Education.
"""

import numpy as np

E_CHARGE = 1.602176634e-19  # C, the fundamental unit of quantized charge


def is_quantized(q, tol=1e-6):
    """Fact 2: is charge q an integer multiple of e (within tolerance tol,
    since q is a float)? Returns (bool, n) where n is the nearest integer
    multiple found."""
    n = q / E_CHARGE
    n_rounded = round(n)
    return abs(n - n_rounded) < tol, n_rounded


def total_charge(charges):
    """Fact 1 (conservation): the total charge of a system, simply summed --
    the quantity that Fact 2 (conservation) asserts stays constant over any
    isolated process, no matter how the individual charges rearrange."""
    return float(np.sum(charges))


def charge_conserved(charges_before, charges_after, tol=1e-6 * E_CHARGE):
    """Fact 2, made checkable: total charge before an interaction (particle
    decay, chemical reaction, circuit transient) must equal total charge
    after, even though individual charges may have been created, destroyed,
    or moved. This is EXACTLY the KCL equation dgs.spice.dc_nodal_analysis
    solves at every circuit node -- a node's charge cannot accumulate at DC,
    so current in must equal current out.

    Default tol is a millionth of an electron charge, NOT a generic 1e-12 --
    charge magnitudes are themselves ~1e-19 C, so a SI-scale absolute
    tolerance like 1e-12 would silently pass even a difference of a whole
    missing electron (1.6e-19 << 1e-12), never catching a real violation."""
    q_before = total_charge(charges_before)
    q_after = total_charge(charges_after)
    return abs(q_before - q_after) < tol


def coulomb_force_sign(q1_sign, q2_sign):
    """Fact 1's sign rule, computed directly from the actual product of
    signs (the physics): +1 = repulsive (like charges), -1 = attractive
    (opposite charges). q1_sign, q2_sign should be +1 or -1."""
    if q1_sign not in (1, -1) or q2_sign not in (1, -1):
        raise ValueError("q1_sign and q2_sign must each be +1 or -1")
    return q1_sign * q2_sign


def charge_truth_table():
    """Fact 1's two-valued nature IS a truth table: map charge sign to a
    boolean (True = positive, False = negative), and Coulomb's law's
    'like charges repel, opposite charges attract' rule becomes exactly
    XNOR(bool1, bool2) -- verified below by comparing the two truth tables
    row by row, not just asserted as an analogy.

    Returns a dict of {(bool1, bool2): (coulomb_says_repulsive, xnor_value)}
    for all four boolean input combinations."""
    table = {}
    for b1 in (True, False):
        for b2 in (True, False):
            sign1 = 1 if b1 else -1
            sign2 = 1 if b2 else -1
            coulomb_repulsive = coulomb_force_sign(sign1, sign2) > 0
            xnor_value = (b1 == b2)   # XNOR: true iff inputs match
            table[(b1, b2)] = (coulomb_repulsive, xnor_value)
    return table


if __name__ == "__main__":
    print("Fact 2 (quantization): is q=3e quantized?",
          is_quantized(3 * E_CHARGE))
    print("Fact 2 (quantization): is q=1.5e quantized (should be False)?",
          is_quantized(1.5 * E_CHARGE))

    print("\nFact 3 (conservation) at a circuit node: charges rearrange,")
    print("total stays fixed --")
    before = [2 * E_CHARGE, -1 * E_CHARGE, 3 * E_CHARGE]
    after = [1 * E_CHARGE, 0 * E_CHARGE, 3 * E_CHARGE]   # 1e moved from particle 1 to particle 2
    print(f"  before: {[round(q/E_CHARGE) for q in before]} e, total={total_charge(before)/E_CHARGE:.0f}e")
    print(f"  after:  {[round(q/E_CHARGE) for q in after]} e, total={total_charge(after)/E_CHARGE:.0f}e")
    print(f"  conserved? {charge_conserved(before, after)}")

    print("\nFact 1, as a truth table -- Coulomb's sign rule vs. logical XNOR:")
    table = charge_truth_table()
    print(f"  {'q1':>6} {'q2':>6} {'repulsive?':>12} {'XNOR(q1,q2)':>14} {'match?':>8}")
    all_match = True
    for (b1, b2), (repulsive, xnor_val) in table.items():
        match = (repulsive == xnor_val)
        all_match &= match
        print(f"  {b1!s:>6} {b2!s:>6} {repulsive!s:>12} {xnor_val!s:>14} {match!s:>8}")
    print(f"\nEVERY row matches: {all_match} -- Coulomb's law sign rule IS an XNOR gate,")
    print("not merely analogous to one.")
