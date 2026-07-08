"""The semi-empirical mass formula: a liquid drop that predicts nuclear binding.

Treat a nucleus as a charged liquid drop and its binding energy is the sum of five
competing terms (Bethe-Weizsacker):

    B(Z, A) =  a_V A                         volume  -- every nucleon binds to neighbors
             - a_S A^(2/3)                   surface -- the skin has fewer neighbors
             - a_C Z(Z-1) / A^(1/3)          Coulomb -- protons repel
             - a_sym (A - 2Z)^2 / A          SYMMETRY -- N != Z costs energy
             + delta(Z, A)                   pairing -- even-even is extra-bound

The SYMMETRY term is the one asked about: (A - 2Z) = N - Z, so the penalty grows with
the square of the neutron-proton imbalance and VANISHES at N = Z. It comes from the
Pauli principle -- protons and neutrons fill separate ladders of levels, and the
lowest total energy has them filled to the same height, i.e. N = Z. That is why light
stable nuclei sit on N = Z; only for heavy nuclei does the Coulomb term (which wants
FEWER protons) win enough to push the stable line to N > Z. The two terms in
competition give the VALLEY OF STABILITY, whose bottom is
        Z*(A) = A / (2 + (a_C / 2 a_sym) A^(2/3)).

Verified: the formula reproduces measured binding energies (iron-56, uranium-238) to
under ~1%, the binding-energy-per-nucleon curve peaks in the iron region, and Z*(A)
lands on the known stable isotopes. Sequel to dgs.proton_proton_chain, which measured
those binding energies; this predicts them. NumPy-free; py-3.13.
"""

import math

# coefficients in MeV (a standard Wapstra-like set)
A_VOLUME = 15.75
A_SURFACE = 17.8
A_COULOMB = 0.711
A_SYMMETRY = 23.7
A_PAIRING = 11.18


def volume_term(A):
    """+a_V A: bulk binding, proportional to the number of nucleons."""
    if A < 1:
        raise ValueError("A must be >= 1")
    return A_VOLUME * A


def surface_term(A):
    """-a_S A^(2/3): surface nucleons are under-bound (fewer neighbors)."""
    if A < 1:
        raise ValueError("A must be >= 1")
    return -A_SURFACE * A ** (2 / 3)


def coulomb_term(Z, A):
    """-a_C Z(Z-1)/A^(1/3): electrostatic repulsion of the Z protons."""
    if Z < 0 or A < 1:
        raise ValueError("need Z >= 0 and A >= 1")
    return -A_COULOMB * Z * (Z - 1) / A ** (1 / 3)


def symmetry_term(Z, A):
    """-a_sym (A - 2Z)^2 / A: the SYMMETRY energy. Zero at N = Z, and a growing
    penalty as the neutron-proton imbalance N - Z = A - 2Z grows."""
    if A < 1:
        raise ValueError("A must be >= 1")
    return -A_SYMMETRY * (A - 2 * Z) ** 2 / A


def pairing_term(Z, A):
    """delta: +a_P/sqrt(A) for even-even nuclei (extra bound), -a_P/sqrt(A) for
    odd-odd, 0 for odd A. Nucleons like to pair up."""
    if A < 1:
        raise ValueError("A must be >= 1")
    N = A - Z
    if A % 2 == 1:
        return 0.0
    return (A_PAIRING if (Z % 2 == 0 and N % 2 == 0) else -A_PAIRING) / math.sqrt(A)


def binding_energy(Z, A):
    """Total nuclear binding energy from the five liquid-drop terms, in MeV."""
    if not 0 <= Z <= A or A < 1:
        raise ValueError("need 1 <= A and 0 <= Z <= A")
    return (volume_term(A) + surface_term(A) + coulomb_term(Z, A)
            + symmetry_term(Z, A) + pairing_term(Z, A))


def binding_energy_per_nucleon(Z, A):
    """B/A -- the tightness of binding, whose curve peaks near iron."""
    return binding_energy(Z, A) / A


def most_stable_Z(A):
    """The proton number that MAXIMIZES binding for a given A -- the bottom of the
    valley of stability, Z* = A/(2 + (a_C/2 a_sym) A^(2/3)), rounded to an integer.
    The symmetry term pulls toward A/2 (N=Z); Coulomb pulls Z below it."""
    if A < 1:
        raise ValueError("A must be >= 1")
    z_star = A / (2 + (A_COULOMB / (2 * A_SYMMETRY)) * A ** (2 / 3))
    return int(round(z_star))


def peak_mass_number(a_range=range(20, 260)):
    """The mass number A where B/A is largest, walking the valley of stability --
    comes out in the iron-nickel region (~56-62), the most tightly bound nuclei."""
    best_A, best_ba = 0, -math.inf
    for A in a_range:
        ba = binding_energy_per_nucleon(most_stable_Z(A), A)
        if ba > best_ba:
            best_A, best_ba = A, ba
    return best_A


if __name__ == "__main__":
    print("semi-empirical binding energies vs measured:")
    known = {"40Ca": (20, 40, 342.05), "56Fe": (26, 56, 492.26),
             "120Sn": (50, 120, 1020.5), "238U": (92, 238, 1801.69)}
    for name, (Z, A, meas) in known.items():
        B = binding_energy(Z, A)
        print(f"  {name:6s}: SEMF {B:8.2f} MeV vs measured {meas:8.2f}  "
              f"({(B-meas)/meas*100:+.2f}%),  B/A = {B/A:.3f}")

    print("\nthe SYMMETRY term (why N = Z is favored):")
    for Z in (28, 26, 24):
        print(f"  56 nucleons, Z={Z} (N-Z={56-2*Z:+d}): "
              f"symmetry energy = {symmetry_term(Z, 56):7.2f} MeV")

    print("\nvalley of stability Z*(A) vs known stable isotopes:")
    for A, iso in [(16, "O-8"), (40, "Ca-20"), (56, "Fe-26"), (238, "U-92")]:
        print(f"  A={A:3d}: Z* = {most_stable_Z(A)}  ({iso})")

    print(f"\nB/A peaks at A = {peak_mass_number()} (the iron region -- most bound)")
