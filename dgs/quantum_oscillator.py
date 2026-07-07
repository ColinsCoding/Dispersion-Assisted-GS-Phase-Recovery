"""omega, the quantum of energy: the harmonic oscillator, three ways.

Trap an atom in a potential well -- an optical dipole trap, an ion trap, a lattice
site -- and near the bottom the well is harmonic, V = (1/2) m omega^2 x^2. Quantum
mechanics then forbids a continuum of energies: the atom can only sit on a LADDER
of equally spaced rungs
        E_n = (n + 1/2) hbar*omega,   n = 0, 1, 2, ...
The spacing between rungs is one QUANTUM of energy, hbar*omega -- the same hbar*omega
that is a photon (a quantum of the light field), a phonon (a quantum of lattice
vibration), and the gap Planck needed to fix the blackbody spectrum. Even the lowest
rung is not zero: the ZERO-POINT energy (1/2) hbar*omega, a purely quantum fact.

This module reaches that ladder three independent ways, each a check on the others:

  1. NUMERICALLY -- "put the atom in the well": diagonalize the finite-difference
     Hamiltonian H = -(hbar^2/2m) d^2/dx^2 + (1/2) m omega^2 x^2 (reusing the same
     second-difference operator as dgs.vibration_modes) and watch the eigenvalues
     land on (n+1/2) hbar*omega, evenly spaced by hbar*omega.

  2. ALGEBRAICALLY -- the ladder operators a, a-dagger with [a, a-dagger] = 1 and
     H = hbar*omega (a-dagger a + 1/2). Here a-dagger literally ADDS one quantum:
     a-dagger|n> = sqrt(n+1)|n+1>. The number operator N = a-dagger a counts quanta.

  3. THERMALLY -- at temperature T the mean number of quanta is the Bose-Einstein
     <n> = 1/(exp(hbar*omega/kT) - 1), and the mean energy hbar*omega(<n>+1/2)
     crosses over to the classical kT (equipartition) when kT >> hbar*omega. The
     partition function Z = 1/(2 sinh(hbar*omega/2kT)) is checked against a sum
     computed in real C (gcc-guarded) -- "C programming for physics experiments."

Units default to hbar = m = k_B = 1 (set omega and T in those units). NumPy + stdlib.
"""

import os
import math
import subprocess

import numpy as np
from dgs.vibration_modes import second_difference_matrix

GCC_DEFAULT = r"C:\msys64\mingw64\bin\gcc.exe"


# ----------------------------------------------------------------------
# 1. The ladder E_n = (n + 1/2) hbar*omega
# ----------------------------------------------------------------------

def energy_levels(n_max, omega=1.0, hbar=1.0):
    """The energy ladder E_n = (n+1/2) hbar*omega for n = 0..n_max."""
    if n_max < 0 or omega <= 0 or hbar <= 0:
        raise ValueError("need n_max >= 0 and omega, hbar > 0")
    n = np.arange(n_max + 1)
    return (n + 0.5) * hbar * omega


def level_spacing(omega=1.0, hbar=1.0):
    """The quantum of energy: the constant gap hbar*omega between rungs."""
    if omega <= 0 or hbar <= 0:
        raise ValueError("omega, hbar must be positive")
    return hbar * omega


def zero_point_energy(omega=1.0, hbar=1.0):
    """The irreducible ground-state energy E_0 = (1/2) hbar*omega."""
    return 0.5 * hbar * omega


def numeric_energies(n_levels=6, n_grid=800, x_max=8.0,
                     omega=1.0, hbar=1.0, mass=1.0):
    """Put the atom in the well and solve: diagonalize the finite-difference
    Hamiltonian H = -(hbar^2/2m) d^2/dx^2 + (1/2) m omega^2 x^2 on [-x_max, x_max].
    Returns the lowest n_levels eigenvalues -- which must land on (n+1/2)hbar*omega."""
    if n_levels < 1 or n_grid < 20 or x_max <= 0:
        raise ValueError("need n_levels>=1, n_grid>=20, x_max>0")
    x = np.linspace(-x_max, x_max, n_grid)
    dx = x[1] - x[0]
    T = -(hbar ** 2) / (2 * mass) * second_difference_matrix(n_grid, dx)
    V = np.diag(0.5 * mass * omega ** 2 * x ** 2)
    return np.sort(np.linalg.eigvalsh(T + V))[:n_levels]


# ----------------------------------------------------------------------
# 2. Ladder operators: a-dagger adds one quantum
# ----------------------------------------------------------------------

def annihilation_operator(N):
    """The lowering operator a (truncated to N levels): a|n> = sqrt(n)|n-1>.
    Removes one quantum of energy."""
    if N < 2:
        raise ValueError("N must be >= 2")
    a = np.zeros((N, N))
    for n in range(1, N):
        a[n - 1, n] = math.sqrt(n)
    return a


def creation_operator(N):
    """The raising operator a-dagger: a-dagger|n> = sqrt(n+1)|n+1>. ADDS one
    quantum -- the operator form of 'omega is the quantum of energy'."""
    return annihilation_operator(N).T


def number_operator(N):
    """N = a-dagger a = diag(0, 1, 2, ...): counts how many quanta occupy the
    mode. Its eigenvalues are the rung indices n."""
    return creation_operator(N) @ annihilation_operator(N)


def hamiltonian_from_ladder(N, omega=1.0, hbar=1.0):
    """H = hbar*omega (a-dagger a + 1/2): the oscillator energy built purely
    from the algebra. Its diagonal is exactly the ladder (n+1/2)hbar*omega."""
    return hbar * omega * (number_operator(N) + 0.5 * np.eye(N))


def commutator_top_block(N):
    """[a, a-dagger] on the truncated space, minus its last row/column (the
    truncation artifact). Equals the identity -- the canonical [a,a-dagger]=1
    from which the whole ladder follows."""
    a = annihilation_operator(N)
    comm = a @ a.T - a.T @ a
    return comm[:N - 1, :N - 1]


# ----------------------------------------------------------------------
# 3. Thermal occupation: quanta at temperature T, and the classical limit
# ----------------------------------------------------------------------

def mean_occupation(omega, T, hbar=1.0, kB=1.0):
    """Bose-Einstein mean number of quanta <n> = 1/(exp(hbar*omega/kT) - 1).
    -> 0 as T -> 0 (frozen in the ground state), -> kT/hbar*omega when hot."""
    if omega <= 0 or T <= 0:
        raise ValueError("omega and T must be positive")
    return 1.0 / (math.exp(hbar * omega / (kB * T)) - 1.0)


def mean_energy(omega, T, hbar=1.0, kB=1.0):
    """<E> = hbar*omega(<n> + 1/2). At kT >> hbar*omega it approaches the
    classical kT (equipartition: 1/2 kT kinetic + 1/2 kT potential)."""
    return hbar * omega * (mean_occupation(omega, T, hbar, kB) + 0.5)


def partition_function(omega, T, hbar=1.0, kB=1.0):
    """Closed-form QHO partition function Z = 1/(2 sinh(hbar*omega/2kT)), the
    sum of exp(-E_n/kT) over the whole ladder done in closed form."""
    if omega <= 0 or T <= 0:
        raise ValueError("omega and T must be positive")
    return 1.0 / (2.0 * math.sinh(hbar * omega / (2 * kB * T)))


# ----------------------------------------------------------------------
# C programming for physics experiments: the partition sum in C
# ----------------------------------------------------------------------

C_SOURCE_QHO = r"""
#include <stdio.h>
#include <math.h>

/* Partition function and mean energy of a quantum harmonic oscillator by
   SUMMING over the energy ladder E_n = (n + 1/2) hbar*omega, in units where
   x = hbar*omega / (kB T). Cross-checks the closed forms Z = 1/(2 sinh(x/2))
   and <E>/hbar*omega = (1/2) coth(x/2). */
int main(void) {
    double x = 1.0;              /* hbar*omega / kT */
    double Z = 0.0, E = 0.0;
    for (int n = 0; n < 200; n++) {
        double En = (n + 0.5);          /* in units of hbar*omega */
        double w = exp(-En * x);        /* Boltzmann weight */
        Z += w;
        E += En * w;
    }
    printf("%.12e %.12e\n", Z, E / Z);  /* Z, <E>/hbar*omega */
    return 0;
}
"""


def gcc_available(gcc_path=GCC_DEFAULT):
    """Whether the C toolchain is present for the compiled partition-sum check."""
    return os.path.exists(gcc_path)


def compile_and_run_c(out_dir, gcc_path=GCC_DEFAULT):
    """Compile and run C_SOURCE_QHO; return the C-summed (Z, mean_energy_over_hw)
    for x = hbar*omega/kT = 1, to compare against the Python closed forms."""
    src = os.path.join(out_dir, "qho.c")
    exe = os.path.join(out_dir, "qho.exe")
    with open(src, "w") as f:
        f.write(C_SOURCE_QHO)
    r = subprocess.run([gcc_path, "-O2", "-o", exe, src, "-lm"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"gcc failed: {r.stderr}")
    out = subprocess.run([exe], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"C program failed: {out.stderr}")
    Z, E_over_hw = map(float, out.stdout.split())
    return {"Z": Z, "mean_energy_over_hw": E_over_hw}


if __name__ == "__main__":
    print("ladder E_n = (n+1/2) hbar*omega (omega=1):", np.round(energy_levels(5), 2))
    print(f"quantum of energy hbar*omega = {level_spacing():.1f}; "
          f"zero-point = {zero_point_energy():.1f}")

    E = numeric_energies(6)
    print("numeric (put atom in the well):", np.round(E, 4),
          " spacing ~", round(float(np.mean(np.diff(E))), 4))

    N = 12
    print("[a, a-dagger] top block == I ?",
          np.allclose(commutator_top_block(N), np.eye(N - 1)))
    adag = creation_operator(N)
    e2 = np.zeros(N); e2[2] = 1
    print("a-dagger|2> = sqrt(3)|3> ?", np.allclose(adag @ e2, math.sqrt(3) * np.eye(N)[3]))
    print("ladder-Hamiltonian diagonal:", np.round(np.diag(hamiltonian_from_ladder(6)), 2))

    print("\nthermal: <E> crosses over from hbar*omega/2 to kT")
    for T in (0.1, 1.0, 10.0, 100.0):
        print(f"  T={T:6.1f}: <n>={mean_occupation(1.0, T):8.3f}  "
              f"<E>={mean_energy(1.0, T):8.3f}  (classical kT={T})")

    print("\nC partition sum (x = hbar*omega/kT = 1):")
    if gcc_available():
        c = compile_and_run_c(os.environ.get("TEMP", "."))
        Z_exact = partition_function(1.0, 1.0)
        E_exact = 0.5 / math.tanh(0.5)            # (1/2) coth(1/2)
        print(f"  C: Z={c['Z']:.6f} (closed {Z_exact:.6f}); "
              f"<E>/hw={c['mean_energy_over_hw']:.6f} (closed {E_exact:.6f})")
    else:
        print("  (gcc not found -- skipping)")
