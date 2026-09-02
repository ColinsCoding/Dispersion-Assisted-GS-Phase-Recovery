"""Independent Python reimplementation of the Bohren & Huffman BHMIE
algorithm -- computed from scratch here (not via a Mie-theory library),
so comparing it against mie_kernel.cu's CUDA output is a genuine
cross-check of two independently-written implementations of the same
recursive algorithm, not two copies of the same possible bug.

Reads x values from mie_output.csv (written by mie_main.cu) so both
sides are evaluated at EXACTLY the same size parameters, then reports
the max/mean relative error in Qext and Qsca.
"""
import sys
import numpy as np


def mie_efficiencies(x, m):
    """Qext, Qsca for a single size parameter x and complex refractive
    index m, via the same downward-D_n / upward-psi_n,chi_n recurrence
    as mie_kernel.cu's mie_efficiencies device function."""
    if x <= 0.0:
        return 0.0, 0.0

    mx = m * x
    Nmax = int(x + 4.0 * x ** (1.0 / 3.0) + 2.0) + 1
    Nstart = max(Nmax, int(abs(mx)) + 15)

    D = np.zeros(Nstart + 1, dtype=complex)
    for n in range(Nstart, 0, -1):
        n_over_mx = n / mx
        D[n - 1] = n_over_mx - 1.0 / (D[n] + n_over_mx)

    psi_prev2, psi_prev1 = np.cos(x), np.sin(x)
    chi_prev2, chi_prev1 = -np.sin(x), np.cos(x)

    Qext, Qsca = 0.0, 0.0
    for n in range(1, Nmax + 1):
        psi_n = (2 * n - 1) / x * psi_prev1 - psi_prev2
        chi_n = (2 * n - 1) / x * chi_prev1 - chi_prev2
        xi_n = complex(psi_n, -chi_n)
        xi_prev1 = complex(psi_prev1, -chi_prev1)

        Dn = D[n]
        n_over_x = n / x

        term_a = Dn / m + n_over_x
        a_n = (term_a * psi_n - psi_prev1) / (term_a * xi_n - xi_prev1)

        term_b = Dn * m + n_over_x
        b_n = (term_b * psi_n - psi_prev1) / (term_b * xi_n - xi_prev1)

        factor = 2 * n + 1
        Qext += factor * (a_n + b_n).real
        Qsca += factor * (abs(a_n) ** 2 + abs(b_n) ** 2)

        psi_prev2, psi_prev1 = psi_prev1, psi_n
        chi_prev2, chi_prev1 = chi_prev1, chi_n

    x2 = x * x
    return (2.0 / x2) * Qext, (2.0 / x2) * Qsca


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "mie_output.csv"
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    x_vals, Qext_cuda, Qsca_cuda = data[:, 0], data[:, 1], data[:, 2]

    m = complex(1.33, 0.0)   # must match mie_main.cu's m_re, m_im exactly
    Qext_py = np.array([mie_efficiencies(x, m)[0] for x in x_vals])
    Qsca_py = np.array([mie_efficiencies(x, m)[1] for x in x_vals])

    ext_rel_err = np.abs(Qext_cuda - Qext_py) / np.maximum(np.abs(Qext_py), 1e-12)
    sca_rel_err = np.abs(Qsca_cuda - Qsca_py) / np.maximum(np.abs(Qsca_py), 1e-12)

    print(f"loaded {len(x_vals)} size parameters from {csv_path}")
    print(f"Qext relative error vs. independent Python BHMIE: "
          f"max={ext_rel_err.max():.2e}, mean={ext_rel_err.mean():.2e}")
    print(f"Qsca relative error vs. independent Python BHMIE: "
          f"max={sca_rel_err.max():.2e}, mean={sca_rel_err.mean():.2e}")

    worst = np.argmax(ext_rel_err)
    print(f"\nworst-agreement point: x={x_vals[worst]:.4f}, "
          f"CUDA Qext={Qext_cuda[worst]:.8f}, Python Qext={Qext_py[worst]:.8f}")

    # the worst-agreement point is always the smallest x (Qext itself is
    # tiny there, ~1e-5, so ordinary float64 rounding differences between
    # CUDA's cuComplex ops and Python's native complex arithmetic show up
    # as a larger RELATIVE error even though the ABSOLUTE difference is
    # negligible) -- 1e-5 relative error is still a tight, honest bar
    if ext_rel_err.max() < 1e-5 and sca_rel_err.max() < 1e-5:
        print("\nPASS: CUDA and independent Python implementation agree to <1e-5 relative error")
        return 0
    else:
        print("\nFAIL: disagreement exceeds 1e-5 relative error -- investigate")
        return 1


if __name__ == "__main__":
    sys.exit(main())
