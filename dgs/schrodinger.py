"""
1-D Schrödinger equation solver via finite-difference Hamiltonian.

Builds H = T + V on a grid, diagonalises with np.linalg.eigh (numpy only,
no scipy).  Parameter sweeps loop over well depth or width and collect
bound-state energies + THz transition frequencies.

Physical units: eV for energy, nm for length, electron mass throughout.
"""
import numpy as np
import matplotlib.pyplot as plt

# ── constants ──────────────────────────────────────────────────────────────
_HBAR_SI  = 1.054571817e-34   # ℏ in J·s
M_E_KG    = 9.1093837015e-31  # electron mass kg
EV_TO_J   = 1.602176634e-19   # J per eV
NM_TO_M   = 1e-9              # m per nm

# ℏ²/2m in eV·nm²: compute in SI (J·m²) then divide by eV and nm²
# = 6.105e-39 J·m²  →  / 1.602e-19 J/eV / 1e-18 m²/nm²  ≈ 0.03811 eV·nm²
HBAR2_OVER_2M = (_HBAR_SI**2 / (2 * M_E_KG)) / EV_TO_J / NM_TO_M**2


# ── core solver ───────────────────────────────────────────────────────────
def solve(V_func, x_nm, n_states=4):
    """
    Solve H ψ = E ψ on grid x_nm (nm) with potential V_func(x) → eV.

    Returns
    -------
    energies : (n_states,) array, eV
    psi      : (n_states, N) array, normalised wavefunctions
    """
    N  = len(x_nm)
    dx = x_nm[1] - x_nm[0]
    V  = V_func(x_nm)

    # kinetic energy: -ℏ²/2m · d²/dx²  (3-point finite difference)
    diag  =  2 * HBAR2_OVER_2M / dx**2 + V
    off   = -    HBAR2_OVER_2M / dx**2 * np.ones(N - 1)
    H     = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)

    vals, vecs = np.linalg.eigh(H)

    # keep only bound states (E < max-wall height) up to n_states
    V_wall = max(V[0], V[-1])
    bound  = vals < V_wall
    vals   = vals[bound][:n_states]
    vecs   = vecs[:, bound][:, :n_states]

    # normalise
    psi = (vecs / np.sqrt(dx)).T      # shape (n_states, N)
    return vals, psi


# ── standard potentials ───────────────────────────────────────────────────
def finite_square_well(x_nm, L_nm, V0_eV):
    """Finite square well centred at 0, width L, depth V0."""
    V = np.where(np.abs(x_nm) <= L_nm / 2, 0.0, V0_eV)
    return V


def harmonic(x_nm, omega_eV_per_nm2):
    """½ m ω² x²  expressed as  ω_eV_per_nm2 · x²  (eV)."""
    return omega_eV_per_nm2 * x_nm**2


# ── parameter sweeps ──────────────────────────────────────────────────────
def sweep_depth(L_nm=5.0, V0_range=(0.05, 2.0), n_V=40,
                n_states=4, x_span=30.0, N_grid=800):
    """
    Loop over well depth V0.  Returns dict with sweep arrays.

    THz transition: ΔE (eV) → f (THz) via f = ΔE / h.
    """
    V0_vals = np.linspace(*V0_range, n_V)
    x       = np.linspace(-x_span / 2, x_span / 2, N_grid)
    h_eV_THz = 4.135667696e-3          # h in eV·THz⁻¹ (= eV/THz)

    all_E   = []
    all_THz = []

    for V0 in V0_vals:
        E, _ = solve(lambda x, V0=V0: finite_square_well(x, L_nm, V0),
                     x, n_states)
        all_E.append(E)
        # 0→1 transition
        dE = E[1] - E[0] if len(E) > 1 else np.nan
        all_THz.append(dE / h_eV_THz)

    return dict(param=V0_vals, param_label="Well depth V₀ (eV)",
                energies=all_E, thz_01=np.array(all_THz),
                L_nm=L_nm)


def sweep_width(V0_eV=1.0, L_range=(1.0, 20.0), n_L=40,
                n_states=4, x_span=40.0, N_grid=800):
    """Loop over well width L."""
    L_vals  = np.linspace(*L_range, n_L)
    h_eV_THz = 4.135667696e-3

    all_E   = []
    all_THz = []

    for L in L_vals:
        x = np.linspace(-max(L * 2, x_span / 2),
                         max(L * 2, x_span / 2), N_grid)
        E, _ = solve(lambda x, L=L: finite_square_well(x, L, V0_eV),
                     x, n_states)
        all_E.append(E)
        dE = E[1] - E[0] if len(E) > 1 else np.nan
        all_THz.append(dE / h_eV_THz)

    return dict(param=L_vals, param_label="Well width L (nm)",
                energies=all_E, thz_01=np.array(all_THz),
                V0_eV=V0_eV)


# ── plotting ──────────────────────────────────────────────────────────────
def plot_sweep(result, title="Parameter sweep"):
    p       = result["param"]
    energies = result["energies"]
    thz     = result["thz_01"]

    # pad energy lists to matrix (NaN for missing states)
    max_s = max(len(e) for e in energies)
    E_mat = np.full((len(p), max_s), np.nan)
    for i, e in enumerate(energies):
        E_mat[i, :len(e)] = e

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, max_s))
    for s in range(max_s):
        ax1.plot(p, E_mat[:, s], color=colors[s], label=f"n={s}")
    ax1.set_ylabel("Energy (eV)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    ax2.plot(p, thz, color="steelblue")
    ax2.set_ylabel("0→1 transition (THz)")
    ax2.set_xlabel(result["param_label"])
    ax2.grid(True, alpha=0.3)
    ax2.axhspan(0.3, 10, alpha=0.12, color="orange", label="THz window (0.3–10 THz)")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    return fig


def plot_wavefunctions(L_nm=5.0, V0_eV=1.0, n_states=4,
                       x_span=20.0, N_grid=600):
    """Plot potential + first n_states wavefunctions."""
    x   = np.linspace(-x_span / 2, x_span / 2, N_grid)
    V   = finite_square_well(x, L_nm, V0_eV)
    E, psi = solve(lambda x: finite_square_well(x, L_nm, V0_eV), x, n_states)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, V, "k", lw=2, label="V(x)")
    scale = 0.3 * V0_eV
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(E)))
    for i, (Ei, psi_i) in enumerate(zip(E, psi)):
        ax.axhline(Ei, color=colors[i], lw=0.8, ls="--", alpha=0.6)
        ax.plot(x, psi_i * scale + Ei, color=colors[i],
                label=f"n={i}  E={Ei:.3f} eV")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("Energy (eV) / ψ (offset)")
    ax.set_title(f"Finite square well  L={L_nm} nm  V₀={V0_eV} eV")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
