"""Static membrane physics -- Jalali lab prereq: PDEs, linear algebra, numerics.

THE PHYSICS:
  A membrane stretched over a frame obeys Laplace's equation (static case):
    nabla^2 u = 0   (no forcing, steady state)
    d^2u/dx^2 + d^2u/dy^2 = 0

  With forcing (drum hit, applied pressure):
    nabla^2 u = -f(x,y) / T   (Poisson equation, T=tension)

  Dynamic (wave) case (Griffiths Ch 9 in 2D):
    d^2u/dt^2 = c^2 * nabla^2 u

JALALI LAB CONNECTION:
  Jalali group uses membrane-analog models for:
  - Serial time-encoded amplified microscopy (STEAM): dispersive temporal lens
  - Optical analogies of quantum membranes (photonic potential wells)
  - Neural membrane potential: Hodgkin-Huxley ODE (see cellular_biophysics.py)
  - Rogue waves in fibers: membrane-like instability (NLSE, see dgs/nlse.py)

MATH PREREQS THIS FILE EXERCISES (application checklist):

  [RANK / LINEAR ALGEBRA]
    Finite difference matrix A is rank N^2 - boundary (singular if not pinned).
    np.linalg.matrix_rank(A) confirms it. Solving Au=f -> linalg.solve.
    COVERED: dgs/hilbert_space.py, dgs/bessel_linalg.py, dgs/statistics.py (PCA/SVD)

  [VECTOR CALCULUS]
    nabla^2 u = div(grad u) -- Laplacian is div of gradient.
    Discretized: (u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j})/h^2
    COVERED: griffiths/vectors.py, dgs/line_integrals.py

  [DIFFERENTIAL EQUATIONS]
    Laplace -> elliptic PDE (boundary value problem, not initial value).
    Solved by: FD matrix inversion, separation of variables, Green's function.
    COVERED: dgs/pde_separation.py, dgs/schrodinger.py

  [LOGARITHMIC DIFFERENTIATION]
    Used to find: d/dx[u^n] = n * u^{n-1} * u'
    Membrane energy: E = T/2 * integral |nabla u|^2 dA
    Minimize E -> Euler-Lagrange -> Laplace equation.
    d/dA[integral f dA] = 0  solved via log-derivative of functional.
    COVERED: dgs/spacetime.py (3 quotient rule forms + log-diff)

  NUMERICAL STABILITY:
    Condition number cond(A) = sigma_max / sigma_min (ratio of largest to smallest singular value).
    For the Laplace matrix: cond ~ O(1/h^2) -> small grid spacing = ill-conditioned.
    Fix: use sparse direct solver or multigrid; here we use numpy (dense, educational).

Run: py -3.13 -c "from dgs.static_membrane import demo; demo()"
"""
import numpy as np


# ── Finite-difference Laplace solver ────────────────────────────────────────

def build_laplace_matrix(N):
    """Build the (N^2) x (N^2) finite-difference matrix for nabla^2 on a grid.

    Interior points of an (N+2)x(N+2) grid (0 and N+1 are boundary).
    5-point stencil: A*u = f  where u is the N^2 interior unknowns.

    A is symmetric, negative semi-definite, diagonally dominant.
    Eigenvalues are all negative -> stable discretization.
    """
    if N < 2:
        raise ValueError("N must be >= 2")
    size = N * N
    A = np.zeros((size, size))
    def idx(i, j):
        return i * N + j
    for i in range(N):
        for j in range(N):
            k = idx(i, j)
            A[k, k] = -4.0
            if i > 0:     A[k, idx(i-1, j)] = 1.0
            if i < N-1:   A[k, idx(i+1, j)] = 1.0
            if j > 0:     A[k, idx(i, j-1)] = 1.0
            if j < N-1:   A[k, idx(i, j+1)] = 1.0
    return A


def solve_static_membrane(N=20, boundary="zero", forcing=None):
    """Solve nabla^2 u = f(x,y) on [0,1]^2 with boundary u=0 (Dirichlet).

    N: interior grid points per dimension (N^2 unknowns total).
    forcing: None -> Laplace (zero RHS), or array (N,N) for Poisson.

    Returns: u (N,N) interior solution, x (N,), y (N,), condition_number.

    NUMERICAL STABILITY:
      Small N (5-10): cond~100-1000 (well conditioned, numpy is fine)
      Large N (100+): cond~10000+ (use scipy.sparse for production)
      Here N<=50 is safe with numpy.linalg.solve.
    """
    if N > 50:
        raise ValueError("N>50: use sparse solver (scipy) for production; keep N<=50 here")

    h = 1.0 / (N + 1)
    x = np.linspace(h, 1-h, N)
    y = np.linspace(h, 1-h, N)
    X, Y = np.meshgrid(x, y)

    A = build_laplace_matrix(N) / h**2

    # default forcing: point load at center (like a fingertip on a drum)
    if forcing is None:
        f = np.zeros((N, N))
        f[N//2, N//2] = -1.0 / h**2   # concentrated load
    else:
        f = np.asarray(forcing, float)
        if f.shape != (N, N):
            raise ValueError(f"forcing must be shape ({N},{N})")

    f_flat = f.flatten()
    u_flat = np.linalg.solve(A, f_flat)
    u = u_flat.reshape(N, N)

    # condition number -- key for numerical stability report
    sv = np.linalg.svd(A, compute_uv=False)
    cond = sv[0] / sv[-1]

    return {"u": u, "x": x, "y": y, "X": X, "Y": Y, "h": h, "N": N,
            "cond": cond, "A_shape": A.shape}


def membrane_modes(N=10):
    """Analytical eigenmodes of a rectangular membrane (separation of variables).

    u_{mn}(x,y) = sin(m*pi*x) * sin(n*pi*y)
    omega_{mn} = pi * c * sqrt(m^2 + n^2)

    These are the EXACT solutions. The FD solver approximates them.
    The ratio FD_eigenvalue / exact eigenvalue -> 1 as h -> 0.

    LINEAR ALGEBRA: eigenmodes = eigenvectors of the Laplace operator.
    Same math as PCA eigenvectors, Schrodinger eigenstates, normal modes.
    """
    h = 1.0 / (N + 1)
    x = np.linspace(h, 1-h, N)
    y = np.linspace(h, 1-h, N)
    X, Y = np.meshgrid(x, y)
    modes = {}
    for m in range(1, 4):
        for n in range(1, 4):
            u_mn = np.sin(m * np.pi * X) * np.sin(n * np.pi * Y)
            omega_mn = np.pi * np.sqrt(m**2 + n**2)   # c=1
            modes[(m, n)] = {"u": u_mn, "omega": omega_mn,
                             "degenerate_with": [
                                 (p, q) for p in range(1,4) for q in range(1,4)
                                 if p != m or q != n
                                 if abs(p**2 + q**2 - m**2 - n**2) < 0.01
                             ]}
    return modes


def membrane_energy(u, h):
    """Membrane elastic energy E = T/2 * integral |nabla u|^2 dA.

    Discretized: E ~ h^2/2 * sum_ij [(u_{i+1,j}-u_{i-1,j})^2/(2h)^2 + ...]
    At minimum energy: Euler-Lagrange -> nabla^2 u = 0 (Laplace equation).

    LOGARITHMIC DIFFERENTIATION connection:
      Functional derivative dE/du = 0
      -> -T * nabla^2 u = 0
      Same as log-diff: d/depsilon[E[u + epsilon*v]]|0 = 0
    """
    du_dx = np.gradient(u, h, axis=1)
    du_dy = np.gradient(u, h, axis=0)
    E = 0.5 * h**2 * np.sum(du_dx**2 + du_dy**2)
    return float(E)


def condition_number_vs_N():
    """Show how cond(A) scales with N -- core numerical stability result.

    Theory: cond(Laplace FD) ~ 4/h^2 * pi^-2 ~ O(N^2)
    This is WHY large meshes need iterative/multigrid solvers.
    """
    results = []
    for N in [5, 10, 15, 20, 30]:
        A = build_laplace_matrix(N)
        sv = np.linalg.svd(A, compute_uv=False)
        cond = sv[0] / sv[-1]
        rank = np.linalg.matrix_rank(A)
        results.append({"N": N, "unknowns": N*N,
                        "cond": round(cond, 1),
                        "rank": rank,
                        "full_rank": rank == N*N})
    return results


# ── Math prereq readiness report ────────────────────────────────────────────

MATH_READINESS = {
    "rank_linear_algebra": {
        "status": "COVERED",
        "where": "dgs/bessel_linalg.py, dgs/hilbert_space.py, dgs/statistics.py (SVD/PCA)",
        "jalali_use": "SVD for STEAM image reconstruction; matrix rank for GS constraint counting",
        "gap": "None -- SVD, eigenvectors, null space all present",
    },
    "vector_calculus": {
        "status": "COVERED",
        "where": "griffiths/vectors.py, dgs/line_integrals.py, dgs/classical_ed.py",
        "jalali_use": "nabla^2 u (Laplacian) for wave/membrane; Poynting S = E x B for optical intensity",
        "gap": "None -- gradient, curl, div, Stokes theorem all present",
    },
    "differential_equations": {
        "status": "COVERED",
        "where": "dgs/pde_separation.py, dgs/schrodinger.py, dgs/nlse.py, dgs/cellular_biophysics.py",
        "jalali_use": "NLSE for fiber; Schrodinger for photonic potential; HH for neural membrane",
        "gap": "None -- ODEs, PDEs, separation of variables, finite difference all present",
    },
    "logarithmic_differentiation": {
        "status": "COVERED",
        "where": "dgs/spacetime.py (3 quotient rule forms, log-diff form verified SymPy)",
        "jalali_use": "d/dx[ln|u|] = u'/u appears in gain-guiding laser equations and NLSE loss term",
        "gap": "None",
    },
    "fourier_analysis": {
        "status": "COVERED",
        "where": "dgs/fourier_tools.py, dgs/fourier_series.py, dgs/causality.py (FFT Hilbert)",
        "jalali_use": "DFT = core of GS algorithm; H(f)=exp(i*pi*D*f^2) is the dispersion operator",
        "gap": "None -- DFT, FFT, convolution theorem, windowing all present",
    },
    "numerical_methods": {
        "status": "COVERED",
        "where": "dgs/numerical_methods.py, dgs/static_membrane.py (FD Laplace)",
        "jalali_use": "Finite difference for beam propagation; condition number for solver stability",
        "gap": "No sparse solver (scipy.sparse) -- add if N>50 mesh needed",
    },
    "probability_statistics": {
        "status": "COVERED",
        "where": "dgs/statistics.py, dgs/hypothesis.py, dgs/bayes_inference.py",
        "jalali_use": "Noise models for photon counting; hypothesis test for phase convergence",
        "gap": "None",
    },
    "physics_programs_count": {
        "status": f"156 modules in dgs/",
        "categories": "optics/photonics/QM/EM/circuits/biophysics/ML/signal processing",
        "flagship": "phase_retrieval.ipynb (129 cells) + ml_course_on_receiver.ipynb",
    },
}


def readiness_report():
    print("=" * 65)
    print("  MATH READINESS for Jalali Lab Application")
    print("=" * 65)
    gaps = []
    for topic, r in MATH_READINESS.items():
        if topic == "physics_programs_count":
            print(f"\n  PROGRAMS: {r['status']}")
            print(f"    Flagship: {r['flagship']}")
            continue
        status = r["status"]
        flag = "[OK]" if status == "COVERED" else "[!!]"
        print(f"\n  {flag} {topic.upper().replace('_',' ')}")
        print(f"       where:  {r['where']}")
        print(f"       Jalali: {r['jalali_use']}")
        if r["gap"] != "None":
            gaps.append(f"{topic}: {r['gap']}")
            print(f"       GAP:    {r['gap']}")

    print("\n" + "=" * 65)
    if not gaps:
        print("  VERDICT: All prereqs covered. You have what you need to apply.")
        print("  Strongest assets for the email:")
        print("    - GS phase retrieval (Project 5, notebooks/phase_retrieval.ipynb)")
        print("    - FNO + NLSE (dgs/gs_fno.py + dgs/nlse.py)")
        print("    - 156 physics modules across EM/QM/biophysics/photonics")
    else:
        print("  GAPS:", ", ".join(gaps))
    return gaps


# ── Demo ──────────────────────────────────────────────────────────────────────

def demo():
    print("=" * 65)
    print("  dgs/static_membrane.py  --  demo")
    print("=" * 65)

    print("\n--- Laplace FD solver: N=10, point load at center ---")
    r = solve_static_membrane(N=10)
    u = r["u"]
    print(f"  Grid: {r['N']}x{r['N']} = {r['N']**2} unknowns")
    print(f"  Condition number: {r['cond']:.1f}  (ratio largest/smallest singular value)")
    print(f"  u_max = {u.max():.4f}  u_min = {u.min():.4f}  (at center: {u[5,5]:.4f})")
    E = membrane_energy(u, r["h"])
    print(f"  Elastic energy E = T/2*|nabla u|^2 dA = {E:.6f}")

    print("\n--- Condition number vs grid size (numerical stability) ---")
    print("  N    unknowns   cond(A)     rank   full_rank")
    for row in condition_number_vs_N():
        print(f"  {row['N']:3d}   {row['unknowns']:6d}    {row['cond']:10.1f}   "
              f"{row['rank']:5d}   {row['full_rank']}")
    print("  cond ~ O(N^2) -> 4x bigger grid = 16x harder to solve")
    print("  Fix for N>50: scipy.sparse.linalg.spsolve or multigrid")

    print("\n--- Membrane eigenmodes (separation of variables) ---")
    modes = membrane_modes(N=20)
    print("  (m,n)   omega_mn      degenerate_with")
    for (m,n), v in sorted(modes.items()):
        deg = v["degenerate_with"]
        print(f"  ({m},{n})     {v['omega']:.4f}    {deg if deg else 'none'}")
    print("  Degenerate modes: same frequency, different spatial pattern")
    print("  Same math as QM degenerate eigenstates (Griffiths Ch 6)")

    print()
    readiness_report()


if __name__ == "__main__":
    demo()
