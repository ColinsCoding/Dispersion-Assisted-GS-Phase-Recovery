"""Paul trap: the RF quadrupole that holds trapped-ion qubits, from SymPy to CUDA.

In plain words: you cannot hold a charged particle still with static voltages alone (Earnshaw), but
you can with a quadrupole whose voltage flips sign millions of times a second. The ion is pushed in
along x while pushed out along y, then the roles swap before it can escape; on average it feels a
gentle bowl. That bowl is how labs hold a single 40Ca+ or 171Yb+ ion as a qubit.

The pieces, each checkable:

  * derive_mathieu()      SymPy: the potential phi = (U - V cos Omega t)(x^2 - y^2)/(2 r0^2) solves
                          Laplace's equation, and Newton's law turns into the Mathieu equation
                          u'' + (a - 2q cos 2 xi) u = 0 with xi = Omega t / 2,
                          a = 4 e U / (m r0^2 Omega^2), q = 2 e V / (m r0^2 Omega^2) (y gets -a, -q).
  * floquet_beta(a, q)    exact stability: integrate one RF period, the monodromy matrix M has
                          trace 2 cos(pi beta). |trace| < 2 -> stable, and the secular (slow)
                          frequency is omega_sec = beta Omega / 2. For small q, beta ~ q / sqrt 2.
  * trap_parameters(...)  numbers for a real ion: q, secular frequency, ground-state size
                          x0 = sqrt(hbar / (2 m omega)), Lamb-Dicke parameter eta = k x0, the knob
                          that sets how well a laser can entangle the ion's spin with its motion.
  * electrode_voltage(t), ion_trajectory(...)   voltage versus time on the electrodes and the ion's
                          motion (slow secular swing plus fast micromotion at Omega), for the scope.
  * stability_map_gpu / stability_map_cpu       the (a, q) stability diagram from brute force: one
                          CUDA thread per (a, q) point integrates the Mathieu equation (RK4, float64)
                          and reports whether the ion stays bounded. Checked against Floquet.

Connects to: dgs.electrostatics_multipoles (the same x^2 - y^2 quadrupole field),
dgs.quadrupole_wavefunction (the l = 2 angular shape), dgs.scope_bench (the simulated oscilloscope
that displays these signals), dgs.qiskit_phase_flip (dephasing of the qubit this trap holds).
"""
from __future__ import annotations

import math

import numpy as np

E_CHARGE = 1.602176634e-19
AMU = 1.66053906660e-27
HBAR = 1.054571817e-34


# ---------------------------------------------------------------- SymPy derivation
def derive_mathieu():
    """Return dict with the SymPy potential, its Laplacian, and the Mathieu a, q for x and y."""
    import sympy as sp
    x, y, t, xi = sp.symbols("x y t xi", real=True)
    e, m, r0, U, V, Om = sp.symbols("e m r_0 U V Omega", positive=True)
    phi = (U - V * sp.cos(Om * t)) * (x ** 2 - y ** 2) / (2 * r0 ** 2)
    laplacian = sp.simplify(sp.diff(phi, x, 2) + sp.diff(phi, y, 2))
    out = {"phi": phi, "laplacian": laplacian}
    for coord, sign in ((x, 1), (y, -1)):
        accel = sp.simplify(-e / m * sp.diff(phi, coord) / coord)          # x'' = accel * x
        # t = 2 xi / Omega  ->  d^2/dt^2 = (Omega^2 / 4) d^2/dxi^2, so u'' = (4 / Omega^2) accel * u
        coeff = sp.expand(sp.simplify(-4 / Om ** 2 * accel.subs(t, 2 * xi / Om)))   # u'' + coeff * u = 0
        a = sp.simplify(coeff.subs(sp.cos(2 * xi), 0))
        q = sp.simplify(-(coeff - a).coeff(sp.cos(2 * xi)) / 2)
        out[str(coord)] = {"a": a, "q": q}
    return out


# ---------------------------------------------------------------- Floquet stability (exact)
def _mathieu_rhs(xi, s, a, q):
    return np.array([s[1], -(a - 2 * q * math.cos(2 * xi)) * s[0]])


def monodromy(a, q, steps=2000):
    """Monodromy matrix of u'' + (a - 2q cos 2xi) u = 0 over one period pi (RK4)."""
    h = math.pi / steps
    M = np.empty((2, 2))
    for col, s0 in enumerate(([1.0, 0.0], [0.0, 1.0])):
        s = np.array(s0)
        xi = 0.0
        for _ in range(steps):
            k1 = _mathieu_rhs(xi, s, a, q)
            k2 = _mathieu_rhs(xi + h / 2, s + h / 2 * k1, a, q)
            k3 = _mathieu_rhs(xi + h / 2, s + h / 2 * k2, a, q)
            k4 = _mathieu_rhs(xi + h, s + h * k3, a, q)
            s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            xi += h
        M[:, col] = s
    return M


def floquet_beta(a, q):
    """Characteristic exponent beta in [0, 1] (lowest zone), or nan if unstable (|trace| >= 2)."""
    tr = float(np.trace(monodromy(a, q)))
    if abs(tr) >= 2:
        return float("nan")
    return math.acos(tr / 2) / math.pi


def q_max(a=0.0, lo=0.1, hi=1.2, tol=1e-6):
    """Edge of the first stability region along fixed a (bisection on the Floquet trace)."""
    stable = lambda q: abs(np.trace(monodromy(a, q))) < 2
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if stable(mid) else (lo, mid)
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- a real trapped-ion qubit
def trap_parameters(mass_amu=40.0, r0=0.5e-3, rf_freq=20e6, V=300.0, U=0.0, laser_nm=729.0, angle_deg=45.0):
    """Numbers for a linear Paul trap holding one singly charged ion (defaults: 40Ca+, 729 nm qubit laser)."""
    m = mass_amu * AMU
    Om = 2 * math.pi * rf_freq
    q = 2 * E_CHARGE * V / (m * r0 ** 2 * Om ** 2)
    a = 4 * E_CHARGE * U / (m * r0 ** 2 * Om ** 2)
    beta = floquet_beta(a, q)
    w_sec = beta * Om / 2
    x0 = math.sqrt(HBAR / (2 * m * w_sec))
    k = 2 * math.pi / (laser_nm * 1e-9) * math.cos(math.radians(angle_deg))
    return dict(q=q, a=a, beta=beta, beta_lowest_order=math.sqrt(a + q ** 2 / 2),
                secular_MHz=w_sec / (2 * math.pi) / 1e6, rf_MHz=rf_freq / 1e6,
                pseudopotential_depth_eV=E_CHARGE * V ** 2 / (4 * m * r0 ** 2 * Om ** 2),
                ground_state_size_nm=x0 * 1e9, lamb_dicke_eta=k * x0)


def electrode_voltage(t, V=300.0, U=0.0, rf_freq=20e6):
    """Voltage on the RF electrode pair versus time: U - V cos(Omega t)."""
    return U - V * np.cos(2 * np.pi * rf_freq * np.asarray(t))


def ion_trajectory(a, q, n_periods=40, steps_per_period=200, u0=1.0, v0=0.0):
    """Ion position u(xi) for the Mathieu equation (RK4). Returns xi, u, du/dxi.
    xi = Omega t / 2, so one RF period is pi in xi."""
    h = math.pi / steps_per_period
    n = n_periods * steps_per_period
    xi = np.arange(n + 1) * h
    out = np.empty((n + 1, 2))
    s = np.array([u0, v0], float)
    out[0] = s
    for k in range(n):
        x_ = xi[k]
        k1 = _mathieu_rhs(x_, s, a, q)
        k2 = _mathieu_rhs(x_ + h / 2, s + h / 2 * k1, a, q)
        k3 = _mathieu_rhs(x_ + h / 2, s + h / 2 * k2, a, q)
        k4 = _mathieu_rhs(x_ + h, s + h * k3, a, q)
        s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[k + 1] = s
    return xi, out[:, 0], out[:, 1]


# ---------------------------------------------------------------- stability diagram: CPU and CUDA
def stability_map_cpu(a_vals, q_vals, n_periods=30, steps_per_period=64, blowup=1e3):
    """Brute-force stability: integrate every (a, q) at once (NumPy, vectorised RK4).
    Returns bool array [len(a_vals), len(q_vals)]: True = stayed bounded."""
    A, Q = np.meshgrid(np.asarray(a_vals, float), np.asarray(q_vals, float), indexing="ij")
    best = np.zeros(A.shape, bool)
    h = math.pi / steps_per_period
    for u0, v0 in ((1.0, 0.0), (0.0, 1.0)):          # both fundamental solutions must stay bounded
        u = np.full(A.shape, u0)
        v = np.full(A.shape, v0)
        f = lambda xi, u_: -(A - 2 * Q * np.cos(2 * xi)) * u_
        xi = 0.0
        for _ in range(n_periods * steps_per_period):
            k1u, k1v = v, f(xi, u)
            k2u, k2v = v + h / 2 * k1v, f(xi + h / 2, u + h / 2 * k1u)
            k3u, k3v = v + h / 2 * k2v, f(xi + h / 2, u + h / 2 * k2u)
            k4u, k4v = v + h * k3v, f(xi + h, u + h * k3u)
            u = u + h / 6 * (k1u + 2 * k2u + 2 * k3u + k4u)
            v = v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
            xi += h
            np.clip(u, -1e30, 1e30, out=u)
            np.clip(v, -1e30, 1e30, out=v)
        best |= ~(np.abs(u) < blowup)
    return ~best


_KERNEL = r"""
extern "C" __global__
void mathieu_rk4(const double* a, const double* q, unsigned char* stable, int n,
                 int n_steps, double h, double blowup) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double A = a[i], Q = q[i];
    unsigned char ok = 1;
    for (int run = 0; run < 2; ++run) {                 // both fundamental solutions
        double u = run == 0 ? 1.0 : 0.0, v = run == 0 ? 0.0 : 1.0, xi = 0.0;
        for (int k = 0; k < n_steps; ++k) {
            double c0 = cos(2.0 * xi), c1 = cos(2.0 * (xi + 0.5 * h)), c2 = cos(2.0 * (xi + h));
            double k1u = v,               k1v = -(A - 2.0 * Q * c0) * u;
            double k2u = v + 0.5*h*k1v,   k2v = -(A - 2.0 * Q * c1) * (u + 0.5*h*k1u);
            double k3u = v + 0.5*h*k2v,   k3v = -(A - 2.0 * Q * c1) * (u + 0.5*h*k2u);
            double k4u = v + h*k3v,       k4v = -(A - 2.0 * Q * c2) * (u + h*k3u);
            u += h / 6.0 * (k1u + 2.0*k2u + 2.0*k3u + k4u);
            v += h / 6.0 * (k1v + 2.0*k2v + 2.0*k3v + k4v);
            xi += h;
            if (fabs(u) > 1e30) break;
        }
        if (!(fabs(u) < blowup)) ok = 0;
    }
    stable[i] = ok;
}
"""


def gpu_available():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def stability_map_gpu(a_vals, q_vals, n_periods=30, steps_per_period=64, blowup=1e3):
    """Same as stability_map_cpu, one CUDA thread per (a, q) point (CuPy RawKernel, float64)."""
    import cupy as cp
    A, Q = np.meshgrid(np.asarray(a_vals, float), np.asarray(q_vals, float), indexing="ij")
    a_d, q_d = cp.asarray(A.ravel()), cp.asarray(Q.ravel())
    out = cp.zeros(a_d.size, dtype=cp.uint8)
    kern = cp.RawKernel(_KERNEL, "mathieu_rk4")
    threads = 256
    blocks = (a_d.size + threads - 1) // threads
    kern((blocks,), (threads,), (a_d, q_d, out, np.int32(a_d.size), np.int32(n_periods * steps_per_period),
                                 np.float64(math.pi / steps_per_period), np.float64(blowup)))
    cp.cuda.Stream.null.synchronize()
    return out.get().reshape(A.shape).astype(bool)


def stability_map_floquet(a_vals, q_vals):
    """Exact reference map from the monodromy trace (slow; use small grids)."""
    return np.array([[abs(np.trace(monodromy(a, q, steps=600))) < 2 for q in q_vals] for a in a_vals])
