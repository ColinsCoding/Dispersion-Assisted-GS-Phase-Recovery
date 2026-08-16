"""Lightweight regression checks for the Python/SymPy equivalents used in
notebooks/synthetic_cell_imaging_pipeline.ipynb -- runnable without Jupyter, for a fast
CI-style sanity check of the same numeric claims the notebook verifies inline. Plain
asserts, run as a script (this project's tests, like the notebook, are self-contained and
do not depend on the parent repo's dgs/griffiths packages)."""
import pathlib
import numpy as np
import sympy as sp

HERE = pathlib.Path(__file__).resolve().parent
MATLAB_DIR = HERE.parent / "matlab"

# ---- Part 1: field theory (point charge) ----
x, y, z = sp.symbols("x y z", real=True)
coords = (x, y, z)
r = sp.sqrt(x**2 + y**2 + z**2)


def grad(f):
    return sp.Matrix([sp.diff(f, c) for c in coords])


def div(F):
    return sum(sp.diff(F[i], coords[i]) for i in range(3))


def curl(F):
    return sp.Matrix([
        sp.diff(F[2], y) - sp.diff(F[1], z),
        sp.diff(F[0], z) - sp.diff(F[2], x),
        sp.diff(F[1], x) - sp.diff(F[0], y),
    ])


V_point = 1 / r
E_point = -grad(V_point)
assert sp.simplify(div(E_point)) == 0
assert sp.simplify(curl(E_point)) == sp.zeros(3, 1)
assert sp.simplify(sum(sp.diff(V_point, c, 2) for c in coords)) == 0

# ---- Part 2: Bessel's equation from cylindrical symmetry ----
rho_s, k_s = sp.symbols("rho_s k_s", positive=True)
J0_expr = sp.besselj(0, k_s * rho_s)
residual = sp.simplify(rho_s**2 * sp.diff(J0_expr, rho_s, 2) + rho_s * sp.diff(J0_expr, rho_s) + k_s**2 * rho_s**2 * J0_expr)
assert residual == 0

from scipy.special import jv
rho_num = np.linspace(0, 20, 4000)
J0_num = jv(0, rho_num)
known_J0_zeros = np.array([2.4048, 5.5201, 8.6537, 11.7915])
sign_changes = np.where(np.diff(np.sign(J0_num)))[0][:4]
found_zeros = np.array([np.interp(0, [J0_num[i], J0_num[i + 1]], [rho_num[i], rho_num[i + 1]]) for i in sign_changes])
assert np.max(np.abs(found_zeros - known_J0_zeros)) < 1e-2

# ---- Part 3: even/odd -> real/imaginary FFT correspondence ----
N = 513
dx = 0.02
half = (N - 1) // 2
xs = (np.arange(N) - half) * dx
g_even = np.exp(-xs**2 / 4)
g_odd = xs * np.exp(-xs**2 / 4)
G_even = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(g_even)))
G_odd = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(g_odd)))
assert np.max(np.abs(G_even.imag)) / np.max(np.abs(G_even.real)) < 1e-8
assert np.max(np.abs(G_odd.real)) / np.max(np.abs(G_odd.imag)) < 1e-8

# ---- Part 4/5: forward model + Tikhonov beats naive pinv ----
def build_cell_object(n):
    lin = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(lin, lin)
    Rr = np.sqrt(X**2 + Y**2)
    membrane = ((Rr > 0.75) & (Rr < 0.9)).astype(float)
    nucleus = 0.9 * np.exp(-((X - 0.15) ** 2 + (Y + 0.1) ** 2) / (2 * 0.18**2))
    return (membrane + nucleus) / (membrane + nucleus).max()


def build_blur_matrix(n, sigma):
    kk = np.arange(n) - n // 2
    h1 = np.exp(-(kk**2) / (2 * sigma**2))
    h1 = h1 / h1.sum()
    H1 = np.array([np.roll(h1, row - n // 2) for row in range(n)])
    return np.kron(H1, H1)


N_img = 16
x_true = build_cell_object(N_img)
H = build_blur_matrix(N_img, 1.5)
rng = np.random.default_rng(0)
y = (H @ x_true.flatten()).reshape(N_img, N_img) + 0.02 * rng.standard_normal((N_img, N_img))

x_hat_naive = np.linalg.pinv(H) @ y.flatten()
naive_error = np.linalg.norm(x_hat_naive - x_true.flatten())

HtH = H.T @ H
Hty = H.T @ y.flatten()
I = np.eye(H.shape[1])
lam = 3e-3
x_hat_reg = np.linalg.solve(HtH + lam * I, Hty)
reg_error = np.linalg.norm(x_hat_reg - x_true.flatten())

assert np.linalg.matrix_rank(H) == H.shape[0]
assert abs(np.linalg.cond(H) - np.linalg.svd(H, compute_uv=False)[0] / np.linalg.svd(H, compute_uv=False)[-1]) < 1e-3
assert reg_error < naive_error, "Tikhonov regularization should beat the naive pinv reconstruction"

# ---- Part 6: kinetics fit on the MATLAB-exported CSV (cross-language check) ----
csv_path = MATLAB_DIR / "part6_kinetics_data.csv"
if csv_path.exists():
    data = np.loadtxt(csv_path, delimiter=",")
    t_data, I_data = data[:, 0], data[:, 1]
    slope, intercept = np.polyfit(t_data, np.log(I_data), 1)
    k_analytic = -slope
    k_true = 0.35
    assert abs(k_analytic - k_true) / k_true < 0.15, \
        f"analytic fit on MATLAB-exported data should recover k close to {k_true}, got {k_analytic}"
else:
    print(f"NOTE: {csv_path} not found -- run matlab/part6_kinetics_fit.m first to exercise this check")

print("all projects/synthetic_cell_imaging_pipeline Python-equivalent checks passed")
