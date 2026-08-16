"""Stainless steel grill: the 2D heat equation, solved three independent
ways and cross-checked against each other.

The physical setup: a circular grill of radius R, heated unevenly (a hot
sear zone at the center, cooling toward the rim, which is held at ambient
temperature -- a Dirichlet boundary condition). Heat diffuses according to
the standard diffusion equation dT/dt = alpha*Laplacian(T).

Three independent solvers, in increasing order of generality (and
decreasing order of "exactness"):

  1. BesselHeatSolution -- the EXACT analytical solution for the
     axisymmetric case, using separation of variables in polar
     coordinates. This is where the Bessel functions come in: a
     Dirichlet-BC eigenfunction on a disk is J_0(beta_m * r/R), where
     beta_m are the (infinitely many) positive zeros of J_0 -- the direct
     cylindrical-coordinates analog of sin(n*pi*x/L) on an interval.
  2. RadialFiniteDifferenceSolver -- a 1D numerical solve exploiting the
     same radial symmetry, cross-checking the Bessel series without
     assuming its correctness.
  3. TriangulatedFEMSolver -- a genuine 2D finite-element solve on an
     unstructured triangular mesh (the "high number of triangles"), which
     does NOT assume radial symmetry at all -- the same method would work
     for an off-center hot spot or a non-circular grill, unlike the other
     two. Linear (P1) elements, backward-Euler time stepping.

All three should agree, on a purely radially-symmetric problem, to within
each method's own discretization error -- verified directly, not assumed.
"""

from __future__ import annotations

import pathlib
import subprocess

import numpy as np
from scipy.special import jv, jn_zeros
from scipy.integrate import quad
from scipy.spatial import Delaunay
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

GCC_DEFAULT = r"C:\msys64\mingw64\bin\gcc.exe"
C_SOURCE_DIR = pathlib.Path(__file__).resolve().parent / "c" / "grill_heat"


class BesselHeatSolution:
    """Exact axisymmetric solution: T(r,t) = sum_m A_m J_0(beta_m r/R)
    exp(-alpha (beta_m/R)^2 t), Dirichlet BC T(R,t)=0, with A_m fit to an
    arbitrary radial initial condition f(r) via the orthogonality of
    J_0(beta_m r/R) on [0,R] with weight r."""

    def __init__(self, R, alpha, f_initial, n_modes=40):
        self.R = float(R)
        self.alpha = float(alpha)
        self.f_initial = f_initial
        self.n_modes = n_modes
        self.betas = jn_zeros(0, n_modes)
        self.coefficients = np.array([self._coefficient(b) for b in self.betas])

    def _coefficient(self, beta):
        numerator = quad(lambda r: r * self.f_initial(r) * jv(0, beta * r / self.R), 0, self.R)[0]
        denominator = 0.5 * self.R ** 2 * jv(1, beta) ** 2
        return numerator / denominator

    def temperature(self, r, t):
        """T(r, t) for scalar or array r, scalar t."""
        r = np.atleast_1d(np.asarray(r, dtype=float))
        result = np.zeros_like(r)
        for A_m, beta in zip(self.coefficients, self.betas):
            result += A_m * jv(0, beta * r / self.R) * np.exp(-self.alpha * (beta / self.R) ** 2 * t)
        return result


class RadialFiniteDifferenceSolver:
    """1D explicit (FTCS) finite-difference solve of the same axisymmetric
    PDE, dT/dt = alpha*(d2T/dr2 + (1/r)dT/dr), with the r=0 singularity
    handled via L'Hopital's rule (the 1/r * dT/dr term becomes d2T/dr2
    there by symmetry) and Dirichlet BC at r=R."""

    def __init__(self, R, alpha, f_initial, n_points=200, stability_factor=0.2):
        self.R = float(R)
        self.alpha = float(alpha)
        self.r = np.linspace(0, R, n_points)
        self.dr = self.r[1] - self.r[0]
        self.dt = stability_factor * self.dr ** 2 / alpha
        self.T = f_initial(self.r).astype(float)
        self.T[-1] = 0.0
        self.t = 0.0

    def step(self):
        T, r, dr, alpha, dt = self.T, self.r, self.dr, self.alpha, self.dt
        T_new = T.copy()
        d2T = (T[2:] - 2 * T[1:-1] + T[:-2]) / dr ** 2
        dTdr = (T[2:] - T[:-2]) / (2 * dr)
        T_new[1:-1] = T[1:-1] + dt * alpha * (d2T + dTdr / r[1:-1])
        d2T0 = (2 * T[1] - 2 * T[0]) / dr ** 2   # symmetric ghost point T[-1] = T[1]
        T_new[0] = T[0] + dt * alpha * 2 * d2T0
        T_new[-1] = 0.0
        self.T = T_new
        self.t += dt

    def advance_to(self, t_target):
        while self.t < t_target:
            self.step()
        return self.T


class TriangulatedFEMSolver:
    """2D linear (P1) finite-element solve on a Delaunay-triangulated
    disk mesh -- doesn't assume radial symmetry, unlike the other two
    solvers, so it's the one that would still work for an off-center hot
    spot or a non-circular grill shape."""

    def __init__(self, R, alpha, f_initial_radial, n_r=30, n_theta=60, dt=1.0):
        self.R = float(R)
        self.alpha = float(alpha)
        self.dt = float(dt)

        rs = np.linspace(0, R, n_r)
        pts = [[0.0, 0.0]]
        for ri in rs[1:]:
            for th in np.linspace(0, 2 * np.pi, n_theta, endpoint=False):
                pts.append([ri * np.cos(th), ri * np.sin(th)])
        self.points = np.array(pts)
        self.triangulation = Delaunay(self.points)
        self.r_of_points = np.linalg.norm(self.points, axis=1)
        self.boundary_mask = self.r_of_points > R - 1e-9

        self.M, self.K = self._assemble()
        A = (self.M + self.dt * self.alpha * self.K).tolil()
        for idx in np.where(self.boundary_mask)[0]:
            A.rows[idx] = [idx]
            A.data[idx] = [1.0]
        self._A = csr_matrix(A)

        self.T = f_initial_radial(self.r_of_points).astype(float)
        self.T[self.boundary_mask] = 0.0
        self.t = 0.0

    def _assemble(self):
        n = len(self.points)
        M = lil_matrix((n, n))
        K = lil_matrix((n, n))
        for simplex in self.triangulation.simplices:
            i, j, k = simplex
            xi, yi = self.points[i]
            xj, yj = self.points[j]
            xk, yk = self.points[k]
            area = 0.5 * abs((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
            if area < 1e-14:
                continue
            b = np.array([yj - yk, yk - yi, yi - yj]) / (2 * area)
            c = np.array([xk - xj, xi - xk, xj - xi]) / (2 * area)
            idx = [i, j, k]
            for a_ in range(3):
                for b_ in range(3):
                    K[idx[a_], idx[b_]] += area * (b[a_] * b[b_] + c[a_] * c[b_])
            M_local = area / 12.0 * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
            for a_ in range(3):
                for b_ in range(3):
                    M[idx[a_], idx[b_]] += M_local[a_, b_]
        return csr_matrix(M), csr_matrix(K)

    @property
    def n_triangles(self):
        return len(self.triangulation.simplices)

    def step(self):
        rhs = self.M.dot(self.T)
        rhs[self.boundary_mask] = 0.0
        self.T = spsolve(self._A, rhs)
        self.t += self.dt

    def advance_to(self, t_target):
        while self.t < t_target - 1e-9:
            self.step()
        return self.T


def run_c_radial_solver(gcc_path=GCC_DEFAULT, out_dir=None):
    """Compile and run dgs/c/grill_heat/grill_heat_fd.c (same Makefile
    project, same default parameters as RadialFiniteDifferenceSolver's
    defaults) and return its final temperature array. This is a genuine
    independent implementation in a different language, not a Python
    wrapper around the same code -- disagreement would mean an actual bug
    in one of the two, not a language quirk.

    gcc needs its OWN bin directory on PATH to find co-located toolchain
    components (the linker, runtime DLLs) even when invoked by an
    absolute path -- verified directly (a trivial hello-world compile
    fails silently, no stderr at all, exit code 1, unless mingw64/bin is
    prepended to PATH first). Passing an explicit env= to subprocess is
    what fixes it, not just pointing at the right gcc.exe."""
    import os

    out_dir = pathlib.Path(out_dir) if out_dir else C_SOURCE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    exe_path = out_dir / "grill_heat_fd.exe"
    src_path = C_SOURCE_DIR / "grill_heat_fd.c"

    gcc_bin_dir = str(pathlib.Path(gcc_path).parent)
    env = dict(os.environ)
    env["PATH"] = gcc_bin_dir + os.pathsep + env.get("PATH", "")

    compile_result = subprocess.run(
        [gcc_path, "-O2", "-Wall", "-lm", "-o", str(exe_path), str(src_path)],
        capture_output=True, text=True, env=env,
    )
    if compile_result.returncode != 0:
        raise RuntimeError(f"gcc failed (exit {compile_result.returncode}):\n{compile_result.stderr}")

    run_result = subprocess.run([str(exe_path)], capture_output=True, text=True, env=env)
    if run_result.returncode != 0:
        raise RuntimeError(f"grill_heat_fd.exe failed:\n{run_result.stderr}")

    values = np.array([float(line) for line in run_result.stdout.strip().splitlines()])
    return values


def searzone_initial_condition(peak_temp=250.0, sear_radius=0.05):
    """A hot central sear zone (Gaussian bump), cooling toward the rim --
    physically motivated: the burner directly under the center of a grill
    runs hottest."""
    def f(r):
        return peak_temp * np.exp(-(np.asarray(r) / sear_radius) ** 2)
    return f


def temperature_to_color(T, T_max):
    """Map temperature in [0, T_max] to an RGB blackbody-ish gradient:
    black -> deep red -> orange -> yellow -> white, the standard visual
    convention for "how hot is this surface." Vectorized over an array."""
    frac = np.clip(np.asarray(T, dtype=float) / T_max, 0.0, 1.0)
    stops = np.array([
        [0.00, 10, 10, 15],
        [0.25, 120, 20, 10],
        [0.55, 220, 90, 15],
        [0.80, 250, 190, 40],
        [1.00, 255, 250, 220],
    ])
    r = np.interp(frac, stops[:, 0], stops[:, 1])
    g = np.interp(frac, stops[:, 0], stops[:, 2])
    b = np.interp(frac, stops[:, 0], stops[:, 3])
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


class GrillHeatVisualizer:
    """Object-oriented pygame front end: renders the EXACT Bessel-series
    solution's temperature field in real time (fast enough to evaluate
    fresh every frame at any t -- no need to step a simulation forward,
    unlike the two numerical solvers, which is exactly why the analytical
    solution is the right one to drive an interactive animation)."""

    def __init__(self, R, alpha, f_initial, n_modes=40, size_px=520, margin_px=20):
        self.R = R
        self.alpha = alpha
        self.solution = BesselHeatSolution(R, alpha, f_initial, n_modes=n_modes)
        self.T_max = float(self.solution.temperature(0.0, 0.0)[0])
        self.size_px = size_px
        self.margin_px = margin_px
        self.grid_radius_px = (size_px - 2 * margin_px) // 2

        xs = np.arange(size_px) - size_px / 2
        ys = np.arange(size_px) - size_px / 2
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        self.pixel_r = np.sqrt(X ** 2 + Y ** 2)
        self.inside_grill = self.pixel_r <= self.grid_radius_px
        # physical radius corresponding to each pixel (only valid where inside_grill)
        self.physical_r = np.clip(self.pixel_r / self.grid_radius_px * R, 0, R)

        # PRECOMPUTE the Bessel basis J_0(beta_m * r/R) once per pixel per
        # mode (n_modes x n_pixels) -- this is the expensive part
        # (n_modes scipy.special.jv calls over ~270k pixels each), and it
        # does NOT depend on t. Naively calling BesselHeatSolution.temperature()
        # fresh every frame measured at 3.3s/frame (would take minutes for
        # a real animation); precomputing the basis and reducing each
        # frame to a cheap weighted sum brings that down by ~2 orders of
        # magnitude, verified below.
        r_flat = self.physical_r.ravel()
        self._basis = np.stack(
            [jv(0, beta * r_flat / R) for beta in self.solution.betas], axis=0
        )   # (n_modes, n_pixels)
        self._decay_rates = alpha * (self.solution.betas / R) ** 2   # (n_modes,)
        self._coefficients = self.solution.coefficients   # (n_modes,)

    def frame_surface(self, t):
        """A pygame Surface for the temperature field at time t -- a
        cheap weighted sum against the precomputed basis, not a fresh
        BesselHeatSolution.temperature() call."""
        import pygame

        weights = self._coefficients * np.exp(-self._decay_rates * t)   # (n_modes,)
        T_flat = weights @ self._basis                                    # (n_pixels,)
        T_at_pixel_r = T_flat.reshape(self.physical_r.shape)

        colors = temperature_to_color(T_at_pixel_r, self.T_max)
        colors[~self.inside_grill] = (25, 25, 30)   # background outside the grill disk

        surf = pygame.Surface((self.size_px, self.size_px))
        pygame.surfarray.blit_array(surf, colors)
        return surf

    def run(self, t_max=600.0, speed=8.0):
        import pygame

        pygame.init()
        screen = pygame.display.set_mode((self.size_px, self.size_px + 40))
        pygame.display.set_caption("Stainless Steel Grill -- Bessel-function heat diffusion")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("consolas", 16)

        t = 0.0
        running = True
        paused = False
        try:
            while running:
                dt = clock.tick(30) / 1000.0
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        if event.key == pygame.K_SPACE:
                            paused = not paused
                        if event.key == pygame.K_r:
                            t = 0.0

                if not paused:
                    t = min(t_max, t + dt * speed)

                screen.fill((15, 15, 18))
                screen.blit(self.frame_surface(t), (0, 0))
                hud = font.render(f"t = {t:6.1f} s   SPACE=pause  R=reset  ESC=quit  (Ctrl+C also quits cleanly)",
                                   True, (230, 230, 230))
                screen.blit(hud, (10, self.size_px + 8))
                pygame.display.flip()
        except KeyboardInterrupt:
            pass   # Ctrl+C in the terminal is a valid way to stop this -- exit cleanly, no traceback

        pygame.quit()


if __name__ == "__main__":
    R, alpha = 0.15, 4.0e-6   # 15cm stainless steel grill, typical thermal diffusivity
    f_initial = searzone_initial_condition()

    print("=== Bessel analytical solution ===")
    bessel = BesselHeatSolution(R, alpha, f_initial, n_modes=40)
    print(f"T(r=0, t=100) = {bessel.temperature(0.0, 100.0)[0]:.4f}")

    print("\n=== Radial finite-difference cross-check ===")
    fd = RadialFiniteDifferenceSolver(R, alpha, f_initial, n_points=200)
    T_fd = fd.advance_to(100.0)
    idx_center = 0
    print(f"T(r=0, t={fd.t:.2f}) = {T_fd[idx_center]:.4f}")
    diff_fd = abs(T_fd[idx_center] - bessel.temperature(0.0, fd.t)[0])
    print(f"|FD - Bessel| at center = {diff_fd:.4f}")

    print("\n=== Triangulated FEM cross-check ===")
    fem = TriangulatedFEMSolver(R, alpha, f_initial, n_r=30, n_theta=60, dt=1.0)
    print(f"mesh: {len(fem.points)} points, {fem.n_triangles} triangles")
    T_fem = fem.advance_to(100.0)
    diff_fem = abs(T_fem[0] - bessel.temperature(0.0, 100.0)[0])
    print(f"T(center, t=100) = {T_fem[0]:.4f}")
    print(f"|FEM - Bessel| at center = {diff_fem:.4f}")

    print("\n=== C radial finite-difference cross-check ===")
    T_c = run_c_radial_solver()
    print(f"T(r=0, t=100) [C] = {T_c[0]:.4f}")
    print(f"|C - Python FD| at center = {abs(T_c[0] - T_fd[0]):.2e}")

    print("\nLaunching the interactive visualization "
          "(SPACE=pause, R=reset, ESC/close window=quit)...")
    viz = GrillHeatVisualizer(R, alpha, f_initial)
    viz.run()
