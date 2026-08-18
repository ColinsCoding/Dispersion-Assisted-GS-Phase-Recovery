"""geometry.py -- 2D refractive-index maps for the photonic mode-geometry
project. Each function returns n[x,y] (a real-valued refractive-index
grid) given physical dimensions in micrometers and a grid spacing --
COMPUTATIONAL GEOMETRY (boolean masks on a grid), not CAD.

Coordinate system: origin at the domain center, x increasing "right",
y increasing "up" (matplotlib imshow convention handled at plot time,
not baked into the grid itself).
"""
import numpy as np


def make_grid(nx: int, ny: int, dx: float, dy: float):
    """Physical coordinate arrays for an nx*ny grid with spacing dx,dy (um),
    centered at the origin. Returns (x, y, X, Y) -- 1D axes and their 2D
    meshgrid (indexing='ij', so X[i,j],Y[i,j] matches array index [i,j])."""
    if nx < 2 or ny < 2:
        raise ValueError(f"nx={nx}, ny={ny}: both must be >= 2")
    if dx <= 0 or dy <= 0:
        raise ValueError("dx and dy must be positive")
    x = (np.arange(nx) - nx / 2) * dx
    y = (np.arange(ny) - ny / 2) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")
    return x, y, X, Y


def _validate_indices(n_core: float, n_clad: float):
    if n_core <= 0 or n_clad <= 0:
        raise ValueError("n_core and n_clad must be positive")


def make_rectangle(nx: int, ny: int, dx: float, dy: float, width: float, height: float,
                    n_core: float = 3.4, n_clad: float = 1.44, center=(0.0, 0.0)):
    """A single rectangular high-index core in a uniform cladding.
    Returns (n[nx,ny], (x,y))."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    _validate_indices(n_core, n_clad)
    x, y, X, Y = make_grid(nx, ny, dx, dy)
    cx, cy = center
    mask = (np.abs(X - cx) <= width / 2) & (np.abs(Y - cy) <= height / 2)
    n = np.full((nx, ny), n_clad, dtype=float)
    n[mask] = n_core
    return n, (x, y)


def make_circle(nx: int, ny: int, dx: float, dy: float, radius: float,
                 n_core: float = 3.4, n_clad: float = 1.44, center=(0.0, 0.0)):
    """A single circular high-index core in a uniform cladding.
    Returns (n[nx,ny], (x,y))."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    _validate_indices(n_core, n_clad)
    x, y, X, Y = make_grid(nx, ny, dx, dy)
    cx, cy = center
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2
    n = np.full((nx, ny), n_clad, dtype=float)
    n[mask] = n_core
    return n, (x, y)


def make_two_core_structure(nx: int, ny: int, dx: float, dy: float,
                             core_width: float, core_height: float, gap: float,
                             n_core: float = 3.4, n_clad: float = 1.44, shape: str = "rectangle"):
    """Two identical cores separated by `gap`, GAP FILLED WITH CLADDING
    (contrast with make_slot below, where the gap is a distinct slot
    material). Matches the layout motif of the attached patent FIG. 7
    (two 80um regions separated by a 5um gap), generalized to any
    core_width/gap. Returns (n[nx,ny], (x,y))."""
    if core_width <= 0 or core_height <= 0 or gap <= 0:
        raise ValueError("core_width, core_height, and gap must be positive")
    if shape not in ("rectangle", "circle"):
        raise ValueError("shape must be 'rectangle' or 'circle'")
    _validate_indices(n_core, n_clad)
    x, y, X, Y = make_grid(nx, ny, dx, dy)
    n = np.full((nx, ny), n_clad, dtype=float)
    center_offset = (core_width + gap) / 2.0
    if shape == "rectangle":
        mask1 = (np.abs(X + center_offset) <= core_width / 2) & (np.abs(Y) <= core_height / 2)
        mask2 = (np.abs(X - center_offset) <= core_width / 2) & (np.abs(Y) <= core_height / 2)
    else:
        r = core_width / 2
        mask1 = (X + center_offset) ** 2 + Y ** 2 <= r ** 2
        mask2 = (X - center_offset) ** 2 + Y ** 2 <= r ** 2
    n[mask1 | mask2] = n_core
    return n, (x, y)


def make_slot(nx: int, ny: int, dx: float, dy: float,
              core_width: float, core_height: float, gap: float,
              n_core: float = 3.4, n_clad: float = 1.44, n_slot: float = 1.0):
    """A slot waveguide: two cores separated by a narrow gap FILLED WITH A
    DISTINCT SLOT MATERIAL n_slot (typically low-index, e.g. air or SiO2)
    -- this is the literal geometry of the attached patent FIG. 7 (two
    80um Al/Si ridge regions separated by a 5um gap/trench down to a thin
    connecting Si layer). n_slot defaults to 1.0 (air-filled slot).
    Returns (n[nx,ny], (x,y))."""
    if n_slot <= 0:
        raise ValueError("n_slot must be positive")
    n, (x, y) = make_two_core_structure(nx, ny, dx, dy, core_width, core_height, gap,
                                         n_core, n_clad, shape="rectangle")
    _, _, X, Y = make_grid(nx, ny, dx, dy)
    slot_mask = (np.abs(X) <= gap / 2) & (np.abs(Y) <= core_height / 2)
    n[slot_mask] = n_slot
    return n, (x, y)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    nx, ny, dx, dy = 64, 64, 0.1, 0.1  # 64x64 grid, 0.1 um/pixel -> 6.4x6.4 um domain

    fig, axs = plt.subplots(2, 2, figsize=(9, 8))
    n1, _ = make_rectangle(nx, ny, dx, dy, width=2.0, height=1.0)
    n2, _ = make_circle(nx, ny, dx, dy, radius=1.0)
    n3, _ = make_two_core_structure(nx, ny, dx, dy, core_width=1.5, core_height=1.0, gap=0.5)
    n4, _ = make_slot(nx, ny, dx, dy, core_width=1.5, core_height=1.0, gap=0.5)

    for ax, n, title in zip(axs.flat, [n1, n2, n3, n4],
                             ["rectangle", "circle", "two cores (cladding gap)", "slot (air gap)"]):
        im = ax.imshow(n.T, origin="lower", cmap="viridis",
                        extent=[-nx * dx / 2, nx * dx / 2, -ny * dy / 2, ny * dy / 2])
        ax.set_title(title); ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
        plt.colorbar(im, ax=ax, label="n")
    plt.tight_layout()
    plt.savefig("geometry_demo.png", dpi=110)
    print("saved geometry_demo.png")
