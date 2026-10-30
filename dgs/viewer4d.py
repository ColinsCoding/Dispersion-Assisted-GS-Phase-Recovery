"""
A 4D time-series viewer: a 3D scalar field sampled over time, f(t, x, y, z).

In plain words
--------------
A 4D dataset can't be drawn directly, so a viewer shows it through 2D and 3D "windows":

* **slice**       fix one axis and a time, and draw the plane (e.g. the z = 0.5 nm plane at t = 2 fs)
* **projection**  collapse one axis with max (maximum-intensity projection) or sum
* **probe**       follow a single point through time, giving an ordinary 1D time series
* **isosurface**  show the voxels above a fraction of the peak as 3D points
* **animate**     step any of these through time

``VolumeSeries`` holds the data and the axes. ``box_superposition`` makes a physical example
from Knight Ch. 40: an electron in a 3D rigid box in a superposition of two energy states.
The probability density |psi|^2 sloshes back and forth with period h / (E2 - E1).

Units: lengths nm, time fs, energies eV.
"""
from dataclasses import dataclass, field

import numpy as np

from dgs import one_d_quantum as Q

HBAR_EV_FS = 0.6582119569          # hbar in eV fs
AXES = {"x": 1, "y": 2, "z": 3}


@dataclass
class VolumeSeries:
    data: np.ndarray            # shape (T, Nx, Ny, Nz)
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    name: str = "f"
    units: dict = field(default_factory=lambda: {"t": "fs", "x": "nm", "y": "nm", "z": "nm"})

    def __post_init__(self):
        self.data = np.asarray(self.data, dtype=float)
        if self.data.ndim != 4:
            raise ValueError(f"data must be 4D (T, Nx, Ny, Nz), got shape {self.data.shape}")
        for name, ax, n in (("t", self.t, 0), ("x", self.x, 1), ("y", self.y, 2), ("z", self.z, 3)):
            if len(ax) != self.data.shape[n]:
                raise ValueError(f"axis {name} has {len(ax)} points but data has {self.data.shape[n]}")
            if len(ax) > 1 and np.any(np.diff(ax) <= 0):
                raise ValueError(f"axis {name} must be strictly increasing")

    def axis(self, name):
        if name not in ("t", "x", "y", "z"):
            raise ValueError(f"unknown axis {name!r}; use t, x, y or z")
        return getattr(self, name)

    def index(self, name, value):
        """Nearest grid index to value on the named axis (raises if outside the axis range)."""
        a = self.axis(name)
        span = (a[-1] - a[0]) if len(a) > 1 else 0
        tol = 0.5 * (a[1] - a[0]) if len(a) > 1 else 0
        if value < a[0] - tol or value > a[-1] + tol:
            raise ValueError(f"{name} = {value} is outside [{a[0]:.4g}, {a[-1]:.4g}] {self.units[name]}")
        return int(np.argmin(np.abs(a - value))) if span else 0

    # ---- views ------------------------------------------------------------------------
    def slice(self, axis, value, t):
        """2D plane perpendicular to `axis` at `value`, at time t. Returns (plane, (h_axis, v_axis))."""
        if axis not in AXES:
            raise ValueError(f"slice axis must be x, y or z, got {axis!r}")
        frame = self.data[self.index("t", t)]
        k = self.index(axis, value)
        plane = np.take(frame, k, axis=AXES[axis] - 1)
        rest = [a for a in "xyz" if a != axis]
        return plane, (rest[0], rest[1])

    def projection(self, axis, t, kind="max"):
        """Collapse `axis` at time t with max (MIP) or sum. Returns (image, (h_axis, v_axis))."""
        if axis not in AXES:
            raise ValueError(f"projection axis must be x, y or z, got {axis!r}")
        if kind not in ("max", "sum", "mean"):
            raise ValueError(f"kind must be max, sum or mean, got {kind!r}")
        frame = self.data[self.index("t", t)]
        img = getattr(np, kind)(frame, axis=AXES[axis] - 1)
        rest = [a for a in "xyz" if a != axis]
        return img, (rest[0], rest[1])

    def probe(self, point):
        """Time series at the voxel nearest to point = (x, y, z)."""
        if len(point) != 3:
            raise ValueError(f"point must be (x, y, z), got {point}")
        i, j, k = (self.index(a, v) for a, v in zip("xyz", point))
        return self.data[:, i, j, k]

    def isosurface_points(self, t, level=0.5):
        """Voxel coordinates where f >= level * max(f at t). Returns (N, 3) array and values."""
        if not 0 < level < 1:
            raise ValueError(f"level must be a fraction in (0, 1), got {level}")
        frame = self.data[self.index("t", t)]
        mask = frame >= level * frame.max()
        ii, jj, kk = np.nonzero(mask)
        return np.column_stack([self.x[ii], self.y[jj], self.z[kk]]), frame[mask]

    def peak(self):
        """(time, (x, y, z), value) of the global maximum."""
        n, i, j, k = np.unravel_index(np.argmax(self.data), self.data.shape)
        return self.t[n], (self.x[i], self.y[j], self.z[k]), self.data[n, i, j, k]

    def center_of_mass(self):
        """Center of mass (x, y, z) of f at every time step, shape (T, 3)."""
        w = self.data.sum(axis=(1, 2, 3))
        cx = np.einsum("tijk,i->t", self.data, self.x) / w
        cy = np.einsum("tijk,j->t", self.data, self.y) / w
        cz = np.einsum("tijk,k->t", self.data, self.z) / w
        return np.column_stack([cx, cy, cz])


def box_superposition(L=1.0, n=32, t_max=None, frames=40, states=((1, 1, 1), (2, 1, 1)), weights=None):
    """|psi(x, y, z, t)|^2 for an electron in a cubic rigid box (side L nm) in a superposition.

    psi = sum_k c_k psi_nx(x) psi_ny(y) psi_nz(z) exp(-i E_k t / hbar), with E_k = E_1(L) (nx^2 + ny^2 + nz^2).
    The default two-state mix sloshes along x with period h / (E_211 - E_111).
    """
    if n < 4 or frames < 1:
        raise ValueError("need n >= 4 grid points per axis and frames >= 1")
    states = [tuple(int(q) for q in s) for s in states]
    w = np.ones(len(states)) if weights is None else np.asarray(weights, dtype=float)
    if len(w) != len(states):
        raise ValueError("weights must match states")
    c = w / np.linalg.norm(w)
    E1 = Q.box_energy(1, L)
    E = np.array([E1 * sum(q * q for q in s) for s in states])
    if t_max is None:
        dE = np.ptp(E) if len(E) > 1 else E[0]
        t_max = 2 * np.pi * HBAR_EV_FS / dE          # one beat period
    x = (np.arange(n) + 0.5) * L / n                  # cell centers, avoids the zero walls
    t = np.linspace(0, t_max, frames)
    psi = np.zeros((frames, n, n, n), dtype=complex)
    for ck, s, Ek in zip(c, states, E):
        spatial = np.einsum("i,j,k->ijk", Q.box_psi(x, s[0], L), Q.box_psi(x, s[1], L), Q.box_psi(x, s[2], L))
        psi += ck * np.exp(-1j * Ek * t / HBAR_EV_FS)[:, None, None, None] * spatial[None]
    return VolumeSeries(np.abs(psi) ** 2, t, x, x.copy(), x.copy(), name="|psi|^2 (nm^-3)")
