"""
hair_sim.py — Damped spring hair / fur simulation.
Ported from powerline_runner catenary jiggle — same ODE.

d^2x/dt^2 + 2*zeta*omega0*dx/dt + omega0^2*(x-x_rest) = F_ext

poodle:    omega0=12, zeta=0.3  (springy, underdamped)
retriever: omega0=6,  zeta=0.5  (slower, more damped)
straight:  omega0=4,  zeta=0.8  (near critically damped)
"""
import numpy as np
from dataclasses import dataclass, field


@dataclass
class HairStrand:
    n_segments : int   = 12
    rest_length: float = 0.08       # metres per segment
    omega0     : float = 10.0       # natural frequency rad/s
    zeta       : float = 0.35       # damping ratio
    curl_kappa : float = 0.6        # curvature (0=straight, 1=tight curl)

    pos: np.ndarray = field(init=False)
    vel: np.ndarray = field(init=False)

    def __post_init__(self):
        # T-pose: hang straight down
        self.pos = np.zeros((self.n_segments, 3))
        for i in range(self.n_segments):
            self.pos[i] = [0, -i * self.rest_length, 0]
        self.vel = np.zeros_like(self.pos)

    def rest_pos(self, root: np.ndarray, root_normal: np.ndarray) -> np.ndarray:
        """Compute rest positions following curl curvature."""
        tangent = root_normal.copy()
        bitangent = np.cross(tangent, np.array([0,0,1]))
        if np.linalg.norm(bitangent) < 1e-6:
            bitangent = np.cross(tangent, np.array([1,0,0]))
        bitangent /= np.linalg.norm(bitangent)

        rest = np.zeros((self.n_segments, 3))
        p = root.copy()
        angle = 0.0
        for i in range(self.n_segments):
            rest[i] = p
            # Curl: rotate tangent by kappa each segment
            angle += self.curl_kappa * 0.25
            t = (np.cos(angle) * tangent +
                 np.sin(angle) * bitangent)
            t /= np.linalg.norm(t) + 1e-9
            p += t * self.rest_length
        return rest

    def update(self, dt: float, root: np.ndarray,
               root_normal: np.ndarray,
               wind: np.ndarray = None,
               gravity: np.ndarray = None):
        if gravity is None:
            gravity = np.array([0, -9.81, 0])
        if wind is None:
            wind = np.zeros(3)

        rest = self.rest_pos(root, root_normal)

        # Constraint: root follows skeleton
        self.pos[0] = root
        self.vel[0] = np.zeros(3)

        for i in range(1, self.n_segments):
            spring = -self.omega0**2 * (self.pos[i] - rest[i])
            damp   = -2 * self.zeta * self.omega0 * self.vel[i]
            ext    = gravity + wind * (i / self.n_segments)
            accel  = spring + damp + ext
            self.vel[i] += accel * dt
            self.pos[i] += self.vel[i] * dt

    def apply_impulse(self, force: np.ndarray, segment: int = None):
        """Zoomies / wind gust."""
        if segment is None:
            self.vel += force
        else:
            self.vel[segment] += force


class DogFur:
    """Collection of strands covering dog surface voxels."""
    def __init__(self, surface_pts: np.ndarray,
                 normals: np.ndarray,
                 breed: str = "golden_doodle"):
        params = {
            "golden_doodle": dict(omega0=10, zeta=0.35, curl_kappa=0.55),
            "poodle":        dict(omega0=14, zeta=0.25, curl_kappa=0.85),
            "retriever":     dict(omega0=6,  zeta=0.55, curl_kappa=0.12),
        }
        p = params.get(breed, params["golden_doodle"])

        self.strands = [
            HairStrand(**p) for _ in range(len(surface_pts))
        ]
        self.roots   = surface_pts
        self.normals = normals

    def update(self, dt: float, wind=None):
        for strand, root, normal in zip(self.strands,
                                        self.roots,
                                        self.normals):
            strand.update(dt, root, normal, wind=wind)

    def zoomies(self):
        """Apply random impulses — dog is running fast."""
        for strand in self.strands:
            impulse = np.random.randn(3) * 2.5
            impulse[1] = abs(impulse[1])
            strand.apply_impulse(impulse)
