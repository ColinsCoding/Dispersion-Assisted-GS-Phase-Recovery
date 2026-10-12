"""Chicken-chase game logic (no pygame) -- the Python twin of
learn/chicken_chase/chicken_chase.c, so it can be tested and drawn by
dgs/viewer_chicken_pygame.py.

The calculus behind the HUD:
  * A chicken fleeing STRAIGHT AWAY from the player: the gap obeys
        dr/dt = -(v_p - v_c)  ->  T = r0 / (v_p - v_c)          (until it hits a wall)
  * Bouguer's pursuit (the "P" lab): target runs perpendicular at v_c = k v_p,
    chaser always aims at it:
        T = d v_p / (v_p^2 - v_c^2)
    derived from the pursuit ODE; compared live with the measured capture time.

Units: world is 128 x 96 cells (same as the C program), speeds in cells/step.
"""
from dataclasses import dataclass, field

import numpy as np

WORLD_W, WORLD_H = 128, 96


@dataclass
class Params:
    v_player: float = 0.9
    v_wander: float = 0.25
    v_flee: float = 0.75
    flee_radius: float = 18.0
    catch_radius: float = 2.0
    n_chickens: int = 12

    def __post_init__(self):
        if min(self.v_player, self.flee_radius, self.catch_radius) <= 0 or self.n_chickens < 1:
            raise ValueError("speeds, radii must be positive; n_chickens >= 1")
        if self.v_flee < 0 or self.v_wander < 0:
            raise ValueError("chicken speeds must be non-negative")


@dataclass
class World:
    params: Params = field(default_factory=Params)
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        p = self.params
        n = p.n_chickens
        self.chickens = np.column_stack([
            8 + self.rng.uniform(size=n) * (WORLD_W - 16),
            8 + self.rng.uniform(size=n) * (WORLD_H - 16),
            2 * np.pi * self.rng.uniform(size=n)])
        self.alive = np.ones(n, bool)
        self.player = np.array([WORLD_W / 2, WORLD_H / 2])
        self.steps = 0
        self.catch_steps = []

    # ── queries ──────────────────────────────────────────────────────
    def nearest(self):
        """(index, distance) of the nearest live chicken, or (None, inf)."""
        if not self.alive.any():
            return None, np.inf
        d = np.hypot(*(self.chickens[:, :2] - self.player).T)
        d[~self.alive] = np.inf
        i = int(np.argmin(d))
        return i, float(d[i])

    def flee_prediction(self):
        """Steps to catch the nearest chicken if it flees straight away.

        Continuous model: dr/dt = -(v_p - v_c). Game-loop detail: the player
        moves BEFORE the catch check, so the effective catch distance is
        r_c + v_p, giving  T = (r - r_c - v_p) / (v_p - v_c).
        """
        i, d = self.nearest()
        p = self.params
        if i is None:
            return None
        if p.v_player <= p.v_flee:
            return np.inf
        return (d - p.catch_radius - p.v_player) / (p.v_player - p.v_flee)

    # ── dynamics ─────────────────────────────────────────────────────
    def step(self, move=None, autopilot=False):
        """Advance one step. move = (dx, dy) direction from the keyboard."""
        p = self.params
        if autopilot:
            i, d = self.nearest()
            if i is not None and d > 0:
                self.player += p.v_player * (self.chickens[i, :2] - self.player) / d
        elif move is not None and (move[0] or move[1]):
            m = np.asarray(move, float)
            self.player += p.v_player * m / np.hypot(*m)
        self.player = np.clip(self.player, 1, [WORLD_W - 2, WORLD_H - 2])

        for i in np.flatnonzero(self.alive):
            x, y, h = self.chickens[i]
            dx, dy = x - self.player[0], y - self.player[1]
            d = np.hypot(dx, dy)
            if d < p.catch_radius:
                self.alive[i] = False
                self.catch_steps.append(self.steps)
                continue
            if d < p.flee_radius:
                h = np.arctan2(dy, dx)
                v = p.v_flee
            else:
                h += 0.6 * (self.rng.uniform() - 0.5)
                v = p.v_wander
            x, y = x + v * np.cos(h), y + v * np.sin(h)
            if x < 2 or x > WORLD_W - 3:
                h, x = np.pi - h, np.clip(x, 2, WORLD_W - 3)
            if y < 2 or y > WORLD_H - 3:
                h, y = -h, np.clip(y, 2, WORLD_H - 3)
            self.chickens[i] = (x, y, h)
        self.steps += 1

    @property
    def caught(self):
        return int((~self.alive).sum())


# ── Bouguer pursuit lab ──────────────────────────────────────────────

def bouguer_capture_time(d, v_p, v_c):
    """T = d v_p / (v_p^2 - v_c^2); None if v_c >= v_p."""
    if d <= 0 or v_p <= 0 or v_c < 0:
        raise ValueError("d > 0, v_p > 0, v_c >= 0")
    return None if v_c >= v_p else d * v_p / (v_p ** 2 - v_c ** 2)


class PursuitLab:
    """Chaser at (x0, y0) aims at a target starting d to its right that runs
    straight up at v_c. Records the curve and the measured capture time."""

    def __init__(self, d=60.0, v_p=0.9, k=0.6, origin=(20.0, 85.0), dt=0.01, eps=0.02):
        if not 0 <= k < 1:
            raise ValueError("0 <= k < 1 for a capture to exist")
        self.v_p, self.v_c, self.d, self.dt, self.eps = v_p, k * v_p, d, dt, eps
        self.chaser = np.array(origin, float)
        self.target = self.chaser + [d, 0.0]
        self.t = 0.0
        self.path = [self.chaser.copy()]
        self.done = False
        self.predicted = bouguer_capture_time(d, v_p, self.v_c)

    def advance(self, steps=1):
        for _ in range(int(steps / self.dt)):
            if self.done:
                return
            r = self.target - self.chaser
            dist = np.hypot(*r)
            if dist < self.eps:
                self.done = True
                return
            self.chaser += self.dt * self.v_p * r / dist
            self.target[1] -= self.dt * self.v_c          # screen "up" = -y
            self.t += self.dt
            self.path.append(self.chaser.copy())


# ── 1-bit framebuffer (same layout as the C program) ─────────────────

def render_bits(world):
    """Draw the world into a (96, 128) 0/1 array; return it and its packed bytes."""
    img = np.zeros((WORLD_H, WORLD_W), np.uint8)
    img[0, :] = img[-1, :] = img[:, 0] = img[:, -1] = 1
    for (x, y, _), a in zip(world.chickens, world.alive):
        if a:
            xi, yi = int(round(x)), int(round(y))
            img[max(yi - 1, 0):yi + 2, max(xi - 1, 0):xi + 2] = 1
    px, py = np.round(world.player).astype(int)
    img[max(py - 2, 0):py + 3, max(px - 2, 0):px + 3] = 1
    a = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    rx = np.round(world.player[0] + world.params.flee_radius * np.cos(a)).astype(int)
    ry = np.round(world.player[1] + world.params.flee_radius * np.sin(a)).astype(int)
    ok = (rx >= 0) & (rx < WORLD_W) & (ry >= 0) & (ry < WORLD_H)
    img[ry[ok], rx[ok]] = 1
    return img, np.packbits(img, axis=None)
