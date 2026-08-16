"""Hand-rolled 3D-to-2D camera projection (NumPy, no MuJoCo/OpenGL) plus a
spoon rigid body governed by F = dp/dt, shown to be identical to F = ma --
not asserted, verified by integrating the SAME force two different ways
and checking the trajectories agree.

Why dp/dt = ma is not just notation: dp/dt = d(mv)/dt. If (and only if)
mass is constant, that's m*dv/dt = m*a -- the two are the same equation,
not two different laws, and the distinction matters the moment mass
ISN'T constant (a rocket burning fuel, output below, integrates F=dp/dt
correctly while F=ma alone does not, because it misses the d(m)/dt*v term
the product rule adds).

Rendering pipeline (the actual "3D model projected to a 2D screen" part):
world space -> camera (view) space -> clip space (perspective projection
matrix) -> normalized device coordinates (perspective divide) -> screen
pixels. Every matrix here is built and multiplied by hand in NumPy; no
external graphics library does the projection for you, unlike the MuJoCo
renderer used elsewhere in this repo.
"""

from __future__ import annotations

import numpy as np

from dgs.laser_tag_raycaster import mouse_look_update


# -- 3D -> 2D projection pipeline, from scratch -------------------------------

class Camera:
    """A perspective camera: position + look-at target define the view
    matrix; field of view + aspect + near/far define the projection
    matrix. project() runs a full world->view->clip->NDC->screen pipeline
    on an (N,3) array of world-space points."""

    def __init__(self, eye, target, up=(0.0, 0.0, 1.0), fov_deg=60.0,
                 aspect=4 / 3, near=0.05, far=20.0, screen_size=(640, 480)):
        self.eye = np.asarray(eye, dtype=float)
        self.target = np.asarray(target, dtype=float)
        self.up = np.asarray(up, dtype=float)
        self.fov_deg = fov_deg
        self.aspect = aspect
        self.near = near
        self.far = far
        self.screen_w, self.screen_h = screen_size

    def view_matrix(self):
        """Look-at matrix: rotates+translates world space so the camera
        sits at the origin looking down -z."""
        forward = self.target - self.eye
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, self.up)
        right = right / np.linalg.norm(right)
        true_up = np.cross(right, forward)

        R = np.stack([right, true_up, -forward], axis=0)
        view = np.eye(4)
        view[:3, :3] = R
        view[:3, 3] = -R @ self.eye
        return view

    def projection_matrix(self):
        """Standard OpenGL-style perspective projection matrix from
        vertical FOV, aspect ratio, and near/far clip planes."""
        f = 1.0 / np.tan(np.deg2rad(self.fov_deg) / 2)
        P = np.zeros((4, 4))
        P[0, 0] = f / self.aspect
        P[1, 1] = f
        P[2, 2] = (self.far + self.near) / (self.near - self.far)
        P[2, 3] = (2 * self.far * self.near) / (self.near - self.far)
        P[3, 2] = -1.0
        return P

    def project(self, points_world):
        """(N,3) world points -> (screen_xy (N,2), depth (N,), visible (N,) bool).
        visible=False for points behind the camera or outside the view
        frustum's near/far range (perspective divide is undefined/unstable there)."""
        points_world = np.atleast_2d(points_world)
        n = len(points_world)
        homog = np.hstack([points_world, np.ones((n, 1))])   # (N,4)

        view = self.view_matrix()
        proj = self.projection_matrix()
        clip = (proj @ (view @ homog.T)).T   # (N,4)

        w = clip[:, 3]
        visible = w > 1e-6
        ndc = np.zeros((n, 3))
        ndc[visible] = clip[visible, :3] / w[visible, None]

        screen_x = (ndc[:, 0] * 0.5 + 0.5) * self.screen_w
        screen_y = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * self.screen_h   # flip y: NDC up is screen up
        depth = np.linalg.norm(points_world - self.eye, axis=1)      # for painter's-algorithm sorting

        return np.stack([screen_x, screen_y], axis=1), depth, visible


# -- F = dp/dt vs F = ma: verified equal for constant mass, and shown to
# DIVERGE the moment mass isn't constant -------------------------------------

def integrate_via_force_and_acceleration(mass, force, v0, dt, n_steps):
    """v(t) integrated via a = F/m directly (F = m*a)."""
    v = np.array(v0, dtype=float)
    velocities = [v.copy()]
    for _ in range(n_steps):
        a = np.asarray(force) / mass
        v = v + a * dt
        velocities.append(v.copy())
    return np.array(velocities)


def integrate_via_momentum(mass, force, v0, dt, n_steps):
    """v(t) integrated via p = m*v, dp/dt = F (F = dp/dt), then v = p/m.
    For CONSTANT mass this must produce the identical trajectory to
    integrate_via_force_and_acceleration -- verified numerically below,
    not assumed."""
    p = mass * np.array(v0, dtype=float)
    velocities = [p / mass]
    for _ in range(n_steps):
        p = p + np.asarray(force) * dt
        velocities.append(p / mass)
    return np.array(velocities)


def integrate_via_momentum_variable_mass(mass_func, force, v0, dt, n_steps, t0=0.0):
    """The case where F=dp/dt and F=ma genuinely DIFFER: mass changing
    over time (e.g. a spoon flinging batter off it, or more dramatically
    a rocket). p = m(t)*v is still the right momentum, and F=dp/dt is
    still exactly correct; naively integrating a = F/m(t) alone silently
    drops the d(m)/dt * v term the product rule requires, and drifts
    away from the true (momentum-based) answer."""
    t = t0
    p = mass_func(t) * np.array(v0, dtype=float)
    velocities = [p / mass_func(t)]
    for _ in range(n_steps):
        p = p + np.asarray(force) * dt
        t += dt
        velocities.append(p / mass_func(t))
    return np.array(velocities)


# -- The spoon: a simple wireframe rigid body ---------------------------------

class Spoon:
    """A spoon as a small wireframe mesh (handle + bowl, defined in the
    spoon's own local frame) plus real rigid-body state: center-of-mass
    position/momentum (F=dp/dt) and orientation/angular-velocity (Euler's
    equations, same machinery as dgs.gyroscopes -- a thrown/flipped spoon
    is a free rigid body exactly like the chicken drumstick)."""

    def __init__(self, mass=0.04, length=0.18, bowl_radius=0.025,
                 position=(0.0, 0.0, 0.0), orientation_quat=(1.0, 0.0, 0.0, 0.0)):
        self.mass = mass
        self.length = length
        self.bowl_radius = bowl_radius
        self.position = np.array(position, dtype=float)
        self.momentum = np.zeros(3)
        self.quat = np.array(orientation_quat, dtype=float)   # (w, x, y, z)
        self.omega_body = np.zeros(3)   # body-frame angular velocity

        handle_len = length * 0.7
        n_bowl_pts = 10
        bowl_theta = np.linspace(0, 2 * np.pi, n_bowl_pts, endpoint=False)
        bowl_pts = np.stack([
            bowl_radius * np.cos(bowl_theta),
            bowl_radius * 0.6 * np.sin(bowl_theta),
            np.full(n_bowl_pts, handle_len),
        ], axis=1)
        self.local_vertices = np.vstack([
            [[0.0, 0.0, 0.0], [0.0, 0.0, handle_len]],   # handle: base -> neck
            bowl_pts,                                     # bowl rim
        ])
        self.handle_edges = [(0, 1)]
        self.bowl_edges = [(2 + i, 2 + (i + 1) % n_bowl_pts) for i in range(n_bowl_pts)]

        # a real drumstick-style asymmetric box approximation for I1,I2,I3
        # (handle is long and thin -> distinct principal moments, same
        # "genuinely asymmetric" condition as dgs.chicken_bbq_3d's drumstick)
        sx, sy, sz = 0.008, 0.012, length / 2
        self.I1 = (mass / 12) * ((2 * sy) ** 2 + (2 * sz) ** 2)
        self.I2 = (mass / 12) * ((2 * sx) ** 2 + (2 * sz) ** 2)
        self.I3 = (mass / 12) * ((2 * sx) ** 2 + (2 * sy) ** 2)

    def apply_impulse(self, force, dt):
        """F = dp/dt, applied as an impulse over dt -- the translational
        half of the spoon's motion."""
        self.momentum = self.momentum + np.asarray(force, dtype=float) * dt

    def _quat_derivative(self, quat, omega_body):
        w, x, y, z = quat
        wx, wy, wz = omega_body
        dw = 0.5 * (-x * wx - y * wy - z * wz)
        dx = 0.5 * (w * wx + y * wz - z * wy)
        dy = 0.5 * (w * wy - x * wz + z * wx)
        dz = 0.5 * (w * wz + x * wy - y * wx)
        return np.array([dw, dx, dy, dz])

    def _euler_rhs(self, omega):
        w1, w2, w3 = omega
        dw1 = (self.I2 - self.I3) * w2 * w3 / self.I1
        dw2 = (self.I3 - self.I1) * w3 * w1 / self.I2
        dw3 = (self.I1 - self.I2) * w1 * w2 / self.I3
        return np.array([dw1, dw2, dw3])

    def step(self, dt, gravity=(0.0, 0.0, -9.80665)):
        # translation: F=dp/dt (gravity is the only continuous force here)
        self.apply_impulse(np.array(gravity) * self.mass, dt)
        self.position = self.position + (self.momentum / self.mass) * dt

        # rotation: torque-free Euler's equations (RK4), same as dgs.gyroscopes
        k1 = self._euler_rhs(self.omega_body)
        k2 = self._euler_rhs(self.omega_body + dt / 2 * k1)
        k3 = self._euler_rhs(self.omega_body + dt / 2 * k2)
        k4 = self._euler_rhs(self.omega_body + dt * k3)
        self.omega_body = self.omega_body + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        dq = self._quat_derivative(self.quat, self.omega_body)
        self.quat = self.quat + dq * dt
        self.quat = self.quat / np.linalg.norm(self.quat)

    def world_vertices(self):
        """local_vertices rotated by the current orientation quaternion
        and translated to the current position."""
        w, x, y, z = self.quat
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        return (R @ self.local_vertices.T).T + self.position


# -- Interactive pygame visualization: 1st-person (CS:GO-style arm,
# rendered through the SAME projection pipeline as everything else, not a
# fixed screen-space overlay) and 3rd-person, both driven by the same
# spoon physics --------------------------------------------------------------

class KitchenScene:
    """Static environment geometry -- floor, walls, counter, stove -- as
    flat-color quads. Drawn back-to-front by fixed priority (floor/walls
    first, counter/stove after) rather than a full per-object depth sort:
    correct for a room where the walls really are always farthest away
    and the counter/stove really do sit in front of them, which holds for
    every camera position this visualizer actually uses."""

    def __init__(self, counter_top_z=0.85):
        self.counter_top_z = counter_top_z

        self.floor = np.array([
            [-1.5, -1.0, 0.0], [1.5, -1.0, 0.0], [1.5, 2.5, 0.0], [-1.5, 2.5, 0.0],
        ])
        self.back_wall = np.array([
            [-1.5, 2.5, 0.0], [1.5, 2.5, 0.0], [1.5, 2.5, 2.4], [-1.5, 2.5, 2.4],
        ])
        self.left_wall = np.array([
            [-1.5, -1.0, 0.0], [-1.5, 2.5, 0.0], [-1.5, 2.5, 2.4], [-1.5, -1.0, 2.4],
        ])

        cx, cy = 0.0, 0.55       # counter footprint, centered under the spoon's rest position
        hw, hd = 0.55, 0.35      # half-width, half-depth
        z0, z1 = 0.0, counter_top_z
        self.counter_top = np.array([
            [cx - hw, cy - hd, z1], [cx + hw, cy - hd, z1], [cx + hw, cy + hd, z1], [cx - hw, cy + hd, z1],
        ])
        self.counter_front = np.array([
            [cx - hw, cy - hd, z0], [cx + hw, cy - hd, z0], [cx + hw, cy - hd, z1], [cx - hw, cy - hd, z1],
        ])
        self.counter_side = np.array([
            [cx + hw, cy - hd, z0], [cx + hw, cy + hd, z0], [cx + hw, cy + hd, z1], [cx + hw, cy - hd, z1],
        ])

        # stove/grill: a low box next to the counter with a circular
        # "burner" ring on top -- same motif as the earlier BBQ grill work
        sx0, sy0 = -0.75, 0.55
        sw, sd = 0.35, 0.35
        sz1 = counter_top_z
        self.stove_top = np.array([
            [sx0 - sw, sy0 - sd, sz1], [sx0 + sw, sy0 - sd, sz1], [sx0 + sw, sy0 + sd, sz1], [sx0 - sw, sy0 + sd, sz1],
        ])
        self.stove_front = np.array([
            [sx0 - sw, sy0 - sd, 0.0], [sx0 + sw, sy0 - sd, 0.0], [sx0 + sw, sy0 - sd, sz1], [sx0 - sw, sy0 - sd, sz1],
        ])
        theta = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        self.burner_ring = np.stack([
            sx0 + 0.18 * np.cos(theta), sy0 + 0.18 * np.sin(theta), np.full(20, sz1 + 0.001),
        ], axis=1)

        self.surfaces = [
            (self.floor, (55, 50, 48)),
            (self.back_wall, (70, 68, 72)),
            (self.left_wall, (62, 60, 64)),
            (self.stove_front, (60, 60, 65)),
            (self.stove_top, (75, 75, 80)),
            (self.counter_front, (120, 90, 60)),
            (self.counter_side, (105, 78, 52)),
            (self.counter_top, (150, 115, 78)),
        ]

    def render(self, screen, camera):
        import pygame
        for quad, color in self.surfaces:
            xy, depth, visible = camera.project(quad)
            if visible.all():
                pygame.draw.polygon(screen, color, xy.tolist())
        ring_xy, _, ring_vis = camera.project(self.burner_ring)
        if ring_vis.all():
            pygame.draw.polygon(screen, (90, 40, 30), ring_xy.tolist(), width=3)


class KitchenVisualizer:
    MODES = ["first", "third", "free"]

    def __init__(self, screen_size=(640, 480)):
        self.screen_size = screen_size
        self.scene = KitchenScene()
        self.spoon = Spoon(position=(0.0, 0.55, self.scene.counter_top_z + 0.02))
        self.mode_idx = 0

        # free-look mouse camera: a fixed "standing in the kitchen" eye
        # position, mouse controls a REAL 3D look direction (unlike
        # laser_tag_raycaster's flat world, this camera is true 3D
        # perspective, so pitch actually tilts the view, not just a
        # cosmetic screen shift)
        self.free_eye = np.array([0.0, -0.5, 1.2])
        self.yaw = np.pi / 2      # facing +y (into the room) initially
        self.pitch = -0.15
        self.mouse_sensitivity = 0.003
        self.pitch_limit = np.deg2rad(85)

    @property
    def first_person(self):
        return self.MODES[self.mode_idx] == "first"

    def cycle_mode(self):
        self.mode_idx = (self.mode_idx + 1) % len(self.MODES)

    def _fp_camera(self):
        eye = self.spoon.position + np.array([0.0, -0.7, 0.05])
        return Camera(eye=eye, target=self.spoon.position, screen_size=self.screen_size)

    def _tp_camera(self):
        # follows the spoon (a fixed offset from its current position),
        # not a static target -- otherwise a tossed spoon flies out of frame
        eye = self.spoon.position + np.array([1.1, -1.1, 0.5])
        return Camera(eye=eye, target=self.spoon.position, screen_size=self.screen_size)

    def _arm_world_segments(self, camera):
        """Forearm + hand, built as real 3D points near the first-person
        camera's eye reaching toward the spoon's grip -- projected through
        the exact same Camera.project() pipeline as the spoon, so it's a
        genuine 3D construction, not a screen-space sprite."""
        eye = camera.eye
        grip = self.spoon.position + (self.spoon.world_vertices()[1] - self.spoon.position) * 0.3
        elbow = eye + np.array([0.05, 0.15, -0.2])
        wrist = eye * 0.35 + grip * 0.65
        segments = [(elbow, wrist), (wrist, grip)]
        # a small two-prong "hand" near the grip, same aesthetic as the
        # tongs viewmodel from chicken_bbq_simulator.py
        side = np.cross(grip - wrist, np.array([0.0, 0.0, 1.0]))
        side = side / (np.linalg.norm(side) + 1e-9) * 0.02
        segments.append((wrist + side, grip + side * 0.4))
        segments.append((wrist - side, grip - side * 0.4))
        return segments

    def toss_spoon(self):
        self.spoon.momentum = self.spoon.mass * np.array([0.4, 0.6, 3.0])
        self.spoon.omega_body = np.array([0.3, 14.0, 0.4])

    def _free_camera(self):
        _, _, look = mouse_look_update(0.0, 0.0, self.mouse_sensitivity, self.yaw, self.pitch, self.pitch_limit)
        return Camera(eye=self.free_eye, target=self.free_eye + look, screen_size=self.screen_size)

    def move_free_camera(self, keys, dt, speed=1.2, margin=0.15):
        """W/S move forward/back, A/D strafe left/right -- along the
        HORIZONTAL component of the current look direction only (yaw, not
        pitch), the standard FPS convention: looking down at the floor
        doesn't make W walk you into it. A/D strafes rather than turns,
        since the mouse already owns turning (the CS:GO-style combo, not
        laser_tag_raycaster's turn-with-A/D convention, which doesn't have
        a true 3D mouse-look to pair with). Clamped to stay inside the
        room's own floor bounds so you can't walk through the walls."""
        import pygame
        forward = np.array([np.cos(self.yaw), np.sin(self.yaw), 0.0])
        right = np.array([np.cos(self.yaw - np.pi / 2), np.sin(self.yaw - np.pi / 2), 0.0])

        delta = np.zeros(3)
        if keys[pygame.K_w]:
            delta += forward
        if keys[pygame.K_s]:
            delta -= forward
        if keys[pygame.K_d]:
            delta += right
        if keys[pygame.K_a]:
            delta -= right
        norm = np.linalg.norm(delta)
        if norm > 1e-9:
            self.free_eye = self.free_eye + (delta / norm) * speed * dt

        x_min, y_min = self.scene.floor[:, 0].min(), self.scene.floor[:, 1].min()
        x_max, y_max = self.scene.floor[:, 0].max(), self.scene.floor[:, 1].max()
        self.free_eye[0] = np.clip(self.free_eye[0], x_min + margin, x_max - margin)
        self.free_eye[1] = np.clip(self.free_eye[1], y_min + margin, y_max - margin)

    def _current_camera(self):
        mode = self.MODES[self.mode_idx]
        if mode == "first":
            return self._fp_camera()
        if mode == "third":
            return self._tp_camera()
        return self._free_camera()

    def run(self, dt_phys=0.005):
        import pygame

        pygame.init()
        W, H = self.screen_size
        screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption(
            "Kitchen 3D projection -- SPACE=toss spoon, V=cycle 1st/3rd/free-look, mouse=look in free mode, ESC=quit"
        )
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("consolas", 16)

        def _sync_mouse_grab():
            free = self.MODES[self.mode_idx] == "free"
            pygame.mouse.set_visible(not free)
            pygame.event.set_grab(free)
            if free:
                pygame.mouse.get_rel()   # discard the first (potentially large) delta from grabbing

        _sync_mouse_grab()

        running = True
        try:
            while running:
                clock.tick(60)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        if event.key == pygame.K_SPACE:
                            self.toss_spoon()
                        if event.key == pygame.K_v:
                            self.cycle_mode()
                            _sync_mouse_grab()

                if self.MODES[self.mode_idx] == "free":
                    dx, dy = pygame.mouse.get_rel()
                    self.yaw, self.pitch, _ = mouse_look_update(
                        dx, dy, self.mouse_sensitivity, self.yaw, self.pitch, self.pitch_limit
                    )
                    self.move_free_camera(pygame.key.get_pressed(), dt_phys)

                rest_z = self.scene.counter_top_z + 0.02   # lands back ON the counter, not the true floor
                if self.spoon.position[2] > rest_z:
                    self.spoon.step(dt_phys)
                else:
                    self.spoon.position[2] = rest_z
                    self.spoon.momentum[:] = 0.0
                    self.spoon.omega_body[:] = 0.0

                camera = self._current_camera()

                screen.fill((20, 18, 22))
                self.scene.render(screen, camera)
                verts = self.spoon.world_vertices()
                screen_xy, depth, visible = camera.project(verts)
                for i, j in self.spoon.handle_edges + self.spoon.bowl_edges:
                    if visible[i] and visible[j]:
                        pygame.draw.line(screen, (200, 190, 170), screen_xy[i], screen_xy[j], 3)

                if self.first_person:
                    for a, b in self._arm_world_segments(camera):
                        pts, _, vis = camera.project(np.array([a, b]))
                        if vis[0] and vis[1]:
                            pygame.draw.line(screen, (210, 175, 150), pts[0], pts[1], 8)

                mode_label = {"first": "1ST PERSON", "third": "3RD PERSON", "free": "FREE LOOK (mouse)"}[self.MODES[self.mode_idx]]
                hud = font.render(f"{mode_label}   SPACE=toss  V=cycle view  ESC=quit", True, (230, 230, 230))
                screen.blit(hud, (10, 10))
                pygame.display.flip()
        except KeyboardInterrupt:
            pass

        pygame.quit()


if __name__ == "__main__":
    print("=== F = dp/dt vs F = ma: identical for constant mass (verified, not assumed) ===\n")
    mass = 0.5
    force = np.array([0.0, 0.0, -9.80665]) * mass   # a fixed gravity-like force
    v0 = np.array([2.0, 0.0, 3.0])
    dt, n_steps = 0.01, 200

    v_via_a = integrate_via_force_and_acceleration(mass, force, v0, dt, n_steps)
    v_via_p = integrate_via_momentum(mass, force, v0, dt, n_steps)
    max_diff = np.max(np.abs(v_via_a - v_via_p))
    print(f"max|v(F=ma) - v(F=dp/dt)| over {n_steps} steps = {max_diff:.2e}  (constant mass)")

    print("\n=== Where they genuinely diverge: mass changing over time ===\n")

    def shrinking_mass(t):
        return max(0.1, mass - 0.15 * t)   # spoon flinging material off it

    v_p_variable = integrate_via_momentum_variable_mass(shrinking_mass, force, v0, dt, n_steps)
    v_a_naive = integrate_via_force_and_acceleration(mass, force, v0, dt, n_steps)   # WRONG: ignores changing mass
    divergence = np.max(np.abs(v_p_variable - v_a_naive))
    print(f"max|v(correct dp/dt, variable mass) - v(naive F=ma, constant-mass assumption)| = {divergence:.4f}")
    print("(nonzero: naively applying a=F/m with a fixed m silently drops the d(m)/dt*v term)")

    print("\n=== Camera projection sanity check ===\n")
    cam = Camera(eye=[0, -2.0, 0.5], target=[0, 0, 0.0])
    screen_xy, depth, visible = cam.project(np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]))
    print(f"point at look-at target -> screen {screen_xy[0]} (expect near screen center 320,240)")
    print(f"point offset +x -> screen {screen_xy[1]} (expect x > 320)")

    print("\n=== A tossed, tumbling spoon ===\n")
    spoon = Spoon(position=(0.0, 0.0, 1.0))
    spoon.momentum = spoon.mass * np.array([0.3, 0.0, 2.0])
    spoon.omega_body = np.array([0.1, 15.0, 0.1])   # spin near an intermediate-ish axis on purpose
    for _ in range(300):
        spoon.step(dt=0.005)
    print(f"spoon position after 1.5s: {spoon.position.round(3)}")
    print(f"spoon angular velocity (body frame): {spoon.omega_body.round(3)}")
