#!/usr/bin/env python3
"""
Indiana Jones Rigid Body Arena
================================
Monte Carlo spawn  |  Rolling rigid bodies  |  Observer camera  |  Gauge invariance

Physics:
  Solid sphere cross-section  I = 2/5 m r^2
  Solid disk                  I = 1/2 m r^2
  Rolling constraint          v = omega * r  (no-slip on floor)
  Elastic collision impulse   j = -(1+e) dv.n / (1/ma + 1/mb)
  Gauge invariance            E_total invariant under camera translation

Controls:
  R       -- respawn Monte Carlo wave
  SPACE   -- spawn extra boulder
  G       -- toggle gauge frame overlay
  ESC     -- quit
"""

import pygame
import numpy as np
import math
import sys

# ── World constants ────────────────────────────────────────────────────────────
W, H         = 1280, 720
FPS          = 60
WORLD_W      = W * 4          # scrollable world width
G_ACC        = 520.0          # px / s^2
FLOOR_Y      = H - 90
RESTITUTION  = 0.62
N_SPAWN      = 20
RNG          = np.random.default_rng(1492)   # 1492: Columbus year, Indy era

# ── Palette ────────────────────────────────────────────────────────────────────
BG           = (8,  10, 18)
FLOOR_COL    = (55, 40, 20)
FLOOR_EDGE   = (110, 80, 35)
INDY_SKIN    = (230, 185, 110)
INDY_HAT     = (55,  30,  8)
INDY_SHIRT   = (180, 130, 60)
HUD_GREEN    = (80, 255, 130)
HUD_BLUE     = (80, 180, 255)
HUD_ORANGE   = (255, 160, 40)
HUD_DIM      = (90,  90, 100)

NEON = [
    (255,  50, 110),   # hot pink
    ( 50, 210, 255),   # cyan
    (255, 210,   0),   # gold
    (170,  50, 255),   # violet
    (  0, 255, 160),   # mint
    (255, 100,  30),   # orange
    (200, 255,  50),   # lime
    ( 30, 160, 255),   # azure
]


# ── Particle system ────────────────────────────────────────────────────────────
class Particle:
    __slots__ = ('x','y','vx','vy','life','max_life','color','r')
    def __init__(self, x, y, color):
        angle = RNG.uniform(0, 2*math.pi)
        speed = RNG.uniform(60, 280)
        self.x, self.y   = float(x), float(y)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed - 120
        self.life = self.max_life = RNG.uniform(0.25, 0.7)
        self.color = color
        self.r = RNG.integers(2, 6)

    def step(self, dt):
        self.vy += G_ACC * dt * 0.3
        self.x  += self.vx * dt
        self.y  += self.vy * dt
        self.life -= dt

    def alive(self):
        return self.life > 0

    def draw(self, surf, cam_x):
        alpha = self.life / self.max_life
        c = tuple(int(v * alpha) for v in self.color)
        pygame.draw.circle(surf, c, (int(self.x - cam_x), int(self.y)), self.r)


# ── Rigid body ─────────────────────────────────────────────────────────────────
class Body:
    def __init__(self, x, y, r, vx, vy, btype, color, idx):
        self.x, self.y   = float(x), float(y)
        self.r           = float(r)
        self.btype       = btype            # 'sphere' | 'disk'
        self.m           = 0.004 * r**2     # mass ~ area
        self.I_fac       = 0.4 if btype == 'sphere' else 0.5
        self.I           = self.I_fac * self.m * r**2
        self.vx          = float(vx)
        self.vy          = float(vy)
        self.omega       = vx / max(r, 1e-6)
        self.theta       = float(RNG.uniform(0, 2*math.pi))
        self.color       = color
        self.idx         = idx
        self.on_floor    = False
        self.trail       = []
        self.flash       = 0.0   # collision flash timer

    # ── Gauge-invariant energy ───────────────────────────────────────────────
    def KE(self):
        return 0.5 * self.m * (self.vx**2 + self.vy**2) + 0.5 * self.I * self.omega**2

    def PE(self):
        h = max(0.0, (FLOOR_Y - self.r) - self.y)
        return self.m * G_ACC * h

    def E_total(self):
        return self.KE() + self.PE()

    # ── Integration ─────────────────────────────────────────────────────────
    def step(self, dt, particles):
        # Gravity
        self.vy += G_ACC * dt

        # Integrate
        self.x     += self.vx    * dt
        self.y     += self.vy    * dt
        self.theta += self.omega * dt

        self.flash = max(0.0, self.flash - dt * 3)

        # ── Floor ──────────────────────────────────────────────────────────
        if self.y + self.r >= FLOOR_Y:
            self.y = FLOOR_Y - self.r
            if self.vy > 20:
                # Bounce particles
                for _ in range(int(min(12, self.r * 0.4))):
                    particles.append(Particle(self.x, FLOOR_Y, self.color))
            self.vy    = -abs(self.vy) * RESTITUTION
            self.on_floor = True

            if abs(self.vy) < 25:
                self.vy = 0.0

            # Rolling friction
            if abs(self.vx) > 2:
                sign   = math.copysign(1, self.vx)
                decel  = sign * 0.45 * G_ACC * dt
                if abs(decel) > abs(self.vx):
                    self.vx = 0.0
                else:
                    self.vx -= decel
            # Rolling constraint
            self.omega = self.vx / self.r
        else:
            self.on_floor = False

        # ── World walls ────────────────────────────────────────────────────
        if self.x - self.r < 0:
            self.x  = self.r
            self.vx =  abs(self.vx) * RESTITUTION
        if self.x + self.r > WORLD_W:
            self.x  = WORLD_W - self.r
            self.vx = -abs(self.vx) * RESTITUTION

        # ── Trail ──────────────────────────────────────────────────────────
        self.trail.append((self.x, self.y))
        if len(self.trail) > 30:
            self.trail.pop(0)

    # ── Draw ─────────────────────────────────────────────────────────────────
    def draw(self, surf, cam_x):
        sx = int(self.x - cam_x)
        sy = int(self.y)
        r  = int(self.r)

        if sx + r < -60 or sx - r > W + 60:
            return

        # Trail
        trail_len = len(self.trail)
        for i, (tx, ty) in enumerate(self.trail):
            frac = i / trail_len
            tr   = max(1, int(r * frac * 0.55))
            c    = tuple(int(v * frac * 0.7) for v in self.color)
            pygame.draw.circle(surf, c, (int(tx - cam_x), int(ty)), tr)

        # Flash ring on collision
        if self.flash > 0:
            ring_r = int(r + self.flash * 18)
            flash_c = tuple(min(255, int(v + 150 * self.flash)) for v in self.color)
            pygame.draw.circle(surf, flash_c, (sx, sy), ring_r, 2)

        # Body
        pygame.draw.circle(surf, self.color, (sx, sy), r)
        # Inner circle (disk has solid fill visual difference)
        if self.btype == 'disk':
            inner = max(2, int(r * 0.45))
            darker = tuple(max(0, v - 60) for v in self.color)
            pygame.draw.circle(surf, darker, (sx, sy), inner)

        # Spoke (rotation indicator)
        ex = sx + int(r * 0.9 * math.cos(self.theta))
        ey = sy + int(r * 0.9 * math.sin(self.theta))
        pygame.draw.line(surf, (255, 255, 255, 160), (sx, sy), (ex, ey), 2)

        # Type label
        # (skip for speed — spoke is the indicator)


# ── Indiana Jones ─────────────────────────────────────────────────────────────
class Indy:
    SPEED_BASE  = 200.0
    SPEED_PANIC = 620.0

    def __init__(self):
        self.x      = 200.0
        self.y      = float(FLOOR_Y)
        self.vx     = 0.0
        self.fear   = 0.0
        self.step_t = 0.0
        self.alive  = True
        self.dead_t = 0.0

    def update(self, bodies, dt):
        if not self.alive:
            self.dead_t += dt
            return

        # Find nearest approaching threat
        nearest      = 1e9
        for b in bodies:
            dx = b.x - self.x
            if dx < 0:
                continue
            dy = b.y - (self.y - 20)
            d  = math.sqrt(dx*dx + dy*dy)
            if d < nearest:
                nearest = d

        self.fear = max(0.0, min(1.0, 1.0 - nearest / 520.0))

        run_spd = self.SPEED_BASE + (self.SPEED_PANIC - self.SPEED_BASE) * self.fear
        self.vx = -run_spd * self.fear if self.fear > 0.05 else 0.0

        self.x += self.vx * dt
        self.x  = max(40.0, self.x)

        self.step_t += dt * (1.0 + self.fear * 4.0)

        # Squash if hit
        for b in bodies:
            dx = abs(b.x - self.x)
            dy = abs(b.y - self.y)
            if dx < b.r + 12 and dy < b.r + 30:
                self.alive = False
                break

    def draw(self, surf, cam_x):
        sx  = int(self.x - cam_x)
        gnd = FLOOR_Y

        if not self.alive:
            # Flattened star
            t = self.dead_t
            for angle in range(0, 360, 45):
                rad = math.radians(angle)
                r   = int(20 + 10 * math.sin(t * 8))
                ex  = sx + int(r * math.cos(rad))
                ey  = gnd - 5 + int(4 * math.sin(rad))
                pygame.draw.line(surf, INDY_SKIN, (sx, gnd - 5), (ex, ey), 2)
            return

        # Leg phase
        lp    = math.sin(self.step_t * 6.0) * 0.55
        panic = self.fear

        # Head
        pygame.draw.circle(surf, INDY_SKIN, (sx, gnd - 50), 9)

        # Hat brim
        pygame.draw.ellipse(surf, INDY_HAT,
                            (sx - 14, gnd - 61, 28, 8))
        # Hat crown
        pygame.draw.rect(surf, INDY_HAT,
                         (sx - 9, gnd - 75, 18, 16))

        # Body
        pygame.draw.line(surf, INDY_SHIRT,
                         (sx, gnd - 41), (sx, gnd - 20), 3)

        # Arms (flailing with fear)
        arm_swing = panic * math.sin(self.step_t * 7) * 20
        pygame.draw.line(surf, INDY_SKIN,
                         (sx, gnd - 38),
                         (sx - 14 + int(arm_swing), gnd - 28), 2)
        pygame.draw.line(surf, INDY_SKIN,
                         (sx, gnd - 38),
                         (sx + 14 - int(arm_swing), gnd - 28), 2)

        # Legs
        pygame.draw.line(surf, INDY_SHIRT,
                         (sx, gnd - 20),
                         (sx + int(14 * math.sin( lp)), gnd), 2)
        pygame.draw.line(surf, INDY_SHIRT,
                         (sx, gnd - 20),
                         (sx + int(14 * math.sin(-lp)), gnd), 2)

        # Whip when panic > 0.5
        if panic > 0.5:
            whip_pts = []
            for k in range(12):
                frac = k / 11
                wx   = sx + int(frac * 55 * (1 + panic))
                wy   = gnd - 35 + int(12 * math.sin(frac * math.pi + self.step_t * 8))
                whip_pts.append((wx, wy))
            if len(whip_pts) > 1:
                pygame.draw.lines(surf, (180, 120, 30), False, whip_pts, 2)


# ── Monte Carlo spawn wave ────────────────────────────────────────────────────
def mc_spawn(n, offset_x=0):
    bodies = []
    for i in range(n):
        r     = float(RNG.uniform(18, 58))
        x     = float(RNG.uniform(W * 0.7 + offset_x, W * 2.8 + offset_x))
        y     = float(RNG.uniform(FLOOR_Y - 380, FLOOR_Y - r - 5))
        vx    = float(RNG.uniform(-380, -60))
        vy    = float(RNG.uniform(-120, 100))
        btype = RNG.choice(['sphere', 'disk'])
        color = NEON[i % len(NEON)]
        bodies.append(Body(x, y, r, vx, vy, btype, color, i))
    return bodies


# ── Impulse collision resolver ────────────────────────────────────────────────
def resolve(a, b, particles):
    dx = b.x - a.x
    dy = b.y - a.y
    d  = math.sqrt(dx*dx + dy*dy)
    mn = a.r + b.r
    if d >= mn or d < 1e-6:
        return

    nx, ny = dx / d, dy / d

    # Separate (push out proportional to mass)
    overlap  = mn - d
    tot_m    = a.m + b.m
    a.x     -= nx * overlap * (b.m / tot_m)
    a.y     -= ny * overlap * (b.m / tot_m)
    b.x     += nx * overlap * (a.m / tot_m)
    b.y     += ny * overlap * (a.m / tot_m)

    # Relative velocity along normal
    dvx = a.vx - b.vx
    dvy = a.vy - b.vy
    vn  = dvx * nx + dvy * ny
    if vn >= 0:
        return   # already separating

    # Impulse magnitude (elastic, no rotation transfer for simplicity)
    j    = -(1.0 + RESTITUTION) * vn / (1.0/a.m + 1.0/b.m)
    a.vx += j / a.m * nx
    a.vy += j / a.m * ny
    b.vx -= j / b.m * nx
    b.vy -= j / b.m * ny

    a.flash = 1.0
    b.flash = 1.0

    # Particles at impact point
    mx = (a.x + b.x) * 0.5
    my = (a.y + b.y) * 0.5
    mix_c = tuple((a.color[k] + b.color[k]) // 2 for k in range(3))
    for _ in range(8):
        particles.append(Particle(mx, my, mix_c))


# ── HUD ────────────────────────────────────────────────────────────────────────
def draw_hud(surf, fonts, bodies, indy, cam_x, t, show_gauge):
    font_b, font_s = fonts
    E_vals   = [b.E_total() for b in bodies]
    KE_vals  = [b.KE()      for b in bodies]
    PE_vals  = [b.PE()      for b in bodies]
    E_tot    = sum(E_vals)
    KE_tot   = sum(KE_vals)
    PE_tot   = sum(PE_vals)

    lines = [
        ('INDIANA JONES  RIGID BODY ARENA', HUD_ORANGE, True),
        (f't = {t:.2f} s      bodies: {len(bodies)}', HUD_DIM, False),
        (f'Sphere I=2/5 mr²   Disk I=1/2 mr²', HUD_DIM, False),
        (f'E_total = {E_tot:7.0f}  |  KE={KE_tot:6.0f}  PE={PE_tot:6.0f}', HUD_GREEN, False),
        (f'Gauge inv: frame translation cam_x={int(cam_x)} does not change E', HUD_BLUE, False),
        (f'Indy  x={indy.x:.0f}  fear={indy.fear:.0%}  {"ALIVE" if indy.alive else "SQUASHED!"}',
         INDY_SKIN if indy.alive else (255, 60, 60), False),
    ]

    for i, (text, color, bold) in enumerate(lines):
        fn = font_b if bold else font_s
        surf.blit(fn.render(text, True, color), (14, 14 + i * 22))

    # KE/PE ratio bar
    bar_x, bar_y, bar_w, bar_h = 14, 152, 300, 13
    ke_w = int(bar_w * KE_tot / max(E_tot, 1))
    pygame.draw.rect(surf, (30, 30, 40),  (bar_x, bar_y, bar_w, bar_h))
    pygame.draw.rect(surf, HUD_ORANGE,    (bar_x, bar_y, ke_w,  bar_h))
    pygame.draw.rect(surf, HUD_BLUE,      (bar_x + ke_w, bar_y, bar_w - ke_w, bar_h))
    surf.blit(font_s.render('KE', True, BG), (bar_x + 4,        bar_y + 1))
    surf.blit(font_s.render('PE', True, BG), (bar_x + ke_w + 4, bar_y + 1))

    # Per-body energy dots (sparkline)
    dot_y = bar_y + bar_h + 8
    max_e  = max(E_vals) if E_vals else 1
    for i, (e, b) in enumerate(zip(E_vals, bodies)):
        dx = int(14 + i * (bar_w / len(bodies)))
        dh = int(40 * e / max_e)
        pygame.draw.rect(surf, b.color, (dx, dot_y + 40 - dh, max(2, int(bar_w/len(bodies))-1), dh))

    # Gauge frame overlay
    if show_gauge:
        _draw_gauge_overlay(surf, font_s, cam_x)

    # Controls
    ctrl_txt = font_s.render('R respawn  SPACE boulder  G gauge  ESC quit', True, (50, 50, 65))
    surf.blit(ctrl_txt, (14, H - 22))


def _draw_gauge_overlay(surf, font, cam_x):
    """Show that same body has same E in two reference frames."""
    ox, oy = W - 340, 14
    w, h   = 320, 120
    s      = pygame.Surface((w, h), pygame.SRCALPHA)
    s.fill((10, 15, 30, 200))
    surf.blit(s, (ox, oy))

    lines = [
        'GAUGE INVARIANCE',
        'E_mech = KE + PE  is frame-independent',
        'Frame A (cam):   same E as Frame B (world)',
        'Boost v->v+V:  KE changes BUT so does PE',
        'Total Hamiltonian H = p^2/2m + mgh  invariant',
        'under time translation (Noether: energy conserved)',
    ]
    colors = [HUD_ORANGE, HUD_DIM, HUD_BLUE, HUD_DIM, HUD_GREEN, HUD_GREEN]
    for i, (line, col) in enumerate(zip(lines, colors)):
        surf.blit(font.render(line, True, col), (ox + 8, oy + 8 + i * 18))


# ── BOULDER flash splash ───────────────────────────────────────────────────────
def boulder_splash(surf, font_b, indy, bodies, cam_x):
    for b in bodies:
        if not indy.alive:
            break
        dx = b.x - indy.x
        if 0 < dx < b.r * 2.5:
            sx = int(indy.x - cam_x)
            size = int(32 + 20 * (1 - dx / (b.r * 2.5)))
            txt  = font_b.render('BOULDER!', True,
                                 (255, int(80 + 175 * dx / (b.r * 2.5)), 0))
            surf.blit(txt, (sx - txt.get_width()//2, FLOOR_Y - 90))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    pygame.init()
    surf  = pygame.display.set_mode((W, H))
    pygame.display.set_caption('Indiana Jones Rigid Body Arena')
    clock = pygame.time.Clock()

    font_b = pygame.font.SysFont('consolas', 15, bold=True)
    font_s = pygame.font.SysFont('consolas', 13)
    fonts  = (font_b, font_s)

    bodies    = mc_spawn(N_SPAWN)
    indy      = Indy()
    particles = []
    cam_x     = 0.0
    t         = 0.0
    show_gauge = False

    running = True
    while running:
        dt = min(clock.tick(FPS) / 1000.0, 0.033)

        # ── Events ──────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    bodies    = mc_spawn(N_SPAWN)
                    indy      = Indy()
                    particles = []
                elif event.key == pygame.K_SPACE:
                    # Spawn one huge boulder right of Indy
                    bx  = indy.x + 600 + float(RNG.uniform(100, 300))
                    r   = float(RNG.uniform(50, 90))
                    b   = Body(bx, FLOOR_Y - r - 5, r,
                               float(RNG.uniform(-500, -200)), 0.0,
                               'sphere', (255, 200, 0), len(bodies))
                    bodies.append(b)
                elif event.key == pygame.K_g:
                    show_gauge = not show_gauge

        # ── Physics ─────────────────────────────────────────────────────────
        for b in bodies:
            b.step(dt, particles)

        # O(N^2) collision, N<=40 so fast enough
        for i in range(len(bodies)):
            for j in range(i + 1, len(bodies)):
                resolve(bodies[i], bodies[j], particles)

        # Particles
        for p in particles:
            p.step(dt)
        particles = [p for p in particles if p.alive()]

        # Indy AI
        indy.update(bodies, dt)

        # Respawn Indy after squash
        if not indy.alive and indy.dead_t > 2.5:
            indy = Indy()
            indy.x = max(40.0, cam_x + 160)

        # Smooth camera follows Indy (parallax target)
        target_cam = indy.x - W * 0.28
        cam_x     += (target_cam - cam_x) * 0.07

        t += dt

        # ── Draw ────────────────────────────────────────────────────────────
        surf.fill(BG)

        # Parallax background grid
        for gx in range(int(-cam_x * 0.2) % 150 - 150, W + 150, 150):
            pygame.draw.line(surf, (16, 16, 28), (gx, 0), (gx, H), 1)
        for gy in range(0, H, 100):
            pygame.draw.line(surf, (16, 16, 28), (0, gy), (W, gy), 1)

        # Distant pillars (parallax layer 2)
        for px in range(int(-cam_x * 0.4) % 280 - 280, W + 280, 280):
            pillar_h = RNG.integers(60, 180)
            pygame.draw.rect(surf, (25, 20, 14),
                             (px - 18, FLOOR_Y - pillar_h, 36, pillar_h))

        # Floor
        pygame.draw.rect(surf, FLOOR_COL, (0, FLOOR_Y, W, H - FLOOR_Y))
        pygame.draw.line(surf, FLOOR_EDGE, (0, FLOOR_Y), (W, FLOOR_Y), 3)
        # Stone tile pattern on floor
        for tx in range(int(-cam_x * 1.0) % 80 - 80, W + 80, 80):
            pygame.draw.line(surf, FLOOR_EDGE, (tx, FLOOR_Y), (tx, H), 1)

        # Bodies
        for b in bodies:
            b.draw(surf, cam_x)

        # Particles (above bodies)
        for p in particles:
            p.draw(surf, cam_x)

        # Indy
        indy.draw(surf, cam_x)

        # Boulder proximity flash
        boulder_splash(surf, font_b, indy, bodies, cam_x)

        # HUD
        draw_hud(surf, fonts, bodies, indy, cam_x, t, show_gauge)

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == '__main__':
    main()
