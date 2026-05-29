#!/usr/bin/env python3
"""
Indiana Jones  –  Circuit & Drivetrain
========================================
Rigid body balls/disks roll through a multi-ramp circuit with a loop-the-loop.
A live gear drivetrain in the corner spins off the nearest ball's omega.

Physics
-------
  Solid sphere   I = 2/5 m r²
  Solid disk     I = 1/2 m r²
  Rolling        v_tangential = ω r   (no-slip on any surface)
  Segment coll.  impulse normal + tangential friction
  Gear train     ω₂/ω₁ = r₁/r₂   τ₂/τ₁ = r₂/r₁

Controls
--------
  R          respawn MC wave
  SPACE      launch a single boulder
  LEFT/RIGHT apply drive force to all balls
  G          gear detail panel
  ESC        quit
"""

import pygame
import numpy as np
import math
import sys

# ── World ──────────────────────────────────────────────────────────────────────
W, H        = 1280, 720
FPS         = 60
WORLD_W     = 5200
G_ACC       = 540.0
RESTITUTION = 0.55
N_SPAWN     = 16
RNG         = np.random.default_rng(1492)

# ── Colours ────────────────────────────────────────────────────────────────────
BG          = (8, 10, 18)
TRACK_FILL  = (48, 38, 22)
TRACK_EDGE  = (120, 88, 38)
INDY_SKIN   = (230, 185, 110)
INDY_HAT    = (55, 30, 8)
HUD_GREEN   = (80, 255, 130)
HUD_BLUE    = (80, 180, 255)
HUD_ORANGE  = (255, 165, 40)
HUD_DIM     = (80, 80, 95)
GEAR_COL    = (160, 140, 70)
GEAR_TOOTH  = (200, 180, 90)

NEON = [
    (255,  50, 110), (50, 210, 255), (255, 210,   0),
    (170,  50, 255), (  0, 255, 160),(255, 100,  30),
    (200, 255,  50), ( 30, 160, 255),
]

# ── Track geometry ─────────────────────────────────────────────────────────────
# All coords in world space.  Segments list: (x1,y1,x2,y2)
# Normal convention: left-hand side of travel direction = surface normal.
FLOOR_Y  = H - 80
RAMP_H   = 210          # height of elevated plateau
LOOP_CX  = 2100         # loop centre-x
LOOP_CY  = FLOOR_Y - RAMP_H - 120   # loop centre-y
LOOP_R   = 120          # loop radius (inner edge)
N_LOOP   = 32           # polygon segments for loop

def build_track():
    segs = []   # each: ((x1,y1),(x2,y2))
    floor = FLOOR_Y

    # ── Section 1: starting flat ───────────────────────────────────────────
    segs.append(((0, floor), (700, floor)))

    # ── Section 2: ramp up ────────────────────────────────────────────────
    segs.append(((700, floor), (950, floor - RAMP_H)))

    # ── Section 3: plateau before loop ────────────────────────────────────
    segs.append(((950, floor - RAMP_H), (LOOP_CX - LOOP_R - 30, floor - RAMP_H)))

    # ── Section 4: loop-the-loop (inner polygon) ──────────────────────────
    # Polygon approximation of full circle inner surface
    angles = [2*math.pi * i / N_LOOP for i in range(N_LOOP + 1)]
    for i in range(N_LOOP):
        a1, a2 = angles[i], angles[i+1]
        x1 = LOOP_CX + (LOOP_R) * math.cos(a1)
        y1 = LOOP_CY + (LOOP_R) * math.sin(a1)
        x2 = LOOP_CX + (LOOP_R) * math.cos(a2)
        y2 = LOOP_CY + (LOOP_R) * math.sin(a2)
        segs.append(((x1, y1), (x2, y2)))

    # ── Section 5: plateau after loop ─────────────────────────────────────
    segs.append(((LOOP_CX + LOOP_R + 30, floor - RAMP_H),
                 (2800, floor - RAMP_H)))

    # ── Section 6: ramp down ──────────────────────────────────────────────
    segs.append(((2800, floor - RAMP_H), (3100, floor)))

    # ── Section 7: long flat + second small ramp ──────────────────────────
    segs.append(((3100, floor), (3700, floor)))
    segs.append(((3700, floor), (3900, floor - 100)))
    segs.append(((3900, floor - 100), (4200, floor - 100)))
    segs.append(((4200, floor - 100), (4400, floor)))
    segs.append(((4400, floor), (WORLD_W, floor)))

    # ── Left wall ─────────────────────────────────────────────────────────
    segs.append(((0, 0), (0, H + 200)))

    return segs


TRACK_SEGS = build_track()


def seg_normal_and_len(p1, p2):
    """Return (unit_tangent, unit_normal_pointing_up, length)."""
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    L = math.sqrt(dx*dx + dy*dy)
    if L < 1e-9:
        return (1,0),(0,-1),0
    tx, ty = dx/L, dy/L
    # Normal: rotate tangent 90° CCW → points "left" of travel = upward for floor
    nx, ny = -ty, tx
    # For loop inner surface we need inward normal — handled per-collision
    return (tx, ty), (nx, ny), L


# ── Particle ───────────────────────────────────────────────────────────────────
class Particle:
    __slots__ = ('x','y','vx','vy','life','max_life','color','r')
    def __init__(self, x, y, color):
        a = RNG.uniform(0, 2*math.pi)
        s = RNG.uniform(50, 250)
        self.x, self.y = float(x), float(y)
        self.vx = math.cos(a)*s; self.vy = math.sin(a)*s - 80
        self.life = self.max_life = RNG.uniform(0.2, 0.6)
        self.color = color
        self.r = int(RNG.integers(2, 5))

    def step(self, dt):
        self.vy += G_ACC*dt*0.25
        self.x  += self.vx*dt; self.y += self.vy*dt
        self.life -= dt

    def draw(self, surf, cam_x):
        a = max(0, self.life/self.max_life)
        c = tuple(int(v*a) for v in self.color)
        pygame.draw.circle(surf, c, (int(self.x-cam_x), int(self.y)), self.r)


# ── Rigid body ─────────────────────────────────────────────────────────────────
class Body:
    def __init__(self, x, y, r, vx, vy, btype, color):
        self.x, self.y = float(x), float(y)
        self.r = float(r)
        self.btype = btype
        self.m = 0.004 * r**2
        self.I_fac = 0.4 if btype == 'sphere' else 0.5
        self.I = self.I_fac * self.m * r**2
        self.vx, self.vy = float(vx), float(vy)
        self.omega = vx / max(r, 1e-6)
        self.theta = float(RNG.uniform(0, 2*math.pi))
        self.color = color
        self.flash = 0.0
        self.trail = []

    def KE(self):
        return 0.5*self.m*(self.vx**2+self.vy**2) + 0.5*self.I*self.omega**2

    def PE(self):
        h = max(0.0, (FLOOR_Y - self.r) - self.y)
        return self.m * G_ACC * h

    def step(self, dt, drive_fx, particles):
        # Drive force
        self.vx += drive_fx / self.m * dt

        # Gravity
        self.vy += G_ACC * dt

        # Integrate
        self.x     += self.vx    * dt
        self.y     += self.vy    * dt
        self.theta += self.omega * dt

        self.flash = max(0.0, self.flash - dt*3)

        # World walls
        if self.x - self.r < 0:
            self.x = self.r
            self.vx = abs(self.vx) * RESTITUTION
        if self.x + self.r > WORLD_W:
            self.x = WORLD_W - self.r
            self.vx = -abs(self.vx) * RESTITUTION
        if self.y - self.r < 0:
            self.y = self.r
            self.vy = abs(self.vy) * RESTITUTION

        # Trail
        self.trail.append((self.x, self.y))
        if len(self.trail) > 28:
            self.trail.pop(0)

    def draw(self, surf, cam_x):
        sx, sy = int(self.x - cam_x), int(self.y)
        r = int(self.r)
        if sx+r < -60 or sx-r > W+60:
            return

        # Trail
        tl = len(self.trail)
        for i,(tx,ty) in enumerate(self.trail):
            frac = i/tl
            tr = max(1, int(r*frac*0.5))
            c  = tuple(int(v*frac*0.6) for v in self.color)
            pygame.draw.circle(surf, c, (int(tx-cam_x), int(ty)), tr)

        if self.flash > 0:
            fc = tuple(min(255, int(v+140*self.flash)) for v in self.color)
            pygame.draw.circle(surf, fc, (sx,sy), int(r+self.flash*16), 2)

        pygame.draw.circle(surf, self.color, (sx, sy), r)
        if self.btype == 'disk':
            dk = tuple(max(0,v-55) for v in self.color)
            pygame.draw.circle(surf, dk, (sx,sy), max(2,int(r*0.42)))

        ex = sx + int(r*0.88*math.cos(self.theta))
        ey = sy + int(r*0.88*math.sin(self.theta))
        pygame.draw.line(surf, (255,255,255), (sx,sy), (ex,ey), 2)


# ── Segment collision ─────────────────────────────────────────────────────────
def collide_segment(b, p1, p2, particles, is_loop=False):
    dx_s = p2[0]-p1[0]; dy_s = p2[1]-p1[1]
    L = math.sqrt(dx_s**2 + dy_s**2)
    if L < 1e-9:
        return False

    tx, ty = dx_s/L, dy_s/L
    # Normal: CCW rotate tangent (points "left" of p1→p2 = inward for loop, upward for floor)
    nx, ny = -ty, tx

    # For loop segments, normal should point inward (toward centre)
    if is_loop:
        # Centre of loop
        mx = (p1[0]+p2[0])/2 - LOOP_CX
        my = (p1[1]+p2[1])/2 - LOOP_CY
        # If normal points away from centre, flip it
        if mx*nx + my*ny > 0:
            nx, ny = -nx, -ny

    # Vector from p1 to ball
    ex = b.x - p1[0]; ey = b.y - p1[1]

    # Projection onto segment (clamped)
    t = max(0.0, min(L, ex*tx + ey*ty))
    # Closest point
    cx = p1[0] + t*tx; cy = p1[1] + t*ty

    # Distance
    ddx = b.x - cx; ddy = b.y - cy
    dist = math.sqrt(ddx**2 + ddy**2)

    if dist >= b.r or dist < 1e-9:
        return False

    # Penetration direction
    pnx = ddx/dist; pny = ddy/dist

    # Check the normal is roughly aligned (within 90° of surface normal)
    if pnx*nx + pny*ny < -0.1:
        return False

    # Push out
    pen = b.r - dist
    b.x += pnx*pen; b.y += pny*pen

    # Velocity along penetration normal
    vn = b.vx*pnx + b.vy*pny
    if vn >= 0:
        return False   # separating

    # Tangent velocity
    tang_x = -pny; tang_y = pnx
    vt = b.vx*tang_x + b.vy*tang_y

    # Reflect normal component
    b.vx -= (1+RESTITUTION)*vn*pnx
    b.vy -= (1+RESTITUTION)*vn*pny

    # Rolling: omega from tangential velocity
    b.omega = vt / b.r

    # Particles on hard hits
    if abs(vn) > 120:
        b.flash = 1.0
        for _ in range(5):
            particles.append(Particle(cx, cy, b.color))

    return True


def collide_all_segments(b, particles):
    for seg in TRACK_SEGS:
        p1, p2 = seg
        is_loop = False
        # Detect if this segment is part of the loop polygon
        # (quick check: both endpoints within loop_r+50 of loop centre)
        d1 = math.sqrt((p1[0]-LOOP_CX)**2+(p1[1]-LOOP_CY)**2)
        d2 = math.sqrt((p2[0]-LOOP_CX)**2+(p2[1]-LOOP_CY)**2)
        if d1 < LOOP_R+25 and d2 < LOOP_R+25:
            is_loop = True
        collide_segment(b, p1, p2, particles, is_loop)


# ── Ball–ball impulse collision ────────────────────────────────────────────────
def resolve_bodies(a, b, particles):
    dx = b.x-a.x; dy = b.y-a.y
    d  = math.sqrt(dx*dx+dy*dy)
    mn = a.r+b.r
    if d >= mn or d < 1e-9:
        return
    nx, ny = dx/d, dy/d
    ov = mn-d; tm = a.m+b.m
    a.x -= nx*ov*(b.m/tm); a.y -= ny*ov*(b.m/tm)
    b.x += nx*ov*(a.m/tm); b.y += ny*ov*(a.m/tm)
    dv = (a.vx-b.vx)*nx + (a.vy-b.vy)*ny
    if dv >= 0:
        return
    j = -(1+RESTITUTION)*dv/(1/a.m+1/b.m)
    a.vx += j/a.m*nx; a.vy += j/a.m*ny
    b.vx -= j/b.m*nx; b.vy -= j/b.m*ny
    a.flash = b.flash = 1.0
    mc = tuple((a.color[k]+b.color[k])//2 for k in range(3))
    mx = (a.x+b.x)/2; my = (a.y+b.y)/2
    for _ in range(7):
        particles.append(Particle(mx, my, mc))


# ── Gear drivetrain ────────────────────────────────────────────────────────────
class GearTrain:
    """
    Three-gear drivetrain:  driver  →  idler  →  output
    Gear ratios:  ω₂ = ω₁ * r₁/r₂   τ₂ = τ₁ * r₂/r₁
    Driven by omega of the nearest ball to Indy.
    """
    def __init__(self):
        # (centre_x, centre_y, radius, teeth) in HUD-local coords
        self.gears = [
            (0,    0,  36, 18),   # driver (input from ball)
            (82,   0,  24, 12),   # idler
            (82+58,0,  18,  9),   # output pinion
        ]
        self.angles  = [0.0, 0.0, 0.0]
        self.omega_in = 0.0   # rad/s of driver gear

    def update(self, omega_ball, r_ball, dt):
        # Scale ball omega to driver gear (conserve surface speed)
        self.omega_in = omega_ball * r_ball / self.gears[0][2]
        # Cascade: ω_next = -ω_prev * r_prev / r_next  (negative = meshed reversal)
        omegas = [self.omega_in]
        for i in range(1, len(self.gears)):
            r_prev = self.gears[i-1][2]
            r_curr = self.gears[i  ][2]
            omegas.append(-omegas[-1] * r_prev / r_curr)
        for i, om in enumerate(omegas):
            self.angles[i] += om * dt

    def draw(self, surf, ox, oy, font, show_detail):
        """Draw at HUD offset (ox, oy)."""
        r0 = self.gears[0][2]
        omegas = [self.omega_in]
        for i in range(1, len(self.gears)):
            rp = self.gears[i-1][2]; rc = self.gears[i][2]
            omegas.append(-omegas[-1]*rp/rc)

        for i, (gx, gy, gr, teeth) in enumerate(self.gears):
            cx, cy = ox+gx, oy+gy
            angle  = self.angles[i]

            # Gear body
            pygame.draw.circle(surf, GEAR_COL, (cx, cy), gr)
            pygame.draw.circle(surf, (30,25,15), (cx, cy), max(4, gr//3))

            # Teeth
            tooth_h = gr * 0.28
            for t in range(teeth):
                ta = angle + 2*math.pi*t/teeth
                inner_r = gr - 3
                outer_r = gr + tooth_h
                # Trapezoidal tooth
                half_w = math.pi / teeth * 0.4
                pts = [
                    (cx + inner_r*math.cos(ta-half_w*1.4),
                     cy + inner_r*math.sin(ta-half_w*1.4)),
                    (cx + outer_r*math.cos(ta-half_w),
                     cy + outer_r*math.sin(ta-half_w)),
                    (cx + outer_r*math.cos(ta+half_w),
                     cy + outer_r*math.sin(ta+half_w)),
                    (cx + inner_r*math.cos(ta+half_w*1.4),
                     cy + inner_r*math.sin(ta+half_w*1.4)),
                ]
                pygame.draw.polygon(surf, GEAR_TOOTH, pts)

            # Spoke
            ex = cx + int(gr*0.7*math.cos(angle))
            ey = cy + int(gr*0.7*math.sin(angle))
            pygame.draw.line(surf, (220,190,80), (cx,cy), (ex,ey), 2)

            # Mesh line between adjacent gears
            if i < len(self.gears)-1:
                gx2 = self.gears[i+1][0]; gy2 = self.gears[i+1][1]
                pygame.draw.line(surf, (60,50,30),
                                 (ox+gx, oy+gy), (ox+gx2, oy+gy2), 1)

            if show_detail:
                om = omegas[i]
                label = font.render(f'w={om:.1f}', True, HUD_ORANGE)
                surf.blit(label, (cx - label.get_width()//2, cy + gr + 5))

        # Drivetrain ratios
        if show_detail:
            ratio_1_3 = (self.gears[0][2] / self.gears[2][2])
            lines = [
                f'DRIVETRAIN',
                f'r1={self.gears[0][2]} r2={self.gears[1][2]} r3={self.gears[2][2]}',
                f'w_out = w_in x {ratio_1_3:.2f}  (gear-up)',
                f't_out = t_in / {ratio_1_3:.2f}  (torque-dn)',
            ]
            for k, line in enumerate(lines):
                col = HUD_ORANGE if k==0 else HUD_DIM
                surf.blit(font.render(line, True, col), (ox-10, oy+70+k*17))


# ── Indiana Jones ──────────────────────────────────────────────────────────────
class Indy:
    def __init__(self):
        self.x     = 300.0
        self.y     = float(FLOOR_Y)
        self.vx    = 0.0
        self.vy    = 0.0
        self.fear  = 0.0
        self.step_t= 0.0
        self.alive = True
        self.dead_t= 0.0
        self.on_ground = True

    def _floor_y_at(self, x):
        """Find floor Y at given x using track segments (simplified)."""
        # Binary bracket: check all non-loop flat/ramp segments
        best_y = FLOOR_Y
        for (p1, p2) in TRACK_SEGS:
            x1,y1 = p1; x2,y2 = p2
            if x1 == x2:
                continue
            if min(x1,x2) <= x <= max(x1,x2):
                t = (x - x1)/(x2-x1)
                y = y1 + t*(y2-y1)
                # Only consider surface below Indy (floor-like)
                if FLOOR_Y - RAMP_H - 30 < y <= FLOOR_Y + 5:
                    if y < best_y:
                        best_y = y
        return best_y

    def update(self, bodies, dt):
        if not self.alive:
            self.dead_t += dt
            return

        nearest = 1e9
        for b in bodies:
            dx = b.x - self.x
            if dx > 0:
                dy = b.y - (self.y - 20)
                dist = math.sqrt(dx*dx+dy*dy)
                if dist < nearest:
                    nearest = dist

        self.fear = max(0.0, min(1.0, 1.0 - nearest/550.0))
        run_spd = 180 + 520*self.fear
        self.vx = -run_spd * max(self.fear, 0.02)

        self.x += self.vx * dt
        self.x  = max(60.0, self.x)

        # Snap to track surface
        ground_y = self._floor_y_at(self.x)
        self.y = ground_y

        self.step_t += dt * (1.0 + self.fear*4.0)

        # Check squash
        for b in bodies:
            if abs(b.x-self.x)<b.r+14 and abs(b.y-self.y)<b.r+32:
                self.alive = False
                break

    def draw(self, surf, cam_x):
        sx  = int(self.x - cam_x)
        gnd = int(self.y)

        if not self.alive:
            t = self.dead_t
            for ang in range(0, 360, 45):
                rad = math.radians(ang)
                rv  = int(18 + 8*math.sin(t*9))
                pygame.draw.line(surf, INDY_SKIN,
                                 (sx, gnd-4),
                                 (sx+int(rv*math.cos(rad)), gnd-4+int(4*math.sin(rad))), 2)
            return

        lp = math.sin(self.step_t*6.0)*0.55

        pygame.draw.circle(surf, INDY_SKIN, (sx, gnd-50), 9)

        # Hat
        pygame.draw.ellipse(surf, INDY_HAT, (sx-14, gnd-61, 28, 8))
        pygame.draw.rect(surf, INDY_HAT, (sx-9, gnd-75, 18, 16))

        # Body
        pygame.draw.line(surf, (160,110,50), (sx,gnd-41),(sx,gnd-20),3)

        # Arms
        sw = self.fear*math.sin(self.step_t*7)*22
        pygame.draw.line(surf, INDY_SKIN,(sx,gnd-38),(sx-14+int(sw),gnd-28),2)
        pygame.draw.line(surf, INDY_SKIN,(sx,gnd-38),(sx+14-int(sw),gnd-28),2)

        # Legs
        pygame.draw.line(surf, (160,110,50),(sx,gnd-20),
                         (sx+int(14*math.sin( lp)),gnd),2)
        pygame.draw.line(surf, (160,110,50),(sx,gnd-20),
                         (sx+int(14*math.sin(-lp)),gnd),2)

        # Whip
        if self.fear > 0.45:
            wpts = [(sx+int(k*6*(1+self.fear)),
                     gnd-34+int(14*math.sin(k*0.6+self.step_t*8)))
                    for k in range(10)]
            if len(wpts) > 1:
                pygame.draw.lines(surf, (160,110,30), False, wpts, 2)


# ── MC Spawn ───────────────────────────────────────────────────────────────────
def mc_spawn(n):
    bodies = []
    for i in range(n):
        r     = float(RNG.uniform(16, 50))
        x     = float(RNG.uniform(500, 2000))
        y     = float(RNG.uniform(FLOOR_Y - 400, FLOOR_Y - r - 10))
        vx    = float(RNG.uniform(-350, -50))
        vy    = float(RNG.uniform(-100, 80))
        btype = RNG.choice(['sphere', 'disk'])
        color = NEON[i % len(NEON)]
        bodies.append(Body(x, y, r, vx, vy, btype, color))
    return bodies


# ── Track draw ────────────────────────────────────────────────────────────────
# Explicit floor / plateau rects — never use segment y-equality check
# (loop polygon segments are near-horizontal near 0° / 180° and would
#  stamp dark rectangles across the whole screen)
_FLOOR_RECTS = [
    # (world_x1, world_x2, top_y)  — filled from top_y down to H
    (0,          700,    FLOOR_Y),
    (950,        LOOP_CX - LOOP_R - 30, FLOOR_Y - RAMP_H),
    (LOOP_CX + LOOP_R + 30, 2800,       FLOOR_Y - RAMP_H),
    (3100,       3700,   FLOOR_Y),
    (3900,       4200,   FLOOR_Y - 100),
    (4400,       WORLD_W, FLOOR_Y),
]

def draw_track(surf, cam_x):
    # ── Filled ground under each flat / elevated section ──────────────────
    for (wx1, wx2, top_y) in _FLOOR_RECTS:
        sx1 = int(wx1 - cam_x)
        sx2 = int(wx2 - cam_x)
        if sx2 < 0 or sx1 > W:
            continue
        pygame.draw.rect(surf, TRACK_FILL,
                         (sx1, int(top_y), sx2 - sx1, H - int(top_y) + 10))

    # ── Track edge lines (all segments, culled off-screen) ────────────────
    for seg in TRACK_SEGS:
        p1, p2 = seg
        sx1 = int(p1[0]-cam_x); sy1 = int(p1[1])
        sx2 = int(p2[0]-cam_x); sy2 = int(p2[1])
        if max(sx1, sx2) < -20 or min(sx1, sx2) > W + 20:
            continue
        pygame.draw.line(surf, TRACK_EDGE, (sx1, sy1), (sx2, sy2), 5)

    # ── Loop highlight ────────────────────────────────────────────────────
    lx = int(LOOP_CX - cam_x); ly = int(LOOP_CY)
    if -LOOP_R - 30 < lx < W + LOOP_R + 30:
        pygame.draw.circle(surf, (90, 68, 28), (lx, ly), LOOP_R + 18, 4)
        pygame.draw.circle(surf, (50, 38, 15), (lx, ly), LOOP_R - 8,  1)


# ── HUD ───────────────────────────────────────────────────────────────────────
def draw_hud(surf, fonts, bodies, indy, cam_x, t, gear, show_gear):
    fb, fs = fonts
    E   = sum(b.KE()+b.PE() for b in bodies)
    KE  = sum(b.KE() for b in bodies)
    PE  = sum(b.PE() for b in bodies)

    lines = [
        ('INDY  CIRCUIT  &  DRIVETRAIN', HUD_ORANGE, True),
        (f't={t:.1f}s  bodies={len(bodies)}', HUD_DIM, False),
        (f'I_sphere=2/5mr²  I_disk=1/2mr²', HUD_DIM, False),
        (f'E={E:.0f}  KE={KE:.0f}  PE={PE:.0f}', HUD_GREEN, False),
        (f'Indy: x={indy.x:.0f}  fear={indy.fear:.0%}  '
         f'{"ALIVE" if indy.alive else "FLAT!"}',
         INDY_SKIN if indy.alive else (255,60,60), False),
    ]
    for i,(txt,col,bold) in enumerate(lines):
        fn = fb if bold else fs
        surf.blit(fn.render(txt,True,col),(14,14+i*22))

    # KE/PE bar
    bw = 280
    ke_w = int(bw*KE/max(E,1))
    pygame.draw.rect(surf,(25,25,35),(14,128,bw,12))
    pygame.draw.rect(surf,HUD_ORANGE,(14,128,ke_w,12))
    pygame.draw.rect(surf,HUD_BLUE, (14+ke_w,128,bw-ke_w,12))

    # Gear train box
    gx, gy = W-260, 14
    box = pygame.Surface((250,160), pygame.SRCALPHA)
    box.fill((10,14,28,210))
    surf.blit(box,(gx,gy))

    # Find nearest ball omega
    nearest_om = 0.0; nearest_r = 30.0
    if bodies:
        bodies_sorted = sorted(bodies, key=lambda b: abs(b.x - indy.x))
        nearest_om = bodies_sorted[0].omega
        nearest_r  = bodies_sorted[0].r

    gear.update(nearest_om, nearest_r, 1/FPS)
    gear.draw(surf, gx+50, gy+55, fs, show_gear)

    # Controls
    ctrl = fs.render('R respawn  SPACE boulder  ←→ drive  G gear  ESC', True,(45,45,60))
    surf.blit(ctrl,(14,H-22))

    # BOULDER warning
    for b in bodies:
        if 0 < b.x - indy.x < b.r*2.8:
            sx = int(indy.x - cam_x)
            frac = (b.x - indy.x) / max(b.r * 2.8, 1.0)   # 0=touching → 1=threshold
            green = min(255, max(0, int(frac * 210)))
            txt = fb.render('BOULDER!', True, (255, green, 0))
            surf.blit(txt,(sx-txt.get_width()//2, FLOOR_Y-100))
            break


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    pygame.init()
    surf  = pygame.display.set_mode((W, H))
    pygame.display.set_caption('Indy Circuit & Drivetrain')
    clock = pygame.time.Clock()
    fb = pygame.font.SysFont('consolas', 15, bold=True)
    fs = pygame.font.SysFont('consolas', 13)

    bodies    = mc_spawn(N_SPAWN)
    indy      = Indy()
    particles = []
    gear      = GearTrain()
    cam_x     = 0.0
    t         = 0.0
    show_gear = False
    drive_fx  = 0.0

    # Pre-compute background pillar positions (static, not re-rolled every frame)
    PILLARS = [(int(i * 290), int(RNG.integers(55, 180)))
               for i in range(WORLD_W // 290 + 3)]

    running = True
    while running:
        dt = min(clock.tick(FPS)/1000.0, 0.033)

        # ── Events ──────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                elif event.key == pygame.K_r:
                    bodies = mc_spawn(N_SPAWN); indy = Indy(); particles = []
                elif event.key == pygame.K_SPACE:
                    r_ = float(RNG.uniform(45,80))
                    b_ = Body(indy.x+550+float(RNG.uniform(50,200)),
                              FLOOR_Y-r_-5, r_,
                              float(RNG.uniform(-450,-180)), 0.0,
                              'sphere', (255,200,40))
                    bodies.append(b_)
                elif event.key == pygame.K_g:
                    show_gear = not show_gear

        # Drive force from held keys
        keys = pygame.key.get_pressed()
        drive_fx = 0.0
        if keys[pygame.K_RIGHT]: drive_fx =  380.0
        if keys[pygame.K_LEFT]:  drive_fx = -380.0

        # ── Physics ─────────────────────────────────────────────────────────
        for b in bodies:
            b.step(dt, drive_fx, particles)
            collide_all_segments(b, particles)

        for i in range(len(bodies)):
            for j in range(i+1,len(bodies)):
                resolve_bodies(bodies[i],bodies[j],particles)

        for p in particles: p.step(dt)
        particles = [p for p in particles if p.life > 0]

        indy.update(bodies, dt)
        if not indy.alive and indy.dead_t > 2.5:
            indy = Indy(); indy.x = max(80.0, cam_x+180)

        # Camera
        target_cam = indy.x - W*0.3
        cam_x += (target_cam - cam_x)*0.075

        t += dt

        # ── Draw ────────────────────────────────────────────────────────────
        surf.fill(BG)

        # BG grid (parallax)
        for gx_ in range(int(-cam_x*0.2)%160-160, W+160, 160):
            pygame.draw.line(surf,(14,14,26),(gx_,0),(gx_,H),1)
        for gy_ in range(0,H,100):
            pygame.draw.line(surf,(14,14,26),(0,gy_),(W,gy_),1)

        # Distant silhouette pillars (parallax, pre-computed heights)
        for (wx, ph) in PILLARS:
            sx = int(wx - cam_x * 0.35)
            if -40 < sx < W + 40:
                pygame.draw.rect(surf, (22, 17, 10), (sx-20, FLOOR_Y-ph, 40, ph))

        draw_track(surf, cam_x)

        for b in bodies: b.draw(surf, cam_x)
        for p in particles: p.draw(surf, cam_x)
        indy.draw(surf, cam_x)
        draw_hud(surf,(fb,fs),bodies,indy,cam_x,t,gear,show_gear)

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == '__main__':
    main()
