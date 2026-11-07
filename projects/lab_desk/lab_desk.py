"""
lab_desk.py - a desk-style lab simulator for dispersion-assisted phase recovery.

You sit at an optical bench (laser -> gas cell -> splitter -> two dispersive fibers -> two
photodiodes) with an oscilloscope and a laptop. Your hand follows the mouse with a little lag and
tremor, Surgeon Simulator style. Pick up the scope probes and drop them on test points to see the
light at each stage. With CH1 and CH2 on the two photodiodes, the laptop can run time-domain
Gerchberg-Saxton (TDGSA) on what the scope captured, and hand the same capture to MATLAB or Jupyter.

Things worth trying: turn the fiber spools to the 2024 settings (353 / 872 ps/nm) and watch the
recovery fail; drop the scope bandwidth to 8 GHz; switch the gas cell to the non-causal model.

Controls
  mouse: move hand   left button: grab / drop a probe, press buttons and knobs
  click a spool: more dispersion (right-click: less)   click the gas cell: causal <-> non-causal
  scope knobs: timebase, bandwidth, noise, THz stretch   T: steady hand   Space: run TDGSA
  V: export the desk for Vision Pro (USDZ) and Windows VR (GLB)   Esc: quit
Run:  python projects/lab_desk/lab_desk.py         (headless check: --shot out.png)
"""
import math
import os
import shutil
import subprocess
import sys

import numpy as np

SHOT = "--shot" in sys.argv
if SHOT:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import desk_physics as dp  # noqa: E402

W, H = 1280, 800
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
TIMEBASE = [2e-12, 5e-12, 20e-12, 100e-12, 500e-12]    # s/div
OKABE = {"orange": (230, 159, 0), "sky": (86, 180, 233), "green": (0, 158, 115), "yellow": (240, 228, 66),
         "blue": (0, 114, 178), "vermillion": (213, 94, 0), "purple": (204, 121, 167)}


def catenary_pts(p, q, slack=1.08, n=24):
    """Points of a hanging cable between p and q (screen coords, y down)."""
    (x1, y1), (x2, y2) = p, q
    L = math.hypot(x2 - x1, y2 - y1)
    if L < 2:
        return [p, q]
    S = slack * L
    a = L / (2 * math.sqrt(6 * (S / L - 1)))
    for _ in range(25):
        g = 2 * a * math.sinh(L / (2 * a)) - S
        dg = 2 * math.sinh(L / (2 * a)) - (L / a) * math.cosh(L / (2 * a))
        a -= g / dg
    out = []
    for k in range(n + 1):
        x = L * k / n
        sag = a * math.cosh((x - L / 2) / a) - a * math.cosh(L / (2 * a))   # <= 0
        out.append((x1 + (x2 - x1) * k / n, y1 + (y2 - y1) * k / n - sag))
    return out


def box3d(surf, rect, depth, col, label=None, font=None):
    """A component as a little 3D box: top face, front face, side face."""
    x, y, w, h = rect
    top = [(x, y), (x + w, y), (x + w + depth, y - depth * 0.6), (x + depth, y - depth * 0.6)]
    side = [(x + w, y), (x + w + depth, y - depth * 0.6), (x + w + depth, y + h - depth * 0.6), (x + w, y + h)]
    pygame.draw.polygon(surf, tuple(min(255, int(c * 1.18)) for c in col), top)
    pygame.draw.polygon(surf, tuple(int(c * 0.7) for c in col), side)
    pygame.draw.rect(surf, col, rect)
    pygame.draw.rect(surf, tuple(int(c * 0.5) for c in col), rect, 1)
    if label and font:
        t = font.render(label, True, (20, 22, 28))
        surf.blit(t, t.get_rect(center=(x + w / 2, y + h / 2)))


class Probe:
    def __init__(self, name, color, anchor, rest):
        self.name, self.color, self.anchor = name, color, anchor
        self.pos = list(rest)
        self.at = None            # test-point name it is clipped to


class Desk:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("Lab desk - dispersion-assisted phase recovery")
        self.fnt = pygame.font.SysFont("segoeui", 15)
        self.fnt_b = pygame.font.SysFont("segoeui", 16, bold=True)
        self.fnt_s = pygame.font.SysFont("segoeui", 12)
        self.fnt_big = pygame.font.SysFont("segoeui", 22, bold=True)
        self.mono = pygame.font.SysFont("consolas", 13)
        self.clock = pygame.time.Clock()
        self.bench = dp.Bench()
        self.tb = 2
        self.bw_i = 4
        self.noise_i = 0
        self.stretch_i = 0
        self.view = "tdgsa"          # laptop tab: "tdgsa" or "topo"
        self._topo = None
        self.hand = [W / 2, H / 2]
        self.hand_v = [0.0, 0.0]
        self.steady = False
        self.held = None
        self.t = 0.0
        self.run = None
        self.iter_target = 0
        self.status = "Pick up a probe (yellow = CH1, cyan = CH2) and clip it onto a gold test point."
        # bench layout (screen coordinates)
        self.parts = {
            "laser": pygame.Rect(70, 300, 150, 70), "cell": pygame.Rect(290, 312, 150, 46),
            "split": pygame.Rect(505, 318, 50, 34), "spool1": pygame.Rect(610, 205, 120, 80),
            "spool2": pygame.Rect(610, 395, 120, 80), "pd1": pygame.Rect(790, 222, 70, 50),
            "pd2": pygame.Rect(790, 412, 70, 50), "scope": pygame.Rect(920, 90, 340, 300),
            "laptop": pygame.Rect(60, 470, 560, 300),
        }
        self.tps = {"laser": (220, 335), "cell": (440, 335), "pd1": (860, 247), "pd2": (860, 437)}
        s = self.parts["scope"]
        self.probes = [Probe("CH1", OKABE["yellow"], (s.x + 60, s.bottom), (980, 470)),
                       Probe("CH2", OKABE["sky"], (s.x + 140, s.bottom), (1060, 480))]
        self.buttons = []
        self._traces = {}

    # ------------------------------------------------ simulation hooks
    def invalidate(self):
        self._traces = {}
        self.run = None
        self._topo = None

    def trace(self, tp):
        if tp not in self._traces:
            self._traces[tp] = self.bench.test_point(tp)
        return self._traces[tp]

    def start_tdgsa(self):
        at = {p.at for p in self.probes}
        if at != {"pd1", "pd2"}:
            self.status = "TDGSA needs both photodiode traces: clip CH1 and CH2 onto PD1 and PD2 first."
            return
        self.run = dp.TDGSARun(self.bench)
        self.iter_target = 200
        self.status = "Running TDGSA on the two captured traces..."

    def open_matlab(self):
        dp.write_capture(self.bench, OUT)
        exe = shutil.which("matlab")
        if not exe:
            for root in (r"C:\Program Files\MATLAB",):
                if os.path.isdir(root):
                    for ver in sorted(os.listdir(root), reverse=True):
                        cand = os.path.join(root, ver, "bin", "matlab.exe")
                        if os.path.exists(cand):
                            exe = cand
                            break
        if exe and not SHOT:
            subprocess.Popen([exe, "-sd", OUT, "-r", "desk_tdgsa"])
            self.status = "MATLAB is starting with out/desk_tdgsa.m and the captured traces (desk_capture.mat)."
        else:
            self.status = f"Wrote out/desk_tdgsa.m + desk_capture.mat ({'MATLAB not found' if not exe else 'not launched'})."

    def open_jupyter(self):
        dp.write_capture(self.bench, OUT)
        nb = os.path.join(OUT, "desk_tdgsa.ipynb")
        if SHOT:
            self.status = "Wrote out/desk_tdgsa.ipynb"
            return
        try:
            import jupyterlab  # noqa: F401
            subprocess.Popen([sys.executable, "-m", "jupyterlab", nb], cwd=OUT)
            self.status = "JupyterLab is opening out/desk_tdgsa.ipynb (use the py312 kernel)."
        except ImportError:
            os.startfile(nb)
            self.status = "Opened out/desk_tdgsa.ipynb with the default app (use the py312 kernel)."

    # ------------------------------------------------ input
    def hand_tip(self):
        return (self.hand[0], self.hand[1])

    def handle(self, ev):
        if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
            return False
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_t:
                self.steady = not self.steady
            elif ev.key == pygame.K_SPACE:
                self.start_tdgsa()
            elif ev.key == pygame.K_v:
                self.press("xr")
        if ev.type == pygame.MOUSEBUTTONDOWN:
            hx, hy = self.hand_tip()
            if ev.button == 1:
                for rect, key in self.buttons:
                    if rect.collidepoint(hx, hy):
                        self.press(key)
                        return True
                if self.held is None:
                    for p in self.probes:
                        if math.hypot(p.pos[0] - hx, p.pos[1] - hy) < 26:
                            self.held, p.at = p, None
                            self.status = f"Holding {p.name}. Drop it on a gold test point."
                            return True
                    self.click_part(hx, hy, +1)
                else:
                    self.drop()
            elif ev.button == 3:
                self.click_part(hx, hy, -1)
            elif ev.button == 4:
                self.click_part(hx, hy, +1)
            elif ev.button == 5:
                self.click_part(hx, hy, -1)
        return True

    def drop(self):
        p = self.held
        self.held = None
        best = min(self.tps.items(), key=lambda kv: math.hypot(kv[1][0] - p.pos[0], kv[1][1] - p.pos[1]))
        if math.hypot(best[1][0] - p.pos[0], best[1][1] - p.pos[1]) < 34:
            p.pos = list(best[1])
            p.at = best[0]
            self.status = f"{p.name} clipped onto {best[0].upper()}."
        else:
            p.pos[1] = max(p.pos[1], 480)
            self.status = f"{p.name} dropped on the desk."

    def click_part(self, x, y, step):
        b = self.bench
        for name in ("spool1", "spool2"):
            if self.parts[name].collidepoint(x, y):
                attr = "D1" if name == "spool1" else "D2"
                cur = getattr(b, attr)
                i = min(range(len(dp.D_STEPS)), key=lambda k: abs(dp.D_STEPS[k] - cur))
                setattr(b, attr, dp.D_STEPS[max(0, min(len(dp.D_STEPS) - 1, i + step))])
                if b.D1 == b.D2:
                    setattr(b, attr, dp.D_STEPS[max(0, min(len(dp.D_STEPS) - 1, i + 2 * step))])
                self.invalidate()
                self.status = f"{attr} = {getattr(b, attr):g} ps/nm  ({b.regime(getattr(b, attr))})"
                return
        if self.parts["cell"].collidepoint(x, y):
            b.causal = not b.causal
            self.invalidate()
            self.status = ("Gas cell: causal lines (absorption and phase are Kramers-Kronig partners)" if b.causal
                           else "Gas cell: non-causal model (absorption only, no matching phase) - the 2024 suspect")

    def press(self, key):
        b = self.bench
        if key == "tb":
            self.tb = (self.tb + 1) % len(TIMEBASE)
        elif key == "bw":
            self.bw_i = (self.bw_i + 1) % len(dp.SCOPE_BW)
            b.bw = dp.SCOPE_BW[self.bw_i][1]
            self.invalidate()
            self.status = f"Scope + photodiode bandwidth: {dp.SCOPE_BW[self.bw_i][0]}"
        elif key == "noise":
            self.noise_i = (self.noise_i + 1) % len(dp.NOISE)
            b.noise = dp.NOISE[self.noise_i]
            self.invalidate()
            self.status = f"Detector noise: {b.noise:.1%} of peak"
        elif key == "stretch":
            self.stretch_i = (self.stretch_i + 1) % len(dp.STRETCH_STEPS)
            b.stretch = dp.STRETCH_STEPS[self.stretch_i]
            self.invalidate()
            eff = b.effective_bw
            self.status = (f"THz probe: time-stretch x{b.stretch}"
                           + (f" -> {eff / 1e9:,.0f} GHz effective bandwidth" if eff else " (scope is ideal anyway)"))
        elif key in ("view_tdgsa", "view_topo"):
            self.view = key[5:]
            self._topo = None
        elif key == "xr":
            import desk_xr
            usdz, glb, tris = desk_xr.export(self, OUT)
            self.status = (f"Exported out/lab_desk.usdz (Vision Pro) + lab_desk.glb (Vive), {tris:,} triangles, "
                           "real scale. AirDrop the .usdz to open it in Quick Look.")
        elif key == "run":
            self.start_tdgsa()
        elif key == "matlab":
            self.open_matlab()
        elif key == "jupyter":
            self.open_jupyter()

    # ------------------------------------------------ update
    def update(self, dt):
        self.t += dt
        mx, my = pygame.mouse.get_pos() if not SHOT else self.hand
        k, c = (900.0, 60.0) if self.steady else (260.0, 22.0)        # spring / damping
        for i, target in enumerate((mx, my)):
            a = k * (target - self.hand[i]) - c * self.hand_v[i]
            self.hand_v[i] += a * dt
            self.hand[i] += self.hand_v[i] * dt
        if not self.steady:   # tremor: a few incommensurate sines
            self.hand[0] += 0.6 * math.sin(self.t * 11.3) + 0.4 * math.sin(self.t * 23.7)
            self.hand[1] += 0.6 * math.sin(self.t * 13.1 + 1) + 0.4 * math.sin(self.t * 19.9)
        if self.held:
            self.held.pos = [self.hand[0] + 6, self.hand[1] + 14]
        if self.run and self.run.it < self.iter_target:
            self.run.step(5)
            if self.run.it >= self.iter_target:
                e = self.run.field_err[-1]
                verdict = "recovered" if e < 0.15 else "did NOT converge"
                self.status = (f"TDGSA {verdict}: field error {e:.3f} after {self.run.it} iterations "
                               f"(D1/D2 = {self.bench.D1:g}/{self.bench.D2:g} ps/nm).")

    # ------------------------------------------------ drawing
    def draw(self):
        scr = self.screen
        self.buttons = []
        scr.fill((60, 42, 30))
        for y in range(0, H, 6):     # wood grain
            pygame.draw.line(scr, (66 + (y * 7) % 9, 46 + (y * 5) % 7, 33), (0, y), (W, y))
        bb = pygame.Rect(40, 150, 860, 330)       # optical breadboard
        pygame.draw.rect(scr, (52, 56, 62), bb, border_radius=6)
        for gx in range(bb.x + 15, bb.right, 25):
            for gy in range(bb.y + 15, bb.bottom, 25):
                pygame.draw.circle(scr, (36, 38, 44), (gx, gy), 3)
        scr.blit(self.fnt_big.render("Lab desk: dispersion-assisted phase recovery", True, (250, 236, 210)), (24, 14))
        scr.blit(self.fnt.render("Time-domain Gerchberg-Saxton (Solli, Gupta, Jalali 2009)  \u00b7  ECE 279AS", True,
                                 (220, 200, 170)), (26, 44))
        self.draw_bench()
        self.draw_scope()
        self.draw_laptop()
        for p in self.probes:          # cables hang from the scope to each probe tip
            pts = catenary_pts(p.anchor, p.pos, 1.06)
            pygame.draw.lines(scr, (20, 20, 24), False, pts, 6)
            pygame.draw.lines(scr, p.color, False, pts, 3)
            pygame.draw.circle(scr, (30, 30, 34), p.pos, 9)
            pygame.draw.circle(scr, p.color, p.pos, 7)
            scr.blit(self.fnt_s.render(p.name, True, (255, 255, 255)), (p.pos[0] + 10, p.pos[1] - 18))
        self.draw_hand()
        bar = pygame.Rect(0, H - 30, W, 30)
        pygame.draw.rect(scr, (20, 16, 12), bar)
        scr.blit(self.fnt.render(self.status, True, (240, 230, 210)), (14, H - 26))
        tip = "T: steady hand (on)" if self.steady else "T: steady hand (off)"
        t = self.fnt_s.render(f"{tip}   Space: run TDGSA   Esc: quit", True, (180, 170, 150))
        scr.blit(t, (W - t.get_width() - 12, H - 24))

    def fiber(self, p, q, slack=1.03):
        pts = catenary_pts(p, q, slack)
        pygame.draw.lines(self.screen, (120, 100, 20), False, [(x, y + 2) for x, y in pts], 5)
        pygame.draw.lines(self.screen, (240, 210, 60), False, pts, 3)

    def draw_bench(self):
        scr, P, b = self.screen, self.parts, self.bench
        self.fiber((220, 335), (290, 335))
        self.fiber((440, 335), (505, 335))
        self.fiber((555, 330), (610, 245), 1.05)
        self.fiber((555, 340), (610, 435), 1.05)
        self.fiber((730, 245), (790, 247))
        self.fiber((730, 435), (790, 437))
        box3d(scr, P["laser"], 14, (190, 196, 206), "1550 nm laser", self.fnt_s)
        pygame.draw.rect(scr, (230, 60, 50), (P["laser"].x + 10, P["laser"].y + 8, 10, 6))
        c = P["cell"]
        pygame.draw.ellipse(scr, (150, 200, 220), c)
        pygame.draw.ellipse(scr, (210, 240, 250), c.inflate(-20, -20))
        pygame.draw.ellipse(scr, (80, 120, 140), c, 2)
        lab = "gas cell (causal)" if b.causal else "gas cell (non-causal)"
        t = self.fnt_s.render(lab, True, (20, 30, 40))
        scr.blit(t, t.get_rect(center=c.center))
        box3d(scr, P["split"], 8, (170, 170, 180), "1\u00d72", self.fnt_s)
        for key, D in (("spool1", b.D1), ("spool2", b.D2)):
            r = P[key]
            pygame.draw.ellipse(scr, (40, 70, 140), (r.x, r.y + r.h - 24, r.w, 24))
            pygame.draw.rect(scr, (60, 100, 180), (r.x, r.y + 12, r.w, r.h - 24))
            pygame.draw.ellipse(scr, (90, 140, 220), (r.x, r.y, r.w, 24))
            for k in range(6):
                yy = r.y + 18 + k * (r.h - 30) / 6
                pygame.draw.line(scr, (240, 210, 60), (r.x + 4, yy), (r.right - 4, yy), 1)
            name = "D1" if key == "spool1" else "D2"
            t = self.fnt_b.render(f"{name} = {D:g} ps/nm", True, (255, 255, 255))
            scr.blit(t, t.get_rect(center=(r.centerx, r.y - 14)))
            reg = b.regime(D)
            t = self.fnt_s.render(reg, True, OKABE["green"] if reg == "near field" else OKABE["orange"])
            scr.blit(t, t.get_rect(center=(r.centerx, r.bottom + 10)))
        for key in ("pd1", "pd2"):
            box3d(scr, P[key], 10, (90, 92, 104), key.upper(), self.fnt_s)
        for name, (x, y) in self.tps.items():
            pygame.draw.circle(scr, (255, 215, 90), (x, y), 8, 2)
            scr.blit(self.fnt_s.render("TP " + name, True, (255, 225, 140)), (x - 18, y + 10))

    def draw_scope(self):
        scr = self.screen
        s = self.parts["scope"]
        box3d(scr, s, 18, (70, 74, 86))
        scr.blit(self.fnt_b.render("Oscilloscope", True, (230, 230, 240)), (s.x + 12, s.y + 6))
        sc = pygame.Rect(s.x + 14, s.y + 32, s.w - 28, 180)
        pygame.draw.rect(scr, (8, 12, 10), sc)
        for k in range(1, 10):
            x = sc.x + sc.w * k / 10
            pygame.draw.line(scr, (24, 40, 30), (x, sc.y), (x, sc.bottom))
        for k in range(1, 8):
            y = sc.y + sc.h * k / 8
            pygame.draw.line(scr, (24, 40, 30), (sc.x, y), (sc.right, y))
        span = TIMEBASE[self.tb] * 10
        b = self.bench
        for p in self.probes:
            if not p.at:
                continue
            I = self.trace(p.at)
            sel = np.abs(b.t) <= span / 2
            tt, yy = b.t[sel], I[sel]
            if len(tt) < 2:
                continue
            step = max(1, len(tt) // 400)
            pk = I.max() + 1e-30
            pts = [(sc.x + (x + span / 2) / span * sc.w, sc.bottom - 8 - (v / pk) * (sc.h - 20))
                   for x, v in zip(tt[::step], yy[::step])]
            pygame.draw.lines(scr, p.color, False, pts, 2)
        tb = TIMEBASE[self.tb]
        bwtxt = dp.SCOPE_BW[self.bw_i][0] + (f" x{b.stretch}" if b.stretch > 1 else "")
        scr.blit(self.mono.render(f"{tb * 1e12:g} ps/div   BW {bwtxt}   noise {b.noise:.1%}", True,
                                  (150, 230, 160)), (sc.x + 6, sc.y + 4))
        lbl = "  ".join(f"{p.name}: {p.at.upper() if p.at else '-'}" for p in self.probes)
        scr.blit(self.mono.render(lbl, True, (200, 220, 200)), (sc.x + 6, sc.bottom - 18))
        for k, (key, text) in enumerate((("tb", "TIME/DIV"), ("bw", "BANDWIDTH"), ("noise", "NOISE"),
                                         ("stretch", "THz STRETCH"))):
            cx = s.x + 45 + k * 82
            cy = s.y + 250
            r = pygame.Rect(cx - 22, cy - 22, 44, 44)
            pygame.draw.circle(scr, (40, 42, 50), (cx, cy), 22)
            pygame.draw.circle(scr, (120, 124, 140), (cx, cy), 22, 2)
            ang = [self.tb, self.bw_i, self.noise_i, self.stretch_i][k] * 0.9
            pygame.draw.line(scr, (240, 240, 250), (cx, cy), (cx + 16 * math.sin(ang), cy - 16 * math.cos(ang)), 3)
            t = self.fnt_s.render(text, True, (220, 220, 230))
            scr.blit(t, t.get_rect(center=(cx, cy + 32)))
            self.buttons.append((r, key))

    def draw_laptop(self):
        scr = self.screen
        lp = self.parts["laptop"]
        pygame.draw.polygon(scr, (40, 42, 48), [(lp.x - 20, lp.bottom), (lp.right + 20, lp.bottom),
                                                (lp.right, lp.bottom - 30), (lp.x, lp.bottom - 30)])
        sc = pygame.Rect(lp.x, lp.y, lp.w, lp.h - 36)
        pygame.draw.rect(scr, (30, 32, 38), sc.inflate(14, 14), border_radius=8)
        pygame.draw.rect(scr, (245, 246, 250), sc)
        bx = sc.x + 10
        for key, text, col in (("run", "Run TDGSA", OKABE["green"]), ("matlab", "MATLAB", OKABE["vermillion"]),
                               ("jupyter", "Jupyter", OKABE["orange"]),
                               ("xr", "Vision Pro", OKABE["blue"])):
            w = self.fnt_b.size(text)[0] + 20
            r = pygame.Rect(bx, sc.y + 8, w, 30)
            pygame.draw.rect(scr, col, r, border_radius=6)
            t = self.fnt_b.render(text, True, (255, 255, 255))
            scr.blit(t, t.get_rect(center=r.center))
            self.buttons.append((r, key))
            bx += w + 8
        for k, (key, text) in enumerate((("view_tdgsa", "TDGSA"), ("view_topo", "Topology"))):
            r = pygame.Rect(sc.right - 148 + k * 72, sc.y + 8, 68, 30)
            on = self.view == key[5:]
            pygame.draw.rect(scr, (60, 70, 96) if on else (200, 204, 214), r, border_radius=6)
            t = self.fnt_b.render(text, True, (255, 255, 255) if on else (40, 44, 60))
            scr.blit(t, t.get_rect(center=r.center))
            self.buttons.append((r, key))
        if self.view == "topo":
            self.draw_topology(sc)
            return
        plot = pygame.Rect(sc.x + 10, sc.y + 48, sc.w * 0.62, sc.h - 58)
        err = pygame.Rect(plot.right + 12, plot.y, sc.right - plot.right - 22, plot.h)
        for r in (plot, err):
            pygame.draw.rect(scr, (250, 250, 252), r)
            pygame.draw.rect(scr, (170, 174, 186), r, 1)
        b = self.bench
        fs = np.fft.fftshift(b.f)
        sel = np.abs(fs) <= 120e9
        truth = np.fft.fftshift(np.abs(b.field_after_cell()))[sel]
        pk = truth.max()
        xs = plot.x + (fs[sel] + 120e9) / 240e9 * plot.w

        def curve(vals, col, width, dash=False):
            pts = list(zip(xs.tolist(), (plot.bottom - 6 - vals / pk * (plot.h - 30)).tolist()))[::3]
            if dash:
                for k in range(0, len(pts) - 1, 3):
                    pygame.draw.line(scr, col, pts[k], pts[k + 1], width)
            else:
                pygame.draw.lines(scr, col, False, pts, width)
        curve(truth, (20, 20, 20), 2)
        if self.run:
            rec = np.fft.fftshift(np.abs(self.run.spectrum()))[sel]
            curve(np.minimum(rec, pk * 1.4), OKABE["vermillion"], 2, dash=True)
        scr.blit(self.fnt_s.render("|E(f)| after the gas cell: true (solid black), recovered (dashed red); "
                                   "\u00b1120 GHz", True, (60, 60, 70)), (plot.x + 6, plot.y + 4))
        scr.blit(self.fnt_s.render("field error vs iteration (log)", True, (60, 60, 70)), (err.x + 6, err.y + 4))
        if self.run and self.run.field_err:
            e = np.array(self.run.field_err)
            lo, hi = -2.5, 0.3
            pts = [(err.x + 6 + (k / max(1, self.iter_target - 1)) * (err.w - 12),
                    err.bottom - 8 - (math.log10(max(v, 10 ** lo)) - lo) / (hi - lo) * (err.h - 30))
                   for k, v in enumerate(e)]
            if len(pts) > 1:
                pygame.draw.lines(scr, OKABE["blue"], False, pts, 2)
            scr.blit(self.mono.render(f"it {self.run.it}  err {e[-1]:.3f}", True, (30, 40, 60)), (err.x + 6, err.bottom - 22))

    def topology(self):
        """Cached STFT images + vortices of the field in PD2's fiber (true and TDGSA-recovered), plus the braid."""
        if self._topo is not None:
            return self._topo
        b = self.bench
        view = np.exp(1j * dp.beta2L_from_psnm(b.D2) * b.omega ** 2 / 2)
        St, ts, fs = dp.stft(np.fft.ifft(b.field_after_cell() * view), b)
        out = dict(ts=ts, fs=fs, true=(St, dp.vortices(St, ts, fs)), rec=None, match=None)
        if self.run and self.run.it >= self.iter_target:
            Sr, _, _ = dp.stft(np.fft.ifft(self.run.spectrum() * view), b)
            out["rec"] = (Sr, dp.vortices(Sr, ts, fs))
            out["match"] = dp.vortex_match(out["true"][1], out["rec"][1], ts, fs)
        out["braid"] = dp.vortex_braid(b, [D for D in dp.D_STEPS if D <= 80])
        self._topo = out
        return out

    def draw_topology(self, sc):
        scr = self.screen
        T = self.topology()
        ts, fs = T["ts"], T["fs"]
        size = sc.h - 110
        for k, (title, data) in enumerate((("true field in the D2 fiber", T["true"]), ("TDGSA-recovered", T["rec"]))):
            r = pygame.Rect(sc.x + 10 + k * (size + 12), sc.y + 64, size, size)
            pygame.draw.rect(scr, (20, 22, 30), r)
            scr.blit(self.fnt_s.render(title, True, (40, 44, 60)), (r.x, r.y - 16))
            if data is None:
                t = self.fnt_s.render("run TDGSA first", True, (200, 200, 210))
                scr.blit(t, t.get_rect(center=r.center))
                continue
            S, vs = data
            img = np.log10(np.abs(S) / np.abs(S).max() + 1e-4)
            img = ((img + 4) / 4 * 255).clip(0, 255)
            rgb = np.stack([img * 0.55, img * 0.75, img], -1).astype(np.uint8)
            # array rows = time -> screen x; columns = frequency -> screen y (flip so +f is up)
            surf = pygame.surfarray.make_surface(rgb[:, ::-1])
            scr.blit(pygame.transform.smoothscale(surf, r.size), r.topleft)
            for t_, f_, q in vs:
                x = r.x + (t_ - ts[0]) / (ts[-1] - ts[0]) * r.w
                y = r.bottom - (f_ - fs[0]) / (fs[-1] - fs[0]) * r.h
                pygame.draw.circle(scr, OKABE["vermillion"] if q < 0 else OKABE["sky"], (x, y), 6, 2)
        br = pygame.Rect(sc.x + 2 * (size + 12) + 16, sc.y + 64, sc.right - (sc.x + 2 * (size + 12) + 26), size)
        pygame.draw.rect(scr, (250, 250, 252), br)
        pygame.draw.rect(scr, (170, 174, 186), br, 1)
        scr.blit(self.fnt_s.render("braid: vortex time vs D", True, (40, 44, 60)), (br.x, br.y - 16))
        Ds = [D for D, _ in T["braid"]]
        for k, (D, vs) in enumerate(T["braid"]):
            x = br.x + 8 + k / max(1, len(Ds) - 1) * (br.w - 16)
            for t_, f_, q in vs:
                y = br.bottom - 6 - (t_ - ts[0]) / (ts[-1] - ts[0]) * (br.h - 12)
                hue = (f_ - fs[0]) / (fs[-1] - fs[0])
                pygame.draw.circle(scr, (int(60 + 180 * hue), 90, int(240 - 180 * hue)), (x, y), 4)
        scr.blit(self.fnt_s.render(f"D {Ds[0]:g} to {Ds[-1]:g} ps/nm; colour = frequency", True, (90, 90, 100)),
                 (br.x + 4, br.bottom + 2))
        msg = "Vortices = zeros of the STFT; each turns the phase by 2\u03c0 (all q = \u22121). x: time \u00b130 ps, y: freq \u00b1120 GHz."
        if T["match"] is not None:
            msg += f"  Recovered field has {T['match']:.0%} of the true vortices in place."
        line1, _, line2 = msg.partition("x: time")
        scr.blit(self.fnt_s.render(line1.strip(), True, (40, 44, 60)), (sc.x + 10, sc.bottom - 30))
        scr.blit(self.fnt_s.render(("x: time" + line2).strip(), True, (40, 44, 60)), (sc.x + 10, sc.bottom - 15))

    def draw_hand(self):
        scr = self.screen
        x, y = self.hand
        glove = (90, 170, 120)
        for k, (dx, dy, L) in enumerate(((-12, -6, 18), (-5, -12, 22), (3, -12, 22), (10, -9, 19))):
            pygame.draw.line(scr, glove, (x + dx, y + dy + 10), (x + dx * 1.1, y + dy - L + 10), 7)
        pygame.draw.line(scr, glove, (x - 14, y + 18), (x - 24, y + 6), 7)
        pygame.draw.ellipse(scr, glove, (x - 17, y + 4, 34, 30))
        pygame.draw.ellipse(scr, (50, 120, 80), (x - 17, y + 4, 34, 30), 2)
        pygame.draw.circle(scr, (255, 255, 255), (x, y - 2), 2)

    def loop(self):
        running = True
        while running:
            dt = min(0.033, self.clock.tick(60) / 1000)
            for ev in pygame.event.get():
                running = self.handle(ev) and running
            self.update(dt)
            self.draw()
            pygame.display.flip()
        pygame.quit()


def shot(path):
    d = Desk()
    d.steady = True
    for p, tp in zip(d.probes, ("pd1", "pd2")):
        p.pos, p.at = list(d.tps[tp]), tp
    d.hand = [700, 600]
    d.start_tdgsa()
    for _ in range(60):
        d.update(1 / 30)
    d.draw()
    pygame.image.save(d.screen, path)
    print(d.status)


if __name__ == "__main__":
    if SHOT:
        shot(sys.argv[sys.argv.index("--shot") + 1] if len(sys.argv) > sys.argv.index("--shot") + 1 else "desk.png")
    else:
        Desk().loop()
