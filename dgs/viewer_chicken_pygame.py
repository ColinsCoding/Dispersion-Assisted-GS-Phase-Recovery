"""Pygame prototype: CS:GO-style chicken chase with live calculus on the HUD.

Run:   py -3.13 -m dgs.viewer_chicken_pygame
Keys:  arrows / WASD  move          TAB  autopilot (pure pursuit of nearest chicken)
       P  Bouguer pursuit lab        B    1-bit view (the C program's framebuffer)
       R  reset                      ESC  quit
Headless (tests / screenshots):
       py -3.13 -m dgs.viewer_chicken_pygame --headless --frames 400 --shot out.png
       (--start lab | bits  opens the pursuit lab or the 1-bit view)

Logic lives in dgs/chicken_chase_game.py; this file only draws and reads keys.
"""
import argparse
import os
import sys

import numpy as np

if __package__ in (None, ""):            # run as a file (VS Code "Run Python File"):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dgs.chicken_chase_game import (
    World, Params, PursuitLab, render_bits, WORLD_W, WORLD_H,
)

SCALE = 6
PANEL = 380
BG, FG, DIM = (18, 18, 24), (235, 235, 235), (120, 120, 140)
CHICK, PLAYER, RING, CURVE = (250, 220, 90), (90, 200, 255), (70, 90, 130), (255, 120, 90)


def _txt(screen, font, s, x, y, col=FG):
    screen.blit(font.render(s, True, col), (x, y))


def _draw_world(pg, screen, world):
    s = SCALE
    pg.draw.rect(screen, DIM, (0, 0, WORLD_W * s, WORLD_H * s), 1)
    px, py = world.player * s
    pg.draw.circle(screen, RING, (int(px), int(py)), int(world.params.flee_radius * s), 1)
    for (x, y, h), a in zip(world.chickens, world.alive):
        if a:
            pg.draw.circle(screen, CHICK, (int(x * s), int(y * s)), 5)
            pg.draw.line(screen, CHICK, (x * s, y * s), (x * s + 9 * np.cos(h), y * s + 9 * np.sin(h)), 2)
    pg.draw.circle(screen, PLAYER, (int(px), int(py)), 8)


def _draw_bits(pg, screen, world):
    img, packed = render_bits(world)
    rgb = np.repeat(img[:, :, None] * 255, 3, axis=2).astype(np.uint8)
    surf = pg.surfarray.make_surface(rgb.transpose(1, 0, 2))
    screen.blit(pg.transform.scale(surf, (WORLD_W * SCALE, WORLD_H * SCALE)), (0, 0))
    return packed.size


def _draw_lab(pg, screen, lab):
    s = SCALE
    if len(lab.path) > 1:
        pg.draw.lines(screen, CURVE, False, [(x * s, y * s) for x, y in lab.path], 2)
    pg.draw.circle(screen, PLAYER, (int(lab.chaser[0] * s), int(lab.chaser[1] * s)), 7)
    pg.draw.circle(screen, CHICK, (int(lab.target[0] * s), int(lab.target[1] * s)), 6)
    x0 = lab.path[0][0] + lab.d
    pg.draw.line(screen, DIM, (x0 * s, lab.path[0][1] * s), (x0 * s, 0), 1)


def run(headless=False, frames=None, shot=None, seed=0, start="game"):
    if start not in ("game", "lab", "bits"):
        raise ValueError("start must be 'game', 'lab' or 'bits'")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame as pg
    pg.init()
    screen = pg.display.set_mode((WORLD_W * SCALE + PANEL, WORLD_H * SCALE))
    pg.display.set_caption("Chicken chase -- pursuit calculus prototype")
    font = pg.font.SysFont("consolas", 16)
    big = pg.font.SysFont("consolas", 20, bold=True)
    clock = pg.time.Clock()

    world = World(Params(), seed=seed)
    autopilot, bits = headless, start == "bits"
    lab = PursuitLab() if start == "lab" else None
    n = 0
    running = True
    while running:
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                running = False
            elif ev.type == pg.KEYDOWN:
                if ev.key == pg.K_ESCAPE:
                    running = False
                elif ev.key == pg.K_TAB:
                    autopilot = not autopilot
                elif ev.key == pg.K_b:
                    bits = not bits
                elif ev.key == pg.K_r:
                    world, lab = World(Params(), seed=np.random.randint(1 << 30)), None
                elif ev.key == pg.K_p:
                    lab = None if lab else PursuitLab()

        keys = pg.key.get_pressed()
        move = (
            (keys[pg.K_RIGHT] or keys[pg.K_d]) - (keys[pg.K_LEFT] or keys[pg.K_a]),
            (keys[pg.K_DOWN] or keys[pg.K_s]) - (keys[pg.K_UP] or keys[pg.K_w]),
        )
        if lab:
            lab.advance(1)
        else:
            world.step(move, autopilot)

        screen.fill(BG)
        nbytes = None
        if lab:
            _draw_lab(pg, screen, lab)
        elif bits:
            nbytes = _draw_bits(pg, screen, world)
        else:
            _draw_world(pg, screen, world)

        x = WORLD_W * SCALE + 14
        y = 12
        if lab:
            _txt(screen, big, "Bouguer pursuit lab", x, y); y += 30
            for line in (f"d = {lab.d:.0f}, v_p = {lab.v_p:.2f}",
                         f"v_c = k v_p = {lab.v_c:.2f}",
                         "", "T = d v_p / (v_p^2 - v_c^2)",
                         f"predicted  T = {lab.predicted:7.2f}",
                         f"elapsed    t = {lab.t:7.2f}",
                         ("CAUGHT: error %.2f%%" % (100 * abs(lab.t - lab.predicted) / lab.predicted))
                         if lab.done else "chasing...", "", "P = back to the game"):
                _txt(screen, font, line, x, y, FG if line else DIM); y += 22
        else:
            p = world.params
            pred = world.flee_prediction()
            _txt(screen, big, f"caught {world.caught}/{p.n_chickens}", x, y); y += 30
            for line in (f"step {world.steps}",
                         f"mode: {'AUTOPILOT' if autopilot else 'keys'}{'  1-bit' if bits else ''}",
                         "", "nearest chicken flees straight:",
                         "  dr/dt = -(v_p - v_c)",
                         f"  T = (r-r_c-v_p)/(v_p-v_c) = {pred:5.1f}" if pred is not None else "  all caught!",
                         f"  v_p = {p.v_player}, v_c = {p.v_flee}",
                         "", (f"framebuffer: {nbytes} bytes (1 bit/px)" if nbytes else ""),
                         "TAB autopilot   P pursuit lab",
                         "B 1-bit view    R reset   ESC quit"):
                _txt(screen, font, line, x, y, FG if line else DIM); y += 22
        pg.display.flip()

        n += 1
        if frames is not None and n >= frames:
            running = False
        if not headless:
            clock.tick(60)

    if shot:
        pg.image.save(screen, shot)
    result = {"caught": world.caught, "steps": world.steps}
    if lab:
        result.update(lab_done=lab.done, lab_t=lab.t, lab_predicted=lab.predicted)
    pg.quit()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--shot", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", choices=("game", "lab", "bits"), default="game")
    a = ap.parse_args()
    print(run(a.headless, a.frames, a.shot, a.seed, a.start))


if __name__ == "__main__":
    main()
