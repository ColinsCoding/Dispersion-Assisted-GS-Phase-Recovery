# Chicken chase prototypes

A CS:GO-style chicken chase, built twice, to teach 1-bit graphics memory, C
toolchains, and the calculus of pursuit.

| prototype | file | run |
|---|---|---|
| C, 1-bit framebuffer, PBM frames | [`chicken_chase.c`](chicken_chase.c) | `gcc -std=c99 -O2 -Wall -Wextra -o chicken_chase chicken_chase.c -lm` then `./chicken_chase sim 7 1500 60 frames/` |
| pygame, playable, live HUD math | [`dgs/viewer_chicken_pygame.py`](../../dgs/viewer_chicken_pygame.py) | `py -3.13 -m dgs.viewer_chicken_pygame` |
| notebook walk-through | [`notebooks/chicken_chase_c_bits_pursuit.ipynb`](../../notebooks/chicken_chase_c_bits_pursuit.ipynb) | — |

**pygame keys:** arrows or WASD to move · TAB autopilot · **P** Bouguer pursuit lab · **B** 1-bit view · R reset · ESC quit.
Headless screenshot: `py -3.13 -m dgs.viewer_chicken_pygame --headless --frames 300 --start lab --shot lab.png`.

## The math on screen

| situation | model | result | checked by |
|---|---|---|---|
| chicken flees straight away | $\dot r = -(v_p - v_c)$ | $T = (r - r_c - v_p)/(v_p - v_c)$ steps (player moves before the catch check), until it hits a wall | `test_straight_flight_capture_time` |
| Bouguer pursuit (P lab) | chaser always aims at a target running perpendicular | $T = d\,v_p/(v_p^2 - v_c^2)$; never if $v_c \ge v_p$ | C sim + pygame lab, within 0.2% |
| 1-bit pixel $i = yW + x$ | byte `i >> 3`, mask `0x80 >> (i & 7)` | 128×96 screen = 1,536 bytes | C selftest, objdump of `px_set` |

## Prototype log

- **v1 (C):** `-std=c99` hides `M_PI` (it's POSIX, not ISO C), so it's defined
  locally. The `(unsigned)x >= W` bounds check compiles to one `cmp`/`ja`.
- **v2 (pygame):** logic in `dgs/chicken_chase_game.py` (tested, no pygame), drawing
  in the viewer. The pursuit lab's first version was 0.7% off because of a coarse time
  step and catch radius. With `dt = 0.01` and `eps = 0.02` it's 0.04% off.
- **Game-loop lesson:** the first straight-flight prediction, $r/(v_p-v_c)$, was 6 steps off.
  The player moves *before* the catch check, which adds $v_p$ to the catch distance. With a
  slow closing speed (0.15/step), that one-step head start is $v_p/(v_p-v_c) = 6$ steps.
- **Known limitation:** the straight-flight prediction ignores walls. A cornered
  chicken is caught early (the straight-flight test deliberately starts far from
  the walls). Modeling the bounce is an open exercise.
