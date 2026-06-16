"""Interactive Pygame viewer for the dispersion-GS phase-recovery results.

Renders the two measured intensities I1, I2 and the Gerchberg-Saxton-recovered
phase vs the hidden truth, and lets you sweep the second dispersion D2 live to
watch the recovery improve or collapse with measurement diversity. Civilian
optical metrology / education.

Run:  py -3.13 viewer_pygame.py        (needs:  pip install pygame)
Keys: UP/DOWN = change |D2| (diversity),  R = re-run recovery,  ESC = quit.
"""

import numpy as np

import dispersion_gs_prototype as dg
import gs_core as gs


def _polyline(surface, color, ys, x0, y0, w, h, lo=None, hi=None):
    import pygame
    lo = ys.min() if lo is None else lo
    hi = ys.max() if hi is None else hi
    rng = (hi - lo) or 1.0
    n = len(ys)
    pts = [(x0 + i * w / (n - 1), y0 + h - (ys[i] - lo) / rng * h) for i in range(n)]
    pygame.draw.lines(surface, color, False, pts, 2)


def main():
    try:
        import pygame
    except ImportError:
        raise SystemExit("Pygame not installed. Run:  pip install pygame")

    pygame.init()
    W, H = 980, 720
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Dispersion-Assisted GS Phase Recovery")
    font = pygame.font.SysFont("consolas", 18)
    clock = pygame.time.Clock()

    # fixed truth; only the second dispersion (diversity) is tunable
    data = dg.make_measurements(N=512, D=6000.0, seed=7)
    x, phi_true = data["x"], data["phi"]
    I1 = np.abs(x) ** 2
    D1 = 5000.0
    D2 = 5750.0

    def recover(d2):
        I2 = np.abs(dg.disperse(x, d2)) ** 2
        phi, _ = gs.retrieve_phase(I1, I2, -D1, -d2, n_iter=80, unit_amplitude=False)
        # align to truth (global offset + twin)
        best = None
        for s in (1, -1):
            off = np.angle(np.mean(np.exp(1j * (phi_true - s * phi))))
            err = np.sqrt(np.mean(np.angle(np.exp(1j * (phi_true - (s * phi + off)))) ** 2))
            if best is None or err < best[0]:
                best = (err, s * phi + off)
        corr = np.corrcoef(I1, I2)[0, 1]
        return I2, best[1], best[0], corr

    I2, phi_rec, rms, corr = recover(D2)
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key in (pygame.K_UP, pygame.K_DOWN, pygame.K_r):
                    if ev.key == pygame.K_UP:
                        D2 = min(D2 + 500, 20000)
                    elif ev.key == pygame.K_DOWN:
                        D2 = max(D2 - 500, 5200)
                    I2, phi_rec, rms, corr = recover(D2)

        screen.fill((13, 17, 23))
        # three stacked panels
        _polyline(screen, (76, 201, 240), I1, 30, 40, W - 60, 160, 0)
        _polyline(screen, (247, 37, 133), I2, 30, 250, W - 60, 160, 0)
        _polyline(screen, (200, 200, 200), phi_true, 30, 470, W - 60, 200)
        _polyline(screen, (255, 90, 90), phi_rec, 30, 470, W - 60, 200,
                  lo=phi_true.min(), hi=phi_true.max())

        labels = [
            (f"I1 (before dispersion)", 18, (76, 201, 240)),
            (f"I2 (after D2 = {D2:.0f})   corr(I1,I2) = {corr:.3f}", 228, (247, 37, 133)),
            (f"phase: true (grey) vs recovered (red)   RMS = {rms:.3f} rad", 448, (255, 255, 255)),
            ("UP/DOWN: change |D2| (diversity)   R: re-run   ESC: quit", 688, (150, 150, 150)),
        ]
        for text, y, col in labels:
            screen.blit(font.render(text, True, col), (30, y))
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    main()
