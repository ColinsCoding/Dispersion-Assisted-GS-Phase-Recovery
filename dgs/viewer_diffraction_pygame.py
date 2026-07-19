"""Interactive Pygame viewer for the N-slit diffraction grating (dgs.diffraction_grating).

Renders the far-field intensity pattern as both a curve and a rendered "screen"
band colored by wavelength_to_rgb -- what the grating pattern would actually look
like projected onto a wall. Live-tunable: wavelength, slit count N, slit spacing d,
slit width a. Civilian optics education.

Run:  py -3.13 -m dgs.viewer_diffraction_pygame     (needs:  pip install pygame)
Keys: LEFT/RIGHT = wavelength,  UP/DOWN = slit count N,
      A/Z = slit width a,  D/C = slit spacing d,  ESC = quit.
"""

import numpy as np

from dgs import diffraction_grating as dgt


def _polyline(surface, color, xs, ys, x0, y0, w, h, lo=None, hi=None):
    import pygame
    lo = ys.min() if lo is None else lo
    hi = ys.max() if hi is None else hi
    rng = (hi - lo) or 1.0
    xlo, xhi = xs.min(), xs.max()
    xrng = (xhi - xlo) or 1.0
    pts = [(x0 + (xs[i] - xlo) / xrng * w, y0 + h - (ys[i] - lo) / rng * h) for i in range(len(xs))]
    pygame.draw.lines(surface, color, False, pts, 2)


def main():
    try:
        import pygame
    except ImportError:
        raise SystemExit("Pygame not installed. Run:  pip install pygame")

    pygame.init()
    W, H = 1000, 640
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("N-Slit Diffraction Grating")
    font = pygame.font.SysFont("consolas", 18)
    clock = pygame.time.Clock()

    wavelength_nm = 589.0   # sodium D line
    N = 6
    d = 2.0e-6              # slit spacing [m]
    a = 0.4e-6               # slit width  [m]
    L = 1.0                  # virtual screen distance [m], for the x-axis label only

    theta = np.linspace(-0.9, 0.9, 2000)  # radians, wide enough for several orders

    def recompute():
        wl = wavelength_nm * 1e-9
        I = dgt.grating_intensity(theta, d, a, N, wl)
        x_screen = dgt.angle_to_screen_position(theta, L)
        m, theta_max = dgt.principal_maxima_angles(d, wl, m_max=10)
        return I, x_screen, m, theta_max

    I, x_screen, orders, theta_max = recompute()
    running = True
    while running:
        dirty = False
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_LEFT:
                    wavelength_nm = max(380.0, wavelength_nm - 10.0); dirty = True
                elif ev.key == pygame.K_RIGHT:
                    wavelength_nm = min(750.0, wavelength_nm + 10.0); dirty = True
                elif ev.key == pygame.K_UP:
                    N = min(N + 1, 40); dirty = True
                elif ev.key == pygame.K_DOWN:
                    N = max(N - 1, 1); dirty = True
                elif ev.key == pygame.K_a:
                    a = min(a * 1.15, d * 0.95); dirty = True
                elif ev.key == pygame.K_z:
                    a = max(a / 1.15, 1e-8); dirty = True
                elif ev.key == pygame.K_d:
                    d = d * 1.1; dirty = True
                elif ev.key == pygame.K_c:
                    d = max(d / 1.1, a * 1.05); dirty = True
        if dirty:
            I, x_screen, orders, theta_max = recompute()

        rgb01 = dgt.wavelength_to_rgb(wavelength_nm)
        rgb = tuple(int(255 * c) for c in rgb01)

        screen.fill((10, 10, 14))

        # panel 1: intensity vs angle curve
        _polyline(screen, rgb, theta, I, 30, 40, W - 60, 220, lo=0.0, hi=1.0)
        for th in theta_max:
            if theta.min() <= th <= theta.max():
                px = 30 + (th - theta.min()) / (theta.max() - theta.min()) * (W - 60)
                pygame.draw.line(screen, (70, 70, 80), (px, 40), (px, 260), 1)

        # panel 2: rendered "screen" -- fringe band colored by wavelength, brightness = I
        band_y0, band_h = 300, 120
        n_px = W - 60
        I_band = np.interp(np.linspace(theta.min(), theta.max(), n_px), theta, I)
        for i, val in enumerate(I_band):
            col = tuple(int(255 * c * val) for c in rgb01)
            pygame.draw.line(screen, col, (30 + i, band_y0), (30 + i, band_y0 + band_h))

        labels = [
            (f"Intensity I(theta)   N={N} slits   d={d*1e6:.2f} um   a={a*1e6:.3f} um   "
             f"lambda={wavelength_nm:.0f} nm", 18, (220, 220, 220)),
            (f"orders visible: {list(orders)}", 268, (150, 150, 160)),
            ("projected screen (color = wavelength, brightness = intensity)", 440, (150, 150, 160)),
            ("LEFT/RIGHT: wavelength   UP/DOWN: N   A/Z: slit width a   D/C: spacing d   ESC: quit",
             600, (140, 140, 140)),
        ]
        for text, y, col in labels:
            screen.blit(font.render(text, True, col), (30, y))

        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    main()
