"""Live Deep-Dispersion-Prior phase retrieval -- a pygame front-end.

Runs the unsupervised optimizer from gs_unsupervised.py one Adam step per frame
and animates, in real time:
  * top panel    -- measured intensity I1 vs the reconstructed |disperse(E,D1)|^2
  * bottom panel -- ground-truth phase vs the recovered phase (global-offset aligned)
plus a live readout of iteration / data-loss / phase-RMSE.

Run:
    py -3.12 deploy/phase_retrieval_pygame.py                 # interactive
    py -3.12 deploy/phase_retrieval_pygame.py --model ddp     # smoothness-prior net
    py -3.12 deploy/phase_retrieval_pygame.py --headless      # CI: N steps, save PNG, exit

Headless mode (or SDL_VIDEODRIVER=dummy) needs no display; it renders to an
off-screen surface and saves a screenshot to figures/.
ESC or window-close quits.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gs_core
import gs_unsupervised as gu


# ── colors / layout ───────────────────────────────────────────────────────────
BG = (16, 18, 24)
GRID = (40, 44, 54)
MEAS = (150, 156, 170)
RECON = (64, 220, 238)
TRUE = (236, 240, 248)
EST = (250, 204, 60)
TEXT = (210, 216, 228)
W, H = 960, 620


def _demo_data(N, snr_db, seed=0):
    """A smooth-phase, unit-amplitude test field and its two dispersed intensities."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, N)
    phi = (1.2*np.sin(2*np.pi*3*t) + 0.7*np.cos(2*np.pi*7*t) + 0.4*np.sin(2*np.pi*11*t))
    E = np.exp(1j*phi)
    D1, D2 = -5000.0, -5750.0
    I = []
    for D in (D1, D2):
        Ik = np.abs(gs_core.disperse(E, D))**2
        if snr_db is not None:
            p = Ik.mean() / (10**(snr_db/10))
            Ik = np.maximum(Ik + rng.normal(0, np.sqrt(p), Ik.shape), 0.0)
        I.append(Ik)
    return phi, I, [D1, D2]


def _polyline(rect, y, ymin, ymax):
    """Map a 1-D array to screen points inside rect=(x,y,w,h)."""
    x0, y0, w, h = rect
    n = len(y)
    span = (ymax - ymin) or 1.0
    xs = x0 + np.arange(n) / (n - 1) * w
    ys = y0 + (1.0 - (np.asarray(y) - ymin) / span) * h
    return list(zip(xs.tolist(), ys.tolist()))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Live DDP phase retrieval (pygame)")
    ap.add_argument("--model", choices=["direct", "ddp"], default="direct")
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--snr", type=float, default=20.0, help="measurement SNR in dB")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--tv", type=float, default=0.0, help="TV phase regularizer weight")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args(argv)

    headless = args.headless or os.environ.get("SDL_VIDEODRIVER") == "dummy"
    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    import torch
    import pygame

    # data + ground truth
    phi_true, I_list, D_list = _demo_data(args.N, args.snr)
    gu._validate(I_list, D_list, args.steps, 1.0)        # reuse the kwarg guards

    # model + optimizer (per-step control, so we drive the loop ourselves)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    Net = gu.PhaseNet if args.model == "ddp" else gu.DirectPhase
    net = Net(args.N, unit_amplitude=True).to(device)
    lr = args.lr if args.lr is not None else (1e-3 if args.model == "ddp" else 5e-2)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    I_meas = [torch.tensor(I, dtype=torch.float32, device=device) for I in I_list]

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(f"DDP phase retrieval [{args.model}] -- {device}")
    font = pygame.font.SysFont("consolas", 18)
    big = pygame.font.SysFont("consolas", 22, bold=True)
    clock = pygame.time.Clock()

    top = (60, 70, W - 120, 210)
    bot = (60, 360, W - 120, 200)
    I1 = I_list[0]
    imin, imax = float(min(x.min() for x in I_list)), float(max(x.max() for x in I_list))
    running, step = True, 0
    while running and step < args.steps:
        if not headless:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (
                        ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                    running = False

        # one optimization step
        opt.zero_grad()
        E = net.field()
        loss = gu.data_consistency_loss(E, I_meas, D_list, args.tv, net.phase())
        loss.backward()
        opt.step()
        loss_val = float(loss.detach())
        step += 1

        # current reconstruction (numpy, for drawing)
        with torch.no_grad():
            recon_I1 = (gu.disperse_torch(net.field(), D_list[0]).abs()**2).cpu().numpy()
            phi_est = net.phase().detach().cpu().numpy()
        phi_aligned = gu.align_global_phase(phi_est, phi_true)
        rmse = min(gu.phase_rmse(phi_est, phi_true), gu.phase_rmse(-phi_est, phi_true))

        # ── draw ──
        screen.fill(BG)
        for rect in (top, bot):
            pygame.draw.rect(screen, GRID, rect, 1)
        pygame.draw.lines(screen, MEAS, False, _polyline(top, I1, imin, imax), 2)
        pygame.draw.lines(screen, RECON, False, _polyline(top, recon_I1, imin, imax), 2)
        pygame.draw.lines(screen, TRUE, False, _polyline(bot, phi_true, -np.pi, np.pi), 2)
        pygame.draw.lines(screen, EST, False, _polyline(bot, phi_aligned, -np.pi, np.pi), 2)

        screen.blit(big.render("Deep Dispersion Prior - unsupervised phase retrieval",
                               True, TEXT), (60, 24))
        screen.blit(font.render("intensity arm 1:  measured", True, MEAS), (70, 80))
        screen.blit(font.render("reconstructed", True, RECON), (320, 80))
        screen.blit(font.render("phase:  true", True, TRUE), (70, 368))
        screen.blit(font.render("recovered", True, EST), (240, 368))
        hud = (f"model={args.model}  step={step:4d}/{args.steps}  "
               f"loss={loss_val:.2e}  phaseRMSE={rmse:.3f} rad  {device}")
        screen.blit(font.render(hud, True, TEXT), (60, H - 34))

        pygame.display.flip()
        if not headless:
            clock.tick(60)

    final_rmse = min(gu.phase_rmse(phi_est, phi_true), gu.phase_rmse(-phi_est, phi_true))
    if headless:
        figs = ROOT / "figures"
        figs.mkdir(exist_ok=True)
        out = figs / f"ddp_pygame_{args.model}.png"
        pygame.image.save(screen, str(out))
        print(f"[headless] {step} steps, final phase RMSE {final_rmse:.3f} rad -> {out}")
    else:
        print(f"done: {step} steps, final phase RMSE {final_rmse:.3f} rad")
    pygame.quit()
    return final_rmse


if __name__ == "__main__":
    main()
