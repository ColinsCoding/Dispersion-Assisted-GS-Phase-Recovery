"""Run PhyCV's PST on your own screen in real time -- point it at any open
window (Minecraft or otherwise) by (partial, case-insensitive) title match.

This only reads pixels that are already visible on YOUR screen, live, for
YOUR own local viewing -- nothing is uploaded, saved to disk, or sent
anywhere, and no game assets are embedded in this script: it has zero
Minecraft content in it, the same way a screen magnifier or an OBS filter
has zero game content in it. It just re-renders whatever is already on
your screen through PhyCV's edge-detecting kernel.

Must be run directly in an interactive terminal (it opens a live window
and reads keypresses) -- it will not do anything useful piped through a
non-interactive process.

Usage:
    python scripts/phycv_pst_live_screen.py --title Minecraft
    python scripts/phycv_pst_live_screen.py --title Minecraft --width 640 --W 25
    python scripts/phycv_pst_live_screen.py --list-windows   # see exact titles first

Press 'q' in the display window to quit.
"""

import argparse
import sys
import time

import cv2
import numpy as np
import torch
import win32gui
from PIL import ImageGrab

from phycv.pst_gpu import PST_GPU

# Some window titles contain characters (emoji, box-drawing glyphs) a
# non-UTF-8 Windows console can't encode -- replace rather than crash.
sys.stdout.reconfigure(errors="replace")


def list_open_windows():
    """All visible top-level window titles -- use this to find the exact
    substring to pass as --title if the default 'Minecraft' doesn't match
    (some launchers/versions title the window differently, e.g. a
    version string or '* - Singleplayer' suffix)."""
    titles = []

    def _cb(hwnd, _extra):
        if win32gui.IsWindowVisible(hwnd):
            t = win32gui.GetWindowText(hwnd)
            if t.strip():
                titles.append(t)

    win32gui.EnumWindows(_cb, None)
    return titles


def find_window_rect(title_substring):
    """Case-insensitive substring search over visible window titles.
    Raises with the full list of open titles if nothing matches, so you
    can immediately see the exact title to use instead."""
    needle = title_substring.lower()
    matches = []

    def _cb(hwnd, _extra):
        if win32gui.IsWindowVisible(hwnd):
            t = win32gui.GetWindowText(hwnd)
            if needle in t.lower():
                matches.append((hwnd, t))

    win32gui.EnumWindows(_cb, None)

    if not matches:
        open_titles = list_open_windows()
        raise ValueError(
            f"No open window title contains '{title_substring}'. "
            f"Open windows right now: {open_titles}"
        )
    hwnd, title = matches[0]
    if win32gui.IsIconic(hwnd):
        raise ValueError(
            f"Found window '{title}', but it's minimized -- Windows won't hand over "
            f"real pixels for a minimized window (you'd just capture a blank frame). "
            f"Restore/un-minimize it and try again."
        )
    return hwnd, title, win32gui.GetWindowRect(hwnd)


def capture_gray(rect, target_width):
    """Grab the window's rect, downscale to target_width (keeping aspect
    ratio) for real-time throughput, return a float32 grayscale array in
    [0, 1] plus the resized BGR frame for display."""
    img = ImageGrab.grab(bbox=rect)
    frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h0, w0 = frame_bgr.shape[:2]
    if w0 == 0 or h0 == 0:
        raise ValueError("captured an empty region -- is the window minimized?")
    scale = target_width / w0
    target_size = (target_width, max(1, int(h0 * scale)))
    frame_bgr = cv2.resize(frame_bgr, target_size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return gray, frame_bgr


def run_live_pst(title_substring="Minecraft", S=0.5, W=15.0, sigma_lpf=0.1,
                  target_width=480, side_by_side=True, device=None):
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hwnd, title, rect = find_window_rect(title_substring)
    print(f"Capturing window: '{title}'  (hwnd={hwnd}, rect={rect})")
    print(f"device: {dev}   S={S}  W={W}  sigma_lpf={sigma_lpf}  target_width={target_width}")
    print("Press 'q' in the display window to quit.")

    gray0, _ = capture_gray(rect, target_width)
    h, w = gray0.shape

    pst = PST_GPU(device=dev)
    pst.h, pst.w = h, w
    pst.init_kernel(S=S, W=W)

    win_name = "PhyCV PST -- live (press q to quit)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    t_fps = time.perf_counter()
    try:
        while True:
            if win32gui.IsIconic(hwnd):
                print("  window minimized -- restore it to resume capture (waiting)...")
                time.sleep(0.5)
                continue

            # re-query the rect every frame in case the window moved/resized
            rect = win32gui.GetWindowRect(hwnd)
            gray, frame_bgr = capture_gray(rect, target_width)

            img_t = torch.from_numpy(gray).unsqueeze(0).to(dev)
            pst.load_img(img_array=img_t)
            pst.apply_kernel(sigma_LPF=sigma_lpf, thresh_min=None, thresh_max=None, morph_flag=0)
            feature = (pst.pst_output.detach().cpu().numpy() * 255).astype(np.uint8)
            feature_bgr = cv2.cvtColor(feature, cv2.COLOR_GRAY2BGR)

            if side_by_side:
                display = np.hstack([frame_bgr, feature_bgr])
            else:
                display = feature_bgr
            cv2.imshow(win_name, display)

            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.perf_counter() - t_fps)
                print(f"  ~{fps:.1f} FPS")
                t_fps = time.perf_counter()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", default="Minecraft",
                         help="Case-insensitive substring of the window title to capture (default: Minecraft)")
    parser.add_argument("--S", type=float, default=0.5, help="PST phase strength")
    parser.add_argument("--W", type=float, default=15.0, help="PST warp strength")
    parser.add_argument("--sigma-lpf", type=float, default=0.1, help="Low-pass filter sigma before the PST kernel")
    parser.add_argument("--width", type=int, default=480, help="Downscaled capture width (height keeps aspect ratio)")
    parser.add_argument("--overlay-only", action="store_true",
                         help="Show only the PST output, not the original side by side")
    parser.add_argument("--list-windows", action="store_true", help="Print open window titles and exit")
    args = parser.parse_args()

    if args.list_windows:
        for t in list_open_windows():
            print(t)
    else:
        run_live_pst(
            title_substring=args.title, S=args.S, W=args.W, sigma_lpf=args.sigma_lpf,
            target_width=args.width, side_by_side=not args.overlay_only,
        )
