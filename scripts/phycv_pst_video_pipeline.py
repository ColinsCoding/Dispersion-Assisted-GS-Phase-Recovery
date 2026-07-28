"""Run PhyCV's Phase-Stretch Transform on a VIDEO, frame by frame.

apply_pst_to_video() works on ANY video file -- point it at your own
screen recording (Minecraft or otherwise) and it will run. This script
does not ship, download, or synthesize Minecraft footage itself: Minecraft's
textures and any recorded gameplay are copyrighted, so the demo below runs
on a procedurally generated "voxel-style" clip instead (a panning camera
over a grid of solid-color blocks, generated from scratch, not game
assets) -- enough blocky structure to see PST do something real without
using anything that isn't ours to use.

Run:
    python scripts/phycv_pst_video_pipeline.py
"""

import pathlib

import numpy as np
import cv2
import torch
import imageio.v3 as iio

from phycv.pst_gpu import PST_GPU

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "notebooks"


def generate_synthetic_voxel_clip(width=384, height=192, n_frames=90, block=32, seed=0):
    """A procedurally generated "world" of solid-color blocks (grass green,
    dirt brown, stone gray, sky blue -- generic palette, not a texture
    ripped from any game), with a camera that pans across it -- gives a
    short clip with real motion and hard block edges, entirely original
    content.
    """
    rng = np.random.default_rng(seed)
    palette = np.array([
        [95, 159, 53],    # grass-like green
        [110, 74, 46],    # dirt-like brown
        [128, 128, 128],  # stone-like gray
        [107, 174, 214],  # sky-like blue
    ], dtype=np.uint8)

    world_w = width + n_frames * 2  # extra width so the camera has room to pan
    n_cols = world_w // block + 1
    n_rows = height // block + 1

    # simple "terrain": sky on top rows, then grass/dirt/stone with depth
    world = np.zeros((n_rows * block, n_cols * block, 3), dtype=np.uint8)
    horizon_row = n_rows // 3
    for row in range(n_rows):
        for col in range(n_cols):
            if row < horizon_row:
                color = palette[3]  # sky
            elif row == horizon_row:
                color = palette[0]  # grass strip
            elif row < horizon_row + 2:
                color = palette[1]  # dirt
            else:
                color = palette[2]  # stone
            jitter = rng.integers(-8, 9, size=3)
            block_color = np.clip(color.astype(int) + jitter, 0, 255).astype(np.uint8)
            world[row * block:(row + 1) * block, col * block:(col + 1) * block] = block_color

    frames = []
    for f in range(n_frames):
        x0 = f * 2  # pan speed: 2 px/frame
        frame = world[:height, x0:x0 + width]
        frames.append(frame.copy())
    return frames


def frames_to_video(frames, out_path, fps=24):
    iio.imwrite(str(out_path), np.stack(frames), fps=fps, codec="libx264")
    print(f"wrote {out_path}")


def apply_pst_to_video(input_path, output_path, S=0.5, W=15, sigma_LPF=0.1, fps=24, device=None):
    """Run PhyCV's torch PST on every frame of input_path, write the
    edge-feature frames to output_path. Works on any video file -- if you
    have your own (legally captured) Minecraft recording, pass its path
    here directly."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {input_path}")

    ok, first_frame = cap.read()
    if not ok:
        raise ValueError(f"video has no readable frames: {input_path}")
    gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape

    # kernel only depends on frame size, not frame content -- build ONCE
    pst = PST_GPU(device=dev)
    pst.h, pst.w = h, w
    pst.init_kernel(S=S, W=W)

    def run_one(gray_frame):
        img_t = torch.from_numpy(gray_frame).unsqueeze(0).to(dev)
        pst.load_img(img_array=img_t)
        pst.apply_kernel(sigma_LPF=sigma_LPF, thresh_min=None, thresh_max=None, morph_flag=0)
        out = (pst.pst_output.detach().cpu().numpy() * 255).astype(np.uint8)
        return out

    out_frames = [run_one(gray)]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        out_frames.append(run_one(gray))
    cap.release()

    frames_to_video(out_frames, output_path, fps=fps)
    return len(out_frames)


if __name__ == "__main__":
    print("Generating a procedural voxel-style clip (not Minecraft assets)...")
    frames = generate_synthetic_voxel_clip()
    original_path = OUT_DIR / "phycv_pst_synthetic_voxel_clip.mp4"
    frames_to_video(frames, original_path, fps=24)

    print("\nRunning PhyCV PST_GPU on every frame...")
    pst_path = OUT_DIR / "phycv_pst_synthetic_voxel_clip_pst.mp4"
    n_frames = apply_pst_to_video(original_path, pst_path)
    print(f"\nProcessed {n_frames} frames.")
    print(f"Original: {original_path}")
    print(f"PST edge output: {pst_path}")
    print("\nTo run this on your own footage instead:")
    print("  from phycv_pst_video_pipeline import apply_pst_to_video")
    print("  apply_pst_to_video('your_video.mp4', 'output_pst.mp4')")
