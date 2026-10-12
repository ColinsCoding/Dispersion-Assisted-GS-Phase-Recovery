"""Command-line interface for the crack_photonics pipeline.

Example
-------
::

    python -m crack_photonics.cli \\
        --input data/example.png \\
        --output results/example \\
        --headless

``--output`` is a DIRECTORY.  The command writes into it:

    crack_mask.png        processed crack mask (white = crack)
    even_horizontal.png   even part under left-right reflection
    odd_horizontal.png    odd part  (mid-gray = 0, signed)
    even_vertical.png     even part under top-bottom reflection
    odd_vertical.png      odd part  (mid-gray = 0, signed)
    edges.png             simplified-PST edge strength
    metrics.json          symmetry scores and crack metrics
    timing.json           time.perf_counter() report per stage
    summary.png           one-page overview figure

Exit status: 0 on success, 2 on a usage or input error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import DEFAULT_SEED, __version__
from . import crack_metrics, io, pst_edges, symmetry
from .timing import StageTimer, format_report

OUTPUT_FILES = (
    "crack_mask.png",
    "even_horizontal.png",
    "odd_horizontal.png",
    "even_vertical.png",
    "odd_vertical.png",
    "edges.png",
    "metrics.json",
    "timing.json",
    "summary.png",
)


# --------------------------------------------------------------------------
# Pure analysis (no files, no plotting) -- reused by the notebook and tests
# --------------------------------------------------------------------------
def analyze(
    raw_image: np.ndarray,
    timer: StageTimer | None = None,
    pixel_size_mm: float | None = None,
) -> dict:
    """Run preprocessing, symmetry, edge detection and metrics on one image.

    Returns a dict with the intermediate arrays (``preprocessed``, ``mask``,
    ``edge_strength``, ``edge_mask``, ``even``/``odd`` per axis) and the two
    JSON-friendly reports ``symmetry`` and ``crack``.
    """
    timer = timer if timer is not None else StageTimer()

    with timer.stage("preprocessing"):
        image = io.preprocess(raw_image)

    with timer.stage("symmetry_analysis"):
        symmetry_report = symmetry.analyze_symmetry(image)
        components = {
            axis: symmetry.even_odd_decompose(image, axis) for axis in symmetry.AXES
        }

    with timer.stage("edge_detection"):
        edge_strength, edge_mask = pst_edges.detect_edges(image)

    with timer.stage("metric_extraction"):
        mask = crack_metrics.segment_crack(image)
        crack_report = crack_metrics.compute_metrics(mask, image, edge_mask, pixel_size_mm)

    return {
        "preprocessed": image,
        "mask": mask,
        "edge_strength": edge_strength,
        "edge_mask": edge_mask,
        "even": {axis: parts[0] for axis, parts in components.items()},
        "odd": {axis: parts[1] for axis, parts in components.items()},
        "symmetry": symmetry_report,
        "crack": crack_report,
    }


# --------------------------------------------------------------------------
# Summary figure
# --------------------------------------------------------------------------
def save_summary_plot(path: Path, raw: np.ndarray, result: dict, headless: bool) -> Path:
    """Save the 2x3 overview figure.  With ``headless`` no window is opened."""
    import matplotlib

    if headless:
        matplotlib.use("Agg")  # file-only backend: needs no display server
    import matplotlib.pyplot as plt

    crack = result["crack"]
    sym = result["symmetry"]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))

    axes[0, 0].imshow(raw, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Input (grayscale)")

    overlay = np.dstack([result["preprocessed"]] * 3)
    overlay[result["mask"]] = [1.0, 0.1, 0.1]
    axes[0, 1].imshow(overlay)
    axes[0, 1].set_title(f"Crack mask ({crack['crack_area_percent']:.2f} % of pixels)")

    axes[0, 2].imshow(result["edge_strength"], cmap="magma")
    axes[0, 2].set_title("Simplified-PST edge strength")

    axes[1, 0].imshow(result["even"]["horizontal"], cmap="gray")
    axes[1, 0].set_title("Even part (left-right mirror)")

    odd = result["odd"]["horizontal"]
    limit = max(float(np.abs(odd).max()), 1e-12)
    axes[1, 1].imshow(odd, cmap="RdBu_r", vmin=-limit, vmax=limit)
    axes[1, 1].set_title("Odd part (left-right mirror)")

    for ax in axes.flat[:5]:
        ax.set_xticks([])
        ax.set_yticks([])

    angle = crack["orientation_deg"]
    text = "\n".join([
        "SYMMETRY",
        f"  S horizontal : {sym['horizontal']['score']:.4f}",
        f"  S vertical   : {sym['vertical']['score']:.4f}",
        f"  odd energy H : {sym['horizontal']['odd_energy_fraction']:.2e}",
        "",
        "CRACK METRICS (pixels)",
        f"  area         : {crack['crack_area_percent']:.3f} %",
        f"  length       : {crack['crack_length_px']:.1f}",
        f"  mean width   : {crack['mean_width_px']:.2f}",
        f"  max width    : {crack['max_width_px']:.2f}",
        f"  orientation  : {'n/a' if angle is None else f'{angle:.1f} deg'}",
        f"  components   : {crack['connected_components']}",
        f"  confidence   : {crack['confidence']:.2f}  (heuristic)",
        "",
        "Symmetry is a feature, not proof.",
        "Pixels are not millimetres.",
    ])
    axes[1, 2].axis("off")
    axes[1, 2].text(0.0, 1.0, text, family="monospace", fontsize=10,
                    va="top", ha="left", transform=axes[1, 2].transAxes)

    fig.suptitle("crack_photonics summary", fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    if not headless:
        plt.show()
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Full pipeline with file output
# --------------------------------------------------------------------------
def run(
    input_path: str | Path,
    output_dir: str | Path,
    headless: bool = True,
    make_example: bool = False,
    seed: int = DEFAULT_SEED,
    pixel_size_mm: float | None = None,
) -> tuple[dict, dict]:
    """Run the whole pipeline on one file and write every output.

    Returns ``(metrics, timing)`` -- the same content as the two JSON files.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"--output must be a directory, but {output_dir} is a file.")

    synthetic_created = False
    if make_example and not input_path.exists():
        sample = io.generate_synthetic_crack(seed=seed)
        io.save_image(input_path, sample.image)
        synthetic_created = True
        print(f"Created synthetic example image (seed={seed}): {input_path}")

    timer = StageTimer()

    with timer.stage("image_loading"):
        raw = io.load_image(input_path)

    result = analyze(raw, timer=timer, pixel_size_mm=pixel_size_mm)

    metrics = {
        "package_version": __version__,
        "input": str(input_path),
        "image_shape_rows_cols": list(raw.shape),
        "synthetic_example_created_this_run": synthetic_created,
        "synthetic_seed": seed if synthetic_created else None,
        "deterministic": True,
        "symmetry": result["symmetry"],
        "crack": result["crack"],
        "disclaimer": (
            "Educational output. Edge detector is a simplified approximation "
            "of the Phase Stretch Transform. Confidence is a heuristic, not a "
            "calibrated probability. Not for structural safety decisions."
        ),
    }

    with timer.stage("output_saving"):
        io.save_image(output_dir / "crack_mask.png", result["mask"])
        io.save_image(output_dir / "edges.png", result["edge_strength"])
        for axis in symmetry.AXES:
            io.save_image(output_dir / f"even_{axis}.png", result["even"][axis])
            io.save_image(output_dir / f"odd_{axis}.png",
                          io.signed_to_display(result["odd"][axis]))
        io.save_json(output_dir / "metrics.json", metrics)
        save_summary_plot(output_dir / "summary.png", raw, result, headless)

    # The clock stops here; writing timing.json itself is not included.
    timer.stop()
    timing = timer.report()
    io.save_json(output_dir / "timing.json", timing)
    return metrics, timing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m crack_photonics.cli",
        description="Photonics-inspired, symmetry-aware crack analysis of one image "
                    "(CPU only, no GUI required).",
    )
    parser.add_argument("--input", required=True, type=Path,
                        help="input image (PNG, JPEG, TIFF, BMP)")
    parser.add_argument("--output", required=True, type=Path,
                        help="output DIRECTORY for masks, JSON reports and the plot")
    parser.add_argument("--headless", action="store_true",
                        help="never open a window; only save files (use on servers/SSH)")
    parser.add_argument("--make-example", action="store_true",
                        help="if --input does not exist, create the deterministic "
                             "synthetic crack image there first")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"random seed for --make-example (default {DEFAULT_SEED})")
    parser.add_argument("--pixel-size-mm", type=float, default=None,
                        help="millimetres per pixel from a scale reference; "
                             "without it, lengths stay in pixels")
    parser.add_argument("--version", action="version", version=f"crack_photonics {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        metrics, timing = run(
            input_path=args.input,
            output_dir=args.output,
            headless=args.headless,
            make_example=args.make_example,
            seed=args.seed,
            pixel_size_mm=args.pixel_size_mm,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        # Expected user-facing problems: report clearly and fail with status 2.
        # Anything unexpected is NOT caught, so the full traceback is shown.
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    crack = metrics["crack"]
    sym = metrics["symmetry"]
    angle = crack["orientation_deg"]
    print(f"Input            : {args.input}  {metrics['image_shape_rows_cols']} (rows, cols)")
    print(f"Symmetry score S : horizontal {sym['horizontal']['score']:.4f}, "
          f"vertical {sym['vertical']['score']:.4f}")
    print(f"Crack area       : {crack['crack_area_percent']:.3f} % of pixels")
    print(f"Crack length     : {crack['crack_length_px']:.1f} px")
    print(f"Crack width      : mean {crack['mean_width_px']:.2f} px, "
          f"max {crack['max_width_px']:.2f} px")
    print(f"Orientation      : {'n/a' if angle is None else f'{angle:.1f} deg'}")
    print(f"Components       : {crack['connected_components']}")
    print(f"Confidence       : {crack['confidence']:.2f} (heuristic, not a probability)")
    if crack["pixel_size_mm"] is None:
        print("Units            : pixels only (no --pixel-size-mm scale reference given)")
    else:
        print(f"Length / width   : {crack['crack_length_mm']:.2f} mm / "
              f"{crack['mean_width_mm']:.3f} mm mean "
              f"(at {crack['pixel_size_mm']} mm per pixel)")
    print()
    print(format_report(timing))
    print()
    print(f"Wrote {len(OUTPUT_FILES)} files to {args.output}{'/' if not str(args.output).endswith('/') else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
