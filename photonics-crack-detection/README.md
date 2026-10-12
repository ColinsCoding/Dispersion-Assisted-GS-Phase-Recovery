# photonics-crack-detection

A small, readable, **headless** Python project that connects four subjects:

| Subject | What appears here |
|---|---|
| Photonics | A simplified, educational approximation of the Phase Stretch Transform (PST), an edge detector modelled on optical dispersion |
| Symmetry | Even/odd decomposition of an image under mirror reflection, and a symmetry score |
| Computer science | A command-line pipeline, unit tests, `time.perf_counter()` instrumentation, complexity analysis |
| Civil engineering | Crack area, length, width, orientation, connected components, and a confidence score |

It runs on any laptop CPU from a Linux terminal. It needs **no GPU, no laser,
no paid cloud service, and no specialised equipment**. It is meant for
community-college classes, home labs and university teaching labs alike.

> **This is a teaching tool.** It does not assess structural safety. See
> [Physics and Engineering Limits](#physics-and-engineering-limits).

No figures are shipped. `make demo` writes a one-page overview to
`results/example/summary.png`.

## Quick start

```bash
git clone <your fork or copy of this repository>
cd photonics-crack-detection

python3 -m venv .venv
source .venv/bin/activate

make install     # pip install -r requirements.txt && pip install -e .
make test        # 61 unit tests
make demo        # headless demo on a synthetic crack image
```

On Debian/Ubuntu, if `python3 -m venv` fails, install it once with
`sudo apt install python3-venv`.

The demo is the same as running:

```bash
python -m crack_photonics.cli \
  --input data/example.png \
  --output results/example \
  --headless \
  --make-example
```

`--make-example` creates `data/example.png` from the built-in generator
(seed 0) if the file does not exist yet. Without that flag a missing input is
an error with exit status 2.

## Command-line reference

| Option | Meaning |
|---|---|
| `--input PATH` | Image to analyse (PNG, JPEG, TIFF, BMP). Colour is converted to grayscale. |
| `--output DIR` | Output **directory** (created if needed). |
| `--headless` | Never open a window; only write files. Use this over SSH and on servers. |
| `--make-example` | Create the synthetic example at `--input` if it is missing. |
| `--seed N` | Seed for `--make-example` (default 0). |
| `--pixel-size-mm X` | Millimetres per pixel from a scale reference. Without it, lengths stay in pixels. |

Files written to the output directory:

| File | Content |
|---|---|
| `crack_mask.png` | Processed crack mask (white = crack) |
| `even_horizontal.png`, `even_vertical.png` | Even component under the left-right / top-bottom mirror |
| `odd_horizontal.png`, `odd_vertical.png` | Odd component (signed: mid-gray = 0) |
| `edges.png` | Simplified-PST edge strength |
| `metrics.json` | Symmetry scores and crack metrics |
| `timing.json` | Seconds per stage from `time.perf_counter()` |
| `summary.png` | One-page overview figure |

## How it works

### 1. Symmetry (`symmetry.py`)

For a reflection $I_R$ of the image $I$:

$$
I_{\mathrm{even}}=\frac{I+I_R}{2},\qquad I_{\mathrm{odd}}=\frac{I-I_R}{2},
\qquad
S=1-\frac{\lVert I-I_R\rVert_2}{\lVert I\rVert_2+\epsilon}.
$$

Reflection is a parity operator ($P^2=1$), so the even and odd parts are its
$+1$ and $-1$ eigen-components. They are orthogonal:
$\lVert I\rVert^2=\lVert I_{\mathrm{even}}\rVert^2+\lVert I_{\mathrm{odd}}\rVert^2$.
`"horizontal"` means the left-right mirror, `"vertical"` the top-bottom mirror.

$S=1$ is perfectly symmetric. For a non-negative image the minimum is
$1-\sqrt2\approx-0.41$, so $S$ is not a probability. Because mean brightness
is trivially symmetric and usually dominates, the report also includes
`score_mean_removed`.

### 2. Photonics-inspired edges (`pst_edges.py`)

**A simplified educational approximation of the Phase Stretch Transform.**
The image spectrum is low-pass filtered, multiplied by an all-pass phase
filter $e^{-j\varphi(r)}$ whose phase grows with spatial frequency (as optical
dispersion delays different frequencies by different amounts), and transformed
back. The *phase* of the result is large at edges.

$$
\varphi(r)=S\,\frac{Wr\arctan(Wr)-\tfrac12\ln\bigl(1+(Wr)^2\bigr)}{\max(\cdot)}
$$

This is a digital filter written from the published equations (Asghari and
Jalali, *Int. J. Biomedical Imaging*, 2015). It is not the reference
implementation and it **does not reproduce laboratory photonic hardware**.

### 3. Crack mask and metrics (`crack_metrics.py`)

The mask comes from a plain rule: a pixel is crack if it is clearly darker
than a wide Gaussian average of its surroundings (robust threshold, small
blobs removed). The PST-style edge map is used as independent support in the
confidence score.

| Metric | Definition |
|---|---|
| `crack_area_percent` | crack pixels / all pixels x 100 |
| `crack_length_px` | length of the one-pixel skeleton (Zhang-Suen thinning); straight step = 1, diagonal step = $\sqrt2$ |
| `mean_width_px` | area / length |
| `max_width_px` | from the Euclidean distance transform on the skeleton |
| `orientation_deg` | principal axis of the crack pixels, in [0, 180), counter-clockwise from the picture's +x axis |
| `connected_components` | number of separate crack pieces (8-connected) |
| `confidence` | 0.4 contrast + 0.3 edge support + 0.3 elongation. A heuristic, **not** a calibrated probability |

### 4. Timing (`timing.py`)

Each stage is timed with `time.perf_counter()`. `timing.json` always contains
`image_loading_s`, `preprocessing_s`, `symmetry_analysis_s`,
`edge_detection_s`, `metric_extraction_s` and `total_runtime_s`, plus
`output_saving_s` and `untimed_overhead_s`.

Example from one run on a desktop CPU (Ubuntu 24.04 under WSL2, Python 3.12,
256 x 256 image):

```text
stage                      seconds    share
image_loading             0.003191     0.4%
preprocessing             0.000862     0.1%
symmetry_analysis         0.028658     3.8%
edge_detection            0.017298     2.3%
metric_extraction         0.015421     2.0%
output_saving             0.687760    91.3%
untimed_overhead          0.000086     0.0%
total_runtime             0.753276   100.0%
```

Most of the wall time is importing Matplotlib and drawing the summary figure,
not the analysis. Single-run numbers include one-time start-up costs and vary
between machines and runs; the notebook shows how to benchmark with repeats.

| Stage | Dominant operation | Time for N pixels |
|---|---|---|
| symmetry analysis | flips, sums, norms | O(N) |
| edge detection | two 2-D FFTs | O(N log N) |
| segmentation | separable Gaussian blur of width k | O(N k) |
| skeleton | p thinning passes, p about half the crack width | O(N p) |

## Reproducibility

- The analysis pipeline contains **no random numbers**. The same input file
  gives the same masks and metrics.
- Random numbers are used only by the synthetic image generators, through
  `numpy.random.default_rng(seed)` with a default seed of 0
  (`crack_photonics.DEFAULT_SEED`). No global random state is touched.
- A unit test runs the command twice and checks that `metrics.json` and
  `crack_mask.png` are identical.
- Timing values are measurements and are never identical between runs.
- Results can differ in the last decimal places between NumPy/SciPy/OpenCV
  versions or CPU types because floating-point libraries differ.

## Notebook

`notebooks/symmetry_crack_detection.ipynb` explains the physical motivation,
image symmetry, crack detection, timing and complexity, and the difference
between a physics-inspired algorithm and a data-driven neural network. It is
stored with its outputs. To re-execute it without opening a browser:

```bash
make notebook
```

## Tests

```bash
make test
```

| File | Covers |
|---|---|
| `tests/test_symmetry.py` | symmetric synthetic image, asymmetric crack image, even/odd reconstruction, energy identity, score bounds, determinism, error messages |
| `tests/test_metrics.py` | percentage calculations, length/width/orientation on known shapes, components, mm conversion, synthetic ground truth, PST step edge, I/O errors |
| `tests/test_timing.py` | required timing keys, timer behaviour, exceptions, end-to-end headless CLI run, CLI error status, run-to-run reproducibility |

## Project layout

```text
photonics-crack-detection/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── Makefile
├── data/
│   └── README.md
├── src/
│   └── crack_photonics/
│       ├── __init__.py
│       ├── io.py              loading, saving, preprocessing, synthetic images
│       ├── symmetry.py        reflection, even/odd parts, symmetry score
│       ├── pst_edges.py       simplified PST edge detector
│       ├── crack_metrics.py   segmentation, skeleton, crack metrics
│       ├── timing.py          perf_counter stage timer
│       └── cli.py             command-line entry point
├── notebooks/
│   └── symmetry_crack_detection.ipynb
├── tests/
│   ├── test_symmetry.py
│   ├── test_metrics.py
│   └── test_timing.py
└── results/
    └── .gitkeep
```

## Physics and Engineering Limits

- **Image symmetry is a feature, not proof of a crack.** Asymmetry also comes
  from stains, shadows, texture, bolts, formwork marks and a camera that is
  not centred. A defect that lies on the mirror line, or a symmetric pair of
  defects, does not lower the score at all. Use the score as one input among
  several, never as a verdict.
- **Synthetic data is not a substitute for calibrated concrete, asphalt, or
  masonry data.** The built-in generator draws one clean dark line on smooth
  noise. It tests whether the code is correct. It says nothing about accuracy
  on real surfaces, which have aggregate, pores, joints, efflorescence, paint,
  moisture and uneven lighting. Thresholds in this repository were chosen on
  synthetic images and will need re-tuning and validation on labelled images
  of your material.
- **Pixels must be converted to millimeters using a scale reference.** A pixel
  is not a unit of length. Photograph a ruler or crack-width gauge in the same
  plane as the crack, with the camera square to the surface, measure
  millimetres per pixel, and pass it as `--pixel-size-mm`. Without it the
  millimetre fields in `metrics.json` are `null`; the code does not guess.
  Width estimates below about three pixels are unreliable at any scale.
- **Train/test splits must be separated by specimen to avoid data leakage.**
  If you extend this project with a learned model, or tune thresholds, keep
  every image and patch from one specimen (one beam, slab, core, wall panel)
  entirely in training or entirely in testing. Patches from the same specimen
  share material, lighting and often the same crack, so a random patch-level
  split reports accuracy that will not hold on a new structure.
- **High-power optical equipment requires institutional safety procedures.**
  This software needs no light source. If you build an optical experiment
  around it, anything beyond an eye-safe source needs your institution's
  laser-safety officer, training, written procedures, eyewear rated for the
  wavelength and power, and beam enclosures. Do not improvise high-power
  optics at home.

Further limits worth stating plainly:

- The edge detector is a simplified approximation of PST and is not validated
  against the reference implementation or any hardware.
- The dark-residual mask assumes cracks are darker than the surface. It will
  flag shadows, joints, pencil lines and dark stains (the notebook shows a
  saw-cut joint reported with confidence 1.00).
- The length estimate counts 0/45/90-degree pixel steps and can overestimate
  by up to about 8 % for lines at other angles.
- The confidence score is a hand-built heuristic. It is not a probability.
- Nothing here is a substitute for inspection by a qualified engineer.

## Open-Source and Copyright

- **Original code.** All source code in this repository was written for this
  project and is released under the MIT License (see `LICENSE`). The PST-style
  filter is an independent implementation of published equations, not a copy
  of any existing code.
- **Permissively licensed dependencies.** NumPy (BSD-3-Clause), SciPy
  (BSD-3-Clause), OpenCV (Apache-2.0), Matplotlib (PSF-style licence), pytest
  (MIT), and Jupyter (BSD-3-Clause). None is bundled; `pip` installs them.
- **User-supplied or openly licensed images only.** The repository ships no
  photographs. The example image is generated by the code. Analyse images you
  took yourself, or images whose licence permits your use, and keep the source
  and licence with them.
- **No copyrighted or proprietary assets.** This project does not download,
  extract or redistribute copyrighted game footage, screenshots, or
  proprietary company assets, and contributions must not add any.

## Assumptions

- "Off screen" is read as headless: with `--headless` Matplotlib uses the
  file-only `Agg` backend and no display server is needed.
- `--output` names a directory.
- There is no CUDA code path. The pipeline is CPU-only by design, so every
  machine runs the same code.
- OpenCV is used in place of scikit-image; the headless wheel is listed so no
  GUI libraries are pulled in.
