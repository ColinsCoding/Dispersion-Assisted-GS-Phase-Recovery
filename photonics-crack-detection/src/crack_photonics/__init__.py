"""crack_photonics -- photonics-inspired, symmetry-aware crack detection.

An educational, CPU-only, headless image-analysis package that connects

* optics      (a simplified Phase Stretch Transform edge detector),
* symmetry    (even / odd decomposition under mirror reflection),
* computing   (timing instrumentation and complexity), and
* civil engineering (crack area, length, width, orientation).

Modules
-------
io             image loading/saving, preprocessing, synthetic test images
symmetry       reflection, even/odd decomposition, symmetry score
pst_edges      simplified Phase Stretch Transform (PST) edge detector
crack_metrics  crack segmentation and civil-engineering metrics
timing         ``time.perf_counter`` stage timer
cli            command-line entry point (``python -m crack_photonics.cli``)
"""

__version__ = "0.1.0"

# Single source of truth for the default random seed used by the synthetic
# image generators.  Every random number in this package comes from
# ``numpy.random.default_rng(seed)``, never from global random state.
DEFAULT_SEED = 0

__all__ = ["__version__", "DEFAULT_SEED"]
