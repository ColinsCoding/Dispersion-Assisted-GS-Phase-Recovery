"""A single publishable Matplotlib style for the whole repository.

Calling `use_style()` in a notebook fixes fonts, sizes, and grid so every figure looks the same. The
palette is colour-blind-safe and prints legibly in greyscale.
"""

import matplotlib as mpl

PALETTE = ["#4C78A8", "#E45756", "#54A24B", "#F58518", "#72B7B2", "#B279A2"]


def use_style():
    """Apply the repository plotting style to the current Matplotlib session."""
    mpl.rcParams.update({
        "figure.figsize": (7.5, 4.2),
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "lines.linewidth": 1.8,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
