"""Shot-noise study: writes figures/shot_noise.png and prints the numbers quoted in the README.

    python scripts/make_shot_noise.py      (about 15 minutes on a laptop CPU; the 3M-shot runs dominate)
"""
import pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dgs import shot_noise as S

SHOTS = [300, 1000, 3000, 10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000]
N_SEEDS = 10
ctx = S.setup()
r = S.sweep(SHOTS, n_seeds=N_SEEDS, ctx=ctx)
for s, sr, m in zip(r["shots"], r["success_rate"], r["median_success"]):
    print(f"{int(s):>9} shots/arm: success {100*sr:5.0f}%   median error when successful {m:.4f} rad")
print(f"exact probabilities (infinite shots): {r['exact']:.2e} rad")
fit_mask = (r["success_rate"] >= 0.8) & np.isfinite(r["median_success"])
slope, pref = S.fit_slope(r["shots"][fit_mask], r["median_success"][fit_mask])
need = S.shots_needed(0.01, slope, pref)
first_reliable = r["shots"][np.argmax(r["success_rate"] >= 0.9)]
print(f"successful-run error ∝ shots^{slope:.2f} (shot-noise theory: −0.50), fitted on {fit_mask.sum()} budgets")
print(f"≥ 90% success from {int(first_reliable):,} shots per arm; 0.01 rad needs ≈ {need:,.0f} shots per arm")

BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#555555"
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.3))
ax[0].semilogx(r["shots"], 100*r["success_rate"], "o-", color=BLUE, lw=2, ms=7)
ax[0].axhline(90, color=GREY, ls=":", lw=1); ax[0].text(r["shots"][0], 92, "90%", color=GREY, fontsize=8)
ax[0].set_xlabel("shots per detector arm"); ax[0].set_ylabel(f"runs recovering the phase (%)\n(RMS error < {S.SUCCESS_RAD} rad)")
ax[0].set_title(f"1. Threshold: GS needs enough shots to work at all ({N_SEEDS} seeds each)", loc="left", fontsize=10)
ax[0].set_ylim(-5, 105)
for s_, row in zip(r["shots"], r["errors"]):
    good = row < S.SUCCESS_RAD
    ax[1].loglog([s_]*good.sum(), row[good], ".", color=BLUE, alpha=0.5, ms=6)
ax[1].loglog(r["shots"][fit_mask], r["median_success"][fit_mask], "o", color=BLUE, ms=8, label="median of successful runs")
xs = np.logspace(np.log10(r["shots"][fit_mask][0]), np.log10(r["shots"][-1]), 50)
ax[1].loglog(xs, pref*xs**slope, "-", color=ORANGE, lw=2, label=f"fit: error ∝ shots^{slope:.2f}")
ax[1].loglog(xs, pref*xs[0]**slope*(xs/xs[0])**-0.5, "--", color=GREY, lw=1, label="shot-noise law ∝ shots^−0.5")
ax[1].set_xlabel("shots per detector arm"); ax[1].set_ylabel("RMS phase error (rad)")
ax[1].set_title("2. Noise floor: once it works, error falls like shots^−½", loc="left", fontsize=10)
ax[1].legend(fontsize=8)
fig.suptitle("Dispersion-assisted GS on a simulated quantum computer: how many measurements? (7 qubits, D = −1000/−1150)",
             fontsize=11, x=0.01, ha="left")
plt.tight_layout(); fig.savefig(ROOT/"figures"/"shot_noise.png", dpi=120, bbox_inches="tight")
print("wrote figures/shot_noise.png")
