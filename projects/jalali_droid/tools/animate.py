"""Turn the droid's per-frame CSV log into a GIF: top-down arena + laser, and the tracking/jitter traces.

Usage: python tools/animate.py out/droid_run.csv out/droid.gif [--every 2]
Okabe-Ito colours; the droid is the black diamond, the foam-sword fighter the orange circle.
"""
import argparse
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

OI = dict(orange="#E69F00", sky="#56B4E9", green="#009E73", blue="#0072B2", red="#D55E00", purple="#CC79A7", black="#000000")


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [{k: float(v) for k, v in r.items()} for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("gif")
    ap.add_argument("--every", type=int, default=2, help="use every k-th frame")
    a = ap.parse_args()
    rows = load(a.csv)[:: a.every]
    t = [r["t"] for r in rows]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9, 4.2), dpi=80, gridspec_kw=dict(width_ratios=[1, 1.15]))
    ax.set(xlim=(-8.5, 8.5), ylim=(-8.5, 8.5), aspect="equal", xlabel="x (m)", ylabel="y (m)", title="Jalali droid, top view")
    ax.plot(0, 0, "D", color=OI["black"], ms=9)
    trail, = ax.plot([], [], "-", color=OI["orange"], lw=1, alpha=0.5)
    player, = ax.plot([], [], "o", color=OI["orange"], ms=10)
    beam, = ax.plot([], [], "-", color=OI["green"], lw=1.5)
    label = ax.text(-8.2, 7.6, "", fontsize=8)
    bx.plot(t, [r["err_deg"] for r in rows], "-", color=OI["blue"], label="aim error (deg)")
    bx.plot(t, [r["spot_offset_mm"] / 100 for r in rows], "--", color=OI["red"], label="spot offset (x100 mm)")
    bx.plot(t, [abs(r["jitter_mrad"]) for r in rows], ":", color=OI["purple"], label="|head jitter| (mrad)")
    bx.axhline(0.5, color=OI["black"], lw=0.7)
    cursor = bx.axvline(t[0], color=OI["black"], lw=1)
    bx.set(xlabel="time (s)", title="tracking (laser fires below 0.5 deg)", ylim=(0, max(5, max(r["err_deg"] for r in rows[5:]) * 1.1)))
    bx.legend(fontsize=7, loc="upper right")
    fig.tight_layout()

    def frame(i):
        r = rows[i]
        trail.set_data([q["player_x"] for q in rows[max(0, i - 40):i + 1]], [q["player_y"] for q in rows[max(0, i - 40):i + 1]])
        player.set_data([r["player_x"]], [r["player_y"]])
        player.set_color(OI["red"] if r["tag"] else OI["orange"])
        pan = math.radians(r["pan_deg"])
        L = r["range_m"] * math.cos(math.radians(r["tilt_deg"]))
        beam.set_data([0, L * math.cos(pan)], [0, L * math.sin(pan)])
        beam.set_linestyle("-" if r["firing"] else ":")
        beam.set_color(OI["green"] if r["firing"] else OI["sky"])
        label.set_text(f"t={r['t']:.2f}s  fov={r['fov_deg']:.0f} deg  " + ("TAG!" if r["tag"] else ("laser" if r["firing"] else "tracking")))
        cursor.set_xdata([r["t"], r["t"]])
        return trail, player, beam, label, cursor

    anim = animation.FuncAnimation(fig, frame, frames=len(rows), blit=False)
    anim.save(a.gif, writer=animation.PillowWriter(fps=15))
    print(f"wrote {a.gif}: {len(rows)} frames. Black diamond = droid, orange circle = foam-sword fighter (red when tagged), "
          "line = camera aim (solid green while the laser fires, dotted blue while tracking). Right: aim error (blue), "
          "laser-spot offset at the target (red dashed, mm/100) and head jitter (purple dotted) against time.")


if __name__ == "__main__":
    main()
