"""Generate the synthetic dataset and save it to results/.

    python scripts/generate_synthetic_data.py [--seed 0]

Writes
    results/synthetic_dataset.npz    arrays (ADC codes, angles, DLS counts, ...)
    results/synthetic_dataset.json   truth, instrument settings, provenance

Everything in these files is SIMULATED. Nothing was measured.
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scarlett import synthetic  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=synthetic.DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)

    data = synthetic.generate_dataset(seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    scan, dark = data["scan"], data["dark"]
    np.savez_compressed(
        args.output / "synthetic_dataset.npz",
        wavelengths_m=data["wavelengths_m"],
        scattering_angles_rad=data["scattering_angles_rad"],
        usable_channels=data["usable_channels"],
        scan_code_sums=scan["code_sums"], scan_volts=scan["volts"],
        scan_sigma_volts=scan["sigma_volts"], scan_clipped=scan["clipped"],
        dark_code_sums=dark["code_sums"], dark_volts=dark["volts"],
        dark_sigma_volts=dark["sigma_volts"],
        dls_counts=data["dls_counts"])

    bench = asdict(data["bench"])
    bench["equalization"] = list(bench["equalization"])
    meta = {
        "synthetic": True,
        "warning": "Simulated data. Not an experimental result.",
        "seed": args.seed,
        "truth": data["truth"],
        "pga_gain": scan["gain"],
        "n_particles_true": scan["n_particles_true"],
        "channel_ripple_rel": data["ripple_rel"],
        "dls": {"dt_s": data["dls_dt_s"], "theta_rad": data["dls_theta_rad"],
                "decay_rate_true_per_s": data["dls_decay_rate_true_per_s"],
                "n_samples": int(data["dls_counts"].size)},
        "optical_bench": bench,
        "sensor": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in asdict(data["sensor"]).items()},
        "units": "SI (metres, radians, watts, volts, seconds); concentration in particles/m^3",
    }
    (args.output / "synthetic_dataset.json").write_text(json.dumps(meta, indent=2) + "\n")

    usable = data["usable_channels"]
    print(f"SYNTHETIC dataset, seed {args.seed}")
    print(f"  truth: dia = {data['truth']['dia']*1e6:.3f} um, npar = {data['truth']['npar']}, "
          f"{scan['n_particles_true']:.0f} particles in the probe volume")
    print(f"  {usable.sum()} usable angle channels of {usable.size}, PGA gain x{scan['gain']}, "
          f"ADC codes {scan['codes'][usable].min():.0f}..{scan['codes'][usable].max():.0f}, "
          f"clipped: {bool(scan['clipped'].any())}")
    print(f"  DLS trace: {data['dls_counts'].size} samples at {1/data['dls_dt_s']:.0f} Hz")
    print(f"  wrote {args.output / 'synthetic_dataset.npz'}")
    print(f"  wrote {args.output / 'synthetic_dataset.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
