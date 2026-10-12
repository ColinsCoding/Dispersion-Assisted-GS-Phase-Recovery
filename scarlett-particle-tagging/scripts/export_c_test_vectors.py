"""Write test vectors for the C estimator and the answers the Python model gives.

    python scripts/export_c_test_vectors.py

Writes to results/c_vectors/
    estimate.txt   measurement + Q15 template table (input of `scarlett_estimator estimate`)
    acf.txt        integer time series               (input of `scarlett_estimator acf`)
    expected.json  what scarlett.asic computes for the same inputs

All values are synthetic.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scarlett import asic, estimator as es, sensor as se, synthetic as sy  # noqa: E402


def main():
    out = ROOT / "results" / "c_vectors"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(sy.DEFAULT_SEED)
    bench, sensor, truth = sy.default_bench(), se.SensorConfig(), sy.SampleTruth()

    lut = es.build_lookup_table(np.arange(4.0e-6, 6.0e-6 + 1e-12, 20e-9), [truth.npar], bench, sensor)
    scan = sy.generate_angular_measurement(truth, bench, sensor, rng=rng)
    dark = sy.generate_dark_measurement(bench, sensor, scan["gain"], rng=rng)
    memory = asic.build_calibration_memory(lut, dark["code_sums"], scan["gain"], sensor)
    meas = asic.subtract_dark(scan["code_sums"], memory)
    volts = scan["volts"] - dark["volts"]
    sigma = es.combined_sigma(volts, np.hypot(scan["sigma_volts"], dark["sigma_volts"]))
    sigma_counts = float(np.sqrt(np.mean(sigma[lut.mask] ** 2))
                         * scan["gain"] * sensor.samples_per_channel / sensor.lsb)

    asic.write_estimate_vectors(out / "estimate.txt", meas, memory.templates_q15, sigma_counts)
    fixed = asic.fixed_point_estimate(meas, memory.templates_q15, sigma_counts)

    counts, _ = sy.generate_dls_trace(truth, bench, 4000, 5e-3, rng=rng)
    asic.write_acf_vectors(out / "acf.txt", counts, 32)
    acf = asic.integer_autocorrelation(counts, 32)

    best = fixed["best_index"]
    expected = {
        "synthetic": True,
        "estimate_stdout": f"{best} {fixed['ci_lo_index']} {fixed['ci_hi_index']} "
                           f"{int(fixed['accepted'].sum())} {int(fixed['ambiguous'])} "
                           f"{int(fixed['corr'][best])} {fixed['energy']} {fixed['chi2_min']!r}",
        "best_diameter_m": float(lut.dia[best]),
        "truth_diameter_m": truth.dia,
        "interval_m": [float(lut.dia[fixed["ci_lo_index"]]), float(lut.dia[fixed["ci_hi_index"]])],
        "acf": [int(a) for a in acf],
    }
    (out / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
    print(f"wrote {out / 'estimate.txt'}, {out / 'acf.txt'}, {out / 'expected.json'}")
    print(f"Python fixed-point model: best diameter {lut.dia[best]*1e6:.3f} um "
          f"(synthetic truth {truth.dia*1e6:.3f} um)")
    print("expected C output:", expected["estimate_stdout"])


if __name__ == "__main__":
    main()
