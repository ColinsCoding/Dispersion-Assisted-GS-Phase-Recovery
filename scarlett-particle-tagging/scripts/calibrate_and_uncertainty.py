"""Calibration and uncertainty analysis on SYNTHETIC data.

    python scripts/calibrate_and_uncertainty.py [--trials 60] [--seed 0]

Steps
    1. Dark calibration          average dark scans -> offset per channel
    2. Gain calibration          a reference bead of known size and count
                                 reveals the unknown system gain
    3. Sample estimate           diameter, index, concentration with intervals
    4. Diameter sensitivity      dia +/- uncertainty sweep, Cramer-Rao bound
    5. Monte Carlo               repeat with fresh noise: bias, spread, coverage
    6. Motion                    diffusion from the intensity autocorrelation
    7. Telemetry                 pack the report, pass it over a simulated link

Writes JSON and PNG files to results/. Every number is simulated: the "truth"
is a choice made in scarlett.synthetic, and the hidden gain error is put in
on purpose so that the calibration has something to find.
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scarlett import estimator as es, sensor as se, synthetic as sy  # noqa: E402
from scarlett import telemetry as tm, visualization as viz  # noqa: E402

HIDDEN_GAIN_ERROR = 1.07     # unknown to the estimator until step 2 finds it
REL_UNCERTAINTY = 0.02       # model/calibration allowance used in the fit


def measure(truth, bench, sensor, rng, gain=None, ripple=None):
    scan = sy.generate_angular_measurement(truth, bench, sensor, rng=rng, gain=gain,
                                           gain_error=HIDDEN_GAIN_ERROR, ripple=ripple)
    return scan


def dark_subtract(scan, dark_volts, dark_sigma):
    volts = scan["volts"] - dark_volts
    noise = np.sqrt(scan["sigma_volts"] ** 2 + dark_sigma ** 2)
    return volts, es.combined_sigma(volts, noise, REL_UNCERTAINTY)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=60, help="Monte Carlo repetitions")
    parser.add_argument("--seed", type=int, default=sy.DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)
    if args.trials < 5:
        parser.error("--trials must be at least 5")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t_start = time.perf_counter()

    bench, sensor, truth = sy.default_bench(), se.SensorConfig(), sy.SampleTruth()
    usable = bench.usable_channels()
    ripple = sy.channel_ripple(bench.n_channels, REL_UNCERTAINTY)
    print("ALL VALUES BELOW ARE SYNTHETIC (simulated instrument, simulated sample)\n")

    # 1. Dark calibration -------------------------------------------------
    sample_scan = measure(truth, bench, sensor, rng, ripple=ripple)
    gain = sample_scan["gain"]
    darks = [sy.generate_dark_measurement(bench, sensor, gain, rng=rng) for _ in range(16)]
    dark_volts = np.mean([d["volts"] for d in darks], axis=0)
    dark_sigma = darks[0]["sigma_volts"] / np.sqrt(len(darks))
    print(f"1. dark calibration: offset {dark_volts.mean()*1e3:.4f} mV "
          f"(expected {sensor.dark_current*sensor.tia_gain*1e3:.4f} mV), PGA gain x{gain}")

    # 2. Gain calibration with a reference bead ---------------------------
    reference = sy.SampleTruth(dia=sy.REFERENCE_DIA, npar=sy.REFERENCE_NPAR,
                               concentration=truth.concentration)
    ref_scan = measure(reference, bench, sensor, rng, ripple=ripple)
    ref_volts = ref_scan["volts"] - sy.generate_dark_measurement(bench, sensor, ref_scan["gain"])["volts"]
    ref_sigma = es.combined_sigma(ref_volts, ref_scan["sigma_volts"], REL_UNCERTAINTY)
    template = es.particle_template(reference.dia, reference.npar, bench, sensor)
    w = 1 / ref_sigma[usable] ** 2
    n_fit = np.sum(w * ref_volts[usable] * template[usable]) / np.sum(w * template[usable] ** 2)
    n_known = reference.concentration * bench.probe_volume
    system_gain = float(n_fit / n_known)
    system_gain_sigma = float(1 / np.sqrt(np.sum(w * template[usable] ** 2)) / n_known)
    print(f"2. gain calibration: system gain = {system_gain:.4f} +/- {system_gain_sigma:.4f} "
          f"(hidden value {HIDDEN_GAIN_ERROR})")
    # The calibrated instrument model: fold the measured gain into the irradiance.
    cal_bench = replace(bench, incident_irradiance=bench.incident_irradiance * system_gain)

    # 3. Estimate the sample ---------------------------------------------
    diameters = np.arange(3.0e-6, 7.0e-6 + 1e-12, 20e-9)
    lut = es.build_lookup_table(diameters, [truth.npar], cal_bench, sensor)
    lut_uncal = es.build_lookup_table(diameters, [truth.npar], bench, sensor)
    volts, sigma = dark_subtract(sample_scan, dark_volts, dark_sigma)
    ok = ~sample_scan["clipped"]
    est = es.estimate_particle(volts, sigma, lut, cal_bench, sensor, channel_ok=ok)
    est_uncal = es.estimate_particle(volts, sigma, lut_uncal, bench, sensor, channel_ok=ok)
    print(f"3. sample estimate (index fixed at {truth.npar}):")
    print(f"     diameter      {est.dia*1e6:.4f} um, 95 % interval "
          f"[{est.dia_ci[0]*1e6:.4f}, {est.dia_ci[1]*1e6:.4f}]   truth {truth.dia*1e6:.4f}")
    print(f"     concentration {est.concentration:.3e} /m^3, 95 % interval "
          f"[{est.concentration_ci[0]:.3e}, {est.concentration_ci[1]:.3e}]   truth {truth.concentration:.3e}")
    print(f"     without gain calibration the concentration reads {est_uncal.concentration:.3e} "
          f"({100*(est_uncal.concentration/truth.concentration-1):+.1f} %); "
          f"the diameter is unchanged ({est_uncal.dia*1e6:.4f} um)")
    print(f"     chi2/dof = {est.chi2:.1f}/{est.dof}, fit probability {est.fit_probability:.2f}, "
          f"ambiguous: {est.ambiguous}")
    viz.save(viz.plot_estimate(lut, est, truth.dia), "estimate_chi2", out)

    # joint diameter + refractive index
    npars = np.round(np.arange(1.53, 1.6101, 0.01), 2)
    lut2 = es.build_lookup_table(np.arange(3.0e-6, 7.0e-6 + 1e-12, 40e-9), npars, cal_bench, sensor)
    est2 = es.estimate_particle(volts, sigma, lut2, cal_bench, sensor, channel_ok=ok)
    print(f"   joint fit of diameter and index: dia = {est2.dia*1e6:.4f} um "
          f"[{est2.dia_ci[0]*1e6:.4f}, {est2.dia_ci[1]*1e6:.4f}], npar = {est2.npar:.2f} "
          f"[{est2.npar_ci[0]:.3f}, {est2.npar_ci[1]:.3f}] (grid step 0.01), ambiguous: {est2.ambiguous}")

    # 4. Diameter sensitivity --------------------------------------------
    nominal = es.particle_template(truth.dia, truth.npar, cal_bench, sensor)[usable]
    sweep = {}
    for rel in (0.001, 0.01, 0.02, 0.05):
        changed = es.particle_template(truth.dia * (1 + rel), truth.npar, cal_bench, sensor)[usable]
        sweep[rel] = float(np.sqrt(np.mean((changed / nominal - 1) ** 2)))
    crb = es.diameter_cramer_rao(truth.dia, truth.npar, est.n_particles, sigma, cal_bench, sensor)
    print("4. diameter sensitivity (RMS relative change of the pattern over the usable channels):")
    for rel, change in sweep.items():
        print(f"     dia + {100*rel:4.1f} %  ->  pattern changes by {100*change:6.1f} %")
    print(f"     Cramer-Rao bound on diameter: {crb*1e9:.2f} nm (1 sigma) "
          f"= {100*crb/truth.dia:.3f} % of the diameter")

    # 5. Monte Carlo -------------------------------------------------------
    print(f"5. Monte Carlo, {args.trials} trials (fresh noise and fresh channel ripple each time) ...")
    estimates, covered, widths = [], 0, []
    for trial in range(args.trials):
        trial_ripple = sy.channel_ripple(bench.n_channels, REL_UNCERTAINTY, seed=10_000 + trial)
        scan = measure(truth, bench, sensor, rng, gain=gain, ripple=trial_ripple)
        v, s = dark_subtract(scan, dark_volts, dark_sigma)
        e = es.estimate_particle(v, s, lut, cal_bench, sensor, channel_ok=~scan["clipped"])
        estimates.append((e.dia, e.concentration))
        covered += e.dia_ci[0] <= truth.dia <= e.dia_ci[1]
        widths.append(e.dia_ci[1] - e.dia_ci[0])
    estimates = np.array(estimates)
    dia_bias = float(estimates[:, 0].mean() - truth.dia)
    dia_std = float(estimates[:, 0].std(ddof=1))
    coverage = covered / args.trials
    print(f"     diameter: bias {dia_bias*1e9:+.2f} nm, spread {dia_std*1e9:.2f} nm (1 sigma), "
          f"Cramer-Rao {crb*1e9:.2f} nm")
    print(f"     95 % interval contained the truth in {100*coverage:.0f} % of trials "
          f"(mean width {np.mean(widths)*1e9:.1f} nm)")
    print(f"     concentration: bias {100*(estimates[:, 1].mean()/truth.concentration-1):+.2f} %, "
          f"spread {100*estimates[:, 1].std(ddof=1)/truth.concentration:.2f} %")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    axes[0].hist((estimates[:, 0] - truth.dia) * 1e9, bins=15)
    axes[0].axvline(0, color="k", ls=":")
    axes[0].set(xlabel="diameter error (nm)", ylabel="trials",
                title=f"Monte Carlo, {args.trials} synthetic trials")
    axes[1].hist(100 * (estimates[:, 1] / truth.concentration - 1), bins=15)
    axes[1].axvline(0, color="k", ls=":")
    axes[1].set(xlabel="concentration error (%)", ylabel="trials", title="Concentration")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    viz.save(fig, "uncertainty_monte_carlo", out)
    plt.close("all")

    # 6. Motion ------------------------------------------------------------
    dls_dt, dls_theta = 5e-3, np.pi / 2
    counts, gamma_true = sy.generate_dls_trace(truth, bench, 120_000, dls_dt, theta=dls_theta, rng=rng)
    motion = es.estimate_motion(counts, dls_dt, bench.lam_center, bench.nmed, dls_theta, max_lag=120,
                                temperature=truth.temperature, viscosity=truth.viscosity)
    print(f"6. motion: decay rate {motion['decay_rate']:.2f} 1/s "
          f"[{motion['decay_rate_ci'][0]:.2f}, {motion['decay_rate_ci'][1]:.2f}] (truth {gamma_true:.2f})")
    print(f"     diffusion {motion['diffusion']:.3e} m^2/s -> hydrodynamic diameter "
          f"{motion['hydrodynamic_dia']*1e6:.2f} um "
          f"[{motion['hydrodynamic_dia_ci'][0]*1e6:.2f}, {motion['hydrodynamic_dia_ci'][1]*1e6:.2f}] "
          f"(truth {truth.dia*1e6:.2f})")
    viz.save(viz.plot_motion(counts, dls_dt, 120, gamma_true), "motion_autocorrelation", out)
    plt.close("all")

    # 7. Telemetry -----------------------------------------------------------
    flags = (tm.FLAG_AMBIGUOUS if est.ambiguous else 0) | (tm.FLAG_CLIPPED if (~ok).any() else 0) \
        | (tm.FLAG_POOR_FIT if est.fit_probability < 0.001 else 0)
    report = tm.TagReport(tag_id=truth.tag_id, timestamp_ms=1_000, dia=est.dia, dia_ci=est.dia_ci,
                          npar=est.npar, confidence_level=est.confidence_level,
                          concentration=est.concentration, diffusion=motion["diffusion"], flags=flags)
    packet = tm.encode_packet(report)
    link = tm.link_budget(distance=5.0)
    received = tm.transmit(packet, link["bit_error_rate"], rng)
    decoded = tm.decode_packet(received)
    print(f"7. telemetry: {len(packet)}-byte packet, tag 0x{report.tag_id:08X}; simulated 5 m link: "
          f"Eb/N0 = {link['ebn0_db']:.1f} dB, packet error rate {link['packet_error_rate']:.1e}")
    print(f"     decoded: dia = {decoded.dia*1e6:.3f} um "
          f"[{decoded.dia_ci[0]*1e6:.3f}, {decoded.dia_ci[1]*1e6:.3f}], CRC ok")

    summary = {
        "synthetic": True,
        "warning": "Simulated data. Not an experimental result.",
        "seed": args.seed,
        "calibration": {"pga_gain": gain, "dark_offset_volts_mean": float(dark_volts.mean()),
                        "system_gain": system_gain, "system_gain_sigma": system_gain_sigma,
                        "hidden_gain_error_in_simulation": HIDDEN_GAIN_ERROR,
                        "rel_model_uncertainty": REL_UNCERTAINTY},
        "truth": {"dia_m": truth.dia, "npar": truth.npar, "concentration_per_m3": truth.concentration},
        "estimate": {"dia_m": est.dia, "dia_ci_m": est.dia_ci, "npar": est.npar,
                     "concentration_per_m3": est.concentration,
                     "concentration_ci_per_m3": est.concentration_ci,
                     "chi2": est.chi2, "dof": est.dof, "fit_probability": est.fit_probability,
                     "ambiguous": est.ambiguous, "confidence_level": est.confidence_level},
        "joint_fit": {"dia_m": est2.dia, "dia_ci_m": est2.dia_ci, "npar": est2.npar,
                      "npar_ci": est2.npar_ci, "ambiguous": est2.ambiguous},
        "sensitivity": {"rms_pattern_change_per_rel_dia_change": {str(k): v for k, v in sweep.items()},
                        "cramer_rao_dia_m": crb},
        "monte_carlo": {"trials": args.trials, "dia_bias_m": dia_bias, "dia_std_m": dia_std,
                        "ci_coverage": coverage, "ci_mean_width_m": float(np.mean(widths))},
        "motion": {**motion, "decay_rate_true_per_s": gamma_true},
        "telemetry": {"packet_bytes": len(packet), "packet_hex": packet.hex(), "link": link},
    }
    (out / "uncertainty_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {out / 'uncertainty_summary.json'} and 3 figures "
          f"({time.perf_counter() - t_start:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
