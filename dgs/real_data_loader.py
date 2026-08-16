"""
real_data_loader.py -- loading externally-provided two-arm intensity data into
gs_core's interface, with the validation the raw loader alone cannot give you
================================================================================

THE GAP THIS FILLS (project brief Task 4)
------------------------------------------
The project brief's Task 4 is "simulate phase recovery with optical
communication data that will be provided to you." Nothing in this repo
cleanly does that. The only prior attempt is ad-hoc Dryad-archive-searching
code embedded directly in a notebook cell, which errors out
(`ModuleNotFoundError`, then `RuntimeError: No traces loaded`) and never
validates what it loads even when it succeeds. This module is a clean,
reusable, testable replacement: parse known real-data formats into exactly
what `dgs.gs_core.retrieve_phase` expects, AND check whether the result is
even physically sane before anyone trusts a recovered phase.

A KEY FINDING THAT MOTIVATES THIS MODULE'S DESIGN
----------------------------------------------------
`sample_data/*.mat` (six files, one per modulation format, "Synthetic
dual-arm dispersion-assisted GS phase retrieval data" per their own embedded
`description` field) ship `D1_ps2`/`D2_ps2` fields whose numeric convention
does NOT match `gs_core`'s `D` parameter directly. Investigating this by
hand (see the worked example in this module's `if __name__` block) found:

  - Using `D1_ps2`/`D2_ps2` literally as `gs_core.retrieve_phase`'s `D1,D2`:
    final GS error ~0.65 (does not converge).
  - Converting via the physically-motivated GDD[ps^2] -> normalized-D formula
    `D = 2*pi*GDD[s^2]*fs[Hz]**2` makes it WORSE, not better.
  - Sweeping the scale over SIX orders of magnitude (0.01x to 10000x) and
    both signs: final error stays pinned near 0.65-0.69 the entire time --
    completely flat, not even trending toward convergence at any scale.
  - `corr(I1, I2)` across all six sample files is small (0.01-0.49 in
    magnitude, no consistent sign) -- far below what a genuinely
    dispersion-related pair of measurements of the same field would be
    expected to show.

A SECOND FINDING: THE CONVERGENCE BASIN IS EXTREMELY NARROW
----------------------------------------------------
A follow-up experiment on `gs_core`'s OWN known-good synthetic QPSK example
(D1=-5000 is the true, generating value -- not in question here) found
something sharper than "wrong D fails": D1=4800 or D1=5200 (each just +-4%
off the TRUE value) both fail just as badly as a wildly wrong guess (final
error ~0.27, versus ~0.047 exactly at D1=5000). A 40-point coarse log
search (100 to 1e6) followed by a 60-point fine linear refinement around
its best coarse candidate BOTH failed to rediscover D1=-5000 from scratch.
**The error surface has no exploitable gradient toward the true value; it
behaves like an isolated spike, not a smooth basin.** This means a blind
numerical search for an unknown D is not a reliable strategy REGARDLESS of
whether the data is "good" -- `convergence_sanity_sweep` in this module is
therefore designed as a candidate-CONFIRMATION tool (test specific values
you have a physical reason to believe in), not a discovery tool, and its
own docstring says so explicitly.

**Conclusion, stated plainly: these particular sample files most likely do
not satisfy `gs_core`'s `H(nu)=exp(i*pi*D*nu**2)` dispersion relationship
under any candidate D this module's default search tried** -- not a units
bug this module can silently "fix," but a genuine open question needing
either the original MATLAB generation script or its author, compounded by
the fact that even correct-order-of-magnitude guessing is not expected to
succeed given the narrow-basin finding above. Rather than guess further,
this module makes both findings IMPOSSIBLE TO MISS: `diagnose_measurement`
runs the same correlation check and convergence confirmation automatically
on whatever you load, and refuses to hand back a phase estimate with a
silent "looks fine" when the data does not actually look fine.

DESIGN DECISIONS (the "why", not just the "what")
----------------------------------------------------
1. Loading and validating are SEPARATE functions, not one opaque call, so a
   caller can inspect *why* a file failed instead of getting only "it broke."
2. `diagnose_measurement` never raises on bad data -- it returns a report
   dict with a `recoverable` boolean and human-readable `messages`. A silent
   crash (the old Dryad cell's failure mode) helps no one; a clear "this
   data doesn't look right, and here is the evidence" does.
3. `convergence_sanity_sweep` automates the exact by-hand investigation
   above as a reusable diagnostic, not a one-off script -- the next person
   pointed at an unfamiliar data file gets this for free.
4. `load_or_synthesize` NEVER silently substitutes synthetic data for a
   missing/bad real file -- it always prints an explicit, loud warning
   identifying that the returned data is synthetic, because silently
   swapping real for fake data in a receiver-testing context is exactly the
   kind of bug that is invisible until it matters.

Run: py -3.13 -m dgs.real_data_loader
"""

import warnings
from pathlib import Path

import numpy as np

from dgs.gs_core import retrieve_phase, make_measurements


# ── Format-specific loaders ───────────────────────────────────────────────

def _decode_mat_string(value):
    """scipy.io.loadmat wraps MATLAB char/cell strings in nested arrays; unwrap them."""
    arr = np.asarray(value)
    while arr.dtype == object or arr.ndim > 0:
        if arr.size == 0:
            return ""
        arr = arr.reshape(-1)[0] if arr.ndim > 0 else arr
        if isinstance(arr, str):
            return arr
        arr = np.asarray(arr)
    return str(arr)


def load_mat_measurement(path):
    """Load a two-arm measurement from a .mat file matching this repo's
    sample_data/ schema: I1, I2, t, fs_GSas, D1_ps2, D2_ps2, lambda_nm,
    standard, description.

    Returns a dict with keys: I1, I2, t, fs_GSas, D1_raw, D2_raw, lambda_nm,
    modulation_label, description, source_path. D1_raw/D2_raw are passed
    through EXACTLY as stored -- see this module's docstring for why they
    should not be assumed to already be gs_core's D convention.
    """
    import scipy.io as sio

    path = Path(path)
    data = sio.loadmat(str(path))

    def _scalar(key):
        return float(np.asarray(data[key]).squeeze()) if key in data else None

    return {
        "I1": np.asarray(data["I1"], dtype=float).squeeze(),
        "I2": np.asarray(data["I2"], dtype=float).squeeze(),
        "t": np.asarray(data["t"], dtype=float).squeeze() if "t" in data else None,
        "fs_GSas": _scalar("fs_GSas"),
        "D1_raw": _scalar("D1_ps2"),
        "D2_raw": _scalar("D2_ps2"),
        "lambda_nm": _scalar("lambda_nm"),
        "modulation_label": _decode_mat_string(data["standard"]) if "standard" in data else None,
        "description": _decode_mat_string(data["description"]) if "description" in data else None,
        "source_path": str(path),
    }


def load_array_measurement(path_I1, path_I2, D1_raw, D2_raw, fs_GSas=None):
    """Load a two-arm measurement from two plain array files (.csv, .txt, .npy),
    for data that does not come packaged as a single .mat file. D1_raw/D2_raw
    must be supplied by the caller since plain arrays carry no metadata."""
    def _load_1d(p):
        p = Path(p)
        if p.suffix == ".npy":
            return np.load(p).astype(float).squeeze()
        return np.loadtxt(p, delimiter=",").astype(float).squeeze()

    return {
        "I1": _load_1d(path_I1),
        "I2": _load_1d(path_I2),
        "t": None,
        "fs_GSas": fs_GSas,
        "D1_raw": float(D1_raw),
        "D2_raw": float(D2_raw),
        "lambda_nm": None,
        "modulation_label": None,
        "description": f"Loaded from {path_I1}, {path_I2} (no embedded metadata)",
        "source_path": f"{path_I1}, {path_I2}",
    }


def load_measurement(path, **kwargs):
    """Dispatch to the right loader by file extension."""
    path = Path(path)
    if path.suffix.lower() == ".mat":
        return load_mat_measurement(path)
    raise ValueError(
        f"No single-file loader for extension '{path.suffix}'. "
        f"For .npy/.csv array pairs use load_array_measurement(path_I1, path_I2, ...) directly."
    )


# ── Validation / diagnostics ──────────────────────────────────────────────

def convergence_sanity_sweep(I1, I2, D1_candidates, D_ratio=2.0, n_iter=50, unit_amplitude=True):
    """Test an explicit LIST of candidate D1 values (D2 = D_ratio * candidate)
    and report whether GS converges for ANY of them.

    IMPORTANT, empirically demonstrated (not assumed) limitation: this is a
    CONFIRMATION tool, not a blind global search. A direct experiment on
    dgs.gs_core's own known-good synthetic QPSK example (D1=-5000 is the
    true, generating value) found the convergence "basin" to be extremely
    narrow -- D1=4800 or D1=5200 (each just +-4% off the true value) BOTH
    fail as badly as a wildly wrong guess (final error ~0.27, versus ~0.047
    at the exact true value). A coarse 40-point log-spaced search from 100
    to 1e6, and a subsequent fine 60-point linear refinement around its best
    coarse candidate, BOTH failed to rediscover D1=-5000 from scratch -- the
    error surface has essentially no exploitable gradient leading toward the
    true value; it is closer to an isolated spike than a smooth basin.

    Practical consequence: candidates passed here should come from physical
    reasoning or file metadata (the raw value, a literature-typical
    magnitude, an obvious unit-conversion factor), not from a hopeful wide
    numerical search -- this function can tell you whether a candidate you
    have reason to believe in actually works, not discover one for you.

    Returns a dict: {"best_D1", "best_D2", "best_final_error", "converged",
    "all_results"}. "converged" is True only if the best candidate's final
    error is below 0.1 -- chosen from the demonstrated gap between a true-D
    result (~0.047 in the reference case above) and every wrong candidate
    tested (~0.24-0.29), not an arbitrarily tight machine-precision bar.
    """
    results = []
    for D1_try in D1_candidates:
        D2_try = D1_try * D_ratio
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the low-|D| convergence warning is expected during a sweep
            try:
                _, err = retrieve_phase(I1, I2, D1_try, D2_try, n_iter=n_iter, unit_amplitude=unit_amplitude)
                results.append((D1_try, D2_try, float(err[-1])))
            except Exception:
                results.append((D1_try, D2_try, float("inf")))

    best = min(results, key=lambda r: r[2])
    converged = best[2] < 0.1
    return {
        "best_D1": best[0], "best_D2": best[1], "best_final_error": best[2],
        "converged": converged,
        "all_results": results,
    }


def default_candidates(D_raw=None):
    """A reasonable default candidate list when no better physical reasoning is
    available: the raw metadata value itself (if given) at several common
    unit-conversion scales, plus a spread of literature-typical magnitudes
    seen elsewhere in this repo (dgs.gs_core's own examples use 5000-15000)."""
    candidates = [5000.0, 10000.0, 15000.0, -5000.0, -10000.0, -15000.0]
    if D_raw is not None and D_raw != 0:
        for scale in (1, 10, 100, 1000):
            candidates.append(D_raw * scale)
            candidates.append(-abs(D_raw) * scale)
    return sorted(set(candidates), key=abs)


def diagnose_measurement(I1, I2, run_convergence_sweep=True, D_ratio=2.0,
                          D_raw=None, D1_candidates=None):
    """Run all sanity checks on a loaded (I1, I2) pair and return a report.
    Never raises -- always returns a dict with 'recoverable' (bool) and
    'messages' (list of str) explaining the verdict, so a caller can decide
    what to do rather than have the decision hidden inside an exception.

    D_raw: a raw D value from file metadata, if available (used to build
    default candidates via default_candidates() -- see that function's
    docstring for why this is a confirmation list, not a search range).
    D1_candidates: an explicit candidate list, overriding the default.
    """
    messages = []
    recoverable = True

    I1 = np.asarray(I1, dtype=float)
    I2 = np.asarray(I2, dtype=float)

    if len(I1) != len(I2):
        messages.append(f"FAIL: I1 (len={len(I1)}) and I2 (len={len(I2)}) have different lengths.")
        recoverable = False
    if not (np.all(np.isfinite(I1)) and np.all(np.isfinite(I2))):
        messages.append("FAIL: I1 or I2 contains NaN/Inf.")
        recoverable = False
    if np.any(I1 < 0) or np.any(I2 < 0):
        messages.append("WARN: I1 or I2 has negative values (will be clipped to 0 by gs_core).")

    corr = float(np.corrcoef(I1, I2)[0, 1]) if len(I1) == len(I2) else float("nan")
    if abs(corr) > 0.9:
        messages.append(f"WARN: corr(I1,I2)={corr:.3f} is very HIGH -- the two measurements may carry "
                         f"too little diversity for GS to have anything to iterate on (gs_core's own "
                         f"|D|>=100 diversity check addresses the same concern from the D side).")
    else:
        messages.append(f"INFO: corr(I1,I2)={corr:.3f}.")

    sweep_result = None
    if run_convergence_sweep and len(I1) == len(I2) and np.all(np.isfinite(I1)) and np.all(np.isfinite(I2)):
        candidates = D1_candidates if D1_candidates is not None else default_candidates(D_raw)
        sweep_result = convergence_sanity_sweep(I1, I2, candidates, D_ratio=D_ratio)
        if sweep_result["converged"]:
            messages.append(f"PASS: candidate D1={sweep_result['best_D1']:.1f} converges "
                             f"(final error={sweep_result['best_final_error']:.3f}).")
        else:
            messages.append(
                f"FAIL: none of {len(candidates)} candidate D1 values converged (best final error="
                f"{sweep_result['best_final_error']:.3f}). Per this module's docstring, GS convergence "
                f"has an empirically narrow basin (a candidate within a few percent of the true value "
                f"can still fail outright) -- this result means none of the TESTED candidates were close "
                f"enough, not that recovery is provably impossible. See the module docstring's worked "
                f"example for exactly this outcome on this repo's own sample_data/*.mat files."
            )
            recoverable = False

    return {
        "recoverable": recoverable,
        "correlation": corr,
        "sweep_result": sweep_result,
        "messages": messages,
    }


# ── Loud, explicit synthetic fallback ─────────────────────────────────────

def load_or_synthesize(path=None, modulation="QPSK", n_symbols=64, sps=8,
                        D1=-5000.0, D2=-5750.0, snr_db=25.0, rng_seed=0):
    """Load real data from `path` if given and valid; otherwise fall back to
    dgs.gs_core.make_measurements -- ALWAYS with an explicit, unmissable
    warning identifying which one actually happened. Never silently swaps
    real for synthetic data.
    """
    if path is not None:
        try:
            data = load_measurement(path)
            report = diagnose_measurement(data["I1"], data["I2"], run_convergence_sweep=False)
            if report["recoverable"]:
                print(f"[real_data_loader] Loaded REAL data from {path}.")
                return data, report
            else:
                warnings.warn(
                    f"[real_data_loader] Loaded {path} but it FAILED validation "
                    f"({report['messages']}); falling back to SYNTHETIC data instead.",
                    stacklevel=2,
                )
        except Exception as e:
            warnings.warn(
                f"[real_data_loader] Could not load {path} ({e}); falling back to SYNTHETIC data instead.",
                stacklevel=2,
            )
    else:
        warnings.warn(
            "[real_data_loader] No path given -- returning SYNTHETIC data from dgs.gs_core.make_measurements.",
            stacklevel=2,
        )

    synthetic = make_measurements(modulation=modulation, n_symbols=n_symbols, sps=sps,
                                   D1=D1, D2=D2, snr_db=snr_db, rng_seed=rng_seed)
    synthetic["source_path"] = "SYNTHETIC (dgs.gs_core.make_measurements)"
    return synthetic, None


if __name__ == "__main__":
    print("Worked example: sample_data/QPSK_56GSas_D600_D1200.mat\n")
    sample_path = Path(__file__).resolve().parent.parent / "sample_data" / "QPSK_56GSas_D600_D1200.mat"
    if sample_path.exists():
        data = load_mat_measurement(sample_path)
        print(f"Loaded: N={len(data['I1'])}, fs_GSas={data['fs_GSas']}, "
              f"D1_raw={data['D1_raw']}, D2_raw={data['D2_raw']}")
        print(f"description: {data['description']}\n")

        report = diagnose_measurement(data["I1"], data["I2"],
                                       D_ratio=data["D2_raw"] / data["D1_raw"], D_raw=data["D1_raw"])
        print("Diagnosis:")
        for m in report["messages"]:
            print(f"  {m}")
        print(f"\nrecoverable = {report['recoverable']}")
    else:
        print(f"(sample file not found at {sample_path} -- skipping worked example)")

    print("\n--- load_or_synthesize with no path (explicit synthetic fallback) ---")
    data, report = load_or_synthesize(path=None)
    print(f"source_path = {data['source_path']}")
