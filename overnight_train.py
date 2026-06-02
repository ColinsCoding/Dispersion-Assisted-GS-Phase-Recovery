"""
overnight_train.py — runs 11 PM to 6 AM (or until killed)
More time = more samples = better coverage of parameter space.

What it does:
  1. Generates GS training data with multi-restart (finds global minimum)
  2. Sweeps full (D1, D2, SNR, modulation, n_symbols) parameter space
  3. Saves checkpoints every 1000 samples — safe to Ctrl+C anytime
  4. Writes a summary report when done

Usage:
    python overnight_train.py                    # runs until 6:00 AM
    python overnight_train.py --hours 2          # runs for 2 hours
    python overnight_train.py --hours 0.1        # 6-minute test run

Output:
    data/overnight/checkpoint_NNNN.npz   — rolling checkpoints
    data/overnight/final.npz             — merged dataset
    data/overnight/report.txt            — convergence statistics
"""

import argparse
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from gs_core import make_measurements, retrieve_phase

# ── Parameter space ───────────────────────────────────────────────────────────
MODULATIONS   = ['QPSK', 'DPSK', 'STEAM', 'Soliton', 'OOK', 'PAM4']
D1_VALUES     = [-4000, -5000, -6000, -7000, -8000, -10000]
D_RATIOS      = [1.10, 1.15, 1.20, 1.33, 1.50, 2.00]
SNR_LEVELS    = [15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
N_SYMBOLS     = [32, 48, 64, 96, 128]
N_RESTARTS    = 3      # multi-start GS — keep best result
N_ITER_BASE   = 100    # base iterations per restart
N_OUT         = 512    # output signal length (all resampled to this)
CHECKPOINT_N  = 1000   # save every N samples


def _resample(arr, n):
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(arr)), arr)


def gs_multi_restart(I1, I2, D1, D2, n_iter, n_restarts, unit_amplitude):
    """Run GS n_restarts times with different random inits, return best."""
    best_phi, best_err = None, float('inf')
    for seed in range(n_restarts):
        rng = np.random.default_rng(seed * 1000 + int(abs(D1)))
        # perturb initial field with random phase noise
        N = len(I1)
        noise = rng.uniform(0, 2 * np.pi, N)
        I1_p = I1 * (1 + 0.01 * rng.standard_normal(N))
        I1_p = np.maximum(I1_p, 0)
        phi, errs = retrieve_phase(I1_p, I2, D1, D2,
                                   n_iter=n_iter,
                                   unit_amplitude=unit_amplitude)
        if errs[-1] < best_err:
            best_err = errs[-1]
            best_phi = phi
    return best_phi, best_err


def generate_one(rng, global_seed):
    mod   = rng.choice(MODULATIONS)
    D1    = float(rng.choice(D1_VALUES))
    ratio = float(rng.choice(D_RATIOS))
    D2    = D1 * ratio
    snr   = float(rng.choice(SNR_LEVELS))
    n_sym = int(rng.choice(N_SYMBOLS))

    d = make_measurements(modulation=mod, n_symbols=n_sym, sps=8,
                          D1=D1, D2=D2, snr_db=snr,
                          rng_seed=int(global_seed))

    phi_rec, gs_err = gs_multi_restart(
        d['I1'], d['I2'], D1, D2,
        n_iter=N_ITER_BASE,
        n_restarts=N_RESTARTS,
        unit_amplitude=d['unit_amplitude'],
    )

    # Phase RMSE with global offset correction
    off   = np.angle(np.mean(np.exp(1j * (d['phi_true'] - phi_rec))))
    delta = np.angle(np.exp(1j * (phi_rec - d['phi_true'] + off)))
    rmse  = float(np.sqrt(np.mean(delta**2)))

    return {
        'I1':       _resample(d['I1'],       N_OUT).astype(np.float32),
        'I2':       _resample(d['I2'],       N_OUT).astype(np.float32),
        'phi_true': _resample(d['phi_true'], N_OUT).astype(np.float32),
        'phi_rec':  _resample(phi_rec,       N_OUT).astype(np.float32),
        'rmse':     np.float32(rmse),
        'gs_err':   np.float32(gs_err),
        'converged':np.bool_(rmse < 0.15),
        'D1':       np.float32(D1),
        'D2':       np.float32(D2),
        'snr_db':   np.float32(snr),
        'mod':      mod,
        'n_symbols':np.int32(n_sym),
    }


def save_checkpoint(records, path, idx):
    p = path / f'checkpoint_{idx:06d}.npz'
    keys = ['I1', 'I2', 'phi_true', 'phi_rec', 'rmse',
            'gs_err', 'converged', 'D1', 'D2', 'snr_db', 'n_symbols']
    arrays = {k: np.stack([r[k] for r in records]) for k in keys}
    arrays['modulation'] = np.array([r['mod'] for r in records])
    np.savez_compressed(str(p), **arrays)
    return p


def write_report(all_records, path, elapsed):
    conv   = [r for r in all_records if r['converged']]
    n      = len(all_records)
    n_conv = len(conv)

    from collections import Counter
    mod_counts = Counter(r['mod'] for r in all_records)
    conv_by_mod = Counter(r['mod'] for r in conv)

    lines = [
        f"overnight_train.py — report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Elapsed:   {elapsed/3600:.2f} hours",
        f"",
        f"Total samples:     {n}",
        f"Converged:         {n_conv} ({n_conv/n*100:.1f}%)",
        f"Rate:              {n/elapsed:.1f} samples/s",
        f"",
        f"Convergence by modulation:",
    ]
    for mod in MODULATIONS:
        total = mod_counts[mod]
        conv_ = conv_by_mod[mod]
        if total:
            lines.append(f"  {mod:<10} {conv_:>5}/{total:<5}  ({conv_/total*100:.1f}%)")

    if conv:
        rmses = [r['rmse'] for r in conv]
        lines += [
            f"",
            f"RMSE (converged only):",
            f"  mean = {np.mean(rmses):.4f} rad",
            f"  std  = {np.std(rmses):.4f} rad",
            f"  min  = {np.min(rmses):.4f} rad",
            f"  max  = {np.max(rmses):.4f} rad",
        ]

    report_path = path / 'report.txt'
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    for line in lines:
        print(line)


def merge_checkpoints(path):
    files = sorted(path.glob('checkpoint_*.npz'))
    if not files:
        return
    all_arrays = {}
    for f in files:
        d = np.load(str(f), allow_pickle=True)
        for k in d.files:
            all_arrays.setdefault(k, []).append(d[k])
    merged = {k: np.concatenate(v) for k, v in all_arrays.items()}
    out = path / 'final.npz'
    np.savez_compressed(str(out), **merged)
    size_mb = out.stat().st_size / 1e6
    print(f"\nMerged {len(files)} checkpoints -> {out}  ({size_mb:.1f} MB)")
    print(f"Final dataset: {merged['I1'].shape[0]} samples")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hours', type=float, default=7.0,
                    help='How many hours to run (default 7 = 11pm to 6am)')
    args = ap.parse_args()

    out_dir = Path('data/overnight')
    out_dir.mkdir(parents=True, exist_ok=True)

    deadline = datetime.now() + timedelta(hours=args.hours)
    print(f"overnight_train.py")
    print(f"Start:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Stop:     {deadline.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Restarts: {N_RESTARTS}x per sample  |  iters: {N_ITER_BASE}")
    print(f"Output:   {out_dir}/")
    print(f"Ctrl+C to stop early — checkpoints saved every {CHECKPOINT_N} samples\n")

    rng = np.random.default_rng(int(time.time()))
    all_records   = []
    batch_records = []
    total = 0
    conv  = 0
    checkpoint_idx = 0
    t0 = time.perf_counter()

    try:
        while datetime.now() < deadline:
            rec = generate_one(rng, global_seed=total)
            all_records.append(rec)
            batch_records.append(rec)
            total += 1
            if rec['converged']:
                conv += 1

            # Progress every 100 samples
            if total % 100 == 0:
                elapsed  = time.perf_counter() - t0
                rate     = total / elapsed
                eta_min  = (deadline - datetime.now()).total_seconds() / 60
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                      f"n={total:>7}  "
                      f"conv={conv/total*100:>5.1f}%  "
                      f"rate={rate:>5.0f}/s  "
                      f"ETA {eta_min:.0f} min")

            # Checkpoint
            if len(batch_records) >= CHECKPOINT_N:
                p = save_checkpoint(batch_records, out_dir, checkpoint_idx)
                print(f"  >> Checkpoint saved: {p.name}  "
                      f"({CHECKPOINT_N} samples, {conv/total*100:.1f}% converged)")
                batch_records = []
                checkpoint_idx += 1

    except KeyboardInterrupt:
        print("\nStopped by user.")

    # Save remaining batch
    if batch_records:
        save_checkpoint(batch_records, out_dir, checkpoint_idx)

    elapsed = time.perf_counter() - t0
    print(f"\nTotal: {total} samples in {elapsed/3600:.2f} hours")

    merge_checkpoints(out_dir)
    write_report(all_records, out_dir, elapsed)


if __name__ == '__main__':
    main()
