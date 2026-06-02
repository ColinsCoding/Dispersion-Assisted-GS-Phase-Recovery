"""
Generate a large GS phase retrieval training dataset overnight.
Saves to data/training_data.npz

Usage:
    python gen_training_data.py              # default 10000 samples
    python gen_training_data.py --n 50000    # larger dataset
    python gen_training_data.py --n 1000 --preview   # quick test + plot
"""
import argparse
import numpy as np
import time
from pathlib import Path
from gs_core import make_measurements, retrieve_phase

MODULATIONS  = ['QPSK', 'DPSK', 'STEAM', 'Soliton']
D_PAIRS      = [(-5000, -5750), (-6000, -7200), (-7000, -8400)]
SNR_LEVELS   = [20.0, 25.0, 30.0, 35.0, 40.0]
N_SYMBOLS_RANGE = [32, 48, 64]
N_OUT        = 512   # all signals resampled to this length


def _resample(arr, n_out):
    x_in  = np.linspace(0, 1, len(arr))
    x_out = np.linspace(0, 1, n_out)
    return np.interp(x_out, x_in, arr)


def generate_one(rng_seed, rng):
    mod    = rng.choice(MODULATIONS)
    D1, D2 = D_PAIRS[rng.integers(len(D_PAIRS))]
    snr    = float(rng.choice(SNR_LEVELS))
    n_sym  = int(rng.choice(N_SYMBOLS_RANGE))

    d = make_measurements(modulation=mod, n_symbols=n_sym, sps=8,
                          D1=D1, D2=D2, snr_db=snr, rng_seed=int(rng_seed))
    phi_rec, errs = retrieve_phase(d['I1'], d['I2'], D1, D2,
                                   n_iter=50, unit_amplitude=d['unit_amplitude'])
    off   = np.angle(np.mean(np.exp(1j * (d['phi_true'] - phi_rec))))
    delta = np.angle(np.exp(1j * (phi_rec - d['phi_true'] + off)))
    rmse  = float(np.sqrt(np.mean(delta**2)))
    converged = rmse < 0.15

    return {
        'I1':        _resample(d['I1'],       N_OUT).astype(np.float32),
        'I2':        _resample(d['I2'],       N_OUT).astype(np.float32),
        'phi_true':  _resample(d['phi_true'], N_OUT).astype(np.float32),
        'phi_rec':   _resample(phi_rec,       N_OUT).astype(np.float32),
        'rmse':      np.float32(rmse),
        'converged': np.bool_(converged),
        'D1':        np.float32(D1),
        'D2':        np.float32(D2),
        'snr_db':    np.float32(snr),
        'mod':       mod,
        'n_symbols': np.int32(n_sym),
        'gs_err':    np.float32(errs[-1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n',       type=int,  default=10_000)
    ap.add_argument('--preview', action='store_true')
    args = ap.parse_args()

    Path('data').mkdir(exist_ok=True)
    rng = np.random.default_rng(42)

    N_SAMPLES = args.n
    print(f"Generating {N_SAMPLES} samples — estimated time: "
          f"{N_SAMPLES * 0.015:.0f}–{N_SAMPLES * 0.03:.0f} s")

    records = []
    t0 = time.perf_counter()
    converged_count = 0

    for i in range(N_SAMPLES):
        rec = generate_one(rng_seed=i, rng=rng)
        records.append(rec)
        converged_count += rec['converged']

        if (i + 1) % 500 == 0:
            elapsed = time.perf_counter() - t0
            rate    = (i + 1) / elapsed
            eta     = (N_SAMPLES - i - 1) / rate
            pct_ok  = converged_count / (i + 1) * 100
            print(f"  {i+1:>6}/{N_SAMPLES}  "
                  f"{rate:.0f} samples/s  "
                  f"ETA {eta/60:.1f} min  "
                  f"converged {pct_ok:.1f}%")

    elapsed = time.perf_counter() - t0
    print(f"\nDone. {N_SAMPLES} samples in {elapsed:.1f} s "
          f"({elapsed/N_SAMPLES*1000:.1f} ms/sample)")
    print(f"Converged: {converged_count}/{N_SAMPLES} "
          f"({converged_count/N_SAMPLES*100:.1f}%)")

    # Stack into arrays
    I1       = np.stack([r['I1']       for r in records])
    I2       = np.stack([r['I2']       for r in records])
    phi_true = np.stack([r['phi_true'] for r in records])
    phi_rec  = np.stack([r['phi_rec']  for r in records])
    rmse     = np.array([r['rmse']     for r in records], dtype=np.float32)
    conv     = np.array([r['converged']for r in records], dtype=bool)
    D1_arr   = np.array([r['D1']       for r in records], dtype=np.float32)
    D2_arr   = np.array([r['D2']       for r in records], dtype=np.float32)
    snr_arr  = np.array([r['snr_db']   for r in records], dtype=np.float32)
    gs_err   = np.array([r['gs_err']   for r in records], dtype=np.float32)
    mods     = np.array([r['mod']      for r in records])

    out = 'data/training_data.npz'
    np.savez_compressed(out,
        I1=I1, I2=I2, phi_true=phi_true, phi_rec=phi_rec,
        rmse=rmse, converged=conv,
        D1=D1_arr, D2=D2_arr, snr_db=snr_arr,
        gs_err=gs_err, modulation=mods,
    )
    size_mb = Path(out).stat().st_size / 1e6
    print(f"Saved {out}  ({size_mb:.1f} MB)")
    print(f"  I1/I2/phi arrays: {I1.shape}  dtype={I1.dtype}")
    print(f"  Converged-only subset: {conv.sum()} samples")

    if args.preview:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))

        axes[0,0].hist(rmse[conv],  bins=40, color='steelblue', label='converged')
        axes[0,0].hist(rmse[~conv], bins=40, color='tomato',    label='not converged', alpha=0.6)
        axes[0,0].set_xlabel('Phase RMSE (rad)'); axes[0,0].set_title('RMSE distribution')
        axes[0,0].legend()

        from collections import Counter
        mc = Counter(mods)
        axes[0,1].bar(mc.keys(), mc.values(), color='mediumseagreen')
        axes[0,1].set_title('Modulation format distribution')

        axes[1,0].scatter(snr_arr[:500], rmse[:500], s=6, alpha=0.4, c=conv[:500],
                          cmap='RdYlGn')
        axes[1,0].set_xlabel('SNR (dB)'); axes[1,0].set_ylabel('RMSE (rad)')
        axes[1,0].set_title('RMSE vs SNR')

        axes[1,1].plot(phi_true[0, :200], label='true')
        axes[1,1].plot(phi_rec[0,  :200], '--', label='recovered')
        axes[1,1].set_title(f'Sample 0  RMSE={rmse[0]:.3f} rad  mod={mods[0]}')
        axes[1,1].legend()

        plt.tight_layout()
        plt.savefig('data/training_preview.png', dpi=120)
        plt.show()
        print('Saved data/training_preview.png')


if __name__ == '__main__':
    main()
