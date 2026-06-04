"""
gs_monitor.py -- real-time performance monitor for GS + FNO pipeline
=====================================================================

Statistical mechanics framing:
  Each GS iteration is a 'collision' -- a discrete state update.
  The monitor counts collisions per second, measures energy dissipation
  (convergence rate), and tracks the phase-space trajectory (error curve).

  Maxwell-Boltzmann analogy:
    Temperature  T  <->  SNR (high SNR = low noise = low T = ordered system)
    Mean free path   <->  iterations to convergence
    Collision rate   <->  FFTs per second
    Equilibrium      <->  GS converged (error plateau)

Run:   python gs_monitor.py
       python gs_monitor.py --n 512 --iter 50 --snr 35 --device cuda
"""

import argparse
import time
import numpy as np

try:
    import torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

from gs_core import make_measurements, retrieve_phase

# ── ANSI terminal colors ───────────────────────────────────────────────────────
R  = '\033[91m'
G  = '\033[92m'
Y  = '\033[93m'
B  = '\033[94m'
W  = '\033[0m'
BOLD = '\033[1m'

# ── single-window measurement ─────────────────────────────────────────────────

def measure_window(n_symbols, sps, D1, D2, snr_db, n_iter, modulation='QPSK'):
    """Run one GS retrieval. Return (rms_deg, errors, elapsed_sec)."""
    m = make_measurements(modulation=modulation, n_symbols=n_symbols, sps=sps,
                          D1=D1, D2=D2, snr_db=snr_db)
    I1, I2, phi_true = m['I1'], m['I2'], m['phi_true']
    N = len(I1)

    t0 = time.perf_counter()
    phi_hat, errors = retrieve_phase(I1, I2, D1=D1, D2=D2, n_iter=n_iter)
    elapsed = time.perf_counter() - t0

    offset  = np.angle(np.mean(np.exp(1j * (phi_true - phi_hat))))
    phi_hat = phi_hat + offset
    rms     = np.degrees(np.sqrt(np.mean((phi_true - phi_hat)**2)))
    return rms, errors, elapsed, N


# ── throughput benchmark (1-second window) ────────────────────────────────────

def throughput_benchmark(n_symbols, sps, D1, D2, snr_db, n_iter,
                         window_sec=1.0, modulation='QPSK'):
    """
    Run as many GS retrievals as possible in window_sec.
    Returns dict of collision statistics (stat-mech framing).
    """
    m = make_measurements(modulation=modulation, n_symbols=n_symbols, sps=sps,
                          D1=D1, D2=D2, snr_db=snr_db)
    I1, I2 = m['I1'], m['I2']
    N = len(I1)

    retrievals   = 0
    total_iters  = 0
    total_ffts   = 0
    rms_list     = []
    phi_true     = m['phi_true']

    deadline = time.perf_counter() + window_sec
    while time.perf_counter() < deadline:
        phi_hat, errors = retrieve_phase(I1, I2, D1=D1, D2=D2, n_iter=n_iter)
        offset  = np.angle(np.mean(np.exp(1j * (phi_true - phi_hat))))
        phi_hat = phi_hat + offset
        rms     = np.degrees(np.sqrt(np.mean((phi_true - phi_hat)**2)))
        rms_list.append(rms)
        retrievals  += 1
        total_iters += n_iter
        total_ffts  += n_iter * 4   # 2 disperse + 2 undisperse per iteration

    elapsed = window_sec

    return {
        'retrievals_per_sec' : retrievals  / elapsed,
        'iterations_per_sec' : total_iters / elapsed,
        'ffts_per_sec'       : total_ffts  / elapsed,
        'samples_per_sec'    : retrievals * N / elapsed,
        'mean_rms_deg'       : float(np.mean(rms_list)),
        'std_rms_deg'        : float(np.std(rms_list)),
        'n_collisions'       : total_iters,   # stat-mech: total collisions
        'signal_length_N'    : N,
        'n_retrievals'       : retrievals,
    }


# ── FNO throughput (if torch available) ───────────────────────────────────────

def fno_throughput_benchmark(n_symbols, sps, D1, D2, snr_db,
                             window_sec=1.0, device=None):
    """Benchmark FNO inference throughput in samples/sec."""
    if not TORCH_OK:
        return None

    from gs_fno import FNO1d, make_fno_dataset, train_fno

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    m = make_measurements(modulation='QPSK', n_symbols=n_symbols, sps=sps,
                          D1=D1, D2=D2, snr_db=snr_db)
    N = len(m['I1'])

    # quick train on tiny dataset
    print(f'  {Y}Training FNO for benchmark (20 epochs)...{W}')
    X_tr, Y_tr = make_fno_dataset(['QPSK'], n_per_format=40, N_t=N,
                                   snr_db=snr_db, D1=D1, D2=D2)
    model = FNO1d(in_channels=2, out_channels=1, modes=32, width=32, n_layers=3)
    train_fno(model, X_tr, Y_tr, n_epochs=20, lr=3e-3, batch_size=32)
    model = model.to(device)
    model.eval()

    # build a batch of inputs
    batch = 64
    X_t = torch.tensor(
        np.tile(X_tr[:1], (batch, 1, 1)), dtype=torch.float32
    ).to(device)

    # warm up
    with torch.no_grad():
        for _ in range(5):
            _ = model(X_t)
    if device == 'cuda':
        torch.cuda.synchronize()

    # timed window
    inferences = 0
    deadline = time.perf_counter() + window_sec
    with torch.no_grad():
        while time.perf_counter() < deadline:
            _ = model(X_t)
            if device == 'cuda':
                torch.cuda.synchronize()
            inferences += batch

    return {
        'device'            : device,
        'inferences_per_sec': inferences / window_sec,
        'signal_length_N'   : N,
        'batch_size'        : batch,
        'vram_mb'           : (torch.cuda.memory_allocated() // 1024**2
                               if device == 'cuda' else 0),
    }


# ── ASCII bar chart ────────────────────────────────────────────────────────────

def bar(value, max_val, width=40, color=G):
    filled = int(width * min(value, max_val) / max_val)
    return color + '#' * filled + W + '-' * (width - filled)


# ── convergence curve (ASCII) ─────────────────────────────────────────────────

def plot_convergence(errors, width=50, height=8):
    if not errors:
        return
    e = np.array(errors, dtype=float)
    e_min, e_max = e.min(), e.max()
    if e_max == e_min:
        e_max = e_min + 1e-9
    print(f'\n  Convergence curve ({len(e)} iterations):')
    for row in range(height, 0, -1):
        threshold = e_min + (e_max - e_min) * row / height
        line = '  |'
        for val in e:
            line += (G + '*' + W) if val >= threshold else ' '
        print(line)
    print('  +' + '-' * len(e))
    print(f'    iter 0                iter {len(e)-1}')
    print(f'    err={e[0]:.4f}               err={e[-1]:.4f}')


# ── main display ──────────────────────────────────────────────────────────────

def run_monitor(n_symbols=64, sps=8, D1=-5000, D2=-5750,
                snr_db=35, n_iter=50, window_sec=1.0,
                modulation='QPSK', show_fno=True):

    print(f'\n{BOLD}{"="*62}{W}')
    print(f'{BOLD}  gs_monitor.py  --  Jalali Lab GS+FNO performance monitor{W}')
    print(f'{BOLD}{"="*62}{W}')

    # ── system info ──
    print(f'\n{B}System:{W}')
    if TORCH_OK and torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory // 1024**2
        print(f'  GPU  : {G}{gpu}{W}  ({vram} MB VRAM)')
        print(f'  CUDA : {torch.version.cuda}  |  torch {torch.__version__}')
    else:
        print(f'  GPU  : {Y}not available -- CPU only{W}')
    print(f'  Signal: N={n_symbols*sps}  ({n_symbols} symbols x {sps} sps)')
    print(f'  Config: D1={D1}  D2={D2}  SNR={snr_db} dB  iter={n_iter}')

    # ── single retrieval detail ──
    print(f'\n{B}Single retrieval:{W}')
    rms, errors, elapsed, N = measure_window(n_symbols, sps, D1, D2,
                                              snr_db, n_iter, modulation)
    print(f'  Time   : {elapsed*1000:.1f} ms')
    print(f'  RMS    : {rms:.1f} deg  ', bar(90 - rms, 90, 30), f'  (lower=better)')
    print(f'  Converged: {"yes" if rms < 30 else "partial"}')
    plot_convergence(errors)

    # ── 1-second throughput window ──
    print(f'\n{B}1-second throughput window (stat-mech collision count):{W}')
    stats = throughput_benchmark(n_symbols, sps, D1, D2, snr_db,
                                  n_iter, window_sec, modulation)
    print(f'  Retrievals/sec  : {G}{stats["retrievals_per_sec"]:.1f}{W}')
    print(f'  Iterations/sec  : {stats["iterations_per_sec"]:.0f}   '
          f'{B}(= collision rate){W}')
    print(f'  FFTs/sec        : {stats["ffts_per_sec"]:.0f}')
    print(f'  Samples/sec     : {stats["samples_per_sec"]:.0f}')
    print(f'  Total collisions: {stats["n_collisions"]}  in {window_sec:.1f}s')
    print(f'  Mean RMS        : {stats["mean_rms_deg"]:.1f} +/- '
          f'{stats["std_rms_deg"]:.1f} deg')

    # stat-mech summary
    kT_analog = 1.0 / (snr_db / 10.0)   # noise ~ 1/SNR ~ temperature
    mfp = n_iter / max(1, stats["retrievals_per_sec"])  # mean free path in time
    print(f'\n  Stat-mech analogy:')
    print(f'    T_analog (1/SNR_linear) = {kT_analog:.4f}')
    print(f'    Mean free path          = {mfp*1000:.2f} ms / retrieval')
    print(f'    Collision rate          = {stats["iterations_per_sec"]:.0f} /s')
    print(f'    Equilibrium RMS         = {stats["mean_rms_deg"]:.1f} deg')

    # ── FNO throughput ──
    if show_fno and TORCH_OK:
        print(f'\n{B}FNO inference throughput:{W}')
        fno_stats = fno_throughput_benchmark(n_symbols, sps, D1, D2, snr_db,
                                              window_sec)
        if fno_stats:
            dev_str = G + fno_stats['device'].upper() + W
            print(f'  Device          : {dev_str}')
            print(f'  Inferences/sec  : {G}{fno_stats["inferences_per_sec"]:.0f}{W}  '
                  f'(batch={fno_stats["batch_size"]})')
            print(f'  Signal length   : N={fno_stats["signal_length_N"]}')
            if fno_stats['device'] == 'cuda':
                print(f'  VRAM used       : {fno_stats["vram_mb"]} MB / '
                      f'{torch.cuda.get_device_properties(0).total_memory//1024**2} MB')
            speedup = fno_stats['inferences_per_sec'] / max(1, stats['retrievals_per_sec'])
            print(f'  FNO vs GS speedup: {Y}{speedup:.0f}x faster{W}  '
                  f'(inference vs iterative)')

    print(f'\n{BOLD}{"="*62}{W}\n')


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='GS+FNO performance monitor')
    p.add_argument('--n',      type=int,   default=64,     help='n_symbols')
    p.add_argument('--sps',    type=int,   default=8,      help='samples per symbol')
    p.add_argument('--iter',   type=int,   default=50,     help='GS iterations')
    p.add_argument('--snr',    type=float, default=35.0,   help='SNR dB')
    p.add_argument('--d1',     type=float, default=-5000,  help='D1')
    p.add_argument('--d2',     type=float, default=-5750,  help='D2')
    p.add_argument('--mod',    type=str,   default='QPSK', help='modulation')
    p.add_argument('--window', type=float, default=1.0,    help='benchmark window (s)')
    p.add_argument('--no-fno', action='store_true',        help='skip FNO benchmark')
    args = p.parse_args()

    run_monitor(
        n_symbols  = args.n,
        sps        = args.sps,
        D1         = args.d1,
        D2         = args.d2,
        snr_db     = args.snr,
        n_iter     = args.iter,
        window_sec = args.window,
        modulation = args.mod,
        show_fno   = not args.no_fno,
    )
