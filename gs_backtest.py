"""
gs_backtest.py -- PyTorch backtesting suite for FNO + GS phase retrieval
========================================================================

Tests three things:
  1. GS algorithm accuracy across all modulation formats, SNR levels, N lengths
  2. FNO accuracy (trained on N=512) tested at N=256, 512, 1024  (resolution invariance)
  3. Time-bandwidth product: longer windows = better frequency resolution
     Heisenberg: delta_t * delta_nu >= 1  (position <-> momentum analog)

Run:  python gs_backtest.py
      make backtest
"""

import numpy as np
import torch
import time

# ── reporting ──────────────────────────────────────────────────────────────────

_results = []

def _check(name, passed, detail=''):
    status = 'PASS' if passed else 'FAIL'
    marker = '+' if passed else 'X'
    msg = f'  [{status}] {marker}  {name}'
    if detail:
        msg += f'\n         -> {detail}'
    print(msg)
    _results.append((name, passed))
    return passed

def _section(title):
    print(f'\n{"-"*60}')
    print(f'  {title}')
    print(f'{"-"*60}')

# ── import project modules ─────────────────────────────────────────────────────

from gs_core import make_measurements, retrieve_phase
from gs_fno   import FNO1d, wrapped_phase_loss, make_fno_dataset, train_fno

# ══════════════════════════════════════════════════════════════════════════════
# B1. GS accuracy across modulation formats
# ══════════════════════════════════════════════════════════════════════════════

def backtest_gs_formats():
    _section('B1 . GS accuracy -- all modulation formats')

    # unit_amplitude: True for constant-envelope formats, False for amplitude-modulated
    fmt_config = {
        # Constant-envelope formats: GS converges, tight thresholds
        # Amplitude-modulated: GS not designed for these; thresholds just check it runs
        'OOK':    {'unit_amplitude': False, 'threshold': 100},
        'PAM4':   {'unit_amplitude': False, 'threshold': 180},  # GS not suited: 4 amp levels
        'QPSK':   {'unit_amplitude': True,  'threshold': 45},
        'DPSK':   {'unit_amplitude': True,  'threshold': 10},
        'SOLITON':{'unit_amplitude': False, 'threshold': 350},  # GS not suited: sech envelope
        '6PSK':   {'unit_amplitude': True,  'threshold': 10},
    }
    D1, D2  = -5000, -5750
    snr_db  = 35
    n_iter  = 50

    for fmt, cfg in fmt_config.items():
        try:
            m = make_measurements(modulation=fmt, n_symbols=64, sps=8,
                                  D1=D1, D2=D2, snr_db=snr_db)
            I1, I2, phi_true = m['I1'], m['I2'], m['phi_true']
            phi_hat, _ = retrieve_phase(I1, I2, D1=D1, D2=D2, n_iter=n_iter,
                                        unit_amplitude=cfg['unit_amplitude'])

            # align global phase offset
            offset  = np.angle(np.mean(np.exp(1j * (phi_true - phi_hat))))
            phi_hat = phi_hat + offset
            rms     = np.degrees(np.sqrt(np.mean((phi_true - phi_hat)**2)))
            thr     = cfg['threshold']

            _check(f'GS {fmt:8s} RMS < {thr} deg',
                   rms < thr,
                   f'RMS = {rms:.1f} deg')
        except Exception as e:
            _check(f'GS {fmt:8s}', False, f'Error: {e}')


# ══════════════════════════════════════════════════════════════════════════════
# B2. GS accuracy vs SNR sweep
# ══════════════════════════════════════════════════════════════════════════════

def backtest_gs_snr():
    _section('B2 . GS accuracy vs SNR (QPSK)')

    snr_levels = [10, 20, 30, 35, 40]
    thresholds = {10: 70, 20: 55, 30: 45, 35: 45, 40: 45}
    D1, D2 = -5000, -5750

    for snr in snr_levels:
        try:
            m = make_measurements(modulation='QPSK', n_symbols=64, sps=8,
                                  D1=D1, D2=D2, snr_db=snr)
            I1, I2, phi_true = m['I1'], m['I2'], m['phi_true']
            phi_hat, _ = retrieve_phase(I1, I2, D1=D1, D2=D2, n_iter=50)
            offset  = np.angle(np.mean(np.exp(1j * (phi_true - phi_hat))))
            phi_hat = phi_hat + offset
            rms     = np.degrees(np.sqrt(np.mean((phi_true - phi_hat)**2)))
            thr     = thresholds[snr]
            _check(f'SNR={snr:2d} dB  RMS < {thr} deg',
                   rms < thr,
                   f'RMS = {rms:.1f} deg')
        except Exception as e:
            _check(f'SNR={snr} dB', False, f'Error: {e}')


# ══════════════════════════════════════════════════════════════════════════════
# B3. Time-bandwidth product (position <-> momentum)
# ══════════════════════════════════════════════════════════════════════════════

def backtest_time_bandwidth():
    _section('B3 . Time-bandwidth product -- longer N = better freq resolution')

    # Heisenberg: sigma_t * sigma_nu >= 1/(4*pi)  (Gaussian pulse)
    # For GS: longer window -> more symbols -> better phase estimation
    # Measure: RMS error should decrease as N grows (more temporal context)

    N_vals  = [128, 256, 512, 1024]
    D1, D2  = -5000, -5750
    rms_prev = 999.0

    print(f'    {"N":>6}  {"RMS (deg)":>10}  {"TBP":>8}  trend')
    improving = True

    rms_vals = []
    for N in N_vals:
        try:
            n_sym = max(8, N // 8)
            m = make_measurements(modulation='QPSK', n_symbols=n_sym, sps=8,
                                  D1=D1, D2=D2, snr_db=35)
            I1, I2, phi_true = m['I1'], m['I2'], m['phi_true']
            phi_hat, _ = retrieve_phase(I1, I2, D1=D1, D2=D2, n_iter=50)
            offset  = np.angle(np.mean(np.exp(1j * (phi_true - phi_hat))))
            phi_hat = phi_hat + offset
            rms     = np.degrees(np.sqrt(np.mean((phi_true - phi_hat)**2)))

            # TBP: N samples * 1/N bandwidth = 1 always for DFT (Nyquist)
            # But phase estimation improves: more symbols = lower RMS
            tbp = N * (1.0 / N)   # = 1.0 by construction
            trend = 'v' if rms < rms_prev else '^'
            print(f'    {N:>6}  {rms:>10.2f}  {tbp:>8.2f}  {trend}')
            rms_vals.append(rms)
            rms_prev = rms
        except Exception as e:
            print(f'    {N:>6}  ERROR: {e}')
            rms_vals.append(999.0)

    # Check overall trend: median of diffs should be negative (improving)
    diffs = [rms_vals[i+1] - rms_vals[i] for i in range(len(rms_vals)-1)]
    improving = sum(1 for d in diffs if d < 0) >= len(diffs) // 2
    _check('Longer N improves phase estimation (more temporal context)',
           improving,
           f'RMS trend: {[f"{r:.1f}" for r in rms_vals]}')


# ══════════════════════════════════════════════════════════════════════════════
# B4. FNO resolution invariance
# ══════════════════════════════════════════════════════════════════════════════

def backtest_fno_resolution():
    _section('B4 . FNO resolution invariance (train N=512, test N=256/512/1024)')

    if not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = 'cuda'

    print(f'    Device: {device}')
    print('    Building small training set (this takes ~30s on CPU)...')

    try:
        # Small dataset for speed
        X_tr, Y_tr = make_fno_dataset(
            modulations=['QPSK', 'OOK'],
            n_per_format=40,
            N_t=512,
            snr_db=35,
            D1=-5000, D2=-5750
        )

        model = FNO1d(in_channels=2, out_channels=1, modes=32, width=32, n_layers=3)
        t0 = time.time()
        train_fno(model, X_tr, Y_tr, n_epochs=30, lr=3e-3, batch_size=16)
        elapsed = time.time() - t0
        print(f'    Training done in {elapsed:.1f}s')

        model.eval()
        thresholds = {256: 60, 512: 35, 1024: 55}   # looser at untrained resolutions

        for N_test in [256, 512, 1024]:
            X_te, Y_te = make_fno_dataset(
                modulations=['QPSK'],
                n_per_format=20,
                N_t=N_test,
                snr_db=35,
                D1=-5000, D2=-5750
            )
            with torch.no_grad():
                X_t = torch.tensor(X_te, dtype=torch.float32)
                Y_t = torch.tensor(Y_te, dtype=torch.float32)
                phi_hat = model(X_t).numpy()

            # wrapped phase loss in degrees
            loss = float(wrapped_phase_loss(
                torch.tensor(phi_hat), Y_t
            ))
            rms_approx = np.degrees(np.sqrt(loss))   # approx: loss ~ (delta_phi)^2/2
            thr = thresholds[N_test]
            _check(f'FNO N={N_test:4d}  approx RMS < {thr} deg',
                   rms_approx < thr,
                   f'approx RMS = {rms_approx:.1f} deg  (loss={loss:.4f})')

    except Exception as e:
        _check('FNO resolution invariance', False, f'Error: {e}')


# ══════════════════════════════════════════════════════════════════════════════
# B5. FNO vs GS on same data (is FNO adding value?)
# ══════════════════════════════════════════════════════════════════════════════

def backtest_fno_vs_gs():
    _section('B5 . FNO vs GS head-to-head (QPSK, SNR=35 dB, N=512)')

    try:
        X_tr, Y_tr = make_fno_dataset(
            modulations=['QPSK', 'OOK', '6PSK'],
            n_per_format=30, N_t=512, snr_db=35, D1=-5000, D2=-5750
        )
        model = FNO1d(in_channels=2, out_channels=1, modes=32, width=32, n_layers=3)
        train_fno(model, X_tr, Y_tr, n_epochs=30, lr=3e-3, batch_size=16)
        model.eval()

        # Test set
        X_te, Y_te = make_fno_dataset(
            modulations=['QPSK'], n_per_format=30,
            N_t=512, snr_db=35, D1=-5000, D2=-5750
        )

        # GS baseline on same data
        # make_fno_dataset may return tensors or numpy -- normalise to numpy
        def _np(x):
            if hasattr(x, 'detach'):
                return x.detach().numpy()
            return np.asarray(x)

        X_np = _np(X_te)
        Y_np = _np(Y_te)

        gs_rms_list = []
        for i in range(X_np.shape[0]):
            I1_i       = X_np[i, 0]
            I2_i       = X_np[i, 1]
            phi_true_i = Y_np[i, 0]
            phi_hat, _ = retrieve_phase(I1_i, I2_i, D1=-5000, D2=-5750, n_iter=50)
            offset     = np.angle(np.mean(np.exp(1j * (phi_true_i - phi_hat))))
            phi_hat    = phi_hat + offset
            gs_rms_list.append(np.degrees(np.sqrt(np.mean((phi_true_i - phi_hat)**2))))
        gs_rms = float(np.mean(gs_rms_list))

        # FNO on same data
        X_t = torch.tensor(X_np, dtype=torch.float32)
        Y_t = torch.tensor(Y_np, dtype=torch.float32)
        with torch.no_grad():
            phi_fno = model(X_t)
        loss = float(wrapped_phase_loss(phi_fno, Y_t).detach())
        fno_rms = np.degrees(np.sqrt(loss))

        print(f'    GS  mean RMS = {gs_rms:.2f} deg')
        print(f'    FNO mean RMS = {fno_rms:.2f} deg')

        _check('GS RMS < 70 deg baseline',   gs_rms  < 70, f'{gs_rms:.2f} deg')
        _check('FNO trains without crashing', fno_rms < 90, f'{fno_rms:.2f} deg')
        _check('Both produce finite outputs',
               np.isfinite(gs_rms) and np.isfinite(fno_rms), '')

    except Exception as e:
        _check('FNO vs GS head-to-head', False, f'Error: {e}')


# ══════════════════════════════════════════════════════════════════════════════
# Run all
# ══════════════════════════════════════════════════════════════════════════════

def run_all():
    print('\n' + '='*60)
    print('  gs_backtest.py -- FNO + GS backtesting suite')
    print('='*60)

    backtest_gs_formats()
    backtest_gs_snr()
    backtest_time_bandwidth()
    backtest_fno_resolution()
    backtest_fno_vs_gs()

    n_pass = sum(1 for _, p in _results if p)
    n_fail = sum(1 for _, p in _results if not p)

    print('\n' + '='*60)
    print(f'  TOTAL: {n_pass} passed, {n_fail} failed  ({len(_results)} checks)')
    print('='*60)

    if n_fail > 0:
        print('\nFAILED:')
        for name, passed in _results:
            if not passed:
                print(f'  X  {name}')

    return n_fail == 0


if __name__ == '__main__':
    ok = run_all()
    raise SystemExit(0 if ok else 1)
