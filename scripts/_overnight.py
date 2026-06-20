"""
_overnight.py -- blast CUDA FNO inference all night
Trains a full FNO on all modulations, then runs continuous inference
and logs throughput + accuracy every 60 seconds.
"""
import time, json, os, logging, traceback
import numpy as np
import torch
from dgs.gs_fno import FNO1d, make_fno_dataset, train_fno, wrapped_phase_loss
from dgs.gs_core import make_measurements, retrieve_phase

LOG      = '_overnight_results.jsonl'
ERR_LOG  = '_overnight_errors.log'

logging.basicConfig(
    filename=ERR_LOG,
    level=logging.ERROR,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

def log_error(context, exc):
    msg = f'[{context}] {type(exc).__name__}: {exc}\n{traceback.format_exc()}'
    logging.error(msg)
    print(f'  ERROR logged -> {ERR_LOG}: {type(exc).__name__}: {exc}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── prevent Windows autosleep ──────────────────────────────────────────────────
import ctypes, platform
if platform.system() == 'Windows':
    ES_CONTINUOUS       = 0x80000000
    ES_SYSTEM_REQUIRED  = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
    print('Autosleep: DISABLED (Windows SetThreadExecutionState)')

start_wall = time.strftime('%Y-%m-%d %H:%M:%S')
print(f'Start:  {start_wall}')
print(f'Device: {device}')
print(f'GPU:    {torch.cuda.get_device_name(0) if device=="cuda" else "cpu"}')
print(f'Log:    {LOG}')
print()

# ── build dataset ──────────────────────────────────────────────────────────────
print('Building dataset (all formats, N=512, SNR sweep 20-40 dB)...')
X_all, Y_all = [], []
for snr in [20, 25, 30, 35, 40]:
    X, Y = make_fno_dataset(
        modulations=['QPSK', 'DPSK', '6PSK', 'OOK'],
        n_per_format=50, N_t=512, snr_db=snr, D1=-5000, D2=-5750
    )
    X_all.append(X); Y_all.append(Y)

X_tr = torch.cat(X_all, dim=0)
Y_tr = torch.cat(Y_all, dim=0)
print(f'Dataset: {X_tr.shape[0]} samples')

# ── train ─────────────────────────────────────────────────────────────────────
print('Training FNO (200 epochs)...')
model = FNO1d(in_channels=2, out_channels=1, modes=32, width=64, n_layers=4)
train_fno(model, X_tr, Y_tr, n_epochs=200, lr=2e-3, batch_size=64)
model = model.to(device)
model.eval()
print('Training done.')
print()

# ── blast inference loop ───────────────────────────────────────────────────────
print('Blasting inference. Ctrl+C to stop. Logging to', LOG)
print(f'{"time":>8}  {"inf/sec":>10}  {"ms/batch":>10}  {"VRAM MB":>8}  {"loss":>8}')
print('-' * 55)

X_t = torch.tensor(X_tr[:64], dtype=torch.float32).to(device)
Y_t = torch.tensor(Y_tr[:64], dtype=torch.float32).to(device)

# warm up
with torch.no_grad():
    for _ in range(20): _ = model(X_t)
if device == 'cuda': torch.cuda.synchronize()

session_start = time.time()
consecutive_errors = 0
MAX_ERRORS = 10

try:
    while True:
        try:
            # 60-second window
            count = 0
            if device == 'cuda': torch.cuda.synchronize()
            t0 = time.perf_counter()
            deadline = t0 + 60.0
            with torch.no_grad():
                while time.perf_counter() < deadline:
                    out = model(X_t)
                    count += 64
            if device == 'cuda': torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            # measure loss on last batch
            with torch.no_grad():
                out  = model(X_t)
                loss = float(wrapped_phase_loss(out, Y_t).detach())

            ips      = count / elapsed
            ms_batch = elapsed / (count / 64) * 1000
            vram     = torch.cuda.memory_allocated() // 1024**2 if device == 'cuda' else 0
            uptime   = int(time.time() - session_start)

            h, m_rem    = divmod(uptime, 3600)
            m2, s       = divmod(m_rem, 60)
            elapsed_str = f'{h:02d}:{m2:02d}:{s:02d}'

            row = {
                'uptime_s'    : uptime,
                'elapsed'     : elapsed_str,
                'wall_time'   : time.strftime('%H:%M:%S'),
                'inf_per_sec' : round(ips, 1),
                'ms_per_batch': round(ms_batch, 3),
                'vram_mb'     : vram,
                'loss'        : round(loss, 5),
            }
            with open(LOG, 'a') as f:
                f.write(json.dumps(row) + '\n')

            print(f'{elapsed_str}  {ips:>10.0f}  {ms_batch:>10.3f}  {vram:>8}  {loss:>8.5f}')
            consecutive_errors = 0   # reset on success

        except torch.cuda.OutOfMemoryError as e:
            log_error('OOM', e)
            torch.cuda.empty_cache()
            consecutive_errors += 1
        except RuntimeError as e:
            log_error('RuntimeError', e)
            consecutive_errors += 1
        except Exception as e:
            log_error('inference_loop', e)
            consecutive_errors += 1

        if consecutive_errors >= MAX_ERRORS:
            logging.critical(f'Stopping: {MAX_ERRORS} consecutive errors')
            print(f'FATAL: {MAX_ERRORS} consecutive errors -- check {ERR_LOG}')
            break

except KeyboardInterrupt:
    print()
    print('Stopped. Results in', LOG)
    # summary
    results = [json.loads(l) for l in open(LOG)]
    if results:
        ips_vals = [r['inf_per_sec'] for r in results]
        print(f'Mean throughput: {np.mean(ips_vals):.0f} inf/sec')
        print(f'Peak throughput: {max(ips_vals):.0f} inf/sec')
        print(f'Final loss:      {results[-1]["loss"]:.5f}')
