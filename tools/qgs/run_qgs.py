#!/usr/bin/env python3
"""Run the compiled wavefunction phase retrieval (C or CUDA) on a test wavefunction and report the fidelity.

    ./run_qgs.py ./gs_wavefunction            (or: python3 run_qgs.py path/to/executable [n_iter])

Exit codes, like the C program: 0 recovered (fidelity > 0.999), 1 ran but did not recover,
2 bad arguments, 3 the executable is missing or failed. Errors are reported as messages and exit
codes; the script never ends with a Python traceback.
"""
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

HBAR_M = 0.11576765                       # hbar / m_e, nm^2/fs
N, L = 1024, 200.0                        # grid points, box length (nm)
TIMES = [0.0, 20.0, 40.0, 80.0, 160.0]    # fs


def test_wavefunction():
    x = (np.arange(N) - N // 2)*(L/N)
    pk = lambda x0, s, k0, ph: np.exp(-(x - x0)**2/(4*s**2) + 1j*k0*x + 1j*ph)
    psi = pk(-10, 3, 0.4, 0) + 0.8*pk(12, 2.5, -0.3, np.pi/2)
    return psi/np.sqrt(np.sum(np.abs(psi)**2)*(L/N))


def evolve(psi, t):
    k = 2*np.pi*np.fft.fftfreq(N, L/N)
    return np.fft.ifft(np.fft.fft(psi)*np.exp(-0.5j*HBAR_M*k**2*t))


def main(argv):
    if len(argv) not in (2, 3):
        print(__doc__.strip().splitlines()[2], file=sys.stderr)
        return 2
    exe = argv[1]
    n_iter = argv[2] if len(argv) == 3 else '300'
    if not n_iter.isdigit():
        print('n_iter must be a whole number', file=sys.stderr)
        return 2
    if not os.path.isfile(exe):
        print(f'executable not found: {exe}', file=sys.stderr)
        return 3
    psi0 = test_wavefunction()
    work = tempfile.mkdtemp(prefix='qgs_')
    inp, out = os.path.join(work, 'densities.bin'), os.path.join(work, 'psi.bin')
    with open(inp, 'wb') as f:
        f.write(struct.pack('<iidd', N, len(TIMES), L/N, HBAR_M))
        f.write(np.array(TIMES, dtype='<f8').tobytes())
        for t in TIMES:
            f.write((np.abs(evolve(psi0, t))**2).astype('<f8').tobytes())
    r = subprocess.run([exe, inp, out, n_iter], capture_output=True, text=True)
    if r.returncode != 0:
        print(f'{os.path.basename(exe)} failed with exit code {r.returncode}: {r.stderr.strip()}', file=sys.stderr)
        return 3
    raw = np.fromfile(out, dtype='<f8')
    psi = raw[0::2] + 1j*raw[1::2]
    fidelity = abs(np.sum(np.conj(psi)*psi0)*(L/N))**2
    lines = r.stdout.split()
    print(f'{os.path.basename(exe)}: misfit {lines[0]}, fidelity |<psi_rec|psi>|^2 = {fidelity:.6f}'
          + (f', GPU time {lines[1]} ms' if len(lines) > 1 else ''))
    return 0 if fidelity > 0.999 else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
