"""Load the two-arm intensity files in sample_data/ and run phase retrieval on them.

In plain words: each .mat file holds two intensity traces, I1 and I2, of the same
optical signal after two different amounts of dispersion. Gerchberg-Saxton finds
the phase that explains both traces at once.

Things to know about these files (checked, not assumed):

* They are synthetic. Each file's own `description` field says so. They are not
  lab measurements, so a good fit here shows the code is self-consistent, not
  that it works on real hardware.
* D1_ps2 / D2_ps2 (-600, -1200) are labeled ps^2, but the data match gs_core's
  normalized dispersion units: used directly, GS converges; converted to
  physical units, it does not.
* Convergence is slow. Mismatch on the arm not enforced last goes from ~40-65 %
  at iteration 1 to ~2-6 % at 500 and ~1-4 % at 2000 (see retrieve()).
* The stored time vector `t` is all zeros, so load() rebuilds it from fs.
* Use unit_amplitude=False. With it on, the fit fails, because these signals
  (OOK, PAM4, STEAM, soliton, and even the shaped QPSK/DPSK) are not constant-envelope.
"""
import pathlib
import numpy as np
import scipy.io as sio

from dgs import gs_core

SAMPLE_DIR = pathlib.Path(__file__).resolve().parent.parent / "sample_data"


def list_files(folder=SAMPLE_DIR):
    return sorted(pathlib.Path(folder).glob("*.mat"))


def load(path):
    """Return a dict with I1, I2 (float arrays), t [ns], fs_GSas, D1, D2, lambda_nm, standard."""
    d = sio.loadmat(path)
    I1 = d["I1"].ravel().astype(float)
    I2 = d["I2"].ravel().astype(float)
    fs = float(d["fs_GSas"].item())
    return dict(
        I1=I1, I2=I2,
        t=np.arange(I1.size) / fs,               # ns, since fs is in GS/s
        fs_GSas=fs,
        D1=float(d["D1_ps2"].item()), D2=float(d["D2_ps2"].item()),
        lambda_nm=float(d["lambda_nm"].item()),
        standard=str(d["standard"][0]),
        description=str(d["description"][0]),
    )


def retrieve(path, n_iter=500):
    """Run GS on one file. Returns (data dict, complex field E(t), residual history).

    Residual = ||  |E dispersed by D1|^2 - I1  || / ||I1||, checked after each
    full iteration. That is the arm NOT enforced last, so it is an honest test.

    Do not use gs_core.retrieve_phase's own error list for this: it compares
    against I2, the constraint the iteration just imposed, so it reads ~1e-16
    after the first iteration whether or not the answer is right.
    """
    data = load(path)
    I1, I2, D1, D2 = data["I1"], data["I2"], data["D1"], data["D2"]
    E = gs_core.undisperse(np.sqrt(I1).astype(complex), D1)
    hist = np.empty(n_iter)
    for k in range(n_iter):
        E = gs_core.gs_iteration(E, I1, I2, D1, D2, unit_amplitude=False)
        hist[k] = np.linalg.norm(np.abs(gs_core.disperse(E, D1)) ** 2 - I1) / np.linalg.norm(I1)
    return data, E, hist
