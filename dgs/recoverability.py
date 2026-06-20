"""Predict whether a measurement is recoverable -- from the intensities alone.

The carrier-less receiver only ever sees I1, I2 -- never the phase. Sometimes
Gerchberg-Saxton recovers phi(t) cleanly; sometimes it can't (too little
dispersion diversity, or too much noise). The question: using ONLY engineered
features of I1 and I2 -- their correlation (diversity), spectral widths, dynamic
range, a high-frequency noise proxy -- can a classifier predict recoverability
*before* running GS?

Yes. Diversity and SNR are imprinted in the intensities, so a simple logistic
model reads them off. This is a triage step: flag un-recoverable shots without
spending the GS iterations. NumPy only (uses dgs.gs_core + dgs.supervised).
"""

import numpy as np

from dgs import gs_core, supervised

FEATURE_NAMES = ["corr_I1_I2",
                 "relstd_1", "dynrange_1", "bandwidth_1", "kurtosis_1",
                 "relstd_2", "dynrange_2", "bandwidth_2", "kurtosis_2",
                 "hf_noise_1", "hf_noise_2"]


def _spectral_bandwidth(I):
    S = np.abs(np.fft.rfft(I - I.mean()))
    f = np.arange(len(S))
    return float(np.sqrt((f**2 * S).sum() / (S.sum() + 1e-12)))


def _hf_noise(I):
    P = np.abs(np.fft.rfft(I - I.mean()))**2
    return float(P[len(P) // 2:].sum() / (P.sum() + 1e-12))


def _single(I):
    m = I.mean() + 1e-12
    return [float(I.std() / m), float(I.max() / m), _spectral_bandwidth(I),
            float(((I - I.mean())**4).mean() / (I.var() + 1e-12)**2)]


def intensity_features(I1, I2):
    """Feature vector from the two intensities only (no phase, no dispersion D).
    Order matches FEATURE_NAMES. corr_I1_I2 is the key diversity signal."""
    I1 = np.asarray(I1, dtype=float)
    I2 = np.asarray(I2, dtype=float)
    corr = float(np.corrcoef(I1, I2)[0, 1])
    return np.array([corr] + _single(I1) + _single(I2) + [_hf_noise(I1), _hf_noise(I2)])


def _recovery_rms(I1, I2, D1, D2, phi_true):
    phi, _ = gs_core.retrieve_phase(I1, I2, D1, D2, n_iter=50)
    best = None
    for s in (1, -1):                                   # resolve twin + global offset
        off = np.angle(np.mean(np.exp(1j * (phi_true - s * phi))))
        e = np.sqrt(np.mean(np.angle(np.exp(1j * (phi_true - (s * phi + off))))**2))
        best = e if best is None else min(best, e)
    return best


def make_recoverability_dataset(n_samples=300, rms_thresh=0.35, seed=0):
    """Simulate measurements across a range of diversity and noise; return
    (X features, y recoverable-labels, feature_names). y=1 if GS RMS < rms_thresh.
    Ranges chosen so recoverability is a balanced (~40%) classification problem."""
    rng = np.random.default_rng(seed)
    D1 = -5000.0
    X, y = [], []
    for _ in range(n_samples):
        D2 = D1 - rng.uniform(600, 1100)                # good (working) dispersion diversity
        snr = rng.uniform(3, 45)                        # WIDE noise range -> the deciding factor
        data = gs_core.make_qpsk_measurements(n_symbols=64, D1=D1, D2=D2,
                                              snr_db=snr, rng_seed=int(rng.integers(2**31)))
        I1 = np.maximum(data["I1"], 0)
        I2 = np.maximum(data["I2"], 0)
        rms = _recovery_rms(I1, I2, D1, D2, data["phi_true"])
        X.append(intensity_features(I1, I2))
        y.append(1 if rms < rms_thresh else 0)
    return np.array(X), np.array(y), FEATURE_NAMES


def _auc(scores, labels):
    """ROC AUC: P(a random recoverable shot scores above a random not). Balance-
    independent -- the right metric when the classes aren't 50/50."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fit_recoverability_classifier(X, y, seed=0):
    """Train a logistic classifier; return (weights, test_accuracy, test_auc, (mean,std))."""
    Xs, mean, std = supervised.standardize(X)
    Xtr, Xte, ytr, yte = supervised.train_test_split(Xs, y, test_frac=0.3, seed=seed)
    w = supervised.logistic_regression(Xtr, ytr, lr=0.2, n_iter=4000)
    scores = supervised.predict_proba(Xte, w)
    acc = supervised.accuracy(yte, (scores >= 0.5).astype(int))
    return w, acc, _auc(scores, yte), (mean, std)


if __name__ == "__main__":
    X, y, names = make_recoverability_dataset(300, seed=1)
    print(f"dataset: {len(y)} shots, {int(y.sum())} recoverable / {int((1-y).sum())} not")
    w, acc, auc, _ = fit_recoverability_classifier(X, y)
    print(f"classifier test accuracy: {acc:.2f}   AUC: {auc:.2f}  (1.0 = perfect, 0.5 = chance)")
    # the corr feature alone separates the classes
    print(f"mean corr(I1,I2): recoverable={X[y==1,0].mean():.3f}, not={X[y==0,0].mean():.3f}")
