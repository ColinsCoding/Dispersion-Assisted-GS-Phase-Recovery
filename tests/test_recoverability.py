"""Test: predict GS recoverability from intensity features alone."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import recoverability as rec

# 1. feature vector is well-formed (right length, all finite)
data_X, data_y, names = rec.make_recoverability_dataset(n_samples=160, seed=3)
assert data_X.shape == (160, len(names))
assert np.all(np.isfinite(data_X))
assert names[0] == "corr_I1_I2"

# 2. the dataset spans both classes (some recoverable, some not)
frac = data_y.mean()
assert 0.15 < frac < 0.85, frac

# 3. recoverability IS predictable from intensities alone: the score RANKS
#    recoverable shots above unrecoverable ones (AUC, the balance-independent metric)
w, acc, auc, _ = rec.fit_recoverability_classifier(data_X, data_y, seed=0)
assert auc > 0.7, auc                                   # 0.5 = chance, 1.0 = perfect

# 4. the signal is real: at least one single feature ranks recoverability well
#    above chance on its own (direction-agnostic -- in this regime the dominant
#    cue is the noise/SNR proxy, not the diversity correlation)
single = [(names[j], abs(rec._auc(data_X[:, j], data_y) - 0.5)) for j in range(data_X.shape[1])]
best_name, best_sep = max(single, key=lambda t: t[1])
assert best_sep > 0.15, single                          # some feature is individually informative

# 5. features come only from I1, I2 -- deterministic, no phase/D leaked in
from dgs import gs_core
shot = gs_core.make_qpsk_measurements(n_symbols=64, D1=-5000, D2=-5750, snr_db=40, rng_seed=7)
f1 = rec.intensity_features(np.maximum(shot["I1"], 0), np.maximum(shot["I2"], 0))
f2 = rec.intensity_features(np.maximum(shot["I1"], 0), np.maximum(shot["I2"], 0))
assert np.allclose(f1, f2) and len(f1) == len(names)

print(f"TEST PASS  ({int(data_y.sum())}/{len(data_y)} recoverable; AUC={auc:.2f}; "
      f"best single feature = {best_name} (sep {best_sep:.2f}))")
