"""Test dgs.exp_decay_classifier: the antiderivative must actually
differentiate back to the decay model (symbolically), the +C must
provably cancel in every definite integral, the parameter fit must
recover known (A,k) from noisy data, and the classifier must assign the
correct parity/side label -- and correctly flag "other" for genuine
outliers, not just always agree with the model."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.exp_decay_classifier import (
    exponential_decay, decay_antiderivative, verify_antiderivative_symbolic,
    cumulative_energy, verify_C_cancels, fit_decay_params, classify_sample, classify_batch,
)

A, k = 5.0, 0.3

# 1. exponential_decay: A at t=0, monotonically decreasing
assert exponential_decay(A, k, 0.0) == A
t_arr = np.linspace(0, 20, 50)
y_arr = exponential_decay(A, k, t_arr)
assert np.all(np.diff(y_arr) < 0)
assert y_arr[-1] < 0.01 * A   # should have decayed substantially by t=20 (k=0.3)

# 2. decay_antiderivative's numerical derivative matches exponential_decay
h = 1e-6
t0_check = 3.0
numerical_deriv = (decay_antiderivative(A, k, t0_check + h) - decay_antiderivative(A, k, t0_check - h)) / (2 * h)
assert abs(numerical_deriv - exponential_decay(A, k, t0_check)) < 1e-4

# 3. symbolic check: d/dt[F] == A*e^(-kt) exactly
sym_check = verify_antiderivative_symbolic()
assert sym_check["matches"] is True

# 4. cumulative_energy matches a fine numerical trapezoid integration
t_fine = np.linspace(1.0, 6.0, 200_000)
numeric_integral = np.trapezoid(exponential_decay(A, k, t_fine), t_fine)
analytic = cumulative_energy(A, k, 1.0, 6.0)
assert abs(numeric_integral - analytic) / analytic < 1e-4

# 5. THE +C CLAIM: cumulative_energy is identical across wildly different C choices
c_check = verify_C_cancels(A, k, 0.0, 3.0, C_values=(0.0, 5.0, -3.7, 1000.0))
assert c_check["spread"] < 1e-8

# 6. fit_decay_params recovers known parameters from noiseless data almost exactly,
# and stays close from moderately noisy data
t_data = np.linspace(0, 10, 40)
y_clean_data = exponential_decay(A, k, t_data)
fit_clean = fit_decay_params(t_data, y_clean_data)
assert abs(fit_clean["A_fit"] - A) < 1e-6
assert abs(fit_clean["k_fit"] - k) < 1e-6

rng = np.random.default_rng(0)
y_noisy = y_clean_data * (1 + 0.03 * rng.standard_normal(len(t_data)))
fit_noisy = fit_decay_params(t_data, y_noisy)
assert abs(fit_noisy["A_fit"] - A) / A < 0.1
assert abs(fit_noisy["k_fit"] - k) / k < 0.1

# 7. fit_decay_params rejects non-positive y (can't take log)
try:
    fit_decay_params([0.0, 1.0], [1.0, -0.5])
    assert False, "should reject non-positive y"
except ValueError:
    pass

# 8. classify_sample: all four parity x side combinations, on-model data
t0 = 5.0
assert classify_sample(index=0, t=2.0, t0=t0, A=A, k=k) == "even_left"    # even, t<t0
assert classify_sample(index=0, t=8.0, t0=t0, A=A, k=k) == "even_right"   # even, t>t0
assert classify_sample(index=1, t=2.0, t0=t0, A=A, k=k) == "odd_left"     # odd, t<t0
assert classify_sample(index=1, t=8.0, t0=t0, A=A, k=k) == "odd_right"   # odd, t>t0

# 9. a sample that fits the model exactly should NOT be flagged "other"
t_ok = 4.0
y_ok = exponential_decay(A, k, t_ok)
assert classify_sample(index=2, t=t_ok, t0=t0, A=A, k=k, y_observed=y_ok) == "even_left"

# 10. a genuine outlier (3x the model prediction) MUST be flagged "other"
y_outlier = exponential_decay(A, k, t_ok) * 3.0
assert classify_sample(index=2, t=t_ok, t0=t0, A=A, k=k, y_observed=y_outlier, residual_tol=0.1) == "other"

# 11. classify_batch: counts sum to the total, and match individual classify_sample calls
indices = np.arange(20)
t_samples = np.linspace(0, 10, 20)
y_clean_batch = exponential_decay(A, k, t_samples)
y_clean_batch[3] *= 3.0   # inject one outlier
result = classify_batch(indices, t_samples, t0, A, k, y_observed=y_clean_batch)
assert sum(result["counts"].values()) == 20
assert result["labels"][3] == "other"
for n in range(20):
    if n == 3:
        continue
    expected = classify_sample(int(indices[n]), float(t_samples[n]), t0, A, k, float(y_clean_batch[n]))
    assert result["labels"][n] == expected

# 12. classify_batch rejects mismatched-length inputs
try:
    classify_batch([0, 1], [0.0, 1.0, 2.0], t0, A, k)
    assert False, "should reject mismatched indices/t_arr lengths"
except ValueError:
    pass

# 13. input validation on the core model
try:
    exponential_decay(-1.0, k, 0.0)
    assert False, "should reject non-positive A"
except ValueError:
    pass

print("all dgs.exp_decay_classifier tests passed")
