"""
pytest suite for simulator.dispersion and simulator.gs.

Run:
    pytest tests/ -v

All tests are deterministic (no RNG) and run in <2 s on CPU.
"""

import numpy as np
import pytest
from simulator.dispersion import propagate, batch_propagate, transfer_function
from simulator.gs import td_gs, _align_global_phase, _rmse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gaussian_pulse():
    """N=4096, dt=1ps Gaussian centred at midpoint."""
    N, dt = 4096, 1e-12
    t = np.arange(N) * dt
    sigma = 50e-12
    A = np.exp(-((t - t.mean()) / sigma) ** 2).astype(complex)
    return A, t, dt


@pytest.fixture
def chirped_pulse():
    """Chirped Gaussian — non-trivial phase, analytic dispersion known."""
    N, dt = 512, 1e-12
    t = np.arange(N) * dt
    t0, sigma, chirp = t.mean(), 50e-12, 1e21   # rad/s²
    A = np.exp(-((t - t0) / sigma) ** 2) * np.exp(1j * 0.5 * chirp * (t - t0) ** 2)
    return A.astype(complex), t, dt


# ---------------------------------------------------------------------------
# transfer_function
# ---------------------------------------------------------------------------

class TestTransferFunction:
    def test_zero_dispersion_is_identity(self):
        H = transfer_function(256, 1e-12, beta2_L=0.0)
        assert np.allclose(H, 1.0), "H(ω) must be 1 when beta2_L=0"

    def test_unit_modulus(self):
        H = transfer_function(1024, 0.5e-12, beta2_L=1e-22)
        assert np.allclose(np.abs(H), 1.0, atol=1e-12), "|H(ω)| must be 1 (lossless)"

    def test_conjugate_symmetry_real_signal(self):
        # H(ω) = exp(-i β ω²/2) is even because ω² is even.
        # In centred order: DC at index N//2.  Positive bin k maps to index N//2+k,
        # negative bin k maps to index N//2-k.  Check H[N//2+k] == H[N//2-k].
        N, dt = 64, 1e-12
        H = transfer_function(N, dt, beta2_L=1e-23)
        dc = N // 2
        # k = 1 … dc-1  (avoid DC and the Nyquist bin which has no mirror)
        k = np.arange(1, dc)
        assert np.allclose(H[dc + k], H[dc - k], atol=1e-14), "H must be even in omega"

    def test_additive_dispersion(self):
        """H(beta1) * H(beta2) == H(beta1 + beta2)."""
        N, dt = 128, 1e-12
        b1, b2 = 1e-23, -3e-23
        H1 = transfer_function(N, dt, b1)
        H2 = transfer_function(N, dt, b2)
        H12 = transfer_function(N, dt, b1 + b2)
        assert np.allclose(H1 * H2, H12, atol=1e-14)


# ---------------------------------------------------------------------------
# propagate — single-signal
# ---------------------------------------------------------------------------

class TestPropagate:
    def test_round_trip(self, gaussian_pulse):
        A, t, dt = gaussian_pulse
        beta2_L = 1e-22
        A2 = propagate(propagate(A, t, beta2_L), t, -beta2_L)
        err = np.max(np.abs(A2 - A))
        assert err < 1e-10, f"Round-trip error {err:.2e} exceeds 1e-10"

    def test_parseval(self, gaussian_pulse):
        """Energy is conserved (lossless dispersive medium)."""
        A, t, _ = gaussian_pulse
        A_out = propagate(A, t, 1.5e-22)
        E_in  = np.sum(np.abs(A) ** 2)
        E_out = np.sum(np.abs(A_out) ** 2)
        assert abs(E_out / E_in - 1) < 1e-12, "Parseval violated"

    def test_zero_dispersion_identity(self, gaussian_pulse):
        A, t, _ = gaussian_pulse
        A_out = propagate(A, t, 0.0)
        assert np.allclose(A_out, A, atol=1e-12)

    def test_chirped_pulse_exact_width(self):
        """Transform-limited Gaussian dispersed by β₂L: output intensity 1/e
        half-width follows σ_out/√2 where σ_out = sqrt(σ² + (β₂L/σ)²)
        (Saleh & Teich §3.1).  The intensity envelope is exp(-2t²/σ²) so the
        1/e half-width of the INTENSITY is σ/√2, not σ."""
        N, dt = 1024, 1e-12
        t = np.arange(N) * dt
        sigma = 50e-12
        A = np.exp(-((t - t.mean()) / sigma) ** 2).astype(complex)
        beta2_L = 2e-22
        A_out = propagate(A, t, beta2_L)
        # Analytic field-envelope sigma after GVD, then intensity half-width = /sqrt(2)
        sigma_out_field = np.sqrt(sigma ** 2 + (beta2_L / sigma) ** 2)
        sigma_out_intensity_hw = sigma_out_field / np.sqrt(2)
        I_out = np.abs(A_out) ** 2
        I_out = I_out / I_out.max()
        idx = np.where(I_out > np.exp(-1))[0]
        hw_sim = (t[idx[-1]] - t[idx[0]]) / 2
        assert abs(hw_sim / sigma_out_intensity_hw - 1) < 0.05, (
            f"Intensity 1/e half-width {hw_sim*1e12:.2f} ps "
            f"vs analytic {sigma_out_intensity_hw*1e12:.2f} ps"
        )

    def test_invalid_nonuniform_axis(self, gaussian_pulse):
        A, t, dt = gaussian_pulse
        t_bad = t.copy()
        t_bad[100] += dt * 0.01   # 1% of dt — well above rtol=1e-6 tolerance
        with pytest.raises(ValueError, match="uniformly spaced"):
            propagate(A, t_bad, 1e-22)

    def test_invalid_shape_mismatch(self, gaussian_pulse):
        A, t, _ = gaussian_pulse
        with pytest.raises(ValueError, match="does not match"):
            propagate(A, t[:-1], 1e-22)

    def test_2d_input_raises(self, gaussian_pulse):
        A, t, _ = gaussian_pulse
        with pytest.raises(ValueError, match="1-D"):
            propagate(A.reshape(64, -1), t, 1e-22)


# ---------------------------------------------------------------------------
# batch_propagate
# ---------------------------------------------------------------------------

class TestBatchPropagate:
    def test_single_signal_many_betas(self, gaussian_pulse):
        A, t, _ = gaussian_pulse
        betas = np.array([-2e-22, -1e-22, 0.0, 1e-22, 2e-22])
        out = batch_propagate(A, t, betas)
        assert out.shape == (5, len(A))
        # Each row must match the scalar propagate result
        for k, b in enumerate(betas):
            ref = propagate(A, t, b)
            assert np.allclose(out[k], ref, atol=1e-12), f"Mismatch at beta index {k}"

    def test_many_signals_scalar_beta(self, gaussian_pulse):
        A, t, _ = gaussian_pulse
        B = 4
        signals = np.stack([A * (k + 1) for k in range(B)])
        beta2_L = 1e-22
        out = batch_propagate(signals, t, beta2_L)
        assert out.shape == (B, len(A))
        for k in range(B):
            ref = propagate(signals[k], t, beta2_L)
            assert np.allclose(out[k], ref, atol=1e-12)

    def test_many_signals_many_betas_elementwise(self, gaussian_pulse):
        A, t, _ = gaussian_pulse
        B = 3
        signals = np.stack([A * (k + 1) for k in range(B)])
        betas = np.array([-1e-22, 0.0, 1e-22])
        out = batch_propagate(signals, t, betas)
        assert out.shape == (B, len(A))
        for k in range(B):
            ref = propagate(signals[k], t, betas[k])
            assert np.allclose(out[k], ref, atol=1e-12)

    def test_batch_parseval(self, gaussian_pulse):
        A, t, _ = gaussian_pulse
        betas = np.linspace(-2e-22, 2e-22, 16)
        out = batch_propagate(A, t, betas)
        E_in  = np.sum(np.abs(A) ** 2)
        for row in out:
            assert abs(np.sum(np.abs(row) ** 2) / E_in - 1) < 1e-11

    def test_scalar_1d_matches_propagate(self, gaussian_pulse):
        A, t, _ = gaussian_pulse
        out_batch = batch_propagate(A, t, 1e-22)
        out_single = propagate(A, t, 1e-22)
        assert np.allclose(out_batch, out_single, atol=1e-12)


# ---------------------------------------------------------------------------
# TD-GS — simulator.gs
# ---------------------------------------------------------------------------

class TestTdGs:
    """End-to-end TD-GS recovery with a small grid for speed."""

    @pytest.fixture
    def small_problem(self):
        # Design rule: sigma_disp = |beta2_L| / sigma must satisfy
        #   sigma << sigma_disp < T_window / 4  (no aliasing, meaningful diversity)
        # N=2048, dt=1ps, T=2048ps, sigma=100ps
        #   sigma_disp1 = 3e-20 / 100e-12 = 300ps < 512ps ✓, >> 100ps ✓
        #   sigma_disp2 = 4.5e-20 / 100e-12 = 450ps < 512ps ✓  (ratio 1.5×)
        N, dt = 2048, 1e-12
        t = np.arange(N) * dt
        sigma = 100e-12
        chirp = 5e19
        t0 = t.mean()
        u_true = (
            np.exp(-((t - t0) / sigma) ** 2)
            * np.exp(1j * 0.5 * chirp * (t - t0) ** 2)
        )
        beta2_L1, beta2_L2 = -3e-20, -4.5e-20   # ratio 1.5×, fits in window
        v1 = propagate(u_true, t, beta2_L1)
        v2 = propagate(u_true, t, beta2_L2)
        I1, I2 = np.abs(v1) ** 2, np.abs(v2) ** 2
        return u_true, I1, I2, t, beta2_L1, beta2_L2

    def test_output_keys(self, small_problem):
        u_true, I1, I2, t, b1, b2 = small_problem
        res = td_gs(I1, I2, t, b1, b2, n_restarts=2, n_iter=50,
                    u_true=u_true, rng=np.random.default_rng(42))
        assert set(res.keys()) == {"u_best", "residual", "rmse", "rmse_history", "res_history"}

    def test_modulus_constraints_satisfied(self, small_problem):
        u_true, I1, I2, t, b1, b2 = small_problem
        res = td_gs(I1, I2, t, b1, b2, n_restarts=4, n_iter=150,
                    rng=np.random.default_rng(0))
        u_best = res["u_best"]
        v1 = propagate(u_best, t, b1)
        v2 = propagate(u_best, t, b2)
        rel_err1 = np.sqrt(np.mean((np.abs(v1) ** 2 - I1) ** 2)) / (I1.max() + 1e-30)
        rel_err2 = np.sqrt(np.mean((np.abs(v2) ** 2 - I2) ** 2)) / (I2.max() + 1e-30)
        assert rel_err1 < 0.05, f"Channel 1 intensity residual {rel_err1:.3f} > 5%"
        assert rel_err2 < 0.05, f"Channel 2 intensity residual {rel_err2:.3f} > 5%"

    def test_true_field_is_fixed_point(self, small_problem):
        """The ground-truth field must be a fixed point of GS.

        If u_true satisfies both intensity constraints, one GS iteration
        starting at u_true should leave it unchanged (up to floating-point
        noise).  This tests algorithmic correctness, not convergence from
        random restarts (which depends on the problem conditioning).
        """
        u_true, I1, I2, t, b1, b2 = small_problem
        from simulator.gs import _residual
        from simulator.dispersion import propagate

        amp1 = np.sqrt(np.maximum(I1, 0.0))
        amp2 = np.sqrt(np.maximum(I2, 0.0))

        # One full GS step starting from u_true
        u = u_true.copy()
        v1 = propagate(u, t, b1)
        mag1 = np.abs(v1); safe = mag1 > 1e-30 * mag1.max()
        v1[safe] = amp1[safe] * v1[safe] / mag1[safe]
        u = propagate(v1, t, -b1)
        v2 = propagate(u, t, b2)
        mag2 = np.abs(v2); safe = mag2 > 1e-30 * mag2.max()
        v2[safe] = amp2[safe] * v2[safe] / mag2[safe]
        u_after = propagate(v2, t, -b2)

        rel_change = np.sqrt(np.mean(np.abs(u_after - u_true) ** 2)) / np.sqrt(np.mean(np.abs(u_true) ** 2))
        assert rel_change < 1e-6, (
            f"u_true is not a fixed point of GS: relative change = {rel_change:.2e}"
        )

    def test_negative_intensity_raises(self, small_problem):
        _, I1, I2, t, b1, b2 = small_problem
        I1_bad = I1.copy(); I1_bad[0] = -1.0
        with pytest.raises(ValueError, match="non-negative"):
            td_gs(I1_bad, I2, t, b1, b2)

    def test_res_history_length(self, small_problem):
        _, I1, I2, t, b1, b2 = small_problem
        n_restarts = 3
        res = td_gs(I1, I2, t, b1, b2, n_restarts=n_restarts, n_iter=10)
        assert len(res["res_history"]) == n_restarts


# ---------------------------------------------------------------------------
# Global-phase alignment helpers
# ---------------------------------------------------------------------------

class TestPhaseAlignment:
    def test_identity_when_aligned(self):
        rng = np.random.default_rng(7)
        u = rng.standard_normal(64) + 1j * rng.standard_normal(64)
        u_aligned = _align_global_phase(u, u)
        assert np.allclose(u_aligned, u, atol=1e-14)

    def test_rotated_field_aligns(self):
        rng = np.random.default_rng(8)
        u_ref = rng.standard_normal(64) + 1j * rng.standard_normal(64)
        phi0 = 1.23456
        u_rot = u_ref * np.exp(1j * phi0)
        u_aligned = _align_global_phase(u_rot, u_ref)
        assert np.allclose(u_aligned, u_ref, atol=1e-13), "Global phase rotation not removed"

    def test_rmse_zero_for_global_rotation(self):
        rng = np.random.default_rng(9)
        u = rng.standard_normal(128) + 1j * rng.standard_normal(128)
        for phi in [0.0, 0.7, np.pi, -2.1]:
            assert _rmse(u * np.exp(1j * phi), u) < 1e-12
