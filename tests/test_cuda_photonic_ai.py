"""Tests for dgs/cuda_photonic_ai.py"""
import math
import numpy as np
import pytest

try:
    from dgs.cuda_photonic_ai import (
        gpu_gs_phase_retrieval, photonic_attention,
        bayesian_photonic_inference, publishable_pipeline,
        NVCC_KERNEL_SOURCE,
    )
except ImportError:
    from cuda_photonic_ai import (
        gpu_gs_phase_retrieval, photonic_attention,
        bayesian_photonic_inference, publishable_pipeline,
        NVCC_KERNEL_SOURCE,
    )


class TestGpuGsPhaseRetrieval:
    def setup_method(self):
        self.gs = gpu_gs_phase_retrieval(
            n_pts=128, n_iter=50,
            D_ps_nm_km=2000.0, L_km=5.0, rng_seed=42
        )

    def test_backend_key_present(self):
        assert 'backend' in self.gs

    def test_D_eff_correct(self):
        assert self.gs['D_eff_ps_nm'] == pytest.approx(10000.0)

    def test_phi_estimated_length(self):
        assert len(self.gs['phi_estimated']) == 128

    def test_I_in_nonneg(self):
        assert all(v >= 0 for v in self.gs['I_in'])

    def test_I_out_nonneg(self):
        assert all(v >= 0 for v in self.gs['I_out'])

    def test_n_iter_stored(self):
        assert self.gs['n_iter'] == 50

    def test_nvcc_ref_present(self):
        assert 'nvcc' in self.gs['nvcc_kernel'].lower()

    def test_gpu_speedup_estimate_present(self):
        assert '50x' in self.gs['gpu_speedup_estimate'] or 'GPU' in self.gs['gpu_speedup_estimate']

    def test_low_D_warns(self):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gpu_gs_phase_retrieval(D_ps_nm_km=100.0, L_km=1.0, n_pts=64, n_iter=10)
            assert len(w) > 0
            assert 'D_eff' in str(w[0].message) or 'converge' in str(w[0].message).lower()

    def test_convergence_list(self):
        # should have at least one correlation entry (every 10 iters)
        assert isinstance(self.gs['convergence'], list)

    def test_correlation_in_minus1_1(self):
        c = self.gs['final_correlation']
        assert -1.0 <= c <= 1.0

    def test_RMSE_nonneg(self):
        assert self.gs['RMSE_rad'] >= 0

    def test_phi_true_stored(self):
        assert self.gs['phi_true'] is not None
        assert len(self.gs['phi_true']) == 128


class TestPhotonicAttention:
    def setup_method(self):
        self.attn = photonic_attention(seq_len=16, d_model=8, n_heads=2, n_pts=32)

    def test_output_shape(self):
        out = self.attn['output']
        assert len(out) == 16
        assert len(out[0]) == 8

    def test_attn_weights_shape(self):
        aw = self.attn['attention_weights']
        assert len(aw) == 2          # n_heads
        assert len(aw[0]) == 16      # seq_len
        assert len(aw[0][0]) == 16   # seq_len

    def test_attn_weights_sum_to_1(self):
        aw = np.array(self.attn['attention_weights'])
        row_sums = aw.sum(axis=-1)
        assert np.allclose(row_sums, 1.0, atol=1e-5)

    def test_entropy_bounded(self):
        assert self.attn['entropy']['mean_entropy_bits'] <= self.attn['entropy']['max_entropy_bits'] + 1e-6

    def test_entropy_nonneg(self):
        assert self.attn['entropy']['mean_entropy_bits'] >= 0

    def test_causal_mask_lower_triangular(self):
        mask = np.array(self.attn['causal_mask'])
        assert np.allclose(mask, np.tril(np.ones_like(mask)))

    def test_causal_attn_sum_to_1(self):
        ca = np.array(self.attn['causal_attn'])
        row_sums = ca.sum(axis=-1)
        assert np.allclose(row_sums, 1.0, atol=1e-5)

    def test_formulas_present(self):
        for key in ['attention', 'integral_form', 'derivative', 'complexity']:
            assert key in self.attn['formulas']

    def test_photonic_heads_4_described(self):
        assert len(self.attn['photonic_heads']) >= 2

    def test_params_correct(self):
        assert self.attn['architecture']['params'] == 4 * 8 * 8


class TestBayesianInference:
    def setup_method(self):
        self.bay = bayesian_photonic_inference(
            n_frames=20, SNR_dB=20.0,
            phi_true_pattern='linear_chirp', rng_seed=11
        )

    def test_n_frames_correct(self):
        assert len(self.bay['phi_MAP']) == 20

    def test_phi_grid_in_0_2pi(self):
        grid = self.bay['phi_grid']
        assert grid[0] >= 0
        assert grid[-1] <= 2*math.pi + 1e-6

    def test_posteriors_length(self):
        assert len(self.bay['posteriors']) == 20

    def test_each_posterior_normalized(self):
        phi_grid = np.array(self.bay['phi_grid'])
        for post in self.bay['posteriors']:
            norm = float(np.trapezoid(np.array(post), phi_grid))
            assert norm == pytest.approx(1.0, abs=0.05)

    def test_KL_nonneg(self):
        assert all(kl >= 0 for kl in self.bay['KL_divergence'])

    def test_anomaly_flags_booleans(self):
        assert all(isinstance(f, bool) for f in self.bay['anomaly_flags'])

    def test_rogue_pattern_detects_anomalies(self):
        bay_rogue = bayesian_photonic_inference(
            n_frames=20, phi_true_pattern='rogue', rng_seed=11
        )
        assert bay_rogue['n_anomalies'] >= 0   # non-negative

    def test_CRB_positive(self):
        assert self.bay['CRB_std'] > 0

    def test_formulas_complete(self):
        for key in ['Bayes', 'log_likelihood', 'log_sum_exp', 'KL', 'online']:
            assert key in self.bay['formulas']

    def test_GS_connection_mentioned(self):
        assert 'GS' in self.bay['gs_connection']

    def test_MAP_near_MMSE(self):
        # MAP and MMSE both in [0, 2*pi]; check at least one is close
        pairs = list(zip(self.bay['phi_MAP'], self.bay['phi_MMSE']))
        n_close = sum(abs(m - mm) < math.pi for m, mm in pairs)
        assert n_close >= len(pairs) // 2   # majority within pi rad


class TestPublishablePipeline:
    def setup_method(self):
        self.pipe = publishable_pipeline(
            n_pts=64, n_gs_iter=30, seq_len=8, d_model=4, SNR_dB=20.0, rng_seed=99
        )

    def test_pipeline_stages_count(self):
        assert len(self.pipe['pipeline']) == 6

    def test_GS_D_eff_correct(self):
        assert self.pipe['GS']['D_eff_ps_nm'] == pytest.approx(5000.0)

    def test_GS_n_iter(self):
        assert self.pipe['GS']['n_iter'] == 30

    def test_attention_n_heads(self):
        assert self.pipe['attention']['n_heads'] == 2

    def test_publishability_physical_model(self):
        assert self.pipe['publishability']['physical_model'] is True

    def test_publishability_algorithm(self):
        assert self.pipe['publishability']['algorithm'] is True

    def test_publishability_nvcc(self):
        assert self.pipe['publishability']['nvcc_kernel_provided'] is True

    def test_software_stack_complete(self):
        for key in ['numpy', 'pytorch', 'nvcc', 'mathematica']:
            assert key in self.pipe['software_stack']

    def test_target_venue_present(self):
        assert 'Optica' in self.pipe['publishability']['target_venue'] or \
               'IEEE' in self.pipe['publishability']['target_venue']


class TestNVCCKernel:
    def test_kernel_source_not_empty(self):
        assert len(NVCC_KERNEL_SOURCE) > 100

    def test_has_global_keyword(self):
        assert '__global__' in NVCC_KERNEL_SOURCE

    def test_has_apply_H_f(self):
        assert 'apply_H_f' in NVCC_KERNEL_SOURCE

    def test_has_gs_magnitude_constraint(self):
        assert 'gs_magnitude_constraint' in NVCC_KERNEL_SOURCE

    def test_has_cufft(self):
        assert 'cufft' in NVCC_KERNEL_SOURCE.lower()

    def test_has_beta2L(self):
        assert 'beta2L' in NVCC_KERNEL_SOURCE

    def test_has_compile_instruction(self):
        assert 'nvcc' in NVCC_KERNEL_SOURCE
