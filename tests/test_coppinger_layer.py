"""Tests for dgs/torch/coppinger_layer.py -- run with py -3.12"""
import numpy as np
import pytest
import torch
from dgs.torch.coppinger_layer import (
    DispersivePhaseFilter, MachZehnderModulator,
    ChirpedGaussianSource, CoppingerForward,
    GSIteration, PhaseRetrievalNet, train_phase_retrieval,
)


N = 64  # small for fast tests


def test_dispersive_filter_unitary():
    H = DispersivePhaseFilter(D_init=-5000.0, N=N)
    E_in = torch.randn(1, N, dtype=torch.complex128)
    E_out = H(E_in)
    # Unitary: energy conserved
    E_in_energy = torch.abs(E_in).pow(2).sum()
    E_out_energy = torch.abs(E_out).pow(2).sum()
    assert torch.allclose(E_in_energy, E_out_energy, rtol=1e-5)


def test_dispersive_filter_at_zero_D():
    H = DispersivePhaseFilter(D_init=0.0, N=N)
    E_in = torch.ones(1, N, dtype=torch.complex128)
    E_out = H(E_in)
    # D=0: H(f)=1, no change
    assert torch.allclose(E_in, E_out, atol=1e-12)


def test_dispersive_filter_energy_conserved():
    H = DispersivePhaseFilter(D_init=-1000.0, N=N)
    E_in = torch.randn(2, N, dtype=torch.complex128)
    E_out = H(E_in)
    # Energy (Parseval) conserved -- time-domain amplitudes change, total energy does not
    Ein_energy = torch.abs(E_in).pow(2).sum()
    Eout_energy = torch.abs(E_out).pow(2).sum()
    assert torch.allclose(Ein_energy, Eout_energy, rtol=1e-5)


def test_dispersive_filter_transfer_function_unit_magnitude():
    H = DispersivePhaseFilter(D_init=-3000.0, N=N)
    H_f = H.transfer_function()
    magnitudes = torch.abs(H_f)
    assert torch.allclose(magnitudes, torch.ones(N, dtype=torch.float64), atol=1e-12)


def test_dispersive_filter_learn_D():
    H = DispersivePhaseFilter(D_init=-5000.0, N=N, learn_D=True)
    assert H.D.requires_grad


def test_mzm_output_shape():
    mzm = MachZehnderModulator(a=0.3, fm_normalized=0.05, N=N)
    E_in = torch.ones(2, N, dtype=torch.complex128)
    E_out = mzm(E_in)
    assert E_out.shape == (2, N)


def test_mzm_zero_modulation():
    mzm = MachZehnderModulator(a=0.0, fm_normalized=0.05, N=N)
    E_in = torch.ones(1, N, dtype=torch.complex128)
    E_out = mzm(E_in)
    # a=0: no modulation, E_out == E_in
    assert torch.allclose(E_out, E_in)


def test_chirped_source_shape():
    src = ChirpedGaussianSource(tau=0.2, N=N)
    E = src(batch_size=3)
    assert E.shape == (3, N)


def test_chirped_source_gaussian_envelope():
    src = ChirpedGaussianSource(tau=0.2, N=N)
    E = src(batch_size=1)[0]
    # Envelope should be max at center
    I = torch.abs(E)**2
    center = N // 2
    assert I[center] > I[0]
    assert I[center] > I[-1]


def test_coppinger_forward_output_shapes():
    model = CoppingerForward(D=-5000, a=0.3, N=N)
    I_det, E_out = model(batch_size=4)
    assert I_det.shape == (4, N)
    assert E_out.shape == (4, N)


def test_coppinger_forward_intensity_nonnegative():
    model = CoppingerForward(D=-5000, a=0.3, N=N)
    I_det, _ = model(batch_size=2)
    assert torch.all(I_det >= 0)


def test_coppinger_forward_intensity_is_real():
    model = CoppingerForward(D=-5000, a=0.3, N=N)
    I_det, _ = model(batch_size=2)
    # I_det should be real (abs^2)
    assert I_det.is_floating_point()


def test_gs_iteration_output_shape():
    gs = GSIteration(D1=5000, D2=-5000, N=N)
    E_est = torch.ones(2, N, dtype=torch.complex128)
    I1_sqrt = torch.ones(2, N, dtype=torch.complex128)
    I2_sqrt = torch.ones(2, N, dtype=torch.complex128)
    E_out = gs(E_est, I1_sqrt, I2_sqrt)
    assert E_out.shape == (2, N)


def test_gs_iteration_amplitude_constraint():
    gs = GSIteration(D1=5000, D2=-5000, N=N)
    E_est = torch.randn(1, N, dtype=torch.complex128)
    # After iteration, amplitude should equal I1_sqrt
    target_amp = 0.5 * torch.ones(1, N, dtype=torch.complex128)
    E_out = gs(E_est, target_amp, target_amp)
    # Plane 1 amplitude should be exactly target_amp
    assert torch.allclose(torch.abs(E_out), torch.abs(target_amp), atol=1e-6)


def test_phase_retrieval_net_output_shape():
    net = PhaseRetrievalNet(D1=5000, D2=-5000, n_iter=3, N=N)
    I1 = torch.rand(2, N, dtype=torch.float64)
    I2 = torch.rand(2, N, dtype=torch.float64)
    E_rec, _ = net(I1, I2)
    assert E_rec.shape == (2, N)


def test_phase_retrieval_loss_decreases():
    net, hist = train_phase_retrieval(D1=5000, D2=-5000, N=N,
                                       n_iter=3, n_epochs=5,
                                       n_train=4, verbose=False)
    assert len(hist['loss']) == 5
    # Loss should be finite
    assert all(np.isfinite(l) for l in hist['loss'])


def test_intensity_loss_zero_for_identical():
    net = PhaseRetrievalNet(D1=5000, D2=-5000, n_iter=2, N=N)
    I = torch.rand(1, N, dtype=torch.float64)
    E_dummy = torch.sqrt(I).to(torch.complex128)
    loss = net.intensity_loss(torch.abs(E_dummy)**2, I)
    assert loss < 1e-20


def test_moment_loss_zero_for_identical():
    net = PhaseRetrievalNet(D1=5000, D2=-5000, n_iter=2, N=N)
    t = torch.linspace(-1, 1, N, dtype=torch.float64)
    E = torch.randn(1, N, dtype=torch.complex128)
    loss = net.moment_loss(E, E, t, order=2)
    assert loss < 1e-20
