"""Tests for projects.seals.inverse.phase_retrieval and .dispersion."""
import sys, pathlib, warnings
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from projects.seals.inverse import phase_retrieval as pr
from projects.seals.inverse import dispersion as disp
from dgs.gs_core import disperse as gs_core_disperse   # read-only reference, not modified


def _make_chirped_gaussian(N=128):
    t = torch.linspace(-8, 8, N, dtype=torch.float64)
    amp = torch.exp(-t ** 2 / (2 * 2.0 ** 2))
    phase_true = 0.5 * 0.3 * t ** 2
    return t, amp, phase_true


def test_wrapped_phase_error_handles_2pi_wraparound():
    phi_true = torch.tensor([0.1, 1.0, -2.0], dtype=torch.float64)
    phi_est = phi_true + 2 * torch.pi   # physically identical phase
    err = pr.wrapped_phase_error(phi_est, phi_true)
    assert torch.allclose(err, torch.zeros_like(err), atol=1e-10)


def test_single_measurement_warns_underdetermined():
    t, amp, phase_true = _make_chirped_gaussian(N=32)
    E_true = amp * torch.exp(1j * phase_true)
    I_meas = [disp.dispersive_operator(E_true, 5.0).abs() ** 2]
    ops = [lambda E: disp.dispersive_operator(E, 5.0)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pr.retrieve_phase(amp, I_meas, ops, n_steps=5)
    assert any("UNDERDETERMINED" in str(w.message) for w in caught)


def test_retrieve_phase_converges():
    """The measurement-matching loss (MSE on intensity, an already-small
    quantity here since amp_true is a narrow Gaussian) should decrease over
    the run and end small in absolute terms."""
    t, amp, phase_true = _make_chirped_gaussian(N=128)
    E_true = amp * torch.exp(1j * phase_true)

    D1, D2 = 0.6, -1.4
    ops = [lambda E: disp.dispersive_operator(E, D1), lambda E: disp.dispersive_operator(E, D2)]
    measurements = [op(E_true).abs() ** 2 for op in ops]

    phase_est, loss_hist = pr.retrieve_phase(amp, measurements, ops, n_steps=400, lr=0.05)

    assert loss_hist[-1] <= loss_hist[0]
    assert loss_hist[-1] < 1e-4


def test_measurement_diversity_reduces_mean_phase_error():
    """
    The scientifically meaningful check, NOT the raw MSE loss: raw intensity
    loss is small in absolute terms here regardless of phase quality (a
    narrow-amplitude field keeps every |E|^2 value small), so a low loss
    does not by itself mean phase was recovered well -- this module's own
    notebook section found exactly that (low loss, large residual phase
    error at low-amplitude points). The real test is whether adding a
    second, differently-dispersed measurement reduces mean phase error
    versus using only one.
    """
    t, amp, phase_true = _make_chirped_gaussian(N=128)
    E_true = amp * torch.exp(1j * phase_true)
    D1, D2 = 0.6, -1.4

    op1 = lambda E: disp.dispersive_operator(E, D1)
    op2 = lambda E: disp.dispersive_operator(E, D2)
    I1 = op1(E_true).abs() ** 2
    I2 = op2(E_true).abs() ** 2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # expected UNDERDETERMINED warning for the single-op case
        phase_single, _ = pr.retrieve_phase(amp, [I1], [op1], n_steps=400, lr=0.05, seed=0)
    phase_two, _ = pr.retrieve_phase(amp, [I1, I2], [op1, op2], n_steps=400, lr=0.05, seed=0)

    mean_err_single = pr.wrapped_phase_error(phase_single, phase_true).abs().mean().item()
    mean_err_two = pr.wrapped_phase_error(phase_two, phase_true).abs().mean().item()

    assert mean_err_two <= mean_err_single


def test_dispersion_operator_matches_gs_core_convention():
    """dispersion.dispersive_operator (torch) must agree with the existing
    dgs.gs_core.disperse (numpy) for the same D -- confirms this module
    genuinely connects to, rather than duplicates, the repo's existing
    time-stretch/dispersion code."""
    rng = np.random.RandomState(0)
    N = 64
    E_np = rng.normal(size=N) + 1j * rng.normal(size=N)
    D = 3.7

    E_ref = gs_core_disperse(E_np, D)
    E_mine = disp.dispersive_operator(torch.tensor(E_np, dtype=torch.complex128), D).numpy()

    assert np.allclose(E_ref, E_mine, atol=1e-10)


def test_dispersion_is_all_pass():
    for D in (-2.3, 0.0, 5.0, 100.0):
        assert disp.is_all_pass(D, N=64)
