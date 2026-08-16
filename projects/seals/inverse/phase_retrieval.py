"""
phase_retrieval.py -- generic multi-measurement phase retrieval via PyTorch autograd.

EXTENSION on top of the validated SEALS forward model -- this is generic
phase retrieval (recovering an arbitrary complex field's phase from
intensity-only measurements), which is a DIFFERENT problem from the
model-based inverse scattering in inverse_scattering.py. See that module's
docstring, and ../README.md, for the distinction.

Detector model: I = |E|^2 (measurement.intensity_measurement). A single such
measurement is invariant to a global phase shift (E -> E*exp(i*c) leaves
|E|^2 unchanged for any constant c) and, more generally, does not determine
an arbitrary phase profile uniquely. This module never claims otherwise --
retrieve_phase() warns explicitly when called with only one measurement.
"""
import warnings

import numpy as np
import torch


def retrieve_phase(amplitude, measurements, forward_operators, n_steps=400, lr=0.05,
                    seed=0, phase_init=None):
    """
    Recover phase(t) given KNOWN amplitude(t) and one or more intensity
    measurements taken through known forward operators.

    E_est = amplitude * exp(i * phase_est)   (amplitude assumed known --
                                               isolates the phase problem)
    L = sum_j || |H_j(E_est)|^2 - measurements[j] ||_2^2

    Parameters
    ----------
    amplitude : real torch.Tensor, shape (N,) -- known field amplitude
    measurements : list of real torch.Tensor, shape (N,) each -- I_j = |H_j(E_true)|^2
    forward_operators : list of callables, same length as measurements;
        each takes a complex torch.Tensor and returns a complex torch.Tensor
        (e.g. the identity, or inverse.dispersion.dispersive_operator(., D_j))
    n_steps, lr, seed : optimizer settings
    phase_init : optional initial phase guess (real torch.Tensor); defaults to
        all-zeros (flat phase)

    Returns
    -------
    phase_est : real torch.Tensor, shape (N,), detached -- the recovered phase
    loss_history : np.ndarray, shape (n_steps,)
    """
    if len(measurements) != len(forward_operators):
        raise ValueError(
            f"measurements ({len(measurements)}) and forward_operators "
            f"({len(forward_operators)}) must have the same length")
    if len(measurements) == 0:
        raise ValueError("need at least one (measurement, forward_operator) pair")
    if len(measurements) == 1:
        warnings.warn(
            "retrieve_phase() called with a single measurement/operator pair. "
            "This is an UNDERDETERMINED phase-retrieval problem: one arbitrary "
            "intensity trace does not uniquely determine phase (global-phase "
            "and, generally, further ambiguities remain). Treat any recovered "
            "phase from this call as illustrative, not a unique solution. "
            "Prefer >=2 measurements with genuine diversity (see dispersion.py).",
            stacklevel=2,
        )

    torch.manual_seed(seed)
    N = amplitude.shape[-1]
    dtype = amplitude.dtype
    if phase_init is None:
        phase_est = torch.zeros(N, dtype=dtype, requires_grad=True)
    else:
        phase_est = phase_init.clone().detach().to(dtype).requires_grad_(True)

    optimizer = torch.optim.Adam([phase_est], lr=lr)
    loss_history = []
    for _ in range(n_steps):
        optimizer.zero_grad()
        E_est = amplitude * torch.exp(1j * phase_est)
        loss = sum(
            torch.mean((H(E_est).abs() ** 2 - I_j) ** 2)
            for H, I_j in zip(forward_operators, measurements)
        )
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

    return phase_est.detach(), np.array(loss_history)


def retrieve_phase_with_history(amplitude, measurements, forward_operators, checkpoints,
                                 lr=0.05, seed=0, phase_init=None):
    """
    Same optimization as retrieve_phase, but returns the phase estimate AND
    loss at each of `checkpoints` (a sorted, strictly increasing iterable of
    cumulative step counts) instead of only the final result -- for studying
    how the fit evolves over training, e.g. whether it overfits noisy
    measurements (loss keeps decreasing past the point where the recovered
    phase stops improving against independent ground truth). See
    seals_to_tdgsa.demonstrate_autograd_overfitting for a concrete example.

    Parameters
    ----------
    Same as retrieve_phase, except n_steps is replaced by `checkpoints`.

    Returns
    -------
    history : list of (step, loss, phase_est) tuples, one per checkpoint;
        phase_est is a detached real torch.Tensor, shape (N,)
    """
    checkpoints = list(checkpoints)
    if not checkpoints or any(c <= 0 for c in checkpoints) or checkpoints != sorted(set(checkpoints)):
        raise ValueError(f"checkpoints must be positive, strictly increasing, and unique, got {checkpoints}")
    if len(measurements) != len(forward_operators):
        raise ValueError(
            f"measurements ({len(measurements)}) and forward_operators "
            f"({len(forward_operators)}) must have the same length")
    if len(measurements) == 0:
        raise ValueError("need at least one (measurement, forward_operator) pair")

    torch.manual_seed(seed)
    N = amplitude.shape[-1]
    dtype = amplitude.dtype
    if phase_init is None:
        phase_est = torch.zeros(N, dtype=dtype, requires_grad=True)
    else:
        phase_est = phase_init.clone().detach().to(dtype).requires_grad_(True)

    optimizer = torch.optim.Adam([phase_est], lr=lr)

    def compute_loss():
        E_est = amplitude * torch.exp(1j * phase_est)
        return sum(
            torch.mean((H(E_est).abs() ** 2 - I_j) ** 2)
            for H, I_j in zip(forward_operators, measurements)
        )

    history = []
    step = 0
    for target in checkpoints:
        while step < target:
            optimizer.zero_grad()
            loss = compute_loss()
            loss.backward()
            optimizer.step()
            step += 1
        with torch.no_grad():
            loss_val = compute_loss().item()
        history.append((step, loss_val, phase_est.detach().clone()))

    return history


def wrapped_phase_error(phase_est: torch.Tensor, phase_true: torch.Tensor) -> torch.Tensor:
    """
    Delta_phi = angle(exp(i*(phase_est - phase_true))), wrapped to (-pi, pi].

    Plain subtraction is wrong here: phase is only defined modulo 2*pi, so
    phase_true + 2*pi is physically identical to phase_true, and naive
    subtraction would report a large, meaningless error there.
    """
    return torch.angle(torch.exp(1j * (phase_est - phase_true)))
