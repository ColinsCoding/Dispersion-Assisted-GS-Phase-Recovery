"""
Neural surrogate for a nanophotonic transmission spectrum: a DNN that replaces the Maxwell solver.

In plain words
--------------
Given the *geometry* of a thin-film optical element (here: the thickness of each layer in a
stack), the transmission spectrum T(lambda) comes from solving Maxwell's equations. For a stack
that is cheap (1-D) but for a real 2-D metasurface it can take minutes to hours per design.
The idea reviewed in Jalali et al. [1] (their ref. [26]) is to run the slow solver many times
once, train a neural network on (geometry -> spectrum) pairs, and then use the network as the
physics model: one forward pass instead of one field solve.

This module is the smallest honest version of that idea:

* ``tmm``            exact Maxwell solution at normal incidence (transfer-matrix method). It is
                     the "slow, trusted" teacher, and it is checked against closed forms.
* ``make_dataset``   random layer thicknesses -> T(lambda) on a wavelength grid.
* ``Surrogate``      small MLP (torch). Uses CUDA when available, CPU otherwise.
* ``fit``            train; ``evaluate`` reports error on held-out geometries.

Complex-conjugate / bra-ket note
--------------------------------
The solver returns a complex amplitude t. The power transmitted is T = t t* = |t|^2, i.e. the
inner product <t|t> of the amplitude with its own conjugate. The network never sees t; it only
learns the real quantity T, so it cannot recover phase. Phase (and hence group delay, which is
what time-stretch imaging cares about) comes from the solver: ``group_delay`` = d(arg t)/d(omega).
Recovering that phase from intensity alone is the phase-retrieval problem the rest of this
repo is about.

Source (IEEE style; paraphrased, not quoted)
--------------------------------------------
[1] B. Jalali, A. Kadambi and V. Roychowdhury, "Physics-AI Symbiosis," arXiv:2109.05959, 2021.
"""
from __future__ import annotations

import numpy as np

C0 = 299_792_458.0
N_HI, N_LO = 2.30, 1.45          # TiO2-like / SiO2-like indices, held constant (no dispersion)


def tmm(thickness_nm, wavelengths_nm, n_layers=None, n_in=1.0, n_out=1.0):
    """Normal-incidence transfer-matrix solution.

    thickness_nm: (L,) layer thicknesses.  wavelengths_nm: (W,).
    n_layers: (L,) indices; default alternates N_HI, N_LO.
    Returns complex (t, r) arrays of shape (W,). Lossless layers => |r|^2 + |t|^2 = 1
    when n_in == n_out.
    """
    d = np.asarray(thickness_nm, float)
    lam = np.atleast_1d(np.asarray(wavelengths_nm, float))
    if d.ndim != 1 or d.size == 0 or np.any(d < 0):
        raise ValueError('thickness_nm must be a non-empty 1-D array of non-negative values')
    if np.any(lam <= 0):
        raise ValueError('wavelengths must be positive')
    n = np.array([N_HI if i % 2 == 0 else N_LO for i in range(d.size)]) if n_layers is None \
        else np.asarray(n_layers, float)
    if n.shape != d.shape:
        raise ValueError('n_layers must match thickness_nm')
    t = np.empty(lam.size, complex)
    r = np.empty(lam.size, complex)
    for k, wl in enumerate(lam):
        M = np.eye(2, dtype=complex)
        for nj, dj in zip(n, d):
            delta = 2 * np.pi * nj * dj / wl
            M = M @ np.array([[np.cos(delta), 1j * np.sin(delta) / nj],
                              [1j * nj * np.sin(delta), np.cos(delta)]])
        B, Cc = M @ np.array([1.0, n_out])
        denom = n_in * B + Cc
        t[k] = 2 * n_in / denom
        r[k] = (n_in * B - Cc) / denom
    return t, r


def transmission(thickness_nm, wavelengths_nm, **kw):
    """T = t * conj(t) (real part), the <t|t> inner product."""
    t, _ = tmm(thickness_nm, wavelengths_nm, **kw)
    return (t * np.conj(t)).real * kw.get('n_out', 1.0) / kw.get('n_in', 1.0)


def group_delay(thickness_nm, wavelengths_nm, **kw):
    """Group delay in seconds, tau = -d(arg t)/d(omega), by finite difference in omega.

    The transfer matrix above uses the engineering e^{+j omega t} convention, so a delay tau
    appears as phase -omega*tau and the minus sign makes tau positive for a delaying slab.
    """
    lam = np.asarray(wavelengths_nm, float)
    omega = 2 * np.pi * C0 / (lam * 1e-9)
    t, _ = tmm(thickness_nm, lam, **kw)
    phi = np.unwrap(np.angle(t))
    return -np.gradient(phi, omega)


def make_dataset(n_samples, n_layers=6, wavelengths_nm=None, d_range=(40.0, 220.0), seed=0):
    """Random geometries X (S, L) in nm and spectra Y (S, W) in [0, 1]."""
    if n_samples <= 0 or n_layers <= 0:
        raise ValueError('n_samples and n_layers must be positive')
    wl = np.linspace(500, 900, 64) if wavelengths_nm is None else np.asarray(wavelengths_nm, float)
    rng = np.random.default_rng(seed)
    X = rng.uniform(*d_range, size=(n_samples, n_layers))
    Y = np.stack([transmission(x, wl) for x in X])
    return X, Y, wl


class Surrogate:
    """MLP geometry -> spectrum. torch is imported lazily so the solver works without it."""

    def __init__(self, n_in, n_out, hidden=128, depth=3, device=None):
        import torch
        from torch import nn
        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        layers, w = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(w, hidden), nn.GELU()]
            w = hidden
        layers += [nn.Linear(w, n_out), nn.Sigmoid()]          # T lives in [0, 1]
        self.net = nn.Sequential(*layers).to(self.device)
        self.x_lo = self.x_hi = None

    def _prep(self, X):
        X = np.asarray(X, np.float32)
        return self.torch.tensor((X - self.x_lo) / (self.x_hi - self.x_lo), device=self.device)

    def fit(self, X, Y, epochs=400, lr=3e-3, batch=128, seed=0):
        torch = self.torch
        torch.manual_seed(seed)
        X = np.asarray(X, np.float32)
        self.x_lo, self.x_hi = X.min(0), X.max(0)
        xt, yt = self._prep(X), torch.tensor(np.asarray(Y, np.float32), device=self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        n = len(xt)
        for _ in range(epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                loss = torch.mean((self.net(xt[idx]) - yt[idx]) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
        return self

    def predict(self, X):
        with self.torch.no_grad():
            return self.net(self._prep(np.atleast_2d(X))).cpu().numpy()

    def evaluate(self, X, Y):
        """Held-out RMS and max error of T (absolute, in [0, 1])."""
        err = self.predict(X) - np.asarray(Y)
        return {'rmse': float(np.sqrt(np.mean(err ** 2))), 'max_abs': float(np.abs(err).max())}


if __name__ == '__main__':
    X, Y, wl = make_dataset(1500)
    s = Surrogate(X.shape[1], Y.shape[1]).fit(X[:1200], Y[:1200])
    print(s.device, s.evaluate(X[1200:], Y[1200:]))
