"""
Discovering partial differential equations from data: sparse regression and a two-network
(physics + prior) deep learning model, on heat, Burgers, Schrodinger and Black-Scholes data.

In plain words
--------------
Give a computer a movie of a field u(x, t) and ask: *which equation made this?* Two routes:

1. **Sparse regression (PDE-FIND / STRidge, Rudy et al., Sci. Adv. 2017).** Estimate u_t and a
   library of candidate terms (u, u_x, u_xx, u u_x, ...), then find the *fewest* terms whose
   weighted sum reproduces u_t. The answer is a readable equation, but the derivatives come from
   finite differences, which amplify noise.
2. **Two networks (deep hidden physics, Raissi, JMLR 2018).** One network u_net(x, t) is a smooth
   prior for the solution itself; a second N_net(u, u_x, u_xx) models the evolution, u_t = N.
   They are trained together: u_net must fit the data and obey N_net. The derivatives then come
   from automatic differentiation of a smooth network rather than from noisy differences, and a
   final sparse fit to those derivatives turns the black box into a readable equation.
   The Jalali group's review *Physics-AI symbiosis* (Mach. Learn.: Sci. Technol., 2022,
   doi:10.1088/2632-2153/ac9215) discusses this two-network idea as a route to automated PDE
   discovery, and also the cost of black-box models.

Data come from exact solutions, so the true equation is known:
``heat`` (u_t = nu u_xx), ``burgers`` (u_t = -u u_x + nu u_xx, exact via Cole-Hopf),
``schrodinger`` (i psi_t = -psi_xx / 2 + x^2 psi / 2, harmonic-oscillator eigenstates) and
``black_scholes`` (V_tau = sigma^2 S^2 V_SS / 2 + r S V_S - r V, closed-form call prices).
``kaggle_package`` writes the datasets as CSV with a Kaggle ``dataset-metadata.json``.

Teaching code: 1-D problems, synthetic data. The Black-Scholes part is a PDE-identification
exercise on model prices, not a trading or investment tool.
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np

try:
    import pandas as pd
except ImportError:          # pragma: no cover - pandas is optional for the core maths
    pd = None


# =========================================================================== data (exact solutions)
def heat(nu=0.1, nx=256, nt=101, T=1.0, modes=((1, 1.0, 0.0), (3, 0.5, 0.7), (5, 0.25, 1.3))):
    """u_t = nu u_xx on [0, 2 pi) periodic: u = sum a_k e^{-nu k^2 t} sin(k x + phase).
    Returns dict(x, t, u (nt, nx), truth)."""
    if nu <= 0:
        raise ValueError('nu must be > 0')
    x = np.linspace(0, 2 * np.pi, nx, endpoint=False); t = np.linspace(0, T, nt)
    u = sum(a * np.exp(-nu * k ** 2 * t)[:, None] * np.sin(k * x + ph)[None, :] for k, a, ph in modes)
    return dict(x=x, t=t, u=u, periodic=True, truth={'u_xx': nu})


def burgers(nu=0.1, nx=256, nt=101, T=1.0, a0=2.0, modes=((1, 1.0), (2, 0.4))):
    """Viscous Burgers u_t + u u_x = nu u_xx, exact by Cole-Hopf: u = -2 nu phi_x / phi with
    phi = a0 + sum a_k e^{-nu k^2 t} cos(k x) > 0 solving the heat equation."""
    if nu <= 0 or a0 <= sum(abs(a) for _, a in modes):
        raise ValueError('need nu > 0 and a0 > sum |a_k| (phi must stay positive)')
    x = np.linspace(0, 2 * np.pi, nx, endpoint=False); t = np.linspace(0, T, nt)
    phi = a0 + sum(a * np.exp(-nu * k ** 2 * t)[:, None] * np.cos(k * x)[None, :] for k, a in modes)
    phix = sum(-k * a * np.exp(-nu * k ** 2 * t)[:, None] * np.sin(k * x)[None, :] for k, a in modes)
    return dict(x=x, t=t, u=-2 * nu * phix / phi, periodic=True, truth={'u*u_x': -1.0, 'u_xx': nu})


def hermite_functions(n_max, x):
    """Normalised harmonic-oscillator eigenfunctions phi_0..phi_n_max (hbar = m = omega = 1)."""
    out = np.zeros((n_max + 1, len(x)))
    out[0] = np.pi ** -0.25 * np.exp(-x ** 2 / 2)
    if n_max >= 1:
        out[1] = math.sqrt(2) * x * out[0]
    for n in range(2, n_max + 1):
        out[n] = math.sqrt(2 / n) * x * out[n - 1] - math.sqrt((n - 1) / n) * out[n - 2]
    return out


def schrodinger(nx=256, nt=101, T=2.0, L=8.0, coeffs=((0, 0.8), (1, 0.5), (2, 0.33j))):
    """Harmonic oscillator i psi_t = -psi_xx/2 + x^2 psi/2: psi = sum c_n phi_n(x) e^{-i (n + 1/2) t}.
    In PDE-FIND form psi_t = (i/2) psi_xx - (i/2) x^2 psi."""
    x = np.linspace(-L, L, nx, endpoint=False); t = np.linspace(0, T, nt)
    H = hermite_functions(max(n for n, _ in coeffs), x)
    psi = sum(c * np.exp(-1j * (n + 0.5) * t)[:, None] * H[n][None, :] for n, c in coeffs)
    psi = psi / math.sqrt(np.sum(np.abs(psi[0]) ** 2) * (x[1] - x[0]))
    return dict(x=x, t=t, u=psi, periodic=True, truth={'u_xx': 0.5j, 'x^2*u': -0.5j})


def _ncdf(z):
    from scipy.special import erf
    return 0.5 * (1 + erf(z / math.sqrt(2)))


def black_scholes(r=0.05, sigma=0.2, K=100.0, S=(50.0, 150.0), tau=(0.1, 1.0), nS=201, nt=91):
    """European call prices V(S, tau) (tau = time to maturity), closed form. They satisfy
    V_tau = (sigma^2/2) S^2 V_SS + r S V_S - r V."""
    if sigma <= 0 or K <= 0:
        raise ValueError('need sigma > 0 and K > 0')
    s = np.linspace(*S, nS); tt = np.linspace(*tau, nt)
    Sg, Tg = np.meshgrid(s, tt)
    d1 = (np.log(Sg / K) + (r + sigma ** 2 / 2) * Tg) / (sigma * np.sqrt(Tg)); d2 = d1 - sigma * np.sqrt(Tg)
    V = Sg * _ncdf(d1) - K * np.exp(-r * Tg) * _ncdf(d2)
    return dict(x=s, t=tt, u=V, periodic=False, truth={'S^2*u_SS': sigma ** 2 / 2, 'S*u_S': r, 'u': -r})


def add_noise(data, level, seed=0):
    """Copy of data with Gaussian noise of std = level * std(u) (complex: on both parts)."""
    rng = np.random.default_rng(seed)
    u = data['u']; s = level * np.std(u)
    n = rng.normal(0, s, u.shape) if not np.iscomplexobj(u) else (rng.normal(0, s / math.sqrt(2), u.shape) + 1j * rng.normal(0, s / math.sqrt(2), u.shape))
    return dict(data, u=u + n, noise=level)


# =========================================================================== derivatives and library
def derivatives(data, trim=5):
    """u_t, u_x, u_xx by central differences (spectral in x when periodic). Returns dict of
    (nt - 2 trim, nx - 2 trim) arrays plus the matching x and t grids."""
    x, t, u = data['x'], data['t'], data['u']
    dt = t[1] - t[0]
    u_t = np.gradient(u, dt, axis=0, edge_order=2)
    if data.get('periodic'):
        k = 2 * np.pi * np.fft.fftfreq(len(x), x[1] - x[0])
        U = np.fft.fft(u, axis=1)
        u_x = np.fft.ifft(1j * k * U, axis=1); u_xx = np.fft.ifft(-(k ** 2) * U, axis=1)
        if not np.iscomplexobj(u):
            u_x, u_xx = u_x.real, u_xx.real
    else:
        dx = x[1] - x[0]
        u_x = np.gradient(u, dx, axis=1, edge_order=2); u_xx = np.gradient(u_x, dx, axis=1, edge_order=2)
    s = (slice(trim, -trim), slice(trim, -trim))
    X, T = np.meshgrid(x, t)
    return dict(u=u[s], u_t=u_t[s], u_x=u_x[s], u_xx=u_xx[s], x=X[s], t=T[s])


def standard_library(d, extra=None):
    """PDE-FIND library {1, u, u^2} x {1, u_x, u_xx} (|u|^2 u for complex fields), plus extras."""
    u, ux, uxx = d['u'], d['u_x'], d['u_xx']
    lib = {'1': np.ones_like(u), 'u': u, 'u_x': ux, 'u_xx': uxx}
    if np.iscomplexobj(u):
        lib.update({'|u|^2*u': np.abs(u) ** 2 * u, 'x^2*u': d['x'] ** 2 * u, 'x*u': d['x'] * u})
    else:
        lib.update({'u^2': u ** 2, 'u*u_x': u * ux, 'u*u_xx': u * uxx, 'u^2*u_x': u ** 2 * ux, 'u^2*u_xx': u ** 2 * uxx})
    if extra:
        lib.update(extra)
    return lib


def finance_library(d):
    """Library for option prices on an S grid: S-weighted derivatives as Black-Scholes would need,
    plus distractors."""
    S, V, VS, VSS = d['x'], d['u'], d['u_x'], d['u_xx']
    return {'1': np.ones_like(V), 'u': V, 'u_S': VS, 'u_SS': VSS, 'S*u_S': S * VS, 'S^2*u_SS': S ** 2 * VSS,
            'S*u': S * V, 'u^2': V ** 2, 'S*u_SS': S * VSS}


# =========================================================================== sparse regression
def stridge(lib, u_t, tol=0.02, lam=1e-6, iters=20):
    """Sequential thresholded ridge regression. A term survives if its contribution
    |w_j| ||theta_j|| / ||u_t|| is at least `tol` (scale-free). Returns {term: coefficient}."""
    names = list(lib)
    Theta = np.stack([lib[n].ravel() for n in names], axis=1)
    y = u_t.ravel()
    norms = np.linalg.norm(Theta, axis=0); norms[norms == 0] = 1.0
    Th = Theta / norms
    keep = np.ones(len(names), bool)
    w = np.zeros(len(names), dtype=np.result_type(Th, y))
    for _ in range(iters):
        A = Th[:, keep]
        wk = np.linalg.solve(A.conj().T @ A + lam * np.eye(A.shape[1]), A.conj().T @ y)
        w[:] = 0; w[keep] = wk
        contrib = np.abs(w) / np.linalg.norm(y)
        new = keep & (contrib >= tol)
        if new.sum() == 0 or np.array_equal(new, keep):
            keep = new if new.sum() else keep
            break
        keep = new
    A = Th[:, keep]
    wk = np.linalg.lstsq(A, y, rcond=None)[0]                  # unbiased refit on the survivors
    w[:] = 0; w[keep] = wk
    coef = w / norms
    return {n: (complex(c) if np.iscomplexobj(coef) else float(c)) for n, c, k in zip(names, coef, keep) if k}


def residual(lib, u_t, found):
    """Relative residual ||Theta w - u_t|| / ||u_t|| of a discovered model."""
    if not found:
        return 1.0
    Th = np.stack([lib[k].ravel() for k in found], axis=1)
    return float(np.linalg.norm(Th @ np.array(list(found.values())) - u_t.ravel()) / np.linalg.norm(u_t))


def stridge_path(lib, u_t, tols=None, slack=1.25):
    """Sweep the threshold and pick the most parsimonious model: the fewest terms whose residual
    is within `slack` times the best residual on the path (a knee of the accuracy/complexity
    curve). Returns (chosen model, list of (tol, n_terms, residual, model))."""
    tols = np.geomspace(0.005, 0.5, 16) if tols is None else tols
    path = []
    for tol in tols:
        f = stridge(lib, u_t, tol=float(tol))
        path.append((float(tol), len(f), residual(lib, u_t, f), f))
    r_best = min(p[2] for p in path)
    ok = [p for p in path if p[2] <= slack * r_best and p[1] > 0]
    chosen = min(ok, key=lambda p: (p[1], p[2]))[3]
    return chosen, path


def discover(data, library='standard', tol='auto', trim=5):
    """Finite differences + STRidge on a dataset (tol='auto': parsimony knee of the threshold
    path). Returns {term: coefficient}."""
    d = derivatives(data, trim)
    lib = finance_library(d) if library == 'finance' else standard_library(d)
    return stridge_path(lib, d['u_t'])[0] if tol == 'auto' else stridge(lib, d['u_t'], tol=tol)


def coefficient_error(found, truth):
    """Largest relative coefficient error over the true terms, and whether the term set is exact."""
    errs = [abs(found.get(k, 0) - v) / abs(v) for k, v in truth.items()]
    return dict(max_rel_err=float(max(errs)), exact_terms=set(found) == set(truth))


def as_table(found, truth=None):
    """pandas DataFrame of discovered terms (and true coefficients when given)."""
    if pd is None:
        raise ImportError('pandas is required for as_table')
    keys = sorted(set(found) | set(truth or {}), key=lambda k: (k not in (truth or {}), k))
    return pd.DataFrame({'term': keys, 'discovered': [found.get(k, 0) for k in keys],
                         'true': [(truth or {}).get(k, 0) for k in keys]})


# =========================================================================== two networks (torch)
def _torch():
    import torch
    return torch


def device(prefer_gpu=True):
    """'cuda' when a GPU is visible and wanted, else 'cpu'."""
    torch = _torch()
    return torch.device('cuda' if prefer_gpu and torch.cuda.is_available() else 'cpu')


def _mlp(n_in, n_out, width, depth):
    torch = _torch(); nn = torch.nn
    layers, d = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.Tanh()]; d = width
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


class HiddenPhysics:
    """Two networks trained together on real-valued u(x, t):
    u_net(x, t) ~ u (the prior / approximate solution) and N_net(u, u_x, u_xx) ~ u_t (the dynamics).
    Loss = mean (u_net - u_data)^2 + w_pde * mean (d u_net/dt - N_net)^2, with all derivatives of
    u_net taken by automatic differentiation."""

    def __init__(self, width=48, depth=4, n_width=32, n_depth=2, seed=0, dev=None):
        torch = _torch()
        torch.manual_seed(seed)
        self.dev = dev or device()
        self.u_net = _mlp(2, 1, width, depth).to(self.dev)
        self.N_net = _mlp(3, 1, n_width, n_depth).to(self.dev)
        self.history = []

    def _scale(self, x, t):
        return (2 * (x - self.x0) / self.xs - 1), (2 * (t - self.t0) / self.ts - 1)

    def u_and_derivs(self, x, t):
        torch = _torch()
        x = x.clone().requires_grad_(True); t = t.clone().requires_grad_(True)
        xs, ts = self._scale(x, t)
        u = self.u_net(torch.stack([xs, ts], 1)).squeeze(1) * self.us + self.um
        g = lambda f, v: torch.autograd.grad(f, v, torch.ones_like(f), create_graph=True)[0]
        u_t, u_x = g(u, t), g(u, x)
        return u, u_t, u_x, g(u_x, x)

    def fit(self, data, n_points=4000, steps=3000, lr=2e-3, w_pde=1.0, seed=0, log_every=500):
        """Train on random samples of (x, t, u) from a real-valued dataset."""
        torch = _torch()
        if np.iscomplexobj(data['u']):
            raise ValueError('HiddenPhysics handles real fields; split complex data into parts first')
        rng = np.random.default_rng(seed)
        X, T = np.meshgrid(data['x'], data['t'])
        idx = rng.choice(X.size, size=min(n_points, X.size), replace=False)
        f = lambda a: torch.tensor(a.ravel()[idx], dtype=torch.float32, device=self.dev)
        x, t, u = f(X), f(T), f(data['u'])
        self.x0, self.xs = float(data['x'].min()), float(np.ptp(data['x']))
        self.t0, self.ts = float(data['t'].min()), float(np.ptp(data['t']))
        self.um, self.us = float(np.mean(data['u'])), float(np.std(data['u']))
        params = list(self.u_net.parameters()) + list(self.N_net.parameters())
        opt = torch.optim.Adam(params, lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for k in range(steps):
            opt.zero_grad()
            uh, ut, ux, uxx = self.u_and_derivs(x, t)
            N = self.N_net(torch.stack([(uh - self.um) / self.us, ux / self.us, uxx / self.us], 1)).squeeze(1) * self.us
            l_data = torch.mean((uh - u) ** 2) / self.us ** 2
            l_pde = torch.mean((ut - N) ** 2) / self.us ** 2
            loss = l_data + w_pde * l_pde
            loss.backward(); opt.step(); sched.step()
            if k % log_every == 0 or k == steps - 1:
                self.history.append((k, l_data.item(), l_pde.item()))
        return self

    def derivative_fields(self, data, trim=5, chunk=20000):
        """u, u_t, u_x, u_xx of the trained prior on the data grid (trimmed like `derivatives`)."""
        torch = _torch()
        X, T = np.meshgrid(data['x'], data['t'])
        outs = {k: [] for k in ('u', 'u_t', 'u_x', 'u_xx')}
        xf, tf = X.ravel(), T.ravel()
        for a in range(0, xf.size, chunk):
            xx = torch.tensor(xf[a:a + chunk], dtype=torch.float32, device=self.dev)
            tt = torch.tensor(tf[a:a + chunk], dtype=torch.float32, device=self.dev)
            vals = self.u_and_derivs(xx, tt)
            for k, v in zip(outs, vals):
                outs[k].append(v.detach().cpu().numpy())
        s = (slice(trim, -trim), slice(trim, -trim))
        d = {k: np.concatenate(v).reshape(X.shape)[s].astype(float) for k, v in outs.items()}
        d.update(x=X[s], t=T[s])
        return d

    def discover(self, data, tol='auto', trim=5):
        """Sparse equation from the network's smooth derivatives (the readable form of N_net)."""
        d = self.derivative_fields(data, trim)
        lib = standard_library(d)
        return stridge_path(lib, d['u_t'])[0] if tol == 'auto' else stridge(lib, d['u_t'], tol=tol)

    def black_box_residual(self, data, trim=5):
        """How well N_net (the black box) explains the prior's own u_t: relative residual."""
        torch = _torch()
        d = self.derivative_fields(data, trim)
        f = lambda a: torch.tensor(a.ravel(), dtype=torch.float32, device=self.dev)
        with torch.no_grad():
            N = self.N_net(torch.stack([(f(d['u']) - self.um) / self.us, f(d['u_x']) / self.us, f(d['u_xx']) / self.us], 1)).squeeze(1) * self.us
        N = N.cpu().numpy()
        return float(np.linalg.norm(N - d['u_t'].ravel()) / np.linalg.norm(d['u_t']))


# =========================================================================== Kaggle packaging
def to_frame(data, name):
    """Long-format DataFrame (x, t, u) or (x, t, u_real, u_imag) for one dataset."""
    if pd is None:
        raise ImportError('pandas is required')
    X, T = np.meshgrid(data['x'], data['t'])
    df = pd.DataFrame({'x': X.ravel(), 't': T.ravel()})
    if np.iscomplexobj(data['u']):
        df['u_real'] = data['u'].real.ravel(); df['u_imag'] = data['u'].imag.ravel()
    else:
        df['u'] = data['u'].ravel()
    df.attrs['name'] = name
    return df


def kaggle_package(out_dir, datasets, title='PDE discovery benchmark: heat, Burgers, Schrodinger, Black-Scholes',
                   slug='pde-discovery-benchmark', username='YOUR-KAGGLE-USERNAME', license_name='CC0-1.0'):
    """Write one CSV per dataset, a README data card and the Kaggle CLI's dataset-metadata.json.
    Nothing is uploaded: publishing is `kaggle datasets create -p <out_dir>` after you set your
    username in the metadata."""
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    resources, lines = [], [f'# {title}', '', 'Synthetic fields from exact solutions; the true PDE of each file is listed.', '']
    def num(v):
        v = complex(v)
        if abs(v.imag) < 1e-12:
            return f'{v.real:.6g}'
        return (f'{v.imag:.6g}i' if abs(v.real) < 1e-12 else f'({v.real:.6g}{v.imag:+.6g}i)')

    for name, data in datasets.items():
        df = to_frame(data, name)
        f = out / f'{name}.csv'
        df.to_csv(f, index=False, float_format='%.10g')
        lhs = 'psi_t' if np.iscomplexobj(data['u']) else ('V_tau' if 'S*u_S' in data['truth'] else 'u_t')
        terms = ' + '.join(f'{num(v)} {k}' for k, v in data['truth'].items()).replace('+ -', '- ')
        if lhs != 'u_t':
            terms = terms.replace('u', 'psi' if lhs == 'psi_t' else 'V')
        eq = f'{lhs} = {terms}'
        resources.append({'path': f.name, 'description': eq + (f" (noise {data['noise']} of the signal std)" if 'noise' in data else '')})
        lines.append(f"- `{f.name}`: {len(df)} rows; {eq}" + (f" (noise {data['noise']})" if 'noise' in data else ''))
    lines += ['', 'Generated by dgs/pde_discovery.py (MIT license code). Black-Scholes prices are model prices, not market data.']
    (out / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    meta = {'title': title, 'id': f'{username}/{slug}', 'licenses': [{'name': license_name}], 'resources': resources}
    (out / 'dataset-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return out
