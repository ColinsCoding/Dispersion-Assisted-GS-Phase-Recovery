"""SciNet: an encoder-decoder that compresses observations of a physical system
into a few latent numbers, then answers a question from them.

    observation o  --encoder E-->  latent z (few neurons)  --decoder D(z, q)-->  answer a

This is Fig. 4(b) of Jalali, Kadambi and Roychowdhury, "Physics-AI Symbiosis"
(arXiv:2109.05959), following R. Iten et al., "Discovering physical concepts
with neural networks," Phys. Rev. Lett. 124, 010508 (2020). The latent layer is
a beta-VAE bottleneck: a KL penalty switches off latent neurons the decoder
does not need, so the number of ACTIVE neurons estimates how many parameters
the system really has.

Three synthetic systems, each with a known answer to compare the latent against:

    uniform motion      o = x(t_i) = x0 + v t_i            q = t'     2 parameters (x0, v)
    damped oscillator   o = x(t_i), x'' + b x' + k x = 0   q = t'     2 parameters (k, b)
    relativistic Doppler  o = f(theta_i) / f0              q = theta' 1 parameter (beta)
                        f / f0 = 1 / (gamma (1 - beta cos theta))    (source frame -> lab;
                        theta measured in the lab; theta = 90 deg is the transverse
                        Doppler shift f0 / gamma = pure time dilation)

What the network does and doesn't find: it finds a REPRESENTATION (the latent
neurons become invertible functions of the true parameters). It does not write
the equation x = x0 + v t; that last step still needs a human or a symbolic-
regression step. The analysis functions below measure exactly how well each
latent neuron tracks each true parameter.
"""
import numpy as np
import sympy as sp
import torch
from torch import nn


# ── symbolic physics ─────────────────────────────────────────────────

def doppler_symbolic():
    """Lorentz-transform the photon 4-wavevector (omega/c, k cos, k sin, 0) from the
    source rest frame to the lab, where the source moves at beta along x. Returns
    (f_lab/f0 in terms of the lab angle theta, its value at theta = pi/2)."""
    beta, theta = sp.symbols("beta theta", real=True)
    gam = 1 / sp.sqrt(1 - beta**2)
    # lab -> source frame: omega' = gamma (omega - v k_x) = gamma omega (1 - beta cos theta)
    ratio = 1 / (gam * (1 - beta * sp.cos(theta)))            # omega_lab / omega_source
    return sp.simplify(ratio), sp.simplify(ratio.subs(theta, sp.pi / 2))


def damped_oscillator_symbolic():
    """x'' + b x' + k x = 0, x(0)=1, x'(0)=0, underdamped: write k = w^2 + b^2/4
    with w > 0 so SymPy returns the decaying-cosine form. Returns (solution,
    residual of the numeric formula used by make_data, which must be 0)."""
    tt = sp.symbols("t", real=True)
    w, b = sp.symbols("omega b", positive=True)
    k = w**2 + b**2 / 4
    xf = sp.Function("x")
    sol = sp.dsolve(xf(tt).diff(tt, 2) + b * xf(tt).diff(tt) + k * xf(tt),
                    ics={xf(0): 1, xf(tt).diff(tt).subs(tt, 0): 0})
    used = sp.exp(-b * tt / 2) * (sp.cos(w * tt) + b / (2 * w) * sp.sin(w * tt))
    return sp.Eq(sol.lhs, sp.simplify(sol.rhs)), sp.simplify(sol.rhs - used)


# ── synthetic data ───────────────────────────────────────────────────

SYSTEMS = {
    "uniform motion": {"params": ("x0", "v"), "ranges": ((-1.0, 1.0), (-1.0, 1.0)),
                       "obs_grid": np.linspace(0, 1, 10), "q_range": (0.0, 2.0)},
    "damped oscillator": {"params": ("k", "b"), "ranges": ((5.0, 10.0), (0.5, 1.0)),
                          "obs_grid": np.linspace(0, 5, 50), "q_range": (0.0, 10.0)},
    "relativistic Doppler": {"params": ("beta",), "ranges": ((0.0, 0.8),),
                             "obs_grid": np.linspace(0, np.pi, 12), "q_range": (0.0, np.pi)},
}


def _evaluate(system, p, s):
    """Answer for parameter rows p (n, n_params) at query points s (n,) or (n, m)."""
    s = np.asarray(s, float)
    if system == "uniform motion":
        return p[:, :1] + p[:, 1:2] * s if s.ndim == 2 else p[:, 0] + p[:, 1] * s
    if system == "damped oscillator":
        k, b = (p[:, 0:1], p[:, 1:2]) if s.ndim == 2 else (p[:, 0], p[:, 1])
        w = np.sqrt(k - b**2 / 4)
        return np.exp(-b * s / 2) * (np.cos(w * s) + b / (2 * w) * np.sin(w * s))
    if system == "relativistic Doppler":
        be = p[:, 0:1] if s.ndim == 2 else p[:, 0]
        return np.sqrt(1 - be**2) / (1 - be * np.cos(s))
    raise ValueError(f"unknown system {system!r}; choose from {list(SYSTEMS)}")


def make_data(system, n=20000, noise=0.0, seed=0):
    """Returns dict of float32 arrays: obs (n, n_obs), q (n, 1), a (n, 1), params (n, n_params)."""
    if system not in SYSTEMS:
        raise ValueError(f"unknown system {system!r}; choose from {list(SYSTEMS)}")
    if n < 1 or noise < 0:
        raise ValueError("need n >= 1 and noise >= 0")
    cfg = SYSTEMS[system]
    rng = np.random.default_rng(seed)
    p = np.column_stack([rng.uniform(lo, hi, n) for lo, hi in cfg["ranges"]])
    grid = np.broadcast_to(cfg["obs_grid"], (n, len(cfg["obs_grid"])))
    obs = _evaluate(system, p, grid) + noise * rng.standard_normal(grid.shape)
    q = rng.uniform(*cfg["q_range"], n)
    a = _evaluate(system, p, q)
    f32 = lambda z: np.asarray(z, np.float32)
    return {"obs": f32(obs), "q": f32(q[:, None]), "a": f32(a[:, None]), "params": p}


# ── the network ──────────────────────────────────────────────────────

class SciNet(nn.Module):
    def __init__(self, n_obs, n_latent=3, hidden=100):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_obs, hidden), nn.ELU(),
                                     nn.Linear(hidden, hidden), nn.ELU(),
                                     nn.Linear(hidden, 2 * n_latent))
        self.decoder = nn.Sequential(nn.Linear(n_latent + 1, hidden), nn.ELU(),
                                     nn.Linear(hidden, hidden), nn.ELU(),
                                     nn.Linear(hidden, 1))
        self.n_latent = n_latent

    def encode(self, obs):
        mu, logvar = self.encoder(obs).chunk(2, dim=-1)
        return mu, logvar

    def forward(self, obs, q, sample=True):
        mu, logvar = self.encode(obs)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        return self.decoder(torch.cat([z, q], dim=-1)), mu, logvar


def train(system, n_latent=3, beta_kl=1e-3, steps=4000, batch=512, lr=1e-3, n=20000,
          noise=0.0, seed=0, device="cpu", verbose=False):
    """Train SciNet on one system. Loss = MSE(answer) + beta_kl * KL(q(z|o) || N(0, I)).
    Returns (model, data, history of (step, mse))."""
    if steps < 1 or batch < 1 or beta_kl < 0:
        raise ValueError("need steps >= 1, batch >= 1, beta_kl >= 0")
    torch.manual_seed(seed)
    data = make_data(system, n=n, noise=noise, seed=seed)
    obs = torch.tensor(data["obs"], device=device)
    q = torch.tensor(data["q"], device=device)
    a = torch.tensor(data["a"], device=device)
    model = SciNet(obs.shape[1], n_latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    hist = []
    for step in range(steps):
        idx = torch.randint(0, len(obs), (batch,), generator=gen).to(device)
        pred, mu, logvar = model(obs[idx], q[idx])
        mse = ((pred - a[idx]) ** 2).mean()
        kl = 0.5 * (mu**2 + logvar.exp() - 1 - logvar).sum(-1).mean()
        loss = mse + beta_kl * kl
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 200 == 0 or step == steps - 1:
            hist.append((step, float(mse.detach())))
            if verbose:
                print(step, float(mse), float(kl))
    return model, data, hist


# ── analysis: did the latent find the physics? ───────────────────────

@torch.no_grad()
def latent_report(model, data, kl_active=0.05):
    """Per latent neuron: mean KL (active if > kl_active), and |Pearson r| with
    each true parameter. Also R^2 of predicting each true parameter from the
    ACTIVE latents with a quadratic least-squares fit (allows a smooth
    invertible map, not just a linear one), and the test MSE of the answer."""
    model.eval()
    dev = next(model.parameters()).device
    obs = torch.tensor(data["obs"], device=dev)
    mu, logvar = model.encode(obs)
    kl = (0.5 * (mu**2 + logvar.exp() - 1 - logvar)).mean(0).cpu().numpy()
    z = mu.cpu().numpy()
    p = data["params"]
    active = np.where(kl > kl_active)[0]
    corr = np.array([[abs(np.corrcoef(z[:, i], p[:, j])[0, 1]) for j in range(p.shape[1])]
                     for i in range(z.shape[1])])
    za = z[:, active]
    feats = [np.ones(len(z))] + [za[:, i] for i in range(za.shape[1])] + \
            [za[:, i] * za[:, j] for i in range(za.shape[1]) for j in range(i, za.shape[1])]
    F = np.column_stack(feats)
    r2 = []
    for j in range(p.shape[1]):
        coef, *_ = np.linalg.lstsq(F, p[:, j], rcond=None)
        r2.append(1 - np.var(p[:, j] - F @ coef) / np.var(p[:, j]))
    pred, _, _ = model(obs, torch.tensor(data["q"], device=dev), sample=False)
    mse = float(((pred.cpu().numpy() - data["a"]) ** 2).mean())
    return {"kl": kl, "active": active, "corr": corr, "r2": np.array(r2), "mse": mse, "z": z}


@torch.no_grad()
def ablation(model, data):
    """For each latent neuron, freeze it at its dataset mean and measure the
    answer MSE, relative to the answer variance. A neuron the decoder really
    uses makes this jump; an unused one leaves it near the baseline."""
    model.eval()
    dev = next(model.parameters()).device
    obs = torch.tensor(data["obs"], device=dev)
    q = torch.tensor(data["q"], device=dev)
    a = data["a"]
    mu, _ = model.encode(obs)
    var = float(np.var(a))
    base = float(((model.decoder(torch.cat([mu, q], -1)).cpu().numpy() - a) ** 2).mean()) / var
    out = []
    for i in range(mu.shape[1]):
        z = mu.clone()
        z[:, i] = z[:, i].mean()
        out.append(float(((model.decoder(torch.cat([z, q], -1)).cpu().numpy() - a) ** 2).mean()) / var)
    return base, np.array(out)
