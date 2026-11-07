"""
desk_physics.py - the bench behind the lab-desk simulator.

In plain words: a short laser pulse goes through a gas cell, which eats a few narrow slices of its
spectrum. The light is split into two fibers with different amounts of dispersion (D1, D2), each
ending in a photodiode, and an oscilloscope can only record the two intensity traces |E|^2. The
time-domain Gerchberg-Saxton algorithm (TDGSA) bounces between the two traces through the known
dispersion and recovers the phase that the photodiodes threw away.

Physics choices, and why:
  * Dispersion in ps/nm at 1550 nm is converted to beta2*L (s^2) with beta2*L = -D lambda^2 / (2 pi c),
    then applied as exp(i beta2 L omega^2 / 2), the same propagator as dgs.temporal_gs.
  * The gas cell is causal: each line is a complex Lorentzian, H = exp(-d g / (g + i(f - f0))), so the
    absorption and the phase it adds are Kramers-Kronig partners (pole in one half-plane). The 2024
    slides listed a non-KK absorption model as a suspect; `causal=False` reproduces that model.
  * Photodiode + scope bandwidth is a Gaussian low-pass on the intensity, and noise is additive on the
    intensity. Both default to ideal; they are the knobs that make the recovery break.

The iteration in `TDGSARun.step` is the algorithm of dgs.temporal_gs.temporal_gs (Solli, Gupta and
Jalali, APL 2009), written as a stepper so the desk can animate it.
"""
from __future__ import annotations

import numpy as np

LAMBDA = 1550e-9
C = 2.99792458e8
D_STEPS = [0.5, 1, 2, 5, 10, 20, 40, 80, 150, 353, 600, 872]       # ps/nm, knob detents
SCOPE_BW = [("8 GHz", 8e9), ("33 GHz", 33e9), ("110 GHz", 110e9), ("1 THz", 1e12), ("ideal", None)]
NOISE = [0.0, 1e-3, 1e-2]


def beta2L_from_psnm(D_psnm, lam=LAMBDA):
    """Accumulated GVD beta2*L (s^2) for a fiber with total dispersion D (ps/nm)."""
    return -D_psnm * 1e-3 * lam ** 2 / (2 * np.pi * C)


class Bench:
    """The optical bench: pulse, gas cell, two dispersive arms, photodiodes, scope."""

    def __init__(self, N=2 ** 13, dt=0.5e-12, T0=1.5e-12,
                 lines=((-30e9, 4e9, 1.2), (10e9, 3e9, 0.9), (45e9, 5e9, 1.5))):
        self.N, self.dt, self.T0, self.lines = N, dt, T0, lines
        self.t = (np.arange(N) - N // 2) * dt
        self.f = np.fft.fftfreq(N, dt)
        self.omega = 2 * np.pi * self.f
        self.D1, self.D2 = 1.0, 40.0          # ps/nm (the near-field pair that works)
        self.causal = True
        self.bw = None                        # Hz, None = ideal detector + scope
        self.noise = 0.0                      # relative to peak intensity
        self.stretch = 1                      # time-stretch factor M of the THz probe front end
        self.seed = 7

    @property
    def effective_bw(self):
        """Time-stretch (photonic ADC) front end: stretching by M slows the signal, so a scope of
        bandwidth B records it as if it had bandwidth M*B (ideal linear stretch, no dispersion penalty)."""
        return None if self.bw is None else self.bw * self.stretch

    # ---------------------------------------------------------------- forward model
    def pulse(self):
        return np.exp(-self.t ** 2 / (2 * self.T0 ** 2)).astype(complex)

    def cell_H(self):
        H = np.ones(self.N, complex)
        for f0, w, d in self.lines:
            g = w / 2
            if self.causal:
                H *= np.exp(-d * g / (g + 1j * (self.f - f0)))
            else:
                H *= 1 - 0.6 * g ** 2 / ((self.f - f0) ** 2 + g ** 2)
        return H

    def field_after_cell(self):
        """Complex spectrum after the gas cell (the thing TDGSA tries to recover)."""
        return np.fft.fft(self.pulse()) * self.cell_H()

    def arm_field(self, D_psnm):
        b = beta2L_from_psnm(D_psnm)
        return np.fft.ifft(self.field_after_cell() * np.exp(1j * b * self.omega ** 2 / 2))

    def detect(self, intensity, salt=0):
        """Photodiode + scope: Gaussian low-pass on intensity, then additive noise."""
        I = intensity.copy()
        if self.bw:
            sigma_f = self.effective_bw / np.sqrt(2 * np.log(2))     # -3 dB at the (stretched) bandwidth
            I = np.real(np.fft.ifft(np.fft.fft(I) * np.exp(-0.5 * (self.f / sigma_f) ** 2)))
        if self.noise:
            rng = np.random.default_rng(self.seed + salt)
            I = I + self.noise * I.max() * rng.standard_normal(self.N)
        return np.maximum(I, 0.0)

    def test_point(self, name):
        """Intensity seen by a scope probe at a named test point."""
        if name == "laser":
            return np.abs(self.pulse()) ** 2
        if name == "cell":
            return np.abs(np.fft.ifft(self.field_after_cell())) ** 2
        if name == "pd1":
            return self.detect(np.abs(self.arm_field(self.D1)) ** 2, 1)
        if name == "pd2":
            return self.detect(np.abs(self.arm_field(self.D2)) ** 2, 2)
        raise KeyError(name)

    def regime(self, D_psnm):
        """Near or far field: dispersion length vs the narrowest spectral feature (paper's condition)."""
        b = abs(beta2L_from_psnm(D_psnm))
        df = min(w for _, w, _ in self.lines)
        # far field when the delay spread across one feature exceeds the pulse duration
        spread = b * 2 * np.pi * df
        return "far field" if spread > self.T0 * 4 else "near field"


class TDGSARun:
    """Step-by-step TDGSA on the two photodiode traces (algorithm of dgs.temporal_gs)."""

    def __init__(self, bench: Bench):
        self.b = bench
        self.f1 = np.sqrt(bench.test_point("pd1"))
        self.f2 = np.sqrt(bench.test_point("pd2"))
        self.D1, self.D2 = beta2L_from_psnm(bench.D1), beta2L_from_psnm(bench.D2)
        w = bench.omega
        self.H12 = np.exp(1j * (self.D2 - self.D1) * w ** 2 / 2)
        I1 = self.f1 ** 2
        t = np.arange(bench.N) * bench.dt
        tm = np.sum(t * I1) / I1.sum()
        T = np.sqrt(max(np.sum((t - tm) ** 2 * I1) / I1.sum(), bench.dt ** 2))
        self.u = self.f1 * np.exp(-1j * np.sign(self.D1 or 1) * (t - tm) ** 2 / (2 * T ** 2))
        self.it = 0
        self.mag_err, self.field_err = [], []
        self.truth = bench.field_after_cell()

    def step(self, n=1):
        for _ in range(n):
            u2 = np.fft.ifft(np.fft.fft(self.u) * self.H12)
            self.mag_err.append(float(np.sqrt(np.mean((np.abs(u2) - self.f2) ** 2)) / (self.f2.max() + 1e-30)))
            u2 = self.f2 * np.exp(1j * np.angle(u2))
            u1 = np.fft.ifft(np.fft.fft(u2) / self.H12)
            self.u = self.f1 * np.exp(1j * np.angle(u1))
            self.it += 1
            self.field_err.append(self.error())
        return self

    def spectrum(self):
        """Recovered complex spectrum after the cell (D1 removed), fitted to the truth's global phase."""
        U = np.fft.fft(self.u) * np.exp(-1j * self.D1 * self.b.omega ** 2 / 2)
        a = np.vdot(U, self.truth) / (np.vdot(U, U) + 1e-300)
        return a * U

    def error(self):
        U = self.spectrum()
        return float(np.linalg.norm(U - self.truth) / np.linalg.norm(self.truth))


# ---------------------------------------------------------------- topology probe: phase vortices
STRETCH_STEPS = [1, 4, 10, 30]


def stft(E, bench, sigma=3e-12, span=60e-12, nt=120, nf=120, fmax=120e9):
    """Gaussian-window short-time Fourier transform on a (time, frequency) grid around t = 0.
    Returns S[nt, nf], ts, fs. One matrix product, so it is fast."""
    t = bench.t
    keep = np.abs(t) < span / 2 + 6 * sigma
    t, E = t[keep], E[keep]
    ts = np.linspace(-span / 2, span / 2, nt)
    fs = np.linspace(-fmax, fmax, nf)
    G = np.exp(-(t[None, :] - ts[:, None]) ** 2 / (2 * sigma ** 2)) * E[None, :]
    S = G @ np.exp(-2j * np.pi * t[:, None] * fs[None, :]) * bench.dt
    return S, ts, fs


def vortices(S, ts=None, fs=None, floor=1e-3):
    """Phase vortices: grid plaquettes whose phase winds by 2*pi*q (q = +-1).
    A vortex is a zero of S; with a Gaussian window S is an analytic function in disguise (the
    Bargmann transform), so every vortex winds the same way. Returns [(t, f, q)] (or indices)."""
    ph = np.angle(S)
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
    w = (wrap(ph[1:, :-1] - ph[:-1, :-1]) + wrap(ph[1:, 1:] - ph[1:, :-1])
         + wrap(ph[:-1, 1:] - ph[1:, 1:]) + wrap(ph[:-1, :-1] - ph[:-1, 1:]))
    q = np.rint(w / (2 * np.pi)).astype(int)
    big = np.abs(S).max()
    out = []
    for i, j in zip(*np.nonzero(q)):
        if np.abs(S[i:i + 2, j:j + 2]).max() < floor * big:     # skip numerical noise in the empty corners
            continue
        if ts is not None:
            out.append((0.5 * (ts[i] + ts[i + 1]), 0.5 * (fs[j] + fs[j + 1]), int(q[i, j])))
        else:
            out.append((i, j, int(q[i, j])))
    return out


def vortex_match(true_v, rec_v, ts, fs, cells=2.5):
    """Fraction of true vortices that have a recovered vortex within `cells` grid cells."""
    if not true_v:
        return 1.0
    dt_, df_ = ts[1] - ts[0], fs[1] - fs[0]
    hits = sum(1 for t, f, _ in true_v
               if any(abs(t - t2) / dt_ <= cells and abs(f - f2) / df_ <= cells for t2, f2, _ in rec_v))
    return hits / len(true_v)


def vortex_braid(bench, D_values, **kw):
    """Vortex positions of the dispersed field for each D: their paths form a braid in (t, f, D)."""
    out = []
    for D in D_values:
        S, ts, fs = stft(bench.arm_field(D), bench, **kw)
        out.append((D, vortices(S, ts, fs)))
    return out


# ---------------------------------------------------------------- hand-off to MATLAB / Jupyter
MATLAB_SCRIPT = r"""%% desk_tdgsa.m - written by the lab-desk simulator
% Loads the two photodiode traces the desk captured and runs time-domain Gerchberg-Saxton.
% Same algorithm as dgs/temporal_gs.py (Solli, Gupta, Jalali, APL 2009).
load('desk_capture.mat');            % t, dt, f, I1, I2, D1_psnm, D2_psnm, truth_re, truth_im
lambda = 1550e-9; c = 299792458;
b1 = -D1_psnm*1e-3*lambda^2/(2*pi*c);  b2 = -D2_psnm*1e-3*lambda^2/(2*pi*c);
w  = 2*pi*f(:).';  H12 = exp(1i*(b2-b1)*w.^2/2);
f1 = sqrt(I1(:).'); f2 = sqrt(I2(:).');
u = f1;  err = zeros(1, n_iter);
for k = 1:n_iter
    u2 = ifft(fft(u).*H12);
    err(k) = sqrt(mean((abs(u2)-f2).^2))/max(f2);
    u2 = f2.*exp(1i*angle(u2));
    u  = f1.*exp(1i*angle(ifft(fft(u2)./H12)));
end
U = fft(u).*exp(-1i*b1*w.^2/2);  truth = truth_re + 1i*truth_im;
a = (U*truth')/(U*U'); U = conj(a)*U;
fprintf('field error vs truth: %.3f\n', norm(U-truth)/norm(truth));
fs = fftshift(f)/1e9;
subplot(2,1,1); plot(fs, fftshift(abs(truth)), 'k', fs, fftshift(abs(U)), 'r--'); xlim([-120 120]);
xlabel('frequency offset (GHz)'); ylabel('|E(f)|'); legend('true','recovered'); title('Spectrum after the gas cell');
subplot(2,1,2); semilogy(err, 'b'); xlabel('iteration'); ylabel('magnitude error'); title('TDGSA convergence');
"""


def write_capture(bench: Bench, folder, n_iter=200):
    """Save the desk's traces for MATLAB (.mat + script) and Jupyter (.npz + small notebook)."""
    import os
    os.makedirs(folder, exist_ok=True)
    I1, I2 = bench.test_point("pd1"), bench.test_point("pd2")
    truth = bench.field_after_cell()
    data = dict(t=bench.t, dt=bench.dt, f=bench.f, I1=I1, I2=I2, D1_psnm=bench.D1, D2_psnm=bench.D2,
                truth_re=truth.real, truth_im=truth.imag, n_iter=n_iter)
    np.savez(os.path.join(folder, "desk_capture.npz"), **data)
    try:
        from scipy.io import savemat
        savemat(os.path.join(folder, "desk_capture.mat"), data)
    except ImportError:
        pass
    with open(os.path.join(folder, "desk_tdgsa.m"), "w", encoding="utf-8") as fh:
        fh.write(MATLAB_SCRIPT)
    _write_notebook(os.path.join(folder, "desk_tdgsa.ipynb"))
    return folder


def _write_notebook(path):
    import nbformat
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell("# Desk capture: TDGSA\n\nThe lab-desk simulator saved the two photodiode traces it "
                          "probed. This runs the same time-domain Gerchberg-Saxton loop on them and compares "
                          "the recovered spectrum with the true one."),
        new_code_cell("import numpy as np, matplotlib.pyplot as plt\n"
                      "d = np.load('desk_capture.npz')\n"
                      "lam, c = 1550e-9, 299792458.0\n"
                      "b1, b2 = (-float(d[k])*1e-3*lam**2/(2*np.pi*c) for k in ('D1_psnm', 'D2_psnm'))\n"
                      "w = 2*np.pi*d['f']; H12 = np.exp(1j*(b2-b1)*w**2/2)\n"
                      "f1, f2 = np.sqrt(d['I1']), np.sqrt(d['I2']); u = f1.astype(complex); err = []\n"
                      "for _ in range(int(d['n_iter'])):\n"
                      "    u2 = np.fft.ifft(np.fft.fft(u)*H12); err.append(np.sqrt(np.mean((abs(u2)-f2)**2))/f2.max())\n"
                      "    u2 = f2*np.exp(1j*np.angle(u2)); u = f1*np.exp(1j*np.angle(np.fft.ifft(np.fft.fft(u2)/H12)))\n"
                      "U = np.fft.fft(u)*np.exp(-1j*b1*w**2/2); T = d['truth_re'] + 1j*d['truth_im']\n"
                      "U *= np.vdot(U, T)/np.vdot(U, U)\n"
                      "print('field error vs truth:', round(float(np.linalg.norm(U-T)/np.linalg.norm(T)), 3))"),
        new_code_cell("fs = np.fft.fftshift(d['f'])/1e9\n"
                      "fig, ax = plt.subplots(2, 1, figsize=(7, 6))\n"
                      "ax[0].plot(fs, np.fft.fftshift(abs(T)), 'k', label='true')\n"
                      "ax[0].plot(fs, np.fft.fftshift(abs(U)), 'r--', label='recovered'); ax[0].set_xlim(-120, 120)\n"
                      "ax[0].set_xlabel('frequency offset (GHz)'); ax[0].legend()\n"
                      "ax[1].semilogy(err); ax[1].set_xlabel('iteration'); ax[1].set_ylabel('magnitude error')\n"
                      "plt.tight_layout(); plt.show()\n"
                      "print('Top: true (solid black) vs recovered (dashed red) spectrum after the gas cell. '\n"
                      "      'Bottom: TDGSA magnitude error per iteration on a log scale.')"),
    ]
    nb.metadata["kernelspec"] = {"name": "py312", "display_name": "py312", "language": "python"}
    nbformat.write(nb, path)
