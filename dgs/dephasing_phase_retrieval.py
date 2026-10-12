"""Dephasing and phase retrieval: the same problem in a qubit and in a time-stretch instrument.

A photodetector, like a computational-basis measurement of a qubit, records
only |amplitude|^2. Phase lives in the *off-diagonal* terms, the coherences,
and two things can happen to it:

  * it is DESTROYED by dephasing: random phase kicks average the coherence
    down, rho_12 -> rho_12 exp(-gamma t) (quantum), or
    <exp(i delta)> = exp(-sigma^2 / 2) for Gaussian phase noise (classical);
  * it is HIDDEN but recoverable: apply a KNOWN unitary before the
    intensity measurement so that phase turns into intensity, then invert.

The recovery step has the same structure in both worlds:

    quantum state tomography     known rotation (H, S^dag H) -> measure Z -> solve for Bloch vector
    dispersion-assisted GS       known filter H(f) = exp(j pi D f^2) -> photodiode -> iterate for phi(t)

Dispersion is a diagonal unitary in the frequency basis and the photodiode
measures in the time basis, which is the classical analog of "rotate, then
measure Z". This module checks both halves: SymPy for the qubit algebra,
NumPy plus the repo's own dgs.gs_core for the instrument. Synthetic data only.
"""
import numpy as np
import sympy as sp

from dgs import gs_core

# Okabe-Ito colorblind-safe palette (Okabe & Ito 2008), used by every figure here
OKABE_ITO = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
             "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "black": "#000000"}


# ── quantum half: dephasing and tomography ───────────────────────────

def dephasing_symbolic():
    """Pure-dephasing Lindblad equation for one qubit, solved by SymPy.
    L = sqrt(gamma/2) sigma_z  gives  d rho_12/dt = -gamma rho_12, populations frozen.
    Returns (rho12(t), d rho11/dt, <sigma_x>(t)); the middle entry is 0."""
    t, g = sp.symbols("t gamma", positive=True)
    a, b = sp.symbols("a b", positive=True)          # |psi> = a|0> + b e^{i phi}|1>
    phi = sp.symbols("phi", real=True)
    psi = sp.Matrix([a, b * sp.exp(sp.I * phi)])
    rho0 = psi * psi.H
    sz = sp.Matrix([[1, 0], [0, -1]])
    Lk = sp.sqrt(g / 2) * sz
    r = sp.Function("r")
    # the off-diagonal ODE from rho' = L rho L^dag - 1/2 {L^dag L, rho}
    lind = Lk * rho0 * Lk.H - (Lk.H * Lk * rho0 + rho0 * Lk.H * Lk) / 2
    rate = sp.simplify(lind[0, 1] / rho0[0, 1])                      # = -gamma
    sol = sp.dsolve(sp.Eq(r(t).diff(t), rate * r(t)), ics={r(0): rho0[0, 1]}).rhs
    sx = sp.simplify(2 * sp.re(sp.expand_complex(sol)))
    return sp.simplify(sol), sp.simplify(lind[0, 0]), sx


def measurement_probabilities_symbolic():
    """P(outcome 0) for Z after each tomography pre-rotation, in terms of the
    Bloch vector (x, y, z). Shows that Z alone sees only z (the 'intensity')."""
    x, y, z = sp.symbols("x y z", real=True)
    rho = (sp.eye(2) + x * sp.Matrix([[0, 1], [1, 0]]) + y * sp.Matrix([[0, -sp.I], [sp.I, 0]])
           + z * sp.Matrix([[1, 0], [0, -1]])) / 2
    Hd = sp.Matrix([[1, 1], [1, -1]]) / sp.sqrt(2)
    Sdg = sp.Matrix([[1, 0], [0, -sp.I]])
    rot = {"Z": sp.eye(2), "X (apply H)": Hd, "Y (apply S^dag then H)": Hd * Sdg}
    return {k: sp.simplify((U * rho * U.H)[0, 0]) for k, U in rot.items()}


def bloch_from_angles(theta, phi):
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


def dephase_bloch(r, gamma_t):
    """x, y shrink by exp(-gamma t); z (the populations) is untouched."""
    r = np.asarray(r, float)
    return np.array([r[0] * np.exp(-gamma_t), r[1] * np.exp(-gamma_t), r[2]])


def tomography(r, shots, bases="XYZ", rng=None):
    """Simulate measuring Z after each pre-rotation with `shots` shots each and
    return the estimated Bloch vector. Unmeasured axes are reported as 0 --
    the honest 'no information' guess."""
    if shots < 1:
        raise ValueError("shots must be >= 1")
    rng = np.random.default_rng(rng)
    est = np.zeros(3)
    for i, ax in enumerate("XYZ"):
        if ax in bases:
            p0 = (1 + r[i]) / 2
            est[i] = 2 * rng.binomial(shots, p0) / shots - 1
    n = np.linalg.norm(est)
    return est / n if n > 1 else est                 # project back into the Bloch ball


def state_fidelity(r_pure, r_est):
    """Fidelity <psi|rho_est|psi> = (1 + r_pure . r_est) / 2 for a pure target."""
    return float((1 + np.dot(r_pure, r_est)) / 2)


# ── classical half: phase noise vs dispersion-assisted phase retrieval ──

def smooth_phase_noise(N, n_harmonics=16, rng=None):
    """Band-limited random phase with unit standard deviation."""
    rng = np.random.default_rng(rng)
    t = np.linspace(0, 1, N)
    s = sum(rng.normal() * np.sin(2 * np.pi * k * t + rng.uniform(0, 2 * np.pi))
            for k in range(1, n_harmonics + 1))
    return s / np.std(s)


def coherence_factor(sigma):
    """|<exp(i delta)>| for Gaussian phase noise of std sigma: exp(-sigma^2/2).
    This is the classical counterpart of exp(-gamma t)."""
    c = np.exp(-np.asarray(sigma, float) ** 2 / 2)
    return float(c) if c.ndim == 0 else c


def dephased_measurements(phi_true, D1, D2, sigma, n_realizations=40, rng=None):
    """Detector-averaged intensities when every pulse carries phi_true plus an
    independent random phase of std sigma: I_k = mean |disperse(E_m, D_k)|^2."""
    rng = np.random.default_rng(rng)
    N = len(phi_true)
    I1, I2 = np.zeros(N), np.zeros(N)
    for _ in range(n_realizations):
        E = np.exp(1j * (phi_true + sigma * smooth_phase_noise(N, rng=rng)))
        I1 += np.abs(gs_core.disperse(E, D1)) ** 2
        I2 += np.abs(gs_core.disperse(E, D2)) ** 2
    return I1 / n_realizations, I2 / n_realizations


def phase_fidelity(phi_rec, phi_true):
    """|<exp(i(phi_rec - phi_true))>|: 1 = perfect up to a global phase."""
    return float(np.abs(np.mean(np.exp(1j * (np.asarray(phi_rec) - np.asarray(phi_true))))))


def fringe_contrast(I):
    """std/mean of the dispersed intensity: the interference visibility that
    carries the phase information. It falls as the phase noise grows."""
    I = np.asarray(I, float)
    return float(np.std(I) / np.mean(I))


def retrieval_vs_dephasing(sigmas, n_realizations=40, n_iter=100, seed=0):
    """Sweep phase-noise strength. Returns dict of arrays: sigma, fidelity,
    contrast, coherence (theory), plus 'chance', the fidelity between two
    independent random phases of the same smoothness (the no-information floor)."""
    d = gs_core.make_qpsk_measurements(snr_db=40, rng_seed=seed)
    phi, D1, D2 = d["phi_true"], d["D1"], d["D2"]
    rng = np.random.default_rng(seed + 1)
    fid, con = [], []
    for s in sigmas:
        I1, I2 = dephased_measurements(phi, D1, D2, s, n_realizations, rng)
        rec, _ = gs_core.retrieve_phase(I1, I2, D1, D2, n_iter=n_iter)
        fid.append(phase_fidelity(rec, phi))
        con.append(fringe_contrast(I1))
    other = gs_core.make_qpsk_measurements(snr_db=40, rng_seed=seed + 100)["phi_true"]
    return {"sigma": np.asarray(sigmas, float), "fidelity": np.array(fid),
            "contrast": np.array(con), "coherence": coherence_factor(np.asarray(sigmas)),
            "chance": phase_fidelity(other, phi)}


# ── the instrument flow diagram ──────────────────────────────────────

CLASSICAL_CHAIN = [
    ("Pulsed laser", "broadband, coherent"),
    ("Sample / signal", "writes phi(t)"),
    ("Dispersion D1, D2", "known H(f)=exp(j pi D f^2)"),
    ("Photodiode", "measures |E|^2 only"),
    ("ADC", "samples I1, I2"),
    ("GS / ML", "recovers phi(t)"),
]
QUANTUM_CHAIN = [
    ("State prep", "a|0> + b e^{i phi}|1>"),
    ("Environment", "dephasing gamma"),
    ("Known rotation", "I, H, or S^dag H"),
    ("Z measurement", "counts 0 / 1"),
    ("Shot counter", "estimates P(0)"),
    ("Tomography", "recovers Bloch vector"),
]


def flow_diagram(ax=None):
    """Draw the two chains stage-for-stage, with the analogy linking them.
    Color is never the only cue: each row is also labeled in text and uses
    a different box edge style. Returns (fig, alt_text)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 5.2))
    else:
        fig = ax.figure
    w, h, gap = 2.1, 1.1, 0.45
    rows = [(CLASSICAL_CHAIN, 3.0, OKABE_ITO["blue"], "-", "Time-stretch instrument (classical)"),
            (QUANTUM_CHAIN, 0.0, OKABE_ITO["vermillion"], "--", "Qubit tomography (quantum)")]
    for chain, y0, col, ls, title in rows:
        ax.text(-0.1, y0 + h + 0.35, title, fontsize=12, weight="bold", color=col)
        for i, (name, sub) in enumerate(chain):
            x0 = i * (w + gap)
            ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.05",
                                        fc="white", ec=col, lw=2, ls=ls))
            ax.text(x0 + w / 2, y0 + 0.72, name, ha="center", fontsize=10.5, weight="bold")
            ax.text(x0 + w / 2, y0 + 0.30, sub, ha="center", fontsize=8.3)
            if i < len(chain) - 1:
                ax.annotate("", (x0 + w + gap, y0 + h / 2), (x0 + w, y0 + h / 2),
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8))
    for i in (1, 2, 3, 5):        # analogous stages
        xc = i * (w + gap) + w / 2
        ax.annotate("", (xc, 0.0 + h + 0.03), (xc, 3.0 - 0.03),
                    arrowprops=dict(arrowstyle="<->", color=OKABE_ITO["green"], lw=1.4, ls=":"))
    ax.text(2 * (w + gap) + w / 2 + 0.08, 2.05, "same math:\nknown unitary,\nthen |.|^2",
            fontsize=8.5, color=OKABE_ITO["green"])
    ax.set_xlim(-0.3, 6 * (w + gap)); ax.set_ylim(-0.4, 4.9); ax.axis("off")
    alt = ("Two six-stage flow diagrams, one above the other. Top, classical: pulsed laser, "
           "sample writes phase, known dispersion D1 and D2, photodiode measures intensity only, "
           "ADC, Gerchberg-Saxton or ML recovers phase. Bottom, quantum: state preparation, "
           "dephasing environment, known rotation, Z measurement, shot counter, tomography "
           "recovers the Bloch vector. Dotted double arrows link sample to environment, "
           "dispersion to rotation, photodiode to Z measurement, and GS to tomography.")
    return fig, alt
