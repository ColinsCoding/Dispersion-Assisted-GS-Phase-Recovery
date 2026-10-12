"""Identifying hydrogen emission lines, with the reduced-mass correction, a
SciNet-style question/answer model, and a slowed-down emission animation.

  1/lambda = R_H (1/m^2 - 1/n^2),   n > m >= 1
  R_inf = m_e e^4 / (8 eps0^2 h^3 c) = 10,973,731.6 1/m  (infinitely heavy nucleus)
  R_H   = R_inf / (1 + m_e/m_p)      = 10,967,758   1/m  (the electron and proton
                                                         orbit their common center of mass)

Example (Knight Ch. 37/38 style): lines at 121.6 nm and 1282 nm are
  n = 2 -> 1 (Lyman alpha, ultraviolet) and n = 5 -> 3 (Paschen beta, infrared).

SciNet-style Q&A (cf. the "discovering physics" figure): an ENCODER compresses a
list of observed lines into a latent representation -- one fitted constant R
plus an integer pair (n, m) per line; a DECODER answers a QUESTION (n, m) with a
predicted wavelength. Here the encoder is an exact search + least squares, not
a neural network, so every step can be checked.

The animation is a schematic (Bohr-model energy levels, which give the right
energies but not the right picture of the electron) with the two emitted
waves drawn at their TRUE relative wavelengths and light slowed by ~1e17 so it
can be watched. Atom and wavelengths cannot share one scale (0.05 nm vs
121.6 nm); the figure says so. Free software only (matplotlib + Pillow).
"""
import numpy as np
import sympy as sp

H = 6.62607015e-34
C = 299792458.0
E_CH = 1.602176634e-19
M_E = 9.1093837015e-31
M_P = 1.67262192369e-27
EPS0 = 8.8541878128e-12
R_INF = 10973731.568160          # 1/m, CODATA 2018

SERIES = {1: "Lyman", 2: "Balmer", 3: "Paschen", 4: "Brackett", 5: "Pfund", 6: "Humphreys"}
GREEK = ["alpha", "beta", "gamma", "delta", "epsilon"]


# ── Rydberg formula ──────────────────────────────────────────────────

def rydberg_symbolic():
    """R_inf from Bohr theory and the reduced-mass version."""
    me, e, eps0, h, c, mp = sp.symbols("m_e e epsilon_0 h c m_p", positive=True)
    Rinf = me * e**4 / (8 * eps0**2 * h**3 * c)
    mu = me * mp / (me + mp)
    RH = sp.simplify(Rinf.subs(me, mu))
    return Rinf, RH, sp.simplify(RH / Rinf - 1 / (1 + me / mp))


def rydberg_constant(reduced_mass=True):
    return R_INF / (1 + M_E / M_P) if reduced_mass else R_INF


def wavelength_nm(n, m, reduced_mass=True):
    """Vacuum wavelength of the n -> m transition (n > m >= 1)."""
    if not (isinstance(n, (int, np.integer)) and isinstance(m, (int, np.integer))) or n <= m or m < 1:
        raise ValueError("need integers n > m >= 1")
    return 1e9 / (rydberg_constant(reduced_mass) * (1 / m**2 - 1 / n**2))


def photon_energy_eV(lambda_nm):
    if lambda_nm <= 0:
        raise ValueError("wavelength must be positive")
    return H * C / (lambda_nm * 1e-9) / E_CH


def line_name(n, m):
    series = SERIES.get(m, f"m={m}")
    k = n - m - 1
    return f"{series} {GREEK[k]}" if k < len(GREEK) else f"{series} ({n}->{m})"


def spectral_region(lambda_nm):
    if lambda_nm < 10:
        return "X-ray"
    if lambda_nm < 400:
        return "ultraviolet"
    if lambda_nm <= 700:
        return "visible"
    if lambda_nm < 1e6:
        return "infrared"
    return "microwave/radio"


def identify_line(lambda_nm, n_max=12, reduced_mass=True):
    """All (n, m) within n_max, ranked by |predicted - observed|. Returns the best
    match and the runner-up gap (how unambiguous the assignment is)."""
    cands = sorted(((abs(wavelength_nm(n, m, reduced_mass) - lambda_nm), n, m)
                    for m in range(1, n_max) for n in range(m + 1, n_max + 1)))
    (err, n, m), (err2, n2, m2) = cands[0], cands[1]
    lam = wavelength_nm(n, m, reduced_mass)
    return {"n": n, "m": m, "name": line_name(n, m), "predicted_nm": lam, "error_nm": err,
            "runner_up": (n2, m2, err2), "energy_eV": photon_energy_eV(lam), "region": spectral_region(lam)}


def solve_question_symbolic(lambda1=sp.Rational(1216, 10), lambda2=1282):
    """For each wavelength, test integer pairs exactly: 1/lambda = R (1/m^2 - 1/n^2)
    -> 1/m^2 - 1/n^2 = 1/(R lambda). Report the pair whose fraction is closest."""
    out = []
    for lam in (lambda1, lambda2):
        target = 1 / (sp.Float(rydberg_constant()) * sp.Float(lam) * sp.Float(1e-9))
        best = min(((abs(sp.Rational(1, mm**2) - sp.Rational(1, nn**2) - target), nn, mm)
                    for mm in range(1, 8) for nn in range(mm + 1, 10)), key=lambda e: e[0])
        out.append((lam, best[1], best[2], sp.Rational(1, best[2]**2) - sp.Rational(1, best[1]**2), float(target)))
    return out


# ── SciNet-style encoder / decoder ───────────────────────────────────

def encode(observed_nm, n_max=10):
    """Latent representation of a set of lines: integer pairs from a coarse
    Rydberg guess, then the constant R refit by least squares over all lines."""
    pairs = [(identify_line(lam, n_max, reduced_mass=False)["n"],
              identify_line(lam, n_max, reduced_mass=False)["m"]) for lam in observed_nm]
    x = np.array([1 / m**2 - 1 / n**2 for n, m in pairs])
    y = 1 / (np.asarray(observed_nm, float) * 1e-9)
    R_fit = float(x @ y / (x @ x))
    return {"R": R_fit, "pairs": pairs}


def decode(latent, question):
    """Answer a question (n, m) from the latent representation."""
    n, m = question
    return 1e9 / (latent["R"] * (1 / m**2 - 1 / n**2))


def reduced_mass_from_R(R_fit):
    """Electron-to-proton mass ratio implied by a fitted R: R_inf / R - 1 = m_e / m_p."""
    return R_INF / R_fit - 1


# ── animation ────────────────────────────────────────────────────────

def emission_animation(path, lines=((2, 1), (5, 3)), frames=120, fps=24, okabe=("#0072B2", "#D55E00")):
    """Two-panel GIF: left, Bohr energy-level ladder with the electron dropping and
    a photon arrow; right, the two emitted wave packets traveling outward at the
    same (slowed) speed with wavelengths in their TRUE ratio. Returns (path, alt_text)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    lams = [wavelength_nm(n, m) for n, m in lines]
    E = lambda n: -13.6057 / (1 + M_E / M_P) / n**2
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 2.2]})
    for n in range(1, 7):
        axL.hlines(E(n), 0, 1, color="0.6", lw=1)
        axL.text(1.03 if (n < 4 or n % 2 == 0) else 1.16, E(n), f"n={n}", va="center", fontsize=8)
    axL.set_xlim(0, 1.3); axL.set_ylim(-14.5, 0.8); axL.set_xticks([])
    axL.set_ylabel("energy (eV)"); axL.set_title("hydrogen energy levels (Bohr model)", fontsize=10)
    dots = [axL.plot([], [], "o", color=c, ms=9)[0] for c in okabe]
    arrows = [axL.annotate("", (0, 0), (0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=2)) for c in okabe]

    L = 6.0 * max(lams)                                   # nm of travel shown
    xs = np.linspace(0, L, 3000)
    waves = [axR.plot([], [], color=c, lw=1.6, ls=ls, label=f"{line_name(n, m)}: {lam:.1f} nm ({spectral_region(lam)})")[0]
             for (n, m), lam, c, ls in zip(lines, lams, okabe, ("-", "--"))]
    axR.set_xlim(0, L); axR.set_ylim(-3.2, 3.2); axR.set_yticks([])
    axR.set_xlabel("distance from the atom (nm)  -- wavelengths to scale; the atom itself (~0.1 nm) is not")
    axR.set_title("light slowed ~1e17x: both photons travel at the same speed c", fontsize=10)
    axR.legend(loc="upper right", fontsize=8)
    t_emit = [0.10, 0.35]

    def wave_at(frac, lam, off):
        front = max(0.0, frac) * L
        env = np.exp(-((xs - (front - 2.5 * lam)) / (1.6 * lam)) ** 2) * (xs <= front)
        return np.cos(2 * np.pi * (xs - front) / lam) * env + off

    def update(i):
        f = i / (frames - 1)
        for k, ((n, m), lam) in enumerate(zip(lines, lams)):
            te = t_emit[k]
            x = 0.25 + 0.4 * k
            if f < te:
                dots[k].set_data([x], [E(n)])
                arrows[k].set_position((x, E(n))); arrows[k].xy = (x, E(n))
            else:
                dots[k].set_data([x], [E(m)])
                arrows[k].set_position((x, E(n))); arrows[k].xy = (x, E(m))
            waves[k].set_data(xs, wave_at((f - te) / (1 - te) * 1.15, lam, 1.6 - 3.2 * k))
        return dots + waves

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    alt = (f"Animation. Left: hydrogen energy levels n=1 to 6; an electron drops from n={lines[0][0]} to n={lines[0][1]} "
           f"and another from n={lines[1][0]} to n={lines[1][1]}, each marked by an arrow. Right: two wave packets leave the "
           f"atom at the same speed; the {lams[0]:.1f} nm wave (solid) has crests about {lams[1]/lams[0]:.1f} times closer "
           f"together than the {lams[1]:.1f} nm wave (dashed).")
    return path, alt
