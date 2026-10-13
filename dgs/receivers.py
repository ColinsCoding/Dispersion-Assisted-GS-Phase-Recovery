"""
Three optical receivers on one simulated fiber link, in physical units.

  transmitter : 32 GBd QPSK with smooth (constant-envelope) phase, 8-symbol frames with 2 pilots
  fiber       : standard single-mode fiber, split-step Fourier solution of the nonlinear
                Schrödinger equation (dispersion beta2, loss alpha, Kerr nonlinearity gamma)
  noise       : amplifier noise (ASE) set by OSNR in 0.1 nm (12.5 GHz), plus receiver thermal noise
  receivers   : (a) coherent: local-oscillator laser + 90-degree hybrid (measures the field directly)
                (b) Kramers-Kronig: a carrier tone sent with the signal, one photodiode, KK phase
                (c) dispersion-assisted GS: the light split into five dispersive branches, five
                    photodiodes, multi-plane Gerchberg-Saxton
Every receiver ends the same way: undo the fiber dispersion digitally, align with the pilots, decide.
"""
from __future__ import annotations
import numpy as np

C0 = 299792458.0
RS = 32e9                     # symbol rate, baud
SPS = 32                      # samples per symbol
FS = RS*SPS                   # simulation sample rate
N_SYM = 8                     # symbols per frame (2 pilots + 6 data)
N = N_SYM*SPS
PILOT = [0, 1]
GRAY = {0b00: 0, 0b01: 1, 0b11: 2, 0b10: 3}; INV = {v: k for k, v in GRAY.items()}
BETA2 = -21.7e-27             # s^2/m   (standard single-mode fiber at 1550 nm)
GAMMA = 1.3e-3                # 1/(W m)
ALPHA_DB_KM = 0.2
OMEGA = 2*np.pi*np.fft.fftfreq(N, 1/FS)


def disperse(E, b2L):
    """Apply accumulated dispersion beta2*L (s^2): A(w) -> A(w) exp(i b2L w^2 / 2)."""
    return np.fft.ifft(np.fft.fft(E)*np.exp(0.5j*b2L*OMEGA**2))


def smooth_phase(sym):
    """QPSK phases joined by cosine-shaped transitions, wrapping smoothly from the last symbol back
    to the first, so the frame is periodic and its spectrum stays compact (no edge jump)."""
    ph = np.pi/4 + np.array(sym)*np.pi/2
    n = len(sym)
    step = np.angle(np.exp(1j*(np.roll(ph, -1) - ph)))       # shortest way to the next symbol (cyclic)
    t = np.arange(n*SPS)/SPS; k = np.floor(t).astype(int); fr = t - k
    return ph[k] + step[k]*0.5*(1 - np.cos(np.pi*fr))


def transmit(data_sym, P_W):
    return np.sqrt(P_W)*np.exp(1j*smooth_phase(PILOT + list(data_sym)))


def fiber(E, L_m, steps=20, nonlinear=True):
    """Symmetric split-step Fourier: half dispersion, full nonlinearity + loss, half dispersion."""
    dz = L_m/steps
    a = ALPHA_DB_KM/1000*np.log(10)/10            # 1/m (power)
    for _ in range(steps):
        E = disperse(E, BETA2*dz/2)
        if nonlinear:
            E = E*np.exp(1j*GAMMA*np.abs(E)**2*dz)
        E = E*np.exp(-a*dz/2)
        E = disperse(E, BETA2*dz/2)
    return E


def ase(E, osnr_db, rng):
    """Add complex white noise so that signal power / noise power in 12.5 GHz = OSNR."""
    Ps = np.mean(np.abs(E)**2)
    N0 = Ps/(10**(osnr_db/10)*12.5e9)            # W/Hz
    sigma = np.sqrt(N0*FS/2)
    return E + sigma*(rng.standard_normal(E.size) + 1j*rng.standard_normal(E.size))


def lowpass(x, B):
    X = np.fft.fft(x); X[np.abs(np.fft.fftfreq(x.size, 1/FS)) > B] = 0
    return np.fft.ifft(X)


def decide(E_tx_plane):
    """Pilot alignment (offset; sign is known for field receivers) and QPSK decisions."""
    z = E_tx_plane[np.arange(N_SYM)*SPS]
    ref = np.exp(1j*(np.pi/4 + np.array(PILOT)*np.pi/2))
    out = []
    for zz in (z, np.conj(z)):
        off = np.angle(np.sum(ref*np.conj(zz[:2])))
        out.append((np.round((np.angle(zz*np.exp(1j*off)) - np.pi/4)/(np.pi/2)) % 4).astype(int))
    return out


def best_decision(cands):
    for d in cands:
        if list(d[:2]) == PILOT:
            return d
    return cands[0]


# ---------------- (a) coherent receiver ----------------
def rx_coherent(E_rx, b2L_fiber, B_e):
    E = lowpass(E_rx, B_e)
    return best_decision(decide(disperse(E, -b2L_fiber)))


# ---------------- (b) Kramers-Kronig receiver ----------------
def kk_tx(data_sym, P_W, cspr_db, f_shift):
    """Signal shifted by f_shift (single sideband) plus a carrier tone: the carrier carries
    10^(CSPR/10) times the signal power. Total power P_W."""
    s = np.exp(1j*smooth_phase(PILOT + list(data_sym)))*np.exp(2j*np.pi*f_shift*np.arange(N)/FS)
    r = 10**(cspr_db/10)
    Ps = P_W/(1 + r)
    return np.sqrt(r*Ps) + np.sqrt(Ps)*s, np.sqrt(r*Ps), np.sqrt(Ps)


def rx_kk(E_rx, b2L_fiber, carrier_amp_rx, s_amp_rx, f_shift, B_e, sigma_th=0.0, rng=None):
    I = np.abs(E_rx)**2
    if sigma_th and rng is not None:
        I = I + sigma_th*rng.standard_normal(N)                    # photodiode thermal noise
    I = np.real(lowpass(I, 2*B_e + f_shift))                       # one photodiode
    I = np.maximum(I, 1e-12*np.mean(I))
    lnA = 0.5*np.log(I)
    phi = np.real(np.fft.ifft(-1j*np.sign(np.fft.fftfreq(N))*np.fft.fft(lnA)))   # KK: phase = Hilbert{ln|E|} (multiplier -i sign(f))
    E = np.sqrt(I)*np.exp(1j*phi)
    s = disperse(E - carrier_amp_rx, -b2L_fiber)              # undo dispersion while the sideband is still shifted
    s = s*np.exp(-2j*np.pi*f_shift*np.arange(N)/FS)           # then bring it back to baseband
    return best_decision(decide(lowpass(s, B_e)))


# ---------------- (c) dispersion-assisted GS receiver ----------------
def gs_branches(E_rx, taps_b2L, sigma_th, rng, B_e):
    """Split into len(taps) branches (each gets 1/len of the power), disperse, detect |E|^2,
    add thermal noise, low-pass."""
    n = len(taps_b2L)
    out = []
    for d in taps_b2L:
        I = np.abs(disperse(E_rx/np.sqrt(n), d))**2
        I = I + sigma_th*rng.standard_normal(N)
        out.append(np.maximum(np.real(lowpass(I, 2*B_e)), 0))
    return out


def rx_gs(Is, taps_b2L, b2L_fiber, rng, n_iter=100, max_tries=20, fit_tol=0.15):
    """Multi-plane GS. Unknown: the transmitted field (constant envelope). Plane i is the TX field
    after fiber + tap i dispersion. After each run, check how well the answer reproduces all the
    measurements (relative misfit); if it is worse than fit_tol, restart from a random phase.
    Keep the best-fitting answer. Returns (decisions, tries, misfit)."""
    n = len(taps_b2L)
    amps = [np.sqrt(I*n) for I in Is]                              # undo the 1/n power split
    planes = [b2L_fiber + d for d in taps_b2L]
    scale = np.sqrt(sum(np.sum(a**2) for a in amps))
    best = (np.inf, None)
    for attempt in range(max_tries):
        E = disperse(amps[0].astype(complex), -planes[0]) if attempt == 0 else np.exp(1j*rng.uniform(-np.pi, np.pi, N))
        E = np.exp(1j*np.angle(E))
        for _ in range(n_iter):
            for a, D in zip(amps, planes):
                E = np.exp(1j*np.angle(disperse(a*np.exp(1j*np.angle(disperse(E, D))), -D)))
        # misfit with the best overall amplitude (the TX power is unknown to the receiver)
        mags = [np.abs(disperse(E, D)) for D in planes]
        k = sum(np.sum(m*a) for m, a in zip(mags, amps))/sum(np.sum(m*m) for m in mags)
        mis = np.sqrt(sum(np.sum((k*m - a)**2) for m, a in zip(mags, amps)))/scale
        if mis < best[0]:
            best = (mis, E)
        if mis < fit_tol:
            break
    return best_decision(decide(best[1])), attempt + 1, best[0]
