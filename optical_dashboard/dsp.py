"""
dsp.py — Optical DSP building blocks for Jalali Lab Dashboard
==============================================================
Modules:
  QPSK modem          : modulate / RRC pulse-shape / AWGN / matched-filter RX / BER
  48-ch DWDM C-band   : ITU-T G.694.1 grid, mux, FFT demux
  TDM 2:1 mux/demux   : bit-interleaving
  Digital primitives  : 2:1 MUX, D-latch, D flip-flop, 8-bit shift register
"""

import numpy as np
from scipy.special import erfc

PI = np.pi
C_LIGHT = 299792458.0  # m/s

# ════════════════════════════════════════════════════════════════════════════
#  QPSK modem
# ════════════════════════════════════════════════════════════════════════════

# Gray-coded QPSK: 2 bits → complex symbol on ±1/√2 ± j/√2 grid
_ENC = {(0,0): ( 1+1j), (0,1): (-1+1j), (1,1): (-1-1j), (1,0): ( 1-1j)}
_DEC = {v: k for k, v in _ENC.items()}

def qpsk_modulate(bits):
    """
    bits: 1-D int array of {0,1}. Length padded to even.
    Returns: complex symbol array (normalised to unit average power).
    """
    bits = np.asarray(bits).ravel().astype(int)
    if len(bits) % 2: bits = np.append(bits, 0)
    pairs   = bits.reshape(-1, 2)
    symbols = np.array([_ENC[tuple(p)] for p in pairs], dtype=complex)
    return symbols / np.sqrt(2)

def rrc_taps(beta=0.35, sps=8, n_taps=63):
    """Root-raised-cosine filter coefficients."""
    n = np.arange(n_taps) - n_taps // 2
    t = n / sps
    h = np.zeros(n_taps)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-9:
            h[i] = (1 - beta + 4*beta/PI)
        elif abs(abs(2*beta*ti) - 1.0) < 1e-9:
            h[i] = (beta/np.sqrt(2)) * ((1 + 2/PI)*np.sin(PI/(4*beta))
                                       + (1 - 2/PI)*np.cos(PI/(4*beta)))
        else:
            num = (np.sin(PI*ti*(1-beta))
                   + 4*beta*ti * np.cos(PI*ti*(1+beta)))
            den = PI * ti * (1 - (4*beta*ti)**2)
            h[i] = num / den
    return h / np.sqrt(np.dot(h, h))

def qpsk_tx(bits, sps=8, beta=0.35):
    """Full TX: bits → pulse-shaped waveform (upsample + RRC filter)."""
    syms = qpsk_modulate(bits)
    up   = np.zeros(len(syms) * sps, dtype=complex)
    up[::sps] = syms
    h = rrc_taps(beta, sps)
    return np.convolve(up, h, mode='same')

def awgn(signal, snr_db):
    """Additive white Gaussian noise channel (complex baseband)."""
    P   = np.mean(np.abs(signal)**2)
    std = np.sqrt(P / (2 * 10**(snr_db / 10)))
    n   = std * (np.random.randn(*signal.shape)
                 + 1j * np.random.randn(*signal.shape))
    return signal + n

def qpsk_rx(rx, sps=8, beta=0.35):
    """
    Matched filter → downsample → hard decision.
    Returns: (rx_symbols, decoded_bits)
    """
    h    = rrc_taps(beta, sps)
    mf   = np.convolve(rx, h, mode='same')
    syms = mf[sps // 2::sps]
    # Hard decision: nearest QPSK point
    decoded = []
    for s in syms:
        best = min(_ENC.keys(),
                   key=lambda k: abs(s - _ENC[k]/np.sqrt(2)))
        decoded.extend(best)
    return syms, np.array(decoded, dtype=int)

def qpsk_ber_theory(snr_db_arr):
    """Theoretical BER for QPSK: Q(sqrt(2*Eb/N0)) = 0.5*erfc(sqrt(Eb/N0))."""
    return 0.5 * erfc(np.sqrt(10**(np.asarray(snr_db_arr) / 10)))

def simulate_link(n_bits=2048, snr_db=10.0, sps=8, beta=0.35, rng_seed=0):
    """
    End-to-end QPSK simulation.
    Returns dict with waveform, rx_symbols, BER, constellation data.
    """
    rng  = np.random.default_rng(rng_seed)
    bits = rng.integers(0, 2, n_bits)
    tx   = qpsk_tx(bits, sps, beta)
    rx   = awgn(tx, snr_db)
    syms_rx, bits_rx = qpsk_rx(rx, sps, beta)

    # Count bit errors (align lengths)
    n_cmp = min(len(bits), len(bits_rx))
    ber   = float(np.sum(bits[:n_cmp] != bits_rx[:n_cmp])) / n_cmp

    snr_range = np.linspace(0, 20, 100)
    ber_theory = qpsk_ber_theory(snr_range)

    return {
        "tx_real":     tx.real[:512].tolist(),
        "rx_real":     rx.real[:512].tolist(),
        "syms_i":      syms_rx.real[:200].tolist(),
        "syms_q":      syms_rx.imag[:200].tolist(),
        "ber":         ber,
        "snr_db":      snr_db,
        "snr_range":   snr_range.tolist(),
        "ber_theory":  ber_theory.tolist(),
        "n_bits":      n_bits,
    }

# ════════════════════════════════════════════════════════════════════════════
#  48-channel DWDM C-band
# ════════════════════════════════════════════════════════════════════════════

def itu_grid(n_ch=48, spacing_ghz=100.0, anchor_thz=193.1):
    """
    ITU-T G.694.1 DWDM channel centre frequencies and wavelengths.
    Returns: (freqs_thz array, wavelengths_nm array) — sorted short→long λ.
    """
    half    = n_ch // 2
    offsets = (np.arange(n_ch) - half) * spacing_ghz * 1e-3  # THz
    freqs   = anchor_thz + offsets                            # THz
    lams    = C_LIGHT / (freqs * 1e12) * 1e9                 # nm
    # Sort by frequency ascending (λ descending)
    idx = np.argsort(freqs)
    return freqs[idx], lams[idx]

def wdm_sim(n_ch=48, bits_per_ch=128, sps=4, snr_db=25.0, rng_seed=1):
    """
    Simulate 48-ch DWDM: each channel carries independent QPSK data.
    Returns spectral data for visualisation.
    """
    rng = np.random.default_rng(rng_seed)
    freqs_thz, lams_nm = itu_grid(n_ch)

    N_samp  = bits_per_ch // 2 * sps      # samples per channel
    fs_THz  = 0.1 * n_ch                  # sampling rate (THz), wide enough

    # Build baseband signal per channel
    all_bits   = [rng.integers(0, 2, bits_per_ch) for _ in range(n_ch)]
    ch_signals = [qpsk_tx(b, sps=sps, beta=0.35)[:N_samp] for b in all_bits]

    t = np.arange(N_samp) / (fs_THz * 1e12)  # seconds

    # Modulate onto channel carriers (use offset from band centre)
    f0     = float(np.mean(freqs_thz)) * 1e12   # Hz
    muxed  = np.zeros(N_samp, dtype=complex)
    for sig, fch in zip(ch_signals, freqs_thz):
        df = (fch * 1e12 - f0)
        muxed += sig * np.exp(2j * PI * df * t)

    # Add global noise
    muxed = awgn(muxed, snr_db)

    # Power spectrum
    freqs_fft = np.fft.fftfreq(N_samp, d=1.0 / (fs_THz * 1e12)) * 1e-12 + float(np.mean(freqs_thz))
    psd       = np.abs(np.fft.fft(muxed))**2 / N_samp

    # Demux: extract per-channel power by windowed bandpass
    ch_powers = []
    for fch in freqs_thz:
        df     = (freqs_fft - fch)
        H      = np.exp(-df**2 / (2 * 0.05**2))   # 50 GHz Gaussian BPF
        P_ch   = float(np.mean(np.abs(np.fft.ifft(np.fft.fft(muxed) * H))**2))
        ch_powers.append(P_ch)

    # Normalise powers to dBm-like relative scale
    P_ref = max(ch_powers)
    ch_powers_db = [10 * np.log10(p / P_ref + 1e-12) for p in ch_powers]

    # Prepare sorted frequency axis for plot (show ±band)
    sort_idx   = np.argsort(freqs_fft)
    freqs_plot = freqs_fft[sort_idx].tolist()
    psd_plot   = (10 * np.log10(psd[sort_idx] / (psd[sort_idx].max() + 1e-30))).tolist()

    return {
        "freqs_thz":      freqs_thz.tolist(),
        "lams_nm":        lams_nm.tolist(),
        "ch_powers_db":   ch_powers_db,
        "freqs_plot_thz": freqs_plot,
        "psd_plot_db":    psd_plot,
        "n_ch":           n_ch,
        "bits_per_ch":    bits_per_ch,
        "snr_db":         snr_db,
    }

# ════════════════════════════════════════════════════════════════════════════
#  TDM 2:1 mux / demux
# ════════════════════════════════════════════════════════════════════════════

def tdm_mux(a, b):
    """Interleave streams a, b: [a0,b0,a1,b1,...]"""
    n   = min(len(a), len(b))
    out = np.empty(2 * n, dtype=np.result_type(a, b))
    out[0::2] = np.asarray(a)[:n]
    out[1::2] = np.asarray(b)[:n]
    return out

def tdm_demux(muxed):
    """Undo interleave → (a, b)."""
    return muxed[0::2], muxed[1::2]

def mux_2to1(a, b, sel):
    """Combinational 2:1 MUX. sel=0→a, sel=1→b. Arrays or scalars."""
    return np.where(np.asarray(sel, dtype=bool),
                    np.asarray(b), np.asarray(a))

# ════════════════════════════════════════════════════════════════════════════
#  Digital logic primitives
# ════════════════════════════════════════════════════════════════════════════

class DLatch:
    """Level-sensitive D latch (data transparent when EN=1)."""
    def __init__(self, width=8):
        self.Q = np.zeros(width, dtype=np.uint8)

    def update(self, D, EN):
        if EN:
            self.Q = np.asarray(D, dtype=np.uint8).copy()
        return self.Q.copy()

    def state_str(self):
        return "".join(str(b) for b in self.Q)


class DFlipFlop:
    """Positive-edge-triggered D flip-flop."""
    def __init__(self, width=8):
        self.Q       = np.zeros(width, dtype=np.uint8)
        self._clk_p  = 0

    def tick(self, D, CLK):
        """Call each simulation step. Captures D on rising CLK edge."""
        if CLK and not self._clk_p:
            self.Q = np.asarray(D, dtype=np.uint8).copy()
        self._clk_p = int(CLK)
        return self.Q.copy()


class ShiftRegister:
    """Serial-in / parallel-out SIPO shift register."""
    def __init__(self, n_bits=8):
        self.reg = np.zeros(n_bits, dtype=np.uint8)
        self.n   = n_bits

    def shift(self, bit_in):
        """Shift in one bit MSB-first; returns current register state."""
        self.reg  = np.roll(self.reg, 1)
        self.reg[0] = int(bool(bit_in))
        return self.reg.copy()

    def parallel_load(self, data):
        self.reg = np.asarray(data[:self.n], dtype=np.uint8).copy()

    def state_str(self):
        return "".join(str(b) for b in self.reg)


def digital_demo(data_bits=0b10110010, n_cycles=16):
    """
    Simulate one cycle of digital logic:
    D latch, D flip-flop, shift register, 2:1 mux over n_cycles.
    Returns per-cycle log for visualisation.
    """
    latch  = DLatch(8)
    ff     = DFlipFlop(8)
    sr     = ShiftRegister(8)

    data_arr = np.array([(data_bits >> (7-i)) & 1 for i in range(8)], dtype=np.uint8)
    log = []
    for cyc in range(n_cycles):
        EN  = int(cyc < n_cycles // 2)      # latch transparent for first half
        CLK = cyc % 2                        # alternating clock
        bit_in = (data_bits >> (7 - (cyc % 8))) & 1

        latch_q = latch.update(data_arr, EN)
        ff_q    = ff.tick(data_arr, CLK)
        sr_q    = sr.shift(bit_in)
        mux_out = mux_2to1(latch_q, ff_q, np.array([cyc % 2]*8))

        log.append({
            "cycle":   cyc,
            "EN":      EN,
            "CLK":     CLK,
            "bit_in":  bit_in,
            "latch":   latch.state_str(),
            "ff":      "".join(str(b) for b in ff_q),
            "sr":      sr.state_str(),
            "mux":     "".join(str(b) for b in mux_out),
        })
    return log


# ════════════════════════════════════════════════════════════════════════════
#  Optical 3-D Voxel Hash  ·  Energy Minimisation  ·  LSH
# ════════════════════════════════════════════════════════════════════════════
#
#  Physical motivation
#  -------------------
#  An optical field E(x, y, λ) is a 3-D complex-valued function over the
#  transverse plane and wavelength.  Storing it in a dense 3-D array is
#  wasteful when the field is sparse (e.g. a Gaussian beam, a sparse
#  scatterer ensemble).  A polynomial spatial hash gives O(1) insert/query
#  while naturally partitioning the (x, y, λ) space into voxels.
#
#  Energy functional (physics ↔ optimisation bridge)
#  --------------------------------------------------
#  H = α · H_TV  +  β · H_pc
#
#  H_TV = Σ_{〈i,j〉} |E_i − E_j|²   (total variation — penalises abrupt field changes)
#  H_pc = Σ_k  (|E_k| − 1)²          (TD-GS pure-phase constraint |T|=1)
#
#  Gradient descent on H drives the stored field toward a smooth, unit-
#  amplitude wavefront — the TD-GS solution.
#
#  LSH (Locality-Sensitive Hashing)
#  ---------------------------------
#  Random projection: h_t(v) = sign(A_t · v + b_t) ∈ {0,1}^n_bits
#  Collision probability ≈ 1 − arccos(sim)/π  for cosine similarity.
#  Multiple tables boost recall; query unions candidate sets.
# ════════════════════════════════════════════════════════════════════════════

_HASH_PRIMES = (73856093, 19349669, 83492791)


class OpticalHash3D:
    """
    Sparse 3-D voxel hash for complex optical field data.

    Coordinate system
    -----------------
    (x [μm], y [μm], λ [nm])  — discretised to integer voxel indices.

    Hash function
    -------------
    h(ix, iy, iλ) = (ix·p1 ⊕ iy·p2 ⊕ iλ·p3) mod N   (polynomial XOR hash)

    Supports
    --------
    insert / query / bulk load
    Total-variation energy  H_TV = Σ |E_i − E_j|²
    Phase-constraint energy H_pc = Σ (|E_k| − 1)²
    Gradient-descent energy minimisation (returns convergence history)
    Hash-bucket distribution histogram
    """

    _NEIGH6 = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

    def __init__(self, dx_um=1.0, dy_um=1.0, dlam_nm=0.1):
        self.dx  = float(dx_um)
        self.dy  = float(dy_um)
        self.dl  = float(dlam_nm)
        self._store: dict = {}          # (ix,iy,iλ) → complex

    # ── voxel key ────────────────────────────────────────────────────────────
    def _key(self, x, y, lam):
        return (int(round(x / self.dx)),
                int(round(y / self.dy)),
                int(round(lam / self.dl)))

    # ── CRUD ─────────────────────────────────────────────────────────────────
    def insert(self, x, y, lam, E):
        self._store[self._key(x, y, lam)] = complex(E)

    def query(self, x, y, lam):
        return self._store.get(self._key(x, y, lam), 0.0 + 0j)

    def bulk_insert(self, x_arr, y_arr, lam_arr, E_arr):
        for x, y, l, E in zip(x_arr, y_arr, lam_arr, E_arr):
            self.insert(x, y, l, E)

    def __len__(self):
        return len(self._store)

    # ── array views ──────────────────────────────────────────────────────────
    def keys_array(self):
        """(N,3) int array of (ix,iy,iλ) voxel indices."""
        return np.array(list(self._store.keys()), dtype=int) if self._store else np.empty((0,3), int)

    def values_array(self):
        """(N,) complex array of field amplitudes."""
        return np.array(list(self._store.values()), dtype=complex)

    def coords_and_field(self):
        """Returns (x_um, y_um, lam_nm, |E|, phase_rad) arrays."""
        if not self._store:
            return [np.array([])]*5
        k = self.keys_array()
        v = self.values_array()
        return (k[:,0]*self.dx, k[:,1]*self.dy, k[:,2]*self.dl,
                np.abs(v), np.angle(v))

    # ── energy functionals ───────────────────────────────────────────────────
    def tv_energy(self):
        """H_TV = 0.5·Σ_{i,j adjacent} |E_i − E_j|²  (each pair counted once)."""
        H = 0.0
        for (ix,iy,il), E in self._store.items():
            for dxi,dyi,dli in self._NEIGH6:
                nb = self._store.get((ix+dxi, iy+dyi, il+dli))
                if nb is not None:
                    H += abs(E - nb)**2
        return float(H * 0.5)

    def phase_energy(self):
        """H_pc = Σ_k (|E_k| − 1)²  — TD-GS unit-amplitude constraint."""
        v = self.values_array()
        return float(np.sum((np.abs(v) - 1.0)**2)) if len(v) else 0.0

    def total_energy(self, alpha=0.1, beta=1.0):
        return alpha * self.tv_energy() + beta * self.phase_energy()

    # ── gradient descent ─────────────────────────────────────────────────────
    def minimise(self, n_iter=80, lr=0.04, alpha=0.1, beta=0.5):
        """
        Gradient descent on H = alpha·H_TV + beta·H_pc.

        ∂H/∂E_k* = alpha · Σ_{j∈N(k)} (E_k − E_j)
                  + beta  · (|E_k| − 1) · E_k / |E_k|

        Returns list of (iter, H_TV, H_pc, H_total).
        """
        history = []
        keys = list(self._store.keys())
        for it in range(n_iter):
            grads = {}
            for k in keys:
                Ek  = self._store[k]
                # TV gradient (Wirtinger ∂/∂E*)
                g_tv = sum(
                    (Ek - self._store[(k[0]+dx,k[1]+dy,k[2]+dl)])
                    for dx,dy,dl in self._NEIGH6
                    if (k[0]+dx,k[1]+dy,k[2]+dl) in self._store
                )
                # Phase-constraint gradient
                amp = abs(Ek) + 1e-12
                g_pc = (amp - 1.0) * Ek / amp
                grads[k] = alpha * g_tv + beta * g_pc
            for k in keys:
                self._store[k] -= lr * grads[k]
            htv = self.tv_energy()
            hpc = self.phase_energy()
            history.append((it, htv, hpc, alpha*htv + beta*hpc))
        return history

    # ── polynomial hash distribution ─────────────────────────────────────────
    @staticmethod
    def poly_hash(ix, iy, il, N=256):
        p1, p2, p3 = _HASH_PRIMES
        return int((ix * p1 ^ iy * p2 ^ il * p3) % N)

    def bucket_histogram(self, N_buckets=64):
        """Histogram of items per hash bucket — measures load balance."""
        counts = np.zeros(N_buckets, dtype=int)
        for k in self._store:
            counts[self.poly_hash(*k, N=N_buckets)] += 1
        return counts


# ─────────────────────────────────────────────────────────────────────────────

class OpticalLSH:
    """
    Locality-Sensitive Hashing for optical field samples.

    Feature vector per sample:  v = [|E|, Re(E), Im(E), x_um, y_um, λ_nm]
    Hash family:  h_t(v) = (sign(A_t @ v + b_t) ≥ 0).astype(int)   ∈ {0,1}^n_bits
    Collision probability ∝ angular similarity between feature vectors.

    n_tables independent hash tables improve recall; query unions candidates.
    """

    def __init__(self, n_bits=12, n_tables=4, seed=0):
        rng = np.random.default_rng(seed)
        d = 6
        self.A  = [rng.standard_normal((n_bits, d)) for _ in range(n_tables)]
        self.b  = [rng.uniform(0, 2*PI, n_bits)     for _ in range(n_tables)]
        self.tables  = [{} for _ in range(n_tables)]
        self._data   = []    # list of ((x,y,λ), feat_vec)
        self.n_bits  = n_bits
        self.n_tables = n_tables

    def _feat(self, x, y, lam, E):
        return np.array([abs(E), E.real, E.imag, x, y, lam], dtype=float)

    def _bucket(self, feat, t):
        return tuple((self.A[t] @ feat + self.b[t] >= 0).astype(int))

    def insert(self, x, y, lam, E):
        feat = self._feat(x, y, lam, E)
        idx  = len(self._data)
        self._data.append(((x, y, lam), feat))
        for t in range(self.n_tables):
            h = self._bucket(feat, t)
            self.tables[t].setdefault(h, []).append(idx)

    def query(self, x, y, lam, E, k=5):
        """Return k approximate nearest neighbours as ((x,y,λ), feat) pairs."""
        feat  = self._feat(x, y, lam, E)
        cands = set()
        for t in range(self.n_tables):
            cands.update(self.tables[t].get(self._bucket(feat, t), []))
        if not cands:
            return []
        ranked = sorted(cands, key=lambda i: float(np.linalg.norm(self._data[i][1] - feat)))
        return [(self._data[i][0], self._data[i][1]) for i in ranked[:k]]

    def collision_rate(self):
        """Fraction of non-empty buckets with > 1 item (collision density)."""
        n_over = n_total = 0
        for t in self.tables:
            for v in t.values():
                n_total += 1
                if len(v) > 1:
                    n_over  += 1
        return n_over / max(n_total, 1)


# ─────────────────────────────────────────────────────────────────────────────

def optical_hash_demo(n_points=256, rng_seed=3):
    """
    End-to-end 3-D optical hash demonstration.

    1. Generate a simulated aberrated Gaussian wavefront E(x, y, λ).
    2. Insert into OpticalHash3D (sparse voxel storage).
    3. Run energy minimisation (TV + phase constraint → unit-amplitude wavefront).
    4. Build OpticalLSH and measure collision rate.
    5. Return arrays for visualisation.
    """
    rng = np.random.default_rng(rng_seed)

    # ── simulate sparse optical field ────────────────────────────────────────
    x_um   = rng.uniform(-10, 10, n_points)
    y_um   = rng.uniform(-10, 10, n_points)
    lam_nm = rng.uniform(1540, 1560, n_points)

    w0  = 5.0   # beam waist [μm]
    amp = np.exp(-(x_um**2 + y_um**2) / w0**2)
    # wavefront: Zernike-like aberration + chromatic tilt
    phi = (rng.normal(0, 0.8, n_points)
           + 0.03 * (lam_nm - 1550.0)
           + 0.002 * (x_um**2 + y_um**2))
    E_true = amp * np.exp(1j * phi)
    noise  = 0.12 * (rng.standard_normal(n_points) + 1j * rng.standard_normal(n_points))
    E_meas = E_true + noise

    # ── populate hash ─────────────────────────────────────────────────────────
    oh = OpticalHash3D(dx_um=0.5, dy_um=0.5, dlam_nm=0.5)
    oh.bulk_insert(x_um, y_um, lam_nm, E_meas)

    H_init = oh.total_energy()

    # ── energy minimisation ──────────────────────────────────────────────────
    history = oh.minimise(n_iter=80, lr=0.035, alpha=0.1, beta=0.4)

    H_final = oh.total_energy()

    # ── LSH ───────────────────────────────────────────────────────────────────
    lsh = OpticalLSH(n_bits=10, n_tables=4, seed=1)
    for xi, yi, li, Ei in zip(x_um, y_um, lam_nm, E_meas):
        lsh.insert(xi, yi, li, Ei)
    coll = lsh.collision_rate()

    # ── query one sample ──────────────────────────────────────────────────────
    q_nbrs = lsh.query(x_um[0], y_um[0], lam_nm[0], E_meas[0], k=5)
    nbr_dists = [float(np.linalg.norm(nb[1][:3] - np.array([x_um[0], y_um[0], lam_nm[0]])))
                 for nb in q_nbrs]

    # ── hash bucket histogram ─────────────────────────────────────────────────
    hist = oh.bucket_histogram(N_buckets=32)

    # ── output arrays ─────────────────────────────────────────────────────────
    x_o, y_o, l_o, amp_o, phi_o = oh.coords_and_field()
    iters  = [h[0] for h in history]
    H_tv   = [h[1] for h in history]
    H_pc   = [h[2] for h in history]
    H_tot  = [h[3] for h in history]

    return {
        # 3-D scatter data
        "x_um":      x_o.tolist(),
        "y_um":      y_o.tolist(),
        "lam_nm":    l_o.tolist(),
        "amp":       amp_o.tolist(),
        "phi_rad":   phi_o.tolist(),
        # energy convergence
        "iters":     iters,
        "H_tv":      H_tv,
        "H_pc":      H_pc,
        "H_total":   H_tot,
        "H_init":    float(H_init),
        "H_final":   float(H_final),
        "reduction": float(1.0 - H_final / (H_init + 1e-30)),
        # hash stats
        "bucket_counts":   hist.tolist(),
        "collision_rate":  float(coll),
        "n_stored":        len(oh),
        "nbr_dists":       nbr_dists,
    }
