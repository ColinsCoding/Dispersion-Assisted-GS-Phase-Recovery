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

# ── MATLAB .mat file loader (Jalali Lab data format) ──────────────────────────
def load_mat(path):
    """
    Load optical intensity data from a MATLAB .mat file.

    Searches for these variable names in priority order:
      I1, I2, data, signal, intensity, y, x

    Returns (t, I1, I2_or_None).
    I2 is returned if both I1 and I2 exist in the file (two-arm measurement).
    """
    from scipy.io import loadmat
    mat = loadmat(str(path), squeeze_me=True, mat_dtype=False)
    # Strip MATLAB meta-keys
    keys = [k for k in mat if not k.startswith('__')]

    def _to_1d(v):
        v = np.asarray(v, dtype=float).ravel()
        return v[np.isfinite(v)]

    # Prefer explicit I1 / I2 pair
    if 'I1' in mat and 'I2' in mat:
        I1 = _to_1d(mat['I1'])
        I2 = _to_1d(mat['I2'])
        N  = min(len(I1), len(I2))
        t  = np.arange(N, dtype=float)
        return t, I1[:N], I2[:N]

    # Fallback: single array
    for name in ('data', 'signal', 'intensity', 'y', 'x'):
        if name in mat:
            y = _to_1d(mat[name])
            return np.arange(len(y), dtype=float), y, None

    # Last resort: first numeric key
    for k in keys:
        v = mat[k]
        if hasattr(v, '__len__') and len(np.asarray(v).ravel()) > 16:
            y = _to_1d(v)
            return np.arange(len(y), dtype=float), y, None

    raise ValueError(f".mat file has no recognisable intensity variable. Keys: {keys}")

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


# ════════════════════════════════════════════════════════════════════════════
#  §75  TX / RX Dual FSM  —  Male (transmit) / Female (receive)
# ════════════════════════════════════════════════════════════════════════════
#
#  Modular arithmetic connection
#  ──────────────────────────────
#  TX cycles through |TX_STATES| = 5 states,  state_idx = step % 5
#  RX cycles through |RX_STATES| = 6 states,  state_idx = step % 6
#  GS inner loop alternates between two constraints:  k % 2
#    k even  →  apply intensity constraint  |E| = sqrt(I)
#    k odd   →  apply dispersion  H(nu) = exp(iπDν²)
#
#  Male FSM  (TX, generates optical signal)
#  IDLE → LOAD_BITS → MODULATE → DISPERSE → SAMPLE → IDLE  (cycle)
#
#  Female FSM  (RX, receives and recovers phase)
#  IDLE → UPLOAD → VALIDATE → PREPROCESS → GS_ITERATE → CONVERGED → IDLE
#
#  Together they form a complementary pair:
#    TX.output  →  (channel)  →  RX.input
#    TX.SAMPLE  ↔  RX.UPLOAD   (handshake)
# ════════════════════════════════════════════════════════════════════════════

class OpticalTxFSM:
    """
    Male FSM — optical transmitter.
    Generates QPSK-modulated, dispersion-stretched intensity measurements.

    States (modulo 5):
      0  IDLE        — waiting for input bits
      1  LOAD_BITS   — accept bit array, set parameters
      2  MODULATE    — QPSK modulate + pulse shape
      3  DISPERSE    — apply H1(nu) and H2(nu)  →  E1(t), E2(t)
      4  SAMPLE      — |E1|², |E2|²  →  I1(t), I2(t)  (ready for TX)
    """
    STATES = ["IDLE", "LOAD_BITS", "MODULATE", "DISPERSE", "SAMPLE"]

    def __init__(self, D1=-600.0, D2=-1200.0, sps=8, beta=0.35):
        self.D1   = D1
        self.D2   = D2
        self.sps  = sps
        self.beta = beta
        self._step  = 0
        self._bits  = None
        self._tx    = None
        self.I1     = None
        self.I2     = None
        self.history = []    # list of (step, state, note)

    @property
    def state(self):
        return self.STATES[self._step % len(self.STATES)]

    def _log(self, note=""):
        self.history.append((self._step, self.state, note))

    def step(self, bits=None):
        """Advance FSM by one state. Pass bits only in IDLE state."""
        s = self.state
        if s == "IDLE":
            if bits is not None:
                self._bits = np.asarray(bits, dtype=int).ravel()
                self._log(f"loaded {len(self._bits)} bits")
            else:
                self._log("idle — no bits")

        elif s == "LOAD_BITS":
            if self._bits is None:
                raise RuntimeError("TX: no bits loaded before LOAD_BITS")
            self._log(f"bits ready: n={len(self._bits)}")

        elif s == "MODULATE":
            self._tx = qpsk_tx(self._bits, self.sps, self.beta)
            self._log(f"QPSK TX waveform: {len(self._tx)} samples")

        elif s == "DISPERSE":
            N   = len(self._tx)
            nu  = np.fft.fftfreq(N)
            H1  = np.exp(1j * PI * self.D1 * nu**2)
            H2  = np.exp(1j * PI * self.D2 * nu**2)
            TX_f = np.fft.fft(self._tx)
            self._E1 = np.fft.ifft(TX_f * H1)
            self._E2 = np.fft.ifft(TX_f * H2)
            self._log(f"dispersed D1={self.D1} D2={self.D2} ps2")

        elif s == "SAMPLE":
            self.I1 = np.abs(self._E1)**2
            self.I2 = np.abs(self._E2)**2
            self._log(f"I1/I2 ready: {len(self.I1)} samples each")

        self._step += 1
        return self.state

    def run(self, bits):
        """Full TX pipeline: bits → I1, I2."""
        self.step(bits)      # IDLE     → load bits
        self.step()          # LOAD_BITS
        self.step()          # MODULATE
        self.step()          # DISPERSE
        self.step()          # SAMPLE
        return self.I1, self.I2


class OpticalRxFSM:
    """
    Female FSM — optical receiver / phase retriever.
    Accepts uploaded I1, I2 and recovers phi(t) via TD-GS.

    States (modulo 6):
      0  IDLE        — waiting for upload
      1  UPLOAD      — accept I1, I2 arrays (or .mat/.npy file path)
      2  VALIDATE    — check shape, remove NaN/Inf, normalise
      3  PREPROCESS  — initialise E_est = sqrt(I1) * exp(i*0)
      4  GS_ITERATE  — run n_iter Gerchberg-Saxton iterations
                        k even: amplitude constraint  (|E| = sqrt(I))
                        k odd:  dispersive constraint (H(nu) applied)
      5  CONVERGED   — phi(t) = angle(E_est) ready for output
    """
    STATES = ["IDLE", "UPLOAD", "VALIDATE", "PREPROCESS", "GS_ITERATE", "CONVERGED"]

    def __init__(self, D1=-600.0, D2=-1200.0, n_iter=50):
        self.D1      = D1
        self.D2      = D2
        self.n_iter  = n_iter
        self._step   = 0
        self.I1      = None
        self.I2      = None
        self._E      = None
        self.phi     = None
        self.residuals = []
        self.history   = []

    @property
    def state(self):
        return self.STATES[self._step % len(self.STATES)]

    def _log(self, note=""):
        self.history.append((self._step, self.state, note))

    def step(self, I1=None, I2=None):
        """Advance FSM by one state. Pass I1/I2 in IDLE state."""
        s = self.state
        if s == "IDLE":
            if I1 is not None:
                self.I1 = np.asarray(I1, dtype=float).ravel()
                self.I2 = np.asarray(I2 if I2 is not None else I1, dtype=float).ravel()
                self._log(f"received I1/I2: {len(self.I1)} samples")
            else:
                self._log("idle — no data")

        elif s == "UPLOAD":
            self._log(f"upload confirmed: N={len(self.I1)}")

        elif s == "VALIDATE":
            N = min(len(self.I1), len(self.I2))
            self.I1 = np.clip(np.nan_to_num(self.I1[:N]), 0, None)
            self.I2 = np.clip(np.nan_to_num(self.I2[:N]), 0, None)
            # Normalise to [0, 1]
            mx = max(self.I1.max(), self.I2.max()) + 1e-12
            self.I1 /= mx;  self.I2 /= mx
            self._log(f"validated N={N}  max_I={mx:.4g}")

        elif s == "PREPROCESS":
            self._E = np.sqrt(self.I1).astype(complex)
            N   = len(self.I1)
            nu  = np.fft.fftfreq(N)
            self._H1 = np.exp(1j * PI * self.D1 * nu**2)
            self._H2 = np.exp(1j * PI * self.D2 * nu**2)
            self._log("E_est = sqrt(I1), transfer functions computed")

        elif s == "GS_ITERATE":
            self.residuals = []
            E = self._E
            sqrt_I1 = np.sqrt(self.I1)
            sqrt_I2 = np.sqrt(self.I2)
            for k in range(self.n_iter):
                if k % 2 == 0:
                    # Even: amplitude constraint  (|T|=1 → unit amplitude)
                    E = sqrt_I1 * np.exp(1j * np.angle(E))
                else:
                    # Odd: dispersive phase constraint
                    E_spec = np.fft.fft(E) * self._H1
                    E_spec = (np.abs(np.fft.fft(sqrt_I2))
                              * np.exp(1j * np.angle(E_spec)))
                    E = np.fft.ifft(E_spec / (self._H1 + 1e-12))
                residual = float(np.mean((np.abs(E) - sqrt_I1)**2))
                self.residuals.append(residual)
            self._E = E
            self._log(f"GS done: {self.n_iter} iter  final_residual={self.residuals[-1]:.4g}")

        elif s == "CONVERGED":
            self.phi = np.angle(self._E)
            self._log(f"phi(t) ready  std={np.std(self.phi):.4f} rad")

        self._step += 1
        return self.state

    def run(self, I1, I2):
        """Full RX pipeline: I1, I2 → phi(t)."""
        self.step(I1, I2)    # IDLE     → accept data
        self.step()          # UPLOAD
        self.step()          # VALIDATE
        self.step()          # PREPROCESS
        self.step()          # GS_ITERATE
        self.step()          # CONVERGED
        return self.phi, self.residuals


def optical_link_fsm_demo(n_bits=256, snr_db=15.0, D1=-600.0, D2=-1200.0, rng_seed=5):
    """
    Full optical communication link using TX/RX dual FSM.

    TX FSM (male)  :  bits → QPSK → disperse → I1, I2
    AWGN channel   :  add noise at snr_db
    RX FSM (female):  I1, I2 → validate → GS → phi(t)

    Returns dict with FSM state histories, residuals, waveforms.
    """
    rng  = np.random.default_rng(rng_seed)
    bits = rng.integers(0, 2, n_bits)

    # ── TX ────────────────────────────────────────────────────────────────
    tx_fsm = OpticalTxFSM(D1=D1, D2=D2)
    I1, I2 = tx_fsm.run(bits)

    # ── Channel: AWGN ─────────────────────────────────────────────────────
    I1_noisy = np.abs(awgn(I1.astype(complex), snr_db))**2
    I2_noisy = np.abs(awgn(I2.astype(complex), snr_db))**2

    # ── RX ────────────────────────────────────────────────────────────────
    rx_fsm = OpticalRxFSM(D1=D1, D2=D2, n_iter=60)
    phi_est, residuals = rx_fsm.run(I1_noisy, I2_noisy)

    # ── State-machine modular arithmetic table ────────────────────────────
    tx_mod = [(i, i % len(OpticalTxFSM.STATES), OpticalTxFSM.STATES[i % len(OpticalTxFSM.STATES)])
              for i in range(len(OpticalTxFSM.STATES) * 2)]
    rx_mod = [(i, i % len(OpticalRxFSM.STATES), OpticalRxFSM.STATES[i % len(OpticalRxFSM.STATES)])
              for i in range(len(OpticalRxFSM.STATES) * 2)]

    # GS k%2 constraint table
    gs_mod = [(k, k % 2, "amplitude" if k % 2 == 0 else "dispersion")
              for k in range(10)]

    return {
        "I1":           I1[:256].tolist(),
        "I2":           I2[:256].tolist(),
        "I1_noisy":     I1_noisy[:256].tolist(),
        "I2_noisy":     I2_noisy[:256].tolist(),
        "phi_est":      phi_est.tolist(),
        "residuals":    residuals,
        "tx_history":   tx_fsm.history,
        "rx_history":   rx_fsm.history,
        "tx_mod_table": tx_mod,
        "rx_mod_table": rx_mod,
        "gs_mod_table": gs_mod,
        "n_bits":       n_bits,
        "snr_db":       snr_db,
        "D1":           D1,
        "D2":           D2,
    }


# ════════════════════════════════════════════════════════════════════════════
#  §74  Odd / Even  ·  Arithmetic Circuits  ·  Ethereum Explicit Trust
# ════════════════════════════════════════════════════════════════════════════
#
#  6-step construction
#  ───────────────────
#  Step 1  Odd/even Cooley-Tukey split
#          X = FFT(x)  is recursively split:
#            x_even = x[0::2],  x_odd = x[1::2]
#            X[k]   = FFT_even[k] + ω^k · FFT_odd[k]     k = 0..N/2-1
#            X[k+N/2] = FFT_even[k] - ω^k · FFT_odd[k]
#
#  Step 2  Butterfly gate  (a, b, ω) → (a + ω·b,  a − ω·b)
#          One gate = two additions + one multiplication = circuit depth 1.
#          N-point FFT needs N/2 · log₂N butterfly gates.
#
#  Step 3  Field mapping  float → F_p
#          Ethereum's secp256k1 prime p = 2²⁵⁶ − 2³² − 977.
#          Map I[n] ∈ [0,1] to  ⌊I[n] · 2³²⌋ mod p  (32-bit fixed-point).
#
#  Step 4  Field butterfly  over F_p  (same gate, modular arithmetic)
#          (a, b, w) → ((a + w·b) mod p,  (a − w·b) mod p)
#          This is exactly one R1CS constraint.
#
#  Step 5  keccak256 commitment
#          C_I1 = keccak256(I1_bytes)
#          C_I2 = keccak256(I2_bytes)
#          C_φ  = keccak256(phi_bytes)
#          Commitments are 32-byte Ethereum bytes32 values.
#
#  Step 6  On-chain explicit trust
#          OpticalPhaseVerifier.sol stores (C_I1, C_I2, C_φ).
#          Anyone with raw data calls verify(id, I1_raw, I2_raw) → bool.
#          No trusted third party. The contract IS the trust.
#
# ════════════════════════════════════════════════════════════════════════════

# ── Step 1 & 2 : Explicit odd/even radix-2 FFT ───────────────────────────────

def fft_butterfly(a: complex, b: complex, w: complex):
    """
    Cooley-Tukey butterfly gate: (a, b, ω) -> (top, bottom).

    top    = a + ω·b    (even combination)
    bottom = a - ω·b    (odd  combination)

    This is one arithmetic circuit gate over C (or F_p for ZK).
    """
    wb = w * b
    return a + wb, a - wb


def fft_radix2_dit(x):
    """
    Radix-2 Decimation-In-Time FFT — explicit odd/even recursive split.

    At every level:
      x_even = x[0::2]   (even-indexed samples)
      x_odd  = x[1::2]   (odd-indexed  samples)
    then butterfly-combine with twiddle factors ω_k = exp(-2πi·k/N).

    Equivalent to numpy.fft.fft but exposing the circuit structure.
    Returns DFT array of same length.  Length must be a power of 2.
    """
    x = np.asarray(x, dtype=complex)
    N = len(x)
    if N == 1:
        return x.copy()
    if N & (N - 1):
        # Pad to next power-of-2 for demonstration
        N2 = 1 << int(np.ceil(np.log2(N)))
        x  = np.pad(x, (0, N2 - N))
        N  = N2

    # ── recursive odd/even split ──────────────────────────────────────────
    X_even = fft_radix2_dit(x[0::2])   # even indices: 0, 2, 4, …
    X_odd  = fft_radix2_dit(x[1::2])   # odd  indices: 1, 3, 5, …

    # ── twiddle factors ───────────────────────────────────────────────────
    k = np.arange(N // 2)
    w = np.exp(-2j * PI * k / N)       # ω_k = e^{-2πik/N}

    # ── butterfly combine ─────────────────────────────────────────────────
    top    = X_even + w * X_odd         # X[k]       k = 0 .. N/2-1
    bottom = X_even - w * X_odd         # X[k + N/2]
    return np.concatenate([top, bottom])


def fft_circuit_stats(N: int):
    """
    Return (n_butterflies, circuit_depth, n_odd_stages, n_even_stages)
    for an N-point radix-2 FFT.

    circuit_depth  = log2(N) levels
    n_butterflies  = (N/2) * log2(N)  gates per full FFT
    Each level alternates between grouping even outputs (top)
    and odd outputs (bottom).
    """
    if N < 2 or (N & (N - 1)):
        return None
    depth = int(np.log2(N))
    return {
        "N":             N,
        "circuit_depth": depth,
        "n_butterflies": (N // 2) * depth,
        "gates_per_GS_iter": 2 * (N // 2) * depth,   # forward + inverse FFT
        "odd_stages":    depth // 2,
        "even_stages":   (depth + 1) // 2,
    }


# ── Step 3 & 4 : Field arithmetic over secp256k1 prime ───────────────────────

# Ethereum secp256k1 field prime
P_SECP256K1 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F

# BN128 (alt_bn128) scalar field prime — used in Groth16 ZK-SNARK on Ethereum
P_BN128 = 0x30644e72e131a029b85045b68181585d2833e84879b9709142e1f0121b70900d

_SCALE_32 = 1 << 32   # 32-bit fixed-point scale


def float_to_field(x: float, p: int = P_BN128, scale: int = _SCALE_32) -> int:
    """Map float x ∈ [0, 1] to prime field element in F_p."""
    return int(round(float(x) * scale)) % p


def field_to_float(fe: int, p: int = P_BN128, scale: int = _SCALE_32) -> float:
    """Inverse: field element → float (approximate, mod p assumed < scale)."""
    return (fe % p) / scale


def field_add(a: int, b: int, p: int = P_BN128) -> int:
    return (a + b) % p


def field_sub(a: int, b: int, p: int = P_BN128) -> int:
    return (a - b) % p


def field_mul(a: int, b: int, p: int = P_BN128) -> int:
    return (a * b) % p


def field_butterfly(a: int, b: int, w: int, p: int = P_BN128):
    """
    Arithmetic circuit butterfly gate over F_p.

    (a, b, w) -> (a + w·b mod p,  a - w·b mod p)

    This is exactly one R1CS (Rank-1 Constraint System) row:
      s·(w) = wb
      top    = a + wb
      bottom = a - wb    (= a + p - wb)
    """
    wb  = field_mul(w, b, p)
    top = field_add(a, wb, p)
    bot = field_sub(a, wb, p)
    return top, bot


def nth_root_of_unity(k: int, N: int, p: int = P_BN128) -> int:
    """
    k-th N-th root of unity in F_p.
    Requires N | (p-1)  (N divides the field order).
    Uses the primitive root g of F_p.
    For BN128: primitive root g = 5.
    """
    g = 5
    # ω_N = g^((p-1)/N)  is a primitive N-th root of unity
    omega_N = pow(g, (p - 1) // N, p)
    return pow(omega_N, k, p)


# ── Step 5 : keccak256 commitment ─────────────────────────────────────────────

def keccak256(data: bytes) -> bytes:
    """
    Ethereum-compatible keccak256 hash.
    Uses pysha3 (pip install pysha3) if available; falls back to sha3_256.
    Note: Python's hashlib.sha3_256 is NIST SHA3, NOT keccak256.
    For production, install: pip install pysha3
    """
    try:
        import sha3 as _sha3
        h = _sha3.keccak_256()
        h.update(data)
        return h.digest()
    except ImportError:
        import hashlib
        # Structural stand-in: sha3_256 (not identical to keccak in last padding)
        return hashlib.sha3_256(data).digest()


def commit_intensity(I: np.ndarray) -> str:
    """keccak256 commitment to intensity array. Returns 0x-prefixed hex string."""
    raw = I.astype(np.float32).tobytes()
    return "0x" + keccak256(raw).hex()


def commit_phase(phi: np.ndarray) -> str:
    """keccak256 commitment to phase solution."""
    raw = phi.astype(np.float32).tobytes()
    return "0x" + keccak256(raw).hex()


# ── Step 6 : Full optical ZK demo ─────────────────────────────────────────────

def optical_zk_demo(N: int = 64, snr_db: float = 20.0, rng_seed: int = 7):
    """
    6-step optical phase retrieval with Ethereum-style explicit trust.

    Step 1  Generate I1, I2 (two dispersed intensity measurements)
    Step 2  Show odd/even FFT split — circuit stats
    Step 3  Map I1[0..15] to BN128 field elements
    Step 4  Run field butterfly on first 4 field elements
    Step 5  Compute keccak256 commitments C_I1, C_I2, C_phi
    Step 6  Simulate OpticalPhaseVerifier on-chain call

    Returns dict with all intermediate values for visualisation.
    """
    rng  = np.random.default_rng(rng_seed)

    # ── Step 1: measurements ──────────────────────────────────────────────
    t   = np.linspace(0, 1, N)
    phi_true = 1.5 * np.sin(2 * PI * 3 * t) + 0.5 * rng.normal(0, 0.1, N)
    D1, D2   = -600.0, -1200.0
    nu        = np.fft.fftfreq(N)
    H1        = np.exp(1j * PI * D1 * nu**2)
    H2        = np.exp(1j * PI * D2 * nu**2)
    E_true    = np.exp(1j * phi_true)
    I1        = np.abs(np.fft.ifft(np.fft.fft(E_true) * H1))**2
    I2        = np.abs(np.fft.ifft(np.fft.fft(E_true) * H2))**2
    snr_lin   = 10**(snr_db / 10)
    noise_std = np.sqrt(I1.mean() / snr_lin)
    I1 += rng.normal(0, noise_std, N)
    I2 += rng.normal(0, noise_std, N)
    I1  = np.clip(I1, 0, None)
    I2  = np.clip(I2, 0, None)

    # ── Step 2: FFT circuit stats ─────────────────────────────────────────
    N_pad   = 1 << int(np.ceil(np.log2(N)))
    stats   = fft_circuit_stats(N_pad)
    X_numpy = np.fft.fft(I1)
    X_radix2 = fft_radix2_dit(I1)
    fft_match = bool(np.allclose(X_numpy[:N], X_radix2[:N], atol=1e-9))

    # Explicit even/odd split at first level
    I1_even = I1[0::2]
    I1_odd  = I1[1::2]

    # ── Step 3: field elements ────────────────────────────────────────────
    n_show = 8
    I1_fe  = [float_to_field(v) for v in I1[:n_show]]
    I1_rec = [field_to_float(fe) for fe in I1_fe]

    # ── Step 4: field butterfly on first 4 pairs ──────────────────────────
    butterfly_results = []
    for i in range(min(4, n_show // 2)):
        a  = I1_fe[i]
        b  = I1_fe[i + n_show // 2]
        w  = nth_root_of_unity(i, n_show)
        top, bot = field_butterfly(a, b, w)
        butterfly_results.append({
            "i": i, "a": hex(a), "b": hex(b), "w": hex(w),
            "top": hex(top), "bot": hex(bot),
        })

    # ── Step 5: commitments ───────────────────────────────────────────────
    # Run one GS iteration to get phi estimate
    E_est = np.sqrt(np.maximum(I1, 0)).astype(complex)
    for _ in range(50):
        E_spec = np.fft.fft(E_est) * H1
        E_spec = np.abs(np.fft.fft(np.sqrt(np.maximum(I2, 0)))) * np.exp(1j * np.angle(E_spec))
        E_est  = np.fft.ifft(E_spec / (H1 + 1e-12))
        E_est  = np.sqrt(np.maximum(I1, 0)) * np.exp(1j * np.angle(E_est))
    phi_est  = np.angle(E_est)

    C_I1 = commit_intensity(I1)
    C_I2 = commit_intensity(I2)
    C_phi = commit_phase(phi_est)

    # ── Step 6: simulate on-chain verification ────────────────────────────
    # In Solidity: verify(id, I1_raw, I2_raw) checks keccak256(I1_raw)==C_I1
    verify_I1 = (commit_intensity(I1)  == C_I1)
    verify_I2 = (commit_intensity(I2)  == C_I2)
    tamper    = I1.copy(); tamper[0] += 1.0
    tamper_rejected = (commit_intensity(tamper) != C_I1)

    return {
        # Step 1
        "I1":            I1.tolist(),
        "I2":            I2.tolist(),
        "phi_true":      phi_true.tolist(),
        "phi_est":       phi_est.tolist(),
        "t":             t.tolist(),
        # Step 2
        "I1_even":       I1_even.tolist(),
        "I1_odd":        I1_odd.tolist(),
        "fft_match":     fft_match,
        "circuit_stats": stats,
        # Step 3
        "I1_float_sample": I1[:n_show].tolist(),
        "I1_field_sample": [hex(fe) for fe in I1_fe],
        "I1_recovered":    I1_rec,
        # Step 4
        "butterfly_results": butterfly_results,
        # Step 5
        "C_I1":          C_I1,
        "C_I2":          C_I2,
        "C_phi":         C_phi,
        # Step 6
        "verify_I1":     verify_I1,
        "verify_I2":     verify_I2,
        "tamper_rejected": tamper_rejected,
        "N":             N,
        "snr_db":        snr_db,
    }
