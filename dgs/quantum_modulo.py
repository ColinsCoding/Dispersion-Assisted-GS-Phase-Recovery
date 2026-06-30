"""Quantum modular arithmetic and phase estimation.

Covers:
  §1  Classical modular exponentiation (fast, used in RSA/Shor)
  §2  Quantum Fourier Transform (QFT) — discrete analog of FFT
  §3  Quantum Phase Estimation (QPE) — measures eigenphase of unitary
  §4  Shor's algorithm structure (period finding via QFT)
  §5  Connection to GHz real-time spectroscopy (quantum noise floor)

SHOR'S ALGORITHM CONNECTION TO JALALI SPECTROSCOPY:
  Shor finds period r of f(x) = a^x mod N using QFT.
  Jalali TS-DFT finds spectral period (absorption line spacing) using FFT.
  Both are period-finding algorithms:
    Shor:   QFT maps |x> -> |k> revealing period r in {a^x mod N}
    Jalali: FFT maps E(t) -> E(omega) revealing spectral lines
  The GHz repetition rate = quantum clock: each pulse = one 'query'
  to the absorption spectrum. Shot noise = quantum measurement limit.
"""

from __future__ import annotations
import numpy as np
import sympy as sp
from typing import Dict, List, Optional, Tuple


# ════════════════════════════════════════════════════════════════════════════
# §1  MODULAR EXPONENTIATION
# ════════════════════════════════════════════════════════════════════════════

def mod_exp(base: int, exp: int, mod: int) -> int:
    """Fast modular exponentiation: base^exp mod mod.

    Uses square-and-multiply (binary exponentiation): O(log exp) multiplications.
    This is the classical subroutine in RSA and Shor's algorithm.
    """
    return pow(base, exp, mod)   # Python built-in is already fast


def mod_exp_sequence(a: int, N: int, x_max: int = 64) -> Dict:
    """Compute f(x) = a^x mod N for x = 0, 1, ..., x_max-1.

    The sequence is PERIODIC with period r (the multiplicative order of a mod N).
    Finding r classically requires O(N) steps.
    Shor's quantum algorithm finds r in O((log N)^3) steps.
    """
    seq = np.array([pow(a, x, N) for x in range(x_max)])
    # Find period r: smallest r such that a^r ≡ 1 (mod N)
    r = None
    for candidate in range(1, N+1):
        if pow(a, candidate, N) == 1:
            r = candidate
            break
    # Verify: f(x+r) = f(x) for all x
    if r is not None and r < x_max:
        periodic = bool(np.all(seq[:x_max-r] == seq[r:x_max]))
    else:
        periodic = False
    return {
        "a": a, "N": N,
        "sequence":  seq,
        "period_r":  r,
        "periodic":  periodic,
        "gcd_check": (sp.gcd(a, N) == 1),   # Shor requires gcd(a,N)=1
    }


def multiplicative_order(a: int, N: int) -> Optional[int]:
    """Find multiplicative order of a mod N (smallest r: a^r ≡ 1 mod N).

    Classical: O(N). Shor quantum: O((log N)^3).
    """
    if sp.gcd(a, N) != 1:
        return None   # a and N not coprime -> no multiplicative order
    r = 1
    val = a % N
    while val != 1:
        val = (val * a) % N
        r += 1
        if r > N:
            return None   # should not happen if gcd=1
    return r


def shor_factor_classical(N: int, n_trials: int = 20,
                           seed: int = 42) -> Dict:
    """Classical simulation of Shor's algorithm for small N.

    Steps:
    1. Choose random a with 1 < a < N, gcd(a,N) = 1
    2. Find period r of a^x mod N (classically, quantum finds via QFT)
    3. If r is even and a^{r/2} ≢ -1 (mod N):
       factor candidates: gcd(a^{r/2} ± 1, N)
    4. Check if factors are non-trivial

    Returns first non-trivial factor found.
    """
    rng = np.random.default_rng(seed)
    for _ in range(n_trials):
        a = int(rng.integers(2, N))
        g = int(sp.gcd(a, N))
        if g > 1:
            return {"factor": g, "method": "gcd_shortcut", "a": a}
        r = multiplicative_order(a, N)
        if r is None or r % 2 != 0:
            continue
        p = pow(a, r//2, N)
        if (p + 1) % N == 0:
            continue
        f1 = int(sp.gcd(p + 1, N))
        f2 = int(sp.gcd(p - 1, N))
        for f in [f1, f2]:
            if 1 < f < N:
                return {"factor": f, "cofactor": N//f, "a": a,
                        "r": r, "p": p, "N": N,
                        "verify": f * (N//f) == N}
    return {"factor": None, "error": "no factor found in n_trials"}


# ════════════════════════════════════════════════════════════════════════════
# §2  QUANTUM FOURIER TRANSFORM (QFT)
# ════════════════════════════════════════════════════════════════════════════

def qft_matrix(n_qubits: int) -> np.ndarray:
    """QFT matrix on n_qubits: U_{jk} = exp(2*pi*i*j*k/N) / sqrt(N).

    N = 2^n_qubits.
    The QFT is the DFT matrix (unitary normalization).
    Classical DFT: O(N^2).  QFT circuit: O(n^2) gates.
    """
    N = 2**n_qubits
    j = np.arange(N)
    k = np.arange(N)
    return np.exp(2j * np.pi * np.outer(j, k) / N) / np.sqrt(N)


def qft_apply(state: np.ndarray) -> np.ndarray:
    """Apply QFT to a quantum state vector |psi> of length 2^n.

    Uses numpy FFT (same operation as DFT, just normalized).
    Classical FFT is O(N log N); QFT circuit is O(n^2) gates.
    """
    N = len(state)
    return np.fft.fft(state) / np.sqrt(N)


def qft_inverse(state: np.ndarray) -> np.ndarray:
    """Inverse QFT (IQFT): U^dagger * |psi>."""
    N = len(state)
    return np.fft.ifft(state) * np.sqrt(N)


def period_from_qft(f_seq: np.ndarray, N_qft: int = None) -> Dict:
    """Find period of f(x) using QFT (simulates the quantum step of Shor).

    The QFT of a periodic sequence with period r shows peaks at
    multiples of N/r (where N = length of sequence).
    """
    n = len(f_seq)
    if N_qft is None:
        N_qft = n
    # Encode: |psi> = (1/sqrt(n)) * sum_x |x>|f(x)>
    # After measuring f(x)=v, state collapses to uniform superposition over x: f(x)=v
    # We approximate: just take FFT of the sequence directly
    F = np.abs(np.fft.fft(f_seq, n=N_qft))**2
    # Find peaks (exclude DC)
    freqs = np.fft.fftfreq(N_qft, d=1)
    F_pos = F[1:N_qft//2]
    k_peak = int(np.argmax(F_pos)) + 1   # dominant frequency bin
    # Period estimate: r = N/k_peak (take rational approximation)
    r_est = N_qft // k_peak if k_peak > 0 else None
    return {
        "F_spectrum":  F,
        "k_peak":      k_peak,
        "r_estimate":  r_est,
        "peak_power":  float(F[k_peak]),
        "freqs":       freqs,
    }


# ════════════════════════════════════════════════════════════════════════════
# §3  QUANTUM PHASE ESTIMATION (QPE)
# ════════════════════════════════════════════════════════════════════════════

def qpe_simulate(phase: float, n_ancilla: int = 8) -> Dict:
    """Simulate Quantum Phase Estimation for unitary U with eigenphase phi.

    U|u> = exp(2*pi*i*phi)|u>

    QPE circuit:
    1. Apply H^n to ancilla: uniform superposition of |0>...|2^n-1>
    2. Apply controlled-U^k for k=0..2^n-1
    3. Apply IQFT to ancilla
    4. Measure ancilla: reads out binary approximation to phi

    Output: measured phase phi_est, error, and probability distribution.

    phase     : true eigenphase phi in [0, 1)
    n_ancilla : number of ancilla qubits (precision = 1/2^n bits)
    """
    N = 2**n_ancilla
    # After IQFT, amplitude at register value k is:
    # a_k = (1/N) * sum_{j=0}^{N-1} exp(2*pi*i*(phi - k/N)*j)
    # = (1/N) * (1 - exp(2*pi*i*(phi - k/N)*N)) / (1 - exp(2*pi*i*(phi - k/N)))
    k_arr = np.arange(N)
    delta = phase - k_arr / N
    # Avoid division by zero at exact multiples
    eps = 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        amp = np.where(
            np.abs(delta) < eps,
            1.0,
            np.sin(np.pi * delta * N) / (N * np.sin(np.pi * delta + eps))
        )
    prob = amp**2
    prob = np.abs(prob)
    prob /= prob.sum()

    k_best = int(np.argmax(prob))
    phi_est = k_best / N
    error   = abs(phase - phi_est)
    precision = 1 / N

    return {
        "phi_true":   phase,
        "phi_est":    phi_est,
        "error":      error,
        "precision":  precision,
        "prob":       prob,
        "k_best":     k_best,
        "n_ancilla":  n_ancilla,
        "N_states":   N,
        "success_prob": float(prob[k_best]),
    }


# ════════════════════════════════════════════════════════════════════════════
# §4  SHOR'S ALGORITHM — FULL STRUCTURE
# ════════════════════════════════════════════════════════════════════════════

def shor_algorithm_steps(N: int, a: int) -> Dict:
    """Trace through all steps of Shor's algorithm for factoring N.

    Classical steps: input validation, gcd shortcuts.
    Quantum step:    QPE / QFT (simulated here).
    Post-processing: period r -> factor candidates.
    """
    steps = []
    # Step 1: Check trivially composite
    if N % 2 == 0:
        return {"factor": 2, "method": "trivial_even", "steps": ["N even"]}
    steps.append(f"N={N} is odd, proceeding.")

    # Step 2: Check perfect power N = p^k
    for k in range(2, int(np.log2(N))+1):
        root = round(N**(1/k))
        for r in [root-1, root, root+1]:
            if r >= 2 and r**k == N:
                return {"factor": r, "method": f"perfect_power_{k}", "steps": steps}

    # Step 3: Classical gcd shortcut
    g = int(sp.gcd(a, N))
    if g > 1:
        return {"factor": g, "method": "gcd", "a": a, "steps": steps}
    steps.append(f"gcd({a},{N})=1, proceeding to period finding.")

    # Step 4: Find period r (classically, simulating quantum step)
    r = multiplicative_order(a, N)
    steps.append(f"Multiplicative order r = {r}")
    if r is None:
        return {"factor": None, "error": "order not found", "steps": steps}

    # Step 5: QFT check — simulate period finding via QFT
    x_vals = np.arange(min(8*r, 256))
    f_vals = np.array([pow(a, int(x), N) for x in x_vals])
    qft_res = period_from_qft(f_vals)
    steps.append(f"QFT peak at k={qft_res['k_peak']}, r_est={qft_res['r_estimate']}")

    # Step 6: Check r is even
    if r % 2 != 0:
        return {"factor": None, "error": f"r={r} is odd", "steps": steps}
    steps.append(f"r={r} is even.")

    # Step 7: Factor candidates
    p = pow(a, r//2, N)
    if (p+1) % N == 0:
        return {"factor": None, "error": "a^(r/2) = -1 mod N", "steps": steps}
    f1 = int(sp.gcd(p+1, N))
    f2 = int(sp.gcd(p-1, N))
    steps.append(f"gcd(a^(r/2)+1, N) = {f1}, gcd(a^(r/2)-1, N) = {f2}")

    for f in [f1, f2]:
        if 1 < f < N:
            steps.append(f"Non-trivial factor found: {f}")
            return {"factor": f, "cofactor": N//f, "a": a, "r": r,
                    "verify": f*(N//f) == N, "steps": steps,
                    "qft_result": qft_res}
    return {"factor": None, "error": "factors trivial", "steps": steps}


# ════════════════════════════════════════════════════════════════════════════
# §5  CONNECTION TO GHz SPECTROSCOPY
# ════════════════════════════════════════════════════════════════════════════

def spectroscopy_quantum_limit(rep_rate_GHz: float,
                                n_shots: int,
                                wavelength_nm: float = 1550.0) -> Dict:
    """Quantum noise floor for real-time TS-DFT spectroscopy.

    Each pulse = one quantum measurement of the spectrum.
    Shot noise limit: SNR ~ sqrt(n_photons_per_pulse).
    At GHz repetition rate, n_shots measurements per second.

    Heisenberg uncertainty: Delta_t * Delta_E >= hbar/2
    -> Delta_nu >= 1/(4*pi*Delta_t)

    Parameters
    ----------
    rep_rate_GHz: laser repetition rate (GHz)
    n_shots:      number of pulses averaged
    wavelength_nm: center wavelength (nm)
    """
    HBAR   = 1.0546e-34
    H_PLANCK = 6.626e-34
    C_LIGHT  = 2.998e8
    K_BOLTZ  = 1.381e-23

    rep_rate_Hz = rep_rate_GHz * 1e9
    T_rep = 1 / rep_rate_Hz              # pulse period (s)
    nu_0  = C_LIGHT / (wavelength_nm * 1e-9)   # optical frequency (Hz)
    E_photon = H_PLANCK * nu_0

    # Heisenberg time-bandwidth: Delta_nu >= 1/(4*pi*T_rep)
    delta_nu_min = 1 / (4 * np.pi * T_rep)

    # Shot noise SNR after n_shots
    # For n_ph photons per pulse: SNR = sqrt(n_ph * n_shots)
    n_ph_typical = 1e6   # 1 million photons per pulse (typical pulsed laser)
    SNR_1shot = np.sqrt(n_ph_typical)
    SNR_avg   = np.sqrt(n_ph_typical * n_shots)

    # Bose-Einstein thermal photon occupation at room temp
    x = HBAR * 2*np.pi*nu_0 / (K_BOLTZ * 300)
    n_BE = 1 / (np.exp(x) - 1) if x < 700 else 0.0

    # Equivalent quantum: one period-finding cycle of Shor uses O(log N)^3 gates
    # One spectral measurement uses O(N*log N) operations (FFT)
    N_fft = int(rep_rate_GHz * 1e3)   # ~GHz bandwidth / MHz resolution = 1000 bins
    ops_fft = N_fft * np.log2(N_fft)

    return {
        "rep_rate_GHz":   rep_rate_GHz,
        "T_rep_ps":       T_rep * 1e12,
        "nu_0_THz":       nu_0 / 1e12,
        "E_photon_eV":    E_photon / 1.602e-19,
        "delta_nu_min_MHz": delta_nu_min / 1e6,
        "SNR_1shot":      float(SNR_1shot),
        "SNR_avg":        float(SNR_avg),
        "n_BE_at_300K":   n_BE,
        "shot_noise_limited": n_BE < 0.01,
        "N_fft_bins":     N_fft,
        "ops_FFT":        ops_fft,
        "spectra_per_sec": rep_rate_GHz * 1e9,
    }


# ════════════════════════════════════════════════════════════════════════════
# §6  SYMPY: 5 KEY QUANTUM MODULO EQUATIONS
# ════════════════════════════════════════════════════════════════════════════

def quantum_modulo_sympy_5() -> Dict:
    """5 key equations: QFT, QPE, modular exponentiation, Shor."""
    n, N, r, k, j, a = sp.symbols("n N r k j a", integer=True, positive=True)
    phi = sp.Symbol("phi", real=True)
    # 1. QFT matrix element
    eq1 = sp.Eq(sp.Symbol("U_jk"),
                sp.exp(2*sp.pi*sp.I*j*k/N) / sp.sqrt(N))
    # 2. QPE: probability at register k
    eq2 = sp.Eq(sp.Symbol("P(k)"),
                sp.Abs(sp.sin(sp.pi*(phi - k/N)*N) /
                       (N * sp.sin(sp.pi*(phi - k/N))))**2)
    # 3. Shor period-finding: QFT of a^x mod N peaks at multiples of N/r
    eq3 = sp.Eq(sp.Symbol("QFT_peak_k"), sp.Symbol("j") * N / r)
    # 4. Modular exponentiation (fast: O(log exp))
    eq4 = sp.Eq(sp.Symbol("a^x mod N"), sp.Mod(a**sp.Symbol("x"), N))
    # 5. Factor from period: gcd(a^{r/2} ± 1, N)
    eq5 = sp.Eq(sp.Symbol("factor"),
                sp.gcd(a**sp.Symbol("r/2") + 1, N))
    return {
        "QFT_element":   eq1,
        "QPE_prob":      eq2,
        "Shor_QFT_peak": eq3,
        "Mod_exp":       eq4,
        "Shor_factor":   eq5,
    }


if __name__ == "__main__":
    sp.init_printing(use_latex=False)

    print("=== Modular Exponentiation Sequence: a=7, N=15 ===")
    seq = mod_exp_sequence(7, 15, 32)
    print(f"  f(x) = 7^x mod 15: {seq['sequence'][:16]}")
    print(f"  Period r = {seq['period_r']}")
    print(f"  Periodic: {seq['periodic']}")

    print("\n=== Shor's Algorithm: Factor N=15 ===")
    result = shor_algorithm_steps(15, 7)
    for step in result["steps"]:
        print(f"  {step}")
    print(f"  Factor: {result.get('factor')} x {result.get('cofactor')} = "
          f"{result.get('factor',0)*result.get('cofactor',0)}")
    print(f"  Verified: {result.get('verify')}")

    print("\n=== Quantum Phase Estimation: phi = 1/3 ===")
    qpe = qpe_simulate(1/3, n_ancilla=8)
    print(f"  True phi = {qpe['phi_true']:.6f}")
    print(f"  Estimated phi = {qpe['phi_est']:.6f}")
    print(f"  Error = {qpe['error']:.2e}  (precision = {qpe['precision']:.2e})")
    print(f"  Success probability = {qpe['success_prob']:.4f}")

    print("\n=== GHz Spectroscopy Quantum Limit (Jalali TS-DFT) ===")
    ql = spectroscopy_quantum_limit(rep_rate_GHz=1.0, n_shots=1000)
    print(f"  Rep rate: {ql['rep_rate_GHz']} GHz")
    print(f"  Heisenberg limit: delta_nu >= {ql['delta_nu_min_MHz']:.3f} MHz")
    print(f"  SNR (1 shot): {ql['SNR_1shot']:.0f}")
    print(f"  SNR (1000 avg): {ql['SNR_avg']:.0f}")
    print(f"  n_BE at 1550nm/300K = {ql['n_BE_at_300K']:.2e} (shot-noise limited: {ql['shot_noise_limited']})")
    print(f"  Spectra/sec = {ql['spectra_per_sec']:.2e}")

    print("\n=== 5 SymPy Equations ===")
    for name, eq in quantum_modulo_sympy_5().items():
        print(f"  [{name}]  {eq}")
