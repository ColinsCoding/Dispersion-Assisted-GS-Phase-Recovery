"""Photonic AI -- optical neural networks, MZI meshes, energy, and neuromorphic photonics.

THE JALALI LAB PROBLEM (and the Columbia/Bell Labs two sides):
  SIDE 1 -- Shannon / Information Theory (Bell Labs heritage):
    Every optical communication channel has a Shannon capacity:
    C = B * log2(1 + SNR) bits/s
    The GS phase retrieval algorithm MAXIMALLY USES this capacity -- it encodes
    information in the PHASE of the field, recovering all degrees of freedom.
    Without GS, a direct-detection receiver wastes half the information (throws
    away the phase).

  SIDE 2 -- Physical AI / Photonic Computing (Columbia Lipson group, MIT, Jalali):
    Matrix-vector multiplications (the inner loop of every neural network) can be
    done OPTICALLY at the speed of light with near-zero energy:
      Electronic multiply: ~1-100 pJ/operation (Joule heating in CMOS)
      Photonic multiply:   ~0.001-0.1 fJ/operation (photon propagation, no heating)
    This is a 10^4 - 10^6x energy advantage for inference at the physical layer.

    The key hardware primitive: the Mach-Zehnder Interferometer (MZI).
    Two MZIs implement an arbitrary 2x2 unitary.
    N*(N-1)/2 MZIs (Reck/Clements decomposition) implement any N×N unitary.
    ANY LINEAR LAYER of a neural network is (up to singular values) a product
    of two unitary matrices (SVD): W = U * Sigma * V^H.
    So: EVERY DENSE LAYER can be implemented in silicon photonics.

TURNING NEURONS ON AND OFF (the Jalali optogenetics connection):
  ChR2 ion channel (blue light ON):  I_ChR2 ~ 5 uA/cm^2 (see cellular_biophysics.py)
  NpHR (yellow light OFF):           I_NpHR ~ -3 uA/cm^2 (hyperpolarizing)
  The OPTICAL NEURAL NETWORK controls which neurons fire:
    - Forward pass through MZI mesh: encodes stimulus as optical field
    - Dispersive fiber (GS element): applies phase transformation H(f) = exp(i*pi*D*f^2)
    - Photodetector: intensity pattern drives optogenetic fiber array
    - ChR2/NpHR in tissue: turns specific neurons on or off
  This is the full loop: photonic AI computes -> light actuates biology -> biology responds.

MZI AS A 2x2 UNITARY (connection to Jones calculus):
  The MZI Jones matrix with phase shifts theta (splitting) and phi (phase):
    U_MZI(theta, phi) = i * [[sin(theta)*e^(i*phi), cos(theta)],
                               [cos(theta)*e^(i*phi), -sin(theta)]]
  THIS IS THE SAME MATH as a half-wave plate in Jones calculus (jones_calculus.py).
  Polarization optics = photonic computing. The chain rule for cascaded MZIs
  is the same chain rule for cascaded wave plates.

CLEMENTS DECOMPOSITION:
  Any N×N unitary can be decomposed into N*(N-1)/2 MZIs arranged in a rectangular
  mesh (Clements et al. Optica 2016). This is the fundamental theorem of
  photonic linear algebra.

  Depth = N (vs Reck's triangular mesh depth = 2N-3).
  Clements is shallower (fewer layers) -> faster, lower loss.

SPIKING OPTICAL NEURON (neuromorphic photonics):
  Optical nonlinearity in microring resonators acts as a threshold device:
  - Below threshold power: ring is off-resonance (linear)
  - Above threshold power: thermal or carrier-induced shift triggers bistable switching
  This is a PHOTONIC integrate-and-fire neuron.
  The Hodgkin-Huxley model (cellular_biophysics.py) is the biological prototype;
  the photonic spiking neuron is its silicon implementation.

ENERGY BREAKDOWN:
  Operation          | Electronic (CMOS) | Photonic (SiPh)
  -------------------|-------------------|------------------
  1-bit multiply     | 0.3 fJ            | 0.01 fJ (MZI)
  N=1024 matmul      | 300 nJ            | 0.01 nJ (10^4x)
  Memory read (DRAM) | 100 pJ/bit        | N/A (no memory wall)
  Interconnect 1m    | 300 fJ/bit        | 3 fJ/bit (100x)
  Photodetection     | N/A               | 0.01 fJ/bit
  ADC (1 GSa/s 8b)   | 500 fJ/conversion | unavoidable (at output)
  BOTTLENECK:        | Memory bandwidth  | ADC at output, laser power

  THE MEMORY WALL: photonic compute avoids the von Neumann bottleneck for
  INFERENCE (weights = fixed optical phases, set once, no memory reads).
  Training still requires digital electronics (gradients need precision).
"""
import numpy as np
import sympy as sp


# ── MZI unitary ───────────────────────────────────────────────────────

def mzi_matrix(theta, phi):
    """Jones matrix of a single Mach-Zehnder Interferometer.

    The MZI consists of:
      1. 50:50 beamsplitter (theta = pi/4 for ideal 50:50)
      2. Phase shifter phi in one arm
      3. 50:50 beamsplitter

    U = i * [[sin(theta)*e^(i*phi), cos(theta)],
              [cos(theta)*e^(i*phi), -sin(theta)]]

    theta in [0, pi/2]: controls splitting ratio (power in each arm)
    phi in [0, 2*pi]:   controls relative phase between arms

    At theta=pi/4: balanced (50:50) -- maximum interference contrast.
    """
    c, s = np.cos(theta), np.sin(theta)
    M = 1j * np.array([
        [s * np.exp(1j * phi), c],
        [c * np.exp(1j * phi), -s]
    ], dtype=complex)
    return M


def mzi_transmission(theta, phi, E_in=None):
    """Transmission of one MZI for an input optical field.

    If E_in is None, assumes [1, 0] (input in port 0 only).
    Returns output field and power in both ports.
    """
    if E_in is None:
        E_in = np.array([1.0, 0.0], dtype=complex)
    M = mzi_matrix(theta, phi)
    E_out = M @ np.asarray(E_in, dtype=complex)
    T0 = float(np.abs(E_out[0])**2)
    T1 = float(np.abs(E_out[1])**2)
    return {"E_out": E_out, "T_port0": T0, "T_port1": T1,
            "total_power": T0 + T1}


# ── Clements decomposition (N×N unitary from MZI mesh) ────────────────

def clements_mzi_count(N):
    """Number of MZIs and phase shifters in a Clements N×N mesh.

    N*(N-1)/2 MZIs + N*(N-1)/2 + N = N^2/2 + N/2 phase shifters total.
    Depth = N layers (rectangular mesh, shallower than Reck's triangular).
    """
    if N < 1:
        raise ValueError("N must be >= 1")
    n_mzi = N * (N - 1) // 2
    n_phase = N * (N - 1) // 2 + N   # MZI phases + output phases
    depth = N
    return {
        "N": N, "n_MZI": n_mzi, "n_phase_shifters": n_phase,
        "depth_layers": depth,
        "note": "Reck decomposition: N*(N-1)/2 MZIs, depth=2N-3 (deeper, older)"
    }


def random_unitary_mzi_params(N, rng=None):
    """Generate random MZI parameters for a Haar-random N×N unitary.

    In practice, training a photonic neural network means OPTIMIZING these
    theta and phi values to minimize a loss function -- same as backprop
    but the parameters are optical phases, not floating-point weights.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n_mzi = N * (N - 1) // 2
    thetas = rng.uniform(0, np.pi / 2, n_mzi)
    phis   = rng.uniform(0, 2 * np.pi, n_mzi)
    output_phases = rng.uniform(0, 2 * np.pi, N)
    return {"thetas": thetas, "phis": phis,
            "output_phases": output_phases, "N": N, "n_mzi": n_mzi}


def optical_matmul_svd(W_matrix):
    """Decompose a real/complex weight matrix W = U * Sigma * V^H for photonic impl.

    Any m×n matrix W can be implemented in silicon photonics as:
      1. V^H: N×N unitary (Clements MZI mesh)
      2. Sigma: diagonal (optical attenuators / VOAs)
      3. U: M×M unitary (Clements MZI mesh)

    Returns the SVD components and the optical hardware cost.
    """
    W = np.asarray(W_matrix, dtype=complex)
    m, n = W.shape
    U, sigma, Vh = np.linalg.svd(W, full_matrices=True)
    condition_number = sigma[0] / sigma[-1] if sigma[-1] > 1e-12 else float("inf")
    return {
        "U": U, "sigma": sigma, "Vh": Vh,
        "rank": int(np.sum(sigma > 1e-10)),
        "condition_number": condition_number,
        "mzi_cost_U": clements_mzi_count(m)["n_MZI"],
        "mzi_cost_Vh": clements_mzi_count(n)["n_MZI"],
        "n_attenuators": min(m, n),
        "total_mzi": clements_mzi_count(m)["n_MZI"] + clements_mzi_count(n)["n_MZI"],
    }


# ── energy comparison: photonic vs electronic ─────────────────────────

ENERGY_TABLE = {
    "electronic_multiply_fJ":  300.0,     # 1 fp32 multiply in CMOS 7nm
    "electronic_add_fJ":       100.0,     # fp32 add
    "dram_read_pJ_per_byte":   10.0,      # HBM2e: ~2 pJ/byte; DDR4: ~10 pJ/byte
    "photonic_mzi_fJ":         0.05,      # MZI phase shift (thermal tuning power * transit time)
    "photonic_propagation_fJ": 0.001,     # waveguide propagation per mm
    "photonic_detector_fJ":    0.01,      # photodetector (avalanche gain near 0.01 fJ/bit)
    "photonic_adc_pJ":         0.5,       # ADC at output (unavoidable digital-analog interface)
    "laser_mW_per_channel":    5.0,       # DFB laser power per wavelength channel (WDM)
}


def energy_per_multiply(N, architecture="electronic_gpu"):
    """Energy per multiply-accumulate (MAC) for N×N matrix multiply.

    Compares electronic GPU (A100) vs photonic MZI mesh.
    Bottleneck for electronics: DRAM reads (weights stored in DRAM).
    Bottleneck for photonics: laser power + ADC at output.
    """
    if N < 1:
        raise ValueError("N must be >= 1")
    n_macs = N * N   # for matrix-vector product

    if architecture == "electronic_gpu":
        # fp32 on A100: peak 312 TFLOPS, power 400W, time = N^2 / 312e12
        time_s = n_macs / 312e12
        energy_compute_J = 400 * time_s
        # DRAM reads: N^2 weights * 4 bytes * 10 pJ/byte
        energy_dram_J = N * N * 4 * ENERGY_TABLE["dram_read_pJ_per_byte"] * 1e-12
        total_J = energy_compute_J + energy_dram_J
        return {
            "architecture": architecture,
            "N": N, "n_macs": n_macs,
            "energy_total_J": total_J,
            "energy_per_mac_fJ": total_J / n_macs * 1e15,
            "bottleneck": "DRAM reads dominate for large N",
        }

    elif architecture == "photonic_mzi":
        # Inference: weights are fixed phases -- no DRAM reads
        # Energy: laser power during signal transit (~ns) + ADC at output
        transit_time_s = 1e-9   # ~1 ns (1 cm chip)
        laser_power_W = ENERGY_TABLE["laser_mW_per_channel"] * 1e-3 * N   # N WDM channels
        energy_laser_J = laser_power_W * transit_time_s
        energy_adc_J = N * ENERGY_TABLE["photonic_adc_pJ"] * 1e-12   # N output ports
        total_J = energy_laser_J + energy_adc_J
        return {
            "architecture": architecture,
            "N": N, "n_macs": n_macs,
            "energy_total_J": total_J,
            "energy_per_mac_fJ": total_J / n_macs * 1e15,
            "bottleneck": "Laser power + ADC at output",
            "advantage_vs_gpu": None,   # computed below
        }
    else:
        raise ValueError(f"Unknown architecture '{architecture}'")


def energy_advantage_ratio(N):
    """Photonic vs electronic energy advantage for N×N matmul."""
    e_elec = energy_per_multiply(N, "electronic_gpu")
    e_phot = energy_per_multiply(N, "photonic_mzi")
    ratio = e_elec["energy_per_mac_fJ"] / e_phot["energy_per_mac_fJ"]
    return {
        "N": N,
        "electronic_fJ_per_mac": e_elec["energy_per_mac_fJ"],
        "photonic_fJ_per_mac": e_phot["energy_per_mac_fJ"],
        "advantage_ratio": ratio,
        "log10_advantage": np.log10(ratio),
    }


# ── Shannon capacity of the photonic GS receiver ─────────────────────

def shannon_capacity(SNR_dB, bandwidth_Hz):
    """Shannon channel capacity: C = B * log2(1 + SNR) bits/s.

    This is the maximum information rate for a Gaussian channel.
    The GS receiver aims to achieve near-Shannon-capacity by recovering
    the PHASE of the field -- direct detection loses log2(e)/2 bits/s
    per mode (throws away the imaginary part of the complex field).
    """
    if bandwidth_Hz <= 0:
        raise ValueError("bandwidth must be positive")
    SNR = 10 ** (SNR_dB / 10)
    C_bps = bandwidth_Hz * np.log2(1 + SNR)
    C_direct_detection = bandwidth_Hz * 0.5 * np.log2(1 + SNR)  # half info (no phase)
    return {
        "C_bps": C_bps,
        "C_Gbps": C_bps / 1e9,
        "C_direct_detection_Gbps": C_direct_detection / 1e9,
        "phase_recovery_gain_bits": (C_bps - C_direct_detection) / bandwidth_Hz,
        "bandwidth_Hz": bandwidth_Hz,
        "SNR_dB": SNR_dB,
        "SNR_linear": SNR,
    }


def gs_as_optical_fourier_layer(N_samples, D_ps2=-5000.0):
    """Model the GS dispersive fiber as a photonic Fourier neural network layer.

    The dispersive transfer function H(f) = exp(i*pi*D*f^2) is a CHIRP MULTIPLICATION
    in frequency domain. The sequence:
      1. Dispersive propagation: multiply by H(f) in freq domain
      2. Intensity detection: |E(t)|^2
      3. GS iteration: infer the phase
    is equivalent to a COMPLEX-VALUED FOURIER FEATURE LAYER (like in Fourier
    Neural Operators, FNO -- already in gs_fno.py).

    The dispersion parameter D plays the role of a LEARNABLE FREQUENCY FILTER:
    larger |D| -> more frequency diversity -> better phase recovery.
    This is why |D| >= 5000 ps^2 is required (see feedback_gs_convergence.md).
    """
    if N_samples < 1:
        raise ValueError("N_samples must be >= 1")
    f = np.fft.fftfreq(N_samples)   # normalized frequency
    H = np.exp(1j * np.pi * D_ps2 * f**2)   # chirp transfer function
    bandwidth_used = np.sum(np.abs(H) > 0.5) / N_samples
    phase_range_rad = np.max(np.angle(H)) - np.min(np.angle(H))
    return {
        "N": N_samples, "D_ps2": D_ps2,
        "H_f": H,
        "H_magnitude": np.abs(H),   # all 1 (all-pass filter)
        "H_phase_range_rad": phase_range_rad,
        "H_phase_range_cycles": phase_range_rad / (2 * np.pi),
        "diversity_metric": 1.0 - np.mean(np.abs(H[:N_samples//2])**2),
        "as_fourier_layer": "H(f) = exp(i*pi*D*f^2) is a learnable quadratic phase filter",
        "fno_connection": "This is a SpectralConv1d with fixed (non-learned) phase weights",
    }


# ── photonic spiking neuron ───────────────────────────────────────────

def photonic_spiking_neuron(P_in_mW, P_threshold_mW=1.0,
                             tau_rise_ns=0.5, tau_fall_ns=2.0, dt_ns=0.01):
    """Simple model of a photonic spiking neuron (microring resonator bistability).

    Above P_threshold: the ring shifts to a new resonance (thermal or free-carrier)
    and emits a brief optical pulse (spike). Below threshold: linear response.

    This is the photonic analogue of the Hodgkin-Huxley integrate-and-fire model
    (cellular_biophysics.py). Same mathematics, optical platform.

    Returns: spike waveform and spike count for a constant input.
    """
    if P_threshold_mW <= 0 or tau_rise_ns <= 0 or tau_fall_ns <= 0:
        raise ValueError("threshold and time constants must be positive")

    t = np.arange(0, 20 * (tau_rise_ns + tau_fall_ns), dt_ns)
    P_out = np.zeros_like(t)
    state = 0.0   # 0=off, 1=on
    integration = 0.0
    tau_int = tau_rise_ns
    spikes = []

    for i, ti in enumerate(t):
        if P_in_mW > P_threshold_mW:
            integration += (P_in_mW - P_threshold_mW) * dt_ns / tau_int
        else:
            integration = max(0, integration - dt_ns / tau_fall_ns)

        if integration > 1.0 and state == 0:
            state = 1.0
            spikes.append(ti)
        elif integration <= 0.0:
            state = 0.0

        # Output: Gaussian pulse shape
        if spikes:
            t_since_last = ti - spikes[-1]
            P_out[i] = state * P_in_mW * np.exp(-0.5 * (t_since_last / (tau_rise_ns/2))**2)
            if t_since_last > 2 * tau_fall_ns:
                state = 0.0
                integration = 0.0

    return {
        "t_ns": t, "P_out_mW": P_out,
        "n_spikes": len(spikes),
        "spike_times_ns": np.array(spikes),
        "P_in_mW": P_in_mW, "P_threshold_mW": P_threshold_mW,
        "is_spiking": P_in_mW > P_threshold_mW,
    }


# ── photonic AI training energy ───────────────────────────────────────

def photonic_training_cost(N_layers, N_size, n_epochs, n_samples,
                           photonic_inference_frac=0.9):
    """Estimate training cost for a photonic neural network.

    Training cannot be done purely optically (needs precise gradient computation).
    Hybrid approach: forward pass in photonics (fast, low energy), backward pass
    in digital (CUDA on GPU).

    photonic_inference_frac: fraction of forward passes done optically.
    """
    if N_layers < 1 or N_size < 1:
        raise ValueError("N_layers and N_size must be positive")
    n_forward_passes = n_epochs * n_samples

    # Digital-only baseline
    e_digital = energy_per_multiply(N_size, "electronic_gpu")
    E_digital_J = n_forward_passes * N_layers * e_digital["energy_total_J"]

    # Photonic forward pass
    e_photonic = energy_per_multiply(N_size, "photonic_mzi")
    E_photonic_J = (n_forward_passes * N_layers *
                    (photonic_inference_frac * e_photonic["energy_total_J"] +
                     (1 - photonic_inference_frac) * e_digital["energy_total_J"]))

    # Backward pass always digital (same order as forward)
    E_backward_J = n_forward_passes * N_layers * e_digital["energy_total_J"]

    E_total_digital_J = E_digital_J + E_backward_J
    E_total_photonic_J = E_photonic_J + E_backward_J

    return {
        "N_layers": N_layers, "N_size": N_size,
        "n_epochs": n_epochs, "n_samples": n_samples,
        "E_digital_training_kWh": E_total_digital_J / 3.6e6,
        "E_photonic_hybrid_kWh": E_total_photonic_J / 3.6e6,
        "training_speedup": E_total_digital_J / E_total_photonic_J,
        "note": "Backward pass always digital; speedup comes from forward inference",
    }


# ── SymPy formalism ───────────────────────────────────────────────────

def photonic_ai_sympy_5():
    """Five key photonic AI equations in SymPy."""
    theta, phi = sp.symbols('theta phi', real=True)
    B_s, SNR_s = sp.symbols('B SNR', positive=True)
    D_s, f_s = sp.symbols('D f', real=True)
    N_s = sp.Symbol('N', positive=True, integer=True)
    U_s, Sigma_s, V_s = sp.symbols('U Sigma V^H')

    return {
        "MZI_matrix":
            sp.Eq(sp.Symbol('U_MZI'),
                  sp.I * sp.Matrix([
                      [sp.sin(theta)*sp.exp(sp.I*phi), sp.cos(theta)],
                      [sp.cos(theta)*sp.exp(sp.I*phi), -sp.sin(theta)]
                  ])),
        "Shannon_capacity":
            sp.Eq(sp.Symbol('C'),
                  B_s * sp.log(1 + SNR_s, 2)),
        "Dispersion_transfer_function":
            sp.Eq(sp.Symbol('H(f)'),
                  sp.exp(sp.I * sp.pi * D_s * f_s**2)),
        "SVD_weight_decomposition":
            sp.Eq(sp.Symbol('W'), U_s * Sigma_s * V_s),
        "Clements_MZI_count":
            sp.Eq(sp.Symbol('n_MZI'),
                  N_s * (N_s - 1) / 2),
    }


if __name__ == "__main__":
    print("=== MZI transmission (theta=pi/4 balanced, phi=0) ===")
    r = mzi_transmission(np.pi/4, 0.0)
    print(f"  T_port0 = {r['T_port0']:.4f},  T_port1 = {r['T_port1']:.4f}")
    print(f"  Total power: {r['total_power']:.6f}")

    print("\n=== Clements N=8 MZI mesh (1 layer deep = 8) ===")
    c = clements_mzi_count(8)
    print(f"  N={c['N']}: {c['n_MZI']} MZIs, {c['n_phase_shifters']} phase shifters, "
          f"depth={c['depth_layers']}")

    print("\n=== Energy advantage: photonic vs electronic ===")
    for N in [64, 512, 4096]:
        r = energy_advantage_ratio(N)
        print(f"  N={N:>5}: electronic={r['electronic_fJ_per_mac']:.1f} fJ/MAC, "
              f"photonic={r['photonic_fJ_per_mac']:.4f} fJ/MAC, "
              f"advantage={r['advantage_ratio']:.0f}x")

    print("\n=== Shannon capacity of GS receiver (10 GHz BW, 20 dB SNR) ===")
    sc = shannon_capacity(20, 10e9)
    print(f"  Full coherent: {sc['C_Gbps']:.1f} Gbps")
    print(f"  Direct detection: {sc['C_direct_detection_Gbps']:.1f} Gbps")
    print(f"  Phase recovery gain: {sc['phase_recovery_gain_bits']:.2f} bits/symbol")

    print("\n=== GS algorithm as optical Fourier layer ===")
    gs = gs_as_optical_fourier_layer(1024, D_ps2=-5000)
    print(f"  D=-5000 ps^2, N=1024: phase range = {gs['H_phase_range_cycles']:.1f} cycles")
    print(f"  FNO connection: {gs['fno_connection']}")

    print("\n=== SVD decomposition of random 8x8 weight matrix ===")
    rng = np.random.default_rng(0)
    W = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
    W /= np.sqrt(8)
    svd = optical_matmul_svd(W)
    print(f"  Rank={svd['rank']}, condition={svd['condition_number']:.2f}")
    print(f"  Total MZIs needed: {svd['total_mzi']} (U: {svd['mzi_cost_U']}, V^H: {svd['mzi_cost_Vh']})")

    print("\n=== Photonic spiking neuron ===")
    spike = photonic_spiking_neuron(2.0, P_threshold_mW=1.0)
    print(f"  P_in=2 mW (above threshold): spiking={spike['is_spiking']}, "
          f"n_spikes={spike['n_spikes']}")
    no_spike = photonic_spiking_neuron(0.5, P_threshold_mW=1.0)
    print(f"  P_in=0.5 mW (below threshold): spiking={no_spike['is_spiking']}")

    print("\n=== Training cost: 4-layer N=64 photonic net ===")
    tc = photonic_training_cost(4, 64, n_epochs=100, n_samples=50000)
    print(f"  Digital-only training: {tc['E_digital_training_kWh']:.4f} kWh")
    print(f"  Photonic hybrid:       {tc['E_photonic_hybrid_kWh']:.4f} kWh")
    print(f"  Speedup: {tc['training_speedup']:.1f}x")

    print("\n=== SymPy 5 ===")
    for k, eq in photonic_ai_sympy_5().items():
        print(f"  {k}: {eq}")
