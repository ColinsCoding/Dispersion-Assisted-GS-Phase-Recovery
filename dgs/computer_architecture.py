"""Von Neumann architecture, data types, CUDA memory hierarchy, and SLURM.

VON NEUMANN DESIGN CHOICES (1945, still dominant in 2026):
  The stored-program computer: instructions and data share the same memory.
  Four key components:
    1. CPU (ALU + control unit + registers) -- executes instructions
    2. Memory (RAM) -- stores both program and data
    3. I/O -- keyboard, disk, network, ADC/DAC
    4. Bus -- connects all components (address, data, control)

  The VON NEUMANN BOTTLENECK: memory bandwidth limits throughput.
  CPU runs at 3-5 GHz; DRAM bandwidth ~50-200 GB/s; L1 cache ~1-5 TB/s.
  Every deep learning matrix multiply is a race against this bottleneck.

  HARVARD ARCHITECTURE alternative: separate instruction and data memory.
  Used in microcontrollers (PIC, AVR, ARM Cortex-M) and GPU shader units.
  Avoids the bottleneck for fixed programs but limits flexibility.

  KEY DESIGN CHOICES that Von Neumann made (and why they still matter):
    - Binary (base-2): simplest reliable switching (on/off = 1/0)
    - Fixed word size: today 64-bit; enables SIMD (AVX-512 = 8x float64)
    - Cache hierarchy: L1/L2/L3 trade off capacity for speed
    - Pipelining: fetch-decode-execute overlap; ~20 stages in modern CPUs
    - Branch prediction: speculative execution (Spectre/Meltdown vuln)
    - Out-of-order execution: reorder instructions to hide memory latency

CUDA MEMORY HIERARCHY (GPU = massively parallel von Neumann):
  Register file:  ~255 regs/thread, <1 cycle latency, ~1 PB/s bandwidth
  Shared memory:  ~48-228 KB/SM, ~1-5 cycles, ~10 TB/s (user-managed L1)
  L2 cache:       6-80 MB per GPU, ~100 cycles
  Global memory:  8-80 GB HBM, ~700 GB/s (A100) -- the bottleneck
  PCIe bus:       ~32-128 GB/s, HIGH LATENCY (microseconds vs nanoseconds)

  Rule: minimize global memory accesses. Fuse kernels. Use shared memory.
  In the GS algorithm: the FFT kernel must read/write N complex floats from
  global memory; the amplitude-replacement kernel is compute-bound. Fusing
  them reduces global memory traffic by 2x.

DATA TYPES AND SIZES:
  Name         | C type   | Python | Bits | Range / Precision     | Torch
  -------------|----------|--------|------|-----------------------|--------
  bool         | bool     | bool   |  1   | {0, 1}                | torch.bool
  int8         | int8_t   | -      |  8   | -128 to 127           | torch.int8
  uint8        | uint8_t  | bytes  |  8   | 0 to 255              | torch.uint8
  int16        | int16_t  | -      | 16   | -32768 to 32767       | torch.int16
  int32        | int32_t  | int    | 32   | +-2.1e9               | torch.int32
  int64        | int64_t  | int    | 64   | +-9.2e18              | torch.int64
  float16      | __half   | -      | 16   | 3 decimal digits      | torch.float16
  bfloat16     | bf16     | -      | 16   | 2 decimal digits, wide| torch.bfloat16
  float32      | float    | -      | 32   | 7 decimal digits      | torch.float32
  float64      | double   | float  | 64   | 15 decimal digits     | torch.float64
  complex64    | -        | -      | 64   | float32 re + float32 im | torch.complex64
  complex128   | -        | complex| 128  | float64 re + float64 im | torch.complex128

  WHY bfloat16 vs float16:
    float16:   1 sign + 5 exponent + 10 mantissa  -> narrow dynamic range
    bfloat16:  1 sign + 8 exponent +  7 mantissa  -> same range as float32
    bfloat16 is preferred for training; float16 for inference on older GPUs.
    For the GS algorithm: complex64 (2x float32) is the natural choice.
"""
import numpy as np
import struct
import sympy as sp


# ── data type table ───────────────────────────────────────────────────

DTYPE_TABLE = [
    # (name, bits, min_val, max_val, decimal_digits, torch_dtype_str)
    ("bool",      1,  0, 1,                 1,  "torch.bool"),
    ("int8",      8,  -128, 127,            3,  "torch.int8"),
    ("uint8",     8,  0, 255,               3,  "torch.uint8"),
    ("int16",     16, -32768, 32767,        5,  "torch.int16"),
    ("int32",     32, -2.1e9, 2.1e9,        10, "torch.int32"),
    ("int64",     64, -9.2e18, 9.2e18,      19, "torch.int64"),
    ("float16",   16, -65504, 65504,        3,  "torch.float16"),
    ("bfloat16",  16, -3.4e38, 3.4e38,      2,  "torch.bfloat16"),
    ("float32",   32, -3.4e38, 3.4e38,      7,  "torch.float32"),
    ("float64",   64, -1.8e308, 1.8e308,    15, "torch.float64"),
    ("complex64", 64, None, None,           7,  "torch.complex64"),
    ("complex128",128, None, None,          15, "torch.complex128"),
]


def dtype_info(name):
    """Look up a data type by name and return its properties."""
    name = name.lower().strip()
    for row in DTYPE_TABLE:
        if row[0] == name:
            return {
                "name": row[0], "bits": row[1], "bytes": row[1] // 8,
                "min_val": row[2], "max_val": row[3],
                "decimal_digits": row[4], "torch_dtype": row[5],
            }
    avail = [r[0] for r in DTYPE_TABLE]
    raise ValueError(f"Unknown dtype '{name}'. Options: {avail}")


def dtype_memory_for_array(dtype_name, shape):
    """Compute memory footprint of a numpy/torch array.

    Parameters
    ----------
    dtype_name : str  -- e.g., 'float32', 'complex64'
    shape : tuple     -- array shape

    Returns
    -------
    dict with n_elements, bytes, KB, MB, GB
    """
    info = dtype_info(dtype_name)
    n = int(np.prod(shape))
    total_bytes = n * info["bytes"]
    return {
        "dtype": dtype_name, "shape": shape,
        "n_elements": n,
        "bytes_per_element": info["bytes"],
        "total_bytes": total_bytes,
        "total_KB": total_bytes / 1024,
        "total_MB": total_bytes / 1024**2,
        "total_GB": total_bytes / 1024**3,
    }


def float32_anatomy(value):
    """Decompose a float32 value into sign, exponent, mantissa bits.

    IEEE 754 float32: 1 sign bit + 8 exponent bits + 23 mantissa bits.
    Stored value = (-1)^S * 2^(E-127) * (1 + M/2^23)
    """
    packed = struct.pack('>f', np.float32(value))
    bits = int.from_bytes(packed, 'big')
    sign = (bits >> 31) & 1
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    reconstructed = ((-1)**sign) * (2**(exponent - 127)) * (1 + mantissa / (2**23))
    return {
        "value": value,
        "sign_bit": sign,
        "exponent_bits": exponent,
        "exponent_decoded": exponent - 127,
        "mantissa_bits": mantissa,
        "mantissa_fraction": mantissa / (2**23),
        "reconstructed": reconstructed,
        "bits_hex": f"0x{bits:08X}",
    }


# ── von Neumann architecture model ───────────────────────────────────

MEMORY_HIERARCHY = {
    "L1_cache":    {"size_KB": 64,       "latency_cycles": 4,    "bandwidth_GBs": 5000},
    "L2_cache":    {"size_KB": 512,      "latency_cycles": 12,   "bandwidth_GBs": 2000},
    "L3_cache":    {"size_MB": 32,       "latency_cycles": 40,   "bandwidth_GBs": 500},
    "DRAM":        {"size_GB": 64,       "latency_cycles": 200,  "bandwidth_GBs": 50},
    "GPU_SHMEM":   {"size_KB": 48,       "latency_cycles": 2,    "bandwidth_GBs": 10000},
    "GPU_HBM":     {"size_GB": 80,       "latency_cycles": 200,  "bandwidth_GBs": 700},
    "NVLink":      {"size_GB": 0,        "latency_cycles": 1000, "bandwidth_GBs": 600},
    "PCIe5":       {"size_GB": 0,        "latency_cycles": 50000,"bandwidth_GBs": 64},
}


def memory_bandwidth_roofline(flops_per_byte, peak_flops_Tflops, peak_BW_GBs):
    """Roofline model: is this operation compute-bound or memory-bound?

    Operational intensity I = FLOPs / bytes accessed.
    Ridge point: I_ridge = peak_FLOPS / peak_bandwidth.

    If I > I_ridge: compute-bound (buy a faster GPU).
    If I < I_ridge: memory-bound (reduce memory traffic).

    For GS algorithm:
      FFT of N complex64 values: ~5*N*log2(N) FLOPs, reads 8*N bytes.
      I = 5*log2(N)/8 ≈ 0.6*log2(N) FLOPs/byte (memory-bound for small N).
    """
    ridge_point = peak_flops_Tflops * 1e12 / (peak_BW_GBs * 1e9)
    if flops_per_byte < ridge_point:
        bound = "memory-bound"
        bottleneck = "reduce data movement: kernel fusion, shared memory"
    else:
        bound = "compute-bound"
        bottleneck = "reduce arithmetic: approximations, mixed precision"
    return {
        "flops_per_byte": flops_per_byte,
        "ridge_point": ridge_point,
        "bound": bound,
        "bottleneck": bottleneck,
        "attainable_Tflops": min(
            flops_per_byte * peak_BW_GBs * 1e9 / 1e12,
            peak_flops_Tflops
        ),
    }


def gs_algorithm_roofline(N_samples=65536):
    """Roofline analysis of the GS phase retrieval FFT on an A100 GPU.

    GS FFT kernel: N complex64 samples, FFT + amplitude replacement.
    FLOPs: 5*N*log2(N) (FFT) + N (amplitude replacement) ~ 5*N*log2(N)
    Bytes: 2*8*N (read input + write output; complex64 = 8 bytes)
    """
    fft_flops = 5 * N_samples * np.log2(max(N_samples, 2))
    bytes_accessed = 2 * 8 * N_samples   # complex64 = 8 bytes
    I = fft_flops / bytes_accessed
    return {
        "N_samples": N_samples,
        "fft_flops": fft_flops,
        "bytes_accessed": bytes_accessed,
        "operational_intensity": I,
        "A100_analysis": memory_bandwidth_roofline(I, 312.0, 2000.0),
    }


# ── SLURM job submission ──────────────────────────────────────────────

def slurm_job_script(job_name, n_gpus=1, n_cpus=8, mem_GB=32,
                     time_hours=4, partition="gpu", python_cmd="python train.py"):
    """Generate a SLURM batch script for GPU training / GS simulation.

    SLURM (Simple Linux Utility for Resource Management) is the scheduler
    on most HPC clusters (UCLA Hoffman2, SDSC Expanse, NERSC Perlmutter).

    Key SBATCH directives:
      --gres=gpu:N     -- request N GPUs
      --cpus-per-task  -- CPU cores (use N_gpus * 4-8 for DataLoader workers)
      --mem            -- RAM in GB
      --time           -- wall-clock limit HH:MM:SS
      --partition      -- queue name (gpu, cpu, high-priority)
    """
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}_%j.out
#SBATCH --error=logs/{job_name}_%j.err
#SBATCH --gres=gpu:{n_gpus}
#SBATCH --cpus-per-task={n_cpus}
#SBATCH --mem={mem_GB}G
#SBATCH --time={time_hours:02d}:00:00
#SBATCH --partition={partition}
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=colincas37@gmail.com

# Load modules (cluster-specific)
module load cuda/12.2
module load python/3.12

# Activate virtual environment
source venv_torch/bin/activate

# Run
{python_cmd}

echo "Job $SLURM_JOB_ID done"
"""
    return script


def slurm_array_job(n_jobs, base_cmd="python gs_sweep.py --D"):
    """SLURM array job for parameter sweep (e.g., sweep dispersion D).

    Array job submits N_jobs at once; each gets SLURM_ARRAY_TASK_ID.
    """
    script = f"""#!/bin/bash
#SBATCH --job-name=gs_sweep
#SBATCH --array=0-{n_jobs - 1}
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu

D_values=({' '.join(str(-i * 500) for i in range(1, n_jobs + 1))})
D=${{D_values[$SLURM_ARRAY_TASK_ID]}}

{base_cmd} $D --output results/D_${{D}}.pkl
"""
    return script


# ── CUDA memory analysis ──────────────────────────────────────────────

def cuda_memory_plan(model_params_M, batch_size, seq_len, hidden_dim,
                     dtype="float32"):
    """Estimate GPU memory usage for a transformer/GS neural network.

    Memory = parameters + gradients + optimizer states + activations + buffers.
    Adam optimizer: 2x extra (first moment m + second moment v).
    Mixed precision training: parameter in float16, optimizer state in float32.
    """
    info = dtype_info(dtype)
    bytes_per = info["bytes"]

    # Parameter memory
    param_bytes = model_params_M * 1e6 * bytes_per

    # Gradient memory (same size as params)
    grad_bytes = param_bytes

    # Adam optimizer states (float32 even in mixed-precision)
    optim_bytes = model_params_M * 1e6 * 4 * 2   # m and v in float32

    # Activation memory (forward pass; depends on architecture)
    # Transformer: O(batch * seq * hidden) per layer
    activation_bytes = batch_size * seq_len * hidden_dim * bytes_per * 12  # rough

    total_bytes = param_bytes + grad_bytes + optim_bytes + activation_bytes

    return {
        "model_params_M": model_params_M,
        "param_GB": param_bytes / 1e9,
        "grad_GB": grad_bytes / 1e9,
        "optim_GB": optim_bytes / 1e9,
        "activation_GB": activation_bytes / 1e9,
        "total_GB": total_bytes / 1e9,
        "fits_on_A100_80GB": total_bytes / 1e9 < 75.0,
        "fits_on_RTX4090_24GB": total_bytes / 1e9 < 22.0,
    }


# ── defense research influence model ─────────────────────────────────

def defense_research_influence(
        trl_current, trl_target=6,
        sbir_phase1_P=0.15, sbir_phase2_P=0.35,
        dod_transition_P=0.20, n_submissions=5):
    """Model the probability of reaching TRL-6 (prototype demonstrated in
    relevant environment) through SBIR Phase I -> Phase II -> DoD transition.

    Technology Readiness Levels (TRL):
      1: basic principles observed
      3: analytical/experimental proof of concept (this repo: TRL 3-4)
      4: validated in lab (bench demo of GS phase retrieval)
      6: prototype demonstrated in relevant environment (fiber link)
      8: system complete and qualified
      9: actual system proven in operational environment

    SBIR path: Phase I (TRL 3->5) -> Phase II (TRL 5->7) -> transition (TRL 7->9)
    """
    if not (1 <= trl_current <= 9):
        raise ValueError("trl_current must be in [1,9]")
    if not (1 <= trl_target <= 9):
        raise ValueError("trl_target must be in [1,9]")

    # P(at least one Phase I win in n submissions)
    p_phase1 = 1 - (1 - sbir_phase1_P) ** n_submissions

    # Conditional on winning Phase I, P(Phase II)
    p_phase2 = p_phase1 * sbir_phase2_P

    # Conditional on Phase II, P(transition to program of record)
    p_transition = p_phase2 * dod_transition_P

    trl_jump = trl_target - trl_current
    return {
        "trl_current": trl_current,
        "trl_target": trl_target,
        "trl_jump_needed": trl_jump,
        "P_phase1_win": p_phase1,
        "P_phase2_win": p_phase2,
        "P_dod_transition": p_transition,
        "expected_dollar_value_M": p_transition * 10,  # ~$10M typical program
        "n_submissions_assumed": n_submissions,
    }


# ── SymPy: von Neumann / data type formalism ──────────────────────────

def computer_arch_sympy_5():
    """Five key computer architecture equations in SymPy."""
    I_s, P_mem, P_cpu = sp.symbols('I P_mem P_cpu', positive=True)
    N_s, log2N = sp.symbols('N log2N', positive=True)
    n_bits = sp.Symbol('n', integer=True, positive=True)
    E_s, S_s = sp.symbols('E S', positive=True)

    return {
        "Roofline_attainable_performance":
            sp.Eq(sp.Symbol('P_attainable'),
                  sp.Min(I_s * P_mem, P_cpu)),
        "FFT_operational_intensity":
            sp.Eq(sp.Symbol('I_FFT'),
                  5 * N_s * log2N / (2 * 8 * N_s)),
        "Float32_value":
            sp.Eq(sp.Symbol('x'),
                  sp.Symbol('(-1)**S') * 2**(E_s - 127) * (1 + sp.Symbol('M/2^23'))),
        "Int_range":
            sp.Eq(sp.Symbol('range'),
                  2**n_bits - 1),
        "GPU_memory_bandwidth":
            sp.Eq(sp.Symbol('BW'),
                  sp.Symbol('transactions_per_sec') * sp.Symbol('bytes_per_transaction')),
    }


if __name__ == "__main__":
    print("=== Data type sizes ===")
    for name in ["int8", "float16", "bfloat16", "float32", "float64", "complex64"]:
        info = dtype_info(name)
        print(f"  {name:<12}: {info['bits']:>4} bits  {info['bytes']:>2} bytes  "
              f"~{info['decimal_digits']} decimal digits  ({info['torch_dtype']})")

    print("\n=== float32 anatomy: 3.14159 ===")
    a = float32_anatomy(3.14159)
    print(f"  bits: {a['bits_hex']}")
    print(f"  sign={a['sign_bit']} exponent={a['exponent_bits']} ({a['exponent_decoded']:+d})"
          f" mantissa=0x{a['mantissa_bits']:06X}")
    print(f"  reconstructed: {a['reconstructed']:.5f}")

    print("\n=== Array memory: GS signal N=65536 complex64 ===")
    m = dtype_memory_for_array("complex64", (65536,))
    print(f"  {m['total_KB']:.1f} KB ({m['n_elements']} elements x {m['bytes_per_element']} bytes)")

    print("\n=== GS algorithm roofline on A100 ===")
    r = gs_algorithm_roofline(65536)
    print(f"  Operational intensity: {r['operational_intensity']:.2f} FLOPs/byte")
    print(f"  Bound: {r['A100_analysis']['bound']}")
    print(f"  Bottleneck: {r['A100_analysis']['bottleneck']}")

    print("\n=== SLURM script for GS sweep ===")
    script = slurm_job_script("gs_phase_retrieval", n_gpus=2, time_hours=8,
                              python_cmd="python -m dgs.gs_core --D -5000 --n_iter 50")
    print(script[:500] + "...")

    print("\n=== Defense research transition probability ===")
    r = defense_research_influence(trl_current=3, trl_target=7, n_submissions=5)
    print(f"  P(Phase I win, 5 submissions): {r['P_phase1_win']:.1%}")
    print(f"  P(Phase II win): {r['P_phase2_win']:.1%}")
    print(f"  P(DoD transition): {r['P_dod_transition']:.1%}")
    print(f"  Expected value: ${r['expected_dollar_value_M']:.2f}M")

    print("\n=== SymPy 5 ===")
    for k, eq in computer_arch_sympy_5().items():
        print(f"  {k}: {eq}")
