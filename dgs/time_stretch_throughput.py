"""Compute-feasibility model for a CUDA-accelerated photonic time-stretch pipeline.

The photonics gives you a fire-hose of data: a time-stretch ADC captures frames at a high rate,
and each frame must be phase-retrieved by the dispersion-GS loop (dgs.dispersion_gs_*). That loop
is FFT-bound -- every Gerchberg-Saxton iteration is two FFTs plus some elementwise work -- which is
exactly why it goes on a GPU (dgs.cuda_time_stretch_runner runs the real cuFFT kernel). This module
answers the feasibility questions BEFORE touching hardware, with first-order arithmetic:

  * How many FLOPs does one frame cost?  An N-point FFT is ~5 N log2 N FLOPs; a GS run is
    n_iter x (2 FFTs + O(N)).
  * Is it compute-bound or memory-bound?  The ROOFLINE: attainable = min(peak_compute,
    arithmetic_intensity x peak_bandwidth). FFT pipelines have low arithmetic intensity
    (~log2 N FLOP/byte), so on a fast GPU they usually sit LEFT of the ridge -- limited by
    memory bandwidth, not raw FLOPs. Knowing that tells you a faster-FLOP card won't help;
    a wider-bandwidth one will.
  * Does the GPU keep up in REAL TIME?  frame_time = FLOPs / sustained_throughput must beat the
    ADC frame period 1/frame_rate. The headroom (>= 1 is feasible) is the number this whole
    calculation exists to produce.
  * What speedup does the GPU buy over the CPU?  Just the ratio of sustained throughputs.

This is a MODEL, not a benchmark -- it sizes the problem and shows where the wall is. Pure
arithmetic; NumPy/math; py-3.13.
"""

import math


def fft_flops(n):
    """FLOPs for one N-point complex FFT, the standard ~5 N log2(N) estimate."""
    if n < 2 or (n & (n - 1)) != 0:
        raise ValueError("n must be a power of two >= 2")
    return 5.0 * n * math.log2(n)


def dispersion_gs_flops(n, n_iter, elementwise_per_sample=20):
    """FLOPs to phase-retrieve one N-sample frame with n_iter Gerchberg-Saxton iterations:
    each iteration is 2 FFTs (forward + inverse) plus O(N) elementwise work (magnitude
    constraint, phase apply). One extra FFT seeds the loop."""
    if n_iter < 1:
        raise ValueError("n_iter must be >= 1")
    per_iter = 2 * fft_flops(n) + elementwise_per_sample * n
    return fft_flops(n) + n_iter * per_iter


def dispersion_gs_bytes(n, n_iter, bytes_per_complex=8, passes_per_fft=2):
    """Bytes moved per frame: each FFT streams the array through memory a few times (read+write).
    Dominates the roofline because FFTs are bandwidth-hungry. Complex64 = 8 bytes/sample."""
    if n_iter < 1:
        raise ValueError("n_iter must be >= 1")
    ffts = 1 + 2 * n_iter
    return ffts * passes_per_fft * n * bytes_per_complex


def arithmetic_intensity(flops, byts):
    """Arithmetic intensity q = FLOPs / bytes -- the x-axis of the roofline plot."""
    if byts <= 0:
        raise ValueError("bytes must be > 0")
    return flops / byts


def ridge_point(peak_gflops, peak_bandwidth_gbs):
    """The roofline ridge: arithmetic intensity q* = peak_compute / peak_bandwidth [FLOP/byte]
    where a kernel switches from memory-bound (left) to compute-bound (right)."""
    if peak_gflops <= 0 or peak_bandwidth_gbs <= 0:
        raise ValueError("peak_gflops and peak_bandwidth_gbs must be > 0")
    return peak_gflops / peak_bandwidth_gbs


def attainable_gflops(q, peak_gflops, peak_bandwidth_gbs):
    """Roofline: the best achievable throughput at arithmetic intensity q is
    min(peak_compute, q * peak_bandwidth)."""
    if q < 0:
        raise ValueError("q must be >= 0")
    return min(peak_gflops, q * peak_bandwidth_gbs)


def is_memory_bound(q, peak_gflops, peak_bandwidth_gbs):
    """True if arithmetic intensity q sits left of the ridge -- bandwidth-limited, so more
    FLOP/s won't help but more GB/s will."""
    return q < ridge_point(peak_gflops, peak_bandwidth_gbs)


def frame_time_s(flops_per_frame, sustained_gflops):
    """Wall-clock time to process one frame at a sustained throughput [s]."""
    if flops_per_frame <= 0 or sustained_gflops <= 0:
        raise ValueError("flops_per_frame and sustained_gflops must be > 0")
    return flops_per_frame / (sustained_gflops * 1e9)


def required_throughput_gflops(flops_per_frame, frame_rate_hz):
    """Sustained throughput needed to process frame_rate_hz frames per second [GFLOP/s]."""
    if flops_per_frame <= 0 or frame_rate_hz <= 0:
        raise ValueError("flops_per_frame and frame_rate_hz must be > 0")
    return flops_per_frame * frame_rate_hz / 1e9


def max_frame_rate_hz(flops_per_frame, sustained_gflops):
    """Highest frame rate a given throughput can keep up with [Hz]."""
    return 1.0 / frame_time_s(flops_per_frame, sustained_gflops)


def realtime_headroom(flops_per_frame, frame_rate_hz, sustained_gflops):
    """Headroom = max_frame_rate / required_frame_rate. >= 1 means real-time is feasible with
    room to spare; < 1 means you fall behind (batch, decimate, or get a faster device)."""
    return max_frame_rate_hz(flops_per_frame, sustained_gflops) / frame_rate_hz


def speedup(cpu_sustained_gflops, gpu_sustained_gflops):
    """GPU-over-CPU speedup from sustained throughputs (same FLOP count cancels)."""
    if cpu_sustained_gflops <= 0 or gpu_sustained_gflops <= 0:
        raise ValueError("throughputs must be > 0")
    return gpu_sustained_gflops / cpu_sustained_gflops


if __name__ == "__main__":
    N, n_iter = 2 ** 20, 50            # 1M-sample frames, 50 GS iterations
    frame_rate = 1e4                   # 10 kHz frames from the time-stretch ADC
    flops = dispersion_gs_flops(N, n_iter)
    byts = dispersion_gs_bytes(N, n_iter)
    q = arithmetic_intensity(flops, byts)

    print(f"one FFT ({N} pts)      = {fft_flops(N)/1e9:.3f} GFLOP")
    print(f"GS frame ({n_iter} iters) = {flops/1e9:.2f} GFLOP, {byts/1e6:.0f} MB moved, "
          f"arithmetic intensity {q:.2f} FLOP/byte")
    print(f"required throughput @ {frame_rate/1e3:.0f} kHz = "
          f"{required_throughput_gflops(flops, frame_rate)/1e3:.2f} TFLOP/s\n")

    # a CPU vs a GPU (peak compute / peak bandwidth / assume ~60% sustained)
    devices = {"CPU (AVX)":   (500.0,   50.0),     # 0.5 TFLOP/s, 50 GB/s
               "GPU (RTX)":   (12000.0, 900.0)}    # 12 TFLOP/s, 900 GB/s
    for name, (peak, bw) in devices.items():
        ridge = ridge_point(peak, bw)
        att = attainable_gflops(q, peak, bw)
        sustained = 0.6 * att
        hr = realtime_headroom(flops, frame_rate, sustained)
        bound = "memory-bound" if is_memory_bound(q, peak, bw) else "compute-bound"
        print(f"{name:11s}: ridge q*={ridge:5.1f}  -> {bound:12s}  "
              f"attainable {att/1e3:5.2f} TFLOP/s  frame {frame_time_s(flops,sustained)*1e3:6.2f} ms"
              f"  real-time headroom x{hr:.2f}")
    print(f"\nGPU/CPU sustained speedup ~ x{speedup(0.6*attainable_gflops(q,500,50), 0.6*attainable_gflops(q,12000,900)):.1f}"
          f"  (both memory-bound here -> set by the {900/50:.0f}x bandwidth ratio, not FLOPs)")
