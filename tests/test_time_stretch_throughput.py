"""Test dgs.time_stretch_throughput: FFT/GS FLOP & byte counts, the roofline (ridge, attainable,
memory- vs compute-bound), frame time / max rate / headroom, and the GPU-over-CPU speedup."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dgs import time_stretch_throughput as tp

# 1. FFT FLOPs: 5 N log2 N, power-of-two only, monotonic
assert math.isclose(tp.fft_flops(1024), 5 * 1024 * 10)
assert math.isclose(tp.fft_flops(2), 5 * 2 * 1)
assert tp.fft_flops(2048) > tp.fft_flops(1024)
for bad in (3, 1, 0, -4):
    try:
        tp.fft_flops(bad); assert False
    except ValueError:
        pass

# 2. GS frame FLOPs: 1 seed FFT + n_iter*(2 FFT + elementwise); linear in n_iter
N = 1024
f1 = tp.dispersion_gs_flops(N, 1)
f2 = tp.dispersion_gs_flops(N, 2)
per_iter = 2 * tp.fft_flops(N) + 20 * N
assert math.isclose(f1, tp.fft_flops(N) + per_iter)
assert math.isclose(f2 - f1, per_iter)                     # each extra iteration adds one unit
assert math.isclose(tp.dispersion_gs_flops(N, 10) - tp.dispersion_gs_flops(N, 5), 5 * per_iter)

# 3. bytes moved: (1 + 2 n_iter) FFTs, passes * N * bytes each
assert math.isclose(tp.dispersion_gs_bytes(N, 3), (1 + 2*3) * 2 * N * 8)
assert tp.dispersion_gs_bytes(N, 10) > tp.dispersion_gs_bytes(N, 5)

# 4. arithmetic intensity
assert math.isclose(tp.arithmetic_intensity(1000.0, 250.0), 4.0)

# 5. roofline: ridge = peak/bw; attainable = min(peak, q*bw)
peak, bw = 12000.0, 900.0
ridge = tp.ridge_point(peak, bw)
assert math.isclose(ridge, peak / bw)
# left of ridge -> bandwidth limited (= q*bw), memory-bound
q_lo = ridge / 2
assert math.isclose(tp.attainable_gflops(q_lo, peak, bw), q_lo * bw)
assert tp.is_memory_bound(q_lo, peak, bw)
# right of ridge -> compute limited (= peak), not memory-bound
q_hi = ridge * 2
assert math.isclose(tp.attainable_gflops(q_hi, peak, bw), peak)
assert not tp.is_memory_bound(q_hi, peak, bw)
# exactly at the ridge the two rooflines meet
assert math.isclose(tp.attainable_gflops(ridge, peak, bw), peak)

# 6. frame time / max rate / required throughput are mutually consistent
flops = tp.dispersion_gs_flops(2**16, 50)
sustained = 3000.0                                         # 3 TFLOP/s sustained
ft = tp.frame_time_s(flops, sustained)
assert math.isclose(ft, flops / (sustained * 1e9))
assert math.isclose(tp.max_frame_rate_hz(flops, sustained), 1.0 / ft)
assert math.isclose(tp.required_throughput_gflops(flops, 1000.0), flops * 1000.0 / 1e9)
# required throughput to hit the max rate equals the sustained throughput
maxr = tp.max_frame_rate_hz(flops, sustained)
assert math.isclose(tp.required_throughput_gflops(flops, maxr), sustained, rel_tol=1e-9)

# 7. real-time headroom: >=1 feasible, <1 falls behind; scales inversely with frame rate
hr = tp.realtime_headroom(flops, 1000.0, sustained)
assert math.isclose(hr, maxr / 1000.0)
assert tp.realtime_headroom(flops, maxr, sustained) == 1.0 or \
       math.isclose(tp.realtime_headroom(flops, maxr, sustained), 1.0, rel_tol=1e-9)
# doubling the frame rate halves the headroom
assert math.isclose(tp.realtime_headroom(flops, 2000.0, sustained), hr / 2)
# more iterations -> more work -> lower max frame rate
assert tp.max_frame_rate_hz(tp.dispersion_gs_flops(2**16, 100), sustained) < \
       tp.max_frame_rate_hz(tp.dispersion_gs_flops(2**16, 50), sustained)

# 8. speedup is the throughput ratio
assert math.isclose(tp.speedup(500.0, 12000.0), 24.0)
# for a memory-bound kernel, GPU/CPU speedup tracks the bandwidth ratio
q = tp.arithmetic_intensity(tp.dispersion_gs_flops(2**20, 50), tp.dispersion_gs_bytes(2**20, 50))
cpu = tp.attainable_gflops(q, 500.0, 50.0)                 # both memory-bound at this q
gpu = tp.attainable_gflops(q, 12000.0, 900.0)
assert tp.is_memory_bound(q, 500.0, 50.0) and tp.is_memory_bound(q, 12000.0, 900.0)
assert math.isclose(tp.speedup(cpu, gpu), 900.0 / 50.0, rel_tol=1e-9)   # = bandwidth ratio

# 9. kwarg bounds
for bad in (lambda: tp.dispersion_gs_flops(1024, 0),
            lambda: tp.arithmetic_intensity(1.0, 0),
            lambda: tp.ridge_point(0, 100),
            lambda: tp.frame_time_s(0, 100),
            lambda: tp.required_throughput_gflops(1e9, 0),
            lambda: tp.speedup(0, 100)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_time_stretch_throughput: all checks passed")
