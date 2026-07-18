"""Generate notebooks/vevid_lut_embedded.ipynb."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from nbkit import md, co, setup_cell, write

cells = [
md(r"""# VEViD as a lookup table: tone mapping for solid-state / embedded hardware

VEViD-lite enhances a dark image with a **pointwise** phase map,
$$V_{\text{out}}=\frac{\arctan\!\big(g\,(V+b)\big)}{\arctan\!\big(g\,(1+b)\big)}.$$
Because it acts on each pixel independently and real images are quantized (8-bit = 256 levels), the whole
transform collapses into a **256-entry lookup table (LUT)**: precompute the arctan once, offline, then at
runtime each pixel is a single **address -> data** read. No arctan, no floating point, no division on the
hot path -- exactly how tone/gamma correction is implemented in **solid-state** hardware: a ROM or block-RAM
LUT in every display, camera ISP, and FPGA pipeline.

This notebook builds the 8-bit VEViD LUT, shows it is **bit-exact** to the direct float computation, times
the speed-up, and **generates the C** (a `const uint8_t table[256]` plus an apply function) ready to drop
into an embedded target. The physics (the arctan phase readout) is paid for once; the silicon just indexes.

Self-contained: NumPy, Matplotlib (integer-only at runtime; no SciPy/GPU)."""),
setup_cell(),

md(r"""## Build the 8-bit VEViD LUT

Evaluate the VEViD-lite map at the 256 possible input bytes and quantize the output to `uint8`. The table
is the entire enhancement -- a monotonic curve that lifts shadows and rolls off highlights."""),
co("""def vevid_lite(value, gain=8.0, bias=0.1):
    \"\"\"Pointwise VEViD-lite tone map on a normalized [0,1] brightness channel.\"\"\"
    value = np.clip(np.asarray(value, dtype=float), 0.0, 1.0)
    return np.arctan(gain*(value + bias))/np.arctan(gain*(1.0 + bias))

GAIN, BIAS = 8.0, 0.1
levels = np.arange(256)
vevid_lut = np.round(vevid_lite(levels/255.0, GAIN, BIAS)*255).astype(np.uint8)
print("LUT[0], LUT[16], LUT[128], LUT[255] =", [int(vevid_lut[i]) for i in (0, 16, 128, 255)])
print("monotonic non-decreasing:", bool(np.all(np.diff(vevid_lut) >= 0)))
assert np.all(np.diff(vevid_lut) >= 0)                       # structure-preserving"""),

md(r"""## The LUT is bit-exact and shadow-lifting

Applying the LUT to a `uint8` image is identical to computing the float map and quantizing -- zero
mismatches -- while differing from the exact continuous curve by at most half a code (quantization). And it
does the job: dark inputs are boosted far more than bright ones."""),
co("""rng = np.random.default_rng(0)
img = np.clip((0.06 + 0.10*rng.random((512, 512)))*255, 0, 255).astype(np.uint8)   # dark 8-bit image
enhanced_lut = vevid_lut[img]                                # runtime: one table lookup per pixel
enhanced_direct = np.round(vevid_lite(img/255.0, GAIN, BIAS)*255).astype(np.uint8)  # reference

mismatches = int(np.sum(enhanced_lut != enhanced_direct))
continuous = vevid_lite(levels/255.0, GAIN, BIAS)*255
max_lsb = float(np.max(np.abs(vevid_lut.astype(float) - continuous)))
print(f"LUT vs direct-float-quantized: {mismatches} mismatches (bit-exact)")
print(f"max |LUT - continuous curve| = {max_lsb:.2f} LSB (quantization only)")
print(f"mean brightness: {img.mean():.1f} -> {enhanced_lut.mean():.1f}  (shadows lifted)")
assert mismatches == 0 and max_lsb <= 0.5"""),

md(r"""## Speed: a table read beats a transcendental

The LUT replaces a per-pixel `arctan` (plus divisions and float ops) with an integer index. On a
half-megapixel image that is several times faster in NumPy -- and on an FPGA/MCU it is the difference
between a multi-cycle CORDIC and a single-cycle memory read."""),
co("""import time
def _time(fn, reps=50):
    best = float('inf')
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best
t_lut = _time(lambda: vevid_lut[img])
t_direct = _time(lambda: np.round(vevid_lite(img/255.0, GAIN, BIAS)*255).astype(np.uint8))
print(f"LUT apply:      {t_lut*1e3:.3f} ms")
print(f"direct arctan:  {t_direct*1e3:.3f} ms")
print(f"speed-up:       {t_direct/t_lut:.1f}x")
assert t_direct > t_lut"""),

md(r"""## Generate the C: a ROM table plus an apply function

The finalized LUT is emitted as portable C -- a `const uint8_t` array (goes in flash/ROM) and an apply
function over an image buffer. This is the embedded artifact: the arctan was computed offline in Python;
the target only indexes. Compiles with any C compiler, no math library needed at runtime."""),
co(r'''def generate_lut_c(table, name="vevid_lut"):
    entries = ", ".join(str(int(v)) for v in table)
    lines = "\n".join("    " + entries[k:k+96] for k in range(0, len(entries), 96))
    header = (f"#ifndef {name.upper()}_H\n#define {name.upper()}_H\n#include <stdint.h>\n\n"
              f"extern const uint8_t {name}[256];\n"
              f"void {name}_apply(const uint8_t *src, uint8_t *dst, int n);\n\n"
              f"#endif\n")
    source = (f'#include "{name}.h"\n\n'
              f"/* Auto-generated VEViD tone-map LUT (gain={GAIN}, bias={BIAS}). */\n"
              f"const uint8_t {name}[256] = {{\n{lines}\n}};\n\n"
              f"void {name}_apply(const uint8_t *src, uint8_t *dst, int n) {{\n"
              f"    for (int i = 0; i < n; ++i) dst[i] = {name}[src[i]];\n}}\n")
    return header, source

header, source = generate_lut_c(vevid_lut)
print(header)
print(source[:420], "...")'''),

md(r"""## Plots"""),
co(r"""fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].plot(levels, vevid_lut, color="#4C78A8", lw=2)
ax[0].plot(levels, levels, ":", color="gray", label="identity")
ax[0].set_xlabel("input code"); ax[0].set_ylabel("output code")
ax[0].set_title("VEViD LUT transfer curve"); ax[0].legend(fontsize=8)
ax[1].imshow(img, cmap="gray", vmin=0, vmax=255); ax[1].set_title("dark input (8-bit)"); ax[1].axis("off")
ax[2].imshow(enhanced_lut, cmap="gray", vmin=0, vmax=255)
ax[2].set_title("LUT-enhanced (one read/pixel)"); ax[2].axis("off")
plt.tight_layout(); plt.show()"""),

md(r"""## Solid-state and embedded context

- **Where LUTs live in hardware.** A LUT is a **ROM / block-RAM**: the input code is the address, the
  stored byte is the output. Displays and camera image-signal processors apply gamma and tone curves this
  way; FPGAs use block-RAM LUTs; even the logic cells of an FPGA are literally *lookup tables*. The
  transcendental is designed once and frozen into silicon.
- **Old-electronics analogue.** Before cheap digital memory, the same nonlinear transfer was built from
  analog circuits (diode function generators, log/antilog amplifiers). The LUT is the solid-state,
  digital-memory replacement: exact, temperature-stable, and reprogrammable.
- **Deployment.** Drop `vevid_lut.c/.h` into a microcontroller or FPGA soft-core; enhancement is a
  memory-bandwidth-bound loop with no floating point -- the portable, GPU-free end of this project's
  Python -> C -> embedded flow.

## Summary

- VEViD-lite, being **pointwise**, reduces to a **256-entry LUT**: build it offline, index it at runtime.
- The LUT is **bit-exact** to the direct float computation (0 mismatches, <=0.5 LSB from the continuous
  curve), monotonic (structure preserved), and several times faster.
- The LUT is emitted as **portable C** (`const uint8_t[256]` + apply loop) for solid-state/embedded targets
  -- ROM/block-RAM tone mapping with no runtime arctan.

Subject-verb-object: the arctan is computed once; the table stores it; the address selects the output; the
silicon just reads."""),
]

write("vevid_lut", "embedded", cells)
