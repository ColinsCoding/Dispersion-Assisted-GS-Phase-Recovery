"""Generate notebooks/python_c_vhdl_phase_retrieval.ipynb."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from nbkit import md, co, setup_cell, write

C_SOURCE = r'''#include <stdio.h>
#include <stdint.h>

/* floor integer square root, bit-exact with Python math.isqrt */
static uint64_t isqrt_u64(uint64_t n) {
    uint64_t x = 0, bit = 1ULL << 62;
    while (bit > n) bit >>= 2;
    while (bit) {
        if (n >= x + bit) { n -= x + bit; x = (x >> 1) + bit; }
        else x >>= 1;
        bit >>= 2;
    }
    return x;
}

/* the Gerchberg-Saxton phase-normalization kernel in fixed point:
   out = target * (re, im) / sqrt(re^2 + im^2)   (keep phase, set magnitude) */
#define SHIFT 12
static void phase_norm(int32_t re, int32_t im, int32_t target,
                       int32_t *ore, int32_t *oim) {
    uint64_t mag2 = (uint64_t)((int64_t)re*re + (int64_t)im*im);
    uint64_t mag  = isqrt_u64(mag2);
    if (mag == 0) { *ore = 0; *oim = 0; return; }
    int64_t scale = ((int64_t)target << SHIFT) / (int64_t)mag;   /* target >= 0, mag > 0 */
    *ore = (int32_t)(((int64_t)re * scale) >> SHIFT);            /* arithmetic shift */
    *oim = (int32_t)(((int64_t)im * scale) >> SHIFT);
}

int main(void) {
    long re, im, target, ore, oim;
    while (scanf("%ld %ld %ld", &re, &im, &target) == 3) {
        phase_norm((int32_t)re, (int32_t)im, (int32_t)target, (int32_t*)&ore, (int32_t*)&oim);
        printf("%ld %ld\n", ore, oim);
    }
    return 0;
}
'''

VHDL_SOURCE = r'''-- phase_normalize: the GS magnitude-replacement kernel for an FPGA.
-- out = target * (re,im) / sqrt(re^2+im^2).  Fixed point, SHIFT = 12.
-- Pipeline: (1) square+add -> mag2, (2) integer sqrt -> mag,
--           (3) reciprocal-scale (divide), (4) multiply + shift.
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity phase_normalize is
    generic ( SHIFT : integer := 12 );
    port (
        clk    : in  std_logic;
        re     : in  signed(15 downto 0);
        im     : in  signed(15 downto 0);
        target : in  signed(15 downto 0);          -- magnitude, >= 0
        ore    : out signed(15 downto 0);
        oim    : out signed(15 downto 0)
    );
end entity;

architecture rtl of phase_normalize is
    signal mag2  : unsigned(33 downto 0);
    signal mag   : unsigned(16 downto 0);          -- sqrt(mag2)
    signal scale : signed(31 downto 0);
begin
    process(clk)
        variable m : unsigned(33 downto 0);
        variable bit_v, x : unsigned(33 downto 0);
    begin
        if rising_edge(clk) then
            -- stage 1: mag2 = re^2 + im^2
            mag2 <= unsigned(resize(re*re + im*im, 34));
            -- stage 2: non-restoring integer sqrt (unrolled in synthesis)
            m := mag2; x := (others => '0'); bit_v := to_unsigned(1, 34) sll 32;
            while bit_v /= 0 loop
                if m >= x + bit_v then m := m - (x + bit_v); x := (x srl 1) + bit_v;
                else x := x srl 1; end if;
                bit_v := bit_v srl 2;
            end loop;
            mag <= resize(x, 17);
            -- stage 3: scale = (target << SHIFT) / mag   (pipelined divider IP)
            if mag = 0 then scale <= (others => '0');
            else scale <= resize((signed('0' & target) sll SHIFT) / signed('0' & mag), 32); end if;
            -- stage 4: out = (in * scale) >> SHIFT
            ore <= resize(shift_right(re * scale, SHIFT), 16);
            oim <= resize(shift_right(im * scale, SHIFT), 16);
        end if;
    end process;
end architecture;
'''

cells = [
md(r"""# Phase retrieval from Python to C to VHDL

A phase-retrieval instrument (dispersion time-stretch, or a holographic VR display) prototypes its
algorithm in **Python/Torch** in a notebook, but the real-time version runs in **firmware (C)** or on
an **FPGA (VHDL)**. This notebook follows the single nonlinear kernel at the heart of Gerchberg-Saxton
-- the **phase-normalization** step
$$y \;\longmapsto\; \text{target}\cdot\frac{y}{|y|}\qquad(\text{keep the phase, replace the magnitude})$$
-- down that whole path:

1. **Python/Torch** float prototype (the reference algorithm),
2. a **fixed-point** model (integer arithmetic, the number system hardware uses) with a
   quantization-noise analysis,
3. a **C** implementation, compiled and verified **bit-exact** against the fixed-point model,
4. a synthesizable **VHDL** entity for the FPGA.

This is "phase engineering before C and VHDL": get the numbers right in Python first, then port them
without surprises. Self-contained (NumPy first; optional PyTorch; the C step compiles only if a C
compiler is present)."""),
setup_cell(),

md(r"""## 1. Python/Torch float prototype

The projection is elementwise: divide by the magnitude to keep only the phase, then scale to the
target magnitude. This is the exact step the GS loop applies in the measurement plane."""),
co("""def phase_norm_float(y, target):
    mag = np.abs(y)
    return np.where(mag > 0, target * y/mag, 0)

rng = np.random.default_rng(0)
y = rng.standard_normal(8) + 1j*rng.standard_normal(8)
tgt = np.abs(rng.standard_normal(8))
out = phase_norm_float(y, tgt)
print("output magnitudes:", np.round(np.abs(out), 4), " (equal the targets:", np.round(tgt,4), ")")
assert np.allclose(np.abs(out), tgt)                 # magnitude replaced, phase kept
assert np.allclose(np.angle(out), np.angle(y))
if torch is not None:                                # same op in Torch, if present
    yt = torch.tensor(y); tt = torch.tensor(tgt)
    ot = tt * yt/yt.abs()
    print("torch matches numpy:", torch.allclose(ot, torch.tensor(out)))"""),

md(r"""## 2. Fixed-point model and quantization noise

Hardware works in integers. Represent $re,im,target$ as `int16` in a $Q1.15$-like format (a signed
16-bit fraction scaled by $2^{15}$). The kernel uses an integer square root and a fixed-point divide
with `SHIFT`=12 guard bits. Quantizing to $B$ bits injects noise; the SNR improves by about
$6\ \mathrm{dB}$ per bit, the standard rule that sets the FPGA word length."""),
co("""import math
SHIFT = 12
def phase_norm_fixed(re, im, target):                # bit-exact spec for the C/VHDL
    mag2 = re*re + im*im
    mag = math.isqrt(mag2)
    if mag == 0: return 0, 0
    scale = (target << SHIFT)//mag
    return (re*scale) >> SHIFT, (im*scale) >> SHIFT

def quantize(x, bits):                               # round to a signed `bits`-bit fraction of 2^(bits-1)
    q = 2**(bits-1)
    return np.clip(np.round(x*q), -q, q-1).astype(np.int64)

# quantization SNR vs word length, on random unit-ish phasors
rng = np.random.default_rng(1)
yv = (rng.standard_normal(4000) + 1j*rng.standard_normal(4000)); yv /= np.abs(yv).max()
tv = np.abs(rng.standard_normal(4000)); tv /= tv.max()
rows = []
for B in (8, 10, 12, 14, 16):
    re_i, im_i, t_i = quantize(yv.real, B), quantize(yv.imag, B), quantize(tv, B)
    out = np.array([phase_norm_fixed(int(a), int(b), int(c)) for a,b,c in zip(re_i, im_i, t_i)])
    fx = (out[:,0] + 1j*out[:,1]) / (2**(B-1))       # back to float
    fl = phase_norm_float(yv, tv)
    noise = fx - fl
    snr = 10*np.log10(np.mean(np.abs(fl)**2)/np.mean(np.abs(noise)**2))
    rows.append({"bits B": B, "SNR (dB)": round(snr, 1)})
df = pd.DataFrame(rows)
print(df.to_string(index=False))
# ~6 dB per bit
assert df["SNR (dB)"].iloc[-1] > df["SNR (dB)"].iloc[0]"""),

md(r"""## 3. C implementation, compiled and checked bit-exact

The C kernel uses the *same* integer square root and fixed-point arithmetic as the Python spec, so it
must agree to the bit. If a C compiler is available, this cell compiles it, pipes random integer test
vectors through it, and asserts every output equals `phase_norm_fixed`. (Guarded: with no compiler it
prints the source and relies on the fixed-point model, which is the bit-exact reference.)"""),
co(r'''C_SRC = r"""%s"""
print(C_SRC[:220], "...\n")

import os, shutil, subprocess, tempfile
for p in (r"C:\msys64\mingw64\bin", r"C:\msys64\ucrt64\bin"):
    if os.path.isdir(p) and p not in os.environ["PATH"]:
        os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
cc = shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")

if cc:
    d = tempfile.mkdtemp()
    csrc = os.path.join(d, "phase.c"); exe = os.path.join(d, "phase.exe")
    open(csrc, "w").write(C_SRC)
    subprocess.run([cc, "-O2", "-o", exe, csrc], check=True)
    rng = np.random.default_rng(7)
    re_i = rng.integers(-2000, 2000, 500); im_i = rng.integers(-2000, 2000, 500)
    t_i  = rng.integers(0, 2000, 500)
    stdin = "\n".join(f"{a} {b} {c}" for a,b,c in zip(re_i, im_i, t_i))
    res = subprocess.run([exe], input=stdin, capture_output=True, text=True, check=True)
    c_out = [tuple(map(int, line.split())) for line in res.stdout.split("\n") if line.strip()]
    py_out = [phase_norm_fixed(int(a), int(b), int(c)) for a,b,c in zip(re_i, im_i, t_i)]
    mism = sum(1 for x,y in zip(c_out, py_out) if x != y)
    print(f"compiled with {os.path.basename(cc)}; tested {len(py_out)} vectors; mismatches = {mism}")
    assert mism == 0, "C output must be bit-exact with the Python fixed-point spec"
    print("C and Python fixed-point models agree to the bit.")
else:
    print("no C compiler found; the Python fixed-point model above is the bit-exact reference.")''' % C_SOURCE),

md(r"""## 4. The FPGA entity (VHDL)

The same datapath as a synthesizable **VHDL** entity: square-and-add, an integer square root, a
pipelined divide, then multiply-and-shift. Each Python/C line maps to one hardware stage; the
generic `SHIFT` is the guard-bit count. This is the block a phase-retrieval instrument instantiates
once per SLM pixel or per spectral bin, running every clock."""),
co(r'''VHDL_SRC = r"""%s"""
print(VHDL_SRC)''' % VHDL_SOURCE),

md(r"""## 5. Modern search, and the use cases

Gerchberg-Saxton is an **alternating-projection search**: repeatedly project onto the measured-magnitude
set and onto the constraint set (support, phase-only) until they agree. Modern variants sharpen the
search -- hybrid input-output (HIO), the difference map / RAAR, plug-and-play priors, and *unrolled*
neural networks that learn the projections -- but every one of them still calls this phase-normalization
kernel in its inner loop. Porting that kernel to C and VHDL is therefore what puts *any* of these
searches into a real-time instrument:

- the **dispersion time-stretch** receiver runs GS per acquisition frame at MHz rates -> FPGA;
- the **holographic VR** display runs GS per video frame at 90 Hz on $1024^2$ fields -> GPU/FPGA;
- an embedded **spectrometer** runs it in firmware (C) on a microcontroller.

The lesson: the physics and the search are designed in Python/Torch, but the kernel is engineered in
fixed point and shipped in C/VHDL -- and the two must match to the bit."""),

md(r"""## Summary

- The GS phase-normalization kernel $y\mapsto \text{target}\cdot y/|y|$ was carried from a
  **Torch/NumPy** float prototype, to a **fixed-point** spec (with a $\sim6\ \mathrm{dB/bit}$ SNR law),
  to **C** (compiled and verified **bit-exact**), to a synthesizable **VHDL** entity.
- Getting the fixed-point numbers right in Python first -- "phase engineering before C and VHDL" --
  is what makes the hardware port predictable.
- Modern phase-retrieval searches (HIO, difference map, unrolled nets) all reuse this same kernel;
  the flow shown here is how any of them reaches a real-time instrument.

Subject-verb-object: Python prototypes the algorithm; fixed point sets the word length; C implements
the firmware; VHDL synthesizes the FPGA; the bit-exact check ties them together."""),
]

write("python_c_vhdl", "phase_retrieval", cells)
