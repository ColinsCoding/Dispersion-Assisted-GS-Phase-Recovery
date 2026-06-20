# FFT8 — C model + Verilog RTL (a hardware-engineering project)

One 8-point radix-2 DIT FFT, implemented two ways that **must produce the same
result** — the standard DSP-hardware workflow (and a clean portfolio piece):

| layer | file | role |
|-------|------|------|
| **C golden model** | [`fft8.c`](fft8.c) | floating-point reference; what you check the hardware against |
| **Verilog RTL** | [`fft8.v`](fft8.v) | the fixed-point hardware implementation (synthesizable) |
| **testbench** | [`fft8_tb.v`](fft8_tb.v) | drives the RTL with test vectors |
| **build** | [`Makefile`](Makefile) | `make c`, `make verilog`, `make all` |

## Run it

```bash
make c          # gcc: build + run the C model
make verilog    # iverilog: compile + simulate the RTL
```

## Verification (both must agree)

| input | expected output |
|-------|-----------------|
| impulse `[1,0,0,0,0,0,0,0]` | flat spectrum — every bin = 1 |
| one-cycle cosine `cos(2πk/8)` | spikes at k=1 and k=7 (height N/2 = 4) |

Both the C model and the Verilog simulation produce exactly this — the C
floating-point reference confirms the fixed-point RTL is correct. Tracked in
git alongside the rest of the dispersion-GS phase-recovery project (the FFT is
the spectral engine the dispersion operator `H(f)=exp(iπDf²)` runs on).
