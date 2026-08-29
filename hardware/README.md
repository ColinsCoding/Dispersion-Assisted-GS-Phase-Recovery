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

---

# TDM scheduler — round-robin time-division multiplexing (cycles/sec math)

One physical detector, clocked at `f_clk` cycles/sec, shared in time across
several optical channels (e.g. the depth planes `dgs/steam_3d_depth_encoding.py`
proposes multiplexing onto one STEAM detector) — the actual cost of hardware
time-sharing, in cycles-per-second terms, plus the RTL that would really run it.

| layer | file | role |
|-------|------|------|
| **C golden model** | [`tdm_scheduler.c`](tdm_scheduler.c) | computes frame period, per-channel effective rate, and the round-robin channel sequence |
| **Verilog RTL** | [`tdm_scheduler.v`](tdm_scheduler.v) | synthesizable mod-N round-robin channel-select counter |
| **testbench** | [`tdm_scheduler_tb.v`](tdm_scheduler_tb.v) | drives 16 cycles, checks against the C model's exact sequence |

```bash
make tdm_c          # gcc: build + run the C model
make tdm_verilog    # iverilog: compile + simulate the RTL
```

**Verification (both must agree):** 4 channels sharing a 100 MHz clock give
a 40 ns frame period and a 25 MHz effective per-channel rate
(`f_clk / N_channels` — the direct cost of time-sharing); the round-robin
sequence `0,1,2,3,0,1,2,3,...` matches exactly between the C model and the
Verilog simulation.

A real bug was caught and fixed building this: the first version of the
testbench deasserted reset with a *blocking* assignment
(`@(posedge clk); rst_n = 1;`) immediately before entering the sampling
loop — a classic race against the DUT's own posedge-triggered reset check.
Depending on simulator scheduling, the DUT could see the *new* `rst_n` on
the very edge meant to be its last reset cycle, take the increment branch
on an still-undefined `channel_sel`, and produce `X` (unknown) forever
(`X+1` is `X`). Fixed with the standard idiom — a non-blocking
`rst_n <= 1;` plus one more full clock edge before sampling — and now
tracked as a comment in the testbench itself, not just fixed silently.
