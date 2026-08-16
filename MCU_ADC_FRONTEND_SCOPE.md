# MCU ADC Front-End — Scope

**Status:** scoping only, nothing built yet. This document exists to make the
build/no-build/build-something-else decision explicit before writing firmware.

## What this is not

This is **not** a replacement for `EQUIPMENT_ACQUISITION_PLAN.md`'s Tier 1 bench.
That plan already specs a USB oscilloscope/digitizer (PicoScope 2000/3000
series or Digilent Analog Discovery 2, ≥100 MSa/s, 2-channel) for capturing the
real I1(t)/I2(t) pair that validates `retrieve_phase` on hardware data. A
general-purpose microcontroller's on-chip ADC cannot get there — common
MCU ADCs top out around 1–5 MSa/s at useful resolution; a 100 MSa/s front end
needs a dedicated high-speed ADC IC plus FPGA capture, a materially different
(and more expensive) project than what's scoped below. **Buy the Tier 1
digitizer for the actual I1/I2 capture; don't wait on this.**

So what is this for? Two honest, separate justifications — pick one (or both)
before starting:

1. **Hardware credential.** A real embedded signal chain — photodiode → TIA →
   anti-alias filter → ADC → digital output — is the physical realization of
   `dgs/photodetector_tia_frontend.py` and the final stage of the time-stretch
   receiver schematic in `dgs/time_stretch_adc.py` ("[Photodetector] →
   [Electronic ADC @ f_s]"). Building it demonstrates you can take a
   Jalali-lab-relevant receiver front end from equation to working hardware,
   even at a modest sample rate. This is the strongest of the two
   justifications for Jalali-lab positioning specifically.
2. **A genuinely useful slow instrument.** Not every measurement in this
   repo's future needs GHz bandwidth. The microplastic-sensing project's
   month 5 ("detector, noise, ADC resolution, sampling") and any DC-to-audio-
   range calibration/monitoring task (e.g. a photodiode power monitor, a slow
   absorption measurement) are realistically within a microcontroller ADC's
   reach (kHz–low-MHz) and would benefit from a real, not simulated,
   front end.

If neither justification matters to you right now, don't build this — it's
optional, unlike the Tier 1 bench.

## Target signal chain

```
[photodiode or function-gen test source]
        v
[TIA: op-amp + Rf || Cf]      <- physical version of dgs/photodetector_tia_frontend.py
        v
[anti-alias low-pass filter]  <- analog RC or active filter, cutoff < fs/2
        v
[MCU on-chip ADC, fs samples/sec]
        v
[firmware: buffer + frame + serial/USB-CDC out]
        v
[host: python reads the stream, writes .npy/.csv]  <- matches REQUESTING_DATA.md's
                                                        accepted format (time, intensity)
        v
[dgs.gs_core / dgs.adc / notebook pipeline]
```

The output format should match what the existing dashboard/notebook pipeline
already accepts (`REQUESTING_DATA.md`: `.npy` 1-D or (N,2), or `.csv` with
`time, intensity` columns, no header) — so a captured trace drops straight
into the same `retrieve_phase` / `ADC.report()` code this repo already has,
no new loader needed.

## Hardware decision (open — needs your input)

No MCU is currently owned per anything in this repo/session, so this is a
recommendation, not a foregone choice:

| Option | ADC | Cost | Fit |
|---|---|---|---|
| **STM32 Nucleo-F401RE (or similar F4)** | 12-bit SAR, up to ~2.4 MSa/s | ~$15–20 | Best match: real DMA-driven ADC, HAL/register-level C, closest to "real embedded" credential. Recommended default. |
| Arduino Uno/Nano (ATmega328) | 10-bit SAR, ~15 kSa/s effective | ~$10–25 | Cheapest/most familiar toolchain, but the slow ADC only fits the "slow instrument" justification, not a credible receiver-front-end demo |
| ESP32 | 12-bit SAR (noisy), ~1–2 MSa/s ideal, worse in practice | ~$8–12 | Adds WiFi/BLE for wireless streaming if that's ever wanted; ADC linearity is known to be poor without calibration — more firmware work to get clean data |

**Recommendation: STM32 Nucleo-F4.** It's the only option with a DMA-driven
ADC fast enough to be a believable "receiver front end" rather than a toy, and
STM32 HAL/register-level C is the more transferable embedded credential.
Confirm before ordering — if you already have one of the others on hand, start
there instead of buying new hardware for this.

## Firmware architecture (matches this repo's existing embedded-C style)

Following the pattern already established in `embedded/fir_lowpass.c` (state
struct + one-sample-at-a-time processing function, no dynamic allocation):

1. **`adc_capture.c`** — ADC + DMA/timer configuration, continuous conversion
   into a double (ping-pong) buffer so capture never stalls waiting on the
   previous buffer to drain.
2. **`frame_protocol.c`** — packs samples into a simple framed serial protocol
   (magic byte, sample count, payload, checksum) over USB-CDC or UART —
   simple enough to parse from a 20-line Python `pyserial` script on the host
   side, no vendor SDK required to read it back.
3. **`fir_lowpass.c`** *(already exists)* — reused as the on-MCU anti-alias /
   smoothing stage before the frame is sent, exactly as its own header comment
   already describes ("this is the shape of code that runs on a
   microcontroller... before the ADC" — currently true only in simulation;
   this project makes it literally true).
4. **`host_capture.py`** (new, host-side, not firmware) — reads the framed
   serial stream, reconstructs the time axis from the known `fs`, writes
   `.npy`/`.csv` in the `REQUESTING_DATA.md` format.

## Milestones (each one a working, demonstrable checkpoint)

1. **Bench test source → ADC → serial → host plot.** Function generator sine
   wave into the Nucleo's ADC pin directly (no TIA yet), stream to a Python
   script, plot it, confirm sample rate and amplitude match the function
   generator. Proves the capture chain end-to-end.
2. **Add the TIA.** Build the physical op-amp transimpedance stage from
   `dgs/photodetector_tia_frontend.py`'s R_f/C_f design, drive it with a real
   photodiode (even a cheap silicon one + an LED as the source), confirm the
   measured rise time matches the ODE solution the module already computes.
3. **Add the anti-alias filter + on-MCU FIR** (`fir_lowpass.c`, now compiled
   for the target instead of `gcc`/desktop), confirm no aliasing artifacts in
   a captured multi-tone test signal.
4. **Round-trip through the existing pipeline.** Feed a captured `.csv` into
   `dgs.adc.ADC.report()` or the dashboard, confirming SQNR/quantization
   numbers from real hardware are in the right ballpark vs. the module's
   theoretical predictions.

Milestone 1 alone is enough to validate the hardware-credential justification;
milestones 2–4 are where it becomes either the "slow instrument" or a genuine
receiver-front-end demo, depending on which justification you're building for.

## Budget

Nucleo board + breadboard TIA components (op-amp, resistors, photodiode,
LED) + USB cable: **under $50** if buying the recommended STM32 option new.
Trivial next to the Tier 1 bench's $1.5–3K, which is the right relative
weight given this is optional.

## Decisions needed before writing any code

1. **Which justification** (hardware credential vs. slow instrument, or both)
   — changes how much the milestone list above matters vs. is optional polish.
2. **MCU choice** — confirm STM32 Nucleo-F4, or say what you already own.
3. **Toolchain** — STM32CubeIDE (GUI, HAL) vs. bare Makefile + arm-none-eabi-gcc
   + CMSIS (closer to `embedded/`'s existing raw-C style, more setup work).
