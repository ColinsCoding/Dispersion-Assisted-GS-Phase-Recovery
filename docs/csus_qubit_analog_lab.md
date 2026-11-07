# Lab protocol: a qubit's Rabi oscillation on a CSUS oscilloscope, with two coupled LC tanks

**Experimenter:** Colin Casey (CSUS)
**Status:** protocol with simulated predictions. **Not yet performed.** Every "measured" cell below is
empty on purpose; the numbers in the "predicted" columns come from `notebooks/qubit_microwave_lab.ipynb`.

## 1. Purpose, in plain words

A superconducting qubit needs a dilution refrigerator, but its basic dynamics don't. Two LC tanks
coupled through the mutual inductance of two coils follow the same 2 × 2 equation (in the
slowly-varying-envelope limit) as a qubit driven by a microwave tone:
- Kick tank 1 and its energy flows into tank 2 and back. That is a Rabi oscillation, at Ω ≈ k ω₀.
- Detune tank 2 and the transfer becomes incomplete and faster, following Ω²/(Ω² + Δ²).
- Sweep the detuning and the two resonance peaks bend away from each other: the avoided crossing
  seen in qubit spectroscopy.

**Limit of the analogy.** It is classical: the voltages are not probability amplitudes, and there is
no superposition of outcomes, no collapse and no entanglement. What it teaches is the equation of
motion and the measurement habits: calibrate the coupling, sweep one knob, fit, and compare.

## 2. Equipment

Generic teaching-lab gear. Use whatever models the CSUS lab has.

| item | requirement |
|---|---|
| digital oscilloscope | 2 channels, ≥ 50 MHz, FFT math, averaging |
| 2 × 10× passive probes | ~10 MΩ ∥ ~12 pF (typical; check the probe body) |
| function generator | square wave, 10 Vpp, 1 kHz |
| LCR meter | inductance and ESR at 100 kHz (or the highest frequency it offers) |
| ruler or calipers, plastic spacers | set the coil gap to ±0.5 mm |

## 3. Parts

| part | value | notes |
|---|---|---|
| 2 coils | 30 turns AWG-24 (0.51 mm) on a 40 mm diameter plastic former, 20 mm winding length | predicted L ≈ 36.2 µH each |
| 2 tank capacitors | 3.3 nF, C0G or polypropylene film | low loss; avoid X7R |
| kick capacitor | 10 pF C0G | couples the generator edge into tank 1 |
| detuning capacitors | 100, 220, 330, 470, 680 pF C0G | added in parallel to one tank |
| breadboard or copper-clad board | | keep leads short |

## 4. Safety

Everything stays at extra-low voltage. The generator supplies at most 10 V, the tanks ring at tens of
millivolts, and nothing stores more than microjoules (`dgs.circuit_rules.safety_check` flags nothing for
this circuit). The real hazards are a hot soldering iron and the bench power cord. Follow the CSUS
lab rules.

## 5. Procedure

1. **Wind the coils.** Wind 30 close-wound turns over 20 mm on each former, and leave 5 cm leads.
2. **Measure each coil alone** on the LCR meter at 100 kHz. Record L and ESR (Table A).
3. **Measure the coupling with the LCR meter.** Mount the coils coaxially with their ends a gap g
   apart. Connect them in series and read L_a. Swap one coil's leads and read L_b. The larger reading
   is L_aid. Then compute M = (L_aid − L_opp)/4 and k = M/√(L₁L₂). Repeat for every gap in Table B.
4. **Build the two tanks.** Each tank is one coil in parallel with one 3.3 nF capacitor, with one end
   of each tank on a common ground.
5. **Wire the kick.** Generator output → 10 pF → top of tank 1, and generator ground → common ground.
   Set a 1 kHz square wave, 0 to 10 V. Each edge kicks tank 1 by about 30 mV.
6. **Probe the tanks.** CH1 goes to tank 1 and CH2 to tank 2, through the 10× probes, with the probe
   grounds on the common ground using the short ground springs if you have them.
7. **Set up the scope.**
   - Vertical: 10 mV/div with the probe factor set to 10×.
   - Timebase: 20 µs/div.
   - Trigger: CH1, rising edge, 5 mV.
   - Averaging: 16 acquisitions, which works because the kick is repetitive.
8. **Watch the beat.** At a 20 mm gap, expect the CH1 envelope to pinch off while CH2 grows, and the
   two to swap roughly every 13.5 µs. Record the swap period from the envelope minima, then save the
   waveform as CSV.
9. **Find the mode splitting.** Turn on the FFT of CH1 (Hann window, 0 to 1 MHz). Read the two peaks
   f₋ < f₊ and compute k = (r² − 1)/(r² + 1) with r = f₊/f₋. Compare with step 3. Repeat steps 8 and 9
   for every gap in Table B.
10. **Sweep the detuning** at the 20 mm gap: the avoided crossing.
    - Add each detuning capacitor in parallel with **tank 2** and record both FFT peaks.
    - Then add the same values to **tank 1** instead; that covers the other side of the crossing.
    - Record everything in Table C.
11. **Measure the ring-down** with CH1 alone, coils far apart (gap > 100 mm). Measure the time for
    the envelope to fall to 1/e. The energy time constant is half of that and should be about L/r.
12. **Save everything.** Save the CSVs and photos, then fill in `MEASURED` in section 6d of the notebook
    and re-run it.

## 6. Data tables

### Table A: each coil alone

| coil | L predicted (µH) | L measured (µH) | ESR measured (Ω) | Q = ωL/ESR |
|---|---|---|---|---|
| 1 | 36.2 | | | |
| 2 | 36.2 | | | |

### Table B: coupling vs gap

| gap (mm) | k predicted | L_aid pred. (µH) | L_opp pred. (µH) | f₋ pred. (kHz) | f₊ pred. (kHz) | swap pred. (µs) | L_aid meas. | L_opp meas. | k (LCR) | f₋ meas. | f₊ meas. | k (FFT) | swap meas. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 0.215 | 88.0 | 56.9 | 416.8 | 518.5 | 5.1 | | | | | | | |
| 10 | 0.150 | 83.4 | 61.6 | 428.3 | 498.4 | 7.2 | | | | | | | |
| 15 | 0.109 | 80.3 | 64.6 | 436.3 | 486.6 | 10.0 | | | | | | | |
| 20 | 0.081 | 78.3 | 66.6 | 441.9 | 479.1 | 13.5 | | | | | | | |
| 30 | 0.047 | 75.9 | 69.1 | 448.9 | 470.7 | 23.0 | | | | | | | |
| 40 | 0.030 | 74.6 | 70.3 | 452.7 | 466.4 | 36.7 | | | | | | | |

The mode-frequency predictions include about 12 pF of probe capacitance on each tank.

### Table C: avoided crossing at a 20 mm gap (capacitor added to tank 2)

| added C (pF) | f₋ pred. (kHz) | f₊ pred. (kHz) | f₋ meas. | f₊ meas. |
|---|---|---|---|---|
| 0 | 441.9 | 479.1 | | |
| 100 | 438.4 | 475.9 | | |
| 220 | 433.5 | 472.9 | | |
| 330 | 428.8 | 470.9 | | |
| 470 | 422.5 | 469.0 | | |
| 680 | 412.9 | 467.2 | | |

Adding the same capacitors to tank 1 mirrors this table, giving the other side of the crossing.

### Table D: ring-down

| quantity | predicted | measured |
|---|---|---|
| amplitude 1/e time (µs) | ~176, uncertain (depends on the real ESR at 460 kHz) | |
| energy time L/r (µs) | ~88 | |

## 7. Analysis

1. Plot k(LCR) and k(FFT) against the gap on top of the predicted curve. They should agree to within
   a few percent. The predictions treat each turn as a thin circular filament, so expect a systematic
   offset of the order of the coil's wire spacing relative to its size.
2. Check that the swap time follows 1/(2 k f₀). This is "Rabi frequency ∝ drive strength".
3. Plot Table C: both measured branches should bend away from each other with a minimum gap of k·f₀,
   about 37 kHz at 20 mm. This is the classical picture of an avoided crossing.
4. Compare the ring-down with ωL/ESR from the LCR meter. A shortfall usually means proximity-effect
   loss in the winding, or dielectric loss in the capacitor if it isn't film or C0G.

## 8. What would make it more qubit-like (later)

- Drive tank 1 continuously near f₀ with a burst instead of kicking it. That is a pulsed Rabi
  experiment: burst length plays pulse length, burst amplitude plays drive strength.
- Ramsey analog: two short bursts separated by a variable wait.
- Read the envelopes with the scope's math (|Hilbert| or a squared, averaged trace) to get the
  "population" curves directly.

## 9. Context

The microwave chain of a *real* qubit (IQ up-conversion, the cold drive line, the readout receiver) is
designed in `notebooks/qubit_microwave_lab.ipynb` as an RF transmitter and receiver problem in the
style of UCLA ECE 279AS, where this repository started. That notebook also maps the work onto the
OUSD(R&E) Critical Technology Areas.
