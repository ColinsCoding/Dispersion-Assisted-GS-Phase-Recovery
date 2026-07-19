# VHDL digital logic teaching set

Structural VHDL mirroring `dgs/digital_logic.py` bit for bit:

| Python (`dgs/digital_logic.py`) | VHDL | Idea |
|---|---|---|
| `half_adder(a, b)` | `half_adder.vhd` | combinational gates as continuous assignment |
| `full_adder(a, b, cin)` (built from 2 half adders) | `full_adder.vhd` | component instantiation = function composition |
| `ripple_carry_add(a_bits, b_bits)` (`for i in range(n)` loop) | `ripple_carry_adder.vhd` | `generate` statement = unrolled-at-elaboration loop |
| -- | `tb_ripple_carry_adder.vhd` | exhaustive self-checking testbench (256 vectors, 4-bit) |

**Correction (2026-07-04): a VHDL simulator IS installed** --
ModelSim-Intel FPGA Starter Edition 10.5b, at
`C:\intelFPGA_lite\19.1\modelsim_ase\win32aloem\vsim` (on PATH in
PowerShell; not on the Bash tool's PATH in this environment, same
mingw64/gcc situation as `dgs/circuits_polyglot.py` -- run VHDL work from
PowerShell). The testbench has actually been compiled and run:

```
vlib work
vcom half_adder.vhd full_adder.vhd ripple_carry_adder.vhd tb_ripple_carry_adder.vhd
vsim -c -do "run -all; quit" tb_ripple_carry_adder
```

Real output: `PASS: all 256 vectors matched a+b exactly` (0 errors, 0
warnings) -- the literal hardware/software equivalence check this
teaching set is built around, now actually verified rather than only
predicted. `scripts/check_vhdl_vs_python_logic.py` separately confirms
the Python model itself against plain integer addition for the same 256
4-bit pairs.

## Behavioral vs. structural, made concrete: `flash_adc_*.vhd`

A second worked example, a 2-bit (4-level) flash ADC digitizing an analog
voltage (the same role `dgs/allpass_dispersion_analog.py`'s `V1(t)`/`V2(t)`
traces would need before feeding into GS phase recovery):

| File | Style | What it describes |
|---|---|---|
| `flash_adc_behavioral.vhd` | behavioral | the ALGORITHM: "which quarter of v_ref is v_in in" (an `if/elsif` chain) -- pure semantics, no circuit topology implied |
| `comparator.vhd` + `flash_adc_structural.vhd` | structural | the CIRCUIT: three comparator instances (a real flash-ADC's actual architecture) feeding a thermometer-to-binary encoder -- syntax/topology, same semantics |
| `tb_flash_adc.vhd` | self-checking testbench | sweeps `v_in` across 201 points and confirms both descriptions agree EXACTLY |

Run:
```
vcom comparator.vhd flash_adc_behavioral.vhd flash_adc_structural.vhd tb_flash_adc.vhd
vsim -c -do "run -all; quit" tb_flash_adc
```
Real output: `PASS: all 201 v_in sweep points gave IDENTICAL
behavioral/structural codes`. Note: `comparator.vhd`'s `real`-typed ports
are valid for simulation (modeling an analog voltage feeding digital
logic) but not synthesizable to actual silicon as written -- a real
comparator is analog transistor circuitry; this is its behavioral
simulation stand-in, same caveat as any AMS-style analog port in a
digital-only VHDL flow.

Also caught a real ModelSim 10.5b (VHDL-93-era) restriction while
building this: inline expressions (e.g. `v_ref => 0.25 * v_ref`) are
rejected as port-map actuals for `real`-typed ports ("not globally
static") -- fixed by computing thresholds into named signals first. And
`to_bstring` (VHDL-2008) isn't available in this tool -- removed from the
testbench's failure-report string.
