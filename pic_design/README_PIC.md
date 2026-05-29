# TD-GS Phase Retrieval — PIC Design

Silicon photonics chip implementing the two-arm dispersive delay lines
from the software pipeline in hardware.

## Architecture

```
fiber_in ──► GC_in ──► Y-splitter ──┬──► spiral_D1 (-600 ps²) ──► GC_out1 ──► PD1 → I₁(t)
                                     │
                                     └──► spiral_D2 (-900 ps²) ──► GC_out2 ──► PD2 → I₂(t)

I₁(t), I₂(t)  →  optical_dashboard/dsp.py TD-GS  →  φ(t)  →  OPA
```

## Directory

```
pic_design/
├── components/
│   ├── dispersive_delay.py     spiral waveguide + chirped Bragg grating arms
│   ├── input_splitter.py       Y-junction and directional coupler
│   └── grating_coupler_io.py   fiber I/O array (127µm pitch)
├── layouts/
│   └── gs_chip.py              top-level chip — routes all components
├── sim/
│   ├── fdtd_dispersion.py      SymPy analytic GVD model, H(ν) plots
│   └── drc_check.py            AIM PDK design rule check
├── gds_output/                 exported .gds files for foundry
├── docs/                       simulation figures
└── requirements.txt
```

## Quickstart

```bash
pip install -r pic_design/requirements.txt

# Simulate dispersion model
cd pic_design
python sim/fdtd_dispersion.py        # → docs/dispersion_model.png

# Render chip layout
python layouts/gs_chip.py            # opens KLayout + writes gds_output/gs_chip_v1.gds

# Run DRC
python sim/drc_check.py
```

## Physical Parameters

| Parameter | Value |
|---|---|
| Platform | AIM Photonics 300mm SOI |
| Wavelength | 1550 nm |
| Waveguide | 220nm Si × 450nm width |
| β₂ | ~−1.0 ps²/mm |
| Arm 1 spiral | D₁ = −600 ps² → 600 mm |
| Arm 2 spiral | D₂ = −900 ps² → 900 mm |
| Die target | 3 mm × 1.5 mm |
| Fiber pitch | 127 µm (standard array) |

## Foundry Submission

```
1. python layouts/gs_chip.py          # generate gds_output/gs_chip_v1.gds
2. python sim/drc_check.py            # must exit 0
3. Email gds_output/gs_chip_v1.gds   →  aim-photonics.com/mpw  (student MPW, free)
```
