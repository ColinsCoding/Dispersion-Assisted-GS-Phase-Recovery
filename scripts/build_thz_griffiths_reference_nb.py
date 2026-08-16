"""Build notebooks/thz_griffiths_reference.ipynb -- the typeset version of
the artifact reference sheet, using real LaTeX ($...$) math cells (Jupyter
renders these via MathJax natively -- no CSS hacks needed here, unlike the
HTML artifact). Same five solved problems, same glossary, now as an actual
saved file with a pandas-rendered glossary table.

Engine: dgs/thz_waveguide_dispersion_relation.py, dgs/thz_griffiths_vocab.py.
"""
import json, pathlib

cells = []

def md(src): cells.append({"cell_type": "markdown", "metadata": {}, "source": [s + "\n" for s in src.splitlines()]})
def code(src): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [s + "\n" for s in src.splitlines()]})

md("""# THz Waveguide Dispersion: Worked Reference (typeset)

The five problems solved in `dgs/thz_waveguide_dispersion_relation.py`, with real LaTeX
math instead of Python spelling, plus a glossary at the end. Companion to the HTML
version published as an artifact; this one is the saved, editable, re-runnable copy.
""")

code("""import sys, pathlib
sys.path.insert(0, str(pathlib.Path('..').resolve()))
import numpy as np
import pandas as pd
from dgs import thz_waveguide_dispersion_relation as thz
from dgs.thz_griffiths_vocab import vocab_table

a = 0.3e-3
k_c_a = 1.8412
omega_c = thz.C_LIGHT * (k_c_a / a)
omega_op = 1.5 * omega_c
print('Setup complete.')
""")

md(r"""## Problem 1 & 2 -- no zero-dispersion point, no two-segment compensation

$$\beta_2(\omega) = \frac{-\omega_c^2}{c(\omega^2-\omega_c^2)^{3/2}} \neq 0 \quad \text{for any finite } \omega$$

The numerator never depends on $\omega$ and is never zero, so $\beta_2$ keeps a fixed
(negative) sign everywhere in the propagating band -- no zero-dispersion frequency, and
no two-segment waveguide link (this same mechanism, two different radii) can cancel its
own dispersion to zero.
""")

code("""ok = thz.verify_gvd_sign_is_fixed()
print(f'beta_2 strictly negative for all omega>omega_c>0, any omega_c: {ok}')
""")

md(r"""## Problem 3 -- which mode disperses least?

Driving TE11, TM01, TE21 at one shared carrier $\omega_0 = 1.5\,\omega_c^{max}$:

$$\Delta t \approx |\beta_2(\omega_0,\,\omega_c^{mode})| \cdot L \cdot \Delta\omega$$
""")

code("""ranking = thz.rank_modes_by_dispersion(a)
df_rank = pd.DataFrame(ranking['modes']).T
df_rank = df_rank.loc[ranking['ranked_best_to_worst']]
df_rank
""")

md(r"""## Problem 4 -- does thermal drift matter?

$$a(T) = a_0(1+\alpha\Delta T), \qquad \alpha_{Cu} \approx 17\times10^{-6}\,\text{K}^{-1}$$

Over a 60 K outdoor swing:
""")

code("""thermal = thz.thermal_broadening_shift(a, omega_op)
pd.Series(thermal, name='60K swing, copper wall').to_frame()
""")

md(r"""## Problem 5 -- is the dispersion relation geometry-independent?

$$\omega^2 = c^2(k^2+k_c^2), \qquad k_c^2 \equiv -\frac{\nabla_t^2 f}{f}$$

True for *any* transverse cross-section $f(x,y)$ -- geometry only decides what $k_c$ is,
never the form of the relation. Checked against a real WR-90 rectangular guide below.
""")

code("""ok_general = thz.verify_dispersion_relation_is_geometry_independent()
f_c_wr90 = thz.rectangular_waveguide_cutoff_frequency(1, 0, 22.86e-3, 10.16e-3)
print(f'general proof (any geometry): {ok_general}')
print(f'WR-90 TE10 computed: {f_c_wr90/1e9:.4f} GHz  (published: 6.557 GHz)')
""")

md("""## Glossary
""")

code("""pd.set_option('display.max_colwidth', 100)
vocab_table()
""")

md("""## Arduino projects for the CSUS THz/benchtop build

Grounded in the actual BOM (`dgs/csus_experiment.py`) and Problem 4's thermal-drift
question above -- not generic Arduino ideas:

1. **Thermal drift logger.** A DS18B20 (or thermocouple + MAX31855) taped to the
   waveguide, logged over a heater/cold-spray-induced temperature sweep, gives real
   data to check against `thermal_broadening_shift`'s -0.10%/-0.45% predictions --
   turns Problem 4 from a calculation into a measurement.
2. **MZM bias controller.** The LiNbO3 modulator's bias point drifts thermally and
   needs to sit at quadrature ($V_\\pi/2$) -- an Arduino + DAC (MCP4922) sweeping bias
   voltage while an ADC (ADS1115) reads the photodetector's DC level can find and
   hold quadrature automatically, replacing manual bias-dithering.
3. **RF source frequency sweep + data capture.** Step the RF signal generator (if
   it has a serial/GPIB-to-TTL bridge) and log photodetector output per frequency via
   Arduino's ADC -- builds the actual dispersion curve from bench data instead of
   only the simulated one in this notebook.
4. **Waveguide-length rail (stepper).** For the rectangular/circular waveguide
   comparison (Problem 5), a small stepper-driven rail varying effective path length
   L, logging broadening vs. L, checks the linear-in-L assumption in
   `thz_pulse_broadening` against real hardware.

None of these need a full DAQ card -- an Uno/Nano plus a couple of $5-15 breakout
boards covers all four, consistent with `csus_experiment.py`'s ~$800-2500 total
budget framing.
""")

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13.0"},
    },
    "cells": cells,
}
out = pathlib.Path("notebooks/thz_griffiths_reference.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Written: {out}  ({len(cells)} cells)")
