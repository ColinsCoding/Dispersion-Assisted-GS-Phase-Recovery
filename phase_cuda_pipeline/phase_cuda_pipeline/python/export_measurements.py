#!/usr/bin/env python3
"""Export sqrt(I1), sqrt(I2), frequency grid from the notebook model to binary files."""
import numpy as np
from visualizer import GHZ, phi2_from_D, make_grid, make_waveform, disperse, intensity

N = 2**14
_, f_hz, w_rad = make_grid(N, 0.10 * GHZ)
phi2_1, phi2_2 = phi2_from_D(300), phi2_from_D(1200)
# Change this to gaussian, chirped_gaussian, gas3, or chirped_nrz.
et0 = make_waveform("gas3", np.fft.fftshift(np.fft.fftfreq(N, d=0.10*GHZ)), f_hz)
amp1 = np.sqrt(intensity(disperse(et0, w_rad, phi2_1))).astype(np.float64)
amp2 = np.sqrt(intensity(disperse(et0, w_rad, phi2_2))).astype(np.float64)
amp1.tofile("data/amp1.bin")
amp2.tofile("data/amp2.bin")
f_hz.astype(np.float64).tofile("data/f_hz.bin")
print("wrote data/amp1.bin data/amp2.bin data/f_hz.bin")
