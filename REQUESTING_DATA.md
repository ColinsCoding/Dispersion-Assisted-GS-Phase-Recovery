# Data Request — ECE 279AS Project 2

**To:** Yiming Zhou &lt;yimingz0416@g.ucla.edu&gt;, Callen MacPhee &lt;cmacphee@g.ucla.edu&gt;  
**Subject:** Project 2 — Optical Communication Data Request (Phase Retrieval)

---

Hello Yiming and Callen,

I am a student enrolled in ECE 279AS working on Project 2: *Dispersion-Assisted Optical Phase Recovery*.
I have completed the Colab/Jupyter notebook simulator and GitHub repository for the Gerchberg–Saxton
time-domain phase retrieval system and am ready to test it on real optical communication data.

**GitHub repository:** https://github.com/ColinsCoding/Dispersion-Assisted-GS-Phase-Recovery  
**Live dashboard:** https://jalabi-dashboard.onrender.com  

## Data format accepted

The dashboard accepts any of the following:

| Format | Content | Notes |
|--------|---------|-------|
| `.npy` | 1-D float array (intensity only) or (N, 2) array (time, intensity) | NumPy save |
| `.mat` | MATLAB workspace — looks for variables: `I1`, `I2`, `data`, `signal`, `intensity` | `save('data.mat','I1','I2')` |
| `.csv` | Two columns: `time, intensity` — no header required | |
| `.txt` | Same as CSV | |

## Parameters needed (to configure the GS algorithm)

- Fiber dispersion **D1** (ps²) — default assumed −600 ps²
- Fiber dispersion **D2** (ps²) — default assumed −1200 ps²
- Center wavelength **λ₀** (nm) — default 1550 nm
- Sample rate **fs** (GSa/s)
- Modulation format (QPSK / other)

## What the simulator does with the data

1. Loads I₁(t) and I₂(t) from the file
2. Computes dispersive transfer functions H(ν) = exp(iπDν²)
3. Runs TD-GS for up to 200 iterations
4. Displays: time-domain waveform · PSD · STFT spectrogram · recovered phase φ(t) · instantaneous frequency · autocorrelation

Please send the data file(s) at your convenience.  
I can also bring a USB drive to the lab if that is easier.

Thank you for the mentorship on this project.

Best regards,  
[Your name]  
UCLA ECE 279AS — Spring 2026
