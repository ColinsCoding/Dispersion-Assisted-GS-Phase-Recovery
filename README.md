# Dispersion-Assisted GS Phase Recovery

> **ECE 279AS — Project 2 · UCLA · Spring 2026 · Complete**  
> Independent student work. Formerly associated with Prof. Jalali's lab.

Intensity-only phase retrieval for carrier-less coherent fiber sensing.
Two dispersed measurements. No local oscillator. No 90° hybrid.

**OUSD(R&E)** — FutureG · Trusted AI · Advanced Computing · Integrated Sensing · Directed Energy · Quantum Science · HMI

---

## Upload optical data

**Local:**
```
.\start.bat          # double-click on Windows → http://localhost:5000/#signal
```

**Direct link to the upload widget** (once the server is running):

> [`http://localhost:5000/#signal`](http://localhost:5000/#signal)

The `#signal` hash opens the **Signal Analysis** tab directly — drag-and-drop
your `.mat` / `.csv` / `.npy` file and it processes immediately.

Upload widget source:
[`optical_dashboard/templates/index.html`](optical_dashboard/templates/index.html)

---

## Acknowledgment

The dispersion-assisted phase retrieval concept originates from the research of
**Prof. Bahram Jalali** and his group at UCLA
(Jalali et al., *Appl. Phys. Lett.* 95, 231108, 2009).
The SEALS and STEAM experimental frameworks are likewise products of that lab.

This repository is an independent student implementation completed for ECE 279AS,
Spring 2026. It does not represent the lab's official code or results.
Any errors are entirely my own.

I am grateful for the mentorship, the lecture material, and the opportunity to
engage with this research direction. I apologize for any confusion caused by
associating this project page with the Jalali Lab name.

---

```
 I₁[n] ── D₁ = −600 ps²  ──┐
                             ├── TD-GS (200 iter) ──► φ(t)  ──► CNN ──► OPA phase commands
 I₂[n] ── D₂ = −900 ps²  ──┘

 H(ν) = exp(iαDν²)     α = πλ₀²/c ≈ 2.515×10⁻⁵     λ₀ = 1550 nm
```

---

## Architecture — Flask + C socket on AWS Graviton

```
┌─────────────────────────────────────────────────────────────────┐
│  Optical front-end  (fiber + dual ADC, 56 GSa/s)               │
│    I₁(t)  D₁ = −600 ps²   I₂(t)  D₂ = −1200 ps²              │
└──────────────────────┬──────────────────────────────────────────┘
                       │ raw float32 stream  (TCP, AF_INET SOCK_STREAM)
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  C socket server  `server/optical_rx.c`                         │
│  AWS Graviton 3 (Neoverse V1, ARMv8.4-A, 64-core)             │
│  recv() → ring buffer → FFTW3f + NEON SVE → TD-GS 200 iter     │
│  Latency target: < 1 ms per frame                               │
└──────────────────────┬──────────────────────────────────────────┘
                       │ JSON over Unix socket  /tmp/gs.sock
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  Python Flask  `optical_dashboard/app.py`                       │
│  /upload  /summary  /uploads/log  /health                       │
│  Signal Analysis · QPSK · WDM · Digital Logic · 3-D Hash       │
└──────────────────────┬──────────────────────────────────────────┘
                       │ φ(t) phase commands
                       ▼
              Optical phased array (OPA)
              laser beamforming  →  directed energy / sensing
```

### C socket layer (`server/`)

| File | Purpose |
|---|---|
| `optical_rx.c` | TCP listener, ADC frame receiver, ring buffer |
| `gs_solver.c` | TD-GS core: FFTW3f complex FFT, NEON SVE butterfly |
| `opa_control.c` | Phase → OPA element voltages (DAC SPI) |
| `Makefile` | `aarch64-linux-gnu-gcc -O3 -march=armv8.4-a+sve` |

```bash
make server   # cross-compile C socket server → server/optical_rx  (Graviton ELF)
make run      # jupyter notebook phase_retrieval.ipynb
make dashboard # python optical_dashboard/app.py → http://localhost:5000
make install  # pip install -r requirements.txt
make execute  # run all notebook cells in place
make firmware # cross-compile → firmware/rogueguard_firmware.elf  (aarch64-linux-gnu-gcc)
make push     # git push origin main
```

---

## Optical Dashboard · `optical_dashboard/`

Interactive Flask web app — dark-theme, no page reloads.

```
python optical_dashboard/app.py   →   http://localhost:5000
```

| Endpoint | Description |
|----------|-------------|
| `POST /upload` | `.mat` / CSV / NPY → time-domain · PSD · STFT · TD-GS phase · autocorrelation. `.mat` with I1/I2 runs 2-arm TD-GS (50 iter). UUID-isolated dirs, 1-hour cleanup. |
| `GET /demo` | Synthetic STEAM pulse — dispersion-stretched Gaussian |
| `GET /qpsk` | Gray-coded QPSK, RRC (β=0.35), AWGN, matched-filter RX · BER vs SNR · eye diagram |
| `GET /wdm` | ITU-T G.694.1 48-ch C-band, 100 GHz spacing · per-channel demux power |
| `GET /digital` | D-latch · D flip-flop · 8-bit shift register · 2:1 MUX · TDM mux/demux |
| `GET /hash3d` | Sparse 3-D voxel hash (x,y,λ) · H=0.1·H_TV+0.4·H_pc energy min · LSH retrieval |
| `GET /uploads/log` | SQLite audit table — every upload row, DSP metadata, lab attribution `?limit=&offset=&status=` |
| `GET /summary` | Aggregate stats — by extension, by lab, two-arm %, avg processing ms |
| `GET /health` | `{"alive":true, "version":"1.2.0", "uploads_total":N}` |

DSP module: `optical_dashboard/dsp.py`

```python
from optical_dashboard import dsp as DSP

result = DSP.simulate_link(n_bits=2048, snr_db=12)   # QPSK
result = DSP.wdm_sim(n_ch=48, snr_db=25)              # WDM
result = DSP.optical_hash_demo(n_points=256)           # 3-D hash + energy min
result = DSP.optical_link_fsm_demo(n_bits=256)         # TX/RX dual FSM
```

Security: Flask container on **internal Docker network (zero outbound internet)** ·
read-only rootfs · `cap_drop=ALL` · filename regex whitelist
`[A-Za-z0-9_-][A-Za-z0-9_-.]{0,60}.(csv|npy|txt|dat|mat)` ·
all query params validated · IP stored as SHA-256 hash only.

```bash
# Local Docker run
docker compose up -d --build
curl http://localhost:5000/health

# Verify with synthetic test data (6 optical standards)
pip install requests scipy
python test_upload.py
```

---

## Notebook · `phase_retrieval.ipynb` · 113 cells

| § | Topic | OUSD Area |
|---|---|---|
| 1–2b | Environment · electromagnetic foundations · H(ν)=exp(iαDν²) derivation | Advanced Computing |
| 3–6 | Forward model · synthetic signals · TD-GS algorithm · baseline recovery | Advanced Computing |
| 7 | PhyCV Phase-Stretch Transform — single-measurement baseline | Advanced Computing |
| 8–9 | Convergence (1–1600 iter) · measurement diversity D₂/D₁ sweep | Advanced Computing |
| 10 | QPSK phase recovery — FutureG communication waveform | FutureG |
| 11 | Neural network phase regression (PyTorch) | Trusted AI & Autonomy |
| 12 | CUDA/GPU acceleration — elementwise complex kernel | Advanced Computing |
| 13 | Ghost imaging — spatial phase via intensity correlations | Integrated Sensing |
| 15–17 | Wirtinger flow · Poisson noise · L1 sparse retrieval | Advanced Computing |
| 19 | Parameter sweep — wavelength, fiber length, Fresnel noise | Advanced Computing |
| 22 | 2D phased array beamforming · HITRAN CO absorption | Directed Energy |
| 23–24 | Deep unrolling (GS as trainable NN) · Kalman phase tracker | Trusted AI & Autonomy |
| 31 | Heterodyne & coherent detection — traditional receiver comparison | FutureG |
| 32 | Algorithm benchmark · research roadmap | Advanced Computing |
| 35 | Speckle statistics — fiber sensing noise floor | Integrated Sensing |
| 43 | Distributed fiber sensing — 100 km phase retrieval | Integrated Sensing |
| 45 | LIGO — most sensitive phase measurement in history | Integrated Sensing |
| 46 | THz time-domain spectroscopy | Directed Energy |
| 47a–b | Qubit · Bra-ket algebra (quantum sensing foundation) | Quantum Science |
| 53 | SEALS — Mie + Rayleigh Python port, wavelength-to-angle | Integrated Sensing |
| 54 | FutureG — carrier-less architecture · SWaP analysis | FutureG |
| 55 | RogueGuard 1U — embedded deployment · ROC · pipeline latency | Trusted AI · HMI |
| 67 | OUSD(R&E) alignment · Thevenin/Norton optical equivalents · STEAM SPICE netlist | FutureG · Advanced Computing |
| 68 | TAM bubble chart — RogueGuard $420M / STEAM-Dx $1800M / TS-QPI $280M / SEALS $650M | Trusted AI |
| 70 | SBIR pathway · Phase I/II checklist · DoD insertion milestones | Trusted AI |
| 71 | C11 beamformer: RF/OPA HPBW · Marx EMP discharge · BLDC PID · combat FSM | Directed Energy · HMI |
| 72 | Neural PID backprop (torch) · EM phase optimisation (Adam) · Lagrangian/Jacobian · laser cavity · Falcon 9 Tsiolkovsky | Trusted AI · Advanced Computing |
| 73 | 3-D optical voxel hash · energy minimisation (H_TV + H_pc Wirtinger GD) · LSH retrieval | Advanced Computing · Trusted AI |
| 74 | Odd/even Cooley-Tukey butterfly circuit · F_p field arithmetic (BN128) · keccak256 commitments · `OpticalPhaseVerifier.sol` explicit trust | Trusted AI · Advanced Computing |
| 75 | TX/RX dual FSM (male/female complementary pair) · modular arithmetic state machines (step%5, step%6, k%2) · MATLAB .mat upload · 2-arm TD-GS 50-iter convergence | FutureG · Advanced Computing |
| 76 | Jalali Lab aggregate study · 6 optical standards (OOK-NRZ, PAM4, QPSK, DPSK, STEAM, Soliton) · comparative TD-GS convergence · project conclusion | All OUSD areas |

## SEALS · `notebooks/seals_simulation.ipynb` · 10 cells

Python port of Jalali Lab MATLAB simulation (Project 1 deliverable).
Lorenz-Mie (exact) + Rayleigh-Debye (approximate) + SEALS wavelength-to-angle mapping.

---

## Hardware — RogueGuard 1U

```
RPi CM4 · 4× ARM Cortex-A72 @ 1.5 GHz · 4 GB RAM
Dual 14-bit ADC · 56 GSa/s · D₁ = −600 ps²  D₂ = −900 ps²
TD-GS (FFTW3f + NEON SIMD) ~1.5 ms · INT8 CNN (NNPACK) ~0.5 ms · Total < 3.5 ms
```

Source: `firmware/` — C11 · `make firmware` → aarch64 ELF

---

## References

```
[1] Gerchberg & Saxton, Optik 35(2):237–246, 1972
    "A practical algorithm for the determination of phase from image and diffraction plane pictures"
    references/gs1972.pdf

[2] Jalali et al., Appl. Phys. Lett. 95, 231108, 2009
    "Optical phase recovery in the dispersive Fourier transform"
    references/jalali2009.pdf

[3] Yao et al., Optica, 2022
    "Neural network enabled time stretch spectral regression"
    references/yao2022.pdf

[4] Adam et al., Opt. Express 21(4), 2013
    "Spectrally Encoded Angular Light Scattering"
```
