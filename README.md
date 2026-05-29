# Dispersion-Assisted GS Phase Recovery

Intensity-only phase retrieval for carrier-less coherent fiber sensing.
Two dispersed measurements. No local oscillator. No 90° hybrid.

**OUSD(R&E)** — FutureG · Trusted AI · Advanced Computing · Integrated Sensing · Directed Energy · Quantum Science · HMI

---

## Live Dashboard

[![Live Dashboard](https://img.shields.io/badge/Live%20Dashboard-%E2%96%B6%20Open-50d8ff?style=for-the-badge&logo=flask&logoColor=black)](https://jalabi-dashboard.onrender.com)
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/ColinsCoding/Dispersion-Assisted-GS-Phase-Recovery)
[![Health](https://img.shields.io/badge/health-/health-00ff9f?style=flat-square)](https://jalabi-dashboard.onrender.com/health)

> **Two clicks to use the dashboard:**
> 1. Click **▶ Open** badge above
> 2. Pick a tab (QPSK · WDM · Digital · 3-D Hash) — or drop a `.csv` / `.npy` file

---

## Deploy your own copy in 5 steps (free)

```
1.  Go to:  https://render.com/deploy?repo=https://github.com/ColinsCoding/Dispersion-Assisted-GS-Phase-Recovery
2.  Click "Deploy to Render"  (no credit card required for free tier)
3.  Wait ~3 min for Docker build
4.  Copy your URL:  https://<your-service-name>.onrender.com
5.  Replace the badge URL in this README, commit
```

**Keep it awake 24/7 for free** (Render free tier sleeps after 15 min idle):
```
uptimerobot.com → New Monitor → HTTP(s) → URL: https://<your-name>.onrender.com/health
Interval: 5 minutes  →  your service never sleeps
```

**Alternative — Cloudflare Tunnel (no server, no sleep, machine must stay on):**
```bash
cloudflared tunnel --url http://localhost:5000
# prints: https://xxxx.trycloudflare.com  ← paste into README badge
```

**Alternative — Railway ($5/mo, true always-on):**
```
railway.app → New Project → Deploy from GitHub
             → select this repo → port 5000 → Deploy
```

---

```
 I₁[n] ── D₁ = −600 ps²  ──┐
                             ├── TD-GS (200 iter) ──► φ(t)  ──► CNN ──► SNMP alert
 I₂[n] ── D₂ = −900 ps²  ──┘

 H(ν) = exp(iαDν²)     α = πλ₀²/c ≈ 2.515×10⁻⁵     λ₀ = 1550 nm
```

---

```
make install   # pip install -r requirements.txt
make run       # jupyter notebook phase_retrieval.ipynb
make seals     # jupyter notebook notebooks/seals_simulation.ipynb
make execute   # run all cells in place
make firmware  # cross-compile → firmware/rogueguard_firmware.elf  (aarch64-linux-gnu-gcc)
make push      # git push origin main
make dashboard # python optical_dashboard/app.py → http://localhost:5000
```

---

## Optical Dashboard · `optical_dashboard/`

Interactive Flask web app — five tabs, all dark-theme, no page reloads.

```
python optical_dashboard/app.py   →   http://localhost:5000
```

| Tab | Endpoint | Description |
|-----|----------|-------------|
| **Signal Analysis** | `POST /upload`, `GET /demo` | Drag-drop CSV/.npy → time-domain · PSD · spectrogram · TD-GS phase · autocorrelation. File auto-processes on select (two-click). UUID-isolated upload dirs, 1-hour auto-cleanup. |
| **QPSK Modem** | `GET /qpsk?snr=&nbits=` | Gray-coded QPSK, RRC pulse shaping (β=0.35), AWGN, matched-filter RX. Constellation · BER vs SNR · eye diagram · phase trellis. |
| **WDM 48-ch** | `GET /wdm?nch=&snr=` | ITU-T G.694.1 C-band, 100 GHz spacing. Per-channel demux power bar · λ-scatter · PSD overlay. |
| **Digital Logic** | `GET /digital?byte=&cycles=` | D-latch · D flip-flop · 8-bit shift register · 2:1 MUX · TDM 2:1 mux/demux. State waterfall + control signals. |
| **3-D Hash** | `GET /hash3d?npts=` | Sparse 3-D voxel hash over (x,y,λ). Energy min: gradient descent on H=0.1·H_TV+0.4·H_pc. LSH nearest-neighbour retrieval. |

DSP module: `optical_dashboard/dsp.py`

```python
from optical_dashboard import dsp as DSP

result = DSP.simulate_link(n_bits=2048, snr_db=12)   # QPSK
result = DSP.wdm_sim(n_ch=48, snr_db=25)              # WDM
result = DSP.optical_hash_demo(n_points=256)           # 3-D hash + energy min
```

### Docker deploy (24/7 uptime)

```bash
# 1. Build + start (restarts automatically on crash or reboot)
docker compose up -d --build

# 2. Check health (boolean status endpoint)
curl http://localhost:5000/health
# {"alive": true, "version": "1.1.0", "uptime_s": 142, ...}

# 3. Logs
docker compose logs -f dashboard

# 4. Stop
docker compose down
```

Security model: Flask container runs on an **internal Docker network with zero outbound internet**.  
Filesystem is **read-only** (`read_only: true`) except the uploads volume.  
All query parameters validated by regex/range guards — bad kwargs return HTTP 400.  
Filenames whitelist-filtered: `[A-Za-z0-9_-][A-Za-z0-9_-.]{0,60}.(csv|npy|txt|dat)`.

### Public HTTPS via Cloudflare Tunnel (free, no port-forward)

```bash
# One-time setup (install + login)
winget install --id Cloudflare.cloudflared
cloudflared tunnel login

# Create tunnel + DNS route
cloudflared tunnel create jalabi-dashboard
cloudflared tunnel route dns jalabi-dashboard dashboard.yourdomain.com

# Add token to .env
cp .env.example .env
# paste CLOUDFLARE_TUNNEL_TOKEN=<token from: cloudflared tunnel token jalabi-dashboard>

# Start with Cloudflare profile
docker compose --profile cloudflare up -d
```

No account? Use the free try.cloudflare.com one-liner:
```bash
docker run --rm --network jalabi_internal cloudflare/cloudflared:latest \
       tunnel --url http://dashboard:5000
# Prints a random public URL good for ~24 h
```

---

## Notebook · `phase_retrieval.ipynb` · 109 cells

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
