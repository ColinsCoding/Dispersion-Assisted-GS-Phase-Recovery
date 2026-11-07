"""Make figures/hero.png and figures/convergence.gif for the README.

    python scripts/make_hero.py

Signal: gs_core.make_measurements("QPSK", ...). Despite the name, that is a smooth,
band-limited random phase on a constant-amplitude carrier (the test signal of the
2009 paper), not discrete QPSK symbols, so the figure calls it what it is. The true
phase is known, so recovery is checked, not just shown. The GPU panel runs a batch
of independent problems with PyTorch and checks the GPU answer against NumPy first.
"""
import pathlib, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dgs import gs_core as gs

OUT = ROOT / "figures"; OUT.mkdir(exist_ok=True)
BLUE, ORANGE, GREEN, RED, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#555555"
SPS, NSYM, N_ITER = 8, 64, 300
D1, D2 = -2000.0, -8000.0            # gs_core's default pair (-5000, -5750) stalls far more often
OK_RAD = 0.15                         # "recovered" = RMS phase error below this
wrap = lambda x: np.angle(np.exp(1j * x))


def run(seed, keep=False):
    m = gs.make_measurements("QPSK", n_symbols=NSYM, sps=SPS, D1=D1, D2=D2, snr_db=30.0, rng_seed=seed)
    I1, I2, ph = m["I1"], m["I2"], m["phi_true"]
    E = gs.undisperse(np.sqrt(I1).astype(complex), D1)
    mis, snaps = [], {}
    for k in range(1, N_ITER + 1):
        E = gs.gs_iteration(E, I1, I2, D1, D2, unit_amplitude=True)
        mis.append(np.linalg.norm(np.abs(gs.disperse(E, D1)) ** 2 - I1) / np.linalg.norm(I1))
        if keep:
            snaps[k] = E.copy()
    align = lambda Eh: Eh * np.exp(-1j * np.angle(np.vdot(np.exp(1j * ph), Eh)))  # global phase is unobservable
    rms = np.sqrt(np.mean(wrap(np.angle(align(E)) - ph) ** 2))
    return dict(m=m, E=E, mis=np.array(mis), rms=rms, snaps=snaps, align=align)


# --- 20 random signals: how often does it work, and can we tell? --------------
trials = [run(s) for s in range(20)]
good = [t for t in trials if t["rms"] < OK_RAD]; bad = [t for t in trials if t["rms"] >= OK_RAD]
g_hi = max(t["mis"][-1] for t in good); b_lo = min(t["mis"][-1] for t in bad) if bad else np.inf
assert g_hi < b_lo, "final mismatch no longer separates successes from failures; update the figure text"
print(f"20 signals: {len(good)} recovered (RMS phase error < {OK_RAD} rad). Final arm-1 mismatch: "
      f"successes <= {100*g_hi:.1f} %, failures >= {100*b_lo:.1f} %")

# show the first successful signal (signal 0 here)
hero = run(next(i for i, t in enumerate(trials) if t["rms"] < OK_RAD), keep=True)
m, I1, I2, ph = hero["m"], hero["m"]["I1"], hero["m"]["I2"], hero["m"]["phi_true"]
phi_rec = np.angle(hero["align"](hero["E"]))
print(f"shown signal: RMS phase error {hero['rms']:.3f} rad, arm-1 mismatch {100*hero['mis'][-1]:.2f} %")

# --- batched GS in torch vs the NumPy loop -------------------------------------
import torch
dev = "cuda" if torch.cuda.is_available() else "cpu"
gpu_name = torch.cuda.get_device_name(0) if dev == "cuda" else "CPU (no CUDA)"
B, N_IT_T = 2048, 100
batch = [gs.make_measurements("QPSK", n_symbols=NSYM, sps=SPS, D1=D1, D2=D2, snr_db=30.0, rng_seed=100 + i) for i in range(B)]
I1b = np.stack([b["I1"] for b in batch]); I2b = np.stack([b["I2"] for b in batch])
N = I1b.shape[1]
nu = np.fft.fftfreq(N)                                        # same signed grid as gs_core.disperse
H1, H2 = np.exp(1j * np.pi * D1 * nu ** 2), np.exp(1j * np.pi * D2 * nu ** 2)


def gs_torch(I1t, I2t, n_it):
    h1 = torch.as_tensor(H1, device=dev); h2 = torch.as_tensor(H2, device=dev)
    a1, a2 = torch.sqrt(I1t), torch.sqrt(I2t)
    E = torch.fft.ifft(torch.fft.fft(a1.to(torch.complex128)) / h1)
    for _ in range(n_it):
        Ed = torch.fft.ifft(torch.fft.fft(E) * h1)
        E = torch.exp(1j * torch.fft.ifft(torch.fft.fft(a1 * torch.exp(1j * Ed.angle())) / h1).angle())
        Ed = torch.fft.ifft(torch.fft.fft(E) * h2)
        E = torch.exp(1j * torch.fft.ifft(torch.fft.fft(a2 * torch.exp(1j * Ed.angle())) / h2).angle())
    return E


E_ref = gs.undisperse(np.sqrt(I1b[0]).astype(complex), D1)
for _ in range(N_IT_T):
    E_ref = gs.gs_iteration(E_ref, I1b[0], I2b[0], D1, D2, unit_amplitude=True)
I1t = torch.as_tensor(I1b, device=dev); I2t = torch.as_tensor(I2b, device=dev)
diff = np.max(np.abs(gs_torch(I1t[:1], I2t[:1], N_IT_T).cpu().numpy()[0] - E_ref))
assert diff < 1e-8, f"GPU and NumPy disagree by {diff}"
print(f"GPU vs NumPy, {N_IT_T} iterations: max |difference| {diff:.1e}")

n_cpu = 64
t0 = time.perf_counter()
for i in range(n_cpu):
    Ec = gs.undisperse(np.sqrt(I1b[i]).astype(complex), D1)
    for _ in range(N_IT_T):
        Ec = gs.gs_iteration(Ec, I1b[i], I2b[i], D1, D2, unit_amplitude=True)
cpu_per = (time.perf_counter() - t0) / n_cpu
gs_torch(I1t[:8], I2t[:8], 2)
if dev == "cuda": torch.cuda.synchronize()
t0 = time.perf_counter(); gs_torch(I1t, I2t, N_IT_T)
if dev == "cuda": torch.cuda.synchronize()
gpu_per = (time.perf_counter() - t0) / B
speedup = cpu_per / gpu_per
print(f"per signal: NumPy CPU {1e3*cpu_per:.2f} ms, {gpu_name} (batch {B}) {1e3*gpu_per:.3f} ms -> {speedup:.0f}x")

# --- hero figure ---------------------------------------------------------------
t = np.arange(N) / SPS
fig = plt.figure(figsize=(13, 7.2))
grid = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.35, 1], hspace=0.42, wspace=0.32)

ax = fig.add_subplot(grid[0, :2])
ax.plot(t, I1, color=BLUE, lw=1.4, label="arm 1 (dispersion D1)")
ax.plot(t, I2, color=ORANGE, lw=1.4, ls="--", label="arm 2 (dispersion D2)")
ax.set_title("1. What two photodetectors measure: intensity only, no phase", loc="left")
ax.set_xlabel("time (samples / 8)"); ax.set_ylabel("intensity"); ax.legend(loc="upper right", fontsize=9); ax.set_xlim(0, 32)

ax = fig.add_subplot(grid[1, :2])
ax.plot(t, ph, color=GREY, lw=4, alpha=0.5, label="true phase")
ax.plot(t, phi_rec, color=GREEN, lw=1.3, label="recovered from the two intensities")
ax.set_title(f"2. Recovered phase: RMS error {hero['rms']:.3f} rad", loc="left")
ax.set_xlabel("time (samples / 8)"); ax.set_ylabel("phase (rad)"); ax.set_xlim(0, 32)
ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]); ax.set_yticklabels(["-π", "-π/2", "0", "π/2", "π"])
ax.legend(loc="upper right", fontsize=9)

ax = fig.add_subplot(grid[0, 2])
it = np.arange(1, N_ITER + 1)
for tr in bad:
    ax.loglog(it, 100 * tr["mis"], color=RED, lw=1, alpha=0.55)
for tr in good:
    ax.loglog(it, 100 * tr["mis"], color=GREEN, lw=1, alpha=0.8, ls="--")
ax.plot([], [], color=GREEN, ls="--", label=f"recovered ({len(good)}/20)")
ax.plot([], [], color=RED, label=f"stalled ({len(bad)}/20)")
ax.set_title("3. Built-in trust check", loc="left"); ax.set_xlabel("iteration"); ax.set_ylabel("arm-1 mismatch (%)")
ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.3, which="both")

ax = fig.add_subplot(grid[1, 2])
label_gpu = "PyTorch\nRTX 4060" if "4060" in gpu_name else "PyTorch\n" + gpu_name
bars = ax.bar(["NumPy\nCPU", label_gpu], [1e3 * cpu_per, 1e3 * gpu_per], color=[GREY, GREEN])
ax.set_yscale("log"); ax.set_ylabel("ms per signal (log)"); ax.set_title(f"4. GPU batch: {speedup:.0f}x faster", loc="left")
for b, v in zip(bars, [cpu_per, gpu_per]):
    ax.text(b.get_x() + b.get_width() / 2, 1e3 * v * 1.15, f"{1e3*v:.1f} ms" if v > 1e-3 else f"{1e6*v:.0f} µs", ha="center")
ax.set_ylim(1e3 * gpu_per / 3, 1e3 * cpu_per * 4)

fig.suptitle("Recovering optical phase from two intensity measurements (dispersion-assisted Gerchberg-Saxton)",
             fontsize=14, x=0.07, ha="left")
fig.text(0.07, -0.025,
         f"Synthetic constant-amplitude signal with smooth random phase, {N} samples, 30 dB SNR, D = {D1:.0f}/{D2:.0f} (normalized). "
         f"Panel 3: 20 random signals; GS recovers {len(good)} and stalls on the rest, and the final mismatch tells them apart\n"
         f"without knowing the answer (recovered <= {100*g_hi:.1f} %, stalled >= {100*b_lo:.1f} %). "
         f"Panel 4: {B} signals x {N} samples x {N_IT_T} iterations; GPU output checked against NumPy before timing.",
         fontsize=8, color=GREY)
fig.savefig(OUT / "hero.png", dpi=130, bbox_inches="tight")

# --- GIF: recovered phase locking onto the true phase ---------------------------
frames = sorted(set(np.round(np.logspace(0, np.log10(N_ITER), 36)).astype(int)))
fig2, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw=dict(width_ratios=[1.7, 1]))
def draw(k):
    a1.clear(); a2.clear()
    a1.plot(t, ph, color=GREY, lw=4, alpha=0.5, label="true")
    a1.plot(t, np.angle(hero["align"](hero["snaps"][k])), color=GREEN, lw=1.4, label="recovered")
    a1.set_xlim(0, 32); a1.set_ylim(-3.5, 3.5); a1.set_xlabel("time"); a1.set_ylabel("phase (rad)")
    a1.set_title(f"iteration {k}"); a1.legend(loc="upper right", fontsize=8)
    a2.loglog(it[:k], 100 * hero["mis"][:k], color=GREEN, lw=2)
    a2.set_xlim(1, N_ITER); a2.set_ylim(1, 100); a2.grid(alpha=0.3, which="both")
    a2.set_title("arm-1 mismatch (%)"); a2.set_xlabel("iteration")
    fig2.tight_layout()
FuncAnimation(fig2, draw, frames=frames + [N_ITER] * 8).save(OUT / "convergence.gif", writer=PillowWriter(fps=7))
print("wrote", OUT / "hero.png", "and", OUT / "convergence.gif")
