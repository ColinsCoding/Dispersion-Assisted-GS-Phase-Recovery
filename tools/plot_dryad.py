# tools/plot_dryad.py
# Usage: python .\tools\plot_dryad.py
import os
import numpy as np
import matplotlib.pyplot as plt

# prefer torch if available for later GPU work; fall back to numpy
try:
    import torch
    has_torch = True
except Exception:
    has_torch = False

# --- load data (try .mat then .npz) ---
base = os.path.join("data", "raw")
mat_path = os.path.join(base, "pipeline_input.mat")
npz_path = os.path.join(base, "pipeline_input.npz")

data = {}
if os.path.exists(mat_path):
    from scipy import io as spio
    mat = spio.loadmat(mat_path)
    # common keys: i1, i2, time, group_delay
    for k in ("i1", "i2", "time", "group_delay"):
        if k in mat:
            data[k] = np.asarray(mat[k]).squeeze()
elif os.path.exists(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    for k in d.files:
        data[k] = np.asarray(d[k])
else:
    raise SystemExit("No pipeline_input.mat or pipeline_input.npz found in data/raw")

# ensure i1 exists
if "i1" not in data:
    raise SystemExit("i1 not found in loaded data")

i1 = data["i1"].astype(np.float32)
i2 = data.get("i2", None)
time = data.get("time", None)
group_delay = data.get("group_delay", None)

# if time missing, build simple axis
if time is None:
    time = np.arange(i1.size)

# convert to torch if available (keeps numpy fallback)
if has_torch:
    t_i1 = torch.from_numpy(i1)
    t_i2 = torch.from_numpy(i2) if i2 is not None else None
else:
    t_i1 = i1
    t_i2 = i2

# FFT (use numpy for plotting simplicity)
n = i1.size
# next power of two for FFT speed
nfft = 1 << (n-1).bit_length()
freq = np.fft.rfftfreq(nfft, d=1.0)  # sampling spacing unknown; freq in sample units
I1f = np.fft.rfft(i1, n=nfft)
mag1 = np.abs(I1f)

if i2 is not None:
    I2f = np.fft.rfft(i2, n=nfft)
    mag2 = np.abs(I2f)
else:
    mag2 = None

# plotting
os.makedirs("figures", exist_ok=True)
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# 1) time domain
axs[0].plot(time, i1, label="i1", lw=1)
if i2 is not None:
    axs[0].plot(time, i2, label="i2", lw=1, alpha=0.8)
axs[0].set_title("Time domain traces")
axs[0].set_xlabel("Sample index")
axs[0].set_ylabel("Amplitude")
axs[0].legend()
axs[0].grid(True)

# 2) frequency magnitude (log scale)
axs[1].plot(freq, mag1, label="|FFT(i1)|", lw=1)
if mag2 is not None:
    axs[1].plot(freq, mag2, label="|FFT(i2)|", lw=1, alpha=0.8)
axs[1].set_title("Frequency magnitude (sample-frequency units)")
axs[1].set_xlabel("Frequency (samples^-1)")
axs[1].set_ylabel("Magnitude")
axs[1].set_yscale("log")
axs[1].legend()
axs[1].grid(True, which="both", ls=":")

# 3) group delay (if present)
if group_delay is not None:
    # group_delay likely matches freq axis length; if not, resample or plot vs index
    gd = np.asarray(group_delay).squeeze()
    if gd.size == freq.size:
        axs[2].plot(freq, gd, color="tab:green")
        axs[2].set_xlabel("Frequency (samples^-1)")
    else:
        axs[2].plot(np.arange(gd.size), gd, color="tab:green")
        axs[2].set_xlabel("Index")
    axs[2].set_title("FBG group delay")
    axs[2].set_ylabel("Group delay (units from file)")
    axs[2].grid(True)
else:
    axs[2].text(0.5, 0.5, "No group_delay found", ha="center", va="center")
    axs[2].set_axis_off()

plt.tight_layout()
out_png = os.path.join("figures", "dryad_plots.png")
plt.savefig(out_png, dpi=200)
print("Saved plot to", out_png)
plt.show()
