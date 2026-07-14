"""Generate notebooks/holographic_3d_reconstruction.ipynb."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from nbkit import md, co, setup_cell, write

cells = [
md(r"""# Holographic 3-D reconstruction and distance estimation

A hologram records not just intensity but the full **complex field** -- amplitude *and* phase -- by
letting the object wave interfere with a coherent reference. That reference is the **local oscillator**
of coherent detection: the same principle as heterodyne radio and terahertz receivers, where mixing an
unknown wave with a known LO recovers its phase. Once the complex field at the sensor is known, the
object can be **numerically refocused** to any depth by propagating the field, and the object's
**distance** is found by locating the depth of sharpest focus. Reconstructing a whole stack of depths
gives a 3-D image (add time for 4-D).

This notebook builds that pipeline on the angular-spectrum propagator from earlier chapters: record a
hologram of objects at different depths, back-propagate to a range of $z$, and estimate each depth from
an autofocus metric. Self-contained: NumPy, SymPy, Pandas, Matplotlib."""),
setup_cell(),

md(r"""## The propagator and the recording

Free-space propagation is the angular spectrum $U(z)=\mathcal F^{-1}\{\mathcal F\{U_0\}\,H_z\}$ with
$H_z=\exp(ikz\sqrt{1-(\lambda f_x)^2-(\lambda f_y)^2})$. A coherent detector (object wave + LO reference)
records the complex field at the sensor plane; here we form it directly as the sum of two point
objects, each propagated from its own depth."""),
co("""lam = 0.5e-6; dx = 8e-6; N = 160
fx = np.fft.fftfreq(N, d=dx); FX, FY = np.meshgrid(fx, fx)
def propagate(U, z):
    arg = 1 - (lam*FX)**2 - (lam*FY)**2
    H = np.exp(1j*(2*np.pi/lam)*z*np.sqrt(arg.astype(complex)))
    H[arg < 0] = 0                                    # band-limited (drop evanescent)
    return np.fft.ifft2(np.fft.fft2(U) * H)

def spot(cx, cy, w=0.8):                              # a small Gaussian 'point' object (tight -> short DOF)
    y, x = np.mgrid[0:N, 0:N]
    return np.exp(-((x-cx)**2+(y-cy)**2)/(2*w**2)).astype(complex)

z_true = [1.0e-3, 3.0e-3]                             # the two object depths
obj = [spot(58, 70), spot(104, 92)]                  # at different transverse positions
hologram = sum(propagate(o, z) for o, z in zip(obj, z_true))   # recorded complex field
print("hologram grid:", hologram.shape, " recorded energy =", round(np.sum(np.abs(hologram)**2), 2))
assert hologram.shape == (N, N)"""),

md(r"""## Numerical refocusing

Back-propagating the recorded field by $-z$ brings whatever sat at depth $z$ into focus while
everything else stays blurred. Refocusing to each true depth recovers a sharp point there; refocusing
to the wrong depth leaves a spread-out blur."""),
co("""def reconstruct(z):
    return propagate(hologram, -z)                    # back-propagate

for z in z_true:
    R = np.abs(reconstruct(z))**2
    peak = R.max()/R.mean()                           # peak-to-mean: high when a point is in focus
    print(f"refocus to z={z*1e3:.1f} mm:  peak/mean = {peak:6.1f}")
# a focused reconstruction is far peakier than a defocused one (midway between the objects)
z_mid = 0.5*(z_true[0]+z_true[1])
sharp = (np.abs(reconstruct(z_true[0]))**2).max()
blur  = (np.abs(reconstruct(z_mid))**2).max()
print(f"in-focus peak {sharp:.3e} >> defocused peak {blur:.3e}")
assert sharp > 3*blur"""),

md(r"""## Autofocus = distance estimation

Sweep the reconstruction depth and score each with a focus metric -- the normalized variance
(Tamura) $T=\sigma_I/\bar I$ of the reconstructed intensity, which is large when the image is a sharp
peak and small when it is smeared. The metric peaks at the true object depths, so its maxima *are* the
estimated distances."""),
co("""def tamura(z):
    I = np.abs(reconstruct(z))**2
    return I.std()/I.mean()

zs = np.linspace(0.4e-3, 4.0e-3, 220)
M = np.array([tamura(z) for z in zs])
# find the two strongest local maxima
loc = [i for i in range(1, len(M)-1) if M[i] > M[i-1] and M[i] > M[i+1]]
loc.sort(key=lambda i: -M[i])
est = sorted(zs[i] for i in loc[:2])
print("true depths (mm):     ", [round(z*1e3, 2) for z in z_true])
print("estimated depths (mm):", [round(z*1e3, 2) for z in est])
for zt, ze in zip(z_true, est):
    assert abs(zt - ze) < 0.25e-3                     # within the depth resolution
print("distance estimation succeeded (within depth resolution)")"""),

md(r"""## The 3-D (and 4-D) picture

The full reconstruction is a *volume*: propagate the one recorded hologram to a stack of depths and
each object lights up in its own plane. A maximum-intensity projection through the stack shows both
points at their true positions -- 3-D from a single 2-D coherent capture. Repeating per video frame
adds time (4-D); repeating per wavelength adds spectral dimensions."""),
co("""z_stack = np.linspace(0.6e-3, 3.6e-3, 40)
vol = np.array([np.abs(reconstruct(z))**2 for z in z_stack])     # (depth, y, x)
mip_xy = vol.max(axis=0)                                          # max-intensity projection (top view)
mip_xz = vol.max(axis=1)                                          # side view (depth vs x)
# each object appears at its own depth slice
d0 = np.argmax(vol.max(axis=(1,2)))                               # brightest slice
print("brightest reconstruction slice at z =", round(z_stack[d0]*1e3, 2), "mm")
print("volume shape (depth, y, x):", vol.shape)"""),

md(r"""## Plots"""),
co(r"""fig, ax = plt.subplots(1, 3, figsize=(13.5, 4))
ax[0].plot(zs*1e3, M, color="#4C78A8")
for zt in z_true: ax[0].axvline(zt*1e3, ls=":", color="#E45756")
ax[0].set_xlabel("reconstruction depth z (mm)"); ax[0].set_ylabel("focus metric (Tamura)")
ax[0].set_title("autofocus: peaks = object distances")
ax[1].imshow(mip_xy, cmap="inferno"); ax[1].set_title("3-D reconstruction (top view, MIP)"); ax[1].axis("off")
ax[2].imshow(mip_xz.T, cmap="inferno", aspect="auto",
             extent=[z_stack[0]*1e3, z_stack[-1]*1e3, 0, N])
ax[2].set_title("side view: objects at their depths"); ax[2].set_xlabel("z (mm)"); ax[2].set_ylabel("x (px)")
plt.tight_layout(); plt.show()""" ),

md(r"""## Summary

- A hologram records the **complex field** by beating the object wave against a coherent **local
  oscillator** -- the heterodyne principle shared with radio and terahertz coherent receivers.
- **Numerical refocusing** back-propagates that field (angular spectrum) to any depth; an object is
  sharp only at its own $z$.
- An **autofocus metric** (normalized variance) peaks at the object depths, so its maxima give the
  **distance estimate** -- here recovering two depths to within the axial resolution.
- Propagating to a **stack of depths** reconstructs a 3-D volume from one 2-D capture; time and
  wavelength extend it to 4-D and beyond. In real time this is a stack of FFT propagations -- a GPU
  workload, the same FFT engine as the rest of the instrument.

Subject-verb-object: the reference beam supplies the local oscillator; the sensor records the complex
field; propagation refocuses each depth; the focus metric estimates the distance; the stack builds the
3-D image."""),
]

write("holographic_3d", "reconstruction", cells)
