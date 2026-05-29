"""
voxel_dog.py
------------
1. 3D wave field  e^(i*(kx*x + ky*y + kz*z))  — symmetrical approximation
2. MRI k-space reconstruction via 3D IFFT (GS pipeline in 3D)
3. Procedural voxel golden doodle — SDF-based, Minecraft chunk system
4. Thin-film interference pattern on dog surface

Run:  py -3.12 sim/voxel_dog.py
Out:  docs/voxel_dog_mri.png
      docs/voxel_wave3d.png
      docs/dog.vox          (MagicaVoxel format — open in any voxel viewer)
"""

import pathlib, struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

_HERE = pathlib.Path(__file__).parent
_DOCS = _HERE.parent / "docs"
_DOCS.mkdir(parents=True, exist_ok=True)

# ── 1. 3D WAVE FIELD  e^(i*(kx*x+ky*y+kz*z)) ────────────────────────────────
# Symmetrical: kx=ky=kz=k0  -> spherical plane wave approximation
# Real part = cos(k·r), Imag part = sin(k·r)
# Intensity = |E|^2 = 1 everywhere (all-pass, like your H(nu))

N  = 64
ax = np.linspace(-np.pi, np.pi, N)
X, Y, Z = np.meshgrid(ax, ax, ax, indexing='ij')

k0  = 2.0                          # spatial frequency
kx  = k0 * np.array([1, 0, 0])
ky  = k0 * np.array([0, 1, 0])
kz  = k0 * np.array([0, 0, 1])

# Superposition of 3 orthogonal plane waves (symmetric)
E3D = (np.exp(1j * k0 * X) +
       np.exp(1j * k0 * Y) +
       np.exp(1j * k0 * Z)) / 3.0

I3D  = np.abs(E3D)**2              # intensity — interference pattern
phi3D = np.angle(E3D)              # phase — what GS recovers

# ── 2. MRI K-SPACE RECONSTRUCTION ─────────────────────────────────────────────
# Simulate a phantom (dog brain cross-section)
# k-space = 3D FFT of the object
# Reconstruction = 3D IFFT (same as your 1D GS but in 3D)

def make_ellipsoid(X, Y, Z, cx, cy, cz, rx, ry, rz, val):
    return val * (((X-cx)/rx)**2 + ((Y-cy)/ry)**2 + ((Z-cz)/rz)**2 < 1)

# Phantom: dog skull approximation (nested ellipsoids)
phantom = np.zeros((N, N, N))
phantom += make_ellipsoid(X, Y, Z,  0,    0,   0,   1.0, 0.7, 0.8, 1.0)  # skull
phantom += make_ellipsoid(X, Y, Z,  0,    0,   0,   0.8, 0.55,0.65,0.5)  # brain
phantom += make_ellipsoid(X, Y, Z,  0,   -0.3, 0.4, 0.3, 0.2, 0.2, 0.8) # snout
phantom += make_ellipsoid(X, Y, Z, -0.5,  0.5,-0.1, 0.2, 0.35,0.15,0.4) # ear L
phantom += make_ellipsoid(X, Y, Z,  0.5,  0.5,-0.1, 0.2, 0.35,0.15,0.4) # ear R
phantom  = np.clip(phantom, 0, 2)

# Forward: object -> k-space (what the MRI scanner measures)
kspace   = np.fft.fftn(phantom)
kspace_mag = np.abs(kspace)

# GS phase retrieval in 3D:
# We only have |kspace| (magnitude), not phase
# Add random phase, iterate between measurement constraint + support constraint
rng       = np.random.default_rng(42)
phase_est = rng.uniform(-np.pi, np.pi, kspace.shape)
E_est     = kspace_mag * np.exp(1j * phase_est)    # random start

for iteration in range(40):
    obj_est   = np.fft.ifftn(E_est)
    obj_est   = np.abs(obj_est)                     # support: positivity
    E_est     = np.fft.fftn(obj_est)
    E_est     = kspace_mag * np.exp(1j * np.angle(E_est))  # k-space amplitude

recon = np.abs(np.fft.ifftn(E_est))

# ── 3. PROCEDURAL VOXEL DOG — SDF + CHUNKS ───────────────────────────────────
VOX_SIZE   = 64
CHUNK_SIZE = 16     # Minecraft-style: 16^3 chunks

voxels = np.zeros((VOX_SIZE, VOX_SIZE, VOX_SIZE), dtype=np.uint8)

def sphere_sdf(px, py, pz, cx, cy, cz, r):
    return np.sqrt((px-cx)**2 + (py-cy)**2 + (pz-cz)**2) - r

def box_sdf(px, py, pz, cx, cy, cz, hx, hy, hz):
    dx = np.abs(px-cx) - hx
    dy = np.abs(py-cy) - hy
    dz = np.abs(pz-cz) - hz
    return np.maximum(dx, np.maximum(dy, dz))

# Voxel grid coordinates
gx = np.arange(VOX_SIZE)
gy = np.arange(VOX_SIZE)
gz = np.arange(VOX_SIZE)
GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing='ij')
# Normalise to [-1, 1]
nx = (GX / VOX_SIZE) * 2 - 1
ny = (GY / VOX_SIZE) * 2 - 1
nz = (GZ / VOX_SIZE) * 2 - 1

# Dog SDF — union of primitives (min of SDFs)
body   = box_sdf(nx, ny, nz,  0,   -0.1, 0,    0.35, 0.25, 0.55)
head   = sphere_sdf(nx, ny, nz, 0,   0.25, 0.55, 0.28)
snout  = box_sdf(nx, ny, nz,  0,   0.10, 0.78,  0.12, 0.10, 0.15)
ear_l  = sphere_sdf(nx, ny, nz,-0.22, 0.40, 0.52, 0.14)
ear_r  = sphere_sdf(nx, ny, nz, 0.22, 0.40, 0.52, 0.14)
leg_fl = box_sdf(nx, ny, nz, -0.22,-0.55, 0.30,  0.08, 0.28, 0.08)
leg_fr = box_sdf(nx, ny, nz,  0.22,-0.55, 0.30,  0.08, 0.28, 0.08)
leg_rl = box_sdf(nx, ny, nz, -0.22,-0.55,-0.30,  0.08, 0.28, 0.08)
leg_rr = box_sdf(nx, ny, nz,  0.22,-0.55,-0.30,  0.08, 0.28, 0.08)
tail   = sphere_sdf(nx, ny, nz, 0,   0.10,-0.72, 0.10)

dog_sdf = np.minimum(body,
          np.minimum(head,
          np.minimum(snout,
          np.minimum(ear_l,
          np.minimum(ear_r,
          np.minimum(leg_fl,
          np.minimum(leg_fr,
          np.minimum(leg_rl,
          np.minimum(leg_rr, tail)))))))))

# Voxel = inside SDF (negative value)
inside = dog_sdf < 0.0

# Color palette (golden doodle)
BODY_COLOR  = 5    # golden
SURFACE_COLOR = 6  # lighter gold — thin film interference on surface

# Chunk processing — like Minecraft
n_chunks = VOX_SIZE // CHUNK_SIZE
for cx in range(n_chunks):
    for cy in range(n_chunks):
        for cz in range(n_chunks):
            # Chunk bounds
            x0,x1 = cx*CHUNK_SIZE, (cx+1)*CHUNK_SIZE
            y0,y1 = cy*CHUNK_SIZE, (cy+1)*CHUNK_SIZE
            z0,z1 = cz*CHUNK_SIZE, (cz+1)*CHUNK_SIZE

            chunk_inside = inside[x0:x1, y0:y1, z0:z1]
            if not chunk_inside.any():
                continue           # skip empty chunks (culling)

            voxels[x0:x1, y0:y1, z0:z1] = np.where(chunk_inside,
                                                     BODY_COLOR, 0)

# Surface voxels — thin film: voxels adjacent to air get interference color
from scipy.ndimage import binary_erosion
interior = binary_erosion(inside, iterations=2)
surface  = inside & ~interior
voxels[surface] = SURFACE_COLOR

print(f"Dog voxels: {inside.sum()} total, {surface.sum()} surface")
print(f"Chunks processed: {n_chunks**3}, non-empty: "
      f"{sum(1 for cx in range(n_chunks) for cy in range(n_chunks) for cz in range(n_chunks) if inside[cx*CHUNK_SIZE:(cx+1)*CHUNK_SIZE, cy*CHUNK_SIZE:(cy+1)*CHUNK_SIZE, cz*CHUNK_SIZE:(cz+1)*CHUNK_SIZE].any())}")

# ── 4. SAVE .VOX (MagicaVoxel format) ────────────────────────────────────────
# Palette: golden doodle colors
PALETTE = [0]*256
PALETTE[5] = 0xFFB830FF    # golden body (ABGR)
PALETTE[6] = 0xFFD878FF    # light gold surface (thin film)
PALETTE[1] = 0xFF000000    # black (nose)

def write_vox(path, voxels, palette):
    """Write MagicaVoxel .vox file — opens in MagicaVoxel, Blender, Godot."""
    sx, sy, sz = voxels.shape
    filled = [(x, y, z, int(voxels[x,y,z]))
              for x in range(sx) for y in range(sy) for z in range(sz)
              if voxels[x,y,z] > 0]

    with open(path, 'wb') as f:
        def chunk(tag, content):
            return tag + struct.pack('<II', len(content), 0) + content

        size_c  = chunk(b'SIZE',
                        struct.pack('<III', sx, sy, sz))
        xyzi_c  = chunk(b'XYZI',
                        struct.pack('<I', len(filled)) +
                        b''.join(struct.pack('BBBB', x,y,z,i)
                                 for x,y,z,i in filled))
        rgba_c  = chunk(b'RGBA',
                        b''.join(struct.pack('<I', palette[i] if i < len(palette) else 0)
                                 for i in range(256)))

        main_content = size_c + xyzi_c + rgba_c
        main_c  = chunk(b'MAIN', main_content)

        f.write(b'VOX ' + struct.pack('<I', 150))
        f.write(main_c)

vox_path = _DOCS / "dog.vox"
write_vox(vox_path, voxels, PALETTE)
print(f"Saved {vox_path}  ({vox_path.stat().st_size//1024} KB)")

# ── 5. PLOT — 4 panels ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor("#0a0a0f")

# Panel 1: 3D wave interference (mid slice)
ax1 = fig.add_subplot(2,2,1)
ax1.set_facecolor("#0a0a0f")
mid = N//2
im1 = ax1.imshow(I3D[:,:,mid], cmap='plasma', origin='lower')
ax1.set_title("e^(i*k*r) interference |E|^2\nXY slice at z=0",
              color='white', fontsize=9)
ax1.set_xlabel("x", color='white'); ax1.set_ylabel("y", color='white')
ax1.tick_params(colors='white', labelsize=7)
plt.colorbar(im1, ax=ax1).ax.yaxis.set_tick_params(color='white')

# Panel 2: MRI k-space (log magnitude)
ax2 = fig.add_subplot(2,2,2)
ax2.set_facecolor("#0a0a0f")
kshow = np.log1p(np.abs(np.fft.fftshift(kspace))[:,:,mid])
im2 = ax2.imshow(kshow, cmap='hot', origin='lower')
ax2.set_title("MRI k-space  log|S(kx,ky)|\n(Fourier domain of dog phantom)",
              color='white', fontsize=9)
ax2.tick_params(colors='white', labelsize=7)
plt.colorbar(im2, ax=ax2).ax.yaxis.set_tick_params(color='white')

# Panel 3: GS reconstruction vs truth
ax3 = fig.add_subplot(2,2,3)
ax3.set_facecolor("#0a0a0f")
ax3.imshow(phantom[:,:,mid], cmap='gray', origin='lower', alpha=0.5, label='truth')
ax3.imshow(recon[:,:,mid],   cmap='plasma', origin='lower', alpha=0.5)
ax3.set_title("MRI GS reconstruction (40 iter)\ngray=truth  plasma=GS estimate",
              color='white', fontsize=9)
ax3.tick_params(colors='white', labelsize=7)

# Panel 4: voxel dog — 3 slices
ax4 = fig.add_subplot(2,2,4)
ax4.set_facecolor("#0a0a0f")
mid_v = VOX_SIZE//2
dog_slice = voxels[:,mid_v,:]
colored = np.zeros((*dog_slice.shape, 3), dtype=np.uint8)
colored[dog_slice==5] = [210,150, 40]   # golden body
colored[dog_slice==6] = [255,220,120]   # light surface
ax4.imshow(colored.transpose(1,0,2), origin='lower')
ax4.set_title(f"Voxel golden doodle XZ slice\n"
              f"{inside.sum()} filled voxels, "
              f"{VOX_SIZE//CHUNK_SIZE}^3 chunks, "
              f"surface=thin film color",
              color='white', fontsize=9)
ax4.tick_params(colors='white', labelsize=7)

plt.suptitle("3D Wave Physics  |  MRI k-space GS  |  Voxel Dog",
             color='#00d4ff', fontsize=11)
plt.tight_layout()
out = _DOCS / "voxel_dog_mri.png"
plt.savefig(out, dpi=140, facecolor="#0a0a0f", bbox_inches='tight')
plt.close()
print(f"Saved {out}")

# Physics summary
print("\n---- Wave / MRI / Voxel Physics --------------------------------")
print("E(r) = exp(i*k0*(x+y+z))/3  symmetric superposition")
print("|E|^2 = interference pattern  (thin film on dog surface)")
print("k-space = FFT3(phantom)       MRI forward model")
print("GS 3D:  E_est = |kspace|*exp(i*angle(FFT3(|IFFT3(E_est)|)))")
print("Chunk culling: skip empty 16^3 blocks (Minecraft trick)")
print("SDF dog: union of 10 primitives, erosion -> surface layer")
print("----------------------------------------------------------------")
