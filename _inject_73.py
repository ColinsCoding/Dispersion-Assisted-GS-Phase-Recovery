"""
_inject_73.py — inject §73 into phase_retrieval.ipynb
======================================================
§73 : 3-D Optical Voxel Hash · Energy Minimisation · LSH
  Cell 105 (markdown) — theory: H_TV + H_pc, polynomial hash, LSH projections
  Cell 106 (code)     — OpticalHash3D + OpticalLSH demo, convergence plots

Run:  python _inject_73.py
"""

import json, pathlib, sys

NB = pathlib.Path("phase_retrieval.ipynb")
if not NB.exists():
    sys.exit(f"[ERROR] {NB} not found — run from repo root")

with NB.open("r", encoding="utf-8") as f:
    nb = json.load(f)

# ── Cell 105 — Markdown §73 ──────────────────────────────────────────────
md73 = r"""## §73 · 3-D Optical Voxel Hash · Energy Minimisation · LSH

### Motivation

An optical field $E(x,y,\lambda)$ is a **3-D complex function** over the transverse plane $(x,y)$ and wavelength $\lambda$.
Dense storage scales as $N_x \times N_y \times N_\lambda$ — prohibitive for wide-band, wide-field systems.
A **sparse voxel hash** with $O(1)$ insert/query collapses this to the set of occupied voxels.

---

### Polynomial Spatial Hash

Voxel key: $(i_x, i_y, i_\lambda) = \bigl(\text{round}(x/\Delta x),\; \text{round}(y/\Delta y),\; \text{round}(\lambda/\Delta\lambda)\bigr)$

$$h(i_x, i_y, i_\lambda) = \bigl(i_x \cdot p_1 \;\oplus\; i_y \cdot p_2 \;\oplus\; i_\lambda \cdot p_3\bigr) \bmod N_{\text{buckets}}$$

with primes $p_1 = 73\,856\,093$, $p_2 = 19\,349\,669$, $p_3 = 83\,492\,791$.

---

### Energy Functional

$$H = \underbrace{\alpha \sum_{\langle i,j\rangle} |E_i - E_j|^2}_{H_{\text{TV}} : \text{total variation}} + \underbrace{\beta \sum_k \bigl(|E_k|-1\bigr)^2}_{H_{\text{pc}} : \text{TD-GS phase constraint}}$$

**$H_{\text{TV}}$** — penalises abrupt field amplitude / phase jumps between adjacent voxels
**$H_{\text{pc}}$** — the Gerchberg–Saxton pure-phase constraint $|T|=1$ written as a differentiable loss

**Wirtinger gradient:**

$$\frac{\partial H}{\partial E_k^*} = \alpha \sum_{j \in \mathcal{N}(k)} (E_k - E_j) + \beta \frac{|E_k|-1}{|E_k|} E_k$$

Gradient descent $E_k \leftarrow E_k - \eta \,\nabla_{E_k^*} H$ drives the field toward a smooth,
unit-amplitude wavefront — the **TD-GS solution** from first principles.

---

### Locality-Sensitive Hashing (LSH)

Feature vector per sample: $\mathbf{v} = [|E|,\;\text{Re}(E),\;\text{Im}(E),\;x,\;y,\;\lambda]$

Random projection family:
$$h_t(\mathbf{v}) = \text{sign}(\mathbf{A}_t \mathbf{v} + \mathbf{b}_t) \in \{0,1\}^{n_{\text{bits}}}$$

Collision probability for two vectors: $\Pr[\text{collide}] = 1 - \frac{\arccos(\cos\theta)}{\pi}$
$\Rightarrow$ nearby field samples land in the same bucket with high probability.
Multiple independent tables boost recall; query **unions** candidate sets.

---

**Physics × AI × CS bridge:**
- Physics: Hamiltonian $H$, Wirtinger calculus, optical wavefront constraint
- AI: gradient-based optimisation, energy-based model (EBM)
- CS: hash table, LSH, $O(1)$ approximate nearest-neighbour retrieval
"""

# ── Cell 106 — Code §73 ──────────────────────────────────────────────────
code73 = r"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PI = np.pi
_HASH_PRIMES = (73856093, 19349669, 83492791)

# ─────────────────────────────────────────────────────────────────────────
# OpticalHash3D
# ─────────────────────────────────────────────────────────────────────────
class OpticalHash3D:
    '''Sparse 3-D voxel hash for complex optical field E(x,y,lambda).'''
    _NEIGH6 = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

    def __init__(self, dx_um=1.0, dy_um=1.0, dlam_nm=0.1):
        self.dx = float(dx_um); self.dy = float(dy_um); self.dl = float(dlam_nm)
        self._store = {}

    def _key(self, x, y, lam):
        return (int(round(x/self.dx)), int(round(y/self.dy)), int(round(lam/self.dl)))

    def insert(self, x, y, lam, E):
        self._store[self._key(x, y, lam)] = complex(E)

    def bulk_insert(self, xs, ys, ls, Es):
        for x,y,l,E in zip(xs,ys,ls,Es): self.insert(x,y,l,E)

    def __len__(self): return len(self._store)

    def keys_array(self):
        return np.array(list(self._store.keys()), dtype=int) if self._store else np.empty((0,3),int)

    def values_array(self):
        return np.array(list(self._store.values()), dtype=complex)

    def coords_and_field(self):
        if not self._store: return [np.array([])]*5
        k = self.keys_array(); v = self.values_array()
        return k[:,0]*self.dx, k[:,1]*self.dy, k[:,2]*self.dl, np.abs(v), np.angle(v)

    def tv_energy(self):
        H = 0.0
        for (ix,iy,il),E in self._store.items():
            for dx,dy,dl in self._NEIGH6:
                nb = self._store.get((ix+dx,iy+dy,il+dl))
                if nb is not None: H += abs(E-nb)**2
        return float(H*0.5)

    def phase_energy(self):
        v = self.values_array()
        return float(np.sum((np.abs(v)-1.0)**2)) if len(v) else 0.0

    def minimise(self, n_iter=80, lr=0.04, alpha=0.1, beta=0.5):
        history = []; keys = list(self._store.keys())
        for it in range(n_iter):
            grads = {}
            for k in keys:
                Ek = self._store[k]
                g_tv = sum((Ek - self._store[(k[0]+dx,k[1]+dy,k[2]+dl)])
                           for dx,dy,dl in self._NEIGH6
                           if (k[0]+dx,k[1]+dy,k[2]+dl) in self._store)
                amp = abs(Ek)+1e-12
                g_pc = (amp-1.0)*Ek/amp
                grads[k] = alpha*g_tv + beta*g_pc
            for k in keys: self._store[k] -= lr*grads[k]
            htv = self.tv_energy(); hpc = self.phase_energy()
            history.append((it, htv, hpc, alpha*htv+beta*hpc))
        return history

    def bucket_histogram(self, N=32):
        counts = np.zeros(N, dtype=int); p1,p2,p3 = _HASH_PRIMES
        for k in self._store:
            counts[int((k[0]*p1^k[1]*p2^k[2]*p3)%N)] += 1
        return counts


# ─────────────────────────────────────────────────────────────────────────
# OpticalLSH
# ─────────────────────────────────────────────────────────────────────────
class OpticalLSH:
    '''Locality-Sensitive Hashing for optical field samples.'''
    def __init__(self, n_bits=10, n_tables=4, seed=0):
        rng = np.random.default_rng(seed); d = 6
        self.A=[rng.standard_normal((n_bits,d)) for _ in range(n_tables)]
        self.b=[rng.uniform(0,2*PI,n_bits) for _ in range(n_tables)]
        self.tables=[{} for _ in range(n_tables)]; self._data=[]; self.n_tables=n_tables

    def _feat(self,x,y,lam,E): return np.array([abs(E),E.real,E.imag,x,y,lam])
    def _bucket(self,feat,t): return tuple((self.A[t]@feat+self.b[t]>=0).astype(int))

    def insert(self,x,y,lam,E):
        feat=self._feat(x,y,lam,E); idx=len(self._data); self._data.append(((x,y,lam),feat))
        for t in range(self.n_tables): self.tables[t].setdefault(self._bucket(feat,t),[]).append(idx)

    def collision_rate(self):
        n_over=n_tot=0
        for t in self.tables:
            for v in t.values(): n_tot+=1; n_over+=(1 if len(v)>1 else 0)
        return n_over/max(n_tot,1)


# ─────────────────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(3)
N   = 256

x_um   = rng.uniform(-10, 10, N)
y_um   = rng.uniform(-10, 10, N)
lam_nm = rng.uniform(1540, 1560, N)

w0    = 5.0
amp   = np.exp(-(x_um**2+y_um**2)/w0**2)
phi   = rng.normal(0,0.8,N) + 0.03*(lam_nm-1550) + 0.002*(x_um**2+y_um**2)
E_meas = amp*np.exp(1j*phi) + 0.12*(rng.standard_normal(N)+1j*rng.standard_normal(N))

oh = OpticalHash3D(dx_um=0.5, dy_um=0.5, dlam_nm=0.5)
oh.bulk_insert(x_um, y_um, lam_nm, E_meas)
H_init = oh.tv_energy()+0.4*oh.phase_energy()

history = oh.minimise(n_iter=80, lr=0.035, alpha=0.1, beta=0.4)
H_final = oh.tv_energy()+0.4*oh.phase_energy()
print(f"OpticalHash3D: {len(oh)} voxels  |  H: {H_init:.3f} → {H_final:.3f}  "
      f"(↓{100*(1-H_final/H_init):.1f}%)")

lsh = OpticalLSH(n_bits=10, n_tables=4, seed=1)
for xi,yi,li,Ei in zip(x_um,y_um,lam_nm,E_meas): lsh.insert(xi,yi,li,Ei)
print(f"OpticalLSH collision rate: {lsh.collision_rate():.2%}")

x_o,y_o,l_o,amp_o,phi_o = oh.coords_and_field()
hist_counts = oh.bucket_histogram(32)
iters = [h[0] for h in history]; H_tv=[h[1] for h in history]
H_pc  = [h[2] for h in history]; H_tot=[h[3] for h in history]

# ─────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────
FG=(0.04,0.04,0.10); BG=(0.06,0.06,0.14)
def dk(ax,title="",xl="",yl=""):
    ax.set_facecolor(BG)
    for s in ax.spines.values(): s.set_color("#334466")
    ax.tick_params(colors="#99aabb",labelsize=8)
    if title: ax.set_title(title,color="white",fontsize=8.5,fontweight="bold",pad=5)
    if xl: ax.set_xlabel(xl,color="#99aabb",fontsize=8)
    if yl: ax.set_ylabel(yl,color="#99aabb",fontsize=8)

fig = plt.figure(figsize=(14,9)); fig.patch.set_facecolor(FG)
gs  = gridspec.GridSpec(2,3,hspace=0.52,wspace=0.40)

ax0 = fig.add_subplot(gs[0,0])
sc0 = ax0.scatter(x_o,y_o,c=amp_o,cmap="plasma",s=12,alpha=0.75)
fig.colorbar(sc0,ax=ax0,label="|E|").ax.tick_params(colors="#99aabb",labelsize=6)
dk(ax0,f"Wavefront |E(x,y)|  N={len(oh)} voxels","x [μm]","y [μm]")

ax1 = fig.add_subplot(gs[0,1])
sc1 = ax1.scatter(l_o,amp_o,c=phi_o,cmap="hsv",s=9,alpha=0.65,vmin=-PI,vmax=PI)
fig.colorbar(sc1,ax=ax1,label="φ [rad]").ax.tick_params(colors="#99aabb",labelsize=6)
dk(ax1,"Amplitude vs Wavelength |E|(λ)","λ [nm]","|E|")

ax2 = fig.add_subplot(gs[0,2])
cols = plt.cm.cool(np.linspace(0.1,0.9,len(hist_counts)))
ax2.bar(range(len(hist_counts)),hist_counts,color=cols,width=0.9)
ax2.axhline(N/len(hist_counts),color="#ffd040",lw=1.2,ls="--",
            label=f"ideal={N/len(hist_counts):.1f}")
ax2.legend(fontsize=7,facecolor=BG,labelcolor="white")
dk(ax2,f"Polynomial Hash Buckets (N=32)\nXOR: h=({N} pts, primes p₁p₂p₃)%32",
   "bucket","count")

ax3 = fig.add_subplot(gs[1,0:2])
ax3.semilogy(iters,H_tot,color="#50d8ff",lw=2.2,
             label=f"H_total  {H_init:.2f}→{H_final:.2f}  (↓{100*(1-H_final/H_init):.1f}%)")
ax3.semilogy(iters,H_tv, color="#ffd040",lw=1.5,ls="--",label="H_TV  (total variation)")
ax3.semilogy(iters,H_pc, color="#ff3278",lw=1.5,ls=":" ,label="H_pc  (phase constraint)")
ax3.legend(fontsize=8,facecolor=BG,labelcolor="white")
ax3.set_xlim(0,len(iters)-1)
dk(ax3,"Energy Minimisation — Gradient Descent on H = 0.1·H_TV + 0.4·H_pc\n"
       "∂H/∂E*_k = α·Σ(E_k−E_j) + β·(|E_k|−1)·E_k/|E_k|",
   "iteration","H (log scale)")

ax4 = fig.add_subplot(gs[1,2])
sc4 = ax4.scatter(x_o,l_o,c=amp_o,cmap="viridis",s=8,alpha=0.6)
fig.colorbar(sc4,ax=ax4,label="|E|").ax.tick_params(colors="#99aabb",labelsize=6)
dk(ax4,"3-D Projection: (x, λ) plane","x [μm]","λ [nm]")

fig.suptitle("§73  3-D Optical Voxel Hash · Energy Minimisation · LSH\n"
             "Physics ↔ CS ↔ AI:  H(TV+phase) → Wirtinger gradient descent → sparse O(1) retrieval",
             color="white",fontsize=10,fontweight="bold")
plt.tight_layout()
plt.show()
print(f"\nPhysics summary\n"
      f"  Voxels stored   : {len(oh)}\n"
      f"  H_TV  (initial) : {H_tv[0]:.4f}  →  {H_tv[-1]:.4f}\n"
      f"  H_pc  (initial) : {H_pc[0]:.4f}  →  {H_pc[-1]:.4f}\n"
      f"  H_tot (initial) : {H_tot[0]:.4f}  →  {H_tot[-1]:.4f}\n"
      f"  Reduction       : {100*(1-H_tot[-1]/H_tot[0]):.1f}%\n"
      f"  LSH collision   : {lsh.collision_rate():.2%}\n"
      f"  Hash load ratio : {max(hist_counts)/max(N/32,1):.2f}×  (max bucket / ideal)")
"""

# ── Build notebook cells ─────────────────────────────────────────────────
def md_cell(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": src.strip().splitlines(keepends=True)}

def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.strip().splitlines(keepends=True)}

new_cells = [md_cell(md73), code_cell(code73)]

before = len(nb["cells"])
nb["cells"].extend(new_cells)
after  = len(nb["cells"])

with NB.open("w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"[OK] {NB.name}: {before} -> {after} cells  (+{after-before} S73 cells)")
