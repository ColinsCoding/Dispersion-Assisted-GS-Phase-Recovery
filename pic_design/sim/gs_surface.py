import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N      = 512
n_iter = 60
phi_true = np.random.randn(N) * np.pi

# ── GS loop, record (residual_1, residual_2, iteration) ──────────────────────
def gs_step(E, I_target, D_ps2, dt=1e-12, lam=1550e-9):
    nu  = np.fft.fftfreq(len(E), d=dt)
    H   = np.exp(1j * np.pi * D_ps2 * (nu * 1e-12)**2)
    Ef  = np.fft.fft(E) * H
    Ef  = np.sqrt(np.abs(I_target)) * np.exp(1j * np.angle(Ef))
    return np.fft.ifft(Ef / H)

E      = np.exp(1j * phi_true)               # true field
I1     = np.abs(np.fft.fft(E * np.exp(1j * np.pi * -600 * np.fft.fftfreq(N)**2)))**2
I2     = np.abs(np.fft.fft(E * np.exp(1j * np.pi * -900 * np.fft.fftfreq(N)**2)))**2

E_est  = np.ones(N, dtype=complex)           # start from flat phase
r1_trace, r2_trace = [], []

for k in range(n_iter):
    E_est  = gs_step(E_est, I1, -600)
    E_est  = gs_step(E_est, I2, -900)
    r1     = np.mean((np.abs(np.fft.fft(E_est))**2 - I1)**2)
    r2     = np.mean((np.abs(np.fft.fft(E_est))**2 - I2)**2)
    r1_trace.append(r1)
    r2_trace.append(r2)

# ── 3D surface: residual_1 × residual_2 landscape, GS path as red line ───────
R1g, R2g = np.meshgrid(
    np.linspace(0, max(r1_trace)*1.1, 50),
    np.linspace(0, max(r2_trace)*1.1, 50)
)
# Energy landscape: bowl — minimum at (0,0)
Z = np.log1p(R1g + R2g)

fig = plt.figure(figsize=(11, 7))
ax  = fig.add_subplot(111, projection='3d')

ax.plot_surface(R1g, R2g, Z, cmap='Blues', alpha=0.4, linewidth=0)

# GS trajectory as a curve on the surface
zpath = np.log1p(np.array(r1_trace) + np.array(r2_trace))
ax.plot(r1_trace, r2_trace, zpath,
        color='red', lw=2.5, label='GS convergence path')

# Mark start and end
ax.scatter([r1_trace[0]],  [r2_trace[0]],  [zpath[0]],
           color='orange', s=80, zorder=5, label='Start')
ax.scatter([r1_trace[-1]], [r2_trace[-1]], [zpath[-1]],
           color='lime',   s=80, zorder=5, label=f'End (iter {n_iter})')

ax.set_xlabel('Residual arm 1')
ax.set_ylabel('Residual arm 2')
ax.set_zlabel('log(R1+R2)')
ax.set_title('Gerchberg-Saxton on the Error Surface')
ax.legend()
plt.tight_layout()
plt.savefig('gs_surface.png', dpi=150)
print('Saved gs_surface.png')