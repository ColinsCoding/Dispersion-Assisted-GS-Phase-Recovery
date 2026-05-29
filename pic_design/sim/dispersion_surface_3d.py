import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# |E_out(t)|^2 as function of D and time — the dispersion surface
D_vals  = np.linspace(-1200, -300, 80)   # ps^2
t_vals  = np.linspace(-50, 50, 80)       # ps
D, T    = np.meshgrid(D_vals, t_vals)

# Gaussian input pulse, width 10ps
sigma = 10.0
E_in  = np.exp(-T**2 / (2*sigma**2))

# Dispersed output intensity: I(t,D) = |FT{Ẽ·exp(iπDν²)}|²
# In large-D limit: I_out(t,D) ≈ |Ẽ(t/D)|² / |D|
# Approximation: stretched Gaussian
sigma_out = np.sqrt(sigma**2 + (1.0/sigma)**2 * D**2)
I_out = (sigma / np.abs(sigma_out)) * np.exp(-T**2 / (2*sigma_out**2))

fig = plt.figure(figsize=(10, 7))
ax  = fig.add_subplot(111, projection='3d')
ax.plot_surface(D, T, I_out, cmap='plasma', alpha=0.85)
ax.set_xlabel('D (ps^2)')
ax.set_ylabel('t (ps)')
ax.set_zlabel('I_out (norm)')
ax.set_title('Dispersive Fourier Transform Surface\nI(t,D) vs GVD and time')
plt.tight_layout()
plt.savefig('dispersion_surface_3d.png', dpi=150)
plt.show()