"""Free-particle Schrödinger wavepacket via simulator.dispersion.propagate().

Same FFT-multiply-IFFT as fibre dispersion - just with electron constants.

Math
----
Fibre:        d/dz E  = -i (beta_2 / 2) d^2/dt^2 E
              => E(omega, L) = E(omega, 0) * exp(-i beta_2 L omega^2 / 2)

Schrodinger:  i hbar d/dt psi = -(hbar^2 / 2m) d^2/dx^2 psi
              => psi(k, t) = psi(k, 0) * exp(-i hbar t k^2 / (2m))

So to make `propagate(A, t_axis, beta2_L=B)` apply Schrodinger evolution
for time T, we pass:
    t_axis  -> the spatial axis x (units: metres)
    beta2_L -> hbar * T / m            (units: m^2)

For a Gaussian wavepacket of half-width sigma_0, the spread is

    sigma(t) = sigma_0 * sqrt(1 + (t / T_spread)**2)

with T_spread = 2 m sigma_0^2 / hbar.  Below we check the simulated FWHM
against that exact formula.
"""

import numpy as np
import matplotlib.pyplot as plt

from simulator.dispersion import propagate

hbar = 1.054_571_817e-34         # J*s
m_e  = 9.109_383_701e-31         # kg


def main():
    sigma_0 = 1e-9                                # 1 nm initial width
    N       = 4096
    x       = np.linspace(-30 * sigma_0, 30 * sigma_0, N, endpoint=False)
    psi_0   = (np.exp(-x ** 2 / (4 * sigma_0 ** 2)) + 0j)

    T_spread = 2 * m_e * sigma_0 ** 2 / hbar      # natural timescale

    # ---- snapshots
    snap_times = [0.0, 0.5, 1.0, 3.0]             # in units of T_spread
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for tau in snap_times:
        T = tau * T_spread
        psi_T = propagate(psi_0, x, beta2_L=hbar * T / m_e)
        ax[0].plot(x * 1e9, np.abs(psi_T) ** 2,
                    label=f't = {tau:.1f} T_spread')
    ax[0].set_xlabel('x (nm)')
    ax[0].set_ylabel(r'$|\psi(x,t)|^2$')
    ax[0].set_title(f'Free electron wavepacket\n'
                    f'sigma_0 = 1 nm, T_spread = {T_spread*1e15:.2f} fs')
    ax[0].legend()
    ax[0].grid(True)

    # ---- sigma vs t
    t_grid = np.linspace(0, 5 * T_spread, 60)
    sigma_sim = np.zeros_like(t_grid)
    for k, T in enumerate(t_grid):
        psi_T = propagate(psi_0, x, beta2_L=hbar * T / m_e)
        I = np.abs(psi_T) ** 2
        I /= I.max()
        idx = np.where(I > 0.5)[0]
        fwhm = (x[idx[-1]] - x[idx[0]]) if len(idx) else 0.0
        sigma_sim[k] = fwhm / (2 * np.sqrt(2 * np.log(2)))
    sigma_th = sigma_0 * np.sqrt(1 + (t_grid / T_spread) ** 2)

    ax[1].plot(t_grid * 1e15, sigma_sim * 1e9, 'o-', label='simulated')
    ax[1].plot(t_grid * 1e15, sigma_th  * 1e9, 'k--', label='analytic')
    ax[1].set_xlabel('t (fs)')
    ax[1].set_ylabel(r'$\sigma$ (nm)')
    ax[1].set_title('Wavepacket width vs time')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig('quantum_wavepacket_demo.png', dpi=120)

    print(f'T_spread = {T_spread*1e15:.3f} fs')
    print(f'sigma(0)       = {sigma_sim[0]*1e9:.3f} nm   (analytic 1.000)')
    print(f'sigma(T_spread)= {sigma_sim[len(t_grid)//5]*1e9:.3f} nm   '
          f'(analytic {np.sqrt(2):.3f})')
    print('Saved quantum_wavepacket_demo.png')


if __name__ == '__main__':
    main()
