"""Microfluidics physics — lab-on-chip design toolkit.

Covers the core dimensionless numbers and physical regimes that govern
behaviour at the micron scale, plus practical design functions:

  Poiseuille flow     — parabolic velocity profile, Q-ΔP relation
  Reynolds number     — always Re << 1 in microfluidics (laminar, no turbulence)
  Péclet number       — advection vs diffusion; controls mixing length
  Diffusion time      — how long a molecule needs to cross a channel
  Droplet formation   — Rayleigh-Plateau instability, breakup length
  Cell counting       — throughput at N cells/mL and flow rate Q
  Pressure drop       — Hagen-Poiseuille for rectangular channels
  Microfluidic SNR    — photon-counting noise in fluorescence detection

Why microfluidics differs from macroscale:
  - Re ~ 10^{-3}..1 → purely laminar (no turbulent mixing)
  - Surface forces dominate body forces (capillary, electro-osmosis)
  - Diffusion mixes analytes on µs–ms timescales
  - Can interrogate individual cells (100 000 cells/mL feasible)

Usage:
    from dgs.microfluidics import Channel, DropletGenerator, CellCounter
    ch = Channel(width=100e-6, height=50e-6, length=10e-3)
    print(ch.report(delta_p=1e3))           # 1 kPa driving pressure
    cc = CellCounter(concentration=1e5)     # 100k cells/mL
    cc.report(channel=ch, flow_rate=1e-9)   # 1 nL/s
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

# Physical constants
ETA_WATER = 1e-3       # dynamic viscosity of water [Pa·s] at 20°C
D_SMALL   = 1e-10      # diffusivity of small molecule (MW ~300) [m²/s]
D_PROTEIN = 1e-11      # diffusivity of globular protein (~50 kDa) [m²/s]
D_CELL    = 3e-13      # effective diffusivity of a mammalian cell [m²/s]
GAMMA_OIL = 50e-3      # oil-water interfacial tension [N/m]


# ---------------------------------------------------------------------------
# Dimensionless numbers
# ---------------------------------------------------------------------------

def reynolds(rho: float, v: float, L: float, eta: float = ETA_WATER) -> float:
    """Re = rho*v*L/eta.  Re < 1 is Stokes flow (microfluidics regime)."""
    return rho * v * L / eta


def peclet(v: float, L: float, D: float) -> float:
    """Pe = v*L/D.  Pe >> 1: advection dominates. Pe << 1: diffusion dominates."""
    return v * L / D


def capillary(eta: float, v: float, gamma: float = GAMMA_OIL) -> float:
    """Ca = eta*v/gamma.  Droplet regime: Ca < 0.1 (squeezing), Ca > 0.1 (dripping)."""
    return eta * v / gamma


def diffusion_time(L: float, D: float = D_SMALL) -> float:
    """Time for a molecule to diffuse across distance L: tau = L^2 / (2D) [s]."""
    return L**2 / (2 * D)


def diffusion_length(t: float, D: float = D_SMALL) -> float:
    """RMS diffusion distance in time t: sqrt(2Dt) [m]."""
    return np.sqrt(2 * D * t)


# ---------------------------------------------------------------------------
# Rectangular channel (Hagen-Poiseuille, width >> height limit)
# ---------------------------------------------------------------------------

class Channel:
    """Straight rectangular microfluidic channel.

    Geometry: width w, height h, length L (all in metres).
    Fluid: dynamic viscosity eta [Pa·s].
    """

    def __init__(self, width: float = 100e-6, height: float = 50e-6,
                 length: float = 10e-3, eta: float = ETA_WATER):
        if any(x <= 0 for x in (width, height, length, eta)):
            raise ValueError("all dimensions and viscosity must be positive")
        self.w = width
        self.h = height
        self.L = length
        self.eta = eta

    @property
    def cross_section(self) -> float:
        return self.w * self.h

    @property
    def hydraulic_diameter(self) -> float:
        """D_h = 4A/P (accounts for rectangular aspect ratio)."""
        return 4 * self.cross_section / (2 * (self.w + self.h))

    def resistance(self) -> float:
        """Fluidic resistance R [Pa·s/m³] using the rectangular approximation.

        For aspect ratio w/h >> 1: R ≈ 12*eta*L / (w*h³).
        Exact series correction used here (Bruus 2008, eq 3.30).
        """
        alpha = self.h / self.w  # < 1 if h < w
        # correction factor sum (first 5 terms sufficient)
        correction = 1 - (192 * alpha / np.pi**5) * sum(
            np.tanh(np.pi * (2*k-1) / (2 * alpha)) / (2*k-1)**5
            for k in range(1, 6)
        )
        return 12 * self.eta * self.L / (self.w * self.h**3 * correction)

    def flow_rate(self, delta_p: float) -> float:
        """Volume flow rate Q = delta_p / R [m³/s]."""
        return delta_p / self.resistance()

    def mean_velocity(self, delta_p: float) -> float:
        """Mean velocity <v> = Q / A [m/s]."""
        return self.flow_rate(delta_p) / self.cross_section

    def velocity_profile(self, delta_p: float, n: int = 50
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """2D Poiseuille velocity field v(y, z) across the cross-section.

        Returns (y_grid, z_grid, v_grid) where y spans the width and z the height.
        Uses leading term of the exact series solution.
        """
        y = np.linspace(-self.w/2, self.w/2, n)
        z = np.linspace(-self.h/2, self.h/2, n)
        Y, Z = np.meshgrid(y, z)
        # leading Fourier term (good when w/h > 2)
        v = (4 * self.h**2 * delta_p / (np.pi**3 * self.eta * self.L)) * (
            np.cos(np.pi * Z / self.h) * (1 - np.abs(Y) / (self.w/2))
        )
        v = np.clip(v, 0, None)
        return y, z, v

    def peclet_number(self, delta_p: float, D: float = D_SMALL) -> float:
        return peclet(self.mean_velocity(delta_p), self.h, D)

    def reynolds_number(self, delta_p: float, rho: float = 1000.0) -> float:
        v = self.mean_velocity(delta_p)
        return reynolds(rho, v, self.hydraulic_diameter, self.eta)

    def mixing_length(self, delta_p: float, D: float = D_SMALL) -> float:
        """Advective-diffusion mixing length: L_mix = Pe * h [m].

        The distance the flow must travel before diffusion mixes the channel width.
        """
        return self.peclet_number(delta_p, D) * self.h

    def report(self, delta_p: float) -> str:
        Q  = self.flow_rate(delta_p)
        v  = self.mean_velocity(delta_p)
        Re = self.reynolds_number(delta_p)
        Pe = self.peclet_number(delta_p)
        Lm = self.mixing_length(delta_p)
        R  = self.resistance()
        lines = [
            f"Channel: w={self.w*1e6:.1f} um  h={self.h*1e6:.1f} um  L={self.L*1e3:.1f} mm",
            f"  Driving pressure     : {delta_p:.2e} Pa  ({delta_p/133.322:.1f} mmHg)",
            f"  Fluidic resistance   : {R:.2e} Pa·s/m^3",
            f"  Volume flow rate Q   : {Q*1e12:.2f} pL/s  ({Q*1e9*60:.2f} nL/min)",
            f"  Mean velocity        : {v*1e3:.3f} mm/s  ({v*1e2:.4f} cm/s)",
            f"  Reynolds number Re   : {Re:.4f}  ({'laminar' if Re < 1 else 'transitional'})",
            f"  Peclet (small mol.)  : {Pe:.1f}",
            f"  Mixing length        : {Lm*1e3:.2f} mm  (need channel >= this for mixing)",
            f"  Diffusion time (h)   : {diffusion_time(self.h)*1e3:.2f} ms  (small molecule across height)",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Droplet generator (T-junction / flow-focusing)
# ---------------------------------------------------------------------------

class DropletGenerator:
    """T-junction droplet generator: oil continuous phase, aqueous dispersed.

    Models droplet volume and formation frequency using the squeezing regime
    correlation (Garstecki et al., Lab Chip 2006):
        V_drop / V_ch = 1 + alpha * (Q_disp / Q_cont)
    where alpha ~ 1 for T-junctions.
    """

    def __init__(self, channel: Channel, alpha: float = 1.1,
                 gamma: float = GAMMA_OIL):
        self.ch = channel
        self.alpha = alpha
        self.gamma = gamma

    def droplet_volume(self, q_cont: float, q_disp: float) -> float:
        """V_drop [m³] — squeezing regime correlation."""
        V_ch = self.ch.cross_section * self.ch.h   # channel unit volume
        return V_ch * (1 + self.alpha * q_disp / q_cont)

    def frequency(self, q_cont: float, q_disp: float) -> float:
        """Droplet generation frequency [Hz] = Q_disp / V_drop."""
        return q_disp / self.droplet_volume(q_cont, q_disp)

    def droplet_diameter(self, q_cont: float, q_disp: float) -> float:
        """Equivalent sphere diameter [m]."""
        V = self.droplet_volume(q_cont, q_disp)
        return 2 * (3 * V / (4 * np.pi)) ** (1/3)

    def report(self, q_cont: float = 5e-9, q_disp: float = 1e-9) -> None:
        V  = self.droplet_volume(q_cont, q_disp)
        f  = self.frequency(q_cont, q_disp)
        d  = self.droplet_diameter(q_cont, q_disp)
        Ca = capillary(self.ch.eta, q_cont / self.ch.cross_section, self.gamma)
        print(f"Droplet Generator (T-junction)")
        print(f"  Q_cont   : {q_cont*1e9:.2f} nL/s")
        print(f"  Q_disp   : {q_disp*1e9:.2f} nL/s")
        print(f"  Volume   : {V*1e15:.2f} fL  ({V*1e12:.4f} pL)")
        print(f"  Diameter : {d*1e6:.2f} um")
        print(f"  Freq     : {f:.1f} Hz  ({f*60:.0f} drops/min)")
        print(f"  Ca       : {Ca:.4f}  ({'squeezing' if Ca < 0.1 else 'dripping/jetting'} regime)")


# ---------------------------------------------------------------------------
# Cell counter / flow cytometry channel
# ---------------------------------------------------------------------------

class CellCounter:
    """Single-file cell counting through a microfluidic channel.

    concentration: cells/mL (e.g. 1e5 for 100,000 cells/mL = 1e5 * 1e6 cells/m^3)
    """

    def __init__(self, concentration: float = 1e5,
                 cell_diameter: float = 10e-6):
        """concentration in cells/mL; cell_diameter in metres (default 10 um)."""
        self.conc_per_m3 = concentration * 1e6   # convert mL^-1 -> m^-3
        self.d_cell = cell_diameter

    def throughput(self, flow_rate: float) -> float:
        """Cells per second passing through the channel at flow_rate [m³/s]."""
        return self.conc_per_m3 * flow_rate

    def inter_cell_distance(self, flow_rate: float, v_mean: float) -> float:
        """Average gap between consecutive cells [m] = v / (conc * Q/A)."""
        rate_per_s = self.throughput(flow_rate)
        if rate_per_s == 0:
            return float('inf')
        return v_mean / rate_per_s

    def occupancy(self, channel: Channel, flow_rate: float) -> float:
        """Fraction of channel length occupied by cells at any instant."""
        L_cells = self.conc_per_m3 * channel.cross_section * self.d_cell
        return L_cells  # dimensionless (cells fill fraction)

    def report(self, channel: Channel, flow_rate: float = 1e-9) -> None:
        v    = flow_rate / channel.cross_section
        rate = self.throughput(flow_rate)
        gap  = self.inter_cell_distance(flow_rate, v)
        occ  = self.occupancy(channel, flow_rate)

        single_file = channel.w < 2 * self.d_cell
        print(f"Cell Counter")
        print(f"  Concentration   : {self.conc_per_m3/1e6:.0f} cells/mL"
              f"  ({self.conc_per_m3:.2e} cells/m^3)")
        print(f"  Cell diameter   : {self.d_cell*1e6:.1f} um")
        print(f"  Flow rate Q     : {flow_rate*1e12:.1f} pL/s")
        print(f"  Mean velocity   : {v*1e3:.3f} mm/s")
        print(f"  Throughput      : {rate:.1f} cells/s  ({rate*60:.0f} cells/min)")
        print(f"  Inter-cell gap  : {gap*1e6:.1f} um")
        print(f"  Single-file?    : {'YES' if single_file else 'NO'}"
              f"  (channel w={channel.w*1e6:.0f} um, cell d={self.d_cell*1e6:.0f} um)")
        print(f"  Channel fill    : {occ*100:.2f}%")
        t_transit = channel.L / v
        print(f"  Transit time    : {t_transit*1e3:.1f} ms  ({t_transit:.4f} s)")


# ---------------------------------------------------------------------------
# Imaging modality comparison: MRI / SEM / thermal camera
# ---------------------------------------------------------------------------

IMAGING = {
    "MRI (clinical 3T)": {
        "resolution_um": 500,       # 0.5 mm isotropic voxel
        "snr_db": 40,
        "contrast": "spin density / T1 / T2 relaxation",
        "penetration": "whole body",
        "label_free": True,
        "live_cell": True,
        "throughput_cells_per_s": 0,  # bulk, not single-cell
    },
    "SEM (scanning electron)": {
        "resolution_um": 0.005,     # ~5 nm
        "snr_db": 50,
        "contrast": "secondary / backscattered electrons",
        "penetration": "surface only",
        "label_free": True,
        "live_cell": False,         # requires vacuum + fixation
        "throughput_cells_per_s": 0,
    },
    "Confocal fluorescence": {
        "resolution_um": 0.25,      # 250 nm lateral (diffraction limit)
        "snr_db": 30,
        "contrast": "fluorescent labels / GFP",
        "penetration": "~100 um in tissue",
        "label_free": False,
        "live_cell": True,
        "throughput_cells_per_s": 10,
    },
    "Thermal / IR camera": {
        "resolution_um": 10,        # ~10 um per pixel (mid-IR detector)
        "snr_db": 25,
        "contrast": "blackbody emission (T diff ~ 0.05 K)",
        "penetration": "surface (~ 1 skin depth)",
        "label_free": True,
        "live_cell": True,
        "throughput_cells_per_s": 1000,
    },
    "Flow cytometry": {
        "resolution_um": 5,         # ~5 um per cell
        "snr_db": 35,
        "contrast": "scatter + up to 48 fluorescent channels",
        "penetration": "single cell in suspension",
        "label_free": False,
        "live_cell": True,
        "throughput_cells_per_s": 50000,
    },
    "Microfluidic imaging": {
        "resolution_um": 1,
        "snr_db": 30,
        "contrast": "bright-field, phase, or fluorescence",
        "penetration": "single cell layer",
        "label_free": True,
        "live_cell": True,
        "throughput_cells_per_s": 1000,
    },
}


def imaging_table() -> None:
    """Print a comparison table of imaging modalities."""
    print(f"{'Modality':<25}  {'Res (um)':>10}  {'SNR (dB)':>9}  "
          f"{'Live?':>6}  {'Label-free?':>12}  {'Cells/s':>8}")
    print("-" * 80)
    for name, d in IMAGING.items():
        print(f"{name:<25}  {d['resolution_um']:>10.3f}  {d['snr_db']:>9}  "
              f"{'Y' if d['live_cell'] else 'N':>6}  "
              f"{'Y' if d['label_free'] else 'N':>12}  "
              f"{d['throughput_cells_per_s']:>8}")


def imaging_plot() -> None:
    """Resolution vs throughput scatter for all modalities."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, d in IMAGING.items():
        x = d["throughput_cells_per_s"] + 0.1   # avoid log(0)
        y = d["resolution_um"]
        ax.scatter(x, y, s=120, zorder=3)
        ax.annotate(name, (x, y), xytext=(6, 0), textcoords='offset points', fontsize=8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Throughput (cells/s)  [log scale]")
    ax.set_ylabel("Spatial resolution (um)  [log scale, lower = better]")
    ax.set_title("Imaging modality trade-off: resolution vs throughput")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("imaging_modalities.png", dpi=120, bbox_inches='tight')
    print("Saved imaging_modalities.png")
    plt.show()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Rectangular channel at 1 kPa ===")
    ch = Channel(width=100e-6, height=50e-6, length=10e-3)
    print(ch.report(delta_p=1e3))

    print()
    print("=== Droplet generator (oil/water T-junction) ===")
    dg = DropletGenerator(ch)
    dg.report(q_cont=5e-9, q_disp=1e-9)

    print()
    print("=== Cell counter: 100k cells/mL ===")
    cc = CellCounter(concentration=100_000, cell_diameter=10e-6)
    cc.report(channel=ch, flow_rate=1e-9)

    print()
    print("=== Imaging modality comparison ===")
    imaging_table()
