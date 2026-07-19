"""Proposed extension: 3D (volumetric, depth-resolved) STEAM microscopy.

STATUS, checked honestly against the literature (WebSearch, 2026): standard
STEAM/time-stretch microscopy is 2D -- Goda et al.'s real design uses a
diffraction GRATING to spectrally encode one transverse axis (x) and a
VIPA (virtually imaged phased array) etalon to spectrally encode the
orthogonal axis (y), giving a full 2D image per laser pulse (see
dgs/steam_imaging.py for the 1D-per-line version already in this repo).
No published "3D STEAM" with depth (z) encoding turned up in this search
-- so what follows is a PROPOSED combination of two independently real,
established techniques, not a claim that this exists in the literature:

  1. Goda's real grating+VIPA 2D spectral encoding (x, y) -- unchanged.
  2. Chromatic-confocal depth sensing (a real, established technique used
     in industrial profilometry and some confocal microscopes): a lens
     with deliberate axial chromatic aberration focuses different
     wavelengths at different depths, so the wavelength that comes back
     in best focus (highest returned signal) tells you the local surface
     height/depth -- no additional detector or mechanical z-scan needed.

The proposed trick: put the xy-image band and the z-depth band in two
NON-OVERLAPPING wavelength sub-ranges of the same broadband pulse. Both
still go through the SAME dispersive fiber, so the same H(f)=exp(i*pi*D*f^2)
(see dgs.gs_core, dgs.steam_imaging) maps each sub-band to its own,
non-overlapping TIME window. A single time-domain ADC capture then
contains the 2D image in one time slice and the depth-per-line-position
signal in a separate later time slice -- demultiplexed by time window,
not by a second detector. This would need real experimental validation;
this module only checks that the numbers are physically self-consistent
and land in realistic device ranges.

Directly relevant to depth-resolved imaging of circulating tumor cells /
tissue microstructure (STEAM's own real cancer-cell-detection use case,
dgs/steam_imaging.py's ULTRAFAST_PHENOMENA["cancer_cell_in_blood"]) --
adding z would let it flag not just THAT a cell passed by, but its
approximate shape/height profile.
"""

import numpy as np

from dgs.photonic_vs_electronic_delay import dispersion_induced_delay_spread_s
from dgs.steam_imaging import time_stretch_pulse


def chromatic_confocal_depth_um(wavelength_nm, wavelength0_nm, axial_dispersion_nm_per_um):
    """Real chromatic-confocal physics: a lens with axial chromatic
    aberration focuses wavelength lambda at depth z = (lambda - lambda0)
    / axial_dispersion. axial_dispersion (nm/um) is a property of the
    chromatic objective's design -- how many nm of wavelength shift
    correspond to 1 um of focal depth shift."""
    if axial_dispersion_nm_per_um <= 0:
        raise ValueError("axial_dispersion_nm_per_um must be positive")
    return (wavelength_nm - wavelength0_nm) / axial_dispersion_nm_per_um


def depth_resolution_um(spectral_resolution_nm, axial_dispersion_nm_per_um):
    """The finest depth step actually resolvable: set by how finely the
    spectrometer/VIPA stage can resolve wavelength, converted to depth
    via the chromatic lens's axial dispersion."""
    if spectral_resolution_nm <= 0:
        raise ValueError("spectral_resolution_nm must be positive")
    if axial_dispersion_nm_per_um <= 0:
        raise ValueError("axial_dispersion_nm_per_um must be positive")
    return spectral_resolution_nm / axial_dispersion_nm_per_um


def depth_range_um(z_band_bandwidth_nm, axial_dispersion_nm_per_um):
    """Total depth range covered by the z-encoding sub-band's bandwidth."""
    if z_band_bandwidth_nm <= 0:
        raise ValueError("z_band_bandwidth_nm must be positive")
    if axial_dispersion_nm_per_um <= 0:
        raise ValueError("axial_dispersion_nm_per_um must be positive")
    return z_band_bandwidth_nm / axial_dispersion_nm_per_um


def split_spectral_budget(total_bandwidth_nm, xy_band_nm):
    """Partition the pulse's total optical bandwidth into a non-
    overlapping xy-image band (grating+VIPA, real Goda scheme) and a
    z-depth band (chromatic confocal). Both bands must be positive and
    fit within the total -- if not, the two encodings collide and can't
    be time-demultiplexed cleanly."""
    if total_bandwidth_nm <= 0:
        raise ValueError("total_bandwidth_nm must be positive")
    if xy_band_nm <= 0:
        raise ValueError("xy_band_nm must be positive")
    z_band_nm = total_bandwidth_nm - xy_band_nm
    if z_band_nm <= 0:
        raise ValueError("xy_band_nm must leave room for a positive z_band "
                          "(xy_band_nm must be < total_bandwidth_nm)")
    return xy_band_nm, z_band_nm


def frame_time_budget(D_ps_per_nm, xy_band_nm, z_band_nm):
    """Both spectral bands pass through the SAME dispersive fiber (same
    D), so each occupies its own, non-overlapping slice of the stretched
    pulse's time axis -- reuses dgs.photonic_vs_electronic_delay's
    already-verified D*delta_lambda group-delay-spread formula rather
    than re-deriving it. Returns each band's time window and the total
    per-frame time (which sets the achievable frame rate)."""
    T_xy = dispersion_induced_delay_spread_s(D_ps_per_nm, xy_band_nm)
    T_z = dispersion_induced_delay_spread_s(D_ps_per_nm, z_band_nm)
    T_total = T_xy + T_z
    return {
        "T_xy_ns": T_xy * 1e9,
        "T_z_ns": T_z * 1e9,
        "T_total_ns": T_total * 1e9,
        "max_frame_rate_hz": 1.0 / T_total,
    }


if __name__ == "__main__":
    print("=== Proposed 3D (depth-resolved) STEAM: xy image + z depth, time-multiplexed ===\n")

    total_bandwidth_nm = 60.0
    xy_band_nm = 40.0   # same 2D-image bandwidth used in dgs.photonic_vs_electronic_delay's demo
    D_ps_per_nm = 800.0
    axial_dispersion_nm_per_um = 0.5   # representative chromatic-confocal lens design
    spectral_resolution_nm = 0.05      # representative VIPA/grating resolution

    xy_band, z_band = split_spectral_budget(total_bandwidth_nm, xy_band_nm)
    print(f"spectral budget: {total_bandwidth_nm} nm total -> "
          f"{xy_band} nm (xy image) + {z_band} nm (z depth)\n")

    timing = frame_time_budget(D_ps_per_nm, xy_band, z_band)
    print(f"xy-image time window:   {timing['T_xy_ns']:.1f} ns")
    print(f"z-depth time window:    {timing['T_z_ns']:.1f} ns")
    print(f"total per-frame window: {timing['T_total_ns']:.1f} ns")
    print(f"implied max frame rate: {timing['max_frame_rate_hz']/1e6:.1f} Mfps "
          f"(real STEAM: 36 Mfps -- same order of magnitude)\n")

    z_res = depth_resolution_um(spectral_resolution_nm, axial_dispersion_nm_per_um)
    z_range = depth_range_um(z_band, axial_dispersion_nm_per_um)
    print(f"depth resolution:  {z_res:.3f} um")
    print(f"depth range:       {z_range:.1f} um "
          f"(order of a single circulating tumor cell / small cell cluster)\n")

    # confirm chromatic_confocal_depth_um is the correct inverse relation
    lambda0 = 1550.0
    test_wavelength = lambda0 + axial_dispersion_nm_per_um * 10.0   # should map to z=10um
    z_check = chromatic_confocal_depth_um(test_wavelength, lambda0, axial_dispersion_nm_per_um)
    print(f"sanity check: wavelength shifted for z=10um -> recovered z = {z_check:.3f} um")

    print("\n=== The point ===")
    print("Real components (grating+VIPA 2D encoding, chromatic-confocal depth,")
    print("dispersive time-stretch) combined via non-overlapping spectral bands +")
    print("time-domain demultiplexing. Depth range/resolution land at the scale of")
    print("a single cell -- directly relevant to STEAM's own cancer-cell-in-blood")
    print("use case (dgs.steam_imaging), now with a shape/height cue, not just a")
    print("flag that a cell passed by. This is a proposed architecture, NOT a")
    print("verified experimental result -- treat it as a research direction to")
    print("check against real literature before pursuing further.")
