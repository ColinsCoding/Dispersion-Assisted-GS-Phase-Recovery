"""Test the proposed 3D STEAM depth-encoding architecture: chromatic-
confocal depth physics, spectral-budget splitting, and time-domain
multiplexing via the same dispersion used throughout dgs.steam_imaging /
dgs.gs_core."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import steam_3d_depth_encoding as s3d

# 1. chromatic_confocal_depth_um is a correct linear inverse: a wavelength
#    shift of (axial_dispersion * z) recovers exactly z
lambda0 = 1550.0
axial_disp = 0.5
for z_true in [0.0, 5.0, 10.0, -20.0]:
    wl = lambda0 + axial_disp * z_true
    z_rec = s3d.chromatic_confocal_depth_um(wl, lambda0, axial_disp)
    assert abs(z_rec - z_true) < 1e-9

# 2. depth_resolution_um: finer spectral resolution -> finer (smaller) depth resolution
res_coarse = s3d.depth_resolution_um(0.1, axial_disp)
res_fine = s3d.depth_resolution_um(0.01, axial_disp)
assert res_fine < res_coarse

# 3. depth_range_um scales linearly with z-band bandwidth
range_small = s3d.depth_range_um(10.0, axial_disp)
range_large = s3d.depth_range_um(20.0, axial_disp)
assert abs(range_large / range_small - 2.0) < 1e-9

# 4. split_spectral_budget: bands sum back to the total, both positive
xy_band, z_band = s3d.split_spectral_budget(60.0, 40.0)
assert abs(xy_band + z_band - 60.0) < 1e-9
assert xy_band > 0 and z_band > 0

# 5. split_spectral_budget rejects a split that leaves no room for z
try:
    s3d.split_spectral_budget(60.0, 60.0)
    assert False, "should have raised ValueError"
except ValueError:
    pass
try:
    s3d.split_spectral_budget(60.0, 70.0)
    assert False, "should have raised ValueError"
except ValueError:
    pass

# 6. frame_time_budget: doubling total bandwidth (both bands) doubles both
#    time windows and doubling total window HALVES the max frame rate
t1 = s3d.frame_time_budget(800.0, 40.0, 20.0)
t2 = s3d.frame_time_budget(800.0, 80.0, 40.0)
assert abs(t2["T_xy_ns"] / t1["T_xy_ns"] - 2.0) < 1e-9
assert abs(t2["T_z_ns"] / t1["T_z_ns"] - 2.0) < 1e-9
assert abs(t1["max_frame_rate_hz"] / t2["max_frame_rate_hz"] - 2.0) < 1e-9

# 7. the two time windows partition the total exactly
assert abs(t1["T_xy_ns"] + t1["T_z_ns"] - t1["T_total_ns"]) < 1e-9

# 8. input validation
for bad_call in [
    lambda: s3d.chromatic_confocal_depth_um(1550.0, 1550.0, -1.0),
    lambda: s3d.depth_resolution_um(-1.0, axial_disp),
    lambda: s3d.depth_resolution_um(0.1, -1.0),
    lambda: s3d.depth_range_um(-1.0, axial_disp),
    lambda: s3d.depth_range_um(10.0, -1.0),
    lambda: s3d.split_spectral_budget(-1.0, 40.0),
    lambda: s3d.split_spectral_budget(60.0, -1.0),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.steam_3d_depth_encoding tests passed")
