"""Test the optical mocap pipeline: a simulated trotting quadruped's IR
markers, projected into two virtual cameras, and reconstructed via
triangulation -- the actual linear-algebra core of Vicon/OptiTrack-style
motion capture."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import optical_mocap_quadruped as mocap

cam1_pos, cam1_fwd, cam1_up, f1 = np.array([0.0, -3.0, 1.0]), np.array([0, 1, -0.2]), np.array([0, 0, 1]), 1.0
cam2_pos, cam2_fwd, cam2_up, f2 = np.array([2.5, -2.0, 1.2]), np.array([-1, 1, -0.2]), np.array([0, 0, 1]), 1.0

# 1. trot gait: diagonal pairs (FL/BR, FR/BL) are in phase with each other
#    (a real trot gait property), tested across several time points
for t in [0.1, 0.3, 0.55, 0.9]:
    markers = mocap.dog_marker_positions(t)
    FL, FR, BL, BR = markers[0], markers[1], markers[2], markers[3]
    # diagonal pairs should have matching (or near-matching) foot heights,
    # since they share the same gait phase
    assert abs(FL[2] - BR[2]) < 1e-9
    assert abs(FR[2] - BL[2]) < 1e-9

# 2. reconstruction from two camera views recovers the true 3D marker
#    position to near machine precision, for EVERY marker at EVERY time
max_err_overall = 0.0
for t in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    markers_3d = mocap.dog_marker_positions(t)
    uv1, _ = mocap.pinhole_project(markers_3d, cam1_pos, cam1_fwd, cam1_up, f1)
    uv2, _ = mocap.pinhole_project(markers_3d, cam2_pos, cam2_fwd, cam2_up, f2)
    for true_pos, p1, p2 in zip(markers_3d, uv1, uv2):
        reconstructed = mocap.triangulate_point(p1, cam1_pos, cam1_fwd, cam1_up, f1,
                                                 p2, cam2_pos, cam2_fwd, cam2_up, f2)
        err = np.linalg.norm(reconstructed - true_pos)
        max_err_overall = max(max_err_overall, err)
assert max_err_overall < 1e-9

# 3. a DIFFERENT camera rig geometry still reconstructs correctly --
#    confirms the triangulation math isn't tuned to one specific rig
cam1b_pos, cam1b_fwd = np.array([-1.0, -4.0, 0.5]), np.array([0.2, 1, 0.1])
cam2b_pos, cam2b_fwd = np.array([1.5, -1.0, 2.0]), np.array([-0.5, 1, -0.5])
markers_3d = mocap.dog_marker_positions(0.35)
uv1b, _ = mocap.pinhole_project(markers_3d, cam1b_pos, cam1b_fwd, cam1_up, f1)
uv2b, _ = mocap.pinhole_project(markers_3d, cam2b_pos, cam2b_fwd, cam2_up, f2)
for true_pos, p1, p2 in zip(markers_3d, uv1b, uv2b):
    reconstructed = mocap.triangulate_point(p1, cam1b_pos, cam1b_fwd, cam1_up, f1,
                                             p2, cam2b_pos, cam2b_fwd, cam2_up, f2)
    assert np.linalg.norm(reconstructed - true_pos) < 1e-9

# 4. a point behind the camera must raise, not silently project garbage
try:
    mocap.pinhole_project(np.array([[0.0, -10.0, 0.0]]), cam1_pos, cam1_fwd, cam1_up, f1)
    assert False, "should have raised ValueError for a point behind the camera"
except ValueError:
    pass

# 5. input validation
for bad_call in [
    lambda: mocap.dog_marker_positions(0.0, body_length=-1.0),
    lambda: mocap.dog_marker_positions(0.0, gait_freq_hz=0.0),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.optical_mocap_quadruped tests passed")
