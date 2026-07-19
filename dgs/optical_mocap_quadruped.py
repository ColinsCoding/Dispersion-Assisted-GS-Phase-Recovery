"""Optical motion capture, the actual pipeline: IR-reflective markers on a
moving subject (here, a simulated dog-sized quadruped -- paws, shoulders,
hips, head, tail), TWO cameras each seeing a 2D projection of the 3D
markers, and TRIANGULATION reconstructing the original 3D positions from
those two 2D views. This is exactly how real systems (Vicon, OptiTrack)
work: IR markers are just retroreflective dots that show up as bright
points in IR camera footage; the hard computational part is triangulating
3D position from 2+ camera views, which is a linear-algebra least-squares
problem, not a machine-learning one.

The quadruped gait model is a simple TROT: diagonal leg pairs (front-left/
back-right, front-right/back-left) move in phase with each other and out
of phase with the other diagonal pair -- the real gait pattern a trotting
dog (golden doodle-sized) actually uses, not an arbitrary animation curve.
"""

import numpy as np


def dog_marker_positions(t, body_length=0.55, body_height=0.45, leg_length=0.35,
                          stride_amplitude=0.18, gait_freq_hz=1.5):
    """3D positions of 8 IR markers on a trotting quadruped at time t:
    4 paws (diagonal pairs in phase, a real trot gait), shoulders, hips,
    head, tail. Dimensions default to a golden-doodle-sized dog (~55cm
    body length, ~45cm shoulder height). Returns an (8,3) array."""
    if body_length <= 0 or body_height <= 0 or leg_length <= 0:
        raise ValueError("body_length, body_height, leg_length must be positive")
    if gait_freq_hz <= 0:
        raise ValueError("gait_freq_hz must be positive")

    omega = 2 * np.pi * gait_freq_hz
    phase_diag1 = omega * t                 # front-left + back-right
    phase_diag2 = omega * t + np.pi          # front-right + back-left (opposite diagonal)

    def paw_xyz(x_body, y_body, phase):
        # a foot's height oscillates during swing phase (clamped to 0 during stance)
        swing = np.maximum(np.sin(phase), 0.0)
        z = swing * 0.08
        x = x_body + stride_amplitude * np.sin(phase) * 0.3
        return np.array([x, y_body, z])

    half_len, half_wid = body_length / 2, 0.15
    FL = paw_xyz(half_len, half_wid, phase_diag1)
    BR = paw_xyz(-half_len, -half_wid, phase_diag1)
    FR = paw_xyz(half_len, -half_wid, phase_diag2)
    BL = paw_xyz(-half_len, half_wid, phase_diag2)

    body_bob = 0.02 * np.sin(2 * omega * t)   # body bobs at 2x stride frequency (both diagonals push)
    shoulders = np.array([half_len * 0.7, 0.0, body_height + body_bob])
    hips = np.array([-half_len * 0.7, 0.0, body_height + body_bob])
    head = np.array([half_len * 1.3, 0.0, body_height + 0.15 + body_bob])
    tail = np.array([-half_len * 1.3, 0.0, body_height - 0.05 + body_bob])

    return np.array([FL, FR, BL, BR, shoulders, hips, head, tail])


def pinhole_project(points_3d, cam_position, cam_forward, cam_up, focal_length):
    """Project 3D world points onto a camera's 2D image plane (pinhole
    model): build the camera's local basis (forward/right/up), express
    each point in camera coordinates, then perspective-divide by depth."""
    points_3d = np.atleast_2d(np.asarray(points_3d, dtype=float))
    cam_position = np.asarray(cam_position, dtype=float)
    forward = np.asarray(cam_forward, dtype=float)
    forward = forward / np.linalg.norm(forward)
    up = np.asarray(cam_up, dtype=float)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)

    rel = points_3d - cam_position
    x_cam = rel @ right
    y_cam = rel @ true_up
    z_cam = rel @ forward   # depth along the camera's viewing direction
    if np.any(z_cam <= 0):
        raise ValueError("point(s) behind the camera (z_cam <= 0) -- cannot project")

    u = focal_length * x_cam / z_cam
    v = focal_length * y_cam / z_cam
    return np.column_stack([u, v]), z_cam


def triangulate_point(uv1, cam1_position, cam1_forward, cam1_up, f1,
                       uv2, cam2_position, cam2_forward, cam2_up, f2):
    """Reconstruct a 3D point from its 2D projections in two cameras, via
    linear least-squares (each camera observation constrains the point to
    lie on a ray; find the 3D point minimizing total squared distance to
    both rays -- the standard two-view triangulation problem)."""
    def ray_direction(uv, forward, up, f):
        forward = np.asarray(forward, dtype=float); forward /= np.linalg.norm(forward)
        up = np.asarray(up, dtype=float)
        right = np.cross(forward, up); right /= np.linalg.norm(right)
        true_up = np.cross(right, forward)
        u, v = uv
        d = forward * f + right * u + true_up * v
        return d / np.linalg.norm(d)

    o1 = np.asarray(cam1_position, dtype=float)
    o2 = np.asarray(cam2_position, dtype=float)
    d1 = ray_direction(uv1, cam1_forward, cam1_up, f1)
    d2 = ray_direction(uv2, cam2_forward, cam2_up, f2)

    # closest point between two skew rays: solve the 2x2 normal-equations
    # system for the two ray parameters (standard least-squares triangulation)
    A = np.array([[d1 @ d1, -d1 @ d2], [d1 @ d2, -d2 @ d2]])
    b = np.array([(o2 - o1) @ d1, (o2 - o1) @ d2])
    t1, t2 = np.linalg.solve(A, b)
    p1 = o1 + t1 * d1
    p2 = o2 + t2 * d2
    return (p1 + p2) / 2   # midpoint of the closest approach between the two rays


if __name__ == "__main__":
    print("=== Simulated golden-doodle-sized quadruped, trot gait, 8 IR markers ===")
    t_test = 0.3
    markers_3d = dog_marker_positions(t_test)
    labels = ["front-left paw", "front-right paw", "back-left paw", "back-right paw",
              "shoulders", "hips", "head", "tail"]
    for label, pos in zip(labels, markers_3d):
        print(f"  {label:16s}: {pos}")

    print("\n=== Two-camera optical mocap rig ===")
    cam1_pos, cam1_fwd, cam1_up, f1 = np.array([0.0, -3.0, 1.0]), np.array([0, 1, -0.2]), np.array([0, 0, 1]), 1.0
    cam2_pos, cam2_fwd, cam2_up, f2 = np.array([2.5, -2.0, 1.2]), np.array([-1, 1, -0.2]), np.array([0, 0, 1]), 1.0

    uv1, _ = pinhole_project(markers_3d, cam1_pos, cam1_fwd, cam1_up, f1)
    uv2, _ = pinhole_project(markers_3d, cam2_pos, cam2_fwd, cam2_up, f2)

    print("Reconstructing each marker's 3D position from ONLY its two 2D camera views:")
    max_err = 0.0
    for label, true_pos, p1, p2 in zip(labels, markers_3d, uv1, uv2):
        reconstructed = triangulate_point(p1, cam1_pos, cam1_fwd, cam1_up, f1,
                                           p2, cam2_pos, cam2_fwd, cam2_up, f2)
        err = np.linalg.norm(reconstructed - true_pos)
        max_err = max(max_err, err)
        print(f"  {label:16s}: true={true_pos.round(4)}  reconstructed={reconstructed.round(4)}  "
              f"error={err:.2e} m")

    print(f"\nmax reconstruction error across all 8 markers: {max_err:.2e} m "
          f"({'sub-millimeter' if max_err < 1e-3 else 'CHECK -- larger than expected'})")
    print("\nThis is the actual computational core of Vicon/OptiTrack-style mocap:")
    print("IR markers just need to be bright/reflective enough to segment in each camera's")
    print("image; triangulation (linear algebra, not ML) does the rest. A game engine can")
    print("drive a golden-doodle character rig directly from these 8 reconstructed 3D points")
    print("per frame -- swap the simulated marker generator for real camera footage and the")
    print("triangulation math is unchanged.")
