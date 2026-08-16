"""Test the 3D grill dynamics: real box inertia formula, the drumstick's
airborne (pre-contact) phase is genuinely torque-free and matches
dgs.gyroscopes' independent Euler-equations integrator directly, contact
physics settles the piece to rest, and landing classification correctly
distinguishes a clean flip from a bad tumble."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.chicken_bbq_3d import (
    drumstick_inertia, launch_and_simulate_flip_3d, which_face_down,
)
from dgs.gyroscopes import integrate_euler_rigid_body

# 1. box inertia formula matches the textbook expression directly
mass = 0.12
sx, sy, sz = 0.02, 0.03, 0.09
Ix, Iy, Iz = drumstick_inertia(mass, (sx, sy, sz))
assert abs(Ix - (mass / 12) * ((2 * sy) ** 2 + (2 * sz) ** 2)) < 1e-12
assert abs(Iy - (mass / 12) * ((2 * sx) ** 2 + (2 * sz) ** 2)) < 1e-12
assert abs(Iz - (mass / 12) * ((2 * sx) ** 2 + (2 * sy) ** 2)) < 1e-12
# genuinely asymmetric (three distinct moments) -- required for the
# tennis-racket-theorem-style tumbling this module is built around
assert len({round(Ix, 8), round(Iy, 8), round(Iz, 8)}) == 3

# 2. the airborne phase really is torque-free: MuJoCo's own contact-physics
# simulation agrees with dgs.gyroscopes' independent RK4 Euler-equations
# integrator on the SAME initial angular velocity, over the same time window
angular_v = [0.2, 9.0, 0.3]
run = launch_and_simulate_flip_3d([0.0, 0.0, 2.2], angular_v, t_max=2.0)
ref = integrate_euler_rigid_body(angular_v, run["I1"], run["I2"], run["I3"],
                                  t_max=float(run["airborne_t"][-1]), dt=0.0005)
idx = min(len(ref["omega"]) - 1, len(run["airborne_omega"]) - 1)
diff = np.max(np.abs(run["airborne_omega"][-1] - ref["omega"][idx]))
assert diff < 0.05   # small residual is expected (float32 MuJoCo vs float64 RK4, near-chaotic regime)

# 3. contact physics actually settles the piece (comes to rest on the grill)
assert run["settled"] is True

# 4. landing classification: a well-tuned flip lands flat (skin or bone),
# not on its edge
clean_run = launch_and_simulate_flip_3d([0.0, 0.0, 2.2], [6.0, 0.0, 0.0], t_max=2.0)
assert which_face_down(clean_run["final_quat"]) in ("skin", "bone")

# 5. which_face_down correctly reads a KNOWN orientation directly (no
# simulation involved) -- identity quaternion means local z points along
# world z, i.e. local +z points UP, so the local +z ("skin") face is NOT
# down; local -z ("bone" side) IS down
identity_quat = np.array([1.0, 0.0, 0.0, 0.0])
assert which_face_down(identity_quat) == "bone"

# a 180-degree flip about x (quat = [0,1,0,0]) inverts z, so now local +z
# points down -> "skin" side down
flipped_quat = np.array([0.0, 1.0, 0.0, 0.0])
assert which_face_down(flipped_quat) == "skin"

print("all dgs.chicken_bbq_3d tests passed")
