"""Test dgs/boat_lagrangian.py: heave and roll as Lagrangian oscillators.
Checks that the symbolic Euler-Lagrange route and the closed-form
naval-architecture formulas describe the same physics (agree exactly),
and that kwarg bounds reject non-physical inputs (GM <= 0, negative mass)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import boat_lagrangian as boat

# 1. Heave: closed-form natural frequency matches the textbook shape
w = boat.heave_natural_frequency(m=5000.0, m_added=800.0, rho=boat.RHO_SEAWATER,
                                  g=boat.G_STANDARD, A_wp=12.0)
expected = np.sqrt(boat.RHO_SEAWATER * boat.G_STANDARD * 12.0 / (5000.0 + 800.0))
assert abs(w - expected) < 1e-12

# 2. Heave: symbolic Euler-Lagrange derivation matches the closed form exactly
heave_check = boat.verify_heave_eom()
assert heave_check["matches"]
assert abs(heave_check["omega_symbolic_rad_s"] - heave_check["omega_closed_form_rad_s"]) < 1e-9

# 3. Added mass increases inertia -> lowers the natural frequency
w_no_added = boat.heave_natural_frequency(m=5000.0, m_added=0.0, rho=boat.RHO_SEAWATER,
                                           g=boat.G_STANDARD, A_wp=12.0)
w_with_added = boat.heave_natural_frequency(m=5000.0, m_added=800.0, rho=boat.RHO_SEAWATER,
                                             g=boat.G_STANDARD, A_wp=12.0)
assert w_with_added < w_no_added

# 4. Seawater (denser) gives a stiffer restoring force -> higher heave frequency
#    than freshwater, for the same hull
w_fresh = boat.heave_natural_frequency(m=5000.0, m_added=800.0, rho=boat.RHO_FRESHWATER,
                                        g=boat.G_STANDARD, A_wp=12.0)
w_sea = boat.heave_natural_frequency(m=5000.0, m_added=800.0, rho=boat.RHO_SEAWATER,
                                      g=boat.G_STANDARD, A_wp=12.0)
assert w_sea > w_fresh

# 5. Roll: closed-form matches the direct sqrt(rho g nabla GM / I) formula
I_roll, nabla, GM = 3600.0, 400.0, 1.2
w_roll = boat.roll_natural_frequency(I_roll, boat.RHO_SEAWATER, boat.G_STANDARD, nabla, GM)
expected_roll = np.sqrt(boat.RHO_SEAWATER * boat.G_STANDARD * nabla * GM / I_roll)
assert abs(w_roll - expected_roll) < 1e-12

# 6. Roll: the mass-cancellation shortcut T = 2*pi*k/sqrt(g*GM) matches the
#    full formula when I_roll = m*k^2, m = rho*nabla (Archimedes)
roll_check = boat.verify_roll_period_shortcut()
assert roll_check["matches"]
assert abs(roll_check["period_from_omega_s"] - roll_check["period_from_shortcut_s"]) < 1e-9

# 7. kwarg bounds: non-physical inputs must raise, not silently produce nan/negative
for bad_call in [
    lambda: boat.heave_natural_frequency(m=-1.0, m_added=0.0, rho=1000.0, g=9.8, A_wp=10.0),
    lambda: boat.heave_natural_frequency(m=1.0, m_added=-5.0, rho=1000.0, g=9.8, A_wp=10.0),
    lambda: boat.heave_natural_frequency(m=1.0, m_added=0.0, rho=-1.0, g=9.8, A_wp=10.0),
    lambda: boat.heave_natural_frequency(m=1.0, m_added=0.0, rho=1000.0, g=9.8, A_wp=0.0),
    lambda: boat.roll_natural_frequency(I_roll=1.0, rho=1000.0, g=9.8, nabla=1.0, GM=0.0),
    lambda: boat.roll_natural_frequency(I_roll=1.0, rho=1000.0, g=9.8, nabla=1.0, GM=-0.5),
    lambda: boat.roll_period_from_radius_of_gyration(k_roll=1.0, GM=-0.1),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.boat_lagrangian tests passed")
