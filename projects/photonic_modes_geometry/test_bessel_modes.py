import numpy as np
import pytest
from bessel_modes import v_number, solve_lp01_u, lp01_effective_index, lp01_radial_profile, _lp01_residual


def test_v_number_matches_formula():
    n_core, n_clad, wl, a = 3.4, 1.44, 1.55, 1.0
    k0 = 2 * np.pi / wl
    expected = k0 * a * np.sqrt(n_core ** 2 - n_clad ** 2)
    assert v_number(n_core, n_clad, wl, a) == pytest.approx(expected)


def test_v_number_rejects_non_guiding_contrast():
    with pytest.raises(ValueError):
        v_number(1.44, 1.44, 1.55, 1.0)


def test_v_number_rejects_nonpositive_geometry():
    with pytest.raises(ValueError):
        v_number(3.4, 1.44, wavelength_um=0.0, radius_um=1.0)


def test_solve_lp01_u_satisfies_characteristic_equation():
    V = 12.485297255556857  # matches the notebook's default circle geometry
    u = solve_lp01_u(V)
    assert abs(_lp01_residual(u, V)) < 1e-6


def test_solve_lp01_u_rejects_nonpositive_V():
    with pytest.raises(ValueError):
        solve_lp01_u(-1.0)


def test_lp01_effective_index_is_guided():
    n_core, n_clad = 3.4, 1.44
    result = lp01_effective_index(n_core, n_clad, wavelength_um=1.55, radius_um=1.0)
    assert n_clad < result["n_eff"] < n_core


def test_lp01_effective_index_matches_fd_solver_closely():
    # cross-check against the finite-difference solver on the same geometry
    # -- these solve the SAME scalar equation two different ways, so
    # agreement should be tight (this checks the FD implementation, not
    # the scalar model's fidelity to real vector-mode physics)
    from geometry import make_circle
    from modes import solve_modes
    n_core, n_clad, wl, a = 3.4, 1.44, 1.55, 1.0
    analytic = lp01_effective_index(n_core, n_clad, wl, a)
    n_grid, _ = make_circle(64, 64, 0.1, 0.1, radius=a, n_core=n_core, n_clad=n_clad)
    fd_modes = solve_modes(n_grid, 0.1, 0.1, wl, n_modes=1)
    rel_err = abs(fd_modes[0]["n_eff"] - analytic["n_eff"]) / analytic["n_eff"]
    assert rel_err < 1e-3


def test_lp01_radial_profile_peaks_at_center():
    result = lp01_effective_index(3.4, 1.44, 1.55, 1.0)
    r = np.linspace(0, 3.0, 200)
    R = lp01_radial_profile(r, result["u"], result["w"], radius_um=1.0)
    assert R[0] == pytest.approx(1.0)
    assert np.argmax(np.abs(R)) == 0


def test_lp01_radial_profile_continuous_at_boundary():
    result = lp01_effective_index(3.4, 1.44, 1.55, 1.0)
    radius_um = 1.0
    eps = 1e-6
    R_inside = lp01_radial_profile(np.array([radius_um - eps]), result["u"], result["w"], radius_um)
    R_outside = lp01_radial_profile(np.array([radius_um + eps]), result["u"], result["w"], radius_um)
    assert R_inside[0] == pytest.approx(R_outside[0], abs=1e-4)


def test_lp01_radial_profile_decays_beyond_core():
    result = lp01_effective_index(3.4, 1.44, 1.55, 1.0)
    radius_um = 1.0
    R = lp01_radial_profile(np.array([radius_um * 1.2, radius_um * 3.0]), result["u"], result["w"], radius_um)
    assert abs(R[1]) < abs(R[0])


def test_lp01_radial_profile_rejects_negative_r():
    with pytest.raises(ValueError):
        lp01_radial_profile(np.array([-1.0]), u=1.0, w=1.0, radius_um=1.0)
