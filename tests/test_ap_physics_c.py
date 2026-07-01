"""Tests for dgs/ap_physics_c.py -- AP Physics C E&M self-tutor"""
import numpy as np
import pytest
from dgs.ap_physics_c import (
    coulombs_law, electric_field_point_charge, gauss_law_surface_integral,
    gauss_law_geometries, electric_potential_point_charge,
    potential_from_field_line_integral, sympy_gradient_demo,
    parallel_plate_capacitor, capacitor_energy, capacitors_series_parallel,
    rc_circuit, kirchhoff_two_loop, biot_savart_segment, ampere_law_geometries,
    lorentz_force, cyclotron_radius, faraday_law, self_inductance_solenoid,
    rl_circuit, lc_oscillator,
    gaussian_measurement, error_propagation_em, chi_squared_test,
    linear_regression_em, rc_fit_from_data,
    run_problem, k_e, eps0, mu0,
)


# ---------------------------------------------------------------------------
# Unit 1: Electrostatics
# ---------------------------------------------------------------------------
def test_coulombs_law_repulsive():
    r = coulombs_law(1e-6, 1e-6, 0.1)
    assert r['F_N'] > 0
    assert r['repulsive']


def test_coulombs_law_attractive():
    r = coulombs_law(1e-6, -1e-6, 0.1)
    assert r['F_N'] < 0
    assert not r['repulsive']


def test_coulombs_law_magnitude():
    # F = k*q^2/r^2 for equal charges 1uC at 1m
    r = coulombs_law(1e-6, 1e-6, 1.0)
    assert abs(r['F_N'] - k_e*1e-12) < 1e-15


def test_coulombs_law_invalid_r():
    with pytest.raises(ValueError):
        coulombs_law(1e-6, 1e-6, 0.0)


def test_electric_field_point_charge():
    # E at r=1m from 1C charge = k_e
    E = electric_field_point_charge(1.0, 1.0)
    assert abs(E - k_e) < 1.0


def test_gauss_law_flux():
    g = gauss_law_surface_integral(1e-6)
    expected = 1e-6 / eps0
    assert abs(g['flux_N_m2_per_C'] - expected) < 1e-4


def test_gauss_law_geometries_sphere():
    g = gauss_law_geometries(Q_total_C=1e-6, r_m=0.2)
    # Outside sphere: E = k*Q/r^2
    expected = k_e * 1e-6 / 0.04
    assert abs(g['sphere_outside_N_per_C'] - expected) < 1.0


def test_gauss_law_geometries_plane():
    g = gauss_law_geometries(Q_total_C=1e-6, r_m=0.1)
    # Infinite plane: E = sigma/(2*eps0) = Q/(2*eps0) for A=1m^2
    expected = 1e-6 / (2*eps0)
    assert abs(g['plane_N_per_C'] - expected) < 1.0


def test_electric_potential_point_charge():
    # V at r=1m from 1C = k_e
    V = electric_potential_point_charge(1.0, 1.0)
    assert abs(V - k_e) < 1.0


def test_potential_line_integral_uniform_field():
    # Uniform E=100 V/m, move from x=0 to x=1m
    # delta_V = -E*d = -100 V
    result = potential_from_field_line_integral(lambda x: 100.0, 0.0, 1.0)
    assert abs(result['delta_V_volts'] - (-100.0)) < 0.1


def test_sympy_gradient_keys():
    g = sympy_gradient_demo()
    for key in ['V', 'Ex', 'Ey', 'Ez', 'E_magnitude']:
        assert key in g


# ---------------------------------------------------------------------------
# Unit 2: Capacitors
# ---------------------------------------------------------------------------
def test_parallel_plate_capacitor():
    c = parallel_plate_capacitor(A_m2=0.01, d_m=0.001)
    expected = eps0 * 0.01 / 0.001
    assert abs(c['C_farads'] - expected) < 1e-16


def test_parallel_plate_dielectric():
    c1 = parallel_plate_capacitor(0.01, 0.001, kappa=1.0)
    c2 = parallel_plate_capacitor(0.01, 0.001, kappa=4.0)
    assert abs(c2['C_farads'] / c1['C_farads'] - 4.0) < 1e-10


def test_capacitor_energy_from_V():
    cap = capacitor_energy(100e-6, V_volts=9.0)
    expected = 0.5 * 100e-6 * 81.0
    assert abs(cap['U_joules'] - expected) < 1e-10


def test_capacitor_energy_from_Q():
    Q = 100e-6 * 9.0
    cap = capacitor_energy(100e-6, Q_coulombs=Q)
    assert abs(cap['U_joules'] - 0.5*100e-6*81.0) < 1e-10


def test_capacitors_parallel():
    c = capacitors_series_parallel([100e-6, 200e-6, 50e-6], 'parallel')
    assert abs(c['C_eq_F'] - 350e-6) < 1e-12


def test_capacitors_series():
    c = capacitors_series_parallel([100e-6, 100e-6], 'series')
    assert abs(c['C_eq_F'] - 50e-6) < 1e-12


# ---------------------------------------------------------------------------
# Unit 3: Circuits
# ---------------------------------------------------------------------------
def test_rc_circuit_tau():
    rc = rc_circuit(10e3, 100e-6, 9.0)
    assert abs(rc['tau_s'] - 1.0) < 1e-10


def test_rc_circuit_at_tau():
    rc = rc_circuit(10e3, 100e-6, 9.0)
    # At t=tau: V_C = V0*(1-1/e) = 0.6321*V0
    assert abs(rc['V_at_tau'] / 9.0 - (1-1/np.e)) < 1e-10


def test_rc_circuit_charge_discharge_complementary():
    rc = rc_circuit(1e3, 1e-3, 5.0)
    # At any t: V_charge + V_discharge = V0
    total = rc['V_charge_V'] + rc['V_discharge_V']
    assert np.allclose(total, 5.0)


def test_rc_circuit_energy():
    rc = rc_circuit(10e3, 100e-6, 9.0)
    expected = 0.5 * 100e-6 * 81.0
    assert abs(rc['energy_stored_J'] - expected) < 1e-12


def test_kirchhoff_two_loop():
    k = kirchhoff_two_loop()
    # Verify KVL loop 1: 12 = I1*4 + (I1-I2)*3 = 7*I1 - 3*I2
    lhs1 = 7*k['I1_A'] - 3*k['I2_A']
    assert abs(lhs1 - 12.0) < 1e-3
    # Verify KVL loop 2: 6 = -3*I1 + 9*I2
    lhs2 = -3*k['I1_A'] + 9*k['I2_A']
    assert abs(lhs2 - 6.0) < 1e-3


def test_kirchhoff_power_positive():
    k = kirchhoff_two_loop()
    assert k['P_R1_W'] > 0
    assert k['P_R2_W'] > 0
    assert k['P_R3_W'] > 0


# ---------------------------------------------------------------------------
# Unit 4: Magnetic fields
# ---------------------------------------------------------------------------
def test_biot_savart_perpendicular():
    # dl along x, r along y -> dB should be along z
    result = biot_savart_segment(1.0, 0.01, [1,0,0], [0,0.1,0])
    dB = result['dB_vec_T']
    assert abs(dB[0]) < 1e-15
    assert abs(dB[1]) < 1e-15
    assert dB[2] > 0  # out of page


def test_biot_savart_magnitude():
    result = biot_savart_segment(1.0, 0.01, [1,0,0], [0,0.1,0])
    expected = mu0 / (4*np.pi) * 1.0 * 0.01 / 0.1**2
    assert abs(result['dB_mag_T'] - expected) < 1e-20


def test_ampere_law_wire():
    amp = ampere_law_geometries(I_A=1.0, r_m=0.01)
    expected = mu0 * 1.0 / (2*np.pi*0.01)
    assert abs(amp['B_wire_T'] - expected) < 1e-15


def test_ampere_law_solenoid():
    amp = ampere_law_geometries(I_A=1.0)
    # n=1000, B = mu0*1000*1 = mu0*1000
    expected = mu0 * 1000 * 1.0
    assert abs(amp['B_solenoid_T'] - expected) < 1e-15


def test_lorentz_force_electric_only():
    # q=1C, v=0, E=1N/C -> F=1N along x
    r = lorentz_force(1.0, [0,0,0], [1,0,0], [0,0,0])
    assert abs(r['F_vec_N'][0] - 1.0) < 1e-10


def test_lorentz_force_magnetic_perpendicular():
    # v along x, B along z -> F along -y
    r = lorentz_force(1.0, [1,0,0], [0,0,0], [0,0,1])
    assert abs(r['F_vec_N'][1] - (-1.0)) < 1e-10


def test_cyclotron_radius_electron():
    m_e = 9.109e-31; q_e = 1.602e-19
    r = cyclotron_radius(m_e, q_e, 1e6, 0.01)
    expected = m_e * 1e6 / (q_e * 0.01)
    assert abs(r['r_m'] - expected) < 1e-10


# ---------------------------------------------------------------------------
# Unit 5: Faraday, RL, LC
# ---------------------------------------------------------------------------
def test_faraday_law_sinusoidal():
    # B(t) = B0*sin(omega*t), dB/dt = B0*omega*cos(omega*t)
    B0 = 0.1; omega = 2*np.pi*60; A = 0.01
    result = faraday_law(lambda t: B0*np.sin(omega*t), A, dt=1e-9, t=0.0)
    # At t=0: dPhi/dt = B0*omega*A*cos(0) = B0*omega*A
    expected_emf = -B0 * omega * A
    assert abs(result['EMF_volts'] - expected_emf) < 0.001


def test_faraday_law_constant_B():
    # Constant B -> EMF = 0
    result = faraday_law(lambda t: 1.0, 0.01)
    assert abs(result['EMF_volts']) < 1e-6


def test_self_inductance_solenoid():
    L = self_inductance_solenoid(N_turns=100, A_m2=1e-4, l_m=0.1)
    expected = mu0 * 100**2 * 1e-4 / 0.1
    assert abs(L['L_H'] - expected) < 1e-12


def test_rl_circuit_tau():
    rl = rl_circuit(100, 0.1, 5.0)
    assert abs(rl['tau_s'] - 0.001) < 1e-10


def test_rl_circuit_final_current():
    rl = rl_circuit(100, 0.1, 5.0)
    assert abs(rl['I_final_A'] - 0.05) < 1e-10


def test_rl_circuit_initial_voltage():
    # At t=0: V_L = V0 (all voltage across inductor, I=0)
    rl = rl_circuit(100, 0.1, 5.0)
    assert abs(rl['V_L_V'][0] - 5.0) < 0.01


def test_lc_oscillator_energy_conserved():
    lc = lc_oscillator(2e-3, 50e-12, 5.0)
    assert lc['energy_conserved']


def test_lc_oscillator_frequency():
    L = 2e-3; C = 50e-12
    lc = lc_oscillator(L, C, 5.0)
    expected = 1/(2*np.pi*np.sqrt(L*C))
    assert abs(lc['f0_Hz'] - expected) < 1.0


def test_lc_oscillator_period():
    lc = lc_oscillator(2e-3, 50e-12, 5.0)
    # Voltage at t=T should equal V0 (one full period)
    t = lc['t_s']
    V = lc['V_C_V']
    T = lc['T_period_s']
    i_T = np.argmin(np.abs(t - T))
    assert abs(V[i_T] - 5.0) < 0.01


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def test_gaussian_measurement_mean():
    g = gaussian_measurement(9.0, 0.1, n_measurements=10000, seed=0)
    assert abs(g['mean'] - 9.0) < 0.01


def test_gaussian_68_rule():
    g = gaussian_measurement(0.0, 1.0, n_measurements=10000, seed=1)
    assert abs(g['within_1sigma'] - 0.6827) < 0.02


def test_gaussian_95_rule():
    g = gaussian_measurement(0.0, 1.0, n_measurements=10000, seed=2)
    assert abs(g['within_2sigma'] - 0.9545) < 0.02


def test_error_propagation_formula():
    err = error_propagation_em(20.0, 0.1, 5.0, 0.05)
    # I = V/R = 0.25 A
    assert abs(err['I_A'] - 0.25) < 1e-10
    # dI/I = sqrt((0.05/5)^2 + (0.1/20)^2) = sqrt(0.0001 + 0.000025) = 0.01118
    expected_rel = np.sqrt((0.05/5)**2 + (0.1/20)**2)
    assert abs(err['I_relative_error'] - expected_rel) < 1e-10


def test_chi_squared_perfect_fit():
    obs = [10, 20, 30]
    exp = [10, 20, 30]
    r = chi_squared_test(obs, exp)
    assert r['chi2'] == pytest.approx(0.0)


def test_chi_squared_poor_fit():
    obs = [10, 20, 30]
    exp = [1, 2, 3]
    r = chi_squared_test(obs, exp)
    assert r['chi2'] > 10


def test_linear_regression_perfect():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 3.0 * x + 1.0
    r = linear_regression_em(x, y)
    assert abs(r['slope'] - 3.0) < 1e-10
    assert abs(r['intercept'] - 1.0) < 1e-10
    assert abs(r['R_squared'] - 1.0) < 1e-10


def test_rc_fit_from_data():
    tau_true = 1.0
    t = np.linspace(0.1, 4.0, 20)
    V = 9.0 * np.exp(-t / tau_true)
    result = rc_fit_from_data(t, V, V0_V=9.0)
    assert abs(result['tau_fit_s'] - tau_true) < 0.01
    assert result['R_squared'] > 0.999


def test_run_problem_lc():
    p = run_problem(2)
    assert p['unit'] == 5
    assert 'f0_MHz' in p['numeric_result']
    assert abs(p['numeric_result']['f0_MHz'] - 0.503) < 0.01


def test_run_problem_keys():
    for i in range(len([0,1,2,3])):
        p = run_problem(i)
        assert 'problem' in p
        assert 'solution' in p
        assert 'numeric_result' in p
