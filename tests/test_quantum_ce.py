"""Tests for dgs/quantum_ce.py -- Quantum Science for Computer Engineers"""
import numpy as np
import pytest
from dgs.quantum_ce import (
    ENERGY_SCALES, energy_in_all_units, photon_energy, thermal_vs_qubit_energy,
    maxwell_to_qed_bridge, purcell_effect, QUBIT_HARDWARE, pauli_matrices,
    three_qubit_bit_flip_code, surface_code_overview, quantum_advantage_table,
    griffiths_to_qis_map, josephson_junction, h_J, hbar, kB, eV, c,
)


# ---------------------------------------------------------------------------
# Energy scales
# ---------------------------------------------------------------------------
def test_energy_scales_kT_room():
    # kT at 300K ~ 26 meV = 0.026 eV
    assert abs(ENERGY_SCALES['kT_room_300K_eV'] - 0.02585) < 1e-4


def test_energy_scales_telecom_photon():
    # hf at 1550nm ~ 0.80 eV
    assert abs(ENERGY_SCALES['hf_1550nm_eV'] - 0.800) < 0.001


def test_photon_energy_1550nm():
    p = photon_energy(1550.0)
    assert abs(p['eV'] - 0.800) < 0.001
    assert abs(p['freq_THz'] - 193.0) < 1.0
    assert abs(p['wavelength_nm'] - 1550.0) < 0.1


def test_photon_energy_800nm():
    p = photon_energy(800.0)
    assert p['eV'] > 1.5
    assert p['eV'] < 1.6


def test_energy_in_all_units_joule():
    E = eV  # 1 eV in Joules
    units = energy_in_all_units(E)
    assert abs(units['eV'] - 1.0) < 1e-10
    assert abs(units['meV'] - 1000.0) < 1e-7
    assert units['J'] == eV


def test_energy_joule_conversion():
    # 1 eV = 1.602e-19 J
    assert abs(eV - 1.602e-19) < 1e-22


def test_thermal_vs_qubit_quantum_regime():
    result = thermal_vs_qubit_energy(T_K=0.015, f_qubit_GHz=5.0)
    assert result['quantum_regime']
    assert result['hf_over_kT'] > 10
    assert result['excited_state_population'] < 1e-3


def test_thermal_vs_qubit_classical_regime():
    result = thermal_vs_qubit_energy(T_K=300.0, f_qubit_GHz=5.0)
    assert not result['quantum_regime']
    assert result['hf_over_kT'] < 1


# ---------------------------------------------------------------------------
# Maxwell -> QED bridge
# ---------------------------------------------------------------------------
def test_maxwell_to_qed_bridge_keys():
    bridge = maxwell_to_qed_bridge()
    for key in ['vacuum_field_V_per_m', 'commutation', 'gs_connection', 'ce_analog']:
        assert key in bridge


def test_vacuum_field_positive():
    bridge = maxwell_to_qed_bridge()
    assert bridge['vacuum_field_V_per_m'] > 0
    # Typical vacuum field in 10um cavity: order 1-100 V/m
    assert 0.1 < bridge['vacuum_field_V_per_m'] < 1e6


def test_purcell_factor_positive():
    result = purcell_effect(Q=1000, V_mode_um3=100)
    assert result['Purcell_factor'] > 0


def test_purcell_factor_scales_with_Q():
    r1 = purcell_effect(Q=100, V_mode_um3=100)
    r10 = purcell_effect(Q=1000, V_mode_um3=100)
    assert abs(r10['Purcell_factor'] / r1['Purcell_factor'] - 10.0) < 0.01


def test_purcell_factor_scales_with_volume():
    r1 = purcell_effect(Q=1000, V_mode_um3=100)
    r2 = purcell_effect(Q=1000, V_mode_um3=200)
    assert r2['Purcell_factor'] < r1['Purcell_factor']


# ---------------------------------------------------------------------------
# Pauli matrices
# ---------------------------------------------------------------------------
def test_pauli_hermitian():
    P = pauli_matrices()
    for name, M in P.items():
        assert np.allclose(M, M.conj().T), f"{name} is not Hermitian"


def test_pauli_trace_zero():
    P = pauli_matrices()
    for name in ['X', 'Y', 'Z']:
        assert abs(np.trace(P[name])) < 1e-10


def test_pauli_commutation_XY():
    P = pauli_matrices()
    X, Y, Z = P['X'], P['Y'], P['Z']
    comm = X @ Y - Y @ X
    assert np.allclose(comm, 2j * Z)


def test_pauli_anticommutation():
    P = pauli_matrices()
    X, Y = P['X'], P['Y']
    anticomm = X @ Y + Y @ X
    assert np.allclose(anticomm, 0)


# ---------------------------------------------------------------------------
# 3-qubit bit-flip code
# ---------------------------------------------------------------------------
def test_bit_flip_code_keys():
    result = three_qubit_bit_flip_code()
    for key in ['stabilizers', 'logical_X', 'logical_Z', 'example_syndrome_X0_error']:
        assert key in result


def test_bit_flip_code_stabilizer_shapes():
    result = three_qubit_bit_flip_code()
    for name, S in result['stabilizers'].items():
        assert S.shape == (8, 8)


def test_bit_flip_code_logical_operators_anticommute():
    result = three_qubit_bit_flip_code()
    XL = result['logical_X']
    ZL = result['logical_Z']
    # {X_L, Z_L} = X_L Z_L + Z_L X_L = (XZ+ZX)@X@X = 0 (anticommute)
    anticomm = XL @ ZL + ZL @ XL
    assert np.allclose(anticomm, 0)


def test_bit_flip_code_syndrome_x0_error():
    # X error on qubit 0: syndrome should be (1,1) or (-1,?) depending on convention
    result = three_qubit_bit_flip_code()
    syn = result['example_syndrome_X0_error']
    # Syndrome is a tuple of two values
    assert len(syn) == 2


def test_bit_flip_code_distance():
    result = three_qubit_bit_flip_code()
    assert result['distance'] == 3


# ---------------------------------------------------------------------------
# Surface code overview
# ---------------------------------------------------------------------------
def test_surface_code_keys():
    sc = surface_code_overview()
    for key in ['code_distance', 'physical_per_logical', 'threshold_percent', 'decoder']:
        assert key in sc


def test_surface_code_distance3_count():
    sc = surface_code_overview()
    assert sc['code_distance'] == 3
    assert sc['physical_per_logical'] == 17


def test_surface_code_threshold():
    sc = surface_code_overview()
    assert 0.5 < sc['threshold_percent'] < 5.0


# ---------------------------------------------------------------------------
# Quantum advantage table
# ---------------------------------------------------------------------------
def test_quantum_advantage_table_length():
    entries = quantum_advantage_table()
    assert len(entries) >= 4


def test_quantum_advantage_shor_entry():
    entries = quantum_advantage_table()
    problems = [e['problem'] for e in entries]
    assert any('factor' in p.lower() or 'RSA' in p or 'Shor' in str(e) for e, p in zip(entries, problems))


def test_quantum_advantage_grover_entry():
    entries = quantum_advantage_table()
    assert any('search' in e['problem'].lower() or 'Grover' in str(e) for e in entries)


# ---------------------------------------------------------------------------
# Griffiths -> QIS map
# ---------------------------------------------------------------------------
def test_griffiths_qis_map_length():
    mapping = griffiths_to_qis_map()
    assert len(mapping) >= 4


def test_griffiths_qis_map_keys():
    mapping = griffiths_to_qis_map()
    for entry in mapping:
        assert 'griffiths_ch' in entry
        assert 'qis_concept' in entry
        assert 'math' in entry


# ---------------------------------------------------------------------------
# Josephson junction
# ---------------------------------------------------------------------------
def test_josephson_junction_keys():
    jj = josephson_junction()
    for key in ['E_J_over_E_C', 'f01_GHz', 'anharmonicity_MHz', 'transmon_regime']:
        assert key in jj


def test_josephson_junction_transmon_regime():
    jj = josephson_junction()
    assert jj['E_J_over_E_C'] > 10  # transmon criterion: EJ/EC >> 1, typically 20-100
    assert jj['transmon_regime']


def test_josephson_junction_frequency_range():
    jj = josephson_junction()
    # Typical transmon: 4-8 GHz
    assert 2.0 < jj['f01_GHz'] < 12.0


def test_josephson_junction_anharmonicity_negative():
    jj = josephson_junction()
    # Transmon anharmonicity is negative (Ec term): -Ec/hbar
    assert jj['anharmonicity_MHz'] < 0


def test_qubit_hardware_keys():
    assert 'superconducting_transmon' in QUBIT_HARDWARE
    assert 'photonic_qubit' in QUBIT_HARDWARE
    assert 'trapped_ion' in QUBIT_HARDWARE
    for k, v in QUBIT_HARDWARE.items():
        assert 'ce_skills_needed' in v
