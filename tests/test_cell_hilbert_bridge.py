"""Tests for dgs.cell_hilbert_bridge: ion-channel master equation <-> dephased dimer."""
import numpy as np
import pytest
import sympy as sp

from dgs import cell_hilbert_bridge as c


def test_thermal_energy_body_temperature():
    assert c.thermal_energy_cm() == pytest.approx(215.5, abs=0.2)
    with pytest.raises(ValueError):
        c.thermal_energy_cm(0)


def test_channel_master_equation_conserves_probability_and_relaxes():
    Q = c.channel_generator(2.0, 3.0)
    assert np.allclose(Q.sum(axis=0), 0)
    p = c.master_evolve(Q, [1, 0], np.linspace(0, 5, 50))
    assert np.allclose(p.sum(axis=1), 1)
    assert p[-1, 1] == pytest.approx(c.stationary_open_probability(2.0, 3.0), abs=1e-6)
    assert c.derive_master_stationary() == sp.Symbol("k_o", positive=True) / (
        sp.Symbol("k_o", positive=True) + sp.Symbol("k_c", positive=True))
    with pytest.raises(ValueError):
        c.channel_generator(-1, 1)


def test_telegraph_time_average_is_ergodic():
    t, s = c.telegraph_trace(1.0, 4.0, 20000.0, rng=0)
    assert c.time_average_open(t, s) == pytest.approx(0.2, abs=0.01)


def test_single_channel_current_reverses_at_nernst():
    E = 1e3 * c.nernst_potential(5.0, 140.0)
    assert c.single_channel_current_pA(E) == pytest.approx(0, abs=1e-12)
    assert c.single_channel_current_pA(0) > 0        # K+ flows out at 0 mV


def test_rabi_formula_symbolic_and_numeric():
    assert c.derive_rabi()[2] == 0
    H = c.dimer_hamiltonian(100, 40)
    ts = np.linspace(0, 400, 9)
    rho = c.evolve_density(H, c.projector(c.site_ket(0)), ts)
    assert np.allclose(rho[:, 1, 1].real, c.rabi_population(ts, 100, 40), atol=1e-10)
    assert all(c.purity(r) == pytest.approx(1) for r in rho)   # unitary: stays pure


def test_dephasing_destroys_coherence_and_purity():
    H = c.dimer_hamiltonian(0, 50)
    rho = c.evolve_density(H, c.projector(c.site_ket(0)), [10000.0], gamma=0.2)[0]
    assert np.allclose(rho, np.eye(2) / 2, atol=1e-3)
    assert c.purity(rho) == pytest.approx(0.5, abs=1e-3)
    assert c.von_neumann_entropy_bits(rho) == pytest.approx(1, abs=1e-3)
    assert np.linalg.norm(c.bloch_vector(rho)) < 1e-2


def test_strong_dephasing_reduces_to_ion_channel_master_equation():
    d, J, g = 150.0, 30.0, 2.0
    k = c.incoherent_rate(d, J, g)
    assert c.derive_incoherent_rate() == sp.sympify("2*J**2*gamma/(Delta**2+gamma**2)",
        locals={s: sp.Symbol(s, positive=True) for s in ("J", "gamma", "Delta")})
    ts = np.linspace(0, 3 / k, 7)
    quantum = c.evolve_density(c.dimer_hamiltonian(d, J), c.projector(c.site_ket(0)), ts, gamma=g)
    classical = c.master_evolve(c.channel_generator(k, k), [1, 0], ts)
    assert np.allclose(quantum[:, 1, 1].real, classical[:, 1], atol=5e-3)


def test_environment_assisted_transport_has_interior_optimum():
    gs = np.logspace(-4, 2, 25)
    eta = np.array([c.transfer_efficiency(300, 50, g) for g in gs])
    i = int(np.argmax(eta))
    assert 0 < i < len(gs) - 1
    assert eta[i] > 2 * eta[0] and eta[i] > 10 * eta[-1]
    assert gs[i] == pytest.approx(c.cm_to_rad_per_fs(300), rel=0.6)   # peak near gamma ~ Delta


def test_hilbert_memory_scaling():
    assert c.hilbert_memory_bytes(10, "vector") == 16 * 1024
    assert c.hilbert_memory_bytes(10, "density") == 16 * 1024**2
    assert c.hilbert_memory_bytes(7, "density", "single") == 16 * 49
    with pytest.raises(ValueError):
        c.hilbert_memory_bytes(0)
