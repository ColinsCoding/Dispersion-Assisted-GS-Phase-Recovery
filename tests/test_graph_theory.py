"""Tests for dgs/graph_theory.py"""
import numpy as np
import pytest

try:
    from dgs.graph_theory import (
        make_graph, graph_spectrum, graph_fourier_transform,
        packet_switching_network, graph_diffusion,
        spectral_graph_theory, gs_as_graph,
    )
except ImportError:
    from graph_theory import (
        make_graph, graph_spectrum, graph_fourier_transform,
        packet_switching_network, graph_diffusion,
        spectral_graph_theory, gs_as_graph,
    )


class TestMakeGraph:
    def test_A_symmetric(self):
        g = make_graph(4, [(0,1),(1,2),(2,3)])
        A = np.array(g['A'])
        assert np.allclose(A, A.T)

    def test_degree_sum_equals_2_edges(self):
        edges = [(0,1),(1,2),(2,3)]
        g = make_graph(4, edges)
        assert int(np.sum(g['degree'])) == 2*len(edges)

    def test_L_row_sum_zero(self):
        g = make_graph(5, [(i,i+1) for i in range(4)])
        L = np.array(g['L'])
        assert np.allclose(L.sum(axis=1), 0)


class TestGraphSpectrum:
    def test_smallest_eigenvalue_zero(self):
        g = make_graph(4, [(0,1),(1,2),(2,3)])
        spec = graph_spectrum(g)
        assert spec['eigenvalues'][0] == pytest.approx(0.0, abs=1e-10)

    def test_connected_ring_fiedler_positive(self):
        g = make_graph(6, [(i,(i+1)%6) for i in range(6)])
        spec = graph_spectrum(g)
        assert spec['fiedler_value'] > 0
        assert spec['is_connected'] is True

    def test_disconnected_graph_two_zeros(self):
        # Two separate edges: 0-1 and 2-3 (not connected)
        g = make_graph(4, [(0,1),(2,3)])
        spec = graph_spectrum(g)
        assert spec['n_components'] == 2

    def test_path_fiedler_small(self):
        # Path graph: low Fiedler (weakly connected)
        g = make_graph(10, [(i,i+1) for i in range(9)])
        spec = graph_spectrum(g)
        assert 0 < spec['fiedler_value'] < 1.0

    def test_complete_K5_fiedler_equals_N(self):
        # K_N: fiedler = N
        N = 5
        edges = [(i,j) for i in range(N) for j in range(i+1,N)]
        g = make_graph(N, edges)
        spec = graph_spectrum(g)
        assert spec['fiedler_value'] == pytest.approx(N, abs=1e-8)

    def test_eigenvalues_nonnegative(self):
        g = make_graph(6, [(i,(i+1)%6) for i in range(6)])
        spec = graph_spectrum(g)
        assert all(v >= -1e-10 for v in spec['eigenvalues'])


class TestGraphFourierTransform:
    def test_reconstruction_exact(self):
        g = make_graph(6, [(i,(i+1)%6) for i in range(6)])
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        result = graph_fourier_transform(g, x)
        assert result['reconstruction_error'] < 1e-10

    def test_DC_component_is_mean(self):
        g = make_graph(4, [(i,(i+1)%4) for i in range(4)])
        x = [1.0, 1.0, 1.0, 1.0]   # constant signal
        result = graph_fourier_transform(g, x)
        # All energy in DC
        assert abs(result['interpretation']['highest_freq_power']) < 1e-10

    def test_lowpass_smooth(self):
        g = make_graph(6, [(i,(i+1)%6) for i in range(6)])
        x = [(-1)**i for i in range(6)]   # alternating signal
        result = graph_fourier_transform(g, x)
        # Lowpass zeros out high freq -> smoother
        x_lp = np.array(result['x_lowpass'])
        x_orig = np.array(x)
        assert np.std(x_lp) < np.std(x_orig)


class TestPacketSwitching:
    @pytest.mark.parametrize("topo", ['ring', 'star', 'path', 'complete', 'mesh'])
    def test_topology_connected(self, topo):
        net = packet_switching_network(topology=topo, n_nodes=6)
        assert net['spectrum']['is_connected'] is True

    def test_H_node_stable(self):
        net = packet_switching_network()
        assert net['node_transfer_fn']['stable'] is True

    def test_H_mag_peaks_at_DC(self):
        net = packet_switching_network()
        H_mag = np.array(net['node_transfer_fn']['H_mag'])
        # H(omega)=1/(1-a*e^{-jw}): max at omega=0 (DC)
        assert H_mag[0] == max(H_mag)

    def test_complete_largest_fiedler(self):
        # Complete graph has largest Fiedler value
        net_complete = packet_switching_network('complete', n_nodes=6)
        net_ring = packet_switching_network('ring', n_nodes=6)
        assert net_complete['spectrum']['fiedler'] > net_ring['spectrum']['fiedler']

    def test_TDGSA_connection_present(self):
        net = packet_switching_network()
        assert 'TDGSA' in net['TDGSA_connection'] or 'GS' in net['TDGSA_connection']


class TestGraphDiffusion:
    def test_signal_spreads(self):
        g = make_graph(6, [(i,(i+1)%6) for i in range(6)])
        x0 = [1,0,0,0,0,0]
        diff = graph_diffusion(g, x0, n_steps=200)
        final = np.array(diff['final_state'])
        # Should approach uniform (1/6 each)
        assert np.std(final) < np.std(np.array(x0)) * 0.5

    def test_total_conserved(self):
        g = make_graph(4, [(i,(i+1)%4) for i in range(4)])
        x0 = [1.0, 2.0, 3.0, 4.0]
        diff = graph_diffusion(g, x0, n_steps=100, alpha=0.1)
        # Total signal conserved (diffusion only redistributes)
        assert sum(diff['final_state']) == pytest.approx(sum(x0), rel=1e-6)

    def test_decay_rates_present(self):
        g = make_graph(4, [(i,(i+1)%4) for i in range(4)])
        diff = graph_diffusion(g, [1,0,0,0])
        assert len(diff['decay_rates_per_mode']) == 4

    def test_connections_present(self):
        g = make_graph(4, [(i,(i+1)%4) for i in range(4)])
        diff = graph_diffusion(g, [1,0,0,0])
        assert 'GVD' in diff['connection'] and 'GS' in diff['connection']


class TestSpectralGraphTheory:
    def test_Petersen_fiedler_positive(self):
        sgt = spectral_graph_theory()
        assert sgt['Petersen']['fiedler'] > 0

    def test_Petersen_connected(self):
        sgt = spectral_graph_theory()
        assert sgt['Petersen']['is_connected'] is True

    def test_K6_fiedler_equals_6(self):
        sgt = spectral_graph_theory()
        assert sgt['complete_graph_K6']['fiedler'] == pytest.approx(6.0, abs=1e-8)

    def test_5_key_theorems(self):
        sgt = spectral_graph_theory()
        assert len(sgt['key_theorems']) >= 5

    def test_GCN_output_shape(self):
        sgt = spectral_graph_theory()
        # Input 3 features -> output 2 features, 10 nodes
        assert sgt['GCN']['H_out_shape'] == [10, 2]

    def test_norm_Laplacian_range(self):
        sgt = spectral_graph_theory()
        evals = np.array(sgt['normalized_Laplacian']['eigenvalues'])
        assert np.all(evals >= -1e-10)
        assert np.all(evals <= 2.0 + 1e-10)

    def test_syntax_keys(self):
        sgt = spectral_graph_theory()
        for k in ['0', '1', '2']:
            assert k in sgt['syntax']


class TestGSAsGraph:
    def test_large_D_converges(self):
        r = gs_as_graph(D=-5000, n_iter=50)
        assert r['convergence']['final_correlation'] > 0.4

    def test_small_D_slower(self):
        r_small = gs_as_graph(D=-600, n_iter=50)
        r_large = gs_as_graph(D=-5000, n_iter=50)
        assert r_large['convergence']['final_correlation'] >= r_small['convergence']['final_correlation'] - 0.05

    def test_graph_view_has_4_edges(self):
        r = gs_as_graph(D=-5000)
        assert len(r['graph_view']['edges']) == 4

    def test_diversity_normalized(self):
        r = gs_as_graph(D=-5000)
        assert 0 <= r['diversity']['normalized_diversity'] <= 1.0

    def test_invalid_D_raises(self):
        with pytest.raises(ValueError):
            gs_as_graph(D=-50)   # |D| < 100

    def test_invalid_n_iter_raises(self):
        with pytest.raises(ValueError):
            gs_as_graph(D=-5000, n_iter=0)

    def test_correlations_positive(self):
        r = gs_as_graph(D=-5000, n_iter=20)
        corrs = r['convergence']['correlations']
        assert all(c >= 0 for c in corrs)
