"""
Graph Theory: Spectral Graph Theory, Packet Switching, Signal Processors at Nodes

WIRE FRAME = GRAPH:
  Every photonic circuit, network, or signal chain is a graph.
  Nodes = signal processors. Edges = wires/fibers/waveguides.
  The Laplacian matrix L = D - A tells you EVERYTHING about wave propagation
  on the graph — eigenvalues are the resonant frequencies.

SYNTAX (0, 1, 2):
  0 edges from node: isolated node. Signal trapped. No flow.
  1 edge from node: leaf. One input or one output. Linear chain.
  2 edges from node: path node. Signal passes through. Series circuit.
  k edges: hub node. Star topology. Bus architecture.

GRAPH LAPLACIAN:
  A = adjacency matrix: A_{ij} = 1 if edge (i,j) exists, else 0
  D = degree matrix:   D_{ii} = degree(i) = sum of row i of A
  L = D - A           [Graph Laplacian]

  Properties:
    - L is symmetric (undirected graph)
    - All eigenvalues >= 0 (positive semidefinite)
    - Smallest eigenvalue = 0, eigenvector = [1,1,...,1] (DC component)
    - Second smallest eigenvalue = algebraic connectivity = Fiedler value
    - Fiedler value > 0 iff graph is connected
    - Largest eigenvalue <= max_degree + max_degree (norm bound)

SPECTRAL GRAPH THEORY:
  Graph Fourier Transform: project node signals onto eigenvectors of L
    x_hat = U^T * x    (U = eigenvectors of L)
  Analogous to: DFT for regular grid = GFT for arbitrary graph
  Graph convolution: (h * x)_hat[k] = h_hat[k] * x_hat[k]
  Graph neural network (GCN): H = sigma(D^{-1/2} A D^{-1/2} H W)

PACKET SWITCHING NODE = SIGNAL PROCESSOR:
  Each node applies a transfer function H(z) to incoming signal.
  H(z) = Z-transform of node's impulse response.
  Network = graph of transfer functions.
  Steady-state: Y(z) = H(z) * X(z) at each node.
  Stability: all poles of H(z) inside unit circle |z| < 1.
  Dispersion: H(f) = exp(j*pi*D*f^2) in fiber nodes.

CONNECTION TO THIS REPO:
  GS algorithm on a fiber = signal flow graph with two nodes:
    Node 1: H(f) dispersion propagator (forward)
    Node 2: Intensity constraint (measurement projection)
  Each iteration = one cycle around the graph.
  Convergence = stable fixed point of the graph dynamical system.
  Spectral gap of the iteration matrix = convergence rate.
"""
import math
import numpy as np
import sympy as sp

# ============================================================
# Graph Construction
# ============================================================

def make_graph(n_nodes, edges, directed=False):
    """
    Build adjacency matrix and Laplacian for a graph.

    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    edges : list of (i, j) or (i, j, weight) tuples.
    directed : bool
        If False, edges are symmetric.

    Returns
    -------
    dict with 'A' (adjacency), 'L' (Laplacian), 'degree' (degree vector).
    """
    A = np.zeros((n_nodes, n_nodes))
    for e in edges:
        i, j = e[0], e[1]
        w = e[2] if len(e) > 2 else 1.0
        A[i, j] += w
        if not directed:
            A[j, i] += w
    degree = A.sum(axis=1)
    D = np.diag(degree)
    L = D - A
    return {'A': A, 'L': L, 'degree': degree, 'n_nodes': n_nodes, 'edges': edges}


def graph_spectrum(g):
    """
    Compute eigenvalues and eigenvectors of the Laplacian.

    Returns eigenvalues sorted ascending, eigenvectors as columns.
    Fiedler value = second smallest eigenvalue.
    Fiedler vector = corresponding eigenvector = graph's 'shape'.
    """
    L = g['L']
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    fiedler_value = eigenvalues[1] if len(eigenvalues) > 1 else 0.0
    fiedler_vector = eigenvectors[:, 1] if len(eigenvalues) > 1 else None
    return {
        'eigenvalues': eigenvalues.tolist(),
        'eigenvectors': eigenvectors.tolist(),
        'fiedler_value': float(fiedler_value),
        'fiedler_vector': fiedler_vector.tolist() if fiedler_vector is not None else [],
        'is_connected': bool(fiedler_value > 1e-10),
        'n_components': int(np.sum(eigenvalues < 1e-10)),
    }


# ============================================================
# Graph Fourier Transform (GFT)
# ============================================================

def graph_fourier_transform(g, x):
    """
    Graph Fourier Transform (GFT): project signal x onto Laplacian eigenvectors.

    For a regular 1D grid with N nodes and edges to neighbors:
      L has eigenvalues 2 - 2*cos(k*pi/(N+1))
      Eigenvectors = sine modes (DST-like basis)
      GFT = Discrete Sine Transform
    For arbitrary graph:
      GFT = U^T * x    where U = eigenvector matrix of L
    Inverse GFT: x = U * x_hat

    This is the graph analog of the DFT.
    In signal processing: each eigenvalue = spatial frequency.
    Small eigenvalue = smooth (slowly varying) mode.
    Large eigenvalue = oscillatory (rapidly varying) mode.

    TDGSA connection:
      GS algorithm = graph filter on the two-node {H(f), Pi_meas} graph.
      Spectral radius of iteration matrix determines convergence rate.
      Faster convergence = larger spectral gap = better diversity (|D| large).
    """
    spec = graph_spectrum(g)
    U = np.array(spec['eigenvectors'])
    x_arr = np.array(x, dtype=float)
    x_hat = U.T @ x_arr   # GFT coefficients
    x_reconstructed = U @ x_hat   # should equal x

    # Low-pass graph filter: zero out high-frequency components
    x_hat_lowpass = x_hat.copy()
    cutoff = len(x_hat) // 2
    x_hat_lowpass[cutoff:] = 0
    x_lowpass = U @ x_hat_lowpass

    return {
        'x_hat': x_hat.tolist(),
        'x_reconstructed': x_reconstructed.tolist(),
        'reconstruction_error': float(np.max(np.abs(x_reconstructed - x_arr))),
        'x_lowpass': x_lowpass.tolist(),
        'eigenvalues': spec['eigenvalues'],
        'interpretation': {
            'DC_component': float(x_hat[0]),
            'lowest_freq_power': float(x_hat[1]**2) if len(x_hat) > 1 else 0,
            'highest_freq_power': float(x_hat[-1]**2),
        },
    }


# ============================================================
# Packet Switching: Node as Signal Processor
# ============================================================

def packet_switching_network(topology='ring', n_nodes=8):
    """
    Model a packet-switching network as a graph of digital signal processors.

    Each node applies a transfer function H(z) to its buffer.
    H(z) = 1/(1 - a*z^{-1})  [IIR accumulator, pole at z=a]
    Stable if |a| < 1.

    Network topologies:
      'ring':  N nodes in a cycle. Circulant graph.
      'star':  1 hub node connected to N-1 leaves.
      'mesh':  2D grid. Used in data center networks.
      'complete': all-to-all. Quantum networks ideal.
      'path':  linear chain. Single fiber span.

    EVERY NODE IS A DIGITAL AUDIO VISUALIZER:
      Each node receives a packet stream (signal x[n]).
      Applies H(z) (its routing/processing algorithm).
      Outputs y[n] = H(z)*x[n] in z-domain.
      Spectral analysis at each node = FFT of packet inter-arrival times.
      Graph Laplacian spectrum = resonant modes of the network.
      A congestion wave travels at group velocity = d(omega)/dk.

    CONNECTION TO DISPATCH (this repo):
      The fiber dispersion chain is a PATH graph.
      Node 0: pulse source (E_in)
      Node 1: fiber propagator H(f)=exp(j*pi*D*f^2)
      Node 2: detector (intensity measurement)
      GS algorithm runs 'forward + backward' over this path graph.
      n_iter GS iterations = n_iter passes over the 3-node path.
    """
    if topology == 'ring':
        edges = [(i, (i+1)%n_nodes) for i in range(n_nodes)]
    elif topology == 'star':
        edges = [(0, i) for i in range(1, n_nodes)]
    elif topology == 'path':
        edges = [(i, i+1) for i in range(n_nodes-1)]
    elif topology == 'complete':
        edges = [(i,j) for i in range(n_nodes) for j in range(i+1,n_nodes)]
    elif topology == 'mesh':
        side = int(np.ceil(np.sqrt(n_nodes)))
        edges = []
        for r in range(side):
            for col in range(side):
                node = r*side + col
                if node >= n_nodes: continue
                if col+1 < side and r*side+(col+1) < n_nodes:
                    edges.append((node, r*side+(col+1)))
                if r+1 < side and (r+1)*side+col < n_nodes:
                    edges.append((node, (r+1)*side+col))
    else:
        raise ValueError(f"Unknown topology: {topology}")

    g = make_graph(n_nodes, edges)
    spec = graph_spectrum(g)

    # Transfer function H(z) at each node: leaky integrator
    # H(z) = 1 / (1 - a*z^{-1})
    # In frequency domain: H(e^{j*omega}) = 1 / (1 - a*e^{-j*omega})
    a = 0.9   # stability: |a| < 1
    omega_arr = np.linspace(0, np.pi, 400)
    H_node = 1.0 / (1 - a*np.exp(-1j*omega_arr))
    H_mag = np.abs(H_node)
    H_phase = np.angle(H_node)

    # Simulate signal propagation along path
    n_steps = 100
    x_in = np.zeros(n_steps); x_in[0] = 1.0   # impulse
    y_node = np.zeros(n_steps)
    for t in range(1, n_steps):
        y_node[t] = a * y_node[t-1] + x_in[t]
    # Packet inter-arrival time spectrum (FFT)
    Y_freq = np.fft.rfft(y_node)

    return {
        'topology': topology,
        'n_nodes': n_nodes,
        'edges': edges,
        'graph': {
            'A': g['A'].tolist(),
            'L': g['L'].tolist(),
            'degree': g['degree'].tolist(),
        },
        'spectrum': {
            'eigenvalues': spec['eigenvalues'],
            'fiedler': spec['fiedler_value'],
            'is_connected': spec['is_connected'],
        },
        'node_transfer_fn': {
            'formula': 'H(z) = 1/(1 - 0.9*z^{-1})',
            'omega_rad': omega_arr.tolist(),
            'H_mag': H_mag.tolist(),
            'H_phase_deg': np.degrees(H_phase).tolist(),
            'pole_at_z': 0.9,
            'stable': True,
        },
        'impulse_response': {
            'y_node': y_node.tolist(),
            'Y_freq_mag': np.abs(Y_freq).tolist(),
        },
        'TDGSA_connection': (
            f'TDGSA = 3-node path graph. '
            f'GS n_iter iterations = n_iter path traversals. '
            f'Spectral gap of Laplacian = convergence rate. '
            f'Fiedler value = {spec["fiedler_value"]:.3f} for {topology} topology.'
        ),
    }


# ============================================================
# Graph Laplacian as Discrete Diffusion Operator
# ============================================================

def graph_diffusion(g, x0, n_steps=50, alpha=0.1):
    """
    Diffuse a signal x0 over the graph using the graph Laplacian.

    Diffusion equation on graph:
      dx/dt = -alpha * L * x
    Solution: x(t) = exp(-alpha * L * t) * x0
    Discrete update: x[k+1] = (I - alpha*L) * x[k]

    This is IDENTICAL to:
      Heat equation: du/dt = kappa * laplacian(u)
      Electric network: current flows from high to low potential
      Random walk: probability diffuses on graph
      GS convergence: error diffuses to zero over iterations

    The diffusion speed is set by the Laplacian eigenvalues.
    Slow modes (small eigenvalue) persist.
    Fast modes (large eigenvalue) decay quickly.
    Same physics as GVD in fiber: different frequencies disperse at different speeds.

    AUDIO VISUALIZER INTERPRETATION:
      x0 = initial signal (spectrum) loaded at one node.
      Each step: signal spreads to neighboring nodes.
      Final state: uniform distribution (DC mode only).
      Middle states: transient spatial patterns = musical notes on the graph.
    """
    n = g['n_nodes']
    L = g['L']
    I = np.eye(n)
    M = I - alpha * L   # diffusion update matrix

    x = np.array(x0, dtype=float)
    history = [x.tolist()]
    for _ in range(n_steps):
        x = M @ x
        history.append(x.tolist())

    spec = graph_spectrum(g)
    # Decay rates of each mode
    decay_rates = [1 - alpha*lam for lam in spec['eigenvalues']]

    return {
        'history': history,
        'final_state': history[-1],
        'steady_state': [float(np.mean(x0))] * n,
        'decay_rates_per_mode': decay_rates,
        'time_constants_steps': [
            1/(-np.log(abs(r))) if abs(r) < 1 and abs(r) > 1e-10 else float('inf')
            for r in decay_rates
        ],
        'fastest_decaying_mode': int(np.argmin(decay_rates)),
        'slowest_decaying_mode': 1,   # DC mode never decays
        'connection': {
            'heat_equation': 'dx/dt = -alpha*L*x  (same as du/dt = kappa*nabla^2 u)',
            'GVD': 'Dispersion H(f) = exp(-j*beta2*omega^2*L/2) = graph diffusion in freq domain',
            'GS': 'GS error diffuses to zero; rate = spectral gap of iteration matrix',
            'audio': 'Each eigenmode = one harmonic; graph diffusion = chord decay',
        },
    }


# ============================================================
# Spectral Graph Theory: Complete Analysis
# ============================================================

def spectral_graph_theory():
    """
    Complete spectral graph theory reference: L, eigenvalues, GCN, clustering.

    Key results:
      1. Fiedler value > 0  <=>  graph is connected
      2. k zero eigenvalues <=>  k connected components
      3. Fiedler vector bipartition: sign of each entry indicates which side of cut
      4. Graph coloring bound: chromatic number <= max eigenvalue + 1
      5. Expander graph: large Fiedler value = fast mixing = robust network

    Graph Neural Network (GCN):
      Kipf & Welling 2017: H^{(l+1)} = sigma(A_hat * H^{(l)} * W^{(l)})
      A_hat = D^{-1/2} (A + I) D^{-1/2}   [normalized with self-loops]
      This is a graph convolution: smooth out node features.
      Equivalent to: spectral filtering with H_hat(lambda) = 1/(1+lambda)

    Compare to this repo:
      GS algorithm on graph: at each node, apply constraint (amplitude or phase).
      Weights W^{(l)} = learned constraint matrices.
      Depth = number of GS iterations.
      GS CONVERGENCE = GCN CONVERGENCE = controlled by spectral radius.
    """
    # Example: Petersen graph (10 nodes, each degree 3, highly symmetric)
    # Used in network design for high connectivity
    petersen_edges = [
        (0,1),(1,2),(2,3),(3,4),(4,0),       # outer pentagon
        (0,5),(1,6),(2,7),(3,8),(4,9),        # spokes
        (5,7),(7,9),(9,6),(6,8),(8,5),        # inner pentagram
    ]
    g_petersen = make_graph(10, petersen_edges)
    spec_petersen = graph_spectrum(g_petersen)

    # Path graph (fiber = linear chain)
    n_path = 20
    path_edges = [(i, i+1) for i in range(n_path-1)]
    g_path = make_graph(n_path, path_edges)
    spec_path = graph_spectrum(g_path)

    # Complete graph K_6
    k6_edges = [(i,j) for i in range(6) for j in range(i+1,6)]
    g_k6 = make_graph(6, k6_edges)
    spec_k6 = graph_spectrum(g_k6)

    # Normalized Laplacian: L_norm = D^{-1/2} L D^{-1/2}
    A = g_petersen['A']; D_deg = np.diag(g_petersen['degree'])
    D_inv_sqrt = np.diag(1/np.sqrt(g_petersen['degree'] + 1e-12))
    L_norm = D_inv_sqrt @ g_petersen['L'] @ D_inv_sqrt
    eval_norm = np.linalg.eigvalsh(L_norm)

    # GCN update (one layer)
    A_hat = D_inv_sqrt @ (g_petersen['A'] + np.eye(10)) @ D_inv_sqrt
    H_features = np.random.randn(10, 3)   # 3 features per node
    W = np.random.randn(3, 2)             # weight matrix
    H_out = np.tanh(A_hat @ H_features @ W)

    # Graph coloring bound
    max_eigenvalue = float(max(spec_petersen['eigenvalues']))
    chromatic_bound = int(max_eigenvalue) + 1

    return {
        'Petersen': {
            'n_nodes': 10, 'degree': 3,
            'eigenvalues': spec_petersen['eigenvalues'],
            'fiedler': spec_petersen['fiedler_value'],
            'is_connected': spec_petersen['is_connected'],
            'chromatic_bound': chromatic_bound,
            'note': 'Petersen graph: high symmetry, high connectivity, used in network design',
        },
        'path_graph': {
            'n_nodes': n_path,
            'eigenvalues': spec_path['eigenvalues'],
            'fiedler': spec_path['fiedler_value'],
            'connection': 'Path graph = linear fiber chain. Eigenmodes = standing waves = cavity modes.',
        },
        'complete_graph_K6': {
            'n_nodes': 6,
            'eigenvalues': spec_k6['eigenvalues'],
            'fiedler': spec_k6['fiedler_value'],
            'note': 'K_N has N-1 degenerate eigenvalue = N (all-to-all connectivity)',
        },
        'normalized_Laplacian': {
            'eigenvalues': eval_norm.tolist(),
            'range': '0 <= lambda <= 2, lambda=2 iff bipartite',
        },
        'GCN': {
            'formula': 'H_out = sigma(A_hat H W)',
            'A_hat': 'D^{-1/2}(A+I)D^{-1/2}',
            'H_out_shape': list(H_out.shape),
            'analogy': 'GCN layer = GS iteration. Depth = n_iter. W = constraint weights.',
        },
        'key_theorems': {
            '1': 'Fiedler > 0  <=>  connected graph',
            '2': 'k zero eigenvalues <=>  k components',
            '3': 'Fiedler vector = optimal graph bisection (spectral clustering)',
            '4': 'Expander: Fiedler/degree_max > const -> fast random walk -> secure network',
            '5': 'GFT: x_hat = U^T x, same as DFT on regular grid',
        },
        'syntax': {
            '0': 'L has 0 eigenvalue -> DC component. Sum of all node values conserved.',
            '1': 'Fiedler vector components: +/- sign -> bipartition of graph.',
            '2': 'Two zero eigenvalues -> two disconnected components. Cut the graph.',
        },
    }


# ============================================================
# GS Algorithm as Graph Dynamical System
# ============================================================

def gs_as_graph(D=-600, n_iter=50, N=256):
    """
    Model the GS algorithm as a dynamical system on a 3-node graph.

    Nodes:
      0: Input plane (measured intensity I_in)
      1: Output plane (measured intensity I_out after dispersion)
      2: Current estimate

    Edges:
      0->2: amplitude constraint (set |E_in| from measurement)
      2->1: propagation H(f) = exp(j*pi*D*f^2)
      1->2: amplitude constraint (set |E_out| from measurement)
      2->0: inverse propagation H^{-1}(f)

    Each GS iteration = one cycle around this graph.
    Convergence = fixed point of the cycle map T = T4 o T3 o T2 o T1.
    Convergence rate = spectral radius of T near fixed point.

    DIVERSITY = |D|:
      Small |D|: H(f) ~ 1, T ~ identity, convergence slow.
      Large |D|: H(f) mixes phases strongly, T has smaller spectral radius.
      This is why |D| >= 5000 is required for reliable convergence.
    """
    if abs(D) < 100:
        raise ValueError(f"|D| = {abs(D)} < 100. Need |D| >= 100 for phase diversity.")
    if n_iter <= 0:
        raise ValueError(f"n_iter = {n_iter} must be > 0.")
    if N <= 0:
        raise ValueError(f"N = {N} must be > 0.")

    f = np.fft.fftfreq(N)
    H = np.exp(1j * np.pi * D * f**2)

    # Simulate GS convergence with a known target
    np.random.seed(42)
    E_true = np.exp(1j * np.random.randn(N))   # true complex field
    I_in  = np.abs(E_true)**2
    I_out = np.abs(np.fft.fft(E_true * H))**2

    # GS iterations
    E_est = np.sqrt(I_in) * np.exp(1j * np.zeros(N))   # initial guess: zero phase
    errors = []
    correlations = []

    for _ in range(n_iter):
        # Forward: apply dispersion, constrain amplitude in output
        E_fwd = np.fft.fft(E_est * H)
        E_fwd = np.sqrt(I_out) * np.exp(1j * np.angle(E_fwd))
        # Backward: apply inverse dispersion, constrain amplitude in input
        E_back = np.fft.ifft(E_fwd) / H
        E_est = np.sqrt(I_in) * np.exp(1j * np.angle(E_back))

        # Error metric
        corr = float(np.abs(np.sum(E_est * np.conj(E_true)))**2 /
                     (np.sum(np.abs(E_est)**2) * np.sum(np.abs(E_true)**2)))
        errors.append(1 - corr)
        correlations.append(corr)

    # Spectral radius of H (proxy for convergence rate)
    # Large |D| -> H oscillates fast -> phases spread -> faster convergence
    phase_range = float(np.max(np.angle(H)) - np.min(np.angle(H)))
    diversity = min(float(np.abs(D)), 5000) / 5000   # normalized diversity [0,1]

    return {
        'D': D, 'n_iter': n_iter, 'N': N,
        'convergence': {
            'errors': errors,
            'correlations': correlations,
            'final_correlation': float(correlations[-1]),
            'converged': float(correlations[-1]) > 0.9,
        },
        'graph_view': {
            'nodes': ['Input plane (I_in)', 'Output plane (I_out)', 'GS estimate'],
            'edges': [
                '0->2: amplitude replace (sqrt(I_in)*exp(j angle))',
                '2->1: dispersion H(f)',
                '1->2: amplitude replace (sqrt(I_out)*exp(j angle))',
                '2->0: inverse dispersion H^{-1}(f)',
            ],
            'cycle_length': 4,
            'fixed_point': 'true phase (when converged)',
        },
        'diversity': {
            'D_value': D,
            'phase_range_rad': phase_range,
            'normalized_diversity': diversity,
            'recommendation': '|D| >= 5000 for reliable convergence',
            'warning': '' if abs(D) >= 5000 else f'|D|={abs(D)} < 5000; convergence may be slow',
        },
    }


def demo():
    print("=== GRAPH THEORY: PACKET SWITCHING + SIGNAL PROCESSORS ===\n")

    print("--- Graph Laplacian: Ring Topology ---")
    g_ring = make_graph(6, [(i,(i+1)%6) for i in range(6)])
    spec = graph_spectrum(g_ring)
    print(f"  Fiedler value (ring 6): {spec['fiedler_value']:.3f}")
    print(f"  Connected: {spec['is_connected']}")
    print(f"  Components: {spec['n_components']}")
    print(f"  Eigenvalues: {[round(v,3) for v in spec['eigenvalues']]}")

    print("\n--- Graph Fourier Transform ---")
    x0 = [1,0,0,0,0,0]   # impulse at node 0
    gft = graph_fourier_transform(g_ring, x0)
    print(f"  GFT error: {gft['reconstruction_error']:.2e}")
    print(f"  DC component: {gft['interpretation']['DC_component']:.4f}")

    print("\n--- Packet Switching Network ---")
    for topo in ['ring', 'star', 'path', 'complete']:
        net = packet_switching_network(topology=topo, n_nodes=8)
        print(f"  {topo:8s}: fiedler={net['spectrum']['fiedler']:.3f}, "
              f"connected={net['spectrum']['is_connected']}")

    print("\n--- Graph Diffusion ---")
    g_path = make_graph(8, [(i,i+1) for i in range(7)])
    x_init = [1.0,0,0,0,0,0,0,0]
    diff = graph_diffusion(g_path, x_init, n_steps=30)
    print(f"  Initial: {x_init}")
    print(f"  After 30 steps: {[round(v,3) for v in diff['final_state']]}")

    print("\n--- Spectral Graph Theory ---")
    sgt = spectral_graph_theory()
    print(f"  Petersen fiedler: {sgt['Petersen']['fiedler']:.3f}")
    print(f"  Petersen chromatic bound: {sgt['Petersen']['chromatic_bound']}")
    print(f"  Path-20 fiedler: {sgt['path_graph']['fiedler']:.4f}")
    print(f"  K6 fiedler: {sgt['complete_graph_K6']['fiedler']:.2f}")
    for k,v in sgt['key_theorems'].items():
        print(f"  Theorem {k}: {v}")

    print("\n--- GS Algorithm as Graph ---")
    gs = gs_as_graph(D=-600, n_iter=50)
    print(f"  D=-600: final_corr={gs['convergence']['final_correlation']:.4f}, "
          f"converged={gs['convergence']['converged']}")
    gs2 = gs_as_graph(D=-5000, n_iter=50)
    print(f"  D=-5000: final_corr={gs2['convergence']['final_correlation']:.4f}, "
          f"converged={gs2['convergence']['converged']}")

    print("\n=== GRAPH THEORY COMPLETE ===")


if __name__ == '__main__':
    demo()
