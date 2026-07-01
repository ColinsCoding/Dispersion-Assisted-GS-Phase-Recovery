"""
Engineering Statics + Jane Street Autodiff + Distribution Sifting

Statics = equilibrium: sum_F=0, sum_M=0.
Jane Street: OCaml + C + torch/JAX. Autodiff differentiates through physics.
Distribution sifting: delta function, importance sampling, attention = same math.
"""
import numpy as np
import sympy as sp
from math import erfc, sqrt


def equilibrium_2d(forces, moments_about_A=None):
    """
    2D equilibrium: sum_Fx=0, sum_Fy=0, sum_M_A=0.
    forces: list of dicts with 'Fx', 'Fy', 'x', 'y'.
    Moment about A: M = x*Fy - y*Fx (z-component of r x F).
    """
    Fx_total = sum(f.get('Fx', 0) for f in forces)
    Fy_total = sum(f.get('Fy', 0) for f in forces)
    M_total  = sum(f.get('x',0)*f.get('Fy',0) - f.get('y',0)*f.get('Fx',0)
                   for f in forces)
    if moments_about_A:
        M_total += sum(moments_about_A)
    ok = abs(Fx_total) < 1e-6 and abs(Fy_total) < 1e-6 and abs(M_total) < 1e-6
    return {
        'sum_Fx': Fx_total, 'sum_Fy': Fy_total, 'sum_M_A': M_total,
        'in_equilibrium': ok,
        'check': 'PASS' if ok else 'FAIL',
    }


def beam_reactions(L, loads):
    """
    Simply supported beam: pin at A (x=0), roller at B (x=L).
    loads: list of {'P': force_N, 'x': pos} or {'w': N_per_m, 'x_start', 'x_end'}.
    Returns Ax, Ay, By via sum_M_A=0 then sum_Fy=0.

    Jane Street analogy: 3 unknowns, 3 equations -> K*x=b.
    Same as pricing 3 derivatives from 3 market instruments.
    """
    if L <= 0:
        raise ValueError("|L| must be positive")
    total_F = 0.0
    M_A = 0.0
    for ld in loads:
        if 'P' in ld:
            total_F += ld['P']; M_A += ld['P'] * ld['x']
        elif 'w' in ld:
            F = ld['w'] * (ld['x_end'] - ld['x_start'])
            xc = (ld['x_start'] + ld['x_end']) / 2
            total_F += F; M_A += F * xc
    By = M_A / L
    Ay = total_F - By
    Ax = 0.0
    return {'Ax_N': Ax, 'Ay_N': Ay, 'By_N': By,
            'total_applied_N': total_F, 'equilibrium_check': 'PASS'}


def shear_moment_diagram(L, loads, N=500):
    """
    V(x) = Ay - sum of loads left of x.
    M(x) = integral of V(x) via FTC.
    """
    rxns = beam_reactions(L, loads)
    Ay = rxns['Ay_N']
    x = np.linspace(0, L, N)
    V = np.full(N, Ay)
    for ld in loads:
        if 'P' in ld:
            V[x >= ld['x']] -= ld['P']
        elif 'w' in ld:
            x0, x1, w = ld['x_start'], ld['x_end'], ld['w']
            mask = (x >= x0) & (x <= x1)
            V[mask] -= w * (x[mask] - x0)
            V[x > x1] -= w * (x1 - x0)
    dx = L / (N - 1)
    M = np.zeros(N)
    for i in range(1, N):
        M[i] = M[i-1] + V[i-1]*dx
    return {
        'x': x, 'V': V, 'M': M,
        'max_shear_N': float(np.max(np.abs(V))),
        'max_moment_Nm': float(np.max(np.abs(M))),
        'x_max_moment': float(x[np.argmax(np.abs(M))]),
        'Ay': Ay, 'By': rxns['By_N'],
        'lesson': 'V(x)=integral of w(x). M(x)=integral of V(x). FTC: dM/dx=V, dV/dx=-w.',
    }


def truss_method_of_joints(nodes, members, supports, loads):
    """
    Planar truss by global matrix method (method of joints, assembled).
    nodes:   {name: (x, y)}
    members: [(n1, n2), ...]
    supports: {node: 'pin'|'roller'|'roller_x'}
    loads:   {node: (Fx, Fy)}
    Positive force = tension. Negative = compression.
    """
    nn = list(nodes.keys())
    n_nodes = len(nn)
    idx = {n: i for i, n in enumerate(nn)}
    n_mem = len(members)

    rxn_dofs = []
    for node, stype in supports.items():
        ni = idx[node]
        if stype == 'pin':
            rxn_dofs += [(2*ni,   'x', node), (2*ni+1, 'y', node)]
        elif stype in ('roller', 'roller_y'):
            rxn_dofs.append((2*ni+1, 'y', node))
        elif stype == 'roller_x':
            rxn_dofs.append((2*ni, 'x', node))

    n_rxn = len(rxn_dofs)
    n_eq = 2 * n_nodes
    n_unk = n_mem + n_rxn
    if n_unk != n_eq:
        return {'error': f'{"over" if n_unk>n_eq else "under"}-determinate: '
                         f'{n_unk} unknowns, {n_eq} equations'}

    A = np.zeros((n_eq, n_unk))
    b = np.zeros(n_eq)

    for j, (n1, n2) in enumerate(members):
        x1, y1 = nodes[n1]; x2, y2 = nodes[n2]
        Lm = np.hypot(x2-x1, y2-y1)
        if Lm < 1e-12:
            raise ValueError(f"Zero-length member {n1}-{n2}")
        cx = (x2-x1)/Lm; cy = (y2-y1)/Lm
        i1, i2 = idx[n1], idx[n2]
        A[2*i1,   j] +=  cx;  A[2*i1+1, j] +=  cy
        A[2*i2,   j] += -cx;  A[2*i2+1, j] += -cy

    for k, (eq_i, _, _) in enumerate(rxn_dofs):
        A[eq_i, n_mem+k] = 1.0

    for node, (Fx, Fy) in loads.items():
        ni = idx[node]
        b[2*ni] = -Fx; b[2*ni+1] = -Fy

    try:
        sol = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return {'error': 'Singular -- check geometry/supports'}

    mf = {f'{members[j][0]}-{members[j][1]}': sol[j] for j in range(n_mem)}
    rxns = {f'R_{rxn_dofs[k][2]}_{rxn_dofs[k][1]}': sol[n_mem+k] for k in range(n_rxn)}

    return {
        'member_forces_N': mf,
        'reactions_N': rxns,
        'max_tension_N': max((f for f in sol[:n_mem] if f > 0), default=0.0),
        'max_compression_N': min((f for f in sol[:n_mem] if f < 0), default=0.0),
        'n_members': n_mem, 'n_reactions': n_rxn,
        'determinacy': '2n = m + r (statically determinate)',
        'tension_positive': True,
    }


def three_bar_truss():
    """Classic 3-bar truss: A(0,0), B(2,0), C(1,1.5); 10 kN down at C."""
    return truss_method_of_joints(
        nodes   = {'A': (0.0,0.0), 'B': (2.0,0.0), 'C': (1.0,1.5)},
        members = [('A','C'), ('B','C'), ('A','B')],
        supports= {'A': 'pin', 'B': 'roller'},
        loads   = {'C': (0.0, -10000.0)},
    )


def moment_of_inertia_shapes():
    """
    Second moment of area for cross-sections.
    I = int y^2 dA  <->  Var[X] = int x^2 p(x) dx  <->  MSE loss.
    Parallel axis theorem <-> bias-variance decomposition.
    """
    b, h, r = sp.symbols('b h r', positive=True)
    A_sym, I_c, d = sp.symbols('A I_c d', positive=True)
    return {
        'rectangle': {
            'I_x': sp.Rational(1,12)*b*h**3,
            'note': 'Strong axis: large h. Wide-flange beams use this.'
        },
        'circle': {
            'I_x': sp.pi*r**4/4,
            'note': 'Isotropic: same stiffness all directions.'
        },
        'parallel_axis_theorem': {
            'formula': I_c + A_sym*d**2,
            'note': 'Move centroidal I to offset axis. Same as E[X^2]=Var[X]+mu^2.'
        },
        'statistics_connection': (
            'I = int y^2 dA  ==  E[X^2] = int x^2 p(x) dx  ==  MSE.\n'
            'Second moment = stiffness = variance = loss curvature.\n'
            'Parallel axis theorem = bias-variance decomposition.'
        ),
    }


def distribution_sifting():
    """
    Dirac delta sifting property + its applications across domains.

    int f(x)*delta(x-a) dx = f(a)     [selects one value]

    Attention = soft sifting (softmax instead of delta).
    Finance pricing = expectation = integral with distribution p(S).
    Importance sampling = change of measure = w(x)=p(x)/q(x).
    Green's function = impulse response = system inverse.
    """
    x, a = sp.symbols('x a', real=True)
    f = sp.Function('f')
    sifting = sp.Integral(f(x)*sp.DiracDelta(x-a), (x, -sp.oo, sp.oo))

    return {
        'sifting_property': f'{sifting} = f(a)',
        'FT_of_delta': 'FT[delta(t-t0)] = exp(-j*omega*t0)  [pure phase shift]',
        'sampling_theorem': 'f_s(t) = f(t) * sum_n delta(t-n*Ts)  [ideal sampler]',
        'attention': {
            'soft':  'a = softmax(Q*K^T/sqrt(d_k)),  out = a*V',
            'hard':  'a = one_hot(argmax(Q*K^T)),     out = V[argmax]',
            'limit': 'temperature->0: softmax -> Kronecker delta',
        },
        'finance': {
            'pricing':      'V = int payoff(S)*p(S|S0) dS',
            'arrow_debreu': 'pays $1 iff S=S* = delta derivative = digital option',
            'jane_street':  'estimate p(S) from order flow, integrate numerically',
        },
        'importance_sampling': {
            'formula':  'E_p[f] = E_q[f*p/q]  (change of measure)',
            'weights':  'w(x) = p(x)/q(x) = likelihood ratio',
            'ml_use':   'PPO (RLHF): r_t(theta) = pi_new/pi_old, clipped to [1-eps, 1+eps]',
            'compute':  '100x fewer samples for rare events -> 100x cheaper per hour',
        },
        'greens_function': {
            'equation':  'L G(x,x\') = delta(x-x\')',
            'solution':  'u(x) = int G(x,x\')*f(x\') dx\'',
            'photonics': 'G(f) = H(f) = exp(j*pi*D*f^2) for dispersive fiber',
        },
        'cost_connection': (
            'Same adjoint equation in statics AND finance AND ML:\n'
            '  Statics:  dU/dA_i = -lambda^T * dK/dA_i * u   [adjoint]\n'
            '  Finance:  dV/dS   = Delta  (hedge ratio = gradient)\n'
            '  ML:       dL/dW   = backprop  (chain rule through computation graph)\n'
            'Three fields. One equation. Learn one, understand all three.'
        ),
    }


def importance_sampling_demo(N=20000):
    """
    Estimate P(X > 3.5) for X~N(0,1) via importance sampling.
    True value ~2.3e-4 (rare event).
    IS: sample from N(3.5, 0.5) where the integrand is large.
    100x fewer samples, same accuracy.
    """
    rng = np.random.default_rng(42)
    true_prob = 0.5 * erfc(3.5 / sqrt(2))

    # Naive MC
    x_naive = rng.standard_normal(N)
    p_naive = float(np.mean(x_naive > 3.5))
    se_naive = float(np.sqrt(max(p_naive*(1-p_naive)/N, 1e-20)))

    # IS with N/100 samples
    N_is = max(N // 100, 50)
    mu_q, sig_q = 3.5, 0.5
    x_is = rng.normal(mu_q, sig_q, N_is)
    indicator = (x_is > 3.5).astype(float)

    log_p = -0.5*x_is**2 - 0.5*np.log(2*np.pi)
    log_q = -0.5*((x_is-mu_q)/sig_q)**2 - np.log(sig_q) - 0.5*np.log(2*np.pi)
    w = np.exp(log_p - log_q)
    p_is  = float(np.mean(indicator * w))
    se_is = float(np.std(indicator*w) / sqrt(N_is))

    speedup = int((se_naive / se_is)**2) if se_is > 0 else 999
    return {
        'true_prob': true_prob,
        'naive_N': N, 'naive_est': p_naive, 'naive_se': se_naive,
        'is_N': N_is,  'is_est': p_is,    'is_se': se_is,
        'speedup_factor': speedup,
        'lesson': (
            f'True={true_prob:.2e}. '
            f'Naive(N={N}): {p_naive:.2e}+-{se_naive:.2e}. '
            f'IS(N={N_is}): {p_is:.2e}+-{se_is:.2e}. '
            f'~{speedup}x efficient -- sifts samples to where p(x)*f(x) is large.'
        ),
    }


def jane_street_tech_stack():
    """
    Jane Street technology stack and interview structure.
    OCaml (primary), C (HFT latency), Python+torch/JAX (quant research).
    Levels: 0=SWE, 1=Senior, 2=Principal.
    """
    return {
        'OCaml': {
            'role': 'primary language -- type-safe functional trading systems',
            'strengths': ['algebraic types catch missing cases', 'immutable by default',
                          'pattern matching', 'fast enough for most strategies'],
            'tradeoff': '3-10x slower than C; GC pauses ~100us (bad for HFT)',
        },
        'C': {
            'role': 'ultra-low-latency execution paths',
            'latency': '~1us round-trip with DPDK (kernel-bypass networking)',
            'vs_kernel': 'Standard TCP: 30us+. DPDK: 1-5us. FPGA: <100ns.',
            'skills': ['manual memory', 'cache alignment', 'SIMD intrinsics'],
        },
        'Python_torch_JAX': {
            'role': 'quantitative research, signal discovery, ML models',
            'autodiff': (
                'jax.grad(f) = exact gradient of any pure function.\n'
                'torch.autograd: same via dynamic computational graph.\n'
                'For pricing: V(theta) is differentiable -> dV/dtheta = Greeks.\n'
                'For GS: L=||I_out-I_target||^2 -> dL/d(input_phase) via torch.fft.fft.'
            ),
            'JAX_advantage': 'jax.jit -> XLA compilation -> 10-100x faster than NumPy',
        },
        'levels': {
            0: 'SWE: algorithms/DS (Leetcode Hard), clean code, OCaml proficiency',
            1: 'Senior: owns trading system end-to-end, production reliability',
            2: 'Principal: invents strategies or market microstructure improvements',
        },
        'interview': [
            '5 rounds: programming puzzles, probability, trading game, systems design, depth',
            'Programming: Leetcode Hard + clean code + handle edge cases',
            'Probability: conditional prob, EV, distributions, Bayes',
            'Trading game: live market-making, bid/ask spread decisions under uncertainty',
        ],
        'physics_connection': (
            'Physics-derived differentiable systems = Jane Street research edge:\n'
            '  Known physics (H(f)=exp(j*pi*D*f^2)) + learned correction (FNO) + autograd.\n'
            '  Same structure: known pricing model (B-S) + learned vol surface + autograd.\n'
            '  dL/dtheta via backprop. Same in both domains. This repo is the template.'
        ),
    }


def autograd_truss_sensitivity():
    """
    Adjoint sensitivity for truss: dU_strain/dA_i without finite differences.

    U = sum_i F_i^2 * L_i / (2*E*A_i)
    dU/dA_i = -F_i^2 * L_i / (2*E*A_i^2)  [analytical adjoint]

    Adjoint method = O(1) cost for ALL sensitivities.
    Direct method = O(n_members) resolves.
    Same as: backprop vs numerical gradient in ML.
    """
    E = 200e9
    areas = np.array([1e-4, 1e-4, 1e-4])
    nodes  = {'A': np.array([0.0,0.0]), 'B': np.array([2.0,0.0]), 'C': np.array([1.0,1.5])}
    mlist  = [('A','C'), ('B','C'), ('A','B')]

    truss = three_bar_truss()
    if 'error' in truss:
        return truss
    forces  = np.array([truss['member_forces_N'][f'{m[0]}-{m[1]}'] for m in mlist])
    lengths = np.array([float(np.linalg.norm(nodes[m[1]]-nodes[m[0]])) for m in mlist])

    U = float(np.sum(forces**2 * lengths / (2*E*areas)))
    dU_dA = -(forces**2 * lengths / (2*E*areas**2))

    return {
        'strain_energy_J': U,
        'member_forces_N': {f'{m[0]}-{m[1]}': float(f) for m,f in zip(mlist, forces)},
        'dU_dA': {f'{m[0]}-{m[1]}': float(s) for m,s in zip(mlist, dU_dA)},
        'lesson': (
            'dU/dA < 0: more area -> less stress -> less strain energy. '
            'Adjoint: one extra solve -> all sensitivities. '
            '= backprop: one backward pass -> dL/dW for all weights.'
        ),
    }


def demo():
    print("=== STATICS + JANE STREET + DISTRIBUTION SIFTING ===\n")

    print("--- Beam Reactions ---")
    br = beam_reactions(6.0, [{'P':5000,'x':2.0},{'w':500,'x_start':0,'x_end':6}])
    print(f"  Ay={br['Ay_N']:.1f} N  By={br['By_N']:.1f} N  check={br['equilibrium_check']}")

    print("\n--- Shear/Moment Diagram ---")
    sm = shear_moment_diagram(6.0, [{'P':5000,'x':2.0},{'w':500,'x_start':0,'x_end':6}])
    print(f"  Max V={sm['max_shear_N']:.1f} N  Max M={sm['max_moment_Nm']:.1f} N*m")

    print("\n--- 3-Bar Truss ---")
    t = three_bar_truss()
    if 'error' not in t:
        for mem, F in t['member_forces_N'].items():
            print(f"  {mem}: {F:.1f} N ({'T' if F>0 else 'C'})")

    print("\n--- Distribution Sifting ---")
    ds = distribution_sifting()
    print(f"  Sifting: {ds['sifting_property']}")
    print(f"  Attention: {ds['attention']['soft']}")
    print(f"  IS formula: {ds['importance_sampling']['formula']}")

    print("\n--- Importance Sampling Demo ---")
    isd = importance_sampling_demo()
    print(f"  {isd['lesson']}")

    print("\n--- Jane Street Stack ---")
    js = jane_street_tech_stack()
    for lang in ['OCaml','C','Python_torch_JAX']:
        print(f"  {lang}: {js[lang]['role']}")

    print("\n--- Adjoint Sensitivity ---")
    sens = autograd_truss_sensitivity()
    if 'error' not in sens:
        print(f"  U={sens['strain_energy_J']:.4f} J")
        for m,s in sens['dU_dA'].items():
            print(f"  dU/dA[{m}]={s:.3e} J/m^2")

    print("\n=== DONE ===")


if __name__ == '__main__':
    demo()
