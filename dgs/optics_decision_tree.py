"""A true/false decision tree for an optical-engineering question, turned back
into algebra by recursion.

QUESTION: is a step-index fiber SINGLE-MODE at wavelength lambda?
    V = 2 pi a NA / lambda < 2.405   (first zero of the Bessel function J_0)
equivalently  a NA / lambda < 2.405 / (2 pi) = 0.3828.

A decision tree (dgs.decision_tree, recursive, from scratch) learns this from
labelled examples. Then, by recursion over the fitted tree:
  * tree_to_piecewise  -> a SymPy Piecewise (the model IS an algebraic formula),
  * tree_to_logic      -> a Boolean formula: OR over root->leaf paths ending in
                          True of the AND of the path's inequalities,
  * leaf_paths         -> every leaf as a bit string (left = 0, right = 1);
    a depth-n tree has at most 2^n leaves -- the same 2^n bit strings that
    label the computational basis of n qubits.
Feature engineering: given the physics variable u = a NA / lambda, ONE split
recovers the exact threshold; given raw (a, NA, lambda), axis-aligned splits
can only build a staircase approximation of the curved boundary.
"""
import numpy as np
import sympy as sp

from dgs.decision_tree import DecisionTree

V_CUTOFF = 2.404825557695773          # first zero of J_0
U_CUTOFF = V_CUTOFF / (2 * np.pi)     # a*NA/lambda threshold


def v_number(a_um, NA, lam_um):
    if np.any(np.asarray(lam_um) <= 0) or np.any(np.asarray(a_um) <= 0) or np.any(np.asarray(NA) <= 0):
        raise ValueError("a, NA, lambda must be positive")
    return 2 * np.pi * np.asarray(a_um) * np.asarray(NA) / np.asarray(lam_um)


def single_mode(a_um, NA, lam_um):
    return v_number(a_um, NA, lam_um) < V_CUTOFF


def fiber_dataset(n=2000, seed=0):
    """Random fibers: core radius 1-10 um (log-uniform), NA 0.05-0.30, lambda 0.4-1.7 um."""
    if n < 2:
        raise ValueError("n must be >= 2")
    rng = np.random.default_rng(seed)
    a = np.exp(rng.uniform(np.log(1.0), np.log(10.0), n))
    NA = rng.uniform(0.05, 0.30, n)
    lam = rng.uniform(0.4, 1.7, n)
    X = np.column_stack([a, NA, lam])
    return X, single_mode(a, NA, lam).astype(int)


def engineered_feature(X):
    """u = a NA / lambda (the physics combination)."""
    return (X[:, 0] * X[:, 1] / X[:, 2])[:, None]


def fit_tree(X, y, max_depth=None, seed=0):
    return DecisionTree(max_depth=max_depth, criterion="gini", random_state=seed).fit(X, y)


# ── recursion over the fitted tree ───────────────────────────────────

def count_nodes_recursive(node):
    if node.is_leaf:
        return 1
    return 1 + count_nodes_recursive(node.left) + count_nodes_recursive(node.right)


def count_nodes_iterative(root):
    """Same count with an explicit stack (a loop instead of recursion)."""
    stack, n = [root], 0
    while stack:
        node = stack.pop()
        n += 1
        if not node.is_leaf:
            stack.extend([node.left, node.right])
    return n


def depth_recursive(node):
    if node.is_leaf:
        return 0
    return 1 + max(depth_recursive(node.left), depth_recursive(node.right))


def tree_to_piecewise(node, syms):
    """Recursive translation into nested SymPy Piecewise (0/1 output)."""
    if node.is_leaf:
        return sp.Integer(int(node.value))
    cond = syms[node.feat] <= sp.Float(node.thresh, 6)
    return sp.Piecewise((tree_to_piecewise(node.left, syms), cond),
                        (tree_to_piecewise(node.right, syms), True))


def leaf_paths(node, prefix="", conds=()):
    """[(bitstring, (conditions...), value)] for every leaf; left = 0, right = 1."""
    if node.is_leaf:
        return [(prefix, conds, int(node.value))]
    return (leaf_paths(node.left, prefix + "0", conds + ((node.feat, "<=", node.thresh),)) +
            leaf_paths(node.right, prefix + "1", conds + ((node.feat, ">", node.thresh),)))


def tree_to_logic(node, syms):
    """Boolean formula: OR over True-leaves of AND(path inequalities)."""
    terms = []
    for _, conds, val in leaf_paths(node):
        if val:
            terms.append(sp.And(*[(syms[f] <= sp.Float(t, 6)) if op == "<=" else (syms[f] > sp.Float(t, 6))
                                  for f, op, t in conds]) if conds else sp.true)
    return sp.Or(*terms) if terms else sp.false


def evaluate_piecewise(expr, syms, X):
    f = sp.lambdify(syms, expr, "numpy")
    return np.asarray(f(*X.T), dtype=int).ravel() if X.shape[0] > 1 else np.atleast_1d(int(f(*X[0])))


# ── 10 binary questions = 10 bits = 10 qubits' basis ─────────────────

def full_tree_bitstrings(depth):
    """All root->leaf paths of a complete binary tree of the given depth, by recursion."""
    if depth < 0:
        raise ValueError("depth must be >= 0")
    if depth == 0:
        return [""]
    sub = full_tree_bitstrings(depth - 1)
    return ["0" + s for s in sub] + ["1" + s for s in sub]


def hadamard_uniform_state(n_qubits):
    """H^{(x)n}|0...0>: equal amplitude 2^{-n/2} on every n-bit string."""
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    psi = np.array([1.0])
    for _ in range(n_qubits):
        psi = np.kron(psi, H @ np.array([1.0, 0.0]))
    return psi
