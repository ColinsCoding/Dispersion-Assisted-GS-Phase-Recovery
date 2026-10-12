"""Tests for dgs/circuit_terminology.py."""
import pytest
import sympy as sp

from dgs.circuit_terminology import (
    nodes, essential_nodes, essential_branches, branch_label, loop_basis, meshes,
    is_planar, table_4_1, branch_current_solution, node_voltage_solution,
    mesh_current_solution, tellegen_sum,
)

R = sp.Integer
A = [("v1", "a", "g", "V", R(40)), ("R1", "a", "b", "R", R(5)), ("R2", "b", "g", "R", R(10)),
     ("R5", "b", "c", "R", R(4)), ("R3", "c", "g", "R", R(20)), ("R7", "c", "g", "R", R(20)),
     ("R6", "c", "d", "R", R(6)), ("R4", "d", "e", "R", R(2)), ("v2", "e", "g", "V", R(10)),
     ("I1", "g", "b", "I", R(2))]
A_NO_I = [e for e in A if e[0] != "I1"]


def k33():
    K, c = [], 0
    for a in ("x1", "x2", "x3"):
        for b in ("y1", "y2", "y3"):
            c += 1
            K.append((f"R{c}", a, b, "R", R(c)))
    K[0] = ("R1", "x1", "m", "R", R(1))
    K.append(("vs", "m", "y1", "V", R(12)))
    return K


def test_table_counts():
    t = table_4_1(A)
    assert (t["n (nodes)"], t["n_e (essential nodes)"]) == (6, 3)
    assert (t["b (branches = elements)"], t["b_e (essential branches)"]) == (10, 7)
    assert (t["KCL equations n_e - 1"], t["KVL equations b_e - (n_e - 1)"]) == (2, 5)
    assert t["planar"] and t["meshes"] == 5
    assert essential_nodes(A) == ["b", "c", "g"]
    assert set(nodes(A)) - set(essential_nodes(A)) == {"a", "d", "e"}


def test_essential_branches_cover_every_element_once():
    brs = essential_branches(A)
    names = [st[0].name for br in brs for st in br.steps]
    assert sorted(names) == sorted(e[0] for e in A)
    labels = {branch_label(b) for b in brs}
    assert "R6 - R4 - v2" in labels and "R1 - v1" in labels


def test_euler_formula_both_levels():
    t = table_4_1(A)
    # meshes = b - n + 1 = b_e - n_e + 1
    assert t["meshes"] == t["b (branches = elements)"] - t["n (nodes)"] + 1
    assert len(loop_basis(A)) == t["meshes"]


def test_each_branch_in_at_most_two_meshes_with_opposite_signs():
    use = {}
    for m in meshes(A):
        for k, s in m:
            use.setdefault(k, []).append(s)
    for signs in use.values():
        assert len(signs) <= 2
        if len(signs) == 2:
            assert sum(signs) == 0


def test_methods_agree_with_current_source():
    _, i_b, v_b = branch_current_solution(A, "g")
    _, i_n, V = node_voltage_solution(A, "g")
    assert i_b == i_n
    k_I = next(k for k, br in enumerate(essential_branches(A))
               if branch_label(br) == "I1")
    br = essential_branches(A)[k_I]
    # I1 pushes 2 A from g to b; branch current is defined start -> end
    assert i_b[k_I] == (2 if br.start == "g" else -2)
    assert tellegen_sum(A, i_b, v_b) == 0
    assert V["g"] == 0


def test_three_methods_agree_planar():
    _, i_b, v_b = branch_current_solution(A_NO_I, "g")
    _, i_n, _ = node_voltage_solution(A_NO_I, "g")
    _, i_m = mesh_current_solution(A_NO_I)
    assert i_b == i_n == i_m
    assert tellegen_sum(A_NO_I, i_b, v_b) == 0


def test_mesh_method_rejects_current_source():
    with pytest.raises(ValueError):
        mesh_current_solution(A)


def test_nonplanar():
    K = k33()
    assert not is_planar(K)
    assert meshes(K) is None
    with pytest.raises(ValueError):
        mesh_current_solution(K)
    _, i_b, v_b = branch_current_solution(K)
    _, i_n, _ = node_voltage_solution(K)
    assert i_b == i_n
    assert tellegen_sum(K, i_b, v_b) == 0
    t = table_4_1(K)
    assert t["KVL equations b_e - (n_e - 1)"] == len(loop_basis(K)) == 4


def test_symbolic_two_mesh():
    v1, v2, R1, R2, R3 = sp.symbols("v1 v2 R1 R2 R3", positive=True)
    C = [("v1", "p", "0", "V", v1), ("R1", "p", "n", "R", R1), ("R2", "n", "0", "R", R2),
         ("R3", "n", "q", "R", R3), ("v2", "q", "0", "V", v2)]
    _, i_b, _ = branch_current_solution(C, "0")
    _, i_n, V = node_voltage_solution(C, "0")
    _, i_m = mesh_current_solution(C)
    for k in i_b:
        assert sp.simplify(i_b[k] - i_n[k]) == 0
        assert sp.simplify(i_b[k] - i_m[k]) == 0
    expected = (v1 / R1 + v2 / R3) / (1 / R1 + 1 / R2 + 1 / R3)   # Millman
    assert sp.simplify(V["n"] - expected) == 0


def test_single_loop_and_voltage_source_branch():
    loop = [("v", "a", "b", "V", R(9)), ("R1", "b", "c", "R", R(1)), ("R2", "c", "a", "R", R(2))]
    _, i_b, _ = branch_current_solution(loop)
    assert [abs(c) for c in i_b.values()] == [3]
    # a branch that is ONLY a voltage source (node-voltage needs an extra unknown)
    C = [("vs", "x", "0", "V", R(6)), ("Ra", "x", "0", "R", R(3)), ("Rb", "x", "0", "R", R(6))]
    _, i_n, V = node_voltage_solution(C, "0")
    _, i_b, _ = branch_current_solution(C, "0")
    assert i_n == i_b and V["x"] == 6


def test_validation():
    with pytest.raises(ValueError):
        table_4_1([("R1", "a", "b", "X", 1)])
    with pytest.raises(ValueError):
        nodes([("R1", "a", "b", "R", 1), ("R2", "b", "c", "R", 1)])      # dangling
    with pytest.raises(ValueError):
        circuit = [("R1", "a", "a", "R", 1)]
        table_4_1(circuit)
    with pytest.raises(ValueError):
        branch_current_solution(A, ground="a")                          # not essential
