"""Cross-validate statics_truss.c against dgs.truss.solve_truss (Python/
NumPy) and dgs.truss_symbolic.solve_truss_symbolic (Python/SymPy) --
three independent implementations of the same method-of-joints statics
(two languages, three solving methods: Gaussian elimination in C,
np.linalg.solve, sp.linsolve), all required to agree, not just each
individually plausible.

Prerequisite: compile statics_truss first (nvcc/gcc needs cl.exe or a
MinGW gcc on PATH -- see this repo's other C build notes):
    cd unix_tools/statics_truss
    gcc -O2 -o statics_truss.exe statics_truss.c -lm   (Windows)
    gcc -O2 -o statics_truss     statics_truss.c -lm   (Linux/macOS)

Run:  py -3.13 unix_tools/statics_truss/verify_cross_language.py
"""
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from dgs.truss import solve_truss
from dgs.truss_symbolic import solve_truss_symbolic


def find_exe():
    for name in ("statics_truss.exe", "statics_truss"):
        p = HERE / name
        if p.exists():
            return p
    print("ERROR: statics_truss binary not found -- compile it first (see this file's docstring)")
    sys.exit(1)


def run_c_program(input_path):
    exe = find_exe()
    result = subprocess.run([str(exe), str(input_path)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"statics_truss exited {result.returncode}: {result.stderr}")
    return result.stdout


def parse_c_output(stdout):
    member_forces = {}
    for m in re.finditer(r"^\s*(\d+)-(\d+):\s*([+-]?[\d.]+)", stdout, re.MULTILINE):
        a, b, force = int(m.group(1)), int(m.group(2)), float(m.group(3))
        member_forces[(a, b)] = force
    reactions = {}
    for m in re.finditer(r"^\s*node (\d+): Rx=([+-]?[\d.]+)\s+Ry=([+-]?[\d.]+)", stdout, re.MULTILINE):
        node, rx, ry = int(m.group(1)), float(m.group(2)), float(m.group(3))
        reactions[node] = (rx, ry)
    return member_forces, reactions


def main():
    print("=== CROSS-VALIDATING statics_truss.c AGAINST TWO INDEPENDENT PYTHON SOLVERS ===\n")

    nodes = {"A": (0.0, 0.0), "B": (2.0, 0.0), "C": (1.0, 1.5)}
    members = [("A", "C"), ("B", "C"), ("A", "B")]
    supports = {"A": (True, True), "B": (False, True)}
    loads = {"C": (0.0, -10000.0)}
    name_to_idx = {"A": 0, "B": 1, "C": 2}

    print("--- Python / NumPy (dgs.truss.solve_truss) ---")
    mf_numpy, rx_numpy = solve_truss(nodes, members, supports, loads)
    for k, v in mf_numpy.items():
        print(f"  {k[0]}-{k[1]}: {v:.4f} N")

    print("\n--- Python / SymPy (dgs.truss_symbolic.solve_truss_symbolic) ---")
    mf_sympy, rx_sympy = solve_truss_symbolic(nodes, members, supports, loads)
    for k, v in mf_sympy.items():
        print(f"  {k[0]}-{k[1]}: {float(v):.4f} N")

    print("\n--- C (statics_truss.c, Gaussian elimination) ---")
    stdout = run_c_program(HERE / "three_bar_truss.txt")
    mf_c_idx, rx_c_idx = parse_c_output(stdout)   # keyed by (node_index_a, node_index_b), matching the .txt file's order
    for (a, b), v in mf_c_idx.items():
        print(f"  {a}-{b}: {v:.4f} N")

    print("\n--- Cross-check: all three must agree to within 1e-3 N ---")
    all_agree = True
    for (name_a, name_b) in mf_numpy:
        idx_pair = (name_to_idx[name_a], name_to_idx[name_b])
        v_numpy, v_sympy, v_c = mf_numpy[(name_a, name_b)], float(mf_sympy[(name_a, name_b)]), mf_c_idx[idx_pair]
        max_diff = max(abs(v_numpy - v_sympy), abs(v_numpy - v_c), abs(v_sympy - v_c))
        status = "OK" if max_diff < 1e-3 else "MISMATCH"   # 1e-3 N: C's printed precision is 4 decimal places
        print(f"  {name_a}-{name_b}: NumPy={v_numpy:.4f}  SymPy={v_sympy:.4f}  C={v_c:.4f}  max diff={max_diff:.2e}  [{status}]")
        if status == "MISMATCH":
            all_agree = False

    assert all_agree, "the three independent implementations disagree -- see mismatches above"
    print("\n[OK] three independent implementations (NumPy linear solve, SymPy symbolic solve,")
    print("     and a from-scratch C Gaussian elimination) all agree on the same truss")


if __name__ == "__main__":
    main()
