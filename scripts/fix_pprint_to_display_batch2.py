"""Same fix as fix_pprint_to_display.py, applied to the other 17 pre-existing
notebooks (not built this session) that have the same sp.pprint() issue.
All 17 already call sp.init_printing(); this just adds the display import
and swaps sp.pprint( -> display(.
"""
import pathlib
import re
import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    "ap_physics_frq_transistor",
    "curvature_measurement_schrodinger_circuits",
    "electrodynamics_to_dispersion_operator",
    "fiber_ray_hamiltonian",
    "franck_hertz_elastic_collision_sympy_torch",
    "griffiths_integrals_for_chemistry",
    "griffiths_norm_conservation_sympy",
    "griffiths_problem_1_8_linear_algebra",
    "magnetism_as_relativistic_electricity",
    "nuclear_sympy_sweeps",
    "pde_separation_em",
    "phase_is_a_spacetime_invariant",
    "phase_retrieval_materials",
    "photonics_calculus_solutions",
    "quantum_mechanics_1d_chapter6",
    "regex_parallel_em_asic_001",
    "thermal_analytical_mechanics_jalali",
]

PPRINT_RE = re.compile(r"\bsp\.pprint\(")

for name in NOTEBOOKS:
    path = ROOT / "notebooks" / f"{name}.ipynb"
    nb = nbf.read(str(path), as_version=4)
    n_fixed = 0
    has_display_import = False

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if "from IPython.display import display" in cell.source:
            has_display_import = True
        if PPRINT_RE.search(cell.source):
            n_fixed += len(PPRINT_RE.findall(cell.source))
            cell.source = PPRINT_RE.sub("display(", cell.source)

    if not has_display_import:
        for cell in nb.cells:
            if cell.cell_type == "code" and "init_printing" in cell.source:
                cell.source = "from IPython.display import display\n" + cell.source
                break

    nbf.write(nb, str(path))
    print(f"{name}: fixed {n_fixed} sp.pprint -> display, "
          f"display import {'already present' if has_display_import else 'added'}")
