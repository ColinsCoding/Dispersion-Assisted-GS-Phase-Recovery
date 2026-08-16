"""dgs.procedural_architecture_blender needs bpy, which is only importable
inside Blender's own Python -- not this test process. This file runs
Blender headless as a subprocess against
tests/_blender_test_procedural_architecture.py (which does the actual
checking using the real check()/checks pattern used everywhere else in
this repo) and verifies it printed ALL_CHECKS_PASSED, exactly the same
grading contract as a notebook's final assertion cell, just crossing a
process boundary because bpy can't be imported directly here."""
import subprocess
import sys
import pathlib

BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe"
TEST_SCRIPT = pathlib.Path(__file__).resolve().parent / "_blender_test_procedural_architecture.py"

if not pathlib.Path(BLENDER_EXE).exists():
    print(f"SKIPPED: Blender not found at {BLENDER_EXE}")
    sys.exit(0)

result = subprocess.run(
    [BLENDER_EXE, "--background", "--python", str(TEST_SCRIPT)],
    capture_output=True, text=True, timeout=120,
)

print(result.stdout)
if result.returncode != 0:
    print(result.stderr)

assert "ALL_CHECKS_PASSED" in result.stdout, (
    f"Blender test script did not report success (exit code {result.returncode}). "
    f"See stdout above for which checks failed.")

print("all dgs.procedural_architecture_blender tests passed")
