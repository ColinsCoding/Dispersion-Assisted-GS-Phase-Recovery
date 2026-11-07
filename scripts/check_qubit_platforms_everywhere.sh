#!/usr/bin/env bash
# Check that the qubit-drive numbers are the same everywhere: Python, Jupyter, MATLAB, Lean.
# Every step's exit code ($?) is recorded; the script exits non-zero if any step fails.
# Source of truth: dgs/qubit_platforms.json.
#
#   bash scripts/check_qubit_platforms_everywhere.sh            (from the repo root, Git Bash or Linux)
#   SKIP_NOTEBOOK=1 bash scripts/check_qubit_platforms_everywhere.sh
#
# Tool locations can be overridden: PY, MATLAB, LAKE, LEAN_PROJECT.

cd "$(dirname "$0")/.." || exit 2
ROOT="$(pwd)"
PY="${PY:-py -3.12}"
MATLAB="${MATLAB:-$(command -v matlab || echo "/c/Program Files/MATLAB/R2025b/bin/matlab")}"
LAKE="${LAKE:-$(command -v lake || echo "$HOME/.elan/bin/lake.exe")}"
LEAN_PROJECT="${LEAN_PROJECT:-$HOME/lean/NavierStokesAndEuler}"
TMPD="$(mktemp -d)"
export PYTHONPATH="$ROOT"
declare -a NAMES CODES
record() { NAMES+=("$1"); CODES+=("$2"); echo "[$1] exit code $2"; }

echo "== 1. regenerate the Lean file from the JSON"
$PY -m dgs.vector_potential_qubit --write-lean
record "lean file generated" $?

echo "== 2. pytest"
$PY -m pytest tests/test_vector_potential_qubit.py -q
record "pytest" $?

echo "== 3. Python table"
$PY -m dgs.vector_potential_qubit --table > "$TMPD/py.txt"
record "python table" $?

echo "== 4. MATLAB table (same JSON, independent code)"
if [ -x "$MATLAB" ] || command -v "$MATLAB" >/dev/null 2>&1; then
    "$MATLAB" -batch "run('learn/matlab/qubit_platforms_table.m')" > "$TMPD/matlab.txt" 2>&1
    record "matlab" $?
    $PY -c "
import sys
from dgs import vector_potential_qubit as v
a = v.parse_table(open(sys.argv[1]).read()); b = v.parse_table(open(sys.argv[2]).read())
bad = v.compare_tables(a, b, rtol=1e-11)
print(f'{len(a)} Python values, {len(b)} MATLAB values, mismatches: {bad or \"none\"}')
sys.exit(1 if bad or not a else 0)" "$TMPD/py.txt" "$TMPD/matlab.txt"
    record "python == matlab" $?
else
    echo "MATLAB not found"; record "matlab (not found)" 127
fi

echo "== 5. Lean (Mathlib) proof check"
if [ -d "$LEAN_PROJECT" ] && { [ -x "$LAKE" ] || command -v "$LAKE" >/dev/null 2>&1; }; then
    LEAN_FILE="$ROOT/learn/lean/QubitPlatforms.lean"
    (cd "$LEAN_PROJECT" && "$LAKE" env lean "$LEAN_FILE") > "$TMPD/lean.txt" 2>&1
    rc=$?
    cat "$TMPD/lean.txt"
    if [ $rc -eq 0 ] && grep -q "sorry" "$TMPD/lean.txt"; then rc=3; echo "Lean reported sorry"; fi
    record "lean" $rc
else
    echo "Lean project or lake not found"; record "lean (not found)" 127
fi

if [ -z "$SKIP_NOTEBOOK" ]; then
    echo "== 6. Jupyter notebook (build + execute, py312 kernel)"
    $PY scripts/build_vector_potential_qubit_nb.py && \
    $PY -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=py312 \
        --ExecutePreprocessor.timeout=2400 notebooks/vector_potential_qubit.ipynb
    record "notebook" $?
fi

echo
echo "== summary"
fail=0
for i in "${!NAMES[@]}"; do
    printf '  %-22s %s\n' "${NAMES[$i]}" "${CODES[$i]}"
    [ "${CODES[$i]}" -ne 0 ] && fail=1
done
rm -rf "$TMPD"
if [ $fail -eq 0 ]; then echo "ALL STEPS EXIT 0: same numbers everywhere"; else echo "SOME STEP FAILED"; fi
exit $fail
