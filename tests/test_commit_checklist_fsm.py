"""Test scripts/commit_checklist_fsm.py: the FSM's structural validity
(every transition target is a real state), the secret/review-ack regex
patterns in isolation, and the full pre-commit/commit-msg pipelines
against REAL staged files in THIS repo (temporarily staged inside a
try/finally so the repo's actual staging area is always restored, even if
an assertion fails partway through)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import os
import subprocess
import tempfile

import commit_checklist_fsm as fsm

REPO_ROOT = fsm.REPO_ROOT


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)


# 1. FSM structural validity: every non-terminal state handler must return
#    a value that's either a known key in STATE_TABLE or a terminal state
#    -- checked by static inspection of the source (can't exhaustively
#    execute every branch), a real structural property of "this IS a
#    proper FSM" rather than just a naming convention
import ast
import inspect

all_valid_targets = set(fsm.STATE_TABLE.keys()) | fsm.TERMINAL_STATES
for state_name, handler in fsm.STATE_TABLE.items():
    src = inspect.getsource(handler)
    tree = ast.parse(src)
    returned_strings = {
        node.value.value for node in ast.walk(tree)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }
    unknown = returned_strings - all_valid_targets
    assert not unknown, f"state {state_name} returns unknown target(s): {unknown}"

assert "COLLECT_STAGED_FILES" in fsm.STATE_TABLE   # the pre-commit entry point exists
assert "CHECK_REVIEW_ACK" in fsm.STATE_TABLE       # the commit-msg entry point exists
assert fsm.TERMINAL_STATES == {"APPROVED", "APPROVED_WITH_WARNING", "REJECTED"}

print("commit_checklist_fsm: structural validity checks passed")

# 2. Secret-pattern regexes: must catch the documented example forms, and
#    must NOT flag an obviously benign line (no false positive on
#    ordinary code)
secret_examples = [
    'api_key = "sk-abcdefghijklmnop1234567890XYZ"',
    "AKIAABCDEFGHIJKLMNOP",
    'password = "hunter22"',
    "-----BEGIN RSA PRIVATE KEY-----",
]
for line in secret_examples:
    matched = any(p.search(line) for p, _ in fsm.SECRET_PATTERNS)
    assert matched, f"expected a secret pattern to match: {line!r}"

benign_examples = [
    "x = np.exp(1j * phi)",
    "def api_key_lookup(name): return REGISTRY[name]",   # mentions api_key but no literal secret
    "password_hint = 'see the vault'",
]
for line in benign_examples:
    matched = any(p.search(line) for p, _ in fsm.SECRET_PATTERNS)
    assert not matched, f"unexpected false-positive secret match: {line!r}"

# 3. REVIEW_ACK_TRAILER: must match the three documented trailer forms,
#    case-insensitively, and not match ordinary prose
for trailer in ("Reviewed-by: alice", "Self-reviewed: yes", "Checklist: done", "reviewed-by: bob"):
    assert fsm.REVIEW_ACK_TRAILER.search(trailer), f"expected trailer match: {trailer!r}"
assert not fsm.REVIEW_ACK_TRAILER.search("this commit was reviewed by the team")

print("commit_checklist_fsm: pattern checks passed")

# 4. Full pipeline against REAL staged files in this repo -- always
#    restored via try/finally regardless of outcome
try:
    # (a) happy path: a real, valid, already-tested module + its test
    git("add", "dgs/wind_tunnel_aerodynamics.py", "tests/test_wind_tunnel_aerodynamics.py")
    result = fsm.run_fsm("COLLECT_STAGED_FILES", mode="pre-commit")
    assert result == 0, "expected the wind-tunnel module + its passing test to be APPROVED"
finally:
    git("reset", "dgs/wind_tunnel_aerodynamics.py", "tests/test_wind_tunnel_aerodynamics.py")

# (b) syntax failure -> REJECTED (exit 1)
bad_path = os.path.join(REPO_ROOT, "_fsm_unittest_syntax_bad.py")
try:
    with open(bad_path, "w") as f:
        f.write("def broken(:\n")
    git("add", "_fsm_unittest_syntax_bad.py")
    result = fsm.run_fsm("COLLECT_STAGED_FILES", mode="pre-commit")
    assert result == 1, "expected a syntax error to be REJECTED"
finally:
    git("reset", "_fsm_unittest_syntax_bad.py")
    if os.path.exists(bad_path):
        os.remove(bad_path)

# (c) secret detected -> REJECTED (exit 1)
secret_path = os.path.join(REPO_ROOT, "_fsm_unittest_secret.py")
try:
    with open(secret_path, "w") as f:
        f.write('api_key = "sk-abcdefghijklmnop1234567890XYZ"\n')
    git("add", "_fsm_unittest_secret.py")
    result = fsm.run_fsm("COLLECT_STAGED_FILES", mode="pre-commit")
    assert result == 1, "expected a detected secret to be REJECTED"
finally:
    git("reset", "_fsm_unittest_secret.py")
    if os.path.exists(secret_path):
        os.remove(secret_path)

# (d) corrupted notebook -> REJECTED (exit 1)
nb_path = os.path.join(REPO_ROOT, "_fsm_unittest_bad.ipynb")
try:
    with open(nb_path, "w") as f:
        f.write("{not valid json")
    git("add", "_fsm_unittest_bad.ipynb")
    result = fsm.run_fsm("COLLECT_STAGED_FILES", mode="pre-commit")
    assert result == 1, "expected a corrupted notebook to be REJECTED"
finally:
    git("reset", "_fsm_unittest_bad.ipynb")
    if os.path.exists(nb_path):
        os.remove(nb_path)

print("commit_checklist_fsm: pre-commit pipeline checks passed")

# 5. commit-msg pipeline: large diff + no trailer -> warning (exit 0);
#    strict mode -> reject (exit 1); with trailer -> clean pass (exit 0)
try:
    git("add", "dgs/wind_tunnel_aerodynamics.py", "tests/test_wind_tunnel_aerodynamics.py")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("quick fix\n")
        short_msg_path = f.name
    with open(short_msg_path, encoding="utf-8") as f:
        msg = f.read()
    result = fsm.run_fsm("CHECK_REVIEW_ACK", mode="commit-msg", commit_message=msg)
    assert result == 0, "expected a large-diff commit with no trailer to WARN, not reject, by default"
    os.remove(short_msg_path)

    os.environ["COMMIT_CHECKLIST_STRICT"] = "1"
    try:
        result = fsm.run_fsm("CHECK_REVIEW_ACK", mode="commit-msg", commit_message="quick fix\n")
        assert result == 1, "expected strict mode to REJECT a large-diff commit with no trailer"
    finally:
        del os.environ["COMMIT_CHECKLIST_STRICT"]

    result = fsm.run_fsm("CHECK_REVIEW_ACK", mode="commit-msg",
                          commit_message="add module\n\nReviewed-by: self\n")
    assert result == 0, "expected a large-diff commit WITH a trailer to pass cleanly"
finally:
    git("reset", "dgs/wind_tunnel_aerodynamics.py", "tests/test_wind_tunnel_aerodynamics.py")

print("commit_checklist_fsm: commit-msg pipeline checks passed")

# 6. confirm the repo's staging area is genuinely clean after all of the above
final_status = git("diff", "--cached", "--name-only").stdout.strip()
assert final_status == "", f"staging area not clean after tests: {final_status!r}"

print("all commit_checklist_fsm tests passed")
