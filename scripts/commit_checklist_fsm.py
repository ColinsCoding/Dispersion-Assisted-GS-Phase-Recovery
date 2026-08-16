"""A real, operational commit-review checklist, implemented as a finite
state machine, run by two tracked git hooks (.githooks/pre-commit and
.githooks/commit-msg).

STATES and TRANSITIONS (this IS the FSM -- see STATE_TABLE below, not a
metaphor):

    INIT -> COLLECT_STAGED_FILES
    COLLECT_STAGED_FILES -> CHECK_SYNTAX            (nothing staged: -> APPROVED, e.g. empty/merge commits)
    CHECK_SYNTAX         -> CHECK_SECRETS  | REJECTED
    CHECK_SECRETS        -> CHECK_TESTS    | REJECTED
    CHECK_TESTS          -> CHECK_NOTEBOOKS| REJECTED
    CHECK_NOTEBOOKS      -> CHECK_REVIEW_ACK | REJECTED
    CHECK_REVIEW_ACK     -> APPROVED | APPROVED_WITH_WARNING | REJECTED (strict mode only)
    APPROVED, APPROVED_WITH_WARNING, REJECTED         -- terminal

Two entry points, matching the two git hook events:
  * `pre-commit`  runs states through CHECK_NOTEBOOKS (nothing about the
    commit MESSAGE exists yet at this point in git's commit flow).
  * `commit-msg <msgfile>` runs CHECK_REVIEW_ACK only, since the message
    doesn't exist until after pre-commit has already passed.

INSTALL (this script does NOT run `git config` itself -- see the repo's
git safety rules; run this one command yourself):

    git config core.hooksPath .githooks

UNINSTALL:

    git config --unset core.hooksPath

BYPASS (standard git behavior, not special to this script):

    git commit --no-verify

STRICT MODE (blocks large commits without a review acknowledgment,
instead of just warning): set COMMIT_CHECKLIST_STRICT=1 in the
environment.
"""

from __future__ import annotations
import fnmatch
import json
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = "py"
PYTHON_VERSION_FLAG = "-3.13"
LARGE_COMMIT_LINE_THRESHOLD = 200
SECRET_PATTERNS = [
    (re.compile(r"-----BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"), "private key block"),
    (re.compile(r"(?i)\b(api[_-]?key|secret[_-]?key|access[_-]?token)\b\s*[:=]\s*['\"][A-Za-z0-9_\-/+]{16,}['\"]"),
     "API key / secret token literal"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key ID"),
    (re.compile(r"(?i)\bpassword\b\s*[:=]\s*['\"][^'\"]{4,}['\"]"), "hardcoded password"),
]
REVIEW_ACK_TRAILER = re.compile(r"(?im)^(Reviewed-by|Self-reviewed|Checklist):")


def run(cmd, cwd=REPO_ROOT):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def staged_files(diff_filter: str = "ACM") -> list[str]:
    result = run(["git", "diff", "--cached", "--name-only", f"--diff-filter={diff_filter}"])
    return [f for f in result.stdout.splitlines() if f.strip()]


# ── State handlers ────────────────────────────────────────────────────────
# Each handler takes a mutable `ctx` dict and returns the NEXT state name.

def state_collect_staged_files(ctx: dict) -> str:
    ctx["staged"] = staged_files()
    if not ctx["staged"]:
        ctx["messages"].append("no staged files -- nothing to check (empty/merge commit)")
        return "APPROVED"
    ctx["messages"].append(f"{len(ctx['staged'])} staged file(s)")
    return "CHECK_SYNTAX"


def state_check_syntax(ctx: dict) -> str:
    py_files = [f for f in ctx["staged"] if f.endswith(".py") and os.path.exists(f)]
    failures = []
    for f in py_files:
        result = run([PYTHON, PYTHON_VERSION_FLAG, "-m", "py_compile", f])
        if result.returncode != 0:
            failures.append((f, result.stderr.strip()))
    if failures:
        ctx["reject_reason"] = "syntax errors:\n" + "\n".join(f"  {f}: {err}" for f, err in failures)
        return "REJECTED"
    ctx["messages"].append(f"syntax OK ({len(py_files)} .py file(s) compiled cleanly)")
    return "CHECK_SECRETS"


def state_check_secrets(ctx: dict) -> str:
    result = run(["git", "diff", "--cached", "-U0", "--", *ctx["staged"]])
    added_lines = [line[1:] for line in result.stdout.splitlines()
                   if line.startswith("+") and not line.startswith("+++")]
    hits = []
    for line in added_lines:
        for pattern, label in SECRET_PATTERNS:
            if pattern.search(line):
                hits.append((label, line.strip()[:100]))
    if hits:
        ctx["reject_reason"] = "possible secret(s) in staged diff:\n" + "\n".join(
            f"  [{label}] {snippet}" for label, snippet in hits)
        return "REJECTED"
    ctx["messages"].append("no obvious secrets detected in the staged diff")
    return "CHECK_TESTS"


def state_check_tests(ctx: dict) -> str:
    to_run = set()
    for f in ctx["staged"]:
        norm = f.replace("\\", "/")
        if norm.startswith("dgs/") and norm.endswith(".py"):
            name = os.path.basename(norm)[:-3]
            candidate = os.path.join("tests", f"test_{name}.py")
            if os.path.exists(os.path.join(REPO_ROOT, candidate)):
                to_run.add(candidate)
        elif norm.startswith("tests/") and norm.endswith(".py") and os.path.exists(norm):
            to_run.add(norm)

    if not to_run:
        ctx["messages"].append("no test files implicated by this commit -- test check skipped")
        return "CHECK_NOTEBOOKS"

    failures = []
    for test_path in sorted(to_run):
        result = run([PYTHON, PYTHON_VERSION_FLAG, test_path])
        if result.returncode != 0:
            failures.append((test_path, (result.stdout + result.stderr).strip()[-500:]))
    if failures:
        ctx["reject_reason"] = "test failure(s):\n" + "\n".join(
            f"  {t}:\n    {out}" for t, out in failures)
        return "REJECTED"
    ctx["messages"].append(f"tests OK ({len(to_run)} test file(s) passed)")
    return "CHECK_NOTEBOOKS"


def state_check_notebooks(ctx: dict) -> str:
    nb_files = [f for f in ctx["staged"] if f.endswith(".ipynb") and os.path.exists(f)]
    broken = []
    for f in nb_files:
        try:
            with open(f, encoding="utf-8") as fh:
                json.load(fh)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            broken.append((f, str(e)))
    if broken:
        ctx["reject_reason"] = "corrupted/invalid notebook JSON:\n" + "\n".join(
            f"  {f}: {err}" for f, err in broken)
        return "REJECTED"
    ctx["messages"].append(f"notebooks OK ({len(nb_files)} .ipynb file(s) are valid JSON)")
    return "CHECK_REVIEW_ACK"


def state_check_review_ack(ctx: dict) -> str:
    """Runs in the commit-msg hook (message exists) OR as a no-op
    placeholder during pre-commit (message doesn't exist yet -- pre-commit
    always transitions straight through to APPROVED and leaves the real
    check to the commit-msg hook)."""
    if ctx.get("mode") == "pre-commit":
        return "APPROVED"

    message = ctx.get("commit_message", "")
    result = run(["git", "diff", "--cached", "--numstat"])
    total_changed_lines = 0
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            for p in parts[:2]:
                if p.isdigit():
                    total_changed_lines += int(p)

    is_large = total_changed_lines > LARGE_COMMIT_LINE_THRESHOLD
    has_ack = bool(REVIEW_ACK_TRAILER.search(message))
    strict = os.environ.get("COMMIT_CHECKLIST_STRICT") == "1"

    if not is_large or has_ack:
        ctx["messages"].append(f"review-acknowledgment check OK (changed lines: {total_changed_lines})")
        return "APPROVED"

    warning = (f"large commit ({total_changed_lines} changed lines) with no "
               f"'Reviewed-by:' / 'Self-reviewed:' / 'Checklist:' trailer in the message -- "
               f"consider running /code-review before committing")
    if strict:
        ctx["reject_reason"] = warning + " (COMMIT_CHECKLIST_STRICT=1: blocking)"
        return "REJECTED"
    ctx["warning"] = warning
    return "APPROVED_WITH_WARNING"


STATE_TABLE = {
    "COLLECT_STAGED_FILES": state_collect_staged_files,
    "CHECK_SYNTAX": state_check_syntax,
    "CHECK_SECRETS": state_check_secrets,
    "CHECK_TESTS": state_check_tests,
    "CHECK_NOTEBOOKS": state_check_notebooks,
    "CHECK_REVIEW_ACK": state_check_review_ack,
}
TERMINAL_STATES = {"APPROVED", "APPROVED_WITH_WARNING", "REJECTED"}


def run_fsm(start_state: str, mode: str, commit_message: str = "") -> int:
    ctx = {"messages": [], "mode": mode, "commit_message": commit_message}
    state = start_state
    while state not in TERMINAL_STATES:
        handler = STATE_TABLE[state]
        state = handler(ctx)

    for m in ctx["messages"]:
        print(f"  [ok] {m}")

    if state == "REJECTED":
        print(f"\n[REJECTED] commit checklist failed:\n{ctx['reject_reason']}")
        print("\n(bypass with `git commit --no-verify` if this is a false positive)")
        return 1
    if state == "APPROVED_WITH_WARNING":
        print(f"\n[WARNING] {ctx['warning']}")
        print("[APPROVED] commit checklist passed with a warning")
        return 0
    print("[APPROVED] commit checklist passed")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("pre-commit", "commit-msg"):
        print("usage: commit_checklist_fsm.py {pre-commit|commit-msg} [commit_msg_file]")
        sys.exit(2)

    hook = sys.argv[1]
    if hook == "pre-commit":
        sys.exit(run_fsm("COLLECT_STAGED_FILES", mode="pre-commit"))
    else:
        if len(sys.argv) < 3:
            print("commit-msg hook requires the commit-message file path")
            sys.exit(2)
        with open(sys.argv[2], encoding="utf-8") as f:
            msg = f.read()
        sys.exit(run_fsm("CHECK_REVIEW_ACK", mode="commit-msg", commit_message=msg))
