"""full_adder_polyglot.py -- the full adder already defined in
dgs.computer_engineering.full_adder (S = XOR(XOR(A,B),Cin),
Cout = majority(A,B,Cin)) as THREE genuinely different formalisms,
cross-validated EXHAUSTIVELY over all 8 input combinations (not
spot-checked -- with only 3 boolean inputs, exhaustive is cheap and
strictly stronger than sampling):

  * Python (dgs.computer_engineering.full_adder): imperative, bitwise ops,
    already tested elsewhere in this repo.
  * C (gate-level, same bitwise operators): compiled and run for real via
    gcc, same subprocess pattern as dgs.circuits_polyglot.
  * VHDL (structural/dataflow hardware description): NOT a sequential
    program at all -- an entity/architecture description of actual GATES
    (xor, and, or) wired together, simulated via GHDL. This is a deeper
    language-formalism gap than dgs.error_propagation_polyglot's
    C-vs-C++ operator-overloading contrast or dgs.vector3_polyglot's
    operand/operator distinction: C and Python both describe "what to
    compute, in what sequence"; VHDL describes "what circuit exists,"
    with every signal assignment updating CONCURRENTLY as the simulator's
    event queue settles, not sequentially top-to-bottom like the other two.

Full adder chosen because it's already this repo's canonical gate-level
circuit (dgs.computer_engineering, dgs.logic_timing both build on the
identical S/Cout formula) -- this module doesn't introduce new physics,
just a third, structurally different way of expressing the same one.

Requires ghdl (winget install ghdl.ghdl.ucrt64.mcode) and gcc.
"""

import os
import re
import subprocess

from dgs.computer_engineering import full_adder as full_adder_python

GCC_DEFAULT = r"C:\msys64\mingw64\bin\gcc.exe"
GHDL_DEFAULT = (r"C:\Users\mrjel\AppData\Local\Microsoft\WinGet\Packages"
                 r"\ghdl.ghdl.ucrt64.mcode_Microsoft.Winget.Source_8wekyb3d8bbwe\bin\ghdl.exe")

ALL_INPUTS = [(a, b, cin) for a in (0, 1) for b in (0, 1) for cin in (0, 1)]

# ── C: gate-level, same bitwise operators as dgs.computer_engineering ──────

C_SOURCE = r"""
#include <stdio.h>

static void full_adder(int a, int b, int cin, int *s, int *cout) {
    int ab = a ^ b;
    *s = ab ^ cin;
    *cout = (a & b) | (cin & ab);   /* majority(a,b,cin) via gate identity */
}

int main(void) {
    for (int a = 0; a <= 1; a++)
        for (int b = 0; b <= 1; b++)
            for (int cin = 0; cin <= 1; cin++) {
                int s, cout;
                full_adder(a, b, cin, &s, &cout);
                printf("RESULT %d %d %d %d %d\n", a, b, cin, s, cout);
            }
    return 0;
}
"""

# ── VHDL: structural/dataflow -- describes GATES, not a sequence of steps ──

VHDL_SOURCE = r"""
library ieee;
use ieee.std_logic_1164.all;

entity full_adder is
    port (
        a, b, cin : in  std_logic;
        s, cout   : out std_logic
    );
end entity full_adder;

architecture gate_level of full_adder is
    signal ab : std_logic;
begin
    -- every assignment below is a CONCURRENT signal, not a sequential
    -- statement -- ab, s, and cout all update whenever their inputs change,
    -- in whatever order the simulator's event queue settles them, not in
    -- the top-to-bottom order they're written
    ab   <= a xor b;
    s    <= ab xor cin;
    cout <= (a and b) or (cin and ab);
end architecture gate_level;


library ieee;
use ieee.std_logic_1164.all;
use std.textio.all;

entity full_adder_tb is
end entity full_adder_tb;

architecture sim of full_adder_tb is
    signal a, b, cin, s, cout : std_logic;
    type bit_vec is array (0 to 7) of std_logic_vector(2 downto 0);
    constant inputs : bit_vec := ("000","001","010","011","100","101","110","111");
begin
    uut: entity work.full_adder port map (a => a, b => b, cin => cin, s => s, cout => cout);

    process
        variable l : line;
    begin
        for i in 0 to 7 loop
            a   <= inputs(i)(2);
            b   <= inputs(i)(1);
            cin <= inputs(i)(0);
            wait for 1 ns;
            write(l, string'("RESULT "));
            write(l, std_logic'image(a)(2));
            write(l, string'(" "));
            write(l, std_logic'image(b)(2));
            write(l, string'(" "));
            write(l, std_logic'image(cin)(2));
            write(l, string'(" "));
            write(l, std_logic'image(s)(2));
            write(l, string'(" "));
            write(l, std_logic'image(cout)(2));
            writeline(output, l);
        end loop;
        wait;
    end process;
end architecture sim;
"""

_RESULT_RE = re.compile(r"RESULT (\d) (\d) (\d) (\d) (\d)")


def run_python() -> dict:
    """The existing dgs.computer_engineering.full_adder, called directly
    (no subprocess -- it's already Python)."""
    out = {}
    for a, b, cin in ALL_INPUTS:
        r = full_adder_python(a, b, cin)
        out[(a, b, cin)] = (r["S"], r["Cout"])
    return out


def compile_c(out_dir, gcc_path=GCC_DEFAULT):
    src_path = os.path.join(out_dir, "full_adder.c")
    exe_path = os.path.join(out_dir, "full_adder.exe")
    with open(src_path, "w") as f:
        f.write(C_SOURCE)
    env = os.environ.copy()
    env["PATH"] = os.path.dirname(gcc_path) + os.pathsep + env.get("PATH", "")
    result = subprocess.run([gcc_path, "-O2", "-o", exe_path, src_path],
                             capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"gcc compile failed: {result.stderr}")
    return exe_path


def run_c(exe_path) -> dict:
    result = subprocess.run([exe_path], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"C binary failed: {result.stderr}")
    return _parse_result_lines(result.stdout)


def run_vhdl(out_dir, ghdl_path=GHDL_DEFAULT) -> dict:
    """Analyze, elaborate, and run the VHDL entity+testbench via GHDL's
    mcode JIT backend (no separate netlist/build-directory bookkeeping
    needed, unlike GHDL's other backends)."""
    if not os.path.exists(ghdl_path):
        raise RuntimeError(f"ghdl not found at {ghdl_path} -- "
                           f"winget install ghdl.ghdl.ucrt64.mcode")
    src_path = os.path.join(out_dir, "full_adder_tb.vhdl")
    with open(src_path, "w") as f:
        f.write(VHDL_SOURCE)

    analyze = subprocess.run([ghdl_path, "-a", src_path],
                             capture_output=True, text=True, cwd=out_dir)
    if analyze.returncode != 0:
        raise RuntimeError(f"ghdl -a (analyze) failed: {analyze.stderr}")

    elaborate = subprocess.run([ghdl_path, "-e", "full_adder_tb"],
                               capture_output=True, text=True, cwd=out_dir)
    if elaborate.returncode != 0:
        raise RuntimeError(f"ghdl -e (elaborate) failed: {elaborate.stderr}")

    run = subprocess.run([ghdl_path, "-r", "full_adder_tb"],
                         capture_output=True, text=True, cwd=out_dir)
    if run.returncode != 0:
        raise RuntimeError(f"ghdl -r (run) failed: {run.stderr}")
    return _parse_result_lines(run.stdout)


def _parse_result_lines(text: str) -> dict:
    out = {}
    for match in _RESULT_RE.finditer(text):
        a, b, cin, s, cout = (int(g) for g in match.groups())
        out[(a, b, cin)] = (s, cout)
    return out


def cross_validate_languages(out_dir, gcc_path=GCC_DEFAULT, ghdl_path=GHDL_DEFAULT,
                              run_c_lang=True, run_vhdl_lang=True) -> dict:
    """Run the full adder EXHAUSTIVELY (all 8 input combinations) in
    Python, C, and VHDL, and report every disagreement -- not a spot
    check, a complete truth-table comparison."""
    results = {"python": run_python()}

    if run_c_lang:
        exe_c = compile_c(out_dir, gcc_path=gcc_path)
        results["c"] = run_c(exe_c)

    if run_vhdl_lang:
        results["vhdl"] = run_vhdl(out_dir, ghdl_path=ghdl_path)

    reference = results["python"]
    mismatches = []
    for lang, table in results.items():
        if lang == "python":
            continue
        if set(table.keys()) != set(ALL_INPUTS):
            mismatches.append(f"{lang}: expected 8 input combinations, got {len(table)}")
            continue
        for inputs in ALL_INPUTS:
            if table[inputs] != reference[inputs]:
                mismatches.append(f"{lang}{inputs}: got {table[inputs]}, expected {reference[inputs]}")

    return {"results": results, "mismatches": mismatches, "all_agree": len(mismatches) == 0}


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        report = cross_validate_languages(tmp)

    print(f"{'A B Cin':10s} {'python':>12s} {'c':>12s} {'vhdl':>12s}")
    for a, b, cin in ALL_INPUTS:
        row = [f"({s},{cout})" for s, cout in
               (report["results"][lang][(a, b, cin)] for lang in ("python", "c", "vhdl"))]
        print(f"{a} {b} {cin}     {row[0]:>12s} {row[1]:>12s} {row[2]:>12s}")

    print(f"\nall 3 languages agree on all 8 input combinations: {report['all_agree']}")
    if report["mismatches"]:
        print("MISMATCHES:")
        for m in report["mismatches"]:
            print(f"  {m}")

    print("\nSame gate identity (S=XOR(XOR(A,B),Cin), Cout=majority(A,B,Cin)),")
    print("expressed as: Python statements, compiled C statements, and VHDL")
    print("concurrent signal assignments describing actual circuit structure --")
    print("three different KINDS of formalism, not just three different syntaxes.")
