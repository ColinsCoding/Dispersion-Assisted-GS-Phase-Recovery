"""voltage_divider_vhdl.py -- an actual, simulatable VHDL source for a
resistive voltage divider, cross-validated against a Python reference,
matching this repo's polyglot cross-validation pattern
(dgs/circuits_polyglot.py: same physics, multiple languages, each checked
against the others -- not asserted to agree, actually run and compared).

    V_out = V_in * R2 / (R1 + R2)

WHY REAL-TYPE VHDL FOR AN ANALOG CIRCUIT: a resistive divider is a passive
analog network, not synthesizable digital logic -- there is no gate-level
VHDL for it. What real EE toolflows do (short of full VHDL-AMS) is model the
analog quantity behaviorally with VHDL's REAL type inside a testbench, for
verifying the DIGITAL logic that reads it (e.g. an ADC front-end) against a
known analog input. VOLTAGE_DIVIDER_VHDL_SOURCE below is exactly that kind
of testbench-level behavioral model -- NOT synthesizable to real analog
hardware, and this module says so rather than implying otherwise.

Requires GHDL (an open-source VHDL simulator) on PATH to actually run the
VHDL and compare it against Python -- run_vhdl_voltage_divider raises a
clear error if it isn't found, rather than silently skipping the check.
"""
from __future__ import annotations
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict


VOLTAGE_DIVIDER_VHDL_TEMPLATE = """\
-- voltage_divider_tb.vhd
-- Behavioral (REAL-valued) VHDL model of a resistive voltage divider:
--   V_out = V_in * R2 / (R1 + R2)
-- Uses VHDL's REAL type for testbench-level simulation of an analog
-- quantity -- standard for mixed-signal verification without full
-- VHDL-AMS. NOT synthesizable to real analog hardware.
--
-- V_IN/R1/R2 are baked into the generic defaults below rather than
-- overridden via `ghdl -g` at run time: this GHDL build's mcode backend
-- accepts -g overrides for integer generics but not `real` ones (confirmed
-- directly -- the identical override syntax works for an integer generic
-- and fails for this real one), so run_vhdl_voltage_divider regenerates
-- this source per test case instead of relying on -g.

entity voltage_divider_tb is
    generic (
        V_IN : real := {v_in};
        R1   : real := {r1};
        R2   : real := {r2}
    );
end entity voltage_divider_tb;

architecture behavioral of voltage_divider_tb is
begin
    process
        variable v_out : real;
    begin
        v_out := V_IN * R2 / (R1 + R2);
        report "V_OUT=" & real'image(v_out);
        wait;
    end process;
end architecture behavioral;
"""

# A representative, browsable copy with the default (5.0, 1000.0, 2000.0) values.
VOLTAGE_DIVIDER_VHDL_SOURCE = VOLTAGE_DIVIDER_VHDL_TEMPLATE.format(v_in=5.0, r1=1000.0, r2=2000.0)


def voltage_divider_output(v_in: float, r1: float, r2: float) -> float:
    """Python reference: V_out = V_in * R2 / (R1 + R2), the same formula the
    VHDL testbench above implements independently."""
    if r1 < 0 or r2 < 0:
        raise ValueError("r1 and r2 must be non-negative")
    if r1 + r2 == 0:
        raise ValueError("r1 + r2 must be nonzero")
    return v_in * r2 / (r1 + r2)


def run_vhdl_voltage_divider(v_in: float, r1: float, r2: float,
                              work_dir: str = None, ghdl_path: str = "ghdl") -> float:
    """Generate VOLTAGE_DIVIDER_VHDL_TEMPLATE with (v_in, r1, r2) baked into
    the generic defaults, analyze/elaborate/run it with GHDL (an
    open-source VHDL simulator), and parse the reported V_out value out of
    its stdout -- an ACTUAL VHDL simulation, not a string match against the
    source text. Raises FileNotFoundError with a clear message if ghdl
    isn't on PATH, rather than silently skipping the check."""
    work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="vdiv_vhdl_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    vhd_path = work_dir / "voltage_divider_tb.vhd"
    vhd_path.write_text(VOLTAGE_DIVIDER_VHDL_TEMPLATE.format(v_in=v_in, r1=r1, r2=r2), encoding="utf-8")

    try:
        subprocess.run([ghdl_path, "-a", "--std=08", str(vhd_path)],
                        cwd=work_dir, capture_output=True, text=True, check=True)
        run = subprocess.run(
            [ghdl_path, "--elab-run", "--std=08", "voltage_divider_tb"],
            cwd=work_dir, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{ghdl_path!r} not found on PATH -- install GHDL to run this VHDL simulation"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"GHDL failed:\n{exc.stdout}\n{exc.stderr}") from exc

    match = re.search(r"V_OUT=([-+0-9.eE]+)", run.stdout)
    if not match:
        raise RuntimeError(f"could not find V_OUT in GHDL output:\n{run.stdout}")
    return float(match.group(1))


def cross_validate_voltage_divider(v_in: float = 5.0, r1: float = 1000.0, r2: float = 2000.0,
                                    ghdl_path: str = "ghdl", tol: float = 1e-6) -> Dict:
    """Run both implementations on the SAME (v_in, r1, r2) and compare --
    the actual cross-validation, not an assumption they agree."""
    python_result = voltage_divider_output(v_in, r1, r2)
    vhdl_result = run_vhdl_voltage_divider(v_in, r1, r2, ghdl_path=ghdl_path)
    diff = abs(python_result - vhdl_result)
    return {
        "python_result": python_result,
        "vhdl_result": vhdl_result,
        "diff": diff,
        "agree": diff < tol,
    }


if __name__ == "__main__":
    print(VOLTAGE_DIVIDER_VHDL_SOURCE)
    print("=== cross-validation: Python vs. actual GHDL simulation ===")
    for v_in, r1, r2 in [(5.0, 1000.0, 2000.0), (3.3, 470.0, 1000.0), (12.0, 100.0, 100.0)]:
        result = cross_validate_voltage_divider(v_in, r1, r2)
        print(f"  V_in={v_in}, R1={r1}, R2={r2}:  "
              f"Python={result['python_result']:.6f}  VHDL={result['vhdl_result']:.6f}  "
              f"agree={result['agree']}")
