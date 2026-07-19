"""Export the radial coefficient ROM as FPGA artifacts (.coe and VHDL).

Purpose:
    Turn the in-software `RadialCoeffRom` (the exact table the node runs) into the files
    an FPGA flow consumes: Xilinx `.coe` block-memory initialization vectors and a VHDL
    block-RAM ROM entity. Software and hardware then share one source of truth.

References:
    - Xilinx Block Memory Generator `.coe` format (memory_initialization_radix/vector).
Assumptions:
    - Signed fixed-point tables (default 8-bit two's-complement).
Limitations:
    - Emits data + a reference RTL entity; place-and-route / IP wiring is the user's flow.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ros2_phycv.pst_core import PstParams
from ros2_phycv.pst_rom import RadialCoeffRom, build_radial_rom

__all__ = ["to_coe", "to_packed_coe", "to_vhdl", "write_rom_artifacts"]


def _byte(value: int) -> int:
    """Two's-complement 8-bit byte for a signed value."""
    return int(value) & 0xFF


def to_coe(values: Sequence[int], radix: int = 16, comment: str = "") -> str:
    """Xilinx `.coe` for one memory: signed values as two's-complement, one per line."""
    if radix == 16:
        cells = [f"{_byte(v):02X}" for v in values]
    elif radix == 2:
        cells = [format(_byte(v), "08b") for v in values]
    else:
        raise ValueError("radix must be 16 or 2")
    header = f"; {comment}\n" if comment else ""
    body = ",\n".join(cells)
    return f"{header}memory_initialization_radix={radix};\nmemory_initialization_vector=\n{body};\n"


def to_packed_coe(rom: RadialCoeffRom, comment: str = "PST coeff ROM {re[15:8], im[7:0]}") -> str:
    """Single `.coe` with real/imag packed into one 16-bit word per address."""
    words = [f"{(_byte(r) << 8) | _byte(i):04X}" for r, i in zip(rom.re, rom.im)]
    body = ",\n".join(words)
    return f"; {comment}\nmemory_initialization_radix=16;\nmemory_initialization_vector=\n{body};\n"


def to_vhdl(rom: RadialCoeffRom, name: str = "pst_coeff_rom") -> str:
    """VHDL block-RAM ROM with signed real/imag outputs (synchronous read => BRAM)."""
    def column(table: Sequence[int]) -> str:
        cells = [f'x"{_byte(v):02X}"' for v in table]
        return ",\n".join("    " + ", ".join(cells[k : k + 16]) for k in range(0, len(cells), 16))

    return f"""library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Auto-generated from ros2_phycv.pst_rom (radial PST coefficient ROM, {rom.n_bits}-bit signed).
entity {name} is
  port (
    clk        : in  std_logic;
    addr       : in  std_logic_vector(7 downto 0);   -- rho-bin address
    re_o, im_o : out std_logic_vector(7 downto 0)    -- signed fixed-point coefficient
  );
end entity {name};

architecture rtl of {name} is
  type rom_type is array (0 to {rom.n_bins - 1}) of std_logic_vector(7 downto 0);
  constant ROM_RE : rom_type := (
{column(rom.re)}
  );
  constant ROM_IM : rom_type := (
{column(rom.im)}
  );
begin
  process(clk)
  begin
    if rising_edge(clk) then                          -- registered read => block RAM
      re_o <= ROM_RE(to_integer(unsigned(addr)));
      im_o <= ROM_IM(to_integer(unsigned(addr)));
    end if;
  end process;
end architecture rtl;
"""


def write_rom_artifacts(rom: RadialCoeffRom, out_dir: str | Path, name: str = "pst_coeff_rom") -> dict[str, Path]:
    """Write the .coe (real, imag, packed) and VHDL files; return the created paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "re_coe": out / f"{name}_re.coe",
        "im_coe": out / f"{name}_im.coe",
        "packed_coe": out / f"{name}_packed.coe",
        "vhdl": out / f"{name}.vhd",
    }
    paths["re_coe"].write_text(to_coe(rom.re, comment=f"{name} real part (signed {rom.n_bits}-bit)"), encoding="utf-8")
    paths["im_coe"].write_text(to_coe(rom.im, comment=f"{name} imag part (signed {rom.n_bits}-bit)"), encoding="utf-8")
    paths["packed_coe"].write_text(to_packed_coe(rom), encoding="utf-8")
    paths["vhdl"].write_text(to_vhdl(rom, name), encoding="utf-8")
    return paths


def _cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export the PST radial coefficient ROM as .coe + VHDL.")
    parser.add_argument("--out", default="fpga", help="output directory")
    parser.add_argument("--name", default="pst_coeff_rom")
    parser.add_argument("--strength", type=float, default=4.0)
    parser.add_argument("--warp", type=float, default=15.0)
    parser.add_argument("--sigma-lpf", type=float, default=0.2)
    parser.add_argument("--bins", type=int, default=256)
    parser.add_argument("--bits", type=int, default=8)
    args = parser.parse_args(argv)
    rom = build_radial_rom(
        PstParams(strength=args.strength, warp=args.warp, sigma_lpf=args.sigma_lpf),
        n_bins=args.bins,
        n_bits=args.bits,
    )
    for kind, path in write_rom_artifacts(rom, args.out, args.name).items():
        print(f"{kind:10s} -> {path}")


if __name__ == "__main__":
    _cli()
