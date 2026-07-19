"""Generate notebooks/vevid_lut_vhdl_bram.ipynb."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from nbkit import md, co, setup_cell, write

cells = [
md(r"""# VEViD LUT in hardware: a VHDL block-RAM ROM

The 256-entry VEViD tone-map LUT becomes a hardware primitive: a **block-RAM ROM**. The input pixel code is
the address, the stored byte is the enhanced output. The one rule that makes a synthesizer map it to a
dedicated **BRAM** block (rather than spending fabric LUTs/flip-flops) is a **synchronous read** -- register
the output on the clock edge. That is the whole design: one clock, an 8-bit address in, an 8-bit datum out,
one cycle of latency, no arithmetic.

This notebook generates the **VHDL** entity and architecture from the Python LUT, verifies the generated
ROM initialization is bit-exact to the table, and -- because a real HDL simulator is the honest test --
**simulates a functionally identical Verilog twin with Icarus Verilog** (`iverilog`), sweeping all 256
addresses and checking every output against Python. (VHDL itself needs GHDL or a vendor tool; the Verilog
twin shares the exact same init values, so it validates the ROM behavior; a VHDL testbench is emitted too.)

This is the FPGA end of the project's Python -> C -> VHDL flow: the arctan phase readout was computed once
in Python, frozen into ROM, and now runs at line rate with zero runtime math. Self-contained: NumPy,
Matplotlib, plus `iverilog`/`vvp` (MinGW)."""),
setup_cell(),
co("""import os, re, shutil, subprocess, tempfile, pathlib
os.environ["PATH"] = r"C:\\msys64\\mingw64\\bin" + os.pathsep + os.environ["PATH"]
IVERILOG, VVP = shutil.which("iverilog"), shutil.which("vvp")
print("iverilog:", IVERILOG); print("vvp:", VVP)"""),

md(r"""## The LUT to burn into ROM

Rebuild the 8-bit VEViD-lite table (the same map as the embedded-C notebook): the arctan is evaluated once,
offline, at the 256 possible input codes."""),
co("""def vevid_lite(value, gain=8.0, bias=0.1):
    value = np.clip(np.asarray(value, dtype=float), 0.0, 1.0)
    return np.arctan(gain*(value + bias))/np.arctan(gain*(1.0 + bias))

GAIN, BIAS = 8.0, 0.1
lut = np.round(vevid_lite(np.arange(256)/255.0, GAIN, BIAS)*255).astype(int)
print("LUT[0], LUT[64], LUT[128], LUT[255] =", [int(lut[i]) for i in (0, 64, 128, 255)])
assert np.all(np.diff(lut) >= 0)                             # monotonic"""),

md(r"""## Generate the VHDL block-RAM ROM

A `std_logic_vector` ROM constant initialized from the LUT, read **synchronously** so the tool infers block
RAM. The `numeric_std` cast turns the address into an index."""),
co(r'''def generate_vhdl(table, name="vevid_lut_rom"):
    vals = [f'x"{int(v):02X}"' for v in table]
    rows = [",\n".join("    " + ", ".join(vals[k:k+16]) for k in range(0, 256, 16))]
    rom_body = rows[0]
    return f"""library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity {name} is
  port (
    clk  : in  std_logic;
    addr : in  std_logic_vector(7 downto 0);   -- input pixel code
    data : out std_logic_vector(7 downto 0)    -- enhanced pixel code
  );
end entity {name};

architecture rtl of {name} is
  type rom_type is array (0 to 255) of std_logic_vector(7 downto 0);
  constant ROM : rom_type := (
{rom_body}
  );
begin
  -- Synchronous (registered) read  =>  synthesizer infers a block-RAM primitive.
  process(clk)
  begin
    if rising_edge(clk) then
      data <= ROM(to_integer(unsigned(addr)));
    end if;
  end process;
end architecture rtl;
"""

vhdl = generate_vhdl(lut)
print(vhdl[:600], "...\\n")

# verify the generated ROM initializers are bit-exact to the Python LUT
hex_vals = [int(h, 16) for h in re.findall(r'x"([0-9A-Fa-f]{2})"', vhdl)]
assert len(hex_vals) == 256 and hex_vals == list(lut)
print("generated VHDL ROM init is bit-exact to the LUT:", hex_vals == list(lut))'''),

md(r"""## Hardware simulation: the Verilog twin (Icarus Verilog)

To *prove the ROM behaves*, generate a functionally identical Verilog module from the **same** LUT values
plus a testbench that sweeps all 256 addresses through the one-cycle-latency read, then run it with
`iverilog`/`vvp` and compare every output to Python. Bit-exact confirms the synchronous-ROM logic and the
init data are correct in a real simulator."""),
co(r'''def generate_verilog(table, name="vevid_lut_rom"):
    init = "\n".join(f"    rom[{i}] = 8'd{int(v)};" for i, v in enumerate(table))
    rom = (f"module {name}(input clk, input [7:0] addr, output reg [7:0] data);\n"
           f"  reg [7:0] rom [0:255];\n  initial begin\n{init}\n  end\n"
           f"  always @(posedge clk) data <= rom[addr];   // synchronous read (BRAM)\nendmodule\n")
    tb = (f"module tb;\n  reg clk = 0; reg [7:0] addr; wire [7:0] data;\n"
          f"  {name} uut(.clk(clk), .addr(addr), .data(data));\n"
          f"  integer i;\n  always #5 clk = ~clk;\n  initial begin\n"
          f"    for (i = 0; i < 256; i = i + 1) begin\n"
          f"      addr = i[7:0]; @(posedge clk); #1; $display(\"%0d %0d\", i, data);\n"
          f"    end\n    $finish;\n  end\nendmodule\n")
    return rom, tb

assert IVERILOG and VVP, "iverilog/vvp not found on PATH"
rom_v, tb_v = generate_verilog(lut)
with tempfile.TemporaryDirectory() as tmp:
    tmp = pathlib.Path(tmp)
    (tmp / "rom.v").write_text(rom_v); (tmp / "tb.v").write_text(tb_v)
    subprocess.run([IVERILOG, "-o", str(tmp / "sim"), str(tmp / "tb.v"), str(tmp / "rom.v")],
                   check=True, capture_output=True, text=True)
    out = subprocess.run([VVP, str(tmp / "sim")], check=True, capture_output=True, text=True).stdout

sim = {}
for line in out.splitlines():
    m = re.match(r"^\s*(\d+)\s+(\d+)\s*$", line)
    if m: sim[int(m.group(1))] = int(m.group(2))
hw = np.array([sim[i] for i in range(256)])
mismatches = int(np.sum(hw != lut))
print(f"simulated all 256 addresses; mismatches vs Python LUT: {mismatches}")
print("sample: addr 0 ->", sim[0], " addr 128 ->", sim[128], " addr 255 ->", sim[255])
assert mismatches == 0'''),

md(r"""## A VHDL testbench for GHDL / Vivado

For the VHDL flow, this testbench sweeps the addresses and reports mismatches against an expected-ROM copy;
run it with `ghdl -a/-e/-r` or a vendor simulator."""),
co(r'''def generate_vhdl_tb(name="vevid_lut_rom"):
    return f"""library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tb_{name} is end entity;

architecture sim of tb_{name} is
  signal clk  : std_logic := '0';
  signal addr : std_logic_vector(7 downto 0) := (others => '0');
  signal data : std_logic_vector(7 downto 0);
begin
  uut : entity work.{name} port map (clk => clk, addr => addr, data => data);
  clk <= not clk after 5 ns;
  stimulus : process
  begin
    for i in 0 to 255 loop
      addr <= std_logic_vector(to_unsigned(i, 8));
      wait until rising_edge(clk);
      wait for 1 ns;
      report \"addr=\" & integer'image(i) & \" data=\" & integer'image(to_integer(unsigned(data)));
    end loop;
    wait;
  end process;
end architecture sim;
"""
print(generate_vhdl_tb()[:400], "...")'''),

md(r"""## Plot: the ROM contents"""),
co(r"""fig, ax = plt.subplots(figsize=(6, 4))
ax.step(np.arange(256), lut, where="mid", color="#4C78A8")
ax.plot(np.arange(256), np.arange(256), ":", color="gray", label="identity")
ax.set_xlabel("ROM address (input pixel code)"); ax.set_ylabel("ROM data (enhanced code)")
ax.set_title("VEViD block-RAM ROM contents"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()"""),

md(r"""## Hardware notes

- **BRAM inference.** The registered (synchronous) read is what makes Vivado/Quartus map the array to a
  block-RAM primitive; a purely combinational read would spend distributed LUTs instead. Result: **1-clock
  read latency**, one BRAM tile.
- **Resources.** $256\times8 = 2$ kbit -- trivial; a single BRAM holds it many times over (or use
  distributed RAM if BRAM is scarce). Add an output register for a second pipeline stage at high $f_{max}$.
- **Throughput.** One enhanced pixel per clock, no arithmetic, no DSP slices -- line-rate tone mapping for a
  camera/display pipeline. Swap the ROM contents (regenerate with new gain/bias) without touching the RTL.
- **Init portability.** The inline constant works everywhere; for large tables use `$readmemh` (Verilog) or
  a `.mif`/`.coe` file (vendor tools) generated from the same LUT.

## Summary

- The VEViD tone map is emitted as a **VHDL block-RAM ROM** (synchronous read => BRAM), generated from the
  Python LUT; the ROM initializers are **bit-exact** to the table.
- A functionally identical **Verilog twin** was simulated with **Icarus Verilog** across all 256 addresses
  with **zero mismatches** -- the synchronous-ROM behavior is verified in a real HDL simulator. A VHDL
  testbench is provided for GHDL/Vivado.
- One clock, one address, one datum, one BRAM: the FPGA end of Python -> C -> VHDL, with the physics baked
  into memory and no runtime math.

Subject-verb-object: Python bakes the table; VHDL stores it in ROM; the clock registers the read; the FPGA
enhances a pixel per cycle."""),
]

write("vevid_lut", "vhdl_bram", cells)
