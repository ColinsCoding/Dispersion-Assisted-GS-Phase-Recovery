"""Build notebooks/master_slave_bus_arbitration.ipynb -- "Master/Slave Bus
Protocols: I2C Arbitration and SPI Daisy Chaining, Bit-Accurately".

Companion notebook to dgs/master_slave_bus_arbitration.py, which fills a
real gap left by dgs/edge_ai_protocols.py: that module's own topology
table calls I2C a "multi-master bus" and SPI "master-slave, daisy," but
its actual functions only ever model one master talking to one slave.
This notebook demonstrates the real mechanisms: I2C's wired-AND
arbitration (why the numerically lower address always wins,
non-destructively) and SPI daisy-chaining (an actual bit-shift
simulation, including a real bug -- bit-reversal from a naive
insert-at-front implementation -- caught by hand-tracing before it
shipped).

Build:   py -3.13 scripts/build_master_slave_bus_arbitration_nb.py
Execute: py -3.13 -m jupyter nbconvert --to notebook --execute --inplace
         notebooks/master_slave_bus_arbitration.ipynb
"""
import pathlib
import nbformat as nbf

nb = nbf.v4.new_notebook()
md = lambda s: nbf.v4.new_markdown_cell(s)
co = lambda s: nbf.v4.new_code_cell(s)
cells = []

cells.append(md(r"""# Master/Slave Bus Protocols: I2C Arbitration and SPI Daisy Chaining, Bit-Accurately

[`dgs/edge_ai_protocols.py`](../dgs/edge_ai_protocols.py)'s own protocol
comparison table calls I2C a **"multi-master bus"** and SPI
**"master-slave, daisy"** — but its `i2c_transaction()` and `spi_frame()`
functions only ever model one master talking to one slave. Neither
multi-master **arbitration** (what happens when two masters start an I2C
transaction at the same instant) nor multi-slave **daisy-chaining** (what
actually happens on a chain of shift-register SPI slaves) was
implemented anywhere in this repo. This notebook, built around
[`dgs/master_slave_bus_arbitration.py`](../dgs/master_slave_bus_arbitration.py),
fills both gaps bit-accurately — actual simulations, not cycle-count
formulas."""))

cells.append(co("""import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))

from dgs.master_slave_bus_arbitration import (
    i2c_bus_arbitration, address_to_bits, spi_daisy_chain_shift, spi_chip_select_polling_time,
)

print("using dgs.master_slave_bus_arbitration directly (not reimplemented)")"""))

# ============================================================ PART 1: I2C arbitration mechanism
cells.append(md(r"""# Part 1 — I2C multi-master arbitration: why lower addresses win

SDA is **open-drain, wired-AND**: any device can pull it LOW, only the
shared pull-up resistor gets it HIGH, so the bus level is the AND of
every driver's intended bit. Two masters starting simultaneously each
drive their address bits (MSB first) and read back what actually
landed — the instant a master drives a 1 but reads back a 0, it has
LOST arbitration and backs off, non-destructively (no data corruption,
unlike an Ethernet-style collision).

Master A wants address `0x2A` (`0b0101010`), Master B wants `0x2C`
(`0b0101100`) — they agree on the first 4 bits, then diverge."""))

cells.append(co("""addr_A, addr_B = 0x2A, 0x2C
bits_A, bits_B = address_to_bits(addr_A), address_to_bits(addr_B)
print(f"Master A: 0x{addr_A:02X} = {bits_A}")
print(f"Master B: 0x{addr_B:02X} = {bits_B}")

result = i2c_bus_arbitration({"A": bits_A, "B": bits_B})
print(f"\\nbus sequence (wired-AND of both, while both still contend): {result['bus_sequence']}")
print(f"loser: {result['losers']}")
print(f"WINNER: Master {result['winner']}")

assert result["winner"] == "A"   # 0x2A < 0x2C -- the lower address must win
assert result["losers"] == [("B", 4)]   # they diverge exactly at bit index 4
print("\\n[OK] confirmed: the numerically lower address wins, and it loses exactly where the bit patterns diverge")"""))

# ============================================================ PART 2: it's a general rule
cells.append(md(r"""# Part 2 — Is this always true, or did that example get lucky?

Test the "lower address always wins" claim across several different
address pairs, including a 3-way arbitration — not just the one
hand-picked example above."""))

cells.append(co("""pairs = [(0x2A, 0x2C), (0x01, 0x7F), (0x10, 0x11), (0x00, 0x01), (0x55, 0x56)]
for a, b in pairs:
    r = i2c_bus_arbitration({"A": address_to_bits(a), "B": address_to_bits(b)})
    expected = "A" if a < b else "B"
    status = "OK" if r["winner"] == expected else "MISMATCH"
    print(f"  0x{a:02X} vs 0x{b:02X}: winner={r['winner']} (expected {expected})  [{status}]")
    assert r["winner"] == expected

r3 = i2c_bus_arbitration({"A": address_to_bits(0x50), "B": address_to_bits(0x40), "C": address_to_bits(0x30)})
print(f"\\nthree-way (0x50, 0x40, 0x30): winner={r3['winner']}  (expect C, the lowest)")
assert r3["winner"] == "C"

# the one case wired-AND genuinely cannot resolve: identical addresses
same = address_to_bits(0x42)
tie = i2c_bus_arbitration({"A": same, "B": same})
print(f"\\nidentical addresses (0x42 vs 0x42): winner={tie['winner']}  (a real, unresolvable tie)")
assert tie["winner"] is None
print("\\n[OK] the rule holds across every pair and a 3-way race, and the one genuine exception")
print("     (identical addresses) is reported honestly as unresolved, not forced to pick one")"""))

# ============================================================ PART 3: SPI daisy chain, and a caught bug
cells.append(md(r"""# Part 3 — SPI daisy chaining: an actual bit-shift, and a bug it caught

Three shift-register slaves share one clock/MOSI/MISO/CS line. After
exactly `3*8=24` clock edges, the master's 3 bytes should have
completely replaced the chain's contents, and the chain's *original*
3 bytes should have completely shifted out on MISO — a clean,
checkable identity, not an approximation.

**A bug this notebook's build process actually caught**: the first
version of `spi_daisy_chain_shift` inserted new bits at a fixed "front"
position each clock (`list.insert(0, bit)`). That silently
**bit-reverses** each byte across a full shift — inserting at a fixed
position, over and over, reverses arrival order. It was caught by
hand-tracing this exact 3-byte example and noticing the result didn't
satisfy the identity above, then fixed by switching to proper FIFO
`deque` semantics (`popleft` at the MISO end, `append` at the MOSI
end)."""))

cells.append(co("""slaves_before = [0xAA, 0xBB, 0xCC]
master_out = [0x11, 0x22, 0x33]

result = spi_daisy_chain_shift(slaves_before, master_out)
print(f"before:        {[hex(b) for b in slaves_before]}")
print(f"master pushes: {[hex(b) for b in master_out]}")
print(f"clocks needed: {result['clocks_required']}")
print(f"after:         {[hex(b) for b in result['new_register_contents']]}")
print(f"MISO saw:      {[hex(b) for b in result['miso_output_bytes']]}")

assert result["new_register_contents"] == master_out
assert result["miso_output_bytes"] == slaves_before
print("\\n[OK] a full N*8-clock shift exactly replaces the register with what the master sent,")
print("     and exactly outputs the chain's original contents on MISO -- the defining identity")
print("     of a complete daisy-chain shift, verified directly rather than assumed")"""))

# ============================================================ PART 4: chip-select vs daisy chain tradeoff
cells.append(md(r"""# Part 4 — The other SPI topology: independent chip-select

Daisy-chaining isn't the only way to put many SPI slaves on one bus —
the more common approach in practice is one dedicated CS line per
slave, addressed one at a time. Different tradeoff: no shift-through
latency, but wiring scales with slave count instead of staying at 4
wires regardless of chain length."""))

cells.append(co("""poll = spi_chip_select_polling_time(n_slaves=8, bits_per_transfer=16, clock_hz=10e6)
print(f"8 independently-selected slaves, 16-bit transfers, 10 MHz clock:")
print(f"  time to poll all 8 once: {poll['total_seconds_to_poll_all']*1e6:.2f} us")
print(f"  max poll rate: {poll['max_poll_rate_hz']:.1f} Hz")

# verify the claimed linear scaling with slave count, not just for N=8
one = spi_chip_select_polling_time(1, 16, 10e6)
sixteen = spi_chip_select_polling_time(16, 16, 10e6)
print(f"\\n1 slave: {one['total_seconds_to_poll_all']*1e6:.3f} us,  16 slaves: {sixteen['total_seconds_to_poll_all']*1e6:.3f} us")
assert abs(sixteen["total_seconds_to_poll_all"] - 16*one["total_seconds_to_poll_all"]) < 1e-9
print("[OK] confirmed: independent chip-select polling time scales exactly linearly with slave count")"""))

# ============================================================ PART 5: summary
cells.append(md(r"""# Part 5 — Summary

| Mechanism | Verified |
|---|---|
| I2C wired-AND arbitration: lower address always wins | 5 pairs + a 3-way race, non-destructively (Parts 1-2) |
| Identical addresses are a genuine, unresolvable tie | reported honestly, not forced (Part 2) |
| SPI daisy chain: full shift exactly swaps in master data, shifts out original contents | exact identity check, both directions (Part 3) |
| SPI independent chip-select: polling time scales linearly with slave count | checked at N=1, 8, 16 (Part 4) |

The real lesson from Part 3's caught bug: a shift-register simulation
that "looks right" (produces *some* plausible-looking bytes) can still
be silently wrong in a way that only shows up when checked against the
actual defining identity of the operation — here, that a complete
N-clock shift must exactly swap old contents for new, not merely
produce different-looking bytes. `dgs/edge_ai_protocols.py`'s existing
single-transaction timing formulas were never wrong in this way because
they never attempted a real bit-level simulation in the first place —
this module's bugs were the direct cost of actually simulating the
mechanism, not something a formula-only approach would have hit."""))

nb["cells"] = cells
out = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "master_slave_bus_arbitration.ipynb"
out.write_text(nbf.writes(nb), encoding="utf-8")
print(f"wrote {out}, {len(cells)} cells")
