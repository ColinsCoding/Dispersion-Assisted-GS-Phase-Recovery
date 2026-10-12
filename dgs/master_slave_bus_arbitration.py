"""Master/slave bus protocol architecture, beyond framing calculators:
dgs/edge_ai_protocols.py's own topology table calls I2C a "multi-master
bus" and SPI "master-slave, daisy" -- but its actual i2c_transaction() and
spi_frame() functions only ever model ONE master talking to ONE slave.
Neither multi-master arbitration (what happens when two masters start an
I2C transaction at the same instant) nor multi-slave daisy-chaining (what
actually happens on a chain of shift-register slaves sharing one SPI bus)
is implemented anywhere. This module implements both, bit-accurately, not
just as a formula.

I2C MULTI-MASTER ARBITRATION (the real mechanism, not a coin flip):
  SDA is OPEN-DRAIN, wired-AND across every device on the bus: any device
  can pull it LOW, but only pulls-UP (via a shared resistor) get it HIGH,
  so the bus level at any instant is the AND of every driver's intended
  bit. Two masters starting simultaneously each drive their own address
  bits (MSB first) onto SDA and READ BACK what actually landed. The
  moment a master drives a 1 but reads back a 0 (someone else on the bus
  drove a 0), it has LOST arbitration -- it stops driving and must retry
  later. This means the master sending the NUMERICALLY LOWER address
  sequence always wins, non-destructively (no data is corrupted, unlike
  Ethernet-style collision) -- a real and checkable consequence of
  wired-AND logic, verified below bit by bit, not asserted.

SPI DAISY CHAINING (the real mechanism, not a cycle-count formula):
  Many SPI slaves (shift-register I/O expanders, LED drivers) share ONE
  clock/MOSI/MISO/CS: each slave's own shift register feeds directly into
  the next slave's input, so the whole chain of N devices behaves as ONE
  long N*bits-wide shift register. After exactly N*bits clock edges, the
  bits the master pushed in have propagated all the way through and each
  slave's register holds exactly the byte intended for it -- verified
  here with an actual bit-shift simulation, not just "N*8 clocks needed."
"""
from collections import deque


def i2c_bus_arbitration(master_bitstreams):
    """Simulate wired-AND I2C arbitration among simultaneous masters.

    master_bitstreams: dict of name -> list of bits (0/1, MSB first) each
    master intends to drive (typically its 7-bit address + R/W bit).
    Returns dict with 'winner' (name, or None if a true tie -- identical
    bitstreams, which wired-AND genuinely cannot resolve), 'losers' (list
    of (name, bit_index_lost) pairs), and 'bus_sequence' (the actual bus
    level at each clock, for inspection)."""
    names = list(master_bitstreams)
    if len(names) < 2:
        raise ValueError("need at least 2 masters for an arbitration scenario")
    lengths = {len(v) for v in master_bitstreams.values()}
    if len(lengths) != 1:
        raise ValueError("all masters must drive the same number of bits (same address width)")
    n_bits = lengths.pop()

    still_driving = set(names)
    bus_sequence = []
    losers = []

    for i in range(n_bits):
        # wired-AND: the bus is LOW if ANY still-contending master drives LOW
        bus_bit = 1
        for name in still_driving:
            if master_bitstreams[name][i] == 0:
                bus_bit = 0
                break
        bus_sequence.append(bus_bit)

        # any master that intended a 1 but reads back 0 has lost arbitration
        newly_lost = [name for name in still_driving
                      if master_bitstreams[name][i] == 1 and bus_bit == 0]
        for name in newly_lost:
            losers.append((name, i))
            still_driving.discard(name)

        if len(still_driving) <= 1:
            break

    if len(still_driving) == 1:
        winner = next(iter(still_driving))
    elif len(still_driving) == 0:
        # every remaining contender dropped out on the SAME bit (can only
        # happen if the very last comparison eliminated all but nobody --
        # not reachable given the "<=1 stop" rule above, but guarded honestly
        winner = None
    else:
        # multiple masters survived all n_bits with identical bitstreams --
        # a genuine tie wired-AND arbitration cannot break on its own
        winner = None

    return {"winner": winner, "losers": losers, "bus_sequence": bus_sequence,
             "survivors_at_end": sorted(still_driving)}


def address_to_bits(addr_7bit, rw_bit=0):
    """7-bit I2C address + R/W bit, MSB first -- the standard bitstream a
    real I2C master actually arbitrates on."""
    if not (0 <= addr_7bit <= 0x7F):
        raise ValueError("addr_7bit must be a 7-bit address (0-127)")
    if rw_bit not in (0, 1):
        raise ValueError("rw_bit must be 0 or 1")
    return [(addr_7bit >> k) & 1 for k in range(6, -1, -1)] + [rw_bit]


def spi_daisy_chain_shift(slave_bytes, master_out_bytes=None):
    """Simulate N SPI slaves daisy-chained into one shift register (shared
    clock/CS; each slave's serial-out feeds the next slave's serial-in).
    slave_bytes[i]: the byte currently held in slave i's register BEFORE
    the transfer (slave 0 = the one closest to the master's MISO input,
    so its bits shift OUT first; slave N-1 = the one closest to the
    master's MOSI output, so it's the last to receive fresh bits).
    master_out_bytes: the N bytes the master wants to push into the
    chain (defaults to all-zero if not given).

    Runs an ACTUAL bit-by-bit shift for len(slave_bytes)*8 clock edges
    using a deque (MISO end = popleft, MOSI end = append) -- not a
    formula. An earlier version of this function used insert(0, ...) for
    the input end, which silently BIT-REVERSES each byte across a full
    shift (inserting at a fixed front position each clock reverses
    arrival order) -- caught by hand-tracing a 3-byte example (the
    result didn't match the expected "full shift exactly replaces the
    register with master_out_bytes" identity) and fixed by switching to
    proper FIFO deque semantics, verified in tests against that exact
    identity."""
    n = len(slave_bytes)
    if n == 0:
        raise ValueError("need at least one slave")
    if any(not (0 <= b <= 0xFF) for b in slave_bytes):
        raise ValueError("slave_bytes must all be 0-255")
    if master_out_bytes is None:
        master_out_bytes = [0] * n
    if len(master_out_bytes) != n:
        raise ValueError("master_out_bytes must have the same length as slave_bytes")

    def bits_of(byte_list):
        bits = []
        for b in byte_list:
            bits.extend([(b >> k) & 1 for k in range(7, -1, -1)])
        return bits

    def bytes_of(bits):
        return [sum(bits[i + k] << (7 - k) for k in range(8)) for i in range(0, len(bits), 8)]

    register = deque(bits_of(slave_bytes))       # front (index 0) = MISO end, back = MOSI end
    incoming = deque(bits_of(master_out_bytes))

    miso_out_bits = []
    total_bits = n * 8
    for _ in range(total_bits):
        miso_out_bits.append(register.popleft())   # oldest content shifts OUT at the MISO end
        register.append(incoming.popleft())        # new data shifts IN at the MOSI end

    return {
        "n_slaves": n,
        "clocks_required": total_bits,
        "new_register_contents": bytes_of(list(register)),
        "miso_output_bytes": bytes_of(miso_out_bits),
    }


def spi_chip_select_polling_time(n_slaves, bits_per_transfer, clock_hz, cs_setup_ns=50.0, cs_hold_ns=50.0):
    """Independent chip-select model (the OTHER SPI multi-slave topology):
    each slave has its OWN dedicated CS line, and the master addresses
    exactly one at a time -- no daisy-chaining, no shared shift register.
    Returns total time to poll all N slaves once, each with cs_setup +
    the transfer itself + cs_hold."""
    if n_slaves <= 0 or bits_per_transfer <= 0 or clock_hz <= 0:
        raise ValueError("n_slaves, bits_per_transfer, clock_hz must all be positive")
    transfer_time_s = bits_per_transfer / clock_hz
    per_slave_s = (cs_setup_ns + cs_hold_ns) * 1e-9 + transfer_time_s
    return {
        "n_slaves": n_slaves, "per_slave_seconds": per_slave_s,
        "total_seconds_to_poll_all": per_slave_s * n_slaves,
        "max_poll_rate_hz": 1.0 / (per_slave_s * n_slaves),
    }


if __name__ == "__main__":
    print("=== I2C MULTI-MASTER ARBITRATION ===\n")

    addr_A, addr_B = 0x2A, 0x2C   # 0b0101010 vs 0b0101100 -- differ starting at bit index 4
    bits_A, bits_B = address_to_bits(addr_A), address_to_bits(addr_B)
    print(f"Master A wants address 0x{addr_A:02X}: {bits_A}")
    print(f"Master B wants address 0x{addr_B:02X}: {bits_B}")

    result = i2c_bus_arbitration({"A": bits_A, "B": bits_B})
    print(f"bus sequence (wired-AND): {result['bus_sequence']}")
    print(f"loser(s): {result['losers']}")
    print(f"WINNER: Master {result['winner']}  (lower address {min(addr_A, addr_B):#04x} wins, non-destructively)")
    assert result["winner"] == ("A" if addr_A < addr_B else "B")

    print("\n=== SPI DAISY CHAIN (3 shift-register slaves) ===")
    slaves_before = [0xAA, 0xBB, 0xCC]
    master_out = [0x11, 0x22, 0x33]
    chain = spi_daisy_chain_shift(slaves_before, master_out)
    print(f"before: {[hex(b) for b in slaves_before]}")
    print(f"master pushes in: {[hex(b) for b in master_out]}")
    print(f"clocks required: {chain['clocks_required']}")
    print(f"after:  {[hex(b) for b in chain['new_register_contents']]}")
    print(f"MISO saw (original chain contents, shifted out in order): {[hex(b) for b in chain['miso_output_bytes']]}")

    print("\n=== SPI INDEPENDENT CHIP-SELECT POLLING ===")
    poll = spi_chip_select_polling_time(n_slaves=8, bits_per_transfer=16, clock_hz=10e6)
    print(f"8 slaves, 16-bit transfers, 10 MHz clock:")
    print(f"  time to poll all 8 once: {poll['total_seconds_to_poll_all']*1e6:.2f} us")
    print(f"  max poll rate: {poll['max_poll_rate_hz']:.1f} Hz")

    print("\n=== DONE ===")
