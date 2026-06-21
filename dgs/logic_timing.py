"""Digital-logic timing -- propagation delay, critical path, and glitches.

Combinational logic is not instantaneous: each gate has a *propagation delay*, and
delays add up along a path. The slowest input-to-output path is the **critical
path**; its delay sets the maximum clock frequency. And when two paths to the same
gate have *different* delays, the output can briefly flip to the wrong value before
settling -- a **glitch/hazard**, the digital cousin of error propagation.

This module models a combinational circuit as a DAG of gates and computes:
  * logic values        -- evaluate(inputs)            (topological order)
  * arrival times       -- arrival()                   (delay accumulated to each node)
  * the critical path   -- critical_path(), fmax()
  * static hazards      -- detect_static_hazard()

The worked example is the ripple-carry adder, whose carry chain makes the delay
grow *linearly* in the bit width -- the reason carry-lookahead adders exist. Pure
Python; the topological sort is the same DFS idea as dgs.algorithms. Education.
"""

from functools import reduce

# default per-gate propagation delays (arbitrary time units; override per gate)
DEFAULT_DELAYS = {"AND": 1, "OR": 1, "NAND": 1, "NOR": 1, "NOT": 1, "XOR": 2, "XNOR": 2}

GATE_FUNCS = {
    "AND":  lambda *b: int(all(b)),
    "OR":   lambda *b: int(any(b)),
    "NAND": lambda *b: int(not all(b)),
    "NOR":  lambda *b: int(not any(b)),
    "NOT":  lambda b: 1 - b,
    "XOR":  lambda *b: reduce(lambda x, y: x ^ y, b),
    "XNOR": lambda *b: 1 - reduce(lambda x, y: x ^ y, b),
}


class Circuit:
    """A combinational circuit: primary inputs plus gates wired into a DAG."""

    def __init__(self):
        self.kind = {}      # node -> "INPUT" or a gate name
        self.inputs = {}    # node -> list of upstream node names
        self.delay = {}     # node -> propagation delay (0 for inputs)
        self.outputs = []   # designated output nodes

    def add_input(self, name):
        self.kind[name] = "INPUT"; self.inputs[name] = []; self.delay[name] = 0
        return name

    def add_gate(self, name, kind, inputs, delay=None):
        if kind not in GATE_FUNCS:
            raise ValueError(f"unknown gate {kind!r}")
        self.kind[name] = kind
        self.inputs[name] = list(inputs)
        self.delay[name] = DEFAULT_DELAYS[kind] if delay is None else delay
        return name

    def mark_output(self, *names):
        self.outputs.extend(names)

    def _topo_order(self):
        """Dependency order (inputs first) via DFS -- the same idea as dgs.algorithms.dfs."""
        order, seen = [], set()
        def visit(n):
            if n in seen:
                return
            seen.add(n)
            for src in self.inputs[n]:
                visit(src)
            order.append(n)
        for n in self.kind:
            visit(n)
        return order

    def evaluate(self, values):
        """Logic value at every node given primary-input `values` {name: 0/1}."""
        val = {}
        for n in self._topo_order():
            if self.kind[n] == "INPUT":
                val[n] = int(values[n])
            else:
                val[n] = GATE_FUNCS[self.kind[n]](*(val[s] for s in self.inputs[n]))
        return val

    def arrival(self, input_arrivals=None):
        """Signal arrival time at each node: max over input arrivals + this gate's delay."""
        arr = {}
        for n in self._topo_order():
            if self.kind[n] == "INPUT":
                arr[n] = 0 if input_arrivals is None else input_arrivals.get(n, 0)
            else:
                arr[n] = max(arr[s] for s in self.inputs[n]) + self.delay[n]
        return arr

    def critical_path(self):
        """Return (delay, path) for the slowest input->output path."""
        arr = self.arrival()
        ends = self.outputs or [n for n in self.kind if n not in
                                {s for ins in self.inputs.values() for s in ins}]
        sink = max(ends, key=lambda n: arr[n])
        path = [sink]
        while self.inputs[sink]:
            sink = max(self.inputs[sink], key=lambda s: arr[s])
            path.append(sink)
        return arr[max(ends, key=lambda n: arr[n])], list(reversed(path))

    def fmax(self):
        """Maximum clock frequency = 1 / critical-path delay (in 1/time-unit)."""
        d, _ = self.critical_path()
        return 1.0 / d if d > 0 else float("inf")


# ── the worked example: ripple-carry adder ──────────────────────────
def ripple_carry_adder(n_bits):
    """Build an n-bit ripple-carry adder as a Circuit. Each full adder:
        sum  = a XOR b XOR cin
        cout = (a AND b) OR (cin AND (a XOR b))
    The carry ripples a[0]->cout[0]->cout[1]->...; that chain is the critical path."""
    c = Circuit()
    cin = c.add_input("cin")
    carry = cin
    for i in range(n_bits):
        a, b = c.add_input(f"a{i}"), c.add_input(f"b{i}")
        axb = c.add_gate(f"axb{i}", "XOR", [a, b])
        c.add_gate(f"sum{i}", "XOR", [axb, carry]); c.mark_output(f"sum{i}")
        ab = c.add_gate(f"ab{i}", "AND", [a, b])
        cab = c.add_gate(f"cab{i}", "AND", [carry, axb])
        carry = c.add_gate(f"cout{i}", "OR", [ab, cab])
    c.mark_output(carry)
    return c


def ripple_carry_delay(n_bits, gate_delay=1):
    """Closed-form critical-path delay of a ripple adder: O(n). The carry enters each
    stage through AND then OR (2 gates), so the chain is ~2n + the first XOR."""
    return 2 * n_bits * gate_delay + DEFAULT_DELAYS["XOR"]


def carry_lookahead_delay(n_bits, gate_delay=1):
    """Carry-lookahead computes all carries in parallel -> roughly CONSTANT depth
    (generate/propagate, then a wide AND-OR), independent of n. The payoff: O(1) vs
    the ripple adder's O(n). (Idealized -- real CLA fan-in forces O(log n).)"""
    return 3 * gate_delay + DEFAULT_DELAYS["XOR"]


# ── glitches / static hazards: unequal path delays cause a transient ─
def detect_static_hazard(fast_delay, slow_delay):
    """A static hazard: two paths reconverge with different delays, so the output can
    momentarily glitch during the window between the fast and slow path settling.
    Returns the glitch window width (0 means no hazard)."""
    return abs(slow_delay - fast_delay)


# ── bit shifts: the timing of a shifter, and shift = multiply/divide ─
def logical_shift(value, k, width=8):
    """Logical shift: <<k multiplies by 2^k, >>k divides by 2^k (masked to `width`)."""
    mask = (1 << width) - 1
    return ((value << k) & mask) if k >= 0 else ((value & mask) >> -k)


def barrel_shifter_levels(width):
    """A barrel shifter shifts by any amount in ceil(log2(width)) mux levels -- so its
    delay is O(log n), not O(n). (Why hardware shifts are 'free': log-depth.)"""
    levels = 0
    while (1 << levels) < width:
        levels += 1
    return levels


if __name__ == "__main__":
    for n in (4, 8, 16):
        add = ripple_carry_adder(n)
        d, path = add.critical_path()
        print(f"{n:2d}-bit ripple adder: critical delay = {d:3d}  "
              f"(closed form {ripple_carry_delay(n)}, CLA {carry_lookahead_delay(n)})  "
              f"fmax ~ {add.fmax():.3f}/unit")
    # check the adder actually adds: 5 + 6 = 11
    add = ripple_carry_adder(4)
    vals = {"cin": 0, **{f"a{i}": (5 >> i) & 1 for i in range(4)},
            **{f"b{i}": (6 >> i) & 1 for i in range(4)}}
    out = add.evaluate(vals)
    s = sum((out[f"sum{i}"] << i) for i in range(4)) + (out["cout3"] << 4)
    print(f"5 + 6 = {s}   |   shift: 5<<1 = {logical_shift(5,1)} (x2), "
          f"shift levels for 32-bit = {barrel_shifter_levels(32)}")
