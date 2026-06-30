"""VHDL / RTL concepts in Python: IEEE std_logic types, combinational and
sequential circuits, finite-state machines with transition diagrams, and
register-transfer-level (RTL) descriptions.

Three levels of abstraction mirrored in this module:
  Level 0 -- gate / boolean (see also dgs/digital_logic.py)
  Level 1 -- RTL: registers + datapath operations between clock edges
  Level 2 -- behavioural/architectural: process blocks, entities, architectures

No external dependencies beyond numpy and sympy (no scipy on py-3.13).
"""

from __future__ import annotations
import numpy as np
import sympy as sp
from typing import Any, Callable, Dict, List, Optional, Tuple


# ── IEEE std_logic encoding ──────────────────────────────────────────────────
STD_LOGIC_VALUES = ("U", "X", "0", "1", "Z", "W", "L", "H", "-")
# U=uninitialised, X=unknown, 0/1 logic, Z=high-Z, W=weak unknown,
# L/H=weak 0/1, - = don't-care

def std_logic_resolve(a: str, b: str) -> str:
    """Resolution table for wired-OR / open-drain lines (IEEE 1164)."""
    priority = {"U": 0, "X": 1, "0": 2, "1": 3, "Z": 4, "W": 5, "L": 6, "H": 7, "-": 8}
    resolve = {
        ("0", "0"): "0", ("1", "1"): "1", ("Z", "Z"): "Z",
        ("0", "Z"): "0", ("Z", "0"): "0",
        ("1", "Z"): "1", ("Z", "1"): "1",
        ("0", "1"): "X", ("1", "0"): "X",
        ("L", "L"): "L", ("H", "H"): "H",
        ("L", "Z"): "L", ("Z", "L"): "L",
        ("H", "Z"): "H", ("Z", "H"): "H",
    }
    key = (a, b) if priority.get(a, 9) <= priority.get(b, 9) else (b, a)
    return resolve.get(key, "X")


def std_logic_and(a: str, b: str) -> str:
    """AND of two std_logic values (handles X/Z/U correctly)."""
    if "0" in (a, b):
        return "0"
    if a == "1" and b == "1":
        return "1"
    if "U" in (a, b):
        return "U"
    return "X"


def std_logic_or(a: str, b: str) -> str:
    if "1" in (a, b):
        return "1"
    if a == "0" and b == "0":
        return "0"
    if "U" in (a, b):
        return "U"
    return "X"


def std_logic_not(a: str) -> str:
    return {"0": "1", "1": "0", "U": "U", "X": "X", "Z": "X",
            "L": "1", "H": "0", "-": "-", "W": "X"}.get(a, "X")


# ── std_logic_vector arithmetic ──────────────────────────────────────────────
def slv_to_int(bits: str, signed: bool = False) -> int:
    """std_logic_vector (MSB-first string e.g. '1010') to integer."""
    n = len(bits)
    val = int(bits, 2)
    if signed and bits[0] == "1":
        val -= (1 << n)
    return val


def int_to_slv(value: int, width: int) -> str:
    """Integer to std_logic_vector of given width (two's complement)."""
    if value < 0:
        value = value + (1 << width)
    return format(value & ((1 << width) - 1), f"0{width}b")


# ── Combinational circuit: LUT-based ────────────────────────────────────────
class CombinationalBlock:
    """A purely combinational block described by a truth table.

    Parameters
    ----------
    input_names : list[str]
        Input port names (MSB-first ordering).
    output_names : list[str]
        Output port names.
    table : dict[tuple, tuple]
        Maps input tuple (0/1 per port) to output tuple.
        Missing entries resolve to 'X' (unknown).
    """

    def __init__(self, input_names: List[str], output_names: List[str],
                 table: Dict[Tuple, Tuple]):
        self.input_names = input_names
        self.output_names = output_names
        self.table = table

    def evaluate(self, **inputs) -> Dict[str, int]:
        key = tuple(inputs[n] for n in self.input_names)
        out = self.table.get(key)
        if out is None:
            return {n: "X" for n in self.output_names}
        return dict(zip(self.output_names, out))

    def truth_table_str(self) -> str:
        header = " ".join(self.input_names) + " | " + " ".join(self.output_names)
        sep = "-" * len(header)
        rows = [header, sep]
        for key, val in sorted(self.table.items()):
            row = " ".join(str(k) for k in key) + " | " + " ".join(str(v) for v in val)
            rows.append(row)
        return "\n".join(rows)


def build_full_adder_block() -> CombinationalBlock:
    """1-bit full adder as a CombinationalBlock (truth table)."""
    table = {}
    for a in (0, 1):
        for b in (0, 1):
            for cin in (0, 1):
                s = a ^ b ^ cin
                cout = (a & b) | (b & cin) | (a & cin)
                table[(a, b, cin)] = (s, cout)
    return CombinationalBlock(["A", "B", "Cin"], ["Sum", "Cout"], table)


# ── D flip-flop ─────────────────────────────────────────────────────────────
class DFlipFlop:
    """Positive-edge triggered D flip-flop with synchronous reset.

    VHDL equivalent:
        process(clk) begin
          if rising_edge(clk) then
            if rst = '1' then Q <= '0';
            else Q <= D; end if;
          end if;
        end process;
    """

    def __init__(self, init: int = 0):
        self.Q = init
        self._next = init

    def setup(self, D: int, rst: int = 0):
        """Set up next state before clock edge."""
        self._next = 0 if rst else (D & 1)

    def clock_edge(self):
        """Positive clock edge: latch _next into Q."""
        self.Q = self._next

    def simulate(self, D_seq: List[int], rst_seq: Optional[List[int]] = None) -> List[int]:
        """Run n clock cycles, return Q trace."""
        n = len(D_seq)
        if rst_seq is None:
            rst_seq = [0] * n
        trace = []
        for d, r in zip(D_seq, rst_seq):
            self.setup(d, r)
            self.clock_edge()
            trace.append(self.Q)
        return trace


# ── Register (N-bit D-FF bank) ───────────────────────────────────────────────
class Register:
    """N-bit parallel register with synchronous load and reset."""

    def __init__(self, width: int = 8):
        self.width = width
        self.value = 0

    def load(self, data: int, en: int = 1, rst: int = 0):
        """Clock edge: load data if en=1, reset if rst=1 (rst takes priority)."""
        if rst:
            self.value = 0
        elif en:
            self.value = int(data) & ((1 << self.width) - 1)

    def read(self) -> int:
        return self.value

    def read_slv(self) -> str:
        return int_to_slv(self.value, self.width)


# ── Finite State Machine ─────────────────────────────────────────────────────
class FiniteStateMachine:
    """Moore FSM with typed states and transition table.

    Parameters
    ----------
    states : list[str]
        All valid state names.
    initial : str
        Initial state.
    transitions : dict[str, dict]
        {state: {input_symbol: next_state}}
    outputs : dict[str, Any]
        {state: output_value}  (Moore: output depends only on state)
    """

    def __init__(self, states: List[str], initial: str,
                 transitions: Dict[str, Dict],
                 outputs: Dict[str, Any]):
        if initial not in states:
            raise ValueError(f"initial state '{initial}' not in states")
        self.states = states
        self.current = initial
        self.initial = initial
        self.transitions = transitions
        self.outputs = outputs
        self._history: List[str] = [initial]

    def reset(self):
        self.current = self.initial
        self._history = [self.initial]

    def step(self, inp) -> Any:
        """Apply one input, advance state, return current output."""
        nxt = self.transitions.get(self.current, {}).get(inp)
        if nxt is None:
            raise ValueError(f"No transition from '{self.current}' on input '{inp}'")
        self.current = nxt
        self._history.append(self.current)
        return self.outputs[self.current]

    def run(self, inputs: List) -> List[Any]:
        return [self.step(i) for i in inputs]

    def transition_diagram_str(self) -> str:
        """ASCII transition diagram."""
        lines = ["FSM Transition Diagram", "=" * 40]
        for state, trans in self.transitions.items():
            out = self.outputs.get(state, "?")
            tag = " [INIT]" if state == self.initial else ""
            lines.append(f"  {state} (out={out}){tag}")
            for inp, nxt in trans.items():
                lines.append(f"    --{inp}--> {nxt}")
        return "\n".join(lines)


def build_traffic_light_fsm() -> FiniteStateMachine:
    """3-state traffic light: GREEN -> YELLOW -> RED -> GREEN.
    Input '1' = timer expired; input '0' = hold current state.
    Output = light color string.
    """
    return FiniteStateMachine(
        states=["GREEN", "YELLOW", "RED"],
        initial="GREEN",
        transitions={
            "GREEN":  {0: "GREEN",  1: "YELLOW"},
            "YELLOW": {0: "YELLOW", 1: "RED"},
            "RED":    {0: "RED",    1: "GREEN"},
        },
        outputs={"GREEN": "green", "YELLOW": "yellow", "RED": "red"},
    )


def build_sequence_detector_fsm() -> FiniteStateMachine:
    """Detect bit sequence '101' (overlapping). Output=1 when detected.
    States: S0=idle, S1=got_1, S2=got_10, S3=got_101
    """
    return FiniteStateMachine(
        states=["S0", "S1", "S2", "S3"],
        initial="S0",
        transitions={
            "S0": {0: "S0", 1: "S1"},
            "S1": {0: "S2", 1: "S1"},
            "S2": {0: "S0", 1: "S3"},
            "S3": {0: "S2", 1: "S1"},  # overlap: S3 on '1' -> S1 (new '1')
        },
        outputs={"S0": 0, "S1": 0, "S2": 0, "S3": 1},
    )


# ── RTL datapath: N-bit ALU ──────────────────────────────────────────────────
class ALU:
    """Simple N-bit ALU with opcode select (RTL level).

    VHDL process equivalent with case statement on op_sel.
    Opcodes:
        0b000 = ADD, 0b001 = SUB, 0b010 = AND, 0b011 = OR,
        0b100 = XOR, 0b101 = NOT A, 0b110 = SHL, 0b111 = SHR
    """

    OPCODES = {
        0b000: "ADD", 0b001: "SUB", 0b010: "AND",
        0b011: "OR",  0b100: "XOR", 0b101: "NOT_A",
        0b110: "SHL", 0b111: "SHR",
    }

    def __init__(self, width: int = 8):
        self.width = width
        self._mask = (1 << width) - 1

    def execute(self, A: int, B: int, op: int) -> Tuple[int, Dict[str, int]]:
        """Return (result, flags) where flags = {Z, N, C, V}."""
        A &= self._mask
        B &= self._mask
        op &= 0b111

        if op == 0b000:   result = A + B
        elif op == 0b001: result = A - B
        elif op == 0b010: result = A & B
        elif op == 0b011: result = A | B
        elif op == 0b100: result = A ^ B
        elif op == 0b101: result = (~A) & self._mask
        elif op == 0b110: result = (A << 1) & self._mask
        else:              result = A >> 1

        carry = int(result > self._mask)
        result &= self._mask
        flags = {
            "Z": int(result == 0),
            "N": int((result >> (self.width - 1)) & 1),
            "C": carry,
            "V": int(op in (0b000, 0b001) and carry),
        }
        return result, flags


# ── SymPy RTL: symbolic register-transfer equations ─────────────────────────
def rtl_pipeline_sympy(n_stages: int = 4) -> Dict[str, Any]:
    """Model an N-stage pipeline symbolically in SymPy.

    Returns clock period T, throughput, and latency as symbolic expressions.
    Stage delays: d_1, d_2, ..., d_n (combinational delays between registers).
    Pipeline register overhead: t_reg (setup + hold).
    """
    sp.init_printing(use_latex=False)
    d = sp.symbols(f"d_1:{n_stages + 1}", positive=True)
    t_reg = sp.Symbol("t_reg", positive=True)

    T_clock = sp.Max(*d) + t_reg
    throughput = sp.Rational(1, 1) / T_clock      # 1 result per clock
    latency = n_stages * T_clock                   # n stages to fill

    return {
        "stage_delays": d,
        "T_clock": T_clock,
        "throughput": throughput,
        "latency": latency,
        "n_stages": n_stages,
    }


# ── Work-energy theorem (separate from RTL, same file for notebook utility) ──
def work_energy_sympy() -> Dict[str, Any]:
    """Symbolic work-energy theorem + kinematic results in SymPy.

    Returns dict of symbolic expressions ready for init_printing display.
    """
    m, v, v0, vf, F, d, t, g, h = sp.symbols(
        "m v v_0 v_f F d t g h", positive=True
    )
    mu, N_n = sp.symbols("mu N", positive=True)   # friction coeff, normal force

    KE = sp.Rational(1, 2) * m * v**2
    KE0 = sp.Rational(1, 2) * m * v0**2
    KEf = sp.Rational(1, 2) * m * vf**2

    W_net = KEf - KE0                              # work-energy theorem
    W_grav = m * g * h                             # work by gravity
    W_friction = -mu * N_n * d                     # work by kinetic friction (negative)

    W = sp.Symbol("W_net", real=True)              # free symbol for vf equation
    vf_from_W = sp.sqrt(v0**2 + 2 * W / m)        # vf from net work
    P_inst = F * v                                  # instantaneous power
    P_avg = W_net / t                               # average power

    return {
        "KE": KE,
        "W_net = DeltaKE": sp.Eq(F * d, W_net),
        "W_grav": W_grav,
        "W_friction": W_friction,
        "vf_from_work": sp.Eq(vf, vf_from_W),   # vf = sqrt(v0^2 + 2*W_net/m)
        "power_inst": sp.Eq(sp.Symbol("P"), P_inst),
        "power_avg": sp.Eq(sp.Symbol("P_avg"), P_avg),
        "conservation_with_friction": sp.Eq(
            KE0 + W_grav + W_friction, KEf
        ),
    }


if __name__ == "__main__":
    print("=== std_logic resolution ===")
    print(f"resolve('0','1') = {std_logic_resolve('0','1')}")
    print(f"resolve('1','Z') = {std_logic_resolve('1','Z')}")
    print(f"AND('X','0') = {std_logic_and('X','0')}")

    print("\n=== std_logic_vector ===")
    print(f"'1010' -> {slv_to_int('1010')} (unsigned), {slv_to_int('1010', signed=True)} (signed)")
    print(f"-6 width=4 -> '{int_to_slv(-6, 4)}'")

    print("\n=== Full adder truth table ===")
    fa = build_full_adder_block()
    print(fa.truth_table_str())

    print("\n=== D flip-flop simulation ===")
    dff = DFlipFlop()
    D_in = [1, 0, 1, 1, 0]
    rst  = [0, 0, 0, 1, 0]
    out  = dff.simulate(D_in, rst)
    print(f"D:   {D_in}")
    print(f"RST: {rst}")
    print(f"Q:   {out}")

    print("\n=== Traffic light FSM ===")
    fsm = build_traffic_light_fsm()
    print(fsm.transition_diagram_str())
    inputs = [1, 1, 1, 0, 1, 1]
    lights = [fsm.outputs[fsm.initial]] + fsm.run(inputs)
    print(f"Inputs: {inputs}")
    print(f"Lights: {lights}")

    print("\n=== Sequence detector '101' ===")
    det = build_sequence_detector_fsm()
    bits = [1, 0, 1, 1, 0, 1, 0, 1]
    detected = det.run(bits)
    print(f"Bits:     {bits}")
    print(f"Detected: {detected}")

    print("\n=== 8-bit ALU ===")
    alu = ALU(8)
    res, flags = alu.execute(0b00001010, 0b00000011, 0b000)  # ADD 10+3
    print(f"ADD 10+3 = {res}, flags={flags}")
    res, flags = alu.execute(0b00000011, 0b00000101, 0b001)  # SUB 3-5
    print(f"SUB 3-5  = {res} (two's comp), N={flags['N']}, C={flags['C']}")

    print("\n=== RTL pipeline (4 stages) ===")
    pipe = rtl_pipeline_sympy(4)
    print(f"T_clock  = {pipe['T_clock']}")
    print(f"latency  = {pipe['latency']}")

    print("\n=== Work-energy theorem ===")
    we = work_energy_sympy()
    for k, v in we.items():
        print(f"  {k}: {v}")
