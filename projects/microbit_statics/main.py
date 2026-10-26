"""BBC micro:bit (MicroPython) -- guyed sensor mast equilibrium + measured cost per operation.

Flash with the micro:bit Python editor (python.microbit.org) or `uflash main.py`, open the
serial console (115200 baud), press button A to run.

What it does on the device:
  1. Solves the guyed mast (weight + storm wind + ice) with the SAME pure-Python core as
     dgs/rigid_body_core.py (the block between the BEGIN/END markers is copied verbatim; a repo
     test checks they match).
  2. Times N solves for several N with time.ticks_us, fits time = overhead + slope*N by least
     squares (pure Python), and prints the cost per solve and per counted arithmetic operation.
  3. Scrolls the cable tension and the cost per operation on the LED display.
Nothing here has been run on hardware by the author of this file; the numbers are yours to measure.
"""
try:
    from microbit import display, button_a, sleep
    ON_DEVICE = True
except ImportError:          # lets the same file run on a PC for checking
    ON_DEVICE = False
import math
try:
    from utime import ticks_us, ticks_diff          # micro:bit MicroPython
except ImportError:
    try:
        from time import ticks_us, ticks_diff       # newer MicroPython ports
    except ImportError:
        import time as _t                           # CPython fallback for checking on a PC
        def ticks_us():
            return int(_t.perf_counter() * 1e6)
        def ticks_diff(a, b):
            return a - b

# --- BEGIN SHARED CORE (copied verbatim into projects/microbit_statics/main.py) ---
OPS = [0]  # running count of + - * / performed by solve3 and moment_rows


def _finite(v):
    return v == v and v not in (float("inf"), float("-inf"))


def solve3(A, b):
    """Solve the 3x3 system A x = b by Gaussian elimination with partial pivoting.
    A is a list of 3 lists of 3 floats, b a list of 3 floats. Raises ValueError if singular."""
    M = [[float(A[i][0]), float(A[i][1]), float(A[i][2]), float(b[i])] for i in range(3)]
    for i in range(3):
        for j in range(4):
            if not _finite(M[i][j]):
                raise ValueError("non-finite input")
    ops = 0
    for col in range(3):
        piv = col
        for r in range(col + 1, 3):
            if abs(M[r][col]) > abs(M[piv][col]):
                piv = r
        if abs(M[piv][col]) < 1e-12:
            raise ValueError("singular: supports form a mechanism")
        M[col], M[piv] = M[piv], M[col]
        for r in range(col + 1, 3):
            f = M[r][col] / M[col][col]
            ops += 1
            for c in range(col, 4):
                M[r][c] -= f * M[col][c]
                ops += 2
    x = [0.0, 0.0, 0.0]
    for i in (2, 1, 0):
        s = M[i][3]
        for j in range(i + 1, 3):
            s -= M[i][j] * x[j]
            ops += 2
        x[i] = s / M[i][i]
        ops += 1
    OPS[0] += ops
    return x


def moment_rows(loads, reactions):
    """Build A, b for sum Fx = sum Fy = sum M_origin = 0.
    loads: list of (Fx, Fy, x, y) known forces. reactions: list of 3 (ux, uy, x, y) unit
    directions of unknown-magnitude forces. Returns (A, b)."""
    if len(reactions) != 3:
        raise ValueError("need exactly 3 reaction components (statically determinate)")
    A = [[0.0] * 3 for _ in range(3)]
    for j in range(3):
        ux, uy, x, y = reactions[j]
        A[0][j] = ux
        A[1][j] = uy
        A[2][j] = x * uy - y * ux
    fx = fy = m = 0.0
    ops = 3
    for (Fx, Fy, x, y) in loads:
        fx += Fx
        fy += Fy
        m += x * Fy - y * Fx
        ops += 6
    OPS[0] += ops
    return A, [-fx, -fy, -m]


def guyed_mast_reactions(H, d, loads):
    """Mast pinned at A = (0, 0), height H, guy cable from the top (0, H) to an anchor at (-d, 0).
    Returns [Ax, Ay, T] (pin components and cable tension; T > 0 means the cable pulls)."""
    if H <= 0 or d <= 0:
        raise ValueError("H and d must be positive")
    L = (d * d + H * H) ** 0.5
    reactions = [(1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (-d / L, -H / L, 0.0, H)]
    A, b = moment_rows(loads, reactions)
    return solve3(A, b)
# --- END SHARED CORE ---

# --- mast loads (pure-Python mirror of dgs.rigid_body_harsh.mast_loads) ---
RHO_AIR, RHO_ICE, G = 1.225, 900.0, 9.81
H, D_ANCHOR, POLE_D = 3.0, 2.0, 0.06
POLE_MASS, HEAD_MASS, HEAD_AREA, HEAD_SURF, CD_POLE, CD_HEAD = 12.0, 6.0, 0.04, 0.25, 1.2, 1.1


def mast_loads(v, t_ice):
    ice_pole = RHO_ICE * math.pi * ((POLE_D / 2 + t_ice) ** 2 - (POLE_D / 2) ** 2) * H
    w_pole = (POLE_MASS + ice_pole) * G
    w_head = (HEAD_MASS + RHO_ICE * HEAD_SURF * t_ice) * G
    side = math.sqrt(HEAD_AREA)
    f_head = 0.5 * RHO_AIR * CD_HEAD * (side + 2 * t_ice) ** 2 * v * v
    f_pole = 0.5 * RHO_AIR * CD_POLE * (POLE_D + 2 * t_ice) * H * v * v
    return [(0.0, -w_pole, 0.0, H / 2), (0.0, -w_head, 0.0, H),
            (f_pole, 0.0, 0.0, H / 2), (f_head, 0.0, 0.0, H)]


def fit_line(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return my - slope * mx, slope


def run(Ns=(20, 40, 80, 160)):
    loads = mast_loads(35.0, 0.025)
    Ax, Ay, T = guyed_mast_reactions(H, D_ANCHOR, loads)
    print("storm 35 m/s + 25 mm ice: Ax=%.1f N  Ay=%.1f N  T=%.1f N" % (Ax, Ay, T))
    OPS[0] = 0
    guyed_mast_reactions(H, D_ANCHOR, loads)
    nops = OPS[0]
    ts = []
    for N in Ns:
        t0 = ticks_us()
        for _ in range(N):
            guyed_mast_reactions(H, D_ANCHOR, loads)
        ts.append(ticks_diff(ticks_us(), t0))
        print("N=%d  time_us=%d" % (N, ts[-1]))
    over, slope = fit_line(list(Ns), ts)
    print("overhead_us=%.1f  us_per_solve=%.2f  ops_per_solve=%d  us_per_op=%.3f" % (over, slope, nops, slope / nops))
    return T, slope / nops


if ON_DEVICE:
    display.scroll("A: run")
    while True:
        if button_a.was_pressed():
            T, us_op = run()
            display.scroll("T=%dN %.1fus/op" % (int(T), us_op))
        sleep(100)
elif __name__ == "__main__":
    run(Ns=(2000, 4000, 8000, 16000))
