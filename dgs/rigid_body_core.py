"""Pure-Python rigid-body equilibrium core (no NumPy): runs on CPython and on BBC micro:bit MicroPython.

In plain words: a rigid body in a plane is in equilibrium when
        sum F_x = 0,   sum F_y = 0,   sum M = 0.
With three unknown reaction components (a pin gives two, a cable or roller one) those three
equations form a 3x3 linear system A r = b. This file solves it with Gaussian elimination and
partial pivoting, written with only lists, floats and arithmetic, so the identical code can be
copied onto a micro:bit (projects/microbit_statics/main.py; a test keeps the two in sync).

It also COUNTS its own arithmetic operations, so a measured time per solve can be turned into
a cost per operation. Kept MicroPython-friendly on purpose: no f-strings, no numpy, no
dataclasses, % formatting only.
"""

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
