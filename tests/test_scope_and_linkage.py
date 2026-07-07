"""Test dgs.scope_and_linkage: the lexical scope chain (shadowing, outward
resolution, assign-to-existing), the K&R storage-class rules, static-local
persistence, and the kinetics integrator where k is an external variable and dA
is a non-leaking block local -- verified against A(t)=A0 exp(-k t). C demo of
external/static/automatic storage is gcc-guarded."""
import sys, pathlib, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import scope_and_linkage as sl

# 1. scope chain: define / lookup / shadowing / outward resolution / depth
g = sl.Scope(name="global"); g.define("x", 1); g.define("k", 0.5)
blk = sl.Scope(parent=g, name="block"); blk.define("x", 99)   # shadows global x
assert blk.lookup("x") == 99 and g.lookup("x") == 1           # inner shadows outer
assert blk.lookup("k") == 0.5                                  # inner sees outer
assert blk.resolve("k") is g and blk.resolve("x") is blk
assert g.depth() == 0 and blk.depth() == 1
try:
    blk.lookup("nope"); assert False
except NameError:
    pass

# assign updates the EXISTING binding in the nearest scope, not a new local
b2 = sl.Scope(parent=g)
b2.assign("x", 5)                    # no local x -> updates the global's x
assert g.lookup("x") == 5 and "x" not in b2.vars
b2.define("x", 99); b2.assign("x", 7)   # now a local x exists -> update that
assert b2.lookup("x") == 7 and g.lookup("x") == 5
try:
    b2.assign("undeclared", 1); assert False
except NameError:
    pass

# 2. K&R storage-class rules
assert sl.declaration_properties(None, at_file_scope=True) == \
    {"scope": "file", "linkage": "external", "storage_duration": "static"}
assert sl.declaration_properties("static", at_file_scope=True)["linkage"] == "internal"
assert sl.declaration_properties(None, at_file_scope=False) == \
    {"scope": "block", "linkage": "none", "storage_duration": "automatic"}
assert sl.declaration_properties("static", at_file_scope=False)["storage_duration"] == "static"
assert sl.declaration_properties("extern", at_file_scope=False)["linkage"] == "external"
for bad in (lambda: sl.declaration_properties("auto", at_file_scope=True),
            lambda: sl.declaration_properties("bogus")):
    try:
        bad(); assert False
    except ValueError:
        pass

# 3. static local persists across calls; a fresh one starts over
tick = sl.make_static_counter()
assert [tick()[0] for _ in range(3)] == [1, 2, 3]
assert sl.make_static_counter()()[0] == 1        # independent object
# the "automatic" scratch is recomputed, never persisted
tick2 = sl.make_static_counter()
_, scratch1 = tick2(); _, scratch2 = tick2()
assert scratch1 == 10 and scratch2 == 20         # tracks count*10, not stale

# 4. kinetics: external k, block-local dA, verified vs analytic decay
A0, k, t_end, dt = 1.0, 0.7, 5.0, 0.0005
traj, gs = sl.kinetics_with_scope(A0, k, t_end, dt)
t, A, B = traj[:, 0], traj[:, 1], traj[:, 2]
assert np.isclose(A[-1], A0 * np.exp(-k * t_end), rtol=1e-2)   # A(t)=A0 e^-kt
assert np.allclose(A + B, A0, atol=1e-9)                       # mass conservation
assert gs.lookup("k") == k                                     # external still visible
try:
    gs.lookup("dA"); assert False                              # block-local didn't leak
except NameError:
    pass
# doubling the external rate constant decays faster (shorter half-life)
traj_fast, _ = sl.kinetics_with_scope(A0, 2 * k, t_end, dt)
assert traj_fast[-1, 1] < A[-1]
# cross-check against the independent dgs.physical_chemistry model
from dgs import physical_chemistry as pc
pc_C = pc.first_order_kinetics(k, A0, np.array([t_end]))["C"][0]
assert np.isclose(A[-1], pc_C, rtol=1e-2)

# 5. C demonstration (gcc-guarded): external / static / automatic storage
if sl.gcc_available():
    with tempfile.TemporaryDirectory() as d:
        c = sl.compile_and_run_c(d)
    assert c["call1"] == 1015 and c["call2"] == 2015   # static calls: 1 then 2
    assert c["g"] == 100                               # the external variable
    assert c["static_persisted"] and c["automatic_reset"]
    print("test_scope_and_linkage: all checks passed (incl. C demo)")
else:
    print("test_scope_and_linkage: all checks passed (C demo skipped: no gcc)")

# 6. kwarg bounds
for bad in (lambda: sl.kinetics_with_scope(-1, 0.5, 1, 0.1),
            lambda: sl.kinetics_with_scope(1, 0.5, 0, 0.1),
            lambda: sl.kinetics_with_scope(1, 0.5, 1, 0)):
    try:
        bad(); assert False
    except ValueError:
        pass
