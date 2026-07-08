"""Test dgs.cpu_scheduling: FCFS, SJF, round-robin, and priority against hand-worked
values -- Gantt order, per-process and average waiting/turnaround/response, the SJF
<= FCFS optimality theorem, round-robin's best response time, and idle handling when a
process arrives after the CPU goes free."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs import cpu_scheduling as cs

procs = [{"pid": "P1", "arrival": 0, "burst": 5, "priority": 2},
         {"pid": "P2", "arrival": 0, "burst": 3, "priority": 1},
         {"pid": "P3", "arrival": 0, "burst": 8, "priority": 3}]

# 1. FCFS: arrival order, convoy effect
f = cs.fcfs(procs)
assert f["gantt"] == [("P1", 0, 5), ("P2", 5, 8), ("P3", 8, 16)]
assert f["metrics"]["P1"]["waiting"] == 0 and f["metrics"]["P2"]["waiting"] == 5
assert f["metrics"]["P3"]["waiting"] == 8
assert math.isclose(f["avg_waiting"], 13/3) and math.isclose(f["avg_turnaround"], 29/3)

# 2. SJF: shortest first, and provably <= FCFS average waiting
s = cs.sjf(procs)
assert s["gantt"] == [("P2", 0, 3), ("P1", 3, 8), ("P3", 8, 16)]
assert math.isclose(s["avg_waiting"], 11/3)
assert s["avg_waiting"] <= f["avg_waiting"]                 # SJF optimality

# 3. round robin q=2 matches the hand-worked schedule and metrics
r = cs.round_robin(procs, 2)
assert r["metrics"]["P1"]["completion"] == 12
assert r["metrics"]["P2"]["completion"] == 9
assert r["metrics"]["P3"]["completion"] == 16
assert math.isclose(r["avg_waiting"], 7.0)
assert math.isclose(r["avg_turnaround"], 37/3)
# round robin gives the BEST average response time of the three
assert r["avg_response"] < f["avg_response"] and r["avg_response"] < s["avg_response"]
assert math.isclose(r["avg_response"], 2.0)                 # P1,P2,P3 first-run at 0,2,4

# 4. priority (lower value = higher): P2(1), P1(2), P3(3)
p = cs.priority_schedule(procs)
assert [seg[0] for seg in p["gantt"]] == ["P2", "P1", "P3"]

# 5. every metric obeys its identity: waiting = turnaround - burst, TA = completion - arrival
for sched in (f, s, r, p):
    for proc in procs:
        m = sched["metrics"][proc["pid"]]
        assert m["waiting"] == m["turnaround"] - proc["burst"]
        assert m["turnaround"] == m["completion"] - proc["arrival"]

# 6. idle handling: a process arriving after the CPU is free starts at its arrival
gap = [{"pid": "A", "arrival": 0, "burst": 2}, {"pid": "B", "arrival": 5, "burst": 2}]
fg = cs.fcfs(gap)
assert fg["gantt"] == [("A", 0, 2), ("B", 5, 7)]           # CPU idle from 2..5
assert fg["metrics"]["B"]["waiting"] == 0                  # B never actually waited

# 7. kwarg bounds
for bad in (lambda: cs.fcfs([]),
            lambda: cs.fcfs([{"pid": "X", "arrival": 0, "burst": 0}]),
            lambda: cs.fcfs([{"pid": "X", "arrival": 0, "burst": 1},
                             {"pid": "X", "arrival": 0, "burst": 1}]),   # dup pid
            lambda: cs.round_robin(procs, 0),
            lambda: cs.priority_schedule([{"pid": "X", "arrival": 0, "burst": 1}])):  # no priority
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_cpu_scheduling: all checks passed")
