"""CPU scheduling: how an operating system decides which process runs next.

One CPU, many processes that each need a burst of time -- the SCHEDULER picks the order,
and the order dramatically changes how long everyone waits. The classic policies:

  FCFS  (first-come, first-served): run them in arrival order. Simple and fair, but a
        long job at the front makes everyone behind it wait -- the "convoy effect".
  SJF   (shortest job first): always run the shortest available burst next. PROVABLY
        minimizes average waiting time (non-preemptive) -- but needs to know burst
        lengths, and can starve long jobs.
  RR    (round robin): give each process a fixed time QUANTUM, then rotate. Great
        responsiveness (no one waits long for a turn) at the cost of context switches,
        so average waiting is usually worse than SJF.
  PRIORITY: run the highest-priority ready process (lower number = higher priority).

For each we report the metrics that matter:
    completion  -- when the process finishes,
    turnaround  -- completion - arrival (total time in the system),
    waiting     -- turnaround - burst (time spent NOT running),
    response    -- first-run time - arrival (how fast it first gets the CPU).

Verified against hand-worked examples, including the theorem that SJF's average waiting
time is <= FCFS's. A process is a dict {pid, arrival, burst, priority}. Pure Python;
py-3.13.
"""

from collections import deque


def _metrics(gantt, processes):
    """Compute per-process and average metrics from a Gantt schedule (a list of
    (pid, start, end) execution segments)."""
    info = {p["pid"]: p for p in processes}
    per = {}
    for pid, p in info.items():
        segs = [(s, e) for (q, s, e) in gantt if q == pid]
        first_start = min(s for s, _ in segs)
        completion = max(e for _, e in segs)
        turnaround = completion - p["arrival"]
        per[pid] = {
            "completion": completion,
            "turnaround": turnaround,
            "waiting": turnaround - p["burst"],
            "response": first_start - p["arrival"],
        }
    n = len(per)
    return {
        "gantt": gantt,
        "metrics": per,
        "avg_waiting": sum(m["waiting"] for m in per.values()) / n,
        "avg_turnaround": sum(m["turnaround"] for m in per.values()) / n,
        "avg_response": sum(m["response"] for m in per.values()) / n,
    }


def _validate(processes):
    if not processes:
        raise ValueError("need at least one process")
    for p in processes:
        if p["burst"] <= 0 or p["arrival"] < 0:
            raise ValueError("burst must be > 0 and arrival >= 0")
    if len({p["pid"] for p in processes}) != len(processes):
        raise ValueError("process ids must be unique")


def fcfs(processes):
    """First-come, first-served: run in arrival order (ties broken by pid)."""
    _validate(processes)
    order = sorted(processes, key=lambda p: (p["arrival"], p["pid"]))
    t, gantt = 0, []
    for p in order:
        start = max(t, p["arrival"])
        gantt.append((p["pid"], start, start + p["burst"]))
        t = start + p["burst"]
    return _metrics(gantt, processes)


def _greedy_nonpreemptive(processes, key):
    """Shared engine for the non-preemptive 'pick the best ready process' policies
    (SJF and priority). `key` selects the next process among those that have arrived."""
    _validate(processes)
    remaining = list(processes)
    t, gantt = 0, []
    while remaining:
        ready = [p for p in remaining if p["arrival"] <= t]
        if not ready:
            t = min(p["arrival"] for p in remaining)      # idle until next arrival
            continue
        p = min(ready, key=key)
        gantt.append((p["pid"], t, t + p["burst"]))
        t += p["burst"]
        remaining.remove(p)
    return _metrics(gantt, processes)


def sjf(processes):
    """Shortest job first (non-preemptive): minimizes average waiting time."""
    return _greedy_nonpreemptive(processes, key=lambda p: (p["burst"], p["arrival"], p["pid"]))


def priority_schedule(processes):
    """Priority (non-preemptive), lower 'priority' value = higher priority."""
    for p in processes:
        if "priority" not in p:
            raise ValueError("each process needs a 'priority' field")
    return _greedy_nonpreemptive(
        processes, key=lambda p: (p["priority"], p["arrival"], p["pid"]))


def round_robin(processes, quantum):
    """Round robin: each ready process gets up to `quantum` time, then rotates to the
    back of the queue. Best response time; more context switches."""
    _validate(processes)
    if quantum <= 0:
        raise ValueError("quantum must be positive")
    order = sorted(processes, key=lambda p: (p["arrival"], p["pid"]))
    rem = {p["pid"]: p["burst"] for p in processes}
    t, idx, gantt = 0, 0, []
    ready = deque()
    completed = 0
    n = len(order)
    while completed < n:
        while idx < n and order[idx]["arrival"] <= t:
            ready.append(order[idx]); idx += 1
        if not ready:
            t = order[idx]["arrival"]; continue           # jump to next arrival
        p = ready.popleft()
        run = min(quantum, rem[p["pid"]])
        gantt.append((p["pid"], t, t + run))
        t += run
        rem[p["pid"]] -= run
        while idx < n and order[idx]["arrival"] <= t:      # arrivals during this slice
            ready.append(order[idx]); idx += 1
        if rem[p["pid"]] > 0:
            ready.append(p)
        else:
            completed += 1
    return _metrics(gantt, processes)


if __name__ == "__main__":
    procs = [{"pid": "P1", "arrival": 0, "burst": 5, "priority": 2},
             {"pid": "P2", "arrival": 0, "burst": 3, "priority": 1},
             {"pid": "P3", "arrival": 0, "burst": 8, "priority": 3}]

    for name, res in [("FCFS", fcfs(procs)), ("SJF", sjf(procs)),
                      ("RR q=2", round_robin(procs, 2)),
                      ("Priority", priority_schedule(procs))]:
        order = " ".join(f"{pid}[{s},{e}]" for pid, s, e in res["gantt"])
        print(f"{name:9s}: {order}")
        print(f"           avg waiting = {res['avg_waiting']:.2f}, "
              f"turnaround = {res['avg_turnaround']:.2f}, "
              f"response = {res['avg_response']:.2f}")

    print(f"\nSJF avg waiting ({sjf(procs)['avg_waiting']:.2f}) "
          f"<= FCFS ({fcfs(procs)['avg_waiting']:.2f})  -- SJF is optimal")
