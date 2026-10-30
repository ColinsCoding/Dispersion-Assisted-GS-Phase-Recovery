"""
Queueing theory for the M/M/c queue: exact answers to check event simulations against.

In plain words
--------------
Jobs arrive at random (Poisson, rate lambda per hour), each takes a random time to serve
(exponential, rate mu per server per hour), and c servers work first-come first-served.
Example: collapse reports in a disaster (lambda) handled by c rescue teams (mu each).

* offered load a = lambda / mu: how many servers would be busy on average
* utilization rho = a / c, which must be < 1 or the line grows forever
* Erlang C: the probability an arriving job has to wait,
      C(c, a) = [a^c / c! * c/(c - a)] / [sum_{k<c} a^k/k! + a^c / c! * c/(c - a)]
* mean wait in line   Wq = C(c, a) / (c mu - lambda)
* mean time in system W  = Wq + 1/mu
* Little's law        L  = lambda W,  Lq = lambda Wq  (true for any stable queue)

``simulate_mmc`` is a small NumPy reference simulator (Kiefer-Wolfowitz recursion), so the
notebook still runs without a C++ or CUDA compiler.

Sources (IEEE style)
--------------------
[1] D. Gross, J. F. Shortle, J. M. Thompson, and C. M. Harris, Fundamentals of Queueing
    Theory, 4th ed. Wiley, 2008, secs. 2.2-2.3 (M/M/1, M/M/c, Erlang C, Little's law).
[2] A. M. Law, Simulation Modeling and Analysis, 5th ed. McGraw-Hill, 2015, ch. 1
    (discrete-event simulation, event lists) and ch. 9 (replications, confidence intervals).
[3] J. Kiefer and J. Wolfowitz, "On the theory of queues with many servers," Trans. Amer.
    Math. Soc., vol. 78, pp. 1-18, 1955.
"""
import math

import numpy as np


def _check(lam, mu, c):
    if not lam > 0 or not mu > 0:
        raise ValueError(f"need lambda > 0 and mu > 0, got lambda={lam}, mu={mu}")
    if int(c) != c or c < 1:
        raise ValueError(f"number of servers c must be an integer >= 1, got {c}")
    if lam >= c * mu:
        raise ValueError(f"unstable: lambda={lam} >= c*mu={c * mu}; the line grows without bound")
    return int(c)


def erlang_c(lam, mu, c):
    """Probability an arriving job waits (all c servers busy), M/M/c steady state."""
    c = _check(lam, mu, c)
    a = lam / mu
    # sum a^k/k! in a numerically safe way
    term, s = 1.0, 1.0
    for k in range(1, c):
        term *= a / k
        s += term
    top = term * a / c * c / (c - a)          # a^c / c! * c/(c - a)
    return top / (s + top)


def mmc_metrics(lam, mu, c):
    """Steady-state M/M/c answers: rho, P_wait, Wq, W, Lq, L (time units of 1/lambda)."""
    c = _check(lam, mu, c)
    pw = erlang_c(lam, mu, c)
    wq = pw / (c * mu - lam)
    w = wq + 1 / mu
    return {"rho": lam / (c * mu), "P_wait": pw, "Wq": wq, "W": w, "Lq": lam * wq, "L": lam * w}


def min_servers(lam, mu, max_wq):
    """Smallest c with mean wait in line Wq <= max_wq."""
    if not max_wq > 0:
        raise ValueError(f"max_wq must be > 0, got {max_wq}")
    c = math.floor(lam / mu) + 1
    while mmc_metrics(lam, mu, c)["Wq"] > max_wq:
        c += 1
    return c


def simulate_mmc(lam, mu, c, n, warmup=0, seed=0):
    """FIFO M/M/c by the Kiefer-Wolfowitz recursion. Returns (mean wait, P(wait > 0))."""
    c = _check(lam, mu, c)
    if int(n) != n or n < 1 or warmup < 0:
        raise ValueError("need integer n >= 1 and warmup >= 0")
    rng = np.random.default_rng(seed)
    total = int(warmup + n)
    arrivals = np.cumsum(rng.exponential(1 / lam, total))
    services = rng.exponential(1 / mu, total)
    free_at = np.zeros(c)
    waits = np.empty(total)
    for i in range(total):
        k = int(np.argmin(free_at))
        start = max(arrivals[i], free_at[k])
        waits[i] = start - arrivals[i]
        free_at[k] = start + services[i]
    w = waits[int(warmup):]
    return float(w.mean()), float(np.mean(w > 0))
