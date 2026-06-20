"""Network-defense simulator -- a pure in-memory model of DDoS dynamics and the
mitigations that counter them.

This generates NO real network traffic and contains NO attack tooling. It is a
discrete-time queueing simulation: legitimate users plus a synthetic flood
arrive at a finite-capacity server, and we measure how defensive mitigations
(per-source rate limiting, anomaly-based blocklisting) restore service to real
users during an attack. For defensive-security education and capacity planning.
Standalone.
"""

import numpy as np


def generate_traffic(T=200, n_legit=200, legit_rate=0.3,
                     attack_window=(80, 150), n_bots=60, bot_rate=40.0, seed=0):
    """Synthesize a request timeline. Returns a list of per-step dicts, each with
    arrays of source ids and a legit/attack flag.

    Legitimate users (ids 0..n_legit-1) arrive at a low Poisson rate throughout;
    during attack_window a botnet (ids >= n_legit) floods at a high per-source
    rate -- the signature DDoS asymmetry the defenses below exploit.
    """
    rng = np.random.default_rng(seed)
    steps = []
    for t in range(T):
        n_l = rng.poisson(n_legit * legit_rate)
        legit_src = rng.integers(0, n_legit, size=n_l)
        if attack_window[0] <= t < attack_window[1]:
            n_a = rng.poisson(n_bots * bot_rate)
            bot_src = n_legit + rng.integers(0, n_bots, size=n_a)
        else:
            bot_src = np.array([], dtype=int)
        src = np.concatenate([legit_src, bot_src])
        is_legit = np.concatenate([np.ones(n_l, bool), np.zeros(len(bot_src), bool)])
        steps.append({"src": src, "legit": is_legit})
    return steps


def _apply_defense(src, legit, policy, counts, rate_limit, block_thresh, blocklist):
    """Filter the requests arriving this step. Returns a boolean 'passed' mask.
    counts/blocklist are mutated to carry state across steps."""
    n = len(src)
    passed = np.ones(n, bool)
    if policy in ("rate_limit", "both"):
        seen = {}
        for i, s in enumerate(src):                 # per-source token bucket (rate_limit/step)
            seen[s] = seen.get(s, 0) + 1
            if seen[s] > rate_limit:
                passed[i] = False
    if policy in ("blocklist", "both"):
        for i, s in enumerate(src):                 # drop already-blocklisted sources
            if s in blocklist:
                passed[i] = False
        for s in src:                               # update rolling counts, blocklist abusers
            counts[s] = counts.get(s, 0) + 1
        for s, c in counts.items():
            if c > block_thresh:
                blocklist.add(s)
        for k in list(counts):                      # decay so legit users are never blocked
            counts[k] *= 0.6
    return passed


def simulate(steps, capacity=100, policy="none", rate_limit=3, block_thresh=15):
    """Run the server with a defense policy. policy in
    {'none','rate_limit','blocklist','both'}. Returns per-step arrays of
    legit_sent, legit_served, attack_served, and overall legit success rate."""
    if policy not in ("none", "rate_limit", "blocklist", "both"):
        raise ValueError("unknown policy")
    if capacity < 1:
        raise ValueError("capacity must be >= 1")
    rng = np.random.default_rng(0)
    counts, blocklist = {}, set()
    L_sent, L_served, A_served, blocked = [], [], [], []
    for st in steps:
        src, legit = st["src"], st["legit"]
        passed = _apply_defense(src, legit, policy, counts, rate_limit, block_thresh, blocklist)
        idx = np.where(passed)[0]
        # server serves up to `capacity` of the admitted requests (random among them)
        if len(idx) > capacity:
            idx = rng.choice(idx, size=capacity, replace=False)
        served_legit = int(legit[idx].sum())
        L_sent.append(int(legit.sum()))
        L_served.append(served_legit)
        A_served.append(int((~legit[idx]).sum()))
        blocked.append(len(blocklist))
    L_sent, L_served = np.array(L_sent), np.array(L_served)
    rate = L_served.sum() / max(L_sent.sum(), 1)
    return {"legit_sent": L_sent, "legit_served": L_served,
            "attack_served": np.array(A_served), "blocklist_size": np.array(blocked),
            "legit_success_rate": float(rate)}
