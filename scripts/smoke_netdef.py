"""Smoke-test the network-defense simulator: attack hurts, defenses help."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import network_defense_sim as nd

steps = nd.generate_traffic(T=200, attack_window=(80, 150), seed=0)

results = {}
for pol in ("none", "rate_limit", "blocklist", "both"):
    r = nd.simulate(steps, capacity=100, policy=pol)
    results[pol] = r
    print(f"  policy={pol:10}: legit success rate during whole run = "
          f"{r['legit_success_rate']*100:5.1f}%  | final blocklist {r['blocklist_size'][-1]}")

# during the attack window, no-defense should crater and blocklist should restore
aw = slice(80, 150)
def attack_rate(r):
    s, v = r["legit_sent"][aw].sum(), r["legit_served"][aw].sum()
    return v / max(s, 1)
print("\nlegit success DURING the attack window:")
for pol in ("none", "blocklist", "both"):
    print(f"  {pol:10}: {attack_rate(results[pol])*100:.1f}%")

assert attack_rate(results["none"]) < 0.5, "attack should hurt without defense"
assert attack_rate(results["both"]) > attack_rate(results["none"]), "defense should help"
assert attack_rate(results["both"]) > 0.7, "combined defense should restore service"

for bad in [lambda: nd.simulate(steps, policy="hack"),
            lambda: nd.simulate(steps, capacity=0)]:
    try:
        bad()
    except ValueError as e:
        print("err ok:", e)
print("SMOKE PASS")
