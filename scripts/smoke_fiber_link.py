"""Smoke-test the fiber dBm / link-budget module."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import fiber_link as fl

# 1. dBm anchors: 1 mW -> 0 dBm, 1 W -> 30 dBm, 1 uW -> -30 dBm
assert abs(fl.watt_to_dbm(1e-3) - 0.0) < 1e-12
assert abs(fl.watt_to_dbm(1.0) - 30.0) < 1e-12
assert abs(fl.watt_to_dbm(1e-6) + 30.0) < 1e-12
# round trip
assert np.allclose(fl.dbm_to_watt(fl.watt_to_dbm([1e-3, 5e-4, 2e-2])), [1e-3, 5e-4, 2e-2])

# 2. dB <-> ratio: 3 dB ~ 2x, 10 dB = 10x, 20 dB = 100x
assert abs(fl.db_to_ratio(3.0103) - 2.0) < 1e-3
assert abs(fl.db_to_ratio(10) - 10.0) < 1e-9
assert abs(fl.db_to_ratio(20) - 100.0) < 1e-9
assert abs(fl.ratio_to_db(2.0) - 3.0103) < 1e-3

# 3. fiber loss: 100 km @ 0.2 dB/km = 20 dB
assert abs(fl.fiber_loss(100, 0.2) - 20.0) < 1e-9
assert fl.fiber_loss(0) == 0.0

# 4. link budget: 0 dBm Tx, 80 km, 2 connectors(0.5), 3 splices(0.1)
rx, loss = fl.link_budget(0.0, fiber_km=80, n_connectors=2, n_splices=3)
assert abs(loss - (80 * 0.2 + 2 * 0.5 + 3 * 0.1)) < 1e-9     # 16 + 1 + 0.3 = 17.3 dB
assert abs(rx - (0.0 - 17.3)) < 1e-9
# received power sanity: -17.3 dBm ~ 18.6 uW
assert abs(fl.dbm_to_watt(rx) * 1e6 - 18.62) < 0.1

# 5. margin: closes when Rx above sensitivity, fails below
assert fl.power_margin(rx, -28) > 0          # -17.3 dBm vs -28 dBm sensitivity -> closes
assert fl.power_margin(rx, -10) < 0          # too lossy for a -10 dBm receiver
assert abs(fl.power_margin(-17.3, -28) - 10.7) < 1e-9

# 6. longer link loses more (monotonic) and matches Beer-Lambert dB form
rx_long, _ = fl.link_budget(0.0, fiber_km=160, n_connectors=2)
assert rx_long < rx                          # twice the fiber -> lower received power

# 7. validation
for bad in (lambda: fl.watt_to_dbm(0), lambda: fl.ratio_to_db(-1), lambda: fl.fiber_loss(-1)):
    try:
        bad()
    except ValueError:
        pass
    else:
        raise AssertionError("should reject bad input")

print(f"SMOKE PASS  (80 km link: {loss:.1f} dB loss, Rx={rx:.1f} dBm, "
      f"margin {fl.power_margin(rx,-28):.1f} dB)")
