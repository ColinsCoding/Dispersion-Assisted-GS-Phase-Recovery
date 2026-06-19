"""Optical power in dB/dBm and the fiber link budget -- a bench skill.

In a fiber lab nobody quotes power in watts; they quote **dBm** (dB relative to
1 mW) and losses in **dB**, because then a whole link is just addition:

    P_rx(dBm) = P_tx(dBm) - (fiber loss + connector loss + splice loss + ...).

This module does the conversions (W <-> dBm, dB <-> ratio) and a link budget
with the usual loss terms, plus the power **margin** above a receiver's
sensitivity. Fiber attenuation is the Beer-Lambert decay of
griffiths.electrodynamics in dB/km form (~0.2 dB/km for SMF at 1550 nm).
NumPy-friendly. Civilian fiber metrology / education.
"""

import numpy as np


# ── power <-> dBm, and dB <-> linear ratio ──────────────────────────
def watt_to_dbm(power_w):
    """Optical power in dBm: 10 log10(P / 1 mW).  1 mW -> 0 dBm, 1 W -> 30 dBm."""
    p = np.asarray(power_w, dtype=float)
    if np.any(p <= 0):
        raise ValueError("power must be > 0 W")
    return 10.0 * np.log10(p / 1e-3)


def dbm_to_watt(dbm):
    """Inverse of watt_to_dbm: P = 1 mW * 10^(dBm/10)."""
    return 1e-3 * 10.0 ** (np.asarray(dbm, dtype=float) / 10.0)


def db_to_ratio(db):
    """Power ratio from dB: 10^(dB/10).  3 dB ~ 2x, 10 dB = 10x, 20 dB = 100x."""
    return 10.0 ** (np.asarray(db, dtype=float) / 10.0)


def ratio_to_db(ratio):
    """dB from a power ratio: 10 log10(ratio)."""
    r = np.asarray(ratio, dtype=float)
    if np.any(r <= 0):
        raise ValueError("ratio must be > 0")
    return 10.0 * np.log10(r)


# ── the link budget ─────────────────────────────────────────────────
def fiber_loss(length_km, atten_db_per_km=0.2):
    """Fiber attenuation in dB: alpha[dB/km] * L[km] (0.2 dB/km typical @1550 nm)."""
    if length_km < 0 or atten_db_per_km < 0:
        raise ValueError("length and attenuation must be >= 0")
    return atten_db_per_km * length_km


def link_budget(tx_dbm, fiber_km=0.0, atten_db_per_km=0.2,
                connector_db=0.5, n_connectors=2, splice_db=0.1, n_splices=0,
                extra_db=0.0):
    """Received power and total loss for a fiber link.

    Returns (rx_dbm, total_loss_db). Loss = fiber + connectors + splices + extra.
    """
    loss = (fiber_loss(fiber_km, atten_db_per_km)
            + connector_db * n_connectors
            + splice_db * n_splices
            + extra_db)
    return tx_dbm - loss, loss


def power_margin(rx_dbm, sensitivity_dbm):
    """Link margin in dB: how far the received power sits above the receiver's
    sensitivity. Positive = the link closes; negative = too lossy."""
    return rx_dbm - sensitivity_dbm


if __name__ == "__main__":
    tx = watt_to_dbm(1e-3)                     # 1 mW source -> 0 dBm
    rx, loss = link_budget(tx, fiber_km=80, n_connectors=2, n_splices=3)
    print(f"Tx = {tx:.1f} dBm")
    print(f"80 km link: loss = {loss:.1f} dB  ->  Rx = {rx:.1f} dBm "
          f"({dbm_to_watt(rx)*1e6:.2f} uW)")
    print(f"margin vs -28 dBm receiver: {power_margin(rx, -28):.1f} dB "
          f"({'closes' if power_margin(rx, -28) > 0 else 'fails'})")
