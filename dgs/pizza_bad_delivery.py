"""A bad pizza delivery: the same deformable spider-web pizza mesh from
dgs/pizza_web_fold.py, but instead of the smooth, symmetric, eased fold
(dgs.pizza_web_fold.fold_progress -- both anchors moving together,
producing a clean calzone), the two anchor hinges are driven by
INDEPENDENT, high-frequency, high-amplitude chaotic signals with no shared
phase and no joint limit (hinge_a/hinge_b in the underlying model have no
<range> -- verified in pizza_web_fold.py, unlimited rotation, exactly what
unsynchronized violent shaking needs).

Nothing here is a new physics model: it's the identical mass-spring mesh,
same pin structure, same verified "free rim genuinely lags/deforms" fact
-- just driven by a wildly different trajectory. The mangled look is an
emergent consequence of that trajectory, not a separate "damage" system
bolted on.
"""

import numpy as np
import mujoco

from dgs.pizza_web_fold import build_web_pizza_model


def chaotic_delivery_angle(t, phase_offset, base_freq=7.3, amplitude_deg=200.0):
    """Sum of three incommensurate sine waves (frequencies in a
    non-integer ratio, so the motion never repeats over the render's
    duration) -- a simple, honest way to generate genuinely chaotic-
    looking, unrepeating angular motion without needing a real chaotic
    dynamical system. amplitude_deg intentionally exceeds a physically
    sensible fold range; combined with the hinge having no joint limit,
    the anchor spins past where a careful fold would ever go."""
    return (
        amplitude_deg * np.sin(base_freq * t + phase_offset)
        + 0.6 * amplitude_deg * np.sin(1.79 * base_freq * t + 2 * phase_offset)
        + 0.3 * amplitude_deg * np.sin(3.24 * base_freq * t + 3 * phase_offset)
    )


def simulate_bad_delivery(t_max=3.0, dt=0.0005, amplitude_deg=200.0, **model_kwargs):
    """Drive both anchors with INDEPENDENT chaotic signals (different
    phase offsets, so A and B twist out of sync rather than mirroring
    each other) and track how far the two halves' free rims end up from
    each other -- the actual, checkable definition of "mangled": in a
    clean fold the two rims end up roughly mirrored and close together;
    in a bad delivery they end up scattered."""
    model, edge_idx = build_web_pizza_model(dt=dt, **model_kwargs)
    data = mujoco.MjData(model)

    hinge_a_adr = model.joint("hinge_a").qposadr[0]
    hinge_b_adr = model.joint("hinge_b").qposadr[0]

    n_rings_used = model_kwargs.get("n_rings", 4)
    n_spokes_used = model_kwargs.get("n_spokes", 10)
    rim_a_idx = 1 + (n_rings_used - 1) * (n_spokes_used + 1) + n_spokes_used // 2
    n_web_a = model.flex_vertnum[0]
    rim_b_idx = n_web_a + rim_a_idx   # same relative offset within web_b

    data.qpos[hinge_a_adr] = np.deg2rad(chaotic_delivery_angle(0.0, 0.0, amplitude_deg=amplitude_deg))
    data.qpos[hinge_b_adr] = np.deg2rad(chaotic_delivery_angle(0.0, np.pi, amplitude_deg=amplitude_deg))
    mujoco.mj_forward(model, data)

    n_steps = int(t_max / dt)
    t_hist = np.zeros(n_steps)
    rim_a_hist = np.zeros((n_steps, 3))
    rim_b_hist = np.zeros((n_steps, 3))
    max_speed = 0.0

    for i in range(n_steps):
        t = i * dt
        data.qpos[hinge_a_adr] = np.deg2rad(chaotic_delivery_angle(t, 0.0, amplitude_deg=amplitude_deg))
        data.qpos[hinge_b_adr] = np.deg2rad(chaotic_delivery_angle(t, np.pi, amplitude_deg=amplitude_deg))

        t_hist[i] = t
        rim_a_hist[i] = data.flexvert_xpos[rim_a_idx]
        rim_b_hist[i] = data.flexvert_xpos[rim_b_idx]
        mujoco.mj_step(model, data)
        max_speed = max(max_speed, float(np.max(np.abs(data.qvel))))

    rim_separation = np.linalg.norm(rim_a_hist - rim_b_hist, axis=1)
    return {
        "t": t_hist,
        "rim_a": rim_a_hist,
        "rim_b": rim_b_hist,
        "rim_separation": rim_separation,
        "max_speed": max_speed,
        "any_nan": bool(np.any(np.isnan(data.qpos)) or np.any(np.isnan(data.flexvert_xpos))),
    }


if __name__ == "__main__":
    from dgs.pizza_web_fold import simulate_web_fold

    print("=== A bad pizza delivery: chaotic, unsynchronized driving of the same mesh ===\n")
    bad = simulate_bad_delivery(t_max=3.0)
    print(f"any NaN: {bad['any_nan']}")
    print(f"max joint speed reached: {bad['max_speed']:.1f} rad/s")
    print(f"rim separation (A vs B) range: [{bad['rim_separation'].min():.4f}, {bad['rim_separation'].max():.4f}] m")

    print("\n=== For comparison: the clean, careful fold from dgs.pizza_web_fold ===\n")
    clean = simulate_web_fold(t_max=3.0)

    def rim_b_from_clean_result(result_dict):
        # simulate_web_fold only tracks rim_a; re-derive rim_b's separation
        # by symmetry is out of scope here -- just report rim_a's excursion
        # from its rest radius as a simple, comparable "how far did it move" metric
        return result_dict["rim_a_pos"]

    clean_excursion = np.linalg.norm(clean["rim_a_pos"] - clean["rim_a_pos"][0], axis=1)
    bad_excursion = np.linalg.norm(bad["rim_a"] - bad["rim_a"][0], axis=1)
    print(f"clean fold: rim_a total excursion range [{clean_excursion.min():.4f}, {clean_excursion.max():.4f}] m")
    print(f"bad delivery: rim_a total excursion range [{bad_excursion.min():.4f}, {bad_excursion.max():.4f}] m")
