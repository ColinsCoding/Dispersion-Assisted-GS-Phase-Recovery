"""Plotly 4D (3 spatial axes + time/iteration) trajectory viewers.

make_4d_viewer() is the original generic molecular-dynamics animator
(user-provided this session: a trajectory object with .elements/.n_frames,
and a (n_frames, n_atoms, 3) positions array) -- kept as-is, unmodified,
since it's already a correct, reusable Plotly Scatter3d animation pattern
and not specific to this repo's physics.

make_gs_convergence_viewer() adapts the SAME animation scaffold (play/pause
buttons, frame slider, fixed axis limits, marker sizing) to this repo's own
domain: instead of atoms moving over MD timesteps, it animates the complex
field estimate E(t) at each GS phase-retrieval ITERATION (dgs.gs_core's
retrieve_phase_with_history) -- x=time-sample index, y=Re(E), z=Im(E),
animation frame=GS iteration. Marker size/color track |E(t)| (instantaneous
amplitude) in place of the original's atomic-species size mapping, since
amplitude is the physically meaningful per-point quantity here. An optional
ground-truth trajectory overlay shows what the algorithm is converging
toward.
"""
import numpy as np
import plotly.graph_objects as go


def make_4d_viewer(traj, positions):
    """Original generic molecular-trajectory viewer, unmodified."""
    sizes = [18 if e == "H" else 34 for e in traj.elements]
    frames = []
    for k in range(traj.n_frames):
        frames.append(go.Frame(data=[go.Scatter3d(
            x=positions[k, :, 0], y=positions[k, :, 1], z=positions[k, :, 2],
            mode="markers+text", text=traj.elements, textposition="top center",
            marker=dict(size=sizes))], name=str(k)))
    p = positions.reshape(-1, 3)
    margin = 0.4
    limits = [[p[:, d].min() - margin, p[:, d].max() + margin] for d in range(3)]
    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        title="Molecular trajectory: x, y, z + time",
        scene=dict(xaxis=dict(range=limits[0], title="x (A)"), yaxis=dict(range=limits[1], title="y (A)"),
                   zaxis=dict(range=limits[2], title="z (A)"), aspectmode="cube"),
        updatemenus=[dict(type="buttons", buttons=[
            dict(label="Play", method="animate", args=[None, {"frame": {"duration": 40, "redraw": True}, "fromcurrent": True}]),
            dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])])],
        sliders=[dict(steps=[dict(method="animate", args=[[str(k)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}], label=str(k)) for k in range(traj.n_frames)],
                     currentvalue={"prefix": "frame: "})])
    return fig


def make_gs_convergence_viewer(E_history, t=None, phi_true=None, title="GS phase retrieval: convergence trajectory"):
    """Animated 3D view of a GS phase-retrieval run's convergence:
    x=time-sample index, y=Re(E(t)), z=Im(E(t)), animated over GS iteration
    (E_history's leading axis). Marker size and color track instantaneous
    |E(t)| -- the amplitude at that sample, that iteration -- replacing the
    original viewer's fixed per-atom size (which never needed to vary
    because atomic species doesn't change frame to frame; amplitude here
    does, every iteration, so it's recomputed per frame).

    Parameters
    ----------
    E_history : complex array, shape (n_frames, N) -- e.g. from
                dgs.gs_core.retrieve_phase_with_history's third return value
    t         : optional float array, shape (N,) -- time-sample axis;
                defaults to sample index 0..N-1
    phi_true  : optional float array, shape (N,) -- if given, overlays the
                true unit-amplitude trajectory exp(i*phi_true) as a static
                black reference curve, so convergence can be judged visually
                against the actual target, not just the error metric
    title     : str

    Returns
    -------
    plotly.graph_objects.Figure
    """
    E_history = np.asarray(E_history, dtype=complex)
    if E_history.ndim != 2:
        raise ValueError("E_history must be 2D: (n_frames, N)")
    n_frames, N = E_history.shape
    if n_frames < 1:
        raise ValueError("E_history must have at least one frame")

    if t is None:
        t = np.arange(N)
    else:
        t = np.asarray(t, dtype=float)
        if t.shape[0] != N:
            raise ValueError("t must have the same length as E_history's sample axis")

    amp_all = np.abs(E_history)
    amp_max = float(amp_all.max()) if amp_all.max() > 0 else 1.0

    def _sizes_and_colors(E_k):
        amp_k = np.abs(E_k)
        sizes_k = 4 + 12 * (amp_k / amp_max)
        return sizes_k, amp_k

    static_data = []
    if phi_true is not None:
        phi_true = np.asarray(phi_true, dtype=float)
        if phi_true.shape[0] != N:
            raise ValueError("phi_true must have the same length as E_history's sample axis")
        E_true = np.exp(1j * phi_true)
        static_data.append(go.Scatter3d(
            x=t, y=np.real(E_true), z=np.imag(E_true),
            mode="lines", line=dict(color="black", width=3), name="true (unit amplitude)"))

    frames = []
    for k in range(n_frames):
        E_k = E_history[k]
        sizes_k, amp_k = _sizes_and_colors(E_k)
        frame_data = [go.Scatter3d(
            x=t, y=np.real(E_k), z=np.imag(E_k),
            mode="markers", name="E(t) estimate",
            marker=dict(size=sizes_k, color=amp_k, colorscale="Viridis", cmin=0, cmax=amp_max,
                       colorbar=dict(title="|E(t)|") if k == 0 else None))]
        frames.append(go.Frame(data=frame_data + static_data, name=str(k)))

    y_all = np.real(E_history)
    z_all = np.imag(E_history)
    if phi_true is not None:
        y_all = np.concatenate([y_all.ravel(), np.real(np.exp(1j * phi_true))])
        z_all = np.concatenate([z_all.ravel(), np.imag(np.exp(1j * phi_true))])
    margin = 0.2
    y_range = [float(y_all.min()) - margin, float(y_all.max()) + margin]
    z_range = [float(z_all.min()) - margin, float(z_all.max()) + margin]

    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(range=[float(t.min()), float(t.max())], title="time sample"),
            yaxis=dict(range=y_range, title="Re(E)"),
            zaxis=dict(range=z_range, title="Im(E)"),
            aspectmode="cube"),
        updatemenus=[dict(type="buttons", buttons=[
            dict(label="Play", method="animate", args=[None, {"frame": {"duration": 80, "redraw": True}, "fromcurrent": True}]),
            dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])])],
        sliders=[dict(steps=[dict(method="animate", args=[[str(k)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}], label=str(k)) for k in range(n_frames)],
                     currentvalue={"prefix": "GS iteration: "})])
    return fig


if __name__ == "__main__":
    from dgs.gs_core import make_measurements, retrieve_phase_with_history

    m = make_measurements('QPSK', n_symbols=32, sps=8, D1=-5000, D2=-5750, snr_db=30, rng_seed=1)
    phi, errors, E_history = retrieve_phase_with_history(
        m['I1'], m['I2'], m['D1'], m['D2'], n_iter=50, unit_amplitude=True)

    fig = make_gs_convergence_viewer(E_history, t=m['t'], phi_true=m['phi_true'])
    print(f"Built convergence viewer: {len(fig.frames)} frames, "
          f"{len(fig.frames[0].data[0].x)} samples per frame")
    print(f"Final phase correlation: {np.corrcoef(phi, m['phi_true'])[0,1]:.4f}")
    print("Call fig.show() in a notebook/browser to view the animation.")
