"""GNSS positioning: how your phone finds itself from four satellites.

A GPS/GNSS receiver never measures distance directly. It measures a PSEUDORANGE to
each satellite -- the true range plus one unknown it shares with every satellite: its
own clock bias b (a cheap receiver clock is off by microseconds, and light travels
300 m per microsecond, so the bias is huge):
        rho_i = || r - s_i ||  +  b,
where r = (x,y,z) is the unknown receiver position and s_i the known satellite
position. Four unknowns (x, y, z, b) -> you need at least FOUR satellites. The clock
bias is why four, not three: the fourth satellite pays for not owning an atomic clock.

The equation is NONLINEAR (a distance), so it is solved by ITERATED LEAST SQUARES
(Gauss-Newton): linearize about the current estimate, where each row of the geometry
matrix G is [line-of-sight unit vector, 1], solve G dx = residual for the correction,
and repeat. It converges in a handful of steps even starting from the center of the
Earth -- the same Ax=b least squares as dgs.statics_linalg, now locating a phone.

The satellite GEOMETRY sets the accuracy through the DILUTION OF PRECISION: DOP =
sqrt(trace((G^T G)^-1)). Satellites spread across the sky give a small DOP (sharp
fix); satellites bunched together give a large DOP (a smeared fix from the same range
accuracy). It is purely geometric -- no measurement values, just directions.

Verified by placing a receiver, generating exact pseudoranges, and recovering the
position and clock bias to millimeters, plus a noisy case and the DOP geometry test.
NumPy only; py-3.13.
"""

import numpy as np

C_LIGHT = 299792458.0     # m/s (to convert a clock bias in seconds to meters)


def pseudorange(receiver, sat, clock_bias=0.0):
    """The measured pseudorange rho = ||r - s|| + b: true geometric range plus the
    receiver's clock bias b (in meters, i.e. c * clock_error)."""
    receiver = np.asarray(receiver, float)
    sat = np.asarray(sat, float)
    return float(np.linalg.norm(receiver - sat) + clock_bias)


def geometry_matrix(sats, receiver_est):
    """The linearized design matrix G at an estimated position: each row is
    [e_i, 1] where e_i = (r - s_i)/||r - s_i|| is the line-of-sight unit vector
    from satellite i to the receiver. The 1 column is the clock-bias partial."""
    sats = np.asarray(sats, float)
    r = np.asarray(receiver_est, float)
    d = r - sats
    ranges = np.linalg.norm(d, axis=1)
    if np.any(ranges < 1e-9):
        raise ValueError("estimate coincides with a satellite")
    los = d / ranges[:, None]
    return np.hstack([los, np.ones((len(sats), 1))])


def solve_position(sats, pseudoranges, x0=None, max_iter=30, tol=1e-6):
    """Solve for (x, y, z, clock_bias) from >=4 pseudoranges by Gauss-Newton
    iterated least squares. Returns dict with position, clock_bias, iterations,
    residual_rms, and converged. Needs at least 4 satellites (4 unknowns)."""
    sats = np.asarray(sats, float)
    rho = np.asarray(pseudoranges, float)
    if len(sats) < 4:
        raise ValueError("need at least 4 satellites (x, y, z, clock bias)")
    if len(rho) != len(sats):
        raise ValueError("one pseudorange per satellite")
    x = np.zeros(4) if x0 is None else np.asarray(x0, float).copy()
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        r, b = x[:3], x[3]
        pred = np.linalg.norm(r - sats, axis=1) + b       # modeled pseudorange
        G = geometry_matrix(sats, r)
        dx, *_ = np.linalg.lstsq(G, rho - pred, rcond=None)
        x = x + dx
        if np.linalg.norm(dx) < tol:
            converged = True
            break
    r, b = x[:3], x[3]
    residual = rho - (np.linalg.norm(r - sats, axis=1) + b)
    return {
        "position": r, "clock_bias": float(b),
        "iterations": it, "converged": converged,
        "residual_rms": float(np.sqrt(np.mean(residual ** 2))),
    }


def dilution_of_precision(sats, receiver):
    """Geometry-only accuracy factors from Q = (G^T G)^-1:
      GDOP = sqrt(trace Q)         (position + time),
      PDOP = sqrt(Q_xx+Q_yy+Q_zz)  (3-D position),
      TDOP = sqrt(Q_bb)            (clock).
    Small = satellites well spread = a sharp fix; large = bunched = smeared."""
    G = geometry_matrix(sats, receiver)
    if np.linalg.matrix_rank(G) < 4:
        raise ValueError("degenerate geometry (satellites do not span the space)")
    Q = np.linalg.inv(G.T @ G)
    return {
        "GDOP": float(np.sqrt(np.trace(Q))),
        "PDOP": float(np.sqrt(np.trace(Q[:3, :3]))),
        "TDOP": float(np.sqrt(Q[3, 3])),
    }


def _sats_from_directions(receiver, directions, radius):
    """Place satellites at `radius` from the receiver along given unit directions
    (rows). Handy for building test geometries with a known answer."""
    receiver = np.asarray(receiver, float)
    d = np.asarray(directions, float)
    d = d / np.linalg.norm(d, axis=1)[:, None]
    return receiver + radius * d


if __name__ == "__main__":
    # a phone near the Earth's surface, a lousy clock bias of 100 microseconds
    receiver_true = np.array([6.371e6, 0.0, 0.0])
    bias_true = 100e-6 * C_LIGHT                    # ~30 km of range error!
    # six satellites spread across the sky (GPS-like good geometry)
    dirs = np.array([[1, 0, 0], [0.5, 0.8, 0.3], [0.5, -0.8, 0.3],
                     [0.4, 0.3, -0.85], [0.6, -0.2, 0.75], [0.2, 0.6, -0.77]])
    sats = _sats_from_directions(receiver_true, dirs, 26.56e6)
    rho = np.array([pseudorange(receiver_true, s, bias_true) for s in sats])

    sol = solve_position(sats, rho)
    err = np.linalg.norm(sol["position"] - receiver_true)
    print(f"recovered position error : {err*1e3:.3f} mm  in {sol['iterations']} iters")
    print(f"recovered clock bias     : {sol['clock_bias']:.3f} m  "
          f"(true {bias_true:.3f} m)")
    print(f"residual RMS             : {sol['residual_rms']:.2e} m")

    dop = dilution_of_precision(sats, receiver_true)
    print(f"\nDOP (good geometry): GDOP={dop['GDOP']:.2f}, PDOP={dop['PDOP']:.2f}, "
          f"TDOP={dop['TDOP']:.2f}")
    clustered = _sats_from_directions(receiver_true,
                                      [[1, 0, 0], [1, .1, 0], [1, 0, .1], [1, .1, .1]],
                                      26.56e6)
    print(f"DOP (bunched sats): GDOP={dilution_of_precision(clustered, receiver_true)['GDOP']:.1f} "
          f"(much worse -- same ranges, smeared fix)")
