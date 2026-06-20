"""Sonoluminescence: shake a liquid in the dark, get a flash of light.

A sound field drives a micron gas bubble. On the low-pressure half-cycle it
balloons; then the liquid crushes it in a near-supersonic collapse that
adiabatically compresses and heats the trapped gas to thousands of kelvin -- and
it radiates a picosecond flash of broadband (~blackbody) light. The bubble
radius R(t) obeys the **Rayleigh-Plesset equation**, a nonlinear 2nd-order ODE:

    rho ( R R'' + 3/2 R'^2 ) = p_gas(R) - p_inf(t) - 2 sigma/R - 4 mu R'/R.

We integrate it numerically in torch (RK4, fully differentiable), watch the
collapse, and estimate the flash temperature from the adiabatic compression.

It's a 'famous differential equation that models reality' AND a light source --
the optics tie to this repo: the hot collapse radiates ~blackbody, so
quantum_statistics.planck_spectral_radiance turns R_min into a spectrum.
torch (py 3.12 here). Civilian physics education.
"""

import numpy as np

# default single-bubble sonoluminescence parameters (air bubble in water)
WATER = dict(
    rho=998.0,        # liquid density [kg/m^3]
    sigma=0.0725,     # surface tension [N/m]
    mu=1.0e-3,        # liquid viscosity [Pa s]
    p0=101325.0,      # ambient pressure [Pa]
    kappa=1.4,        # polytropic exponent of the gas (adiabatic)
    R0=4.5e-6,        # ambient bubble radius [m]
    Pa=1.3 * 101325,  # acoustic drive amplitude [Pa]
    f=26500.0,        # drive frequency [Hz]
    a_ratio=8.86,     # van der Waals hardcore: R cannot fall below R0/a_ratio
)


def _accel(R, Rdot, t, P):
    """Rayleigh-Plesset radial acceleration R'' (torch tensors).

    R is clamped just above the van der Waals hardcore so the gas pressure stays
    finite through the violent collapse (otherwise R^3 - a^3 -> 0 blows up).
    """
    import torch
    a = P["R0"] / P["a_ratio"]                                   # hardcore radius
    Rc = torch.clamp(R, min=a * 1.001)                          # keep gas pressure finite
    # adiabatic gas pressure with a hardcore so the collapse stays bounded
    p_gas = (P["p0"] + 2 * P["sigma"] / P["R0"]) * \
            ((P["R0"]**3 - a**3) / (Rc**3 - a**3))**P["kappa"]
    p_inf = P["p0"] + P["Pa"] * torch.sin(2 * np.pi * P["f"] * t)
    rhs = (p_gas - p_inf - 2 * P["sigma"] / Rc - 4 * P["mu"] * Rdot / Rc) / P["rho"]
    return (rhs - 1.5 * Rdot**2) / Rc


def simulate(params=None, n_periods=3.0, steps_per_period=20000, **overrides):
    """Integrate Rayleigh-Plesset with RK4. Returns dict(t, R, Rdot) as numpy.

    Keyword overrides patch the default WATER parameters (e.g. Pa=1.4*101325).
    """
    import torch
    P = dict(WATER)
    if params:
        P.update(params)
    P.update(overrides)
    if P["Pa"] < 0 or P["R0"] <= 0 or P["f"] <= 0:
        raise ValueError("Pa >= 0, R0 > 0, f > 0 required")

    T = n_periods / P["f"]
    N = int(n_periods * steps_per_period)
    dt = T / N
    t = torch.zeros(N + 1, dtype=torch.float64)
    R = torch.zeros(N + 1, dtype=torch.float64)
    Rd = torch.zeros(N + 1, dtype=torch.float64)
    R[0] = P["R0"]

    a_hard = P["R0"] / P["a_ratio"]                  # gas can't be crushed below this
    floor = a_hard * 1.001
    r, rd = torch.tensor(P["R0"], dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64)
    for i in range(N):
        ti = torch.tensor(i * dt, dtype=torch.float64)
        k1r, k1v = rd, _accel(r, rd, ti, P)
        k2r, k2v = rd + 0.5*dt*k1v, _accel(r + 0.5*dt*k1r, rd + 0.5*dt*k1v, ti + 0.5*dt, P)
        k3r, k3v = rd + 0.5*dt*k2v, _accel(r + 0.5*dt*k2r, rd + 0.5*dt*k2v, ti + 0.5*dt, P)
        k4r, k4v = rd + dt*k3v,     _accel(r + dt*k3r,     rd + dt*k3v,     ti + dt, P)
        r = r + dt/6 * (k1r + 2*k2r + 2*k3r + k4r)
        rd = rd + dt/6 * (k1v + 2*k2v + 2*k3v + k4v)
        if r < floor:                                # enforce the hardcore: bounce off it
            r = torch.tensor(floor, dtype=torch.float64)
            rd = torch.clamp(rd, min=torch.tensor(0.0, dtype=torch.float64))
        t[i+1], R[i+1], Rd[i+1] = (i+1)*dt, r, rd
    return {"t": t.numpy(), "R": R.numpy(), "Rdot": Rd.numpy(),
            "R0": P["R0"], "kappa": P["kappa"]}


def flash_temperature(R_min, R0, kappa=1.4, T0=300.0):
    """Peak gas temperature from adiabatic compression: T = T0 (R0/R_min)^{3(kappa-1)}.

    The collapse compresses the gas adiabatically (T R^{3(kappa-1)} = const), so a
    10x radius collapse with kappa=1.4 multiplies the temperature ~16x -> the flash.
    """
    if R_min <= 0 or R0 <= 0:
        raise ValueError("radii must be > 0")
    return T0 * (R0 / R_min)**(3 * (kappa - 1))


def collapse_summary(sol, T0=300.0):
    """Pull the headline numbers out of a simulation: R_max, R_min, peak T, flash time."""
    R = sol["R"]
    i_min = int(np.argmin(R))
    R_min, R_max = float(R[i_min]), float(R.max())
    return {
        "R_max": R_max, "R_min": R_min,
        "expansion_ratio": R_max / sol["R0"],
        "collapse_ratio": sol["R0"] / R_min,
        "flash_T": flash_temperature(R_min, sol["R0"], sol["kappa"], T0),
        "flash_time": float(sol["t"][i_min]),
    }


if __name__ == "__main__":
    sol = simulate()
    s = collapse_summary(sol)
    print(f"ambient R0 = {sol['R0']*1e6:.2f} um")
    print(f"expands to R_max = {s['R_max']*1e6:.2f} um  ({s['expansion_ratio']:.1f}x)")
    print(f"collapses to R_min = {s['R_min']*1e6:.3f} um  ({s['collapse_ratio']:.1f}x)")
    print(f"flash temperature ~ {s['flash_T']:.0f} K  at t = {s['flash_time']*1e6:.2f} us")
