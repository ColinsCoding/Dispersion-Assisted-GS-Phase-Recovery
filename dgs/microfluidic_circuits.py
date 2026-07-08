"""Microfluidics as a circuit: pressure is voltage, flow is current, channels are resistors.

At the micron scale, water flows in perfectly ordered LAMINAR sheets -- the Reynolds
number Re = rho v L / mu sits far below the ~2000 turbulence threshold (usually well
under 1), so there is no turbulence. In that regime a channel obeys a linear law that
is Ohm's law in disguise:
        Delta_P = Q * R_hyd            (pressure drop = flow rate * hydraulic resistance),
with pressure playing the role of voltage, volumetric flow rate Q the role of current,
and the HYDRAULIC RESISTANCE
        R_hyd = 8 mu L / (pi r^4)       (circular channel, Hagen-Poiseuille).
So a microfluidic chip is a resistor network: channels in SERIES add their resistances,
in PARALLEL add their conductances, and a pressure source drives flow through them
exactly like a battery drives current. This is why "lab on a chip" design borrows the
whole toolkit of circuit analysis (dgs.spice, dgs.second_order_systems).

The r^4 in R_hyd is the fact that dominates the field: halving a channel's radius makes
it SIXTEEN times harder to push fluid through. And because flow is laminar, two streams
merged side by side do NOT mix by stirring -- only by DIFFUSION, set by the Peclet number
Pe = v L / D (advection vs diffusion). High Pe (the usual case) means the streams stay
stubbornly separate, which micro-mixers must design around.

Verified: the r^4 scaling, the fluidic Ohm's law round trip, series/parallel combination
matching resistor rules, a parallel-channel flow divider, and Re << 1 / high Pe for a
typical chip -- with a gcc C cross-check of the Hagen-Poiseuille resistance and flow.
Complements dgs.microfluidics (Reynolds, Stokes, droplets). NumPy-free; py-3.13.
"""

import os
import math
import subprocess

MU_WATER = 1.0e-3          # Pa s, dynamic viscosity of water at ~20 C
RHO_WATER = 1000.0         # kg/m^3
D_SMALL_MOLECULE = 1e-9    # m^2/s, typical diffusion coefficient
GCC_DEFAULT = r"C:\msys64\mingw64\bin\gcc.exe"


# ----------------------------------------------------------------------
# Hydraulic resistance and the fluidic Ohm's law
# ----------------------------------------------------------------------

def hydraulic_resistance_circular(radius, length, mu=MU_WATER):
    """Hagen-Poiseuille resistance of a circular channel, R = 8 mu L/(pi r^4)
    [Pa s/m^3]. The r^4 makes narrow channels enormously resistive."""
    if radius <= 0 or length <= 0 or mu <= 0:
        raise ValueError("radius, length, mu must be positive")
    return 8 * mu * length / (math.pi * radius ** 4)


def hydraulic_resistance_rectangular(width, height, length, mu=MU_WATER):
    """Resistance of a shallow rectangular channel (the shape soft-lithography
    actually makes), R ~= 12 mu L / (w h^3 (1 - 0.63 h/w)), valid for h <= w."""
    if width <= 0 or height <= 0 or length <= 0 or mu <= 0:
        raise ValueError("dimensions and mu must be positive")
    if height > width:
        width, height = height, width          # use the shorter side as h
    return 12 * mu * length / (width * height ** 3 * (1 - 0.63 * height / width))


def flow_rate(delta_P, R_hyd):
    """Fluidic Ohm's law: Q = Delta_P / R_hyd [m^3/s]."""
    if R_hyd <= 0:
        raise ValueError("R_hyd must be positive")
    return delta_P / R_hyd


def pressure_drop(Q, R_hyd):
    """Fluidic Ohm's law: Delta_P = Q * R_hyd [Pa]."""
    if R_hyd <= 0:
        raise ValueError("R_hyd must be positive")
    return Q * R_hyd


def series_resistance(*resistances):
    """Channels in series: resistances add (same flow through each)."""
    if not resistances:
        raise ValueError("need at least one channel")
    return sum(resistances)


def parallel_resistance(*resistances):
    """Channels in parallel: conductances add (same pressure across each)."""
    if not resistances or any(r <= 0 for r in resistances):
        raise ValueError("need positive resistances")
    return 1.0 / sum(1.0 / r for r in resistances)


def parallel_channel_flows(delta_P, resistances):
    """A pressure source across several parallel channels -- a flow divider.
    Returns the flow in each channel (Q_i = Delta_P/R_i) and the total; the low-
    resistance channels hog the flow, exactly like a current divider."""
    if any(r <= 0 for r in resistances):
        raise ValueError("resistances must be positive")
    flows = [flow_rate(delta_P, r) for r in resistances]
    return flows, sum(flows)


# ----------------------------------------------------------------------
# Flow regime: laminar (Re) and mixing (Pe)
# ----------------------------------------------------------------------

def reynolds_number(velocity, length, rho=RHO_WATER, mu=MU_WATER):
    """Re = rho v L / mu. Below the ~2000 threshold (and usually well under 1 in
    microfluidics) flow is laminar and reversible -- no turbulence, no inertial
    mixing."""
    if velocity < 0 or length <= 0 or mu <= 0:
        raise ValueError("need v>=0, L>0, mu>0")
    return rho * velocity * length / mu


def peclet_number(velocity, length, D=D_SMALL_MOLECULE):
    """Pe = v L / D: advection vs diffusion. High Pe -> merged laminar streams
    barely mix across the channel over their transit; low Pe -> diffusion wins."""
    if velocity < 0 or length <= 0 or D <= 0:
        raise ValueError("need v>=0, L>0, D>0")
    return velocity * length / D


def mean_velocity(Q, radius):
    """Mean flow speed v = Q / (pi r^2) in a circular channel."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    return Q / (math.pi * radius ** 2)


# ----------------------------------------------------------------------
# C cross-check of the Hagen-Poiseuille resistance and flow
# ----------------------------------------------------------------------

C_SOURCE = r"""
#include <stdio.h>
#include <math.h>
#define PI 3.14159265358979323846
int main(void) {
    double mu = 1e-3, r = 5e-5, L = 0.01, dP = 1000.0;   /* 50 um, 1 cm, 10 mbar */
    double R = 8.0 * mu * L / (PI * pow(r, 4.0));         /* hydraulic resistance */
    double Q = dP / R;                                    /* fluidic Ohm's law   */
    printf("%.12e %.12e\n", R, Q);
    return 0;
}
"""


def gcc_available(gcc_path=GCC_DEFAULT):
    """Whether a C toolchain is present for the compiled cross-check."""
    return os.path.exists(gcc_path)


def compile_and_run_c(out_dir, gcc_path=GCC_DEFAULT):
    """Compile and run C_SOURCE; return the C-computed (R_hyd, Q) for a 50 um,
    1 cm channel at 1000 Pa, to compare against the Python formulas."""
    src = os.path.join(out_dir, "microfluidic.c")
    exe = os.path.join(out_dir, "microfluidic.exe")
    with open(src, "w") as f:
        f.write(C_SOURCE)
    r = subprocess.run([gcc_path, "-O2", "-o", exe, src, "-lm"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"gcc failed: {r.stderr}")
    out = subprocess.run([exe], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"C program failed: {out.stderr}")
    R, Q = map(float, out.stdout.split())
    return {"R_hyd": R, "Q": Q}


if __name__ == "__main__":
    r, L = 50e-6, 0.01
    R = hydraulic_resistance_circular(r, L)
    dP = 1000.0
    Q = flow_rate(dP, R)
    print(f"circular channel r=50um L=1cm: R_hyd = {R:.3e} Pa.s/m^3")
    print(f"  at 1000 Pa: Q = {Q:.3e} m^3/s = {Q*6e10:.1f} uL/min")
    print(f"  halve the radius -> R x {hydraulic_resistance_circular(r/2, L)/R:.0f} "
          f"(the r^4 law)")

    print("\nfluidic circuit: two channels in parallel (one 2x wider bore)")
    R1 = hydraulic_resistance_circular(50e-6, L)
    R2 = hydraulic_resistance_circular(100e-6, L)
    flows, total = parallel_channel_flows(dP, [R1, R2])
    print(f"  R_parallel = {parallel_resistance(R1, R2):.3e}, "
          f"flow split = {[f'{f/total*100:.1f}%' for f in flows]} (wide bore hogs it)")

    v = mean_velocity(Q, r)
    print(f"\nregime: v = {v*1e3:.2f} mm/s -> Re = {reynolds_number(v, 2*r):.2e} (laminar), "
          f"Pe = {peclet_number(v, 2*r):.0f} (diffusion-limited mixing)")

    if gcc_available():
        c = compile_and_run_c(os.environ.get("TEMP", "."))
        print(f"\nC cross-check: R={c['R_hyd']:.3e}, Q={c['Q']:.3e}  "
              f"matches Python? {math.isclose(c['R_hyd'], R) and math.isclose(c['Q'], Q)}")
    else:
        print("\n(gcc not found -- skipping C cross-check)")
