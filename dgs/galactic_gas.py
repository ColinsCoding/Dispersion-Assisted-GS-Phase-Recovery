"""Hot galactic gas: gravity vs thermal pressure -- virial temperature and the Jeans collapse.

The gas filling galaxies and galaxy clusters is HOT -- millions of kelvin -- and whether a
cloud of it collapses into stars or sits in equilibrium is a contest between two things:
its own gravity pulling in, and its thermal pressure pushing back. Both are governed by the
same equipartition energy 1/2 kT per particle from dgs.degrees_of_freedom / maxwell_boltzmann;
here that thermal energy is set against the gravitational binding energy.

VIRIAL THEOREM.  A self-gravitating system in equilibrium obeys 2K + U = 0. With translational
kinetic energy K = (3/2) N kT and (uniform-sphere) potential energy U = -(3/5) G M^2 / R, the
gas settles at the VIRIAL TEMPERATURE
        kT_vir = (1/5) G M mu m_H / R,
i.e. gravity heats the gas until its particles move fast enough to hold it up. A galaxy
cluster (M ~ 10^15 M_sun, R ~ few Mpc) virializes at ~3x10^7 K -- which is exactly why the
intracluster medium shines in X-rays.

JEANS INSTABILITY.  Run it the other way: a cold, dense cloud collapses if gravity beats
pressure. Pressure support propagates at the sound speed c_s = sqrt(gamma kT / mu m_H) (the SAME
c = sqrt(gamma RT/M) as the speed of sound in air, with mu m_H the mean particle mass), so a
perturbation collapses if it is bigger than the distance sound can cross before gravity pulls it
in -- the JEANS LENGTH lambda_J = c_s sqrt(pi / G rho), enclosing the JEANS MASS
M_J = (4/3) pi rho (lambda_J/2)^3. Above M_J, collapse; the whole thing falls in a FREE-FALL
TIME t_ff = sqrt(3 pi / 32 G rho). This is the acoustic threshold of star formation.

Verified against known scales: cold molecular cloud (10 K) -> M_J ~ a few M_sun; cluster gas
-> T_vir ~ 10^7 K. Pure-Python (float in, float out); py-3.13.
"""

import math

G_GRAV = 6.674e-11          # N m^2 / kg^2
K_BOLTZ = 1.380649e-23      # J / K
M_H = 1.6735575e-27         # hydrogen-atom mass, kg
M_SUN = 1.98892e30          # kg
PARSEC = 3.0856775815e16    # m
MPC = 1e6 * PARSEC          # megaparsec, m
YEAR = 3.1557e7             # s

# Mean molecular weight mu (mean particle mass in units of m_H):
MU_IONIZED = 0.61           # fully ionized primordial/solar plasma (hot gas)
MU_ATOMIC = 1.3             # neutral atomic H+He
MU_MOLECULAR = 2.33         # molecular H2 + He cloud


def number_to_mass_density(n, mu=MU_MOLECULAR):
    """Mass density rho = n * mu * m_H from particle number density n [1/m^3]."""
    if n < 0 or mu <= 0:
        raise ValueError("n must be >= 0 and mu > 0")
    return n * mu * M_H


def sound_speed(T, mu=MU_IONIZED, gamma=1.0):
    """Speed of sound in a gas of mean particle mass mu*m_H:
        c_s = sqrt(gamma k T / (mu m_H)).
    gamma=1 gives the ISOTHERMAL sound speed used in the Jeans analysis; gamma=5/3 the
    adiabatic one. Same formula as the speed of sound in air (dgs.degrees_of_freedom)."""
    if T <= 0 or mu <= 0 or gamma <= 0:
        raise ValueError("T, mu, gamma must all be > 0")
    return math.sqrt(gamma * K_BOLTZ * T / (mu * M_H))


def free_fall_time(rho):
    """Gravitational free-fall (collapse) time of a cloud of density rho:
        t_ff = sqrt(3 pi / (32 G rho))  [s]."""
    if rho <= 0:
        raise ValueError("rho must be > 0")
    return math.sqrt(3 * math.pi / (32 * G_GRAV * rho))


def jeans_length(T, rho, mu=MU_MOLECULAR):
    """Jeans length lambda_J = c_s sqrt(pi / (G rho))  [m] -- the smallest perturbation that
    collapses. Uses the isothermal sound speed at temperature T."""
    if T <= 0 or rho <= 0:
        raise ValueError("T and rho must be > 0")
    cs = sound_speed(T, mu=mu, gamma=1.0)
    return cs * math.sqrt(math.pi / (G_GRAV * rho))


def jeans_mass(T, rho, mu=MU_MOLECULAR):
    """Jeans mass -- the gas enclosed in a sphere of one Jeans length across:
        M_J = (4/3) pi rho (lambda_J / 2)^3  [kg].
    A cloud heavier than this collapses under its own gravity."""
    lam = jeans_length(T, rho, mu=mu)
    return (4.0 / 3.0) * math.pi * rho * (lam / 2.0) ** 3


def virial_temperature(M, R, mu=MU_IONIZED):
    """Virial temperature of a self-gravitating gas sphere (2K + U = 0, uniform sphere):
        kT_vir = (1/5) G M mu m_H / R   ->   T_vir = G M mu m_H / (5 k R)  [K].
    The temperature gravity heats the gas to in order to support itself."""
    if M <= 0 or R <= 0 or mu <= 0:
        raise ValueError("M, R, mu must all be > 0")
    return G_GRAV * M * mu * M_H / (5.0 * K_BOLTZ * R)


def is_jeans_unstable(mass, T, rho, mu=MU_MOLECULAR):
    """True if a cloud of given `mass`, temperature and density will gravitationally
    collapse (mass exceeds the Jeans mass)."""
    return mass > jeans_mass(T, rho, mu=mu)


if __name__ == "__main__":
    print("=== virial temperature: gravity heats the gas ===")
    for name, M, R in [("galaxy cluster", 1e15 * M_SUN, 2 * MPC),
                       ("Milky Way hot halo", 1e12 * M_SUN, 100 * 1e3 * PARSEC)]:
        T = virial_temperature(M, R)
        print(f"  {name:20s}  T_vir = {T:.2e} K  (c_s = {sound_speed(T):.0f} m/s)")

    print("\n=== Jeans collapse: a cold molecular cloud ===")
    n = 1e4 * 1e6              # 1e4 per cm^3 -> per m^3
    rho = number_to_mass_density(n, mu=MU_MOLECULAR)
    T = 10.0
    print(f"  T = {T} K, n = 1e4 /cm^3, rho = {rho:.2e} kg/m^3")
    print(f"  sound speed   c_s   = {sound_speed(T, mu=MU_MOLECULAR):.0f} m/s")
    print(f"  Jeans length  lam_J = {jeans_length(T, rho)/PARSEC:.3f} pc")
    print(f"  Jeans mass    M_J   = {jeans_mass(T, rho)/M_SUN:.2f} M_sun")
    print(f"  free-fall time t_ff = {free_fall_time(rho)/YEAR:.2e} yr")
    print(f"  a 100 M_sun clump unstable? {is_jeans_unstable(100*M_SUN, T, rho)}")
