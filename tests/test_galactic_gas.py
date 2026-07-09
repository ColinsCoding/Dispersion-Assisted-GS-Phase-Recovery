"""Test dgs.galactic_gas: virial temperature (2K+U=0), Jeans length/mass, free-fall time,
the sound-speed tie to dgs.maxwell_boltzmann, and known astrophysical scales."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs import galactic_gas as gg

G, k, mH, Msun, pc = gg.G_GRAV, gg.K_BOLTZ, gg.M_H, gg.M_SUN, gg.PARSEC

# 1. sound speed matches the closed form c_s = sqrt(gamma k T / mu m_H)
assert math.isclose(gg.sound_speed(1e4, mu=0.6, gamma=1.0),
                    math.sqrt(k * 1e4 / (0.6 * mH)), rel_tol=1e-12)
# adiabatic (gamma=5/3) faster than isothermal (gamma=1)
assert gg.sound_speed(1e4, gamma=5/3) > gg.sound_speed(1e4, gamma=1.0)

# 2. sound speed ties to Maxwell-Boltzmann rms speed: adiabatic c_s = v_rms*sqrt(gamma/3)
from dgs import maxwell_boltzmann as mb
m_particle = 0.61 * mH
v_rms = math.sqrt(3 * k * 3e7 / m_particle)          # rms speed of a plasma proton-ish particle
cs_adiabatic = gg.sound_speed(3e7, mu=0.61, gamma=5/3)
assert math.isclose(cs_adiabatic, v_rms * math.sqrt((5/3) / 3), rel_tol=1e-9)

# 3. virial temperature = G M mu m_H / (5 k R)
M, R = 1e15 * Msun, 2 * gg.MPC
assert math.isclose(gg.virial_temperature(M, R, mu=0.61),
                    G * M * 0.61 * mH / (5 * k * R), rel_tol=1e-12)
# a galaxy cluster virializes around 10^7 K (X-ray emitting intracluster medium)
assert 1e7 < gg.virial_temperature(M, R) < 1e8
# hotter/more massive OR more compact -> higher T_vir
assert gg.virial_temperature(2 * M, R) > gg.virial_temperature(M, R)
assert gg.virial_temperature(M, R / 2) > gg.virial_temperature(M, R)

# 4. Jeans length and mass consistency: M_J = (4/3) pi rho (lambda_J/2)^3
rho = gg.number_to_mass_density(1e10, mu=gg.MU_MOLECULAR)   # 1e4 /cm^3
lam = gg.jeans_length(10.0, rho)
assert math.isclose(gg.jeans_mass(10.0, rho),
                    (4/3) * math.pi * rho * (lam / 2) ** 3, rel_tol=1e-12)
# cold molecular cloud -> Jeans mass of order a few solar masses
assert 0.5 < gg.jeans_mass(10.0, rho) / Msun < 20

# 5. scaling laws: hotter gas resists collapse (bigger M_J); denser gas collapses easier
assert gg.jeans_mass(20.0, rho) > gg.jeans_mass(10.0, rho)          # M_J ~ T^{3/2}
assert gg.jeans_mass(10.0, 4 * rho) < gg.jeans_mass(10.0, rho)      # M_J ~ rho^{-1/2}
# M_J ~ T^{3/2}: doubling T multiplies M_J by 2^{3/2}
assert math.isclose(gg.jeans_mass(20.0, rho) / gg.jeans_mass(10.0, rho),
                    2 ** 1.5, rel_tol=1e-9)
# M_J ~ rho^{-1/2}: quadrupling rho halves M_J
assert math.isclose(gg.jeans_mass(10.0, 4 * rho) / gg.jeans_mass(10.0, rho),
                    0.5, rel_tol=1e-9)

# 6. free-fall time = sqrt(3 pi / 32 G rho), denser -> faster collapse
assert math.isclose(gg.free_fall_time(rho),
                    math.sqrt(3 * math.pi / (32 * G * rho)), rel_tol=1e-12)
assert gg.free_fall_time(4 * rho) < gg.free_fall_time(rho)

# 7. the instability flag
assert gg.is_jeans_unstable(100 * Msun, 10.0, rho)          # heavy clump collapses
assert not gg.is_jeans_unstable(0.01 * Msun, 10.0, rho)     # tiny clump stable

# 8. kwarg bounds
for bad in (lambda: gg.sound_speed(0),
            lambda: gg.sound_speed(1e4, gamma=0),
            lambda: gg.virial_temperature(0, 1e22),
            lambda: gg.virial_temperature(1e40, 0),
            lambda: gg.free_fall_time(0),
            lambda: gg.jeans_length(0, rho),
            lambda: gg.number_to_mass_density(-1)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_galactic_gas: all checks passed")
