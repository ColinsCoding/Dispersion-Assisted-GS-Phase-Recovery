"""Test dgs.proton_proton_chain: the net 4H->He4 energy (~26.7 MeV) reached two
ways (mass defect and step sum), the binding-energy-per-nucleon curve peaking at
iron-56, and the Sun's fusion rate (protons/s, hydrogen kg/s, mass->energy kg/s)
from its luminosity."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import proton_proton_chain as pp

# 1. net pp-chain energy (4 m_H - m_He4) c^2 ~= 26.73 MeV
Q = pp.pp_chain_net_energy()
assert np.isclose(Q, 26.73, atol=0.02)
assert np.isclose(Q, pp.mass_defect_energy([pp.M["1H"]] * 4, [pp.M["4He"]]))
# reversing the reaction costs the same energy (endothermic -> negative Q)
assert np.isclose(pp.mass_defect_energy([pp.M["4He"]], [pp.M["1H"]] * 4), -Q)

# 2. the three steps, weighted by multiplicity, sum to the same net energy
assert np.isclose(pp.pp_chain_step_sum(), Q, atol=0.02)     # independent check

# 3. neutrinos carry ~0.59 MeV away; the rest becomes light
assert np.isclose(pp.energy_as_light(), Q - pp.NEUTRINO_LOSS_MEV)
assert pp.energy_as_light() < Q                              # some energy escapes

# 4. binding energies vs known values
assert np.isclose(pp.binding_energy(2, 2, pp.M["4He"]), 28.30, atol=0.05)   # He-4
assert np.isclose(pp.binding_energy(1, 1, pp.M["2H"]), 2.22, atol=0.02)     # deuteron
assert np.isclose(pp.binding_energy(26, 30, pp.M["56Fe"]), 492.26, atol=0.5)
assert np.isclose(pp.binding_energy_per_nucleon(2, 2, pp.M["4He"]), 7.07, atol=0.02)
assert np.isclose(pp.binding_energy_per_nucleon(26, 30, pp.M["56Fe"]), 8.79, atol=0.02)

# 5. THE IRON PEAK: B/A is highest at iron-56 -- above lighter He-4 AND heavier U-238
ba = {name: pp.binding_energy_per_nucleon(Z, N, pp.M[name])
      for name, Z, N in [("2H", 1, 1), ("4He", 2, 2), ("56Fe", 26, 30), ("238U", 92, 146)]}
assert ba["56Fe"] == max(ba.values())
assert ba["56Fe"] > ba["4He"] and ba["56Fe"] > ba["238U"]   # fusion up, fission down
assert ba["2H"] < ba["4He"]                                 # rising toward the peak

# 6. solar fusion rate from luminosity, and its internal consistency
L = 3.828e26
r = pp.solar_fusion_rate(L)
# each He-4 releases energy_as_light -> he4/s * that = luminosity
assert np.isclose(r["he4_per_s"] * pp.energy_as_light() * pp.MEV_TO_J, L, rtol=1e-9)
assert np.isclose(r["protons_per_s"], 4 * r["he4_per_s"])   # 4 protons per He-4
assert np.isclose(r["mass_to_energy_kg_per_s"], L / pp.C_LIGHT**2)   # ~4.26e9 kg/s
assert np.isclose(r["mass_to_energy_kg_per_s"], 4.26e9, rtol=0.02)
assert np.isclose(r["hydrogen_kg_per_s"], 6.1e11, rtol=0.05)  # ~600 million tonnes/s
# fusing hydrogen mass >> the mass actually converted to energy (~0.7% defect)
assert r["hydrogen_kg_per_s"] > 100 * r["mass_to_energy_kg_per_s"]

# 7. kwarg bounds
for bad in (lambda: pp.binding_energy(-1, 2, 4.0),
            lambda: pp.binding_energy_per_nucleon(0, 0, 0.0),
            lambda: pp.solar_fusion_rate(0)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_proton_proton_chain: all checks passed")
