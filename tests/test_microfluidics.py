"""Tests for dgs/microfluidics.py"""
import numpy as np
import pytest

try:
    from dgs.microfluidics import (
        reynolds_number, stokes_flow_channel, capillary_number,
        diffusion_timescale, pdms_fabrication_protocol,
        droplet_generator, atomic_emission_spectra,
        linux_lab_server_setup, group_z2_binary,
        MU_WATER, RHO_WATER,
    )
except ImportError:
    from microfluidics import (
        reynolds_number, stokes_flow_channel, capillary_number,
        diffusion_timescale, pdms_fabrication_protocol,
        droplet_generator, atomic_emission_spectra,
        linux_lab_server_setup, group_z2_binary,
        MU_WATER, RHO_WATER,
    )


class TestReynoldsNumber:
    def test_microfluidics_laminar(self):
        r = reynolds_number(RHO_WATER, 1e-3, 10e-6, MU_WATER)
        assert r['Re'] < 1
        assert r['regime'] == 'laminar'

    def test_pipe_turbulent(self):
        r = reynolds_number(RHO_WATER, 5.0, 0.01, MU_WATER)
        assert r['Re'] > 4000
        assert r['regime'] == 'turbulent'

    def test_reversible_at_low_re(self):
        r = reynolds_number(RHO_WATER, 1e-4, 10e-6, MU_WATER)
        assert r['reversible'] is True


class TestStokesFlow:
    def test_flow_rate_positive(self):
        f = stokes_flow_channel(100, 50)
        assert f['Q_nL_per_s'] > 0
        assert f['u_max_mm_per_s'] > 0

    def test_max_gt_avg(self):
        f = stokes_flow_channel(100, 50)
        assert f['u_max_mm_per_s'] > f['u_avg_mm_per_s']

    def test_laminar_regime(self):
        f = stokes_flow_channel(100, 50)
        assert f['Re'] < 100

    def test_invalid_dims(self):
        with pytest.raises(ValueError):
            stokes_flow_channel(-10, 50)

    def test_parabolic_profile(self):
        f = stokes_flow_channel(100, 50)
        u = f['u_profile_mm_per_s']
        # max at center, zero at walls
        assert u[0] == pytest.approx(0.0, abs=1e-10)
        assert u[-1] == pytest.approx(0.0, abs=1e-3)
        assert np.argmax(u) in range(40, 60)  # peak near center


class TestCapillaryNumber:
    def test_droplet_regime(self):
        ca = capillary_number(mu=5e-3, v=1e-4)
        assert ca['Ca'] < 0.1
        assert 'droplet' in ca['regime']

    def test_continuous_regime(self):
        ca = capillary_number(mu=5e-3, v=100.0)
        assert ca['Ca'] > 1
        assert 'continuous' in ca['regime']

    def test_ca_value(self):
        from dgs.microfluidics import GAMMA_WATER_AIR
        mu, v = 1e-3, 0.1
        ca = capillary_number(mu, v, GAMMA_WATER_AIR)
        assert ca['Ca'] == pytest.approx(mu*v/GAMMA_WATER_AIR, rel=1e-6)


class TestDiffusion:
    def test_tau_quadratic_in_L(self):
        dt1 = diffusion_timescale(1e-10, 10.0)
        dt2 = diffusion_timescale(1e-10, 20.0)
        assert dt2['tau_ms'] == pytest.approx(4 * dt1['tau_ms'], rel=1e-4)

    def test_positive_timescale(self):
        dt = diffusion_timescale(8e-11, 50.0)
        assert dt['tau_ms'] > 0

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            diffusion_timescale(0, 10)

    def test_small_molecule_faster(self):
        dt = diffusion_timescale(1e-10, 50.0)
        ts = dt['species_timescales_s']
        assert ts['small_molecule_100Da'] < ts['DNA_10kbp']


class TestDropletGenerator:
    def test_droplet_size_positive(self):
        dg = droplet_generator(10, 2)
        assert dg['droplet_diameter_um'] > 0
        assert dg['droplet_volume_pL'] > 0

    def test_larger_water_flow_larger_droplet(self):
        dg1 = droplet_generator(10, 1)
        dg2 = droplet_generator(10, 5)
        assert dg2['droplet_diameter_um'] > dg1['droplet_diameter_um']

    def test_invalid_flow_rates(self):
        with pytest.raises(ValueError):
            droplet_generator(0, 1)

    def test_frequency_positive(self):
        dg = droplet_generator(5, 1)
        assert dg['frequency_Hz'] > 0


class TestAtomicEmission:
    def test_five_colors(self):
        ae = atomic_emission_spectra()
        assert len(ae['colors']) == 5

    def test_photon_energies_visible(self):
        ae = atomic_emission_spectra()
        for name, d in ae['colors'].items():
            assert 1.5 < d['E_photon_eV'] < 3.5   # visible range

    def test_yellow_Na_589(self):
        ae = atomic_emission_spectra()
        assert ae['colors']['yellow_Na']['lambda_nm'] == 589

    def test_blue_shorter_than_red(self):
        ae = atomic_emission_spectra()
        assert ae['colors']['blue_Cu']['lambda_nm'] < ae['colors']['red_Sr']['lambda_nm']

    def test_steam_connection(self):
        ae = atomic_emission_spectra()
        assert 'STEAM' in ae['jalali_ultrafast'] or 'time-stretch' in ae['jalali_ultrafast']


class TestLinuxBoot:
    def test_boot_sequence_steps(self):
        r = linux_lab_server_setup()
        assert len(r['boot_sequence']) >= 4
        assert any('GRUB' in s or 'kernel' in s.lower() for s in r['boot_sequence'])

    def test_systemd_in_sequence(self):
        r = linux_lab_server_setup()
        assert any('systemd' in s.lower() for s in r['boot_sequence'])

    def test_rpi_connection(self):
        r = linux_lab_server_setup()
        assert 'RPi' in r['rpi_for_rogueguard'] or 'RogueGuard' in r['rpi_for_rogueguard']


class TestGroupZ2:
    def test_cayley_table(self):
        g = group_z2_binary()
        T = g['cayley_table']
        # XOR table: T[0,0]=0, T[0,1]=1, T[1,0]=1, T[1,1]=0
        assert T[0,0] == 0
        assert T[0,1] == 1
        assert T[1,1] == 0

    def test_identity_is_0(self):
        g = group_z2_binary()
        assert g['axioms']['identity'] == 0

    def test_parity_detects_error(self):
        g = group_z2_binary()
        assert g['parity_check']['error_detected'] is True

    def test_xor_decryption(self):
        g = group_z2_binary()
        assert g['xor_encryption']['recovered'] is True

    def test_ft_connection(self):
        g = group_z2_binary()
        conn = g['FT_connection']
        assert 'parity' in conn.lower() or 'FT' in conn
