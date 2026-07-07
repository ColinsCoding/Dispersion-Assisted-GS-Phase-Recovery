"""Test dgs.eye_diagram: PRBS7 period, clean-eye Q/BER sanity, noise and
dispersion each close the eye, D=0 is a no-op, ber_from_q hits the classic
Q=6 -> 1e-9 line and round-trips through q_for_target_ber, and the DFT
parity theorem holds to machine precision (even -> real, odd -> imaginary,
Hermitian symmetry for the sum)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import eye_diagram as ed

sps = 32
bits = ed.prbs_bits(600)

# 1. PRBS7 has period 127 and is balanced (64 ones, 63 zeros per period)
period = ed.prbs_bits(254)
assert np.array_equal(period[:127], period[127:254])
assert sum(period[:127]) in (63, 64)

# 2. clean band-limited NRZ: wide-open eye, huge Q, negligible BER
clean = ed.pam_waveform(bits, sps=sps)
m_clean = ed.eye_metrics(ed.fold_eye(clean, sps=sps), sps=sps)
assert m_clean["q_factor"] > 15
assert m_clean["ber"] < 1e-30
assert m_clean["eye_height"] > 0.5
assert m_clean["eye_width"] > 0.4

# 3. noise closes the eye vertically: Q drops, BER rises, height shrinks
noisy = ed.add_noise(clean, 0.06)
m_noisy = ed.eye_metrics(ed.fold_eye(noisy, sps=sps), sps=sps)
assert m_noisy["q_factor"] < m_clean["q_factor"]
assert m_noisy["ber"] > m_clean["ber"]
assert m_noisy["eye_height"] < m_clean["eye_height"]

# 4. Q ~= 1/(2*sigma) for a unit eye with equal-noise rails: sigma=0.06
#    predicts Q ~= 8.3; measured must land in that neighborhood
assert 5 < m_noisy["q_factor"] < 12

# 5. the all-pass dispersion phase changes NO spectral amplitude, yet the
#    detected eye closes -- phase becomes amplitude ISI through |.|^2
dispersed = ed.apply_fiber_dispersion(clean, sps=sps, D=0.4)
m_disp = ed.eye_metrics(ed.fold_eye(dispersed, sps=sps), sps=sps)
assert m_disp["q_factor"] < 25 < m_clean["q_factor"]     # real penalty...
assert m_disp["eye_height"] < 0.9 * m_clean["eye_height"]
assert m_disp["ber"] < 1e-9                              # ...but still a link
# energy is conserved by an all-pass filter (Parseval, field domain)
field_in = np.sqrt(np.clip(clean, 0, None))
assert np.isclose(np.sum(field_in**2), np.sum(dispersed))

# 6. D=0 is the identity (up to the sqrt/square round trip)
assert np.allclose(ed.apply_fiber_dispersion(clean, sps=sps, D=0.0),
                   np.clip(clean, 0, None), atol=1e-12)

# 7. the classic BER lines: Q=0 -> coin flip, Q=6 -> ~1e-9 (9.87e-10)
assert ed.ber_from_q(0.0) == 0.5
assert 9.5e-10 < ed.ber_from_q(6.0) < 1.05e-9
# inverse round-trips: Q for 1e-12 is the textbook ~7.03
q12 = ed.q_for_target_ber(1e-12)
assert 7.0 < q12 < 7.1
assert np.isclose(ed.ber_from_q(q12), 1e-12, rtol=1e-6)

# 8. DFT parity theorem at machine precision: even part -> purely real
#    spectrum, odd part -> purely imaginary, sum -> Hermitian symmetric
chk = ed.fourier_parity_check(clean)
scale = np.max(np.abs(chk["even_spectrum"]))
assert chk["even_spectrum_max_imag"] < 1e-9 * scale
assert chk["odd_spectrum_max_real"] < 1e-9 * scale
assert chk["hermitian_residual"] < 1e-9 * scale
# and the circular split reconstructs the signal exactly
assert np.allclose(ed.circular_even_part(clean) + ed.circular_odd_part(clean),
                   clean)

# 9. kwarg bounds: clear errors on bad input
for bad in (lambda: ed.prbs_bits(0),
            lambda: ed.pam_waveform(bits, sps=4),
            lambda: ed.pam_waveform(bits, levels=1),
            lambda: ed.pam_waveform([0, 1, 5, 2], levels=4),
            lambda: ed.pam_waveform(bits, bandwidth=0),
            lambda: ed.add_noise(clean, -1),
            lambda: ed.ber_from_q(-1),
            lambda: ed.q_for_target_ber(0.7),
            lambda: ed.fold_eye(clean[:100], sps=sps)):
    try:
        bad()
        assert False, "expected ValueError"
    except ValueError:
        pass

# 10. PAM4 waveform generation: four distinct levels survive the filter
pam4 = ed.pam_waveform(ed.prbs_bits(1200)[:600] + 2 * ed.prbs_bits(600, seed=0x33),
                       sps=sps, levels=4)
centers = ed.fold_eye(pam4, sps=sps)[:, sps // 2]
assert len(np.unique(np.round(centers * 3))) == 4

print("test_eye_diagram: all checks passed")
