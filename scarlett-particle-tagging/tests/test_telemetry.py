"""RF telemetry model: packet format, CRC, link budget, corrupted packets."""
import math

import numpy as np
import pytest

from scarlett import telemetry as tm

REPORT = tm.TagReport(tag_id=0x5CA41E77, timestamp_ms=123_456, dia=4.998e-6,
                      dia_ci=(4.993e-6, 5.003e-6), npar=1.57, confidence_level=0.95,
                      concentration=1.995e13, diffusion=8.6e-14)


def test_crc16_known_answer():
    assert tm.crc16_ccitt(b"123456789") == 0x29B1          # CRC-16/CCITT-FALSE check value


def test_packet_round_trip():
    packet = tm.encode_packet(REPORT)
    assert len(packet) == tm.PACKET_BYTES == 42
    decoded = tm.decode_packet(packet)
    assert decoded.tag_id == REPORT.tag_id and decoded.timestamp_ms == REPORT.timestamp_ms
    assert decoded.dia == pytest.approx(REPORT.dia, abs=0.5e-9)       # 1 nm resolution
    assert decoded.dia_ci == pytest.approx(REPORT.dia_ci, abs=0.5e-9)
    assert decoded.npar == pytest.approx(1.57) and decoded.confidence_level == pytest.approx(0.95)
    assert decoded.concentration == pytest.approx(REPORT.concentration, rel=1e-6)
    assert decoded.diffusion == pytest.approx(REPORT.diffusion, rel=1e-6)
    assert decoded.flags == 0


def test_missing_motion_sets_a_flag():
    report = tm.TagReport(1, 0, 5e-6, (4.9e-6, 5.1e-6), 1.57, 0.95, 1e13)
    decoded = tm.decode_packet(tm.encode_packet(report))
    assert math.isnan(decoded.diffusion) and decoded.flags & tm.FLAG_NO_MOTION


def test_every_single_bit_error_is_detected():
    packet = tm.encode_packet(REPORT)
    for bit in range(8 * len(packet)):
        corrupted = bytearray(packet)
        corrupted[bit // 8] ^= 1 << (bit % 8)
        with pytest.raises(ValueError):
            tm.decode_packet(bytes(corrupted))


def test_decode_rejects_wrong_length():
    with pytest.raises(ValueError, match="42 bytes"):
        tm.decode_packet(b"\x00" * 10)


@pytest.mark.parametrize("change", [dict(tag_id=2 ** 32), dict(dia_ci=(5.1e-6, 5.2e-6)),
                                    dict(npar=-1.0), dict(confidence_level=1.0),
                                    dict(timestamp_ms=-1)])
def test_encode_rejects_inconsistent_reports(change):
    fields = {**REPORT.__dict__, **change}
    with pytest.raises(ValueError):
        tm.encode_packet(tm.TagReport(**fields))


def test_link_budget_free_space_values():
    one_metre = tm.link_budget(1.0)
    assert one_metre["path_loss_db"] == pytest.approx(40.2, abs=0.1)     # 2.44 GHz at 1 m
    ten_metres = tm.link_budget(10.0)
    assert ten_metres["path_loss_db"] - one_metre["path_loss_db"] == pytest.approx(20.0)
    assert one_metre["noise_dbm"] == pytest.approx(-174 + 10 * np.log10(250e3) + 10, abs=0.1)
    assert tm.link_budget(3000.0)["bit_error_rate"] > tm.link_budget(1000.0)["bit_error_rate"] > 0
    with pytest.raises(ValueError):
        tm.link_budget(0.0)


def test_transmit_without_errors_is_identity_and_noisy_link_is_caught():
    packet = tm.encode_packet(REPORT)
    rng = np.random.default_rng(0)
    assert tm.transmit(packet, 0.0, rng) == packet
    rejected = 0
    for _ in range(200):
        received = tm.transmit(packet, 0.02, rng)
        if received == packet:
            continue
        with pytest.raises(ValueError):
            tm.decode_packet(received)
        rejected += 1
    assert rejected > 150                                      # most packets get at least one flip
    with pytest.raises(ValueError):
        tm.transmit(packet, 1.5, rng)
