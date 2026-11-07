import math
import zlib

import numpy as np
import pytest

from dgs import gates_to_internet as g


@pytest.mark.parametrize("name", g.GATES)
def test_nand_universality(name):
    f, n = g.nand_only(name)
    assert all(f(*bits) == out for bits, out in g.truth_table(name))
    assert n == {"NAND": 1, "NOT": 1, "AND": 2, "OR": 3, "NOR": 4, "XOR": 4, "XNOR": 5}[name]


def test_truth_tables():
    assert [o for _, o in g.truth_table("XOR")] == [0, 1, 1, 0]
    assert [o for _, o in g.truth_table("NOR")] == [1, 0, 0, 0]
    assert [o for _, o in g.truth_table("NOT")] == [1, 0]
    with pytest.raises(ValueError):
        g.gate("AND", 1, 2)
    with pytest.raises(ValueError):
        g.gate("MAYBE", 1, 0)


def test_levels_and_delay():
    nm = g.noise_margins()
    assert nm["NM_H"] == pytest.approx(0.4) and nm["NM_L"] == pytest.approx(0.4)
    assert g.read_logic(3.0) == 1 and g.read_logic(0.2) == 0 and g.read_logic(1.4) is None
    assert g.gate_delay(1e3, 10e-12) == pytest.approx(1e-8 * math.log(2))


def test_parity_and_crc():
    rng = np.random.default_rng(3)
    data = bytes(rng.integers(0, 256, 40, dtype=np.uint8))
    assert g.crc32_bitwise(b"123456789") == 0xCBF43926                # the standard CRC-32 check value
    assert g.crc32_bitwise(data) == zlib.crc32(data)
    bits = g.bytes_to_bits(data)
    assert g.bits_to_bytes(bits) == data
    assert g.parity_bit(bits) == sum(bits) % 2
    flipped = bytearray(data); flipped[7] ^= 0b00000100
    assert zlib.crc32(bytes(flipped)) != zlib.crc32(data)


def test_osi_round_trip_and_checks():
    assert [L["n"] for L in g.OSI_LAYERS] == [7, 6, 5, 4, 3, 2, 1]
    fr = g.encapsulate("voltages -> internet")
    assert len(fr["L2"]) >= 64
    out = g.decapsulate(fr["L2"])
    assert out["message"] == "voltages -> internet" and all(out["checks"].values())
    bad = bytearray(fr["L2"]); bad[30] ^= 1
    assert not g.decapsulate(bytes(bad))["checks"]["L2 CRC-32"]


def test_internet_checksum_rfc1071_example():
    # RFC 1071 section 3 example words 0001 f203 f4f5 f6f7 sum to ddf2 -> checksum 220d
    assert g.internet_checksum(bytes.fromhex("0001f203f4f5f6f7")) == 0x220D


@pytest.mark.parametrize("rise", [None, 5e-9])
def test_manchester_round_trip(rise):
    bits = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]
    f = g.line_waveform(bits, 10e6, "manchester", rise_time=rise)
    assert g.manchester_decode(f, len(bits), 10e6) == bits
    assert f.duration == pytest.approx(len(bits) / 10e6)
    nrz = g.line_waveform(bits, 10e6, "nrz")
    assert nrz(0.5e-7) == 2.5 and nrz(1.5e-7) == -2.5


def test_manchester_has_no_dc_and_an_edge_every_bit():
    bits = [0] * 50 + [1] * 50
    chips = np.array(g.manchester_chips(bits))
    assert chips.sum() == 0
    assert all(chips[2 * i] != chips[2 * i + 1] for i in range(len(bits)))
