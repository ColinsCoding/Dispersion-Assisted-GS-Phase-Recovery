"""RF telemetry model: a separate identification and reporting layer.

RF tagging here is NOT a way of sensing particles and does not replace the
optical scattering physics. The tag ID names a SAMPLE CONTAINER or instrument
channel on a lab bench (a cuvette, a vial, a flow cell). The radio only
carries the result the optical estimator already produced:

    tag ID, timestamp, particle estimate, confidence interval.

It is not a tracker: there is no position, no person, and no micron-scale
"RF particle". A particle a few micrometres across cannot act as an antenna
at radio wavelengths (centimetres), so nothing here tags individual particles.

Contents
    packet format with CRC-16          encode_packet / decode_packet
    link budget (Friis, free space)    link_budget
    bit errors on the link             transmit
Behavioural simulation only. A real transmitter must use a certified radio
module and follow the local radio regulations.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

MAGIC = b"SP"
VERSION = 1
# little-endian: magic(2s) version(B) flags(B) tag_id(I) timestamp_ms(Q)
#   dia_nm(I) dia_lo_nm(I) dia_hi_nm(I) npar_milli(H) confidence_pct(B) pad(B)
#   concentration_per_m3(f) diffusion_m2_per_s(f)
_BODY = struct.Struct("<2sBBIQIIIHBxff")
PACKET_BYTES = _BODY.size + 2          # + CRC-16

FLAG_AMBIGUOUS = 0x01      # confidence set is not one contiguous diameter range
FLAG_CLIPPED = 0x02        # an ADC sample saturated
FLAG_POOR_FIT = 0x04       # chi-square says the model does not describe the data
FLAG_NO_MOTION = 0x08      # diffusion field not measured (set to NaN)


def crc16_ccitt(data: bytes, init=0xFFFF):
    """CRC-16/CCITT-FALSE (polynomial 0x1021), bit by bit."""
    crc = init
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


@dataclass(frozen=True)
class TagReport:
    """One telemetry record. SI units; diameters rounded to 1 nm on the wire."""
    tag_id: int                 # identifies the sample container (0 .. 2^32 - 1)
    timestamp_ms: int           # milliseconds since the instrument epoch
    dia: float                  # [m]
    dia_ci: tuple               # [m]
    npar: float
    confidence_level: float     # e.g. 0.95
    concentration: float        # [particles/m^3]
    diffusion: float = float("nan")   # [m^2/s]; NaN = not measured
    flags: int = 0


def encode_packet(report: TagReport) -> bytes:
    """Serialize a report into PACKET_BYTES bytes, CRC appended."""
    if not 0 <= report.tag_id < 2 ** 32:
        raise ValueError("tag_id must fit in 32 bits")
    if not 0 <= report.timestamp_ms < 2 ** 64:
        raise ValueError("timestamp_ms must fit in 64 bits")
    lo, hi = report.dia_ci
    nm = [int(round(v * 1e9)) for v in (report.dia, lo, hi)]
    if not (0 <= nm[1] <= nm[0] <= nm[2] < 2 ** 32):
        raise ValueError("need 0 <= ci_low <= dia <= ci_high (in nm, 32-bit)")
    if not 0 < report.npar < 65.535:
        raise ValueError("npar out of range for the 0.001-resolution field")
    if not 0 < report.confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    flags = report.flags | (FLAG_NO_MOTION if np.isnan(report.diffusion) else 0)
    body = _BODY.pack(MAGIC, VERSION, flags & 0xFF, report.tag_id, report.timestamp_ms,
                      nm[0], nm[1], nm[2], int(round(report.npar * 1000)),
                      int(round(report.confidence_level * 100)),
                      report.concentration, report.diffusion)
    return body + struct.pack("<H", crc16_ccitt(body))


def decode_packet(packet: bytes) -> TagReport:
    """Parse a packet. Raises ValueError on wrong length, magic, version or CRC."""
    if len(packet) != PACKET_BYTES:
        raise ValueError(f"packet must be {PACKET_BYTES} bytes, got {len(packet)}")
    body, (crc,) = packet[:-2], struct.unpack("<H", packet[-2:])
    if crc16_ccitt(body) != crc:
        raise ValueError("CRC mismatch: packet corrupted")
    magic, version, flags, tag_id, ts, dia, lo, hi, npar, conf, conc, diff = _BODY.unpack(body)
    if magic != MAGIC or version != VERSION:
        raise ValueError("not a Scarlett packet (bad magic or version)")
    return TagReport(tag_id, ts, dia * 1e-9, (lo * 1e-9, hi * 1e-9), npar / 1000,
                     conf / 100, conc, diff, flags)


def link_budget(distance, frequency=2.44e9, tx_power_dbm=0.0, tx_gain_dbi=0.0,
                rx_gain_dbi=0.0, bit_rate=250e3, noise_figure_db=10.0,
                temperature=290.0):
    """Free-space link budget (Friis). Returns a dict in dB units.

        path loss  = 20 log10(4 pi d f / c)
        P_rx       = P_tx + G_tx + G_rx - path loss
        noise      = k T * bit_rate, plus the receiver noise figure
        Eb/N0      = P_rx - noise      (noise bandwidth taken equal to the bit rate)
    Indoor multipath, antenna detuning and interference are NOT modelled;
    treat the margin as optimistic.
    """
    if distance <= 0 or frequency <= 0 or bit_rate <= 0:
        raise ValueError("distance, frequency and bit_rate must be positive")
    c = 299_792_458.0
    path_loss = 20 * np.log10(4 * np.pi * distance * frequency / c)
    p_rx = tx_power_dbm + tx_gain_dbi + rx_gain_dbi - path_loss
    noise = 10 * np.log10(1.380649e-23 * temperature * bit_rate / 1e-3) + noise_figure_db
    ebn0_db = p_rx - noise
    ebn0 = 10 ** (ebn0_db / 10)
    ber = 0.5 * np.exp(-ebn0 / 2)          # non-coherent binary FSK
    return {"path_loss_db": float(path_loss), "rx_power_dbm": float(p_rx),
            "noise_dbm": float(noise), "ebn0_db": float(ebn0_db), "bit_error_rate": float(ber),
            "packet_error_rate": float(1 - (1 - ber) ** (8 * PACKET_BYTES))}


def transmit(packet: bytes, bit_error_rate, rng):
    """Flip each bit independently with probability `bit_error_rate`."""
    if not 0 <= bit_error_rate <= 1:
        raise ValueError("bit_error_rate must be in [0, 1]")
    bits = np.unpackbits(np.frombuffer(packet, dtype=np.uint8))
    flips = rng.random(bits.size) < bit_error_rate
    return np.packbits(bits ^ flips).tobytes()
