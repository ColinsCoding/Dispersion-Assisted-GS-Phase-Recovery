"""From 7 logic gates to the 7 OSI layers: how voltages become the internet, with checks at every step.

In plain words: a computer network is a stack of agreements. At the bottom, a wire carries
voltages; transistors switch those voltages and make logic gates; gates compute bits, add error
checks, and build frames; frames carry packets; packets carry the web page. Each layer only talks to
the layer right below it and hides the details. This module builds a small but real slice of the
stack so every claim can be checked:

  * The 7 basic gates (AND, OR, NOT, NAND, NOR, XOR, XNOR): truth tables, every gate rebuilt from
    NAND alone (NAND is universal), static-CMOS transistor counts, LVCMOS/LVTTL 3.3 V logic levels
    and noise margins, and the RC gate delay t_50 = R_on C_load ln 2 (derived from the axioms in
    dgs.circuit_rules).
  * XOR gates -> error detection: a parity tree built from XOR, and CRC-32 (the Ethernet frame check)
    computed bit-serially with the same shift-register + XOR logic the hardware uses, checked
    against zlib.crc32.
  * The OSI model as data (layer, unit of data, job, examples, physical reality) and a REAL
    encapsulation: a message -> UDP datagram -> IPv4 packet (with the RFC 791/1071 header checksum)
    -> Ethernet II frame (with CRC-32 FCS) -> bits -> Manchester-coded voltages on the wire, then all
    the way back up. decapsulate() verifies every checksum on the way.
  * Layer 1 waveforms: NRZ and Manchester line codes as functions of time (with a finite rise time),
    ready for dgs.scope_bench.Scope, plus a decoder that samples a waveform back into bits.

Numbers written from memory (check a datasheet / the IEEE 802.3 text before relying on them):
LVTTL-compatible 3.3 V levels V_IH = 2.0 V, V_IL = 0.8 V, V_OH = 2.4 V, V_OL = 0.4 V; static CMOS
counts (XOR/XNOR = 12 in the fully complementary form, fewer with pass-transistor designs);
10BASE-T uses Manchester coding at 10 Mb/s with about +/-2.5 V differential, with IEEE 802.3's
convention 0 = high-to-low and 1 = low-to-high mid-bit transition.

Connects to: dgs.logic_timing (gate delays, critical paths, adders), dgs.circuit_rules (RC delay),
dgs.scope_bench (put the wire on a scope), dgs.network_defense_sim (what attacks the upper layers).
"""
from __future__ import annotations

import math
import struct
import zlib
from itertools import product

import numpy as np

from dgs.logic_timing import GATE_FUNCS

GATES = ["AND", "OR", "NOT", "NAND", "NOR", "XOR", "XNOR"]

# ---------------------------------------------------------------- gates
CMOS_TRANSISTORS = {"NOT": 2, "NAND": 4, "NOR": 4, "AND": 6, "OR": 6, "XOR": 12, "XNOR": 12}   # 2-input static CMOS

LOGIC_LEVELS_3V3 = {"VCC": 3.3, "VOH": 2.4, "VOL": 0.4, "VIH": 2.0, "VIL": 0.8}   # LVTTL-compatible, from memory


def gate(name, *bits):
    name = name.upper()
    if name not in GATE_FUNCS:
        raise ValueError(f"unknown gate {name!r}; choose from {GATES}")
    for b in bits:
        if b not in (0, 1):
            raise ValueError(f"logic inputs must be 0 or 1; got {b!r}")
    if name == "NOT" and len(bits) != 1:
        raise ValueError("NOT takes exactly one input")
    return GATE_FUNCS[name](*bits)


def truth_table(name, n_inputs=None):
    n = 1 if name.upper() == "NOT" else (2 if n_inputs is None else n_inputs)
    return [(bits, gate(name, *bits)) for bits in product((0, 1), repeat=n)]


def _nand(a, b):
    return 1 - (a & b)


def nand_only(name):
    """(function, number of NAND gates) building `name` from 2-input NANDs only."""
    n = name.upper()
    if n == "NAND":
        return (lambda a, b: _nand(a, b)), 1
    if n == "NOT":
        return (lambda a: _nand(a, a)), 1
    if n == "AND":
        return (lambda a, b: _nand(_nand(a, b), _nand(a, b))), 2
    if n == "OR":
        return (lambda a, b: _nand(_nand(a, a), _nand(b, b))), 3
    if n == "NOR":
        def f(a, b):
            o = _nand(_nand(a, a), _nand(b, b))
            return _nand(o, o)
        return f, 4
    if n == "XOR":
        def f(a, b):
            m = _nand(a, b)
            return _nand(_nand(a, m), _nand(b, m))
        return f, 4
    if n == "XNOR":
        def f(a, b):
            m = _nand(a, b)
            x = _nand(_nand(a, m), _nand(b, m))
            return _nand(x, x)
        return f, 5
    raise ValueError(f"unknown gate {name!r}")


def noise_margins(levels=None):
    """NM_H = V_OH - V_IH and NM_L = V_IL - V_OL: how much noise a wire can pick up and still read right."""
    L = LOGIC_LEVELS_3V3 if levels is None else levels
    return {"NM_H": L["VOH"] - L["VIH"], "NM_L": L["VIL"] - L["VOL"]}


def read_logic(v, levels=None):
    """Voltage -> 1, 0, or None (forbidden zone between V_IL and V_IH)."""
    L = LOGIC_LEVELS_3V3 if levels is None else levels
    return 1 if v >= L["VIH"] else 0 if v <= L["VIL"] else None


def gate_delay(R_on, C_load):
    """50 % propagation delay of a gate driving C_load through its on-resistance: R C ln 2."""
    if R_on <= 0 or C_load <= 0:
        raise ValueError("R_on and C_load must be positive")
    return R_on * C_load * math.log(2)


# ---------------------------------------------------------------- error detection built from XOR
def parity_bit(bits):
    """Even parity via a chain of XOR gates: 1 if the data has an odd number of ones."""
    p = 0
    for b in bits:
        p = gate("XOR", p, b)
    return p


def crc32_bitwise(data: bytes) -> int:
    """CRC-32 (IEEE 802.3, reflected polynomial 0xEDB88320) one bit at a time: shift register + XOR."""
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0xEDB88320 if crc & 1 else 0)
    return crc ^ 0xFFFFFFFF


def bytes_to_bits(data: bytes):
    """LSB-first per byte, the order Ethernet puts bits on the wire."""
    return [(byte >> i) & 1 for byte in data for i in range(8)]


def bits_to_bytes(bits):
    if len(bits) % 8:
        raise ValueError("bit count must be a multiple of 8")
    return bytes(sum(bits[8 * j + i] << i for i in range(8)) for j in range(len(bits) // 8))


# ---------------------------------------------------------------- OSI model
OSI_LAYERS = [
    dict(n=7, name="Application", pdu="data", job="what the program means: a web request, a DNS query",
         examples="HTTP, DNS, SMTP, SSH", physical="bytes in a program's memory"),
    dict(n=6, name="Presentation", pdu="data", job="how the meaning is written down: text encoding, "
         "compression, encryption", examples="UTF-8, JPEG, TLS record encryption", physical="byte transformations"),
    dict(n=5, name="Session", pdu="data", job="keeping a conversation going: open, resume, close",
         examples="TLS sessions, RPC, NetBIOS", physical="state in software"),
    dict(n=4, name="Transport", pdu="segment / datagram", job="which program on the host; reliability "
         "and ordering if wanted", examples="TCP, UDP (ports)", physical="header bytes + checksum"),
    dict(n=3, name="Network", pdu="packet", job="which host, across many links; routing", examples="IPv4, IPv6, ICMP",
         physical="header bytes + header checksum (IPv4)"),
    dict(n=2, name="Data link", pdu="frame", job="the next hop on one shared link; MAC addresses; error "
         "detection", examples="Ethernet, Wi-Fi MAC", physical="frame + CRC-32 computed by XOR shift registers"),
    dict(n=1, name="Physical", pdu="bit / symbol", job="turn bits into something a wire carries", examples="10BASE-T "
         "Manchester, 1000BASE-T PAM-5, fiber on/off light", physical="volts on copper, photons in fibre"),
]


def internet_checksum(data: bytes) -> int:
    """RFC 1071 16-bit one's-complement checksum (IPv4 header, UDP)."""
    if len(data) % 2:
        data += b"\x00"
    total = sum(struct.unpack(f"!{len(data)//2}H", data))
    while total >> 16:
        total = (total & 0xFFFF) + (total >> 16)
    return ~total & 0xFFFF


def _ip(addr):
    parts = [int(x) for x in addr.split(".")]
    if len(parts) != 4 or any(not 0 <= p <= 255 for p in parts):
        raise ValueError(f"bad IPv4 address {addr!r}")
    return bytes(parts)


def _mac(addr):
    parts = bytes(int(x, 16) for x in addr.split(":"))
    if len(parts) != 6:
        raise ValueError(f"bad MAC address {addr!r}")
    return parts


def encapsulate(message, src_ip="192.168.1.10", dst_ip="192.168.1.20", src_port=40000, dst_port=5005,
                src_mac="02:00:00:00:00:01", dst_mac="02:00:00:00:00:02", ttl=64):
    """Wrap a message down the stack. Returns dict of the PDU at each layer (bytes) plus the wire bits."""
    payload = message.encode("utf-8") if isinstance(message, str) else bytes(message)   # L7 -> L6 (UTF-8)
    s_ip, d_ip = _ip(src_ip), _ip(dst_ip)
    ulen = 8 + len(payload)
    pseudo = s_ip + d_ip + struct.pack("!BBH", 0, 17, ulen)
    udp0 = struct.pack("!HHHH", src_port, dst_port, ulen, 0) + payload
    ucs = internet_checksum(pseudo + udp0) or 0xFFFF
    udp = struct.pack("!HHHH", src_port, dst_port, ulen, ucs) + payload                   # L4
    total = 20 + len(udp)
    hdr0 = struct.pack("!BBHHHBBH4s4s", 0x45, 0, total, 0x1234, 0x4000, ttl, 17, 0, s_ip, d_ip)
    hdr = hdr0[:10] + struct.pack("!H", internet_checksum(hdr0)) + hdr0[12:]
    ip = hdr + udp                                                                        # L3
    body = _mac(dst_mac) + _mac(src_mac) + struct.pack("!H", 0x0800) + ip
    body += b"\x00" * max(0, 60 - len(body))                                              # pad to 64 B minimum with FCS
    frame = body + struct.pack("<I", zlib.crc32(body))                                     # L2 (FCS)
    preamble = b"\x55" * 7 + b"\xd5"
    return {"L7": message, "L6": payload, "L4": udp, "L3": ip, "L2": frame,
            "L1_bits": bytes_to_bits(preamble + frame)}


def decapsulate(frame: bytes):
    """Walk back up the stack, verifying each check. Returns dict with the message and check results."""
    body, fcs = frame[:-4], struct.unpack("<I", frame[-4:])[0]
    checks = {"L2 CRC-32": zlib.crc32(body) == fcs}
    if struct.unpack("!H", body[12:14])[0] != 0x0800:
        raise ValueError("not an IPv4 frame")
    ip = body[14:]
    ihl = (ip[0] & 0x0F) * 4
    total = struct.unpack("!H", ip[2:4])[0]
    checks["L3 IPv4 header checksum"] = internet_checksum(ip[:ihl]) == 0
    udp = ip[ihl:total]
    ulen = struct.unpack("!H", udp[4:6])[0]
    pseudo = ip[12:16] + ip[16:20] + struct.pack("!BBH", 0, 17, ulen)
    checks["L4 UDP checksum"] = internet_checksum(pseudo + udp[:ulen]) == 0
    payload = udp[8:ulen]
    return {"message": payload.decode("utf-8"), "checks": checks,
            "src_port": struct.unpack("!H", udp[0:2])[0], "dst_port": struct.unpack("!H", udp[2:4])[0],
            "ttl": ip[8]}


# ---------------------------------------------------------------- layer 1: line codes as voltages
def manchester_chips(bits):
    """IEEE 802.3 convention: 0 -> (high, low), 1 -> (low, high). Two half-bit chips per bit, as +1/-1."""
    out = []
    for b in bits:
        if b not in (0, 1):
            raise ValueError("bits must be 0/1")
        out += [-1, 1] if b else [1, -1]
    return out


def line_waveform(bits, bitrate, code="manchester", v_high=2.5, v_low=-2.5, rise_time=None, t0=0.0):
    """Return f(t) -> volts for the coded bit stream (single-pole edges if rise_time is given)."""
    if bitrate <= 0:
        raise ValueError("bitrate must be positive")
    code = code.lower()
    if code == "manchester":
        chips, chip_t = manchester_chips(bits), 0.5 / bitrate
    elif code == "nrz":
        chips, chip_t = [1 if b else -1 for b in bits], 1.0 / bitrate
    else:
        raise ValueError("code must be 'manchester' or 'nrz'")
    levels = np.where(np.array(chips) > 0, v_high, v_low).astype(float)
    tau = None if rise_time is None else rise_time / math.log(9)        # 10-90 % rise of a single pole
    T = len(chips) * chip_t

    def f(t):
        t = np.asarray(t, float) - t0
        idx = np.clip((t // chip_t).astype(int), 0, len(chips) - 1)
        if tau is None:
            v = levels[idx]
        else:
            # sum of step responses: each level change starts an exponential approach
            v = np.full_like(t, levels[0])
            changes = np.nonzero(np.diff(levels))[0] + 1
            for c in changes:
                tc = c * chip_t
                dv = levels[c] - levels[c - 1]
                v = v + np.where(t >= tc, dv * (1 - np.exp(-np.clip(t - tc, 0, None) / tau)), 0.0)
        return np.where((t >= 0) & (t < T), v, 0.0)
    f.duration, f.chip_time = T, chip_t
    return f


def manchester_decode(f, n_bits, bitrate, t0=0.0):
    """Sample each half-bit at its 3/4 point (after the edges settle): low->high = 1, high->low = 0."""
    T = 1.0 / bitrate
    bits = []
    for i in range(n_bits):
        a = float(f(t0 + i * T + 0.375 * T))
        b = float(f(t0 + i * T + 0.875 * T))
        bits.append(1 if b > a else 0)
    return bits
