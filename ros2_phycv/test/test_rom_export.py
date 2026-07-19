"""Tests for exporting the coefficient ROM as .coe and VHDL."""
from __future__ import annotations

import re
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ros2_phycv.pst_rom import build_radial_rom
from ros2_phycv.rom_export import to_coe, to_packed_coe, to_vhdl, write_rom_artifacts


def _to_signed(byte: int) -> int:
    return byte - 256 if byte >= 128 else byte


def _coe_values(text: str) -> list[int]:
    vec = text.split("memory_initialization_vector=")[1]
    return [_to_signed(int(tok, 16)) for tok in re.findall(r"[0-9A-Fa-f]{2}", vec)]


def test_coe_header_and_values_match_table() -> None:
    rom = build_radial_rom()
    coe = to_coe(rom.re, comment="real")
    assert "memory_initialization_radix=16;" in coe
    assert _coe_values(coe) == list(rom.re)                      # two's-complement round-trip


def test_packed_coe_recovers_both_channels() -> None:
    rom = build_radial_rom()
    packed = to_packed_coe(rom)
    words = re.findall(r"[0-9A-Fa-f]{4}", packed.split("memory_initialization_vector=")[1])
    re_vals = [_to_signed(int(w, 16) >> 8) for w in words]
    im_vals = [_to_signed(int(w, 16) & 0xFF) for w in words]
    assert re_vals == list(rom.re) and im_vals == list(rom.im)


def test_vhdl_contains_all_values() -> None:
    rom = build_radial_rom()
    vhdl = to_vhdl(rom, name="pst_coeff_rom")
    assert "entity pst_coeff_rom" in vhdl and "rising_edge(clk)" in vhdl
    hexes = [_to_signed(int(h, 16)) for h in re.findall(r'x"([0-9A-Fa-f]{2})"', vhdl)]
    assert hexes[:256] == list(rom.re) and hexes[256:512] == list(rom.im)


def test_write_rom_artifacts_creates_files(tmp_path) -> None:
    rom = build_radial_rom()
    paths = write_rom_artifacts(rom, tmp_path, name="pst_coeff_rom")
    for key in ("re_coe", "im_coe", "packed_coe", "vhdl"):
        assert paths[key].exists() and paths[key].stat().st_size > 0
    assert _coe_values(paths["re_coe"].read_text()) == list(rom.re)
