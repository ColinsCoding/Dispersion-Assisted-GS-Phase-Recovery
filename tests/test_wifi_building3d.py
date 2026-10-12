"""Tests for dgs/wifi_building3d.py (+ opt-in MATLAB twin: RUN_MATLAB=1)."""
import os
import pathlib
import re
import shutil
import subprocess

import numpy as np
import pytest

from dgs.wifi_building3d import (
    building_grid, received_power_3d, floor_statistics, empirical_cdf, hallway_path,
    export_walls_csv, wall_loss_vectorized, walls_as_arrays,
)
from dgs.wifi_floorplan import default_floorplan, coverage_map, wall_loss_db

W, H, WALLS = default_floorplan()


def test_vectorized_matches_loop_reference():
    pts, fl, _, _ = building_grid(W, H, 1, res=0.4, rx_height=1.0)
    for f in (2.4e9, 5e9):
        P3 = received_power_3d((10, 3, 1.0), pts, fl, WALLS, f)
        _, _, P2 = coverage_map((10, 3), WALLS, W, H, f, res=0.4)
        assert np.max(np.abs(P3 - P2.ravel())) < 1e-9
    A, B, L = walls_as_arrays(WALLS)
    q = np.array([[18.0, 12.0], [1.0, 1.0]])
    assert np.allclose(wall_loss_vectorized((10, 3), q, A, B, L),
                       [wall_loss_db((10, 3), tuple(p), WALLS) for p in q])


def test_slabs_and_band_statistics():
    pts, fl, _, _ = building_grid(W, H, 3, res=0.5)
    P24 = received_power_3d((10, 3, 6.0), pts, fl, WALLS, 2.4e9)
    P5 = received_power_3d((10, 3, 6.0), pts, fl, WALLS, 5e9)
    s24, s5 = floor_statistics(P24, fl), floor_statistics(P5, fl)
    assert s24[1]["coverage %"] > s24[0]["coverage %"] and s24[1]["coverage %"] > s24[2]["coverage %"]
    assert all(a["median dBm"] > b["median dBm"] for a, b in zip(s24, s5))
    assert all(r["p10 dBm"] <= r["median dBm"] for r in s24)


def test_cdf_and_path():
    v, F = empirical_cdf([3, 1, 2])
    assert list(v) == [1, 2, 3] and F[-1] == 1.0
    path = hallway_path(10, floor=1)
    assert path.shape == (10, 3) and np.all(path[:, 2] == 3.5 + 2.5)


def test_validation():
    pts, fl, _, _ = building_grid(W, H, 1, res=1.0)
    with pytest.raises(ValueError):
        received_power_3d((1, 1, 1), pts, fl, WALLS, f_hz=0)


@pytest.mark.skipif(os.environ.get("RUN_MATLAB") != "1", reason="set RUN_MATLAB=1 (MATLAB startup is slow)")
def test_matlab_twin_matches(tmp_path):
    exe = shutil.which("matlab") or r"C:\Program Files\MATLAB\R2025b\bin\matlab.exe"
    root = pathlib.Path(__file__).resolve().parent.parent
    csv = export_walls_csv(tmp_path / "walls.csv", WALLS)
    out = subprocess.run([exe, "-batch", f"addpath('learn/matlab'); wifi_building3d('{csv.as_posix()}', [10 3 6], 2.4, '')"],
                         cwd=root, capture_output=True, text=True, timeout=600).stdout
    got = [float(x) for x in re.findall(r"coverage\s+([\d.]+) %", out)]
    pts, fl, _, _ = building_grid(W, H, 3, res=0.25)
    ref = [r["coverage %"] for r in floor_statistics(received_power_3d((10, 3, 6.0), pts, fl, WALLS, 2.4e9), fl)]
    assert np.allclose(got, ref, atol=0.01)
