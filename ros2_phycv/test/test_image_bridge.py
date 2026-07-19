"""Unit tests for the ROS-independent image bridge (NumPy only)."""
from __future__ import annotations

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ros2_phycv.image_bridge import decode_to_gray, encode_mono8


def test_mono8_roundtrip() -> None:
    h, w = 8, 5
    gray = (np.arange(h * w).reshape(h, w) % 256).astype(np.uint8)
    decoded = decode_to_gray(gray.tobytes(), h, w, "mono8")
    assert decoded.shape == (h, w)
    assert np.allclose(decoded, gray / 255.0)


def test_rgb8_luma() -> None:
    # a pure-green pixel -> luma 0.587
    data = bytes([0, 255, 0])
    gray = decode_to_gray(data, 1, 1, "rgb8")
    assert abs(float(gray[0, 0]) - 0.587) < 1e-3


def test_bgr8_channel_order() -> None:
    # bgr8 pure-red pixel is bytes (0, 0, 255) -> luma 0.299
    gray = decode_to_gray(bytes([0, 0, 255]), 1, 1, "bgr8")
    assert abs(float(gray[0, 0]) - 0.299) < 1e-3


def test_encode_mono8_shape_and_values() -> None:
    arr = np.array([[0.0, 1.0], [0.5, 0.25]])
    data, h, w = encode_mono8(arr)
    assert (h, w) == (2, 2)
    assert np.frombuffer(data, dtype=np.uint8)[0] == 0
    assert np.frombuffer(data, dtype=np.uint8)[1] == 255


def test_unsupported_encoding_raises() -> None:
    try:
        decode_to_gray(b"\x00", 1, 1, "yuv422")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unsupported encoding")
