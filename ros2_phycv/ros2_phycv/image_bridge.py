"""Convert between ROS `sensor_msgs/Image` byte buffers and NumPy arrays.

Purpose:
    A minimal, dependency-free image bridge (no cv_bridge required) so the node works
    with a plain NumPy/rclpy install. Kept ROS-agnostic -- functions take raw fields --
    so they are unit-testable without a ROS2 runtime.

Supported encodings: mono8, rgb8, bgr8.
Limitations:
    - 8-bit encodings only; extend for mono16 / float32 as needed.
"""
from __future__ import annotations

import numpy as np

__all__ = ["decode_to_gray", "encode_mono8"]

_RGB_WEIGHTS = np.array([0.299, 0.587, 0.114])  # ITU-R BT.601 luma


def decode_to_gray(data: bytes, height: int, width: int, encoding: str) -> np.ndarray:
    """Decode a ROS image byte buffer to a float grayscale array in [0, 1]."""
    buf = np.frombuffer(data, dtype=np.uint8)
    enc = encoding.lower()
    if enc == "mono8":
        gray = buf.reshape(height, width).astype(float)
    elif enc in ("rgb8", "bgr8"):
        img = buf.reshape(height, width, 3).astype(float)
        weights = _RGB_WEIGHTS if enc == "rgb8" else _RGB_WEIGHTS[::-1]
        gray = img @ weights
    else:
        raise ValueError(f"unsupported encoding: {encoding!r} (want mono8/rgb8/bgr8)")
    return gray / 255.0


def encode_mono8(gray01: np.ndarray) -> tuple[bytes, int, int]:
    """Encode a float [0, 1] grayscale array to a mono8 byte buffer; return (data, h, w)."""
    arr = np.clip(np.asarray(gray01, dtype=float), 0.0, 1.0)
    u8 = np.round(arr * 255).astype(np.uint8)
    h, w = u8.shape
    return u8.tobytes(), h, w
