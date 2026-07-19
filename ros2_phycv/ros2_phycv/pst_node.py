"""ROS2 node: subscribe to a camera image, publish the PhyCV PST edge map.

Purpose:
    Wrap the pure-NumPy `pst_core` in an rclpy node. Subscribes to a
    `sensor_msgs/Image` topic, runs the Phase Stretch Transform, and publishes the
    binary edge map as a `mono8` image. Parameters expose the PST knobs.

Run (inside a sourced ROS2 workspace):
    ros2 run ros2_phycv pst_node --ros-args \
        -p input_topic:=/camera/image_raw -p output_topic:=/phycv/edges -p strength:=4.0

Note:
    Requires a ROS2 environment (rclpy, sensor_msgs). The image processing itself is
    verified by the unit tests on `pst_core` / `image_bridge`, which need only NumPy.
"""
from __future__ import annotations

import numpy as np

try:  # ROS2 is optional at import time so the module can be inspected/tested without it
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image

    _HAVE_ROS = True
except ImportError:  # pragma: no cover
    _HAVE_ROS = False
    Node = object  # type: ignore

from ros2_phycv.image_bridge import decode_to_gray, encode_mono8
from ros2_phycv.pst_core import PstParams, phase_stretch_transform
from ros2_phycv.pst_rom import build_radial_rom, phase_stretch_transform_rom


class PstNode(Node):  # type: ignore[misc]
    """Applies PST to each incoming image and republishes the edge map.

    With ``use_rom`` (default true) the node runs the **ROM-quantized** transform -- the
    signed fixed-point radial coefficient ROM built once at start-up -- so its output
    bit-matches the FPGA block-RAM data path. Set ``use_rom:=false`` for the exact float
    kernel. Kernel parameters are read once at start-up (the ROM is fixed, like hardware);
    change them and relaunch, as you would reload an FPGA ROM.
    """

    def __init__(self) -> None:
        super().__init__("phycv_pst")
        self.declare_parameter("input_topic", "/camera/image_raw")
        self.declare_parameter("output_topic", "/phycv/edges")
        self.declare_parameter("strength", 4.0)
        self.declare_parameter("warp", 15.0)
        self.declare_parameter("sigma_lpf", 0.2)
        self.declare_parameter("threshold", 0.3)
        self.declare_parameter("use_rom", True)
        self.declare_parameter("rom_bits", 8)
        self.declare_parameter("rom_bins", 256)

        self._params = PstParams(
            strength=float(self.get_parameter("strength").value),
            warp=float(self.get_parameter("warp").value),
            sigma_lpf=float(self.get_parameter("sigma_lpf").value),
            threshold=float(self.get_parameter("threshold").value),
        )
        self._use_rom = bool(self.get_parameter("use_rom").value)
        self._rom = (
            build_radial_rom(
                self._params,
                n_bins=int(self.get_parameter("rom_bins").value),
                n_bits=int(self.get_parameter("rom_bits").value),
            )
            if self._use_rom
            else None
        )

        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value
        self._subscription = self.create_subscription(Image, input_topic, self._on_image, 10)
        self._publisher = self.create_publisher(Image, output_topic, 10)
        mode = f"ROM-quantized ({self._rom.n_bits}-bit x {self._rom.n_bins} bins)" if self._use_rom else "float kernel"
        self.get_logger().info(f"PhyCV PST node [{mode}]: {input_topic} -> {output_topic}")

    def _on_image(self, msg: "Image") -> None:
        gray = decode_to_gray(bytes(msg.data), msg.height, msg.width, msg.encoding)
        if self._use_rom:
            result = phase_stretch_transform_rom(gray, self._params, self._rom)
        else:
            result = phase_stretch_transform(gray, self._params)
        edges = result.edges.astype(np.float64)
        data, height, width = encode_mono8(edges)

        out = Image()
        out.header = msg.header
        out.height, out.width = height, width
        out.encoding = "mono8"
        out.is_bigendian = 0
        out.step = width
        out.data = list(data)
        self._publisher.publish(out)


def main(args: list[str] | None = None) -> None:
    """Entry point: spin the PST node until interrupted."""
    if not _HAVE_ROS:  # pragma: no cover
        raise RuntimeError("rclpy/sensor_msgs not found. Source a ROS2 (Humble/Jazzy) environment first.")
    rclpy.init(args=args)
    node = PstNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
