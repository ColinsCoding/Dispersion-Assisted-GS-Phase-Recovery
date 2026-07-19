# ros2_phycv

A ROS2 (`ament_python`) package that applies the **PhyCV Phase Stretch Transform** edge
detector to a live camera stream: subscribe to a `sensor_msgs/Image` topic, run PST, and
publish the binary edge map as a `mono8` image.

```
sensor_msgs/Image  ──▶  PstNode  ──▶  sensor_msgs/Image
 /camera/image_raw      (PST)         /phycv/edges  (mono8 edges)
```

## Package layout

```
ros2_phycv/
  package.xml            ament_python manifest (rclpy, sensor_msgs deps)
  setup.py / setup.cfg   entry point: pst_node = ros2_phycv.pst_node:main
  resource/ros2_phycv    ament resource marker
  ros2_phycv/
    pst_core.py          Phase Stretch Transform (pure NumPy, ROS-free)  ← unit-tested
    image_bridge.py      sensor_msgs/Image <-> NumPy (mono8/rgb8/bgr8)   ← unit-tested
    pst_node.py          the rclpy node (imports the two modules above)
  test/                  pytest for pst_core and image_bridge
```

## Build & run (inside a ROS2 workspace)

```bash
# put this package under <ws>/src/ros2_phycv, then:
cd <ws>
colcon build --packages-select ros2_phycv
source install/setup.bash
ros2 run ros2_phycv pst_node --ros-args \
    -p input_topic:=/camera/image_raw \
    -p output_topic:=/phycv/edges \
    -p strength:=4.0 -p warp:=15.0 -p sigma_lpf:=0.2 -p threshold:=0.3
# view the result:
ros2 run rqt_image_view rqt_image_view /phycv/edges
```

## Parameters

| name | default | meaning |
|------|---------|---------|
| `input_topic`  | `/camera/image_raw` | image topic to subscribe to |
| `output_topic` | `/phycv/edges`      | edge-map topic to publish |
| `strength`     | `4.0`  | PST phase strength S |
| `warp`         | `15.0` | PST warp W |
| `sigma_lpf`    | `0.2`  | localization low-pass width |
| `threshold`    | `0.3`  | edge threshold (fraction of peak phase) |

## Testing

The **algorithm** (PST core) and the **image bridge** are pure NumPy and unit-tested
without any ROS install:

```bash
python -m pytest ros2_phycv/test -q
```

> **Note.** The `pst_node.py` rclpy plumbing requires a sourced ROS2 environment
> (Humble/Jazzy: `rclpy`, `sensor_msgs`) to launch — that runtime is not needed to run
> the tests, which cover the processing that the node performs. `pst_core` mirrors the
> verified `physics_repo/notebooks/phycv_phase_stretch_transform.ipynb`.
```
