from setuptools import find_packages, setup

package_name = "ros2_phycv"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="Photonics ML Lab",
    maintainer_email="dev@example.com",
    description="ROS2 node applying the PhyCV Phase Stretch Transform edge detector to camera images.",
    license="BSD-3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "pst_node = ros2_phycv.pst_node:main",
        ],
    },
)
