"""
scene.py — Scene graph node + transform hierarchy.
All world objects attach here. Renderer walks the tree each frame.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class Transform:
    position : np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation : np.ndarray = field(default_factory=lambda: np.zeros(3))  # Euler XYZ deg
    scale    : np.ndarray = field(default_factory=lambda: np.ones(3))

    def matrix(self) -> np.ndarray:
        """4x4 TRS matrix."""
        rx, ry, rz = np.radians(self.rotation)
        Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
        Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
        Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
        R  = Rz @ Ry @ Rx
        M  = np.eye(4)
        M[:3,:3] = R * self.scale
        M[:3, 3] = self.position
        return M


class SceneNode:
    def __init__(self, name: str):
        self.name      : str              = name
        self.transform : Transform        = Transform()
        self.parent    : Optional[SceneNode] = None
        self.children  : List[SceneNode]  = []
        self.mesh      = None
        self.material  = None

    def add_child(self, node: SceneNode) -> SceneNode:
        node.parent = self
        self.children.append(node)
        return node

    def world_matrix(self) -> np.ndarray:
        if self.parent is None:
            return self.transform.matrix()
        return self.parent.world_matrix() @ self.transform.matrix()

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()


class Scene:
    def __init__(self):
        self.root   = SceneNode("root")
        self.lights : list = []
        self.camera = None

    def add(self, node: SceneNode) -> SceneNode:
        return self.root.add_child(node)

    def all_nodes(self):
        return list(self.root.walk())
