"""Runs INSIDE Blender (bpy is not available in a normal Python interpreter):
    blender --background --python tests/_blender_test_procedural_architecture.py
Invoked as a subprocess by tests/test_procedural_architecture_blender.py,
which checks this script's stdout for the final pass/fail line -- the
same role a notebook's final grading cell plays elsewhere in this repo,
just via subprocess instead of nbconvert since bpy can't be imported
directly into the normal test process."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs.procedural_architecture_blender import (
    generate_building_mesh, uv_unwrap, build_style_comparison, STYLE_PRESETS, clear_scene,
)

checks = []


def check(label, condition):
    checks.append((label, bool(condition)))
    print(f"{'PASS' if condition else 'FAIL'}  --  {label}")


# 1. the boolean window-cutting actually changes the geometry -- the exact
# regression this module hit during development (a scale/depsgraph bug
# silently left the building as an unmodified 8-vert, 6-face cube)
clear_scene()
building = generate_building_mesh("test_building", width=4.0, depth=3.0, height=8.0,
                                   floors=5, bays_x=3)
n_verts, n_faces = len(building.data.vertices), len(building.data.polygons)
check("window-cut building has far more geometry than a plain 8-vert/6-face cube",
      n_verts > 20 and n_faces > 10)

# 2. the mesh generation is deterministic: same params -> same vertex/face count
clear_scene()
building2 = generate_building_mesh("test_building2", width=4.0, depth=3.0, height=8.0,
                                    floors=5, bays_x=3)
check("identical generation params produce identical vertex/face counts",
      len(building2.data.vertices) == n_verts and len(building2.data.polygons) == n_faces)

# 3. different params produce different geometry (not a hardcoded/fixed mesh).
# height is scaled up with floor count to keep floor spacing > window height
# (floors=8 on the SAME height=8 as the baseline would make floor spacing
# 1.0 < window_h=1.1, so adjacent windows overlap and merge into fewer,
# larger cavities -- a real, correct CSG interaction, not a bug, but it
# would make "more floors -> more geometry" false for a reason unrelated to
# what this check is actually testing)
clear_scene()
building3 = generate_building_mesh("test_building3", width=4.0, depth=3.0, height=14.0,
                                    floors=8, bays_x=5)
check("more floors/bays (with proportional height, no window overlap) produces more geometry",
      len(building3.data.vertices) > n_verts)

# 4. UV unwrapping succeeds and produces a real UV layer
clear_scene()
building4 = generate_building_mesh("test_building4", width=4.0, depth=3.0, height=8.0,
                                    floors=4, bays_x=3)
uv_ok = uv_unwrap(building4)
check("UV unwrap produces an active UV layer", uv_ok)
uv_layer = building4.data.uv_layers.active
uv_coords = [tuple(loop.uv) for loop in uv_layer.data]
check("UV coordinates are non-degenerate (more than one distinct UV position)",
      len(set(uv_coords)) > 1)

# 5. THE core "commitment to the same geometry" claim: every style copy has
# IDENTICAL vertex and face counts -- styles differ only in material
clear_scene()
copies, base_verts, base_faces = build_style_comparison(
    styles=tuple(STYLE_PRESETS.keys()), width=4.0, depth=3.0, height=8.0, floors=5, bays_x=3)
vert_counts = [len(o.data.vertices) for o in copies]
face_counts = [len(o.data.polygons) for o in copies]
check("all style copies share IDENTICAL vertex counts (same underlying geometry)",
      len(set(vert_counts)) == 1)
check("all style copies share IDENTICAL face counts (same underlying geometry)",
      len(set(face_counts)) == 1)
check("style copies actually have the real window-cut geometry, not a plain cube",
      vert_counts[0] > 20)

# 6. each style copy has its OWN distinct material (styles actually differ visually)
materials = [o.data.materials[0] for o in copies]
colors = [tuple(m.node_tree.nodes.get("Principled BSDF").inputs["Base Color"].default_value)
          for m in materials]
check("every style copy has a distinct base color (styles are visually different)",
      len(set(colors)) == len(colors))

failures = [label for label, ok in checks if not ok]
print(f"\n{len(checks) - len(failures)}/{len(checks)} checks passed")
if failures:
    print("FAILURES:", "; ".join(failures))
    sys.exit(1)
else:
    print("ALL_CHECKS_PASSED")
