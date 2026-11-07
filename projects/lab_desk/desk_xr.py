"""
desk_xr.py - export the lab desk as a 3D scene for Apple Vision Pro (USDZ) and Windows VR (glTF/GLB).

In plain words: this rebuilds the desk at real size (a 1.6 m x 0.8 m desk, 75 cm high) with the laser,
gas cell, splitter, fiber spools, photodiodes, oscilloscope and laptop as 3D objects. The scope and
laptop screens are textured with whatever the 2D desk is showing right now, the fibers and probe
cables hang as catenaries, and the desk top gets a procedural wood texture.

  lab_desk.usdz -> visionOS: AirDrop or Files to the Vision Pro, opens in Quick Look (place it on a
                   real table); also loads in Reality Composer Pro / RealityKit.
  lab_desk.glb  -> Windows VR (HTC Vive via SteamVR/OpenXR): open in Blender, Unity, Unreal or Godot.

Geometry is computed in float64 and written as float32, which is what USDZ/RealityKit and glTF use
for vertex positions (sub-micrometre resolution at desk scale). No third-party packages: USDZ is an
uncompressed zip of a USDA layer plus PNG textures with every file 64-byte aligned, per the USDZ spec.
"""
import io
import json
import math
import os
import re
import struct
import zlib

import numpy as np
import pygame

DESK_W, DESK_D, DESK_H = 1.6, 0.8, 0.75      # metres
SCREEN_W, SCREEN_H = 1280, 800               # the 2D desk's pixel space


# ---------------------------------------------------------------- mesh scene
class Scene:
    def __init__(self, name):
        self.name, self.meshes, self.materials = name, [], {}

    def material(self, name, color=(1, 1, 1), png=None, roughness=0.8, metallic=0.0, emissive=None):
        self.materials.setdefault(name, dict(color=color, png=png, roughness=roughness, metallic=metallic,
                                             emissive=emissive))
        return name

    def add(self, name, P, N, UV, I, mat):
        self.meshes.append((name, np.asarray(P, np.float64), np.asarray(N, np.float64), np.asarray(UV, np.float64),
                            np.asarray(I, np.uint32), mat))

    def box(self, name, x0, y0, z0, x1, y1, z1, mat, uv=1.0, top_mat=None):
        faces = [((1, 0, 0), [(x1, y0, z1), (x1, y0, z0), (x1, y1, z0), (x1, y1, z1)]),
                 ((-1, 0, 0), [(x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)]),
                 ((0, -1, 0), [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)]),
                 ((0, 0, 1), [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]),
                 ((0, 0, -1), [(x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0)])]
        top = ((0, 1, 0), [(x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (x0, y1, z0)])
        for group, m in ((faces, mat), ([top], top_mat or mat)):
            P, N, U, I = [], [], [], []
            for n, quad in group:
                b = len(P)
                a_, b_, _, d_ = (np.array(v, float) for v in quad)
                w, h = np.linalg.norm(b_ - a_) * uv, np.linalg.norm(d_ - a_) * uv
                P += quad
                N += [n] * 4
                U += [(0, h), (w, h), (w, 0), (0, 0)] if m == mat or top_mat is None else [(0, 1), (1, 1), (1, 0), (0, 0)]
                I += [b, b + 1, b + 2, b, b + 2, b + 3]
            self.add(name if m == mat else name + "_top", P, N, U, I, m)

    def quad(self, name, corners, normal, mat):
        """Textured rectangle (corners in order bottom-left, bottom-right, top-right, top-left)."""
        self.add(name, corners, [normal] * 4, [(0, 1), (1, 1), (1, 0), (0, 0)], [0, 1, 2, 0, 2, 3], mat)

    def cylinder(self, name, c, r, h, mat, axis="y", sides=32):
        """Closed cylinder centred at c, axis 'y' (vertical) or 'x' (lying along x)."""
        P, N, U, I = [], [], [], []
        for k in range(sides + 1):
            a = 2 * math.pi * k / sides
            u, v = math.cos(a), math.sin(a)
            for s in (-0.5, 0.5):
                if axis == "y":
                    P.append((c[0] + r * u, c[1] + s * h, c[2] + r * v))
                    N.append((u, 0, v))
                else:
                    P.append((c[0] + s * h, c[1] + r * u, c[2] + r * v))
                    N.append((0, u, v))
                U.append((k / sides, s + 0.5))
        for k in range(sides):
            a, b = 2 * k, 2 * k + 2
            I += [a, b, b + 1, a, b + 1, a + 1]
        for s in (-0.5, 0.5):           # caps
            base = len(P)
            n = (0, 1 if s > 0 else -1, 0) if axis == "y" else (1 if s > 0 else -1, 0, 0)
            P.append((c[0], c[1] + s * h, c[2]) if axis == "y" else (c[0] + s * h, c[1], c[2]))
            N.append(n)
            U.append((0.5, 0.5))
            for k in range(sides + 1):
                a = 2 * math.pi * k / sides
                u, v = math.cos(a), math.sin(a)
                P.append((c[0] + r * u, c[1] + s * h, c[2] + r * v) if axis == "y" else (c[0] + s * h, c[1] + r * u, c[2] + r * v))
                N.append(n)
                U.append((0.5 + 0.5 * u, 0.5 + 0.5 * v))
            for k in range(sides):
                tri = [base, base + 1 + k, base + 2 + k]
                I += tri if s > 0 else tri[::-1]
        self.add(name, P, N, U, I, mat)

    def tube(self, name, pts, r, mat, sides=12):
        pts = np.asarray(pts, float)
        T = np.gradient(pts, axis=0)
        T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
        up = np.array([0.0, 1.0, 0.0]) if abs(T[0][1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        n = np.cross(T[0], up)
        n /= np.linalg.norm(n)
        P, N, U, I = [], [], [], []
        for k in range(len(pts)):
            n = n - T[k] * np.dot(n, T[k])
            n /= np.linalg.norm(n) + 1e-12
            b = np.cross(T[k], n)
            for s in range(sides + 1):
                a = 2 * math.pi * s / sides
                d = math.cos(a) * n + math.sin(a) * b
                P.append(tuple(pts[k] + r * d))
                N.append(tuple(d))
                U.append((s / sides, k / max(1, len(pts) - 1)))
        ring = sides + 1
        for k in range(len(pts) - 1):
            for s in range(sides):
                a, b_, c, d = k * ring + s, k * ring + s + 1, (k + 1) * ring + s + 1, (k + 1) * ring + s
                I += [a, d, c, a, c, b_]
        self.add(name, P, N, U, I, mat)


def catenary3d(p, q, slack=1.06, n=32):
    """Cable hanging between 3D points p and q (supports at any heights; sag computed for the span)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    L = float(np.linalg.norm(q - p))
    S = slack * L
    a = L / (2 * math.sqrt(6 * (S / L - 1)))
    for _ in range(40):
        g = 2 * a * math.sinh(L / (2 * a)) - S
        a -= g / (2 * math.sinh(L / (2 * a)) - (L / a) * math.cosh(L / (2 * a)))
    out = []
    for k in range(n + 1):
        x = L * k / n
        sag = a * math.cosh((x - L / 2) / a) - a * math.cosh(L / (2 * a))
        out.append(p + (q - p) * k / n + np.array([0.0, sag, 0.0]))
    return out


# ---------------------------------------------------------------- textures
def png_bytes(surface):
    buf = io.BytesIO()
    pygame.image.save(surface, buf, "tex.png")
    return buf.getvalue()


def wood_png(n=512):
    y, x = np.mgrid[0:n, 0:n] / n
    rng = np.random.default_rng(3)
    grain = np.sin((x * 3 + 0.15 * np.sin(y * 9) + 0.02 * rng.standard_normal((n, n))) * 2 * math.pi * 9)
    t = 0.5 + 0.5 * grain
    rgb = np.stack([110 + 50 * t, 74 + 34 * t, 46 + 20 * t], -1)
    return png_bytes(pygame.surfarray.make_surface(rgb.astype(np.uint8).transpose(1, 0, 2).copy()))


def breadboard_png(n=512, holes=20):
    s = pygame.Surface((n, n))
    s.fill((52, 56, 62))
    step = n / holes
    for i in range(holes):
        for j in range(holes):
            pygame.draw.circle(s, (30, 32, 38), ((i + 0.5) * step, (j + 0.5) * step), step * 0.16)
    return png_bytes(s)


def label_png(text, w=512, h=128, bg=(190, 196, 206), fg=(20, 22, 28)):
    s = pygame.Surface((w, h))
    s.fill(bg)
    f = pygame.font.SysFont("segoeui", int(h * 0.38), bold=True)
    t = f.render(text, True, fg)
    s.blit(t, t.get_rect(center=(w // 2, h // 2)))
    return png_bytes(s)


# ---------------------------------------------------------------- the desk
def build_scene(desk):
    """3D scene of the desk in its current state (desk = lab_desk.Desk, already drawn once)."""
    desk.draw()
    shot = desk.screen.copy()
    s = Scene("LabDesk")
    # where things sit: 2D desk pixels -> metres on the desk top (x right, z toward the viewer)
    X = lambda px: (px / SCREEN_W - 0.5) * DESK_W
    Z = lambda py: (py / SCREEN_H - 0.5) * DESK_D
    top = DESK_H
    wood = s.material("wood", png=wood_png(), roughness=0.7)
    leg = s.material("leg", color=(0.2, 0.2, 0.22), roughness=0.4, metallic=0.8)
    board = s.material("breadboard", png=breadboard_png(), roughness=0.5, metallic=0.6)
    metal = s.material("instrument", color=(0.74, 0.76, 0.8), roughness=0.35, metallic=0.7)
    dark = s.material("dark_case", color=(0.27, 0.29, 0.33), roughness=0.5, metallic=0.4)
    glass = s.material("glass_cell", color=(0.6, 0.8, 0.88), roughness=0.05, metallic=0.0)
    spool = s.material("spool", color=(0.24, 0.4, 0.72), roughness=0.5)
    fiber = s.material("fiber", color=(0.94, 0.8, 0.2), roughness=0.6)
    ch1 = s.material("probe_ch1", color=(0.94, 0.89, 0.26), roughness=0.5)
    ch2 = s.material("probe_ch2", color=(0.34, 0.7, 0.91), roughness=0.5)
    s.box("desk_top", -DESK_W / 2, top - 0.04, -DESK_D / 2, DESK_W / 2, top, DESK_D / 2, wood, uv=1.0)
    for sx in (-1, 1):
        for sz in (-1, 1):
            cx, cz = sx * (DESK_W / 2 - 0.06), sz * (DESK_D / 2 - 0.06)
            s.box("leg", cx - 0.025, 0, cz - 0.025, cx + 0.025, top - 0.04, cz + 0.025, leg)
    bb = (40, 150, 900, 480)    # breadboard rect in pixels
    s.box("breadboard", X(bb[0]), top, Z(bb[1]), X(bb[2]), top + 0.012, Z(bb[3]), board, uv=4.0)
    y0 = top + 0.012
    parts = desk.parts

    def footprint(r):
        return X(r.x), Z(r.y), X(r.right), Z(r.bottom)

    lab_tex = {name: s.material(f"label_{name}", png=label_png(text)) for name, text in
               (("laser", "1550 nm laser"), ("pd1", "PD1"), ("pd2", "PD2"), ("split", "1x2"))}
    for name, h in (("laser", 0.09), ("split", 0.04), ("pd1", 0.06), ("pd2", 0.06)):
        x0, z0, x1, z1 = footprint(parts[name])
        s.box(name, x0, y0, z0, x1, y0 + h, z1, metal, top_mat=lab_tex[name])
    x0, z0, x1, z1 = footprint(parts["cell"])
    s.cylinder("gas_cell", ((x0 + x1) / 2, y0 + 0.03, (z0 + z1) / 2), 0.025, x1 - x0, glass, axis="x")
    for name in ("spool1", "spool2"):
        x0, z0, x1, z1 = footprint(parts[name])
        s.cylinder(name, ((x0 + x1) / 2, y0 + 0.025, (z0 + z1) / 2), min(x1 - x0, z1 - z0) * 0.55, 0.05, spool)
    # oscilloscope with the live screen as its texture
    sc = parts["scope"]
    x0, z0, x1, z1 = footprint(sc)
    z0, z1 = Z(140), Z(260)            # scope stands near the back of the desk, facing the user
    s.box("scope_case", x0, top, z0, x1, top + 0.26, z1, dark)
    scr_rect = pygame.Rect(sc.x + 14, sc.y + 32, sc.w - 28, 180)
    scope_mat = s.material("scope_screen", png=png_bytes(shot.subsurface(scr_rect).copy()), roughness=0.2,
                           emissive=(0.35, 0.35, 0.35))
    pad = 0.02
    s.quad("scope_screen", [(x0 + pad, top + 0.09, z1 + 0.001), (x1 - pad, top + 0.09, z1 + 0.001),
                            (x1 - pad, top + 0.24, z1 + 0.001), (x0 + pad, top + 0.24, z1 + 0.001)], (0, 0, 1), scope_mat)
    # laptop: base + tilted lid with the live laptop panel as its texture
    lp = parts["laptop"]
    lcx = X(lp.centerx)
    lx0, lz0, lx1, lz1 = lcx - 0.17, Z(600), lcx + 0.17, Z(600) + 0.23     # a 34 cm laptop
    s.box("laptop_base", lx0, top, lz0, lx1, top + 0.015, lz1, dark)
    lid_rect = pygame.Rect(lp.x, lp.y, lp.w, lp.h - 36)
    lid_mat = s.material("laptop_screen", png=png_bytes(shot.subsurface(lid_rect).copy()), roughness=0.2,
                         emissive=(0.3, 0.3, 0.3))
    hinge_z, lid_h, tilt = lz0, (lx1 - lx0) * lid_rect.h / lid_rect.w, math.radians(15)
    yb, yt = top + 0.015, top + 0.015 + lid_h * math.cos(tilt)
    zt = hinge_z - lid_h * math.sin(tilt)
    s.quad("laptop_screen", [(lx0, yb, hinge_z), (lx1, yb, hinge_z), (lx1, yt, zt), (lx0, yt, zt)],
           (0, math.sin(tilt), math.cos(tilt)), lid_mat)
    # fibers (yellow) between the components, hanging as catenaries
    port_h = y0 + 0.03
    runs = [((220, 335), (290, 335)), ((440, 335), (505, 335)), ((555, 330), (610, 245)), ((555, 340), (610, 435)),
            ((730, 245), (790, 247)), ((730, 435), (790, 437))]
    for k, ((a, b), (c, d)) in enumerate(runs):
        pts = catenary3d((X(a), port_h, Z(b)), (X(c), port_h, Z(d)), 1.04)
        s.tube(f"fiber_{k}", pts, 0.0025, fiber)
    # probe cables from the scope front to wherever the probes are
    for probe, mat in zip(desk.probes, (ch1, ch2)):
        ax, _ = probe.anchor
        start = (X(ax), top + 0.06, z1 + 0.005)
        tip_y = port_h + 0.02 if probe.at else y0 + 0.005
        end = (X(probe.pos[0]), tip_y, Z(probe.pos[1]))
        s.tube(f"probe_{probe.name}", catenary3d(start, end, 1.12), 0.003, mat)
        s.cylinder(f"probe_tip_{probe.name}", end, 0.006, 0.02, mat)
    return s


# ---------------------------------------------------------------- writers
def write_glb(scene, path):
    bin_ = bytearray()
    views, accessors, meshes, nodes, images, textures, materials = [], [], [], [], [], [], []

    def view(data, target=None):
        while len(bin_) % 4:
            bin_.append(0)
        v = {"buffer": 0, "byteOffset": len(bin_), "byteLength": len(data)}
        bin_.extend(data)
        if target:
            v["target"] = target
        views.append(v)
        return len(views) - 1

    def acc(arr, comp, typ, target, mm=False):
        a = {"bufferView": view(arr.tobytes(), target), "componentType": comp, "count": int(len(arr)), "type": typ}
        if mm:
            a["min"], a["max"] = [float(x) for x in arr.min(0)], [float(x) for x in arr.max(0)]
        accessors.append(a)
        return len(accessors) - 1

    idx = {}
    for name, m in scene.materials.items():
        pbr = {"metallicFactor": m["metallic"], "roughnessFactor": m["roughness"]}
        if m["png"]:
            images.append({"bufferView": view(m["png"]), "mimeType": "image/png"})
            textures.append({"sampler": 0, "source": len(images) - 1})
            pbr["baseColorTexture"] = {"index": len(textures) - 1}
        else:
            pbr["baseColorFactor"] = [*m["color"], 1]
        mat = {"name": name, "pbrMetallicRoughness": pbr}
        if m["emissive"]:
            mat["emissiveFactor"] = list(m["emissive"])
            if m["png"]:
                mat["emissiveTexture"] = {"index": len(textures) - 1}
        materials.append(mat)
        idx[name] = len(materials) - 1
    for name, P, N, UV, I, mname in scene.meshes:
        prim = {"attributes": {"POSITION": acc(P.astype(np.float32), 5126, "VEC3", 34962, True),
                               "NORMAL": acc(N.astype(np.float32), 5126, "VEC3", 34962),
                               "TEXCOORD_0": acc(UV.astype(np.float32), 5126, "VEC2", 34962)},
                "indices": acc(I, 5125, "SCALAR", 34963), "material": idx[mname]}
        meshes.append({"name": name, "primitives": [prim]})
        nodes.append({"name": name, "mesh": len(meshes) - 1})
    while len(bin_) % 4:
        bin_.append(0)
    doc = {"asset": {"version": "2.0", "generator": "lab_desk desk_xr"}, "scene": 0,
           "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes, "meshes": meshes, "materials": materials,
           "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(bin_)}],
           "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]}
    if images:
        doc["images"], doc["textures"] = images, textures
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, 28 + len(js) + len(bin_)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        f.write(struct.pack("<II", len(bin_), 0x004E4942) + bytes(bin_))


def _id(s):
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    return s if s and not s[0].isdigit() else "_" + s


def usda_text(scene):
    f3 = lambda v: f"({v[0]:.7g}, {v[1]:.7g}, {v[2]:.7g})"
    f2 = lambda v: f"({v[0]:.6g}, {v[1]:.6g})"
    arr = lambda vals, fmt: "[" + ", ".join(fmt(v) for v in vals) + "]"
    root = _id(scene.name)
    out = ["#usda 1.0", "(", f'    defaultPrim = "{root}"', "    metersPerUnit = 1", '    upAxis = "Y"', ")", "",
           f'def Xform "{root}" (', '    kind = "component"', ")", "{"]
    seen = {}
    for name, P, N, UV, I, mat in scene.meshes:
        ident = _id(name)
        seen[ident] = seen.get(ident, 0) + 1
        if seen[ident] > 1:
            ident = f"{ident}_{seen[ident]}"
        out += [f'    def Mesh "{ident}" (', '        prepend apiSchemas = ["MaterialBindingAPI"]', "    )", "    {",
                f"        float3[] extent = [{f3(P.min(0))}, {f3(P.max(0))}]",
                f"        int[] faceVertexCounts = {arr([3] * (len(I) // 3), str)}",
                f"        int[] faceVertexIndices = {arr(I.tolist(), str)}",
                f"        point3f[] points = {arr(P.tolist(), f3)}",
                f"        normal3f[] normals = {arr(N.tolist(), f3)} (", '            interpolation = "vertex"', "        )",
                f"        texCoord2f[] primvars:st = {arr(UV.tolist(), f2)} (", '            interpolation = "vertex"', "        )",
                '        uniform token subdivisionScheme = "none"',
                f"        rel material:binding = </{root}/Materials/{_id(mat)}>", "    }", ""]
    out += ['    def Scope "Materials"', "    {"]
    for name, m in scene.materials.items():
        mid = _id(name)
        mp = f"/{root}/Materials/{mid}"
        out += [f'        def Material "{mid}"', "        {", f"            token outputs:surface.connect = <{mp}/PBR.outputs:surface>",
                '            def Shader "PBR"', "            {", '                uniform token info:id = "UsdPreviewSurface"']
        if m["png"]:
            out.append(f"                color3f inputs:diffuseColor.connect = <{mp}/Tex.outputs:rgb>")
            if m["emissive"]:
                out.append(f"                color3f inputs:emissiveColor.connect = <{mp}/Tex.outputs:rgb>")
        else:
            c = m["color"]
            out.append(f"                color3f inputs:diffuseColor = ({c[0]:.4g}, {c[1]:.4g}, {c[2]:.4g})")
        out += [f'                float inputs:roughness = {m["roughness"]}', f'                float inputs:metallic = {m["metallic"]}',
                "                token outputs:surface", "            }"]
        if m["png"]:
            out += ['            def Shader "StReader"', "            {", '                uniform token info:id = "UsdPrimvarReader_float2"',
                    '                string inputs:varname = "st"', "                float2 outputs:result", "            }",
                    '            def Shader "Tex"', "            {", '                uniform token info:id = "UsdUVTexture"',
                    f"                asset inputs:file = @textures/{mid}.png@",
                    f"                float2 inputs:st.connect = <{mp}/StReader.outputs:result>",
                    '                token inputs:wrapS = "repeat"', '                token inputs:wrapT = "repeat"',
                    "                float3 outputs:rgb", "            }"]
        out.append("        }")
    out += ["    }", "}", ""]
    return "\n".join(out)


def write_usdz(scene, path):
    entries = [("scene.usda", usda_text(scene).encode())]
    entries += [(f"textures/{_id(n)}.png", m["png"]) for n, m in scene.materials.items() if m["png"]]
    local, central = bytearray(), bytearray()
    for name, data in entries:
        nb, crc = name.encode(), zlib.crc32(data) & 0xFFFFFFFF
        off = len(local)
        pad = (-(off + 30 + len(nb) + 4)) % 64
        extra = struct.pack("<HH", 0x1986, pad) + b"\0" * pad
        local += struct.pack("<IHHHHHIIIHH", 0x04034B50, 20, 0, 0, 0, 0, crc, len(data), len(data), len(nb), len(extra))
        local += nb + extra + data
        central += struct.pack("<IHHHHHHIIIHHHHHII", 0x02014B50, 20, 20, 0, 0, 0, 0, crc, len(data), len(data),
                               len(nb), 0, 0, 0, 0, 0, off) + nb
    end = struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, len(entries), len(entries), len(central), len(local), 0)
    with open(path, "wb") as f:
        f.write(bytes(local) + bytes(central) + end)


def export(desk, folder):
    os.makedirs(folder, exist_ok=True)
    scene = build_scene(desk)
    usdz, glb = os.path.join(folder, "lab_desk.usdz"), os.path.join(folder, "lab_desk.glb")
    write_usdz(scene, usdz)
    write_glb(scene, glb)
    tris = sum(len(m[4]) // 3 for m in scene.meshes)
    return usdz, glb, tris
