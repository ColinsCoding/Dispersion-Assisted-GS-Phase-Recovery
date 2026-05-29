"""
main.py — Lumion-style photorealism renderer entry point.
Pygame window + numpy software rasteriser.
No GPU required. PBR + sky + fur physics.

Run:  py -3.12 renderer/main.py
Keys: SPACE=zoomies  W/S=sun elevation  A/D=sun azimuth  ESC=quit
"""
import sys, pathlib, time
import numpy as np
import pygame

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from renderer.core.scene         import Scene, SceneNode, Transform
from renderer.materials.pbr      import PBRMaterial
from renderer.materials.thin_film import thin_film_rgb
from renderer.lighting.sky       import preetham_sky, sun_direction
from renderer.physics.hair_sim   import DogFur

W, H   = 800, 600
FOV    = 60.0
FPS    = 30

# ── Build scene ───────────────────────────────────────────────────────────────
scene  = Scene()

dog_node = SceneNode("golden_doodle")
dog_node.material = PBRMaterial(
    albedo   = (0.82, 0.55, 0.18),
    metallic  = 0.0,
    roughness = 0.75,
)
scene.add(dog_node)

# Load voxel surface points from dog.vox bounding box approximation
# (full vox parser omitted — use SDF surface from voxel_dog.py)
N_STRANDS = 200
rng = np.random.default_rng(42)
surface_pts = rng.uniform(-0.4, 0.4, (N_STRANDS, 3))
surface_pts[:,1] = np.abs(surface_pts[:,1]) * 0.5
normals     = surface_pts / (np.linalg.norm(surface_pts, axis=1, keepdims=True) + 1e-9)

fur = DogFur(surface_pts, normals, breed="golden_doodle")

# ── Pygame software renderer ──────────────────────────────────────────────────
pygame.init()
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Lumion-style Renderer — Golden Doodle")
clock  = pygame.time.Clock()

# Sky parameters
sun_el  = 35.0
sun_az  = 45.0
wind    = np.array([0.3, 0.0, 0.1])

def render_frame(surf, sun_el, sun_az, fur, t):
    """Software rasterise: sky gradient + dog silhouette + fur strands."""
    # Sky — sample Preetham across vertical strip
    for row in range(H):
        view_theta = np.radians((row / H) * 80)
        sun_theta  = np.radians(90 - sun_el)
        gamma      = abs(view_theta - sun_theta)
        rgb = preetham_sky(sun_theta, view_theta, gamma, turbidity=2.5)
        color = tuple(int(np.clip(c * 255, 0, 255)) for c in rgb)
        pygame.draw.line(surf, color, (0, row), (W, row))

    # Ground plane
    ground_y = int(H * 0.65)
    pygame.draw.rect(surf, (34, 52, 28),
                     (0, ground_y, W, H - ground_y))

    # Dog body — ellipse silhouette (PBR shaded)
    sun_dir = sun_direction(sun_el, sun_az)
    mat     = dog_node.material
    cx, cy  = W//2, ground_y - 80
    for px in range(cx-120, cx+120, 2):
        for py in range(cy-70, cy+70, 2):
            dx = (px-cx)/120; dy = (py-cy)/70
            if dx**2 + dy**2 < 1.0:
                normal = np.array([dx, -dy, np.sqrt(max(0, 1-dx**2-dy**2))])
                view   = np.array([0, 0, 1])
                rgb    = mat.shade(normal, view, sun_dir,
                                   light_color=(1.0, 0.95, 0.85))
                # Thin film iridescence on surface edge
                if dx**2 + dy**2 > 0.75:
                    angle_deg = np.degrees(np.arccos(np.clip(normal[2],0,1)))
                    film_rgb  = thin_film_rgb(angle_deg, d_nm=180)
                    rgb = rgb * 0.7 + film_rgb * 0.3

                color = tuple(int(np.clip(c*255,0,255)) for c in rgb)
                surf.set_at((px, py), color)

    # Fur strands — project 3D -> 2D
    scale  = 200
    offset = np.array([cx, cy, 0])
    for strand in fur.strands[:80]:      # draw first 80 for speed
        pts = strand.pos
        s0  = pts[0]
        sx0 = int(cx + s0[0]*scale)
        sy0 = int(cy - s0[1]*scale)
        if not (0 <= sx0 < W and 0 <= sy0 < H):
            continue
        for i in range(1, len(pts)):
            sx1 = int(cx + pts[i][0]*scale)
            sy1 = int(cy - pts[i][1]*scale)
            t_  = i / len(pts)
            r   = int(lerp(220, 160, t_))
            g   = int(lerp(150, 100, t_))
            b   = int(lerp(60,  30,  t_))
            if 0 <= sx1 < W and 0 <= sy1 < H:
                pygame.draw.line(surf, (r,g,b), (sx0,sy0), (sx1,sy1), 1)
            sx0, sy0 = sx1, sy1

    # HUD
    font = pygame.font.SysFont("monospace", 12)
    lines = [
        f"Sun: el={sun_el:.0f} az={sun_az:.0f}  W/S A/D",
        f"Fur strands: {len(fur.strands)}  breed: golden_doodle",
        f"PBR: roughness={mat.roughness:.2f} metallic={mat.metallic:.2f}",
        f"SPACE=zoomies  ESC=quit",
    ]
    for i, line in enumerate(lines):
        surf.blit(font.render(line, True, (200,220,255)), (8, 8+i*16))


def lerp(a, b, t): return a + (b-a)*t


# ── Main loop ─────────────────────────────────────────────────────────────────
t = 0.0
running = True
while running:
    dt = clock.tick(FPS) / 1000.0
    t += dt

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:  running = False
            if event.key == pygame.K_SPACE:   fur.zoomies()

    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]: sun_el = min(sun_el + 1, 89)
    if keys[pygame.K_s]: sun_el = max(sun_el - 1,  1)
    if keys[pygame.K_a]: sun_az = (sun_az - 2) % 360
    if keys[pygame.K_d]: sun_az = (sun_az + 2) % 360

    # Gentle breeze
    wind = np.array([0.4*np.sin(t*0.7), 0, 0.2*np.cos(t*0.5)])
    fur.update(dt, wind=wind)

    render_frame(screen, sun_el, sun_az, fur, t)
    pygame.display.flip()

pygame.quit()
