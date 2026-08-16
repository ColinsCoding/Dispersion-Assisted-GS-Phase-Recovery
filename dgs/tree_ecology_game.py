"""A pygame ecology game built on the recursive tree-generation idea from
dgs/speedtree_recursion.py, ported to 2D (fast enough to draw dozens of
trees per frame, which MuJoCo's rigid-body simulation is not built for).

VARIANCE ECOLOGY + DNA EVOLUTION: every tree has genetics (branch angle,
length/radius decay, branching factor, max depth). The FIRST generation is
randomly sampled (real variance, not one species repeated); every
generation after that is EVOLVED -- mature, well-lit (healthy) trees
reproduce, passing their own genetics to a child with small random
mutations (dgs.tree_ecology_game.mutate_genetics), while starved trees die
without reproducing. That's an actual selection process, not just
decoration: genetics that survive shading pressure in a given spot are
the ones that get copied (with drift) into the next generation there.

EQUIPOTENTIAL SOIL FIELD: a scalar "soil quality" potential V(x), built
from a few Gaussian sources exactly like dgs.potential_field_sandbox's
beacons, sets a per-location growth-rate multiplier -- trees literally grow
faster in richer soil. Equipotential markers (vertical ticks where V(x)
crosses evenly-spaced threshold levels, the 1D analog of contour lines) are
drawn along the ground so the field is visible, not just felt.

TIMING: +/- keys change how many real frames correspond to one simulated
day, i.e. the actual simulation SPEED, shown on the HUD.

MULTIPLAYER (same screen, for playing with friends): left-click plants for
Player 1 (blue tint), right-click plants for Player 2 (red tint); every
tree is tagged with its planter/lineage's player_id (inherited by its
descendants), and the HUD tracks each player's living population as a
simple competitive score -- whose genetic lineage dominates the garden.

DATASET: unchanged in spirit from before -- every simulated day is logged
(population, average height, diversity index, and now population per
player), and every tree's full life record (genetics, lineage, birth/death
day, final height) is logged. Exported as two CSVs via export_dataset().
"""

import math
import random
import csv
from dataclasses import dataclass


@dataclass
class TreeGenetics:
    branch_angle_deg: float
    length_decay: float
    radius_decay: float
    n_children: int
    max_depth: int
    base_length: float
    hue: float   # 0..1, used for rendering color variation


def random_genetics(rng):
    """Sample one individual's genetics from scratch -- used only for the
    FIRST generation. Every later generation is produced by mutate_genetics
    from a surviving parent instead (real inheritance, not fresh sampling)."""
    return TreeGenetics(
        branch_angle_deg=rng.uniform(18, 42),
        length_decay=rng.uniform(0.62, 0.80),
        radius_decay=rng.uniform(0.60, 0.78),
        n_children=rng.choice([2, 2, 2, 3]),
        max_depth=rng.choice([4, 5, 5, 6]),
        base_length=rng.uniform(28, 42),
        hue=rng.uniform(0.22, 0.42),   # yellow-green to blue-green range
    )


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def mutate_genetics(parent, rng, mutation_strength=1.0):
    """DNA inheritance: a child's genetics = the parent's, each trait
    nudged by a small random amount (continuous traits) or rarely flipped
    to a neighboring value (discrete traits n_children/max_depth) -- the
    actual evolutionary-algorithm step. mutation_strength scales all the
    continuous-trait mutation sizes together."""
    return TreeGenetics(
        branch_angle_deg=_clip(parent.branch_angle_deg + rng.gauss(0, 3.0 * mutation_strength), 8, 55),
        length_decay=_clip(parent.length_decay + rng.gauss(0, 0.03 * mutation_strength), 0.45, 0.88),
        radius_decay=_clip(parent.radius_decay + rng.gauss(0, 0.03 * mutation_strength), 0.45, 0.85),
        n_children=parent.n_children if rng.random() > 0.08 else rng.choice([2, 3]),
        max_depth=parent.max_depth if rng.random() > 0.08 else _clip(parent.max_depth + rng.choice([-1, 1]), 3, 7),
        base_length=_clip(parent.base_length + rng.gauss(0, 2.5 * mutation_strength), 15, 55),
        hue=_clip(parent.hue + rng.gauss(0, 0.02 * mutation_strength), 0.0, 1.0),
    )


def generate_tree_segments(genetics, rng):
    """Recursively build the FULL segment list once (fixed for this tree's
    lifetime): each entry is (x0, y0, x1, y1, width, depth), local
    coordinates rooted at (0, 0), +y is UP."""
    segments = []

    def branch(x0, y0, angle_deg, length, radius, depth):
        if depth > genetics.max_depth or length < 1.5:
            return
        rad = math.radians(angle_deg)
        x1 = x0 + length * math.sin(rad)
        y1 = y0 + length * math.cos(rad)
        segments.append((x0, y0, x1, y1, max(0.6, radius), depth))
        if depth < genetics.max_depth:
            child_length = length * genetics.length_decay
            child_radius = radius * genetics.radius_decay
            for i in range(genetics.n_children):
                side = 1 if i % 2 == 0 else -1
                spread = genetics.branch_angle_deg + rng.uniform(-6, 6)
                child_angle = angle_deg + side * spread + rng.uniform(-4, 4)
                branch(x1, y1, child_angle, child_length, child_radius, depth + 1)

    branch(0.0, 0.0, 0.0, genetics.base_length, genetics.base_length * 0.12, 0)
    return segments


# ── equipotential soil field ─────────────────────────────────────────────

def make_soil_sources(n=3, x_range=(-400, 400), seed=0):
    rng = random.Random(seed)
    return [(rng.uniform(*x_range), rng.uniform(0.6, 1.6), rng.uniform(60, 140)) for _ in range(n)]


def soil_potential(x, sources):
    """V(x) = sum of Gaussian-bump sources -- exactly the same style of
    scalar potential as dgs.potential_field_sandbox's beacons, just 1D."""
    return sum(strength * math.exp(-((x - cx) ** 2) / (2 * sigma ** 2)) for (cx, strength, sigma) in sources)


def soil_growth_multiplier(x, sources):
    """0.5x growth in the poorest soil, up to ~1.8x in the richest -- soil
    quality actually changes outcomes, not just cosmetics."""
    return _clip(0.5 + 0.9 * soil_potential(x, sources), 0.4, 1.8)


def equipotential_marks(sources, x_range=(-450, 450), n_levels=5, n_samples=400):
    """1D analog of equipotential contour lines: the x-positions where V(x)
    crosses each of n_levels evenly-spaced threshold values."""
    max_v = max((soil_potential(x, sources) for x in
                 [x_range[0] + i * (x_range[1] - x_range[0]) / n_samples for i in range(n_samples + 1)]), default=0.0)
    if max_v <= 0:
        return []
    levels = [max_v * (k + 1) / (n_levels + 1) for k in range(n_levels)]
    xs = [x_range[0] + i * (x_range[1] - x_range[0]) / n_samples for i in range(n_samples + 1)]
    vs = [soil_potential(x, sources) for x in xs]
    marks = []
    for level in levels:
        for i in range(len(xs) - 1):
            if (vs[i] - level) * (vs[i + 1] - level) < 0:
                marks.append(xs[i])
    return marks


class Tree:
    _next_id = [0]

    def __init__(self, x_pos, genetics, birth_day, rng, player_id=0, parent_id=None):
        self.id = Tree._next_id[0]
        Tree._next_id[0] += 1
        self.x_pos = x_pos
        self.genetics = genetics
        self.segments = generate_tree_segments(genetics, rng)
        self.birth_day = birth_day
        self.growth_stage = 0.0     # continuous: floor() = deepest fully-grown depth
        self.light = 1.0
        self.alive = True
        self.death_day = None
        self.days_starved = 0
        self.max_depth_local = max((s[5] for s in self.segments), default=0)
        self.player_id = player_id
        self.parent_id = parent_id
        self.days_healthy = 0       # consecutive well-lit days, gates reproduction

    def visible_segments(self):
        """Segments up to the current growth stage; the segment at the
        currently-growing depth is drawn partway (lerped 0->full length)
        for a smooth, continuous growth animation instead of discrete pops."""
        stage_floor = int(self.growth_stage)
        frac = self.growth_stage - stage_floor
        out = []
        for (x0, y0, x1, y1, w, depth) in self.segments:
            if depth < stage_floor:
                out.append((x0, y0, x1, y1, w))
            elif depth == stage_floor:
                out.append((x0, y0, x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac, w))
        return out

    def height(self):
        vis = self.visible_segments()
        if not vis:
            return 0.0
        return max(max(s[1], s[3]) for s in vis)

    def canopy_half_width(self):
        vis = self.visible_segments()
        if not vis:
            return 0.0
        xs = [s[0] for s in vis] + [s[2] for s in vis]
        return max(abs(v) for v in xs)

    def is_mature(self):
        return self.growth_stage >= self.max_depth_local - 0.25

    def record(self):
        return {
            "id": self.id, "parent_id": self.parent_id, "player_id": self.player_id,
            "x_pos": round(self.x_pos, 2),
            "branch_angle_deg": round(self.genetics.branch_angle_deg, 3),
            "length_decay": round(self.genetics.length_decay, 4),
            "radius_decay": round(self.genetics.radius_decay, 4),
            "n_children": self.genetics.n_children, "max_depth": self.genetics.max_depth,
            "birth_day": self.birth_day, "death_day": self.death_day,
            "final_growth_stage": round(self.growth_stage, 3),
            "final_height": round(self.height(), 2),
            "alive_at_end": self.alive,
        }


def compute_light(trees, shade_reach=90.0, shade_strength=0.6):
    """Each alive tree's light in [0,1]: reduced by taller neighbors whose
    canopy overlaps it. A simple, checkable ecological rule -- not
    decorative -- taller + closer neighbors shade more."""
    alive = [t for t in trees if t.alive]
    for t in alive:
        light = 1.0
        h_t = t.height()
        for other in alive:
            if other is t:
                continue
            dx = abs(other.x_pos - t.x_pos)
            if dx > shade_reach:
                continue
            h_o = other.height()
            if h_o <= h_t:
                continue
            overlap = max(0.0, 1.0 - dx / shade_reach)
            height_advantage = min(1.0, (h_o - h_t) / max(h_o, 1.0))
            light -= shade_strength * overlap * height_advantage
        t.light = max(0.0, min(1.0, light))
    return {t.id: t.light for t in alive}


def simulation_step(trees, rng, day, growth_rate=0.12, starve_threshold=0.15,
                     starve_days_to_die=25, shade_reach=90.0, shade_strength=0.6,
                     soil_sources=(), reproduce_chance=0.03, reproduce_health_days=15,
                     seed_dispersal=70.0, mutation_strength=1.0, max_population=250):
    """Advance one day: compute light, grow (rate modulated by local soil
    potential if soil_sources given), possibly kill starved trees, and let
    mature/healthy trees reproduce (DNA inheritance with mutation) into new
    trees appended directly to `trees`. Returns this day's aggregate stats
    dict -- one row of the dataset's per-day time series."""
    compute_light(trees, shade_reach=shade_reach, shade_strength=shade_strength)
    newborns = []
    for t in trees:
        if not t.alive:
            continue
        soil_mult = soil_growth_multiplier(t.x_pos, soil_sources) if soil_sources else 1.0
        t.growth_stage = min(t.max_depth_local + 0.999, t.growth_stage + growth_rate * t.light * soil_mult)

        if t.light < starve_threshold:
            t.days_starved += 1
            t.days_healthy = 0
        else:
            t.days_starved = max(0, t.days_starved - 1)
            t.days_healthy += 1
        if t.days_starved >= starve_days_to_die:
            t.alive = False
            t.death_day = day
            continue

        if (len(trees) + len(newborns) < max_population and t.is_mature()
                and t.days_healthy >= reproduce_health_days and rng.random() < reproduce_chance):
            child_genetics = mutate_genetics(t.genetics, rng, mutation_strength=mutation_strength)
            child_x = t.x_pos + rng.uniform(-seed_dispersal, seed_dispersal)
            newborns.append(Tree(child_x, child_genetics, birth_day=day, rng=rng,
                                  player_id=t.player_id, parent_id=t.id))

    trees.extend(newborns)

    alive = [t for t in trees if t.alive]
    heights = [t.height() for t in alive]
    angles = [t.genetics.branch_angle_deg for t in alive]
    diversity_index = (sum((a - sum(angles)/len(angles))**2 for a in angles) / len(angles))**0.5 if len(angles) > 1 else 0.0
    pop_by_player = {}
    for t in alive:
        pop_by_player[t.player_id] = pop_by_player.get(t.player_id, 0) + 1
    return {
        "day": day,
        "population": len(alive),
        "births": len(newborns),
        "avg_height": round(sum(heights) / len(heights), 3) if heights else 0.0,
        "avg_light": round(sum(t.light for t in alive) / len(alive), 4) if alive else 0.0,
        "diversity_index": round(diversity_index, 4),
        "avg_branch_angle": round(sum(angles) / len(angles), 3) if angles else 0.0,
        "population_by_player": dict(pop_by_player),
    }


def run_simulation(n_days, n_initial_trees=10, planting_rate=0.15, x_range=(-400, 400),
                    seed=0, use_soil=True, **step_kwargs):
    """Pure-function ecology driver (no pygame): plant an initial population,
    step the simulation for n_days (with reproduction/evolution and an
    optional soil field), and return (final_tree_records, per_day_records)
    -- the dataset."""
    rng = random.Random(seed)
    soil_sources = make_soil_sources(seed=seed) if use_soil else ()
    trees = []
    for _ in range(n_initial_trees):
        x = rng.uniform(*x_range)
        trees.append(Tree(x, random_genetics(rng), birth_day=0, rng=rng))

    per_day_records = []
    for day in range(n_days):
        if rng.random() < planting_rate:
            x = rng.uniform(*x_range)
            trees.append(Tree(x, random_genetics(rng), birth_day=day, rng=rng))
        per_day_records.append(simulation_step(trees, rng, day, soil_sources=soil_sources, **step_kwargs))

    tree_records = [t.record() for t in trees]
    return tree_records, per_day_records


def export_dataset(tree_records, per_day_records, tree_csv_path, daily_csv_path):
    with open(tree_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(tree_records[0].keys()))
        writer.writeheader()
        writer.writerows(tree_records)
    daily_fields = [k for k in per_day_records[0].keys() if k != "population_by_player"]
    with open(daily_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=daily_fields)
        writer.writeheader()
        for row in per_day_records:
            writer.writerow({k: row[k] for k in daily_fields})


# ── pygame game loop ─────────────────────────────────────────────────────

PLAYER_COLORS = {0: None, 1: (90, 140, 230), 2: (230, 90, 90)}   # None = neutral/no player


def main():
    import pygame

    WIDTH, HEIGHT = 900, 600
    GROUND_Y = HEIGHT - 60

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Variance Ecology: evolving trees, for playing with friends")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)

    rng = random.Random(1)
    soil_sources = make_soil_sources(seed=1)
    marks = equipotential_marks(soil_sources)
    trees = [Tree(rng.uniform(-380, 380), random_genetics(rng), birth_day=0, rng=rng) for _ in range(8)]
    per_day_records = []
    day = 0
    frame_in_day = 0
    day_frames = 12   # frames per simulated day -- TIMING, adjustable with +/-
    paused = False
    running = True

    def world_to_screen(x, y):
        return (WIDTH // 2 + x, GROUND_Y - y)

    def draw_tree(surf, tree):
        base_rgb = PLAYER_COLORS.get(tree.player_id)
        for (x0, y0, x1, y1, w) in tree.visible_segments():
            p0 = world_to_screen(tree.x_pos + x0, y0)
            p1 = world_to_screen(tree.x_pos + x1, y1)
            depth_frac = 0.3 + 0.7 * (y0 / max(1.0, tree.height() + 1))
            if base_rgb is not None:
                color = pygame.Color(*base_rgb)
                color = pygame.Color(int(color.r * (0.5 + 0.5*depth_frac)),
                                      int(color.g * (0.5 + 0.5*depth_frac)),
                                      int(color.b * (0.5 + 0.5*depth_frac)))
            else:
                color = pygame.Color(0, 0, 0)
                color.hsva = (tree.genetics.hue * 300, 60, 40 + 50 * depth_frac, 100)
            pygame.draw.line(surf, color, p0, p1, max(1, int(w)))
        if not tree.alive:
            marker = world_to_screen(tree.x_pos, tree.height() + 8)
            pygame.draw.circle(surf, (90, 40, 40), marker, 3)

    def draw_soil(surf):
        for sx in range(-WIDTH // 2, WIDTH // 2, 4):
            v = soil_potential(sx, soil_sources)
            shade = _clip(int(20 + v * 40), 20, 90)
            pygame.draw.line(surf, (30, shade, 25), world_to_screen(sx, 0), world_to_screen(sx, -6), 4)
        for mx in marks:
            p0 = world_to_screen(mx, 0)
            pygame.draw.line(surf, (200, 200, 100), (p0[0], p0[1] - 8), (p0[0], p0[1] + 2), 1)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    day_frames = max(2, day_frames - 2)     # fewer frames/day = faster
                elif event.key == pygame.K_MINUS:
                    day_frames = min(60, day_frames + 2)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                world_x = mx - WIDTH // 2
                if event.button == 1:                       # Player 1: left click
                    trees.append(Tree(world_x, random_genetics(rng), birth_day=day, rng=rng, player_id=1))
                elif event.button == 3:                      # Player 2: right click
                    trees.append(Tree(world_x, random_genetics(rng), birth_day=day, rng=rng, player_id=2))

        if not paused:
            frame_in_day += 1
            if frame_in_day >= day_frames:
                frame_in_day = 0
                per_day_records.append(simulation_step(trees, rng, day, soil_sources=soil_sources))
                day += 1

        screen.fill((18, 24, 34))
        pygame.draw.rect(screen, (35, 45, 30), (0, GROUND_Y, WIDTH, HEIGHT - GROUND_Y))
        draw_soil(screen)
        for t in sorted(trees, key=lambda t: t.x_pos):
            draw_tree(screen, t)

        alive = [t for t in trees if t.alive]
        pop = len(alive)
        angles = [t.genetics.branch_angle_deg for t in alive]
        div = (sum((a - sum(angles)/len(angles))**2 for a in angles)/len(angles))**0.5 if len(angles) > 1 else 0.0
        p1 = sum(1 for t in alive if t.player_id == 1)
        p2 = sum(1 for t in alive if t.player_id == 2)
        speed = round(60 / day_frames, 1)
        hud1 = f"day {day}   population {pop}   diversity {div:.2f}   speed {speed}x days/s"
        hud2 = f"Player1(blue) {p1}   Player2(red) {p2}   [L-click]=P1 plant  [R-click]=P2 plant  [+/-]=speed  [space]=pause"
        screen.blit(font.render(hud1, True, (230, 230, 230)), (10, 6))
        screen.blit(font.render(hud2, True, (200, 210, 220)), (10, 26))

        pygame.display.flip()
        clock.tick(60)

    if per_day_records:
        tree_records = [t.record() for t in trees]
        export_dataset(tree_records, per_day_records, "tree_ecology_final.csv", "tree_ecology_daily.csv")
    pygame.quit()


if __name__ == "__main__":
    main()
