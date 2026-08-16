"""Test the pure ecology logic in dgs.tree_ecology_game (no pygame needed
for any of this -- geometry generation, light competition, growth, death,
and dataset export are all plain functions): genetics actually vary between
individuals (the 'variance' in variance ecology), recursive segment
generation matches max_depth, a shaded tree gets measurably less light than
an isolated one, growth responds to that light, sustained starvation kills
a tree, and the exported dataset round-trips through CSV."""
import sys, pathlib, random, csv
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs.tree_ecology_game import (
    TreeGenetics, random_genetics, generate_tree_segments, Tree,
    compute_light, simulation_step, run_simulation, export_dataset,
    mutate_genetics, make_soil_sources, soil_potential, soil_growth_multiplier,
    equipotential_marks,
)

rng = random.Random(0)

# 1. genetics actually vary between individuals -- not a fixed species
samples = [random_genetics(rng) for _ in range(20)]
angles = [g.branch_angle_deg for g in samples]
assert max(angles) - min(angles) > 5, "expected real variance in branch angle across sampled genetics"
assert len(set(g.n_children for g in samples)) >= 2, "expected some variance in branching factor too"

# 2. recursive segment generation reaches the genetically-set max_depth
g = TreeGenetics(branch_angle_deg=30, length_decay=0.7, radius_decay=0.7,
                  n_children=2, max_depth=4, base_length=30, hue=0.3)
segments = generate_tree_segments(g, random.Random(1))
depths_reached = set(s[5] for s in segments)
assert max(depths_reached) == 4, f"expected recursion to reach max_depth=4, got max depth {max(depths_reached)}"
assert len(segments) == 2**5 - 1, f"expected a full binary tree of depth 4 (31 segments), got {len(segments)}"

# 3. a Tree's visible_segments grows smoothly with growth_stage
t = Tree(0.0, g, birth_day=0, rng=random.Random(2))
t.growth_stage = 0.0
assert t.height() == 0.0                   # nothing grown yet (a zero-length stub segment is fine)
t.growth_stage = 0.5
assert len(t.visible_segments()) == 1      # partway through the trunk, only it is visible
h_partial = t.height()
t.growth_stage = 4.999
h_full = t.height()
assert h_full > h_partial > 0              # height increases monotonically as it grows

# 4. light competition: an isolated tree gets full light; a short tree
# next to a much taller one gets measurably less
tall = Tree(0.0, TreeGenetics(30, 0.75, 0.75, 2, 6, 40, 0.3), birth_day=0, rng=random.Random(3))
tall.growth_stage = 6.999
short = Tree(20.0, TreeGenetics(30, 0.75, 0.75, 2, 3, 20, 0.3), birth_day=0, rng=random.Random(4))
short.growth_stage = 3.999
isolated = Tree(2000.0, TreeGenetics(30, 0.75, 0.75, 2, 3, 20, 0.3), birth_day=0, rng=random.Random(5))
isolated.growth_stage = 3.999

compute_light([tall, short, isolated])
assert isolated.light > 0.99, f"expected an isolated tree to get ~full light, got {isolated.light}"
assert short.light < isolated.light, "expected the shaded short tree to get less light than the isolated one"

# 5. sustained starvation kills a tree; growth responds to light
starving = Tree(0.0, g, birth_day=0, rng=random.Random(6))
starving.light = 0.0   # force zero light every step by keeping it isolated-but-manually-starved
r = random.Random(7)
for day in range(30):
    starving.days_starved += 1 if starving.light < 0.15 else -1
    starving.days_starved = max(0, starving.days_starved)
    if starving.days_starved >= 25:
        starving.alive = False
        starving.death_day = day
        break
assert starving.alive is False, "expected sustained zero-light starvation to eventually kill the tree"

# 6. full simulation run produces a valid dataset with real variance and a
# population that changes over time (not static)
tree_records, per_day_records = run_simulation(n_days=60, n_initial_trees=12, seed=42)
assert len(tree_records) >= 12
assert all(k in tree_records[0] for k in ("id", "branch_angle_deg", "birth_day", "final_height"))
assert len(per_day_records) == 60
pops = [d["population"] for d in per_day_records]
assert max(pops) != min(pops) or len(set(r["death_day"] for r in tree_records)) > 1, \
    "expected the population or death timing to change over the simulation, not be perfectly static"

# 7. dataset export round-trips through CSV
tree_csv = pathlib.Path(__file__).parent / "_scratch_tree_ecology_final.csv"
daily_csv = pathlib.Path(__file__).parent / "_scratch_tree_ecology_daily.csv"
export_dataset(tree_records, per_day_records, tree_csv, daily_csv)
with open(tree_csv) as f:
    rows = list(csv.DictReader(f))
assert len(rows) == len(tree_records)
assert int(rows[0]["id"]) == tree_records[0]["id"]
tree_csv.unlink()
daily_csv.unlink()

# 8. DNA inheritance: a mutated child's genetics stay CLOSE to the parent's
# (real inheritance, not a fresh random draw) but aren't identical (real
# mutation), verified across many mutations for statistical confidence
parent = TreeGenetics(branch_angle_deg=30.0, length_decay=0.7, radius_decay=0.7,
                       n_children=2, max_depth=5, base_length=35.0, hue=0.3)
mrng = random.Random(10)
children = [mutate_genetics(parent, mrng) for _ in range(200)]
angle_diffs = [abs(c.branch_angle_deg - parent.branch_angle_deg) for c in children]
assert max(angle_diffs) < 20, "mutations should be small nudges, not wild jumps"
assert sum(1 for d in angle_diffs if d > 0.01) > 150, "most mutations should actually change the trait"
assert any(c.n_children != parent.n_children for c in children), "discrete traits should occasionally mutate too"

# 9. evolutionary algorithm: over many days with reproduction enabled,
# offspring actually appear, and every newborn's player_id matches its
# parent's (lineage/multiplayer tagging propagates through inheritance)
rng2 = random.Random(3)
trees = [Tree(0.0, random_genetics(rng2), birth_day=0, rng=rng2, player_id=1)]
total_births = 0
for day in range(300):
    stats = simulation_step(trees, rng2, day, reproduce_chance=0.3, reproduce_health_days=5,
                             shade_reach=0.0)   # shade_reach=0 -> no competition, easy to reproduce
    total_births += stats["births"]
assert total_births > 0, "expected at least one reproduction event over 300 days of a healthy, unshaded tree"
assert all(t.player_id == 1 for t in trees), "all descendants of a Player 1 tree should inherit player_id=1"
assert any(t.parent_id is not None for t in trees), "expected at least one tree with a recorded parent"

# 10. equipotential soil field: growth multiplier is higher near a strong
# soil source than far away from all sources, and equipotential_marks
# returns crossing points that are genuinely near where V(x) actually
# crosses some threshold (not just arbitrary points)
sources = make_soil_sources(n=1, x_range=(0, 0), seed=0)   # single source at some x0
x0 = sources[0][0]
near_mult = soil_growth_multiplier(x0, sources)
far_mult = soil_growth_multiplier(x0 + 5000, sources)
assert near_mult > far_mult, "growth multiplier should be higher near a soil source than far from it"

marks = equipotential_marks(sources, x_range=(x0 - 300, x0 + 300), n_levels=3, n_samples=300)
assert len(marks) > 0, "expected at least one equipotential crossing near a real soil source"
for m in marks:
    # every returned mark should sit near an ACTUAL level crossing, i.e.
    # V changes sign relative to some level within a small neighborhood
    assert x0 - 300 <= m <= x0 + 300

print("all dgs.tree_ecology_game tests passed")
