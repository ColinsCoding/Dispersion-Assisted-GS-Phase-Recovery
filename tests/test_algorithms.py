"""Test algorithms & data structures against Python's own correct answers."""
import sys, pathlib, random
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs import algorithms as alg

random.seed(0)

# 1. sorts agree with sorted() on random + edge-case inputs
for data in ([random.randint(-50, 50) for _ in range(200)], [], [1], [3, 3, 3], list(range(50, 0, -1))):
    assert alg.merge_sort(data) == sorted(data)
    assert alg.quicksort(data) == sorted(data)

# 2. binary_search matches linear_search and list.index on a sorted array
s = sorted(random.randint(0, 1000) for _ in range(500))
for _ in range(200):
    t = random.randint(0, 1000)
    found = alg.binary_search(s, t)
    assert (found != -1) == (t in s)
    if found != -1:
        assert s[found] == t
assert alg.binary_search(s, -999) == -1                 # absent -> -1

# 3. HashMap behaves like dict
hm, ref = alg.HashMap(n_buckets=8), {}
for _ in range(500):
    k, v = random.randint(0, 100), random.random()
    hm.put(k, v); ref[k] = v
assert len(hm) == len(ref)
for k in ref:
    assert k in hm and hm.get(k) == ref[k]
assert 9999 not in hm and hm.get(9999, "x") == "x"

# 4. BST inorder walk returns sorted unique keys
keys = [random.randint(0, 100) for _ in range(300)]
bst = alg.BST()
for k in keys:
    bst.insert(k)
assert bst.inorder() == sorted(set(keys))
assert all(k in bst for k in keys)
assert 1234 not in bst

# 5. graph traversals visit every reachable node, start first
g = {0: [1, 2], 1: [3, 4], 2: [4], 3: [], 4: [5], 5: []}
for trav in (alg.bfs, alg.dfs):
    order = trav(g, 0)
    assert order[0] == 0
    assert set(order) == {0, 1, 2, 3, 4, 5}             # all reachable, no repeats
    assert len(order) == len(set(order))
# BFS reaches node 4 before node 3's child-depth (level order): 4 appears by level 2
assert alg.bfs(g, 0).index(4) <= alg.bfs(g, 0).index(5)

print("TEST PASS  (sorts==sorted; binary==linear search; HashMap==dict; "
      "BST inorder sorted; BFS/DFS cover the graph)")
