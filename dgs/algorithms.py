"""Algorithms & data structures -- the CS core, clean and tested.

The set that every CS program (and coding interview) is built on: searching,
sorting, hashing, trees, and graphs -- each tagged with its time complexity so
the Big-O is right next to the code. The punchline for THIS repo is at the
bottom: the FFT the receiver runs is a divide-and-conquer algorithm, the same
idea as merge sort, which is why phase retrieval is O(n log n) and not O(n^2).

Pure-Python, no dependencies. Education.
"""


# ── searching ───────────────────────────────────────────────────────
def linear_search(arr, target):
    """Scan left to right. O(n). Works on any list."""
    for i, v in enumerate(arr):
        if v == target:
            return i
    return -1


def binary_search(arr, target):
    """Halve the search interval each step. O(log n). REQUIRES a sorted list.
    This is why we sort first: a sorted array turns search from O(n) into O(log n)."""
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


# ── sorting ─────────────────────────────────────────────────────────
def merge_sort(arr):
    """Divide-and-conquer: split in half, sort each, merge. O(n log n), stable.
    The 'log n' is the depth of halving; the 'n' is the merge at each level.
    SAME shape as the FFT -- split, solve halves, combine."""
    if len(arr) <= 1:
        return list(arr)
    mid = len(arr) // 2
    left, right = merge_sort(arr[:mid]), merge_sort(arr[mid:])
    return _merge(left, right)


def _merge(left, right):
    out, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i]); i += 1
        else:
            out.append(right[j]); j += 1
    out.extend(left[i:]); out.extend(right[j:])
    return out


def quicksort(arr):
    """Pick a pivot, partition smaller/larger, recurse. O(n log n) average,
    O(n^2) worst (bad pivots). In place is faster, but this reads cleaner."""
    if len(arr) <= 1:
        return list(arr)
    pivot = arr[len(arr) // 2]
    less = [v for v in arr if v < pivot]
    equal = [v for v in arr if v == pivot]
    greater = [v for v in arr if v > pivot]
    return quicksort(less) + equal + quicksort(greater)


# ── hashing: a dictionary from scratch ──────────────────────────────
class HashMap:
    """A hash table with separate chaining -- how Python's dict works underneath.
    get/put/contains are O(1) *average* (O(n) worst if everything collides)."""

    def __init__(self, n_buckets=64):
        self._buckets = [[] for _ in range(n_buckets)]
        self._n = 0

    def _bucket(self, key):
        return self._buckets[hash(key) % len(self._buckets)]

    def put(self, key, value):
        b = self._bucket(key)
        for i, (k, _) in enumerate(b):
            if k == key:
                b[i] = (key, value); return
        b.append((key, value)); self._n += 1

    def get(self, key, default=None):
        for k, v in self._bucket(key):
            if k == key:
                return v
        return default

    def __contains__(self, key):
        return any(k == key for k, _ in self._bucket(key))

    def __len__(self):
        return self._n


# ── trees: a binary search tree ─────────────────────────────────────
class BST:
    """Binary search tree: left < node < right. insert/contains are O(log n) when
    balanced, O(n) when degenerate (inserting already-sorted data). inorder walk
    returns the keys SORTED -- a tree is a sorting algorithm in disguise."""

    class _Node:
        __slots__ = ("key", "left", "right")
        def __init__(self, key):
            self.key, self.left, self.right = key, None, None

    def __init__(self):
        self.root = None

    def insert(self, key):
        self.root = self._insert(self.root, key)

    def _insert(self, node, key):
        if node is None:
            return BST._Node(key)
        if key < node.key:
            node.left = self._insert(node.left, key)
        elif key > node.key:
            node.right = self._insert(node.right, key)
        return node

    def __contains__(self, key):
        node = self.root
        while node is not None:
            if key == node.key:
                return True
            node = node.left if key < node.key else node.right
        return False

    def inorder(self):
        out = []
        def walk(n):
            if n is None:
                return
            walk(n.left); out.append(n.key); walk(n.right)
        walk(self.root)
        return out


# ── graphs: breadth-first and depth-first traversal ─────────────────
def bfs(graph, start):
    """Breadth-first: explore level by level (a queue). O(V+E). Finds the
    fewest-edges path in an unweighted graph. `graph` is {node: [neighbors]}."""
    from collections import deque
    seen, order, q = {start}, [], deque([start])
    while q:
        node = q.popleft(); order.append(node)
        for nb in graph.get(node, ()):
            if nb not in seen:
                seen.add(nb); q.append(nb)
    return order


def dfs(graph, start):
    """Depth-first: go as deep as possible before backtracking (a stack). O(V+E)."""
    seen, order, stack = set(), [], [start]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node); order.append(node)
        for nb in reversed(graph.get(node, ())):
            if nb not in seen:
                stack.append(nb)
    return order


# ── the repo connection ─────────────────────────────────────────────
def fft_is_divide_and_conquer():
    """Why this module matters HERE: the FFT (np.fft, the heart of the dispersion
    operator and the GS phase-retrieval loop) is the SAME divide-and-conquer idea
    as merge_sort -- split the n-point transform into two n/2-point transforms,
    recurse, combine. That recursion is why it is O(n log n) instead of the naive
    O(n^2) DFT. On a 1e6-point signal that is the difference between a second and
    a week. The algorithm IS the receiver."""
    return "FFT: O(n log n) divide-and-conquer -- merge_sort's idea, on complex roots of unity"


if __name__ == "__main__":
    import random
    data = [random.randint(0, 1000) for _ in range(20)]
    print("merge_sort ok:", merge_sort(data) == sorted(data))
    print("quicksort ok :", quicksort(data) == sorted(data))
    s = sorted(data)
    print("binary_search ok:", all(binary_search(s, v) != -1 for v in s))
    g = {0: [1, 2], 1: [3], 2: [3, 4], 3: [], 4: []}
    print("bfs:", bfs(g, 0), " dfs:", dfs(g, 0))
    print(fft_is_divide_and_conquer())
