"""Test dgs.card_counting: the Hi-Lo weights, the balanced-deck property (20/20/12),
true-count normalization by decks remaining, the linear edge with break-even at
true count +1, and Kelly betting only above break-even."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs import card_counting as cc

# 1. Hi-Lo weights: 2-6 -> +1, 7-9 -> 0, 10-A -> -1 (ints and string aliases)
assert [cc.hi_lo_value(r) for r in (2, 3, 4, 5, 6)] == [1, 1, 1, 1, 1]
assert [cc.hi_lo_value(r) for r in (7, 8, 9)] == [0, 0, 0]
assert [cc.hi_lo_value(r) for r in (10, 11, 12, 13, 14)] == [-1, -1, -1, -1, -1]
assert cc.hi_lo_value("A") == -1 and cc.hi_lo_value("K") == -1 and cc.hi_lo_value("T") == -1
assert cc.hi_lo_value("10") == -1 and cc.hi_lo_value("5") == 1

# 2. running count is the sum of weights
assert cc.running_count([5, 3, "K", 6]) == 1 + 1 - 1 + 1

# 3. balanced system: a full deck has 20 +1s, 20 -1s, 12 zeros -> sums to 0
deck = cc.full_deck()
assert len(deck) == 52
assert sum(1 for c in deck if cc.hi_lo_value(c) == 1) == 20
assert sum(1 for c in deck if cc.hi_lo_value(c) == -1) == 20
assert sum(1 for c in deck if cc.hi_lo_value(c) == 0) == 12
assert cc.is_balanced() and cc.running_count(deck) == 0

# 4. true count normalizes by decks remaining
assert cc.true_count(6, 1) == 6.0
assert math.isclose(cc.true_count(6, 3), 2.0)
assert cc.true_count(6, 6) < cc.true_count(6, 2)      # same RC, more decks -> weaker

# 5. edge is linear in true count, break-even at TC = +1
assert math.isclose(cc.player_edge_percent(0), -0.5)   # base house edge
assert math.isclose(cc.player_edge_percent(1), 0.0)    # break-even
assert math.isclose(cc.player_edge_percent(4), 1.5)    # +1.5%
assert cc.player_edge_percent(5) > cc.player_edge_percent(2)   # monotonic

# 6. bet ramp: minimum when edge <= 0, scales up when it's positive
assert cc.recommended_bet_units(0) == 1
assert cc.recommended_bet_units(1) == 1                # exactly break-even -> min
assert cc.recommended_bet_units(4) == 4                # advantage -> bet up
assert cc.recommended_bet_units(-3) == 1

# 7. Kelly fraction: zero at/below break-even, positive above, equals edge/100
assert cc.kelly_fraction(0) == 0.0 and cc.kelly_fraction(1) == 0.0
assert cc.kelly_fraction(-2) == 0.0
assert math.isclose(cc.kelly_fraction(4), cc.player_edge_percent(4) / 100)
assert cc.kelly_fraction(4) > cc.kelly_fraction(2) > 0

# 8. end-to-end tracker
res = cc.deal_and_count([5, 5, 6, 4, 3])               # all low -> positive count
assert res["running_count"] == 5 and res["true_count"] > 0

# 9. kwarg bounds
for bad in (lambda: cc.hi_lo_value(1),
            lambda: cc.hi_lo_value("Z"),
            lambda: cc.true_count(5, 0)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_card_counting: all checks passed")
