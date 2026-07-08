"""Card counting: turning a shoe of cards into an expected-value edge (Hi-Lo).

Blackjack has a small built-in house edge, but the edge SHIFTS as cards are dealt: a
deck rich in tens and aces favors the player (more blackjacks, dealer busts more), a
deck rich in low cards favors the house. Card counting is just tracking that shift with
one running number.

The Hi-Lo system assigns each rank a weight:
        2-6  -> +1   (low cards; their removal is GOOD for the player)
        7-9  ->  0   (neutral)
        10,J,Q,K,A -> -1   (high cards; their removal is BAD for the player)
The RUNNING COUNT is the sum of weights seen. Hi-Lo is a BALANCED system: a full
52-card deck has exactly twenty +1s (2-6, four each), twenty -1s (ten-ace), and twelve
0s, so a fully dealt deck returns the count to zero -- the property that makes the
count meaningful.

The running count must be normalized by how many cards are LEFT, because +5 with one
deck to go is far stronger than +5 with six decks to go:
        TRUE COUNT = running count / decks remaining.
The true count maps roughly linearly to the player's edge -- each +1 of true count is
worth about +0.5%, starting from the ~-0.5% base house edge, so the player breaks even
near true count +1 and has the advantage above it. Bet more when the count is high
(a Kelly-style bet ~ proportional to the edge), the minimum when it is not.

This is expected-value reasoning end to end (very much a Feynman "figure the odds"
exercise). Verified: the Hi-Lo weights, the balanced-deck property, true-count
normalization, the edge/break-even, and a dealt-deck simulation. NumPy-free; py-3.13.
"""

_LOW = {2, 3, 4, 5, 6}
_NEUTRAL = {7, 8, 9}
_HIGH = {10, 11, 12, 13, 14}          # 10, J, Q, K, A
_RANK_ALIASES = {"T": 10, "J": 11, "Q": 12, "K": 13, "A": 14, "10": 10}


def _rank(card):
    """Normalize a card to an integer rank 2..14 (J=11,Q=12,K=13,A=14)."""
    if isinstance(card, str):
        c = card.strip().upper()
        if c in _RANK_ALIASES:
            return _RANK_ALIASES[c]
        if c.isdigit():
            card = int(c)
        else:
            raise ValueError(f"unknown card {card!r}")
    if not (2 <= card <= 14):
        raise ValueError(f"rank {card} out of range 2..14")
    return card


def hi_lo_value(card):
    """The Hi-Lo weight of a card: +1 for 2-6, 0 for 7-9, -1 for 10-A."""
    r = _rank(card)
    if r in _LOW:
        return +1
    if r in _NEUTRAL:
        return 0
    return -1


def running_count(cards):
    """Sum of Hi-Lo weights over the cards seen so far."""
    return sum(hi_lo_value(c) for c in cards)


def full_deck():
    """One 52-card deck as integer ranks (each of 2..14 appears 4 times)."""
    return [r for r in range(2, 15) for _ in range(4)]


def is_balanced():
    """Hi-Lo is balanced: the weights over a full deck sum to zero (20 +1s,
    20 -1s, 12 zeros). This is why a fully dealt deck returns the count to 0."""
    return running_count(full_deck()) == 0


def true_count(run_count, decks_remaining):
    """Normalize the running count by decks left: TC = running / decks_remaining.
    +5 with 1 deck left is a much bigger edge than +5 with 5 decks left."""
    if decks_remaining <= 0:
        raise ValueError("decks_remaining must be positive")
    return run_count / decks_remaining


def player_edge_percent(tc, base_house_edge=0.5):
    """Approximate player advantage (%) as a linear function of the true count:
    edge ~= 0.5 * TC - base_house_edge. Break-even near TC = +1; positive above."""
    return 0.5 * tc - base_house_edge


def recommended_bet_units(tc):
    """A simple bet ramp: 1 unit (table minimum) when the edge is non-positive,
    else scale up with the true count (bet more when you have the advantage)."""
    if player_edge_percent(tc) <= 0:
        return 1
    return max(1, int(tc))


def kelly_fraction(tc):
    """Kelly bet fraction of bankroll ~ edge (blackjack variance is ~1 per unit),
    so bet the edge fraction when positive, nothing when the house is ahead."""
    edge = player_edge_percent(tc) / 100.0
    return max(0.0, edge)


def deal_and_count(cards):
    """Deal a list of cards, returning (running_count, decks_remaining_estimate,
    true_count) assuming a shoe that started at ceil(len/52) decks -- a small
    end-to-end tracker."""
    rc = running_count(cards)
    started_decks = max(1, round(len(cards) / 52) or 1)
    remaining = started_decks - len(cards) / 52
    tc = true_count(rc, remaining) if remaining > 0 else float(rc)
    return {"running_count": rc, "decks_remaining": remaining, "true_count": tc}


if __name__ == "__main__":
    print("Hi-Lo weights: 2-6 -> +1, 7-9 -> 0, 10-A -> -1")
    print(f"  2:{hi_lo_value(2):+d}  7:{hi_lo_value(7):+d}  K:{hi_lo_value('K'):+d}  "
          f"A:{hi_lo_value('A'):+d}")
    print(f"  balanced system (full deck sums to 0)? {is_balanced()}")

    seen = [5, 3, "K", 6, 2, 4, "A", 10, 6, 5]     # lots of low cards -> high count
    rc = running_count(seen)
    print(f"\ndealt {seen}")
    print(f"  running count = {rc:+d}")
    for decks in (1.0, 3.0, 6.0):
        tc = true_count(rc, decks)
        print(f"  {decks:.0f} decks left: true count {tc:+.2f}, "
              f"edge {player_edge_percent(tc):+.2f}%, bet {recommended_bet_units(tc)} unit(s)")

    print("\nedge vs true count (break-even near +1):")
    for tc in (-2, 0, 1, 2, 4):
        print(f"  TC={tc:+d}: edge {player_edge_percent(tc):+.2f}%, "
              f"Kelly {kelly_fraction(tc)*100:.2f}% of bankroll")
