"""Smoke-test algebra.py: groups, rings, fields over finite sets."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs import algebra as al

# Z/n under addition: cyclic abelian group
for n in (5, 6, 26):
    g = al.is_abelian_group(al.Zn(n), al.add_mod(n))
    print(f"Z/{n} under + : abelian group? {g}")

# Z/n as a ring (add + mul)
for n in (5, 6, 26):
    r = al.is_ring(al.Zn(n), al.add_mod(n), al.mul_mod(n))
    f = al.is_field(al.Zn(n), al.add_mod(n), al.mul_mod(n))
    print(f"Z/{n} : ring? {r}   field? {f}   (prime? {al.is_prime(n)})")

# fields exactly at prime n
fields = [n for n in range(2, 14) if al.is_field(al.Zn(n), al.add_mod(n), al.mul_mod(n))]
primes = [n for n in range(2, 14) if al.is_prime(n)]
print("\nZ/n is a field exactly when n is prime:", fields == primes, fields)

# zero divisor in Z/6 breaks the field (2*3 = 0)
print("Z/6 zero divisor: 2*3 mod 6 =", al.mul_mod(6)(2, 3), "-> not a field")

# GF(2): the field behind binary codes
print("GF(2) field?", al.is_field([0, 1], al.add_mod(2), al.mul_mod(2)))

# symmetric group S_3: a non-abelian group
S3 = al.symmetric_group(3)
print(f"\nS_3 ({len(S3)} elements): group? {al.is_group(S3, al.compose)}  "
      f"abelian? {al.is_abelian_group(S3, al.compose)} (expect non-abelian)")

# parity group {+1,-1} ~ Z/2 (Griffiths pseudovectors)
pe, pop = al.parity_group()
print("parity group {+1,-1}: abelian group?", al.is_abelian_group(pe, pop),
      "| identity =", al.identity(pe, pop))

# Cayley table of Z/4 (cyclic -> diagonal stripes)
print("\nCayley table Z/4 under +:")
for row in al.cayley_table(al.Zn(4), al.add_mod(4)):
    print("  ", row)

# validation
for bad in [lambda: al.Zn(0)]:
    try:
        bad()
    except ValueError as e:
        print("err ok:", e)

assert fields == primes
assert al.is_group(S3, al.compose) and not al.is_abelian_group(S3, al.compose)
assert al.is_field([0, 1], al.add_mod(2), al.mul_mod(2))
print("SMOKE PASS")
