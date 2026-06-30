"""Cybersecurity fundamentals: classical ciphers, symmetric XOR, RSA key
generation and encryption, Diffie-Hellman key exchange, hash avalanche
effect, and a parameterized security model that maps (key_bits, compute_ops)
to time-to-crack.

PARAMETERIZATION is the central theme: every security guarantee is a
function of parameters (key size, keyspace, attacker compute budget).
Sweeping those parameters reveals the security/cost tradeoff surface --
the same way sweeping dispersion D parameterizes GS convergence in this repo.

All cryptography here is EDUCATIONAL (pure Python/NumPy/SymPy, no hardening
against timing attacks or side channels). Do not use for real secrets.
"""
import hashlib
import os
import sympy as sp
import numpy as np


# -- Classical ciphers --------------------------------------------------------

def caesar_cipher(text, shift, decrypt=False):
    """Caesar cipher: shift each letter by `shift` positions mod 26."""
    shift = shift % 26
    if decrypt:
        shift = -shift % 26
    result = []
    for ch in text:
        if ch.isalpha():
            base = ord('A') if ch.isupper() else ord('a')
            result.append(chr((ord(ch) - base + shift) % 26 + base))
        else:
            result.append(ch)
    return ''.join(result)


def vigenere_cipher(text, key, decrypt=False):
    """Vigenere cipher: key-dependent shift per character (repeating key)."""
    key = key.upper()
    result = []
    ki = 0
    for ch in text:
        if ch.isalpha():
            shift = ord(key[ki % len(key)]) - ord('A')
            result.append(caesar_cipher(ch, shift, decrypt=decrypt))
            ki += 1
        else:
            result.append(ch)
    return ''.join(result)


def xor_cipher(plaintext_bytes, key_bytes):
    """Symmetric XOR cipher: ciphertext = plaintext XOR key (repeating key).
    XOR is its own inverse: decrypt by XORing the ciphertext with the same key."""
    key = key_bytes
    ct = bytes(plaintext_bytes[i] ^ key[i % len(key)] for i in range(len(plaintext_bytes)))
    return ct


# -- Hash / avalanche effect --------------------------------------------------

def sha256_hex(data):
    """SHA-256 hash of a string, returned as a hex digest."""
    return hashlib.sha256(data.encode()).hexdigest()


def avalanche_demo(message, flip_bit_index=0):
    """Show the avalanche effect: flipping ONE bit in the input changes ~50%
    of the hash bits. Returns original and flipped hashes + Hamming distance."""
    h1 = sha256_hex(message)
    msg_bytes = bytearray(message.encode())
    byte_idx, bit_pos = divmod(flip_bit_index, 8)
    if byte_idx < len(msg_bytes):
        msg_bytes[byte_idx] ^= (1 << bit_pos)
    h2 = sha256_hex(msg_bytes.decode(errors='replace'))
    bits1 = bin(int(h1, 16))[2:].zfill(256)
    bits2 = bin(int(h2, 16))[2:].zfill(256)
    hamming = sum(b1 != b2 for b1, b2 in zip(bits1, bits2))
    return {"hash_original": h1, "hash_flipped": h2,
            "hamming_distance": hamming, "fraction_changed": hamming / 256}


# -- RSA (educational, SymPy-based) ------------------------------------------

def rsa_keygen(bits=16):
    """Generate an RSA key pair using two random primes of ~bits/2 bits.
    Educational only -- real RSA uses 2048+ bits and cryptographic RNG."""
    half = bits // 2
    lo, hi = 2**(half-1), 2**half - 1
    # pick random primes in range
    p = sp.nextprime(int(lo + os.urandom(2)[0] % (hi - lo)))
    q = sp.nextprime(int(lo + os.urandom(2)[0] % (hi - lo)))
    while q == p:
        q = sp.nextprime(q)
    n = p * q
    phi = (p - 1) * (q - 1)
    e = 65537
    while sp.gcd(e, phi) != 1:
        e += 2
    d = int(sp.mod_inverse(e, phi))
    return {"public": (e, n), "private": (d, n), "p": p, "q": q, "phi": phi}


def rsa_encrypt(message_int, public_key):
    """RSA encryption: c = m^e mod n. message_int must be < n."""
    e, n = public_key
    if message_int >= n:
        raise ValueError(f"message {message_int} >= n={n}; use a larger key")
    return pow(message_int, e, n)


def rsa_decrypt(ciphertext_int, private_key):
    """RSA decryption: m = c^d mod n."""
    d, n = private_key
    return pow(ciphertext_int, d, n)


# -- Diffie-Hellman key exchange (educational) --------------------------------

def diffie_hellman_keygen(g=2, p=None):
    """Generate a DH private/public key pair over a prime field p.
    Default p is a small 32-bit safe prime (educational, not secure)."""
    if p is None:
        p = 4294967311  # next prime above 2^32
    private = int.from_bytes(os.urandom(4), 'big') % (p - 2) + 2
    public = pow(g, private, p)
    return {"g": g, "p": p, "private": private, "public": public}


def diffie_hellman_shared_secret(their_public, my_private, p):
    """shared_secret = their_public^my_private mod p. Both parties compute
    the same value without ever transmitting the private keys."""
    return pow(their_public, my_private, p)


# -- Parameterized security model --------------------------------------------

def security_bits(key_bits, cipher="symmetric"):
    """How many bits of security does a key of `key_bits` bits provide?
    symmetric: security_bits = key_bits (exhaustive search)
    RSA/DH: far fewer, because number-field sieve factoring beats exhaustive search.
    Uses NIST SP 800-57 approximations."""
    if cipher == "symmetric":
        return float(key_bits)
    elif cipher == "rsa":
        # NIST approximation: security ~ 1.92*(ln(n))^(1/3)*(ln(ln(n)))^(2/3)
        # simplified fit: security ≈ key_bits / 7.6 for key_bits in [1024,4096]
        n_bits = float(key_bits)
        if n_bits < 512:
            return n_bits / 10.0
        return n_bits / 7.6
    elif cipher == "ecc":
        # ECC: security ~ key_bits / 2
        return float(key_bits) / 2.0
    else:
        raise ValueError(f"unknown cipher type '{cipher}'; use symmetric/rsa/ecc")


def time_to_crack_seconds(security_bits_val, ops_per_second=1e12):
    """Time for an exhaustive attack given 2^security_bits work units and
    `ops_per_second` compute rate. Returns seconds; converts to human scale."""
    total_ops = 2.0 ** security_bits_val
    seconds = total_ops / ops_per_second
    minutes = seconds / 60
    hours   = minutes / 60
    days    = hours / 24
    years   = days / 365.25
    return {"seconds": seconds, "years": years, "minutes": minutes,
            "total_ops": total_ops, "ops_per_second": ops_per_second}


def security_sweep(key_bit_range, cipher="symmetric", ops_per_second=1e12):
    """Sweep key sizes -> security bits -> time-to-crack. The parameterization
    that answers 'how many bits do I need?' Returns arrays for plotting."""
    bits = np.array(key_bit_range, float)
    sec = np.array([security_bits(b, cipher) for b in bits])
    years = np.array([time_to_crack_seconds(s, ops_per_second)["years"] for s in sec])
    return {"key_bits": bits, "security_bits": sec, "years_to_crack": years}


def cybersecurity_sympy_5():
    """Five symbolic equations: RSA encrypt/decrypt, DH shared secret,
    XOR self-inverse, time-to-crack, and the exhaustive search bound."""
    m, e, d, n = sp.symbols('m e d n', positive=True, integer=True)
    g, a, b, p = sp.symbols('g a b p', positive=True, integer=True)
    s, r = sp.symbols('s r', positive=True)   # security bits, ops/s

    return {
        "RSA_encrypt":
            sp.Eq(sp.Symbol('c'), sp.Mod(m**e, n)),
        "RSA_decrypt":
            sp.Eq(m, sp.Mod(sp.Symbol('c')**d, n)),
        "DH_shared_secret":
            sp.Eq(sp.Symbol('K'), sp.Mod(sp.Symbol('A')**b, p)),
        "Exhaustive_search_time":
            sp.Eq(sp.Symbol('T'), 2**s / r),
        "XOR_self_inverse":
            sp.Eq(sp.Symbol('plaintext'),
                  sp.Symbol('ciphertext') ^ sp.Symbol('key')),
    }


if __name__ == "__main__":
    print("=== Caesar cipher ===")
    ct = caesar_cipher("HELLO WORLD", 13)
    pt = caesar_cipher(ct, 13, decrypt=True)
    print(f"  'HELLO WORLD' -> '{ct}' -> '{pt}'")

    print("\n=== XOR cipher ===")
    msg = b"photonics"
    key = b"secret"
    enc = xor_cipher(msg, key)
    dec = xor_cipher(enc, key)
    print(f"  plaintext={msg}, decrypted={dec}")

    print("\n=== SHA-256 avalanche effect ===")
    av = avalanche_demo("Jalali Lab GS phase retrieval", flip_bit_index=0)
    print(f"  flip 1 bit -> {av['fraction_changed']*100:.1f}% of hash bits changed")

    print("\n=== RSA keygen (16-bit educational) ===")
    keys = rsa_keygen(bits=16)
    m = 42
    c = rsa_encrypt(m, keys["public"])
    m2 = rsa_decrypt(c, keys["private"])
    print(f"  encrypt(42) = {c}, decrypt -> {m2}, round-trip ok: {m==m2}")

    print("\n=== Diffie-Hellman ===")
    alice = diffie_hellman_keygen()
    bob   = diffie_hellman_keygen(g=alice["g"], p=alice["p"])
    s_a = diffie_hellman_shared_secret(bob["public"], alice["private"], alice["p"])
    s_b = diffie_hellman_shared_secret(alice["public"], bob["private"], alice["p"])
    print(f"  shared secret match: {s_a == s_b}")

    print("\n=== Security parameterization sweep (symmetric) ===")
    sw = security_sweep([64, 80, 112, 128, 192, 256], "symmetric", ops_per_second=1e15)
    for kb, sec, yr in zip(sw["key_bits"], sw["security_bits"], sw["years_to_crack"]):
        print(f"  {int(kb)}-bit symmetric: {sec:.0f} security bits, {yr:.2e} years to crack at 1 petaop/s")

    print("\n=== SymPy 5 ===")
    for k, eq in cybersecurity_sympy_5().items():
        print(f"  {k}: {eq}")
