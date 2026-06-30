import pytest
import numpy as np
from dgs.cybersecurity import (
    caesar_cipher, vigenere_cipher, xor_cipher,
    sha256_hex, avalanche_demo,
    rsa_keygen, rsa_encrypt, rsa_decrypt,
    diffie_hellman_keygen, diffie_hellman_shared_secret,
    security_bits, time_to_crack_seconds, security_sweep,
    cybersecurity_sympy_5,
)


# -- classical ciphers --------------------------------------------------------

def test_caesar_round_trip():
    assert caesar_cipher(caesar_cipher("HELLO", 7), 7, decrypt=True) == "HELLO"


def test_caesar_rot13():
    assert caesar_cipher("ABC", 13) == "NOP"


def test_caesar_preserves_non_alpha():
    assert caesar_cipher("A B!", 1) == "B C!"


def test_vigenere_round_trip():
    ct = vigenere_cipher("PHYSICS", "KEY")
    assert vigenere_cipher(ct, "KEY", decrypt=True) == "PHYSICS"


def test_xor_self_inverse():
    msg = b"feynman diagrams"
    key = b"jalabi"
    assert xor_cipher(xor_cipher(msg, key), key) == msg


def test_xor_different_key_gives_different_ciphertext():
    msg = b"hello"
    assert xor_cipher(msg, b"key1") != xor_cipher(msg, b"key2")


# -- hash / avalanche ---------------------------------------------------------

def test_sha256_deterministic():
    assert sha256_hex("same input") == sha256_hex("same input")


def test_sha256_different_inputs_differ():
    assert sha256_hex("input1") != sha256_hex("input2")


def test_avalanche_roughly_half_bits_change():
    av = avalanche_demo("Jalali Lab TS-DFT phase retrieval system", flip_bit_index=3)
    assert 0.30 < av["fraction_changed"] < 0.70


def test_avalanche_returns_hex_hashes():
    av = avalanche_demo("test")
    assert len(av["hash_original"]) == 64
    assert len(av["hash_flipped"]) == 64


# -- RSA ----------------------------------------------------------------------

def test_rsa_round_trip():
    keys = rsa_keygen(bits=16)
    m = 7
    c = rsa_encrypt(m, keys["public"])
    assert rsa_decrypt(c, keys["private"]) == m


def test_rsa_encrypt_changes_value():
    keys = rsa_keygen(bits=16)
    m = 5
    c = rsa_encrypt(m, keys["public"])
    assert c != m


def test_rsa_message_too_large_raises():
    keys = rsa_keygen(bits=16)
    _, n = keys["public"]
    with pytest.raises(ValueError):
        rsa_encrypt(n + 1, keys["public"])


# -- Diffie-Hellman -----------------------------------------------------------

def test_dh_shared_secret_matches():
    alice = diffie_hellman_keygen()
    bob   = diffie_hellman_keygen(g=alice["g"], p=alice["p"])
    s_a = diffie_hellman_shared_secret(bob["public"], alice["private"], alice["p"])
    s_b = diffie_hellman_shared_secret(alice["public"], bob["private"], alice["p"])
    assert s_a == s_b


def test_dh_public_keys_differ():
    alice = diffie_hellman_keygen()
    bob   = diffie_hellman_keygen(g=alice["g"], p=alice["p"])
    assert alice["public"] != bob["public"]


# -- parameterized security model --------------------------------------------

def test_security_bits_symmetric_equals_key_bits():
    assert security_bits(128, "symmetric") == 128.0


def test_security_bits_rsa_less_than_key_bits():
    assert security_bits(2048, "rsa") < 2048


def test_security_bits_ecc_half_key_bits():
    assert security_bits(256, "ecc") == 128.0


def test_security_bits_invalid_cipher_raises():
    with pytest.raises(ValueError):
        security_bits(128, "vigenere")


def test_time_to_crack_increases_with_security_bits():
    t1 = time_to_crack_seconds(64)["years"]
    t2 = time_to_crack_seconds(128)["years"]
    assert t2 > t1


def test_security_sweep_returns_correct_shape():
    sw = security_sweep([64, 128, 256])
    assert len(sw["key_bits"]) == 3
    assert len(sw["years_to_crack"]) == 3


def test_security_sweep_monotone():
    sw = security_sweep([64, 128, 192, 256])
    years = sw["years_to_crack"]
    assert all(years[i] < years[i+1] for i in range(len(years)-1))


# -- sympy equations ----------------------------------------------------------

def test_cybersecurity_sympy_5_count_and_type():
    import sympy as sp
    eqs = cybersecurity_sympy_5()
    assert len(eqs) == 5
    for eq in eqs.values():
        assert isinstance(eq, sp.Eq)
