"""
repl/_repl_binary_vectors.py
Binary vectors: bits, Hamming, BPSK, one-hot, GS unit constraint.
The "straitjacket": constraining a vector to live on a set.
"""
import numpy as np
import sympy as sp
import pandas as pd
sp.init_printing(use_unicode=False, wrap_line=False)

print("=" * 60)
print("BINARY VECTORS + CONSTRAINTS")
print("=" * 60)
print()

# ============================================================
# 1. Binary vector basics
# ============================================================
print("=== 1. Binary Vector Basics ===")

# pack/unpack bits
x = np.array([0xDE, 0xAD, 0xBE, 0xEF], dtype=np.uint8)
bits = np.unpackbits(x)
print(f"bytes: {[hex(v) for v in x]}")
print(f"bits:  {bits}")
print(f"shape: {bits.shape}  (4 bytes x 8 bits = 32-bit vector)")
print()

# bit operations as vector ops
a = np.array([1,0,1,1,0,1,0,0], dtype=np.uint8)
b = np.array([1,1,0,1,0,0,1,0], dtype=np.uint8)
print(f"a     = {a}")
print(f"b     = {b}")
print(f"a AND b = {a & b}")
print(f"a OR  b = {a | b}")
print(f"a XOR b = {a ^ b}  <- Hamming distance = {np.sum(a^b)}")
print(f"NOT a   = {1 - a}   (1-a for binary)")
print()

# ============================================================
# 2. Hamming distance and weight
# ============================================================
print("=== 2. Hamming Distance + Weight ===")
print("""
Hamming distance d(a,b): number of positions where a and b differ
  d(a,b) = sum(a XOR b) = popcount(a XOR b)

Hamming weight w(a): number of 1s
  w(a)   = sum(a) = popcount(a)

Error correction:
  d >= 2t+1 -> can correct t bit-flip errors
  d >= t+1  -> can detect t errors

Repetition code (n=3): 0->000, 1->111
  d_min = 3 -> corrects 1 error
  receive 001 -> majority vote -> 0 (one flip corrected)
""")

# build distance matrix for all 3-bit codewords
n = 3
codewords = np.array([[int(b) for b in format(i, f'0{n}b')] for i in range(2**n)], dtype=np.uint8)
D = np.zeros((2**n, 2**n), dtype=int)
for i in range(2**n):
    for j in range(2**n):
        D[i,j] = int(np.sum(codewords[i] ^ codewords[j]))

labels = [format(i, f'0{n}b') for i in range(2**n)]
df_d = pd.DataFrame(D, index=labels, columns=labels)
print("3-bit Hamming distance matrix:")
print(df_d.to_string())
print()

# ============================================================
# 3. The straitjacket: constraint sets
# ============================================================
print("=== 3. The Straitjacket: Constraint Sets ===")
print("""
A straitjacket in signal processing = a hard constraint set C
  Projection P_C(x): find the closest point in C to x

Three straitjackets in your project:

  C_binary  = {0,1}^N      each element is 0 or 1
    P(x_i) = round(x_i)    project to nearest bit

  C_unit    = {z : |z|=1}  unit circle in complex plane
    P(z) = z/|z| = exp(i*angle(z))    <- GS constraint

  C_intensity = {E : |disperse(E,D)|^2 = I}  measured intensity
    P(E) = sqrt(I) * exp(i*angle(disperse(E,D)))  <- GS constraint

GS alternates: E -> P_C2(P_C1(E)) -> fixed point = solution
Binary LDPC:   v -> P_check(P_bit(v))  -> same structure
JPEG:          coeff -> P_quantize(P_DCT(block)) -> same structure
""")

# numerical: projection onto unit circle
rng = np.random.default_rng(42)
z = rng.normal(size=6) + 1j*rng.normal(size=6)   # random complex
z_proj = z / np.abs(z)   # project onto unit circle

print("Projection onto unit circle |z|=1:")
print(f"{'z':>25}  {'|z|':>6}  {'proj(z)':>25}  {'|proj|':>6}")
for zi, zp in zip(z, z_proj):
    print(f"  {zi.real:+.3f}{zi.imag:+.3f}j  {abs(zi):>6.3f}  "
          f"  {zp.real:+.3f}{zp.imag:+.3f}j  {abs(zp):>6.3f}")
print()

# ============================================================
# 4. BPSK: binary -> unit circle -> binary
# ============================================================
print("=== 4. BPSK: binary <-> unit circle ===")
print("""
Binary phase-shift keying:
  bit 0 -> +1  = exp(i*0)
  bit 1 -> -1  = exp(i*pi)

  Transmitted:  s = 1 - 2*b   (maps {0,1} -> {+1,-1})
  Received:     r = s + noise
  Decoded:      b = (sign(r) < 0)  -> back to {0,1}

This is the simplest straitjacket: project real line onto {-1,+1}
""")

bits_tx = np.array([0,1,0,0,1,1,0,1])
s_tx    = 1 - 2*bits_tx.astype(float)   # BPSK symbols
noise   = rng.normal(0, 0.3, len(s_tx)) # AWGN
r_rx    = s_tx + noise
bits_rx = (r_rx < 0).astype(int)
ber     = np.mean(bits_tx != bits_rx)

df_bpsk = pd.DataFrame({
    'bit_tx': bits_tx, 'symbol': s_tx.astype(int),
    'received': np.round(r_rx, 3), 'bit_rx': bits_rx,
    'error': (bits_tx != bits_rx).astype(int)
})
print(df_bpsk.to_string(index=False))
print(f"\nBER = {ber:.3f}  (sigma=0.3, SNR~10 dB)")
print()

# ============================================================
# 5. One-hot encoding: sparse binary vector
# ============================================================
print("=== 5. One-hot: sparse binary vector ===")
print("""
One-hot vector: exactly one 1, rest 0s
  class k -> e_k = [0,...,0,1,0,...,0]  (1 at position k)

Used in:
  neural nets: output layer softmax -> argmax -> one-hot
  digital logic: decoder output (3->8 decoder)
  attention: hard attention = one-hot over sequence positions
  GS: if you discretize phase to N levels -> one-hot per sample
""")

def one_hot(k, N):
    v = np.zeros(N, dtype=int)
    v[k] = 1
    return v

N_classes = 8
print(f"3->8 decoder (one-hot, N={N_classes}):")
for k in range(N_classes):
    binary_in = format(k, '03b')
    oh = one_hot(k, N_classes)
    print(f"  {binary_in} -> {oh}")
print()

# ============================================================
# 6. Binary vector in GS context: phase quantization
# ============================================================
print("=== 6. Phase quantization: continuous -> binary ===")
print("""
GS recovers continuous phase phi in [-pi, pi]
Quantize to B bits -> 2^B phase levels

  phi_q = round(phi / (2*pi) * 2^B) * (2*pi) / 2^B

B=1: BPSK  {0, pi}
B=2: QPSK  {0, pi/2, pi, 3pi/2}
B=4: 16PSK (standard optical modulation)
B=8: 256 levels -> < 0.025 rad quantization error

Quantization noise (uniform):
  sigma_q^2 = (delta)^2 / 12  where delta = 2*pi / 2^B
""")

phi_true = rng.uniform(-np.pi, np.pi, 1000)
rows_q = []
for B in [1, 2, 4, 8]:
    levels = 2**B
    delta  = 2*np.pi / levels
    phi_q  = np.round(phi_true / delta) * delta
    # wrap to [-pi, pi]
    phi_q  = (phi_q + np.pi) % (2*np.pi) - np.pi
    rms_err = np.sqrt(np.mean((phi_true - phi_q)**2))
    sigma_q  = delta / np.sqrt(12)
    rows_q.append({'B': B, 'levels': levels, 'delta_deg': round(np.degrees(delta),2),
                   'RMS_err_rad': round(rms_err,4), 'theory_rad': round(sigma_q,4)})
print(pd.DataFrame(rows_q).to_string(index=False))
print()
print("RMS ~ theory (delta/sqrt(12)): uniform quantization noise confirmed")
print()

# ============================================================
# 7. SymPy: binary polynomial (GF2 arithmetic)
# ============================================================
print("=== 7. GF(2) arithmetic: binary polynomials ===")
print("""
GF(2): arithmetic mod 2
  addition    = XOR  (no carry)
  subtraction = XOR  (same as addition, -1 = 1 mod 2)
  multiply    = AND

CRC (cyclic redundancy check): polynomial division over GF(2)
  generator poly G(x) = x^4 + x + 1  (CRC-4)
  message M(x) = x^3 + x + 1  (binary: 1011)

  remainder R = M * x^4 mod G   <- 4-bit CRC appended to message
""")

# GF(2) polynomial multiplication (manual, no sympy for GF2 easily)
def gf2_mod(a, g):
    """Divide polynomial a by g over GF(2), return remainder."""
    a = int(a); g = int(g)
    dg = g.bit_length() - 1
    while a.bit_length() > dg:
        shift = a.bit_length() - g.bit_length()
        a ^= (g << shift)
    return a

G = 0b10011    # x^4 + x + 1  (CRC-4-ITU)
msgs = [0b1011, 0b1101, 0b1111, 0b1000]
print(f"Generator G = {bin(G)} (x^4 + x + 1)")
print(f"{'msg':>8}  {'msg<<4':>8}  {'CRC':>8}  {'codeword':>12}")
for m in msgs:
    m_shifted = m << 4
    crc = gf2_mod(m_shifted, G)
    codeword = m_shifted | crc
    print(f"  {bin(m):>8}  {bin(m_shifted):>10}  {bin(crc):>6}  {bin(codeword):>12}")
print()
print("Append CRC to message. Receiver: codeword mod G = 0 if no errors.")
