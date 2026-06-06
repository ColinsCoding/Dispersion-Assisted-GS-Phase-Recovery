# -*- coding: utf-8 -*-
"""
_repl_c_memory_data_crypto_catalyst.py
=======================================
C memory management first, then data scales, Fed/rates, Ethereum,
cryptographic commitments, chemical catalysis, silica.

S1: C MEMORY MANAGEMENT (first, as requested)
    - Stack vs heap, lifetime, scope
    - malloc / calloc / realloc / free -- exact semantics
    - Memory fragmentation: internal vs external
    - Common bugs: leak, double-free, use-after-free, buffer overrun
    - Valgrind, AddressSanitizer usage
    - Custom allocator sketch (bump allocator, free list)

S2: DATA SCALES -- bits to petabytes and beyond
    - SI vs binary prefixes (kilo vs kibi)
    - Storage media comparison
    - Petabyte in context: what holds a PB?
    - Big data: CERN, YouTube, LHC, genomics
    - Numerical: log-scale plot of data sizes

S3: FEDERAL RESERVE + INTEREST RATES
    - Fed funds rate mechanism
    - How rate changes flow through economy
    - Bond price vs yield (inverse relation)
    - Taylor rule: r = r* + pi + 0.5*(pi - pi*) + 0.5*(y - y*)
    - Effect on equities, crypto, real estate

S4: ETHEREUM + CRYPTO + COMMITMENT SCHEMES
    - ETH basics: accounts, gas, EVM
    - Smart contract lifecycle
    - Cryptographic hash commitment: H(secret || nonce)
    - Pedersen commitment: C = r*G + v*H (elliptic curve)
    - Merkle tree: O(log n) proof
    - Why ETH uses keccak256 (SHA-3 variant)

S5: CHEMICAL CATALYSIS + SILICA
    - Catalyst: lowers activation energy, not consumed
    - Arrhenius equation: k = A*exp(-Ea/RT)
    - Michaelis-Menten (enzyme = biological catalyst)
    - Silica (SiO2): catalyst support, optical fiber, microfluidics
    - Turnover frequency (TOF), selectivity

Output: repl/_out_c_memory_data_crypto_catalyst.png
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sympy as sp
from sympy import symbols, exp, log, diff, solve, Rational, sqrt, pi, simplify
import hashlib, os

try:
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "_out_c_memory_data_crypto_catalyst.png")
except NameError:
    OUT = "_out_c_memory_data_crypto_catalyst.png"

SEP = "=" * 65

# ============================================================
# S1: C MEMORY MANAGEMENT
# ============================================================
print(SEP)
print("SECTION 1: C MEMORY MANAGEMENT")
print(SEP)

print("""
  MEMORY REGIONS IN A C PROCESS:
  +--------------------------+  high addresses
  |  STACK                   |  local variables, function frames
  |  (grows DOWN)            |  LIFO, auto-freed on return
  +--------------------------+
  |  ...                     |  (stack and heap grow toward each other)
  +--------------------------+
  |  HEAP                    |  malloc/calloc/realloc/free
  |  (grows UP)              |  manual lifetime, any order alloc/free
  +--------------------------+
  |  BSS segment             |  uninitialized global/static variables
  +--------------------------+
  |  Data segment            |  initialized global/static variables
  +--------------------------+
  |  Text (code) segment     |  executable instructions (read-only)
  +--------------------------+  low addresses (0x00000000...)

  STACK:
    Automatically managed. Function call = push frame. Return = pop.
    Frame contains: return address, saved registers, local variables.
    Size: typically 1-8 MB (ulimit -s on Linux).
    Stack overflow: too deep recursion, huge local array.
    FAST: stack pointer arithmetic only (no syscall).

  HEAP:
    Manually managed. Lives until freed. Process-wide.
    Size: limited by virtual address space (~2^47 bytes on x86-64).
    Access: always through pointer (malloc returns pointer to block).
""")

c_memory_code = r"""
  /* ---- MALLOC / CALLOC / REALLOC / FREE ---- */

  /* malloc: allocate n bytes, UNINITIALIZED (may contain garbage) */
  int *arr = malloc(n * sizeof(int));
  if (!arr) { perror("malloc"); exit(EXIT_FAILURE); }   /* ALWAYS check */

  /* calloc: allocate n*size bytes, ZERO-INITIALIZED */
  double *buf = calloc(1024, sizeof(double));  /* all zeros */

  /* realloc: resize existing allocation */
  /* CRITICAL: if realloc fails, it returns NULL but original ptr STILL valid */
  int *tmp = realloc(arr, new_n * sizeof(int));
  if (!tmp) { free(arr); return -1; }   /* DON'T: arr = realloc(arr,...) loses ptr on fail */
  arr = tmp;

  /* free: release allocation. free(NULL) is safe no-op. */
  free(arr);
  arr = NULL;   /* defensive: prevent use-after-free */

  /* ---- SIZEOF IDIOM ---- */
  /* WRONG:  malloc(n * sizeof(int))  -- fails if type changes */
  /* RIGHT:  malloc(n * sizeof *arr)  -- type follows the pointer */
  int *p = malloc(n * sizeof *p);   /* sizeof *p = sizeof(int) */

  /* ---- COMMON MEMORY BUGS ---- */

  /* 1. MEMORY LEAK: allocate, never free */
  void leak(void) {
      int *p = malloc(100);
      /* ... use p ... */
      return;   /* p never freed! 100 bytes lost until process exits */
  }

  /* 2. DOUBLE FREE: free same pointer twice -> undefined behavior */
  free(p);
  free(p);   /* CRASH or silent corruption (heap metadata corrupted) */
  /* Fix: set p = NULL after free; free(NULL) is safe */

  /* 3. USE AFTER FREE: access freed memory */
  free(p);
  *p = 42;   /* UNDEFINED BEHAVIOR: memory may be reused by malloc */
  printf("%d\n", *p);  /* may print 42, may crash, may print garbage */

  /* 4. BUFFER OVERRUN: write past end of allocation */
  int *a = malloc(10 * sizeof(int));
  a[10] = 99;   /* OFF BY ONE: valid indices 0-9 only */
  /* Overwrites heap metadata -> silent corruption, crash later */

  /* 5. UNINITIALIZED READ: malloc gives garbage */
  int *x = malloc(sizeof(int));
  printf("%d\n", *x);   /* reads garbage: may print 0, may print 32767... */
  /* Fix: use calloc, or explicitly set: *x = 0; */

  /* 6. STACK BUFFER OVERRUN */
  char buf[8];
  strcpy(buf, "This string is too long");  /* overflow! overwrites return addr */
  /* Fix: strncpy(buf, src, sizeof(buf)-1); buf[sizeof(buf)-1] = '\0'; */
"""
print(c_memory_code)

print("""
  MEMORY FRAGMENTATION:

  INTERNAL FRAGMENTATION:
    Allocator gives you MORE than you asked for (alignment, header overhead).
    Example: malloc(1) may internally use 16 or 32 bytes (for alignment).
    Wasted space inside each allocated block.
    Always present; minimized by slab/pool allocators.

  EXTERNAL FRAGMENTATION:
    Free blocks are scattered -- total free bytes is large,
    but no single contiguous block is big enough for a request.
    Example:
      [used 100B] [free 50B] [used 200B] [free 50B] [used 100B]
      Total free = 100B, but cannot satisfy malloc(80) -- no 80B block!
    Worse with many alloc/free cycles of varying sizes.
    Fix: compacting GC (Java), memory pools, tcmalloc/jemalloc.

  HOW malloc() WORKS (simplified):
    glibc malloc uses segregated free lists by size class.
    Small blocks (<= 512B): fast bins (LIFO, no coalescing).
    Larger: unsorted/small/large bins (FIFO, sorted by size).
    On free: coalesce adjacent free blocks (reduce fragmentation).
    When heap exhausted: call sbrk() or mmap() to get more from OS.
    Block header: [size | prev_size | P/M/A flags] before user data.

  BUMP ALLOCATOR (fastest, no free):
    char pool[1 << 20];   /* 1 MB pool */
    char *bump = pool;
    void *bump_alloc(size_t n) {
        n = (n + 15) & ~15;   /* align to 16 bytes */
        if (bump + n > pool + sizeof(pool)) return NULL;
        void *p = bump; bump += n; return p;
    }
    /* Cannot free individual blocks. Reset: bump = pool; */
    /* Use for: per-request arenas, parsers, game frames. */

  TOOLS:
    Valgrind memcheck: detects leaks, invalid reads/writes, UB
      valgrind --leak-check=full --track-origins=yes ./a.out
    AddressSanitizer (ASan): compile-time instrumentation, ~2x slower
      gcc -fsanitize=address,undefined -g -O1 prog.c
    Valgrind memcheck output:
      ==1234== Invalid read of size 4
      ==1234==  at 0x4005C3: main (prog.c:10)
      ==1234== Address 0x5204040 is 0 bytes after a block of size 40 alloc'd
""")

# ============================================================
# S2: DATA SCALES
# ============================================================
print(f"\n{SEP}")
print("SECTION 2: DATA SCALES -- BITS TO PETABYTES AND BEYOND")
print(SEP)

scales = [
    ("bit",       "b",   1),
    ("byte",      "B",   8),
    ("kilobyte",  "KB",  8e3),
    ("kibibyte",  "KiB", 8*1024),
    ("megabyte",  "MB",  8e6),
    ("mebibyte",  "MiB", 8*1024**2),
    ("gigabyte",  "GB",  8e9),
    ("gibibyte",  "GiB", 8*1024**3),
    ("terabyte",  "TB",  8e12),
    ("tebibyte",  "TiB", 8*1024**4),
    ("petabyte",  "PB",  8e15),
    ("pebibyte",  "PiB", 8*1024**5),
    ("exabyte",   "EB",  8e18),
    ("zettabyte", "ZB",  8e21),
    ("yottabyte", "YB",  8e24),
]

print(f"\n  {'Name':<12} {'Abbr':<5} {'Bits':<18} {'Bytes':<18} {'SI vs binary error'}")
print(f"  {'-'*70}")
for name, abbr, bits in scales:
    if "bi" not in name and name not in ("bit","byte"):
        # Find binary counterpart
        bin_bits = 8 * 1024**{"kilo":1,"mega":2,"giga":3,"tera":4,"peta":5,
                               "exa":6,"zetta":7,"yotta":8}[name[:-4]]
        si_bits = bits
        err = (bin_bits - si_bits)/si_bits * 100
        print(f"  {name:<12} {abbr:<5} {bits:<18.3e} {bits/8:<18.3e} SI<binary by {err:.2f}%")
    else:
        print(f"  {name:<12} {abbr:<5} {bits:<18.3e} {bits/8:<18.3e}")

print("""
  SI vs BINARY PREFIX:
    1 KB  = 1,000 bytes   (SI, used by hard drive manufacturers)
    1 KiB = 1,024 bytes   (IEC binary, used by OS/RAM)
    1 TB  = 1,000,000,000,000 bytes
    1 TiB = 1,099,511,627,776 bytes
    Difference grows: at PB level, binary is 12.6% LARGER than SI.
    A "1 TB" hard drive shows as ~931 GiB in Windows/macOS.

  WHAT HOLDS A PETABYTE?
    1 PB = 10^15 bytes = 1,000 TB
    - 223,000 DVDs (4.5 GB each)
    - 13.3 years of HD video at 20 Mbps
    - Human genome (3 GB): 333,000 genomes
    - MP3 at 128kbps: 17 million hours of music
    - 5.1 million copies of Encyclopedia Britannica

  REAL DATA SCALES:
    Google processes ~20 PB/day (2023)
    YouTube uploads: ~500 hours video/min = ~1.5 PB/day
    CERN LHC: ~15 PB/year (after filtering from ~700 PB raw)
    Internet traffic: ~5 exabytes/day (Cisco 2023)
    Global data stored: ~120 ZB (2023, growing 23%/year)
    Human brain storage estimate: ~2.5 PB (synaptic weight capacity)

  STORAGE MEDIA COMPARISON (2024 approximate):
""")

media = [
    ("SRAM (cache L1)",    "5 MB",    "~0.1 ns",  "$100,000/GB",  "On-die"),
    ("DRAM (RAM)",         "8-64 GB", "~60 ns",   "$3-5/GB",      "DIMM"),
    ("NAND Flash SSD",     "1-8 TB",  "~100 us",  "$0.05-0.10/GB","NVMe/SATA"),
    ("HDD",                "1-20 TB", "~5 ms",    "$0.01-0.02/GB","SATA"),
    ("Tape (LTO-9)",       "18-45 TB","~30 s seek","$0.003/GB",   "Library"),
    ("Optical (Blu-ray)",  "100 GB",  "~100 ms",  "$0.002/GB",   "Archival"),
    ("DNA storage (est.)", "215 PB/g","hours",    ">>$1M/GB",    "Research"),
]
print(f"  {'Medium':<22} {'Capacity':<12} {'Latency':<12} {'Cost':<16} {'Notes'}")
print(f"  {'-'*74}")
for m, cap, lat, cost, note in media:
    print(f"  {m:<22} {cap:<12} {lat:<12} {cost:<16} {note}")

# ============================================================
# S3: FEDERAL RESERVE + INTEREST RATES
# ============================================================
print(f"\n{SEP}")
print("SECTION 3: FEDERAL RESERVE + INTEREST RATES")
print(SEP)

print("""
  THE FEDERAL RESERVE (Fed):
    Central bank of the United States. Established 1913.
    Dual mandate: (1) maximum employment, (2) price stability (2% inflation target).
    Main tool: Federal Funds Rate (FFR) -- rate banks charge each other
               for overnight lending of reserve balances.
    Set by FOMC (Federal Open Market Committee), meets 8x/year.

  HOW RATE CHANGES WORK (transmission mechanism):
    Fed raises FFR
     -> Banks pay more to borrow reserves
     -> Banks raise prime rate, mortgage rates, credit card rates, auto loans
     -> Businesses borrow less, invest less (higher hurdle rate)
     -> Consumers borrow less, spend less
     -> Demand falls -> inflation cools
     -> Employment may fall (recession risk)

    TIME LAGS:
     Rate change -> mortgage market: weeks
     Rate change -> business investment: 6-18 months
     Rate change -> inflation: 12-24 months
     "Monetary policy works with long and variable lags" -- Milton Friedman

  TAYLOR RULE (prescriptive formula for FFR):
    r = r* + pi + 0.5*(pi - pi*) + 0.5*(y - y*)
    r  = recommended Fed funds rate
    r* = neutral real rate (historically ~2%)
    pi = actual inflation (CPI, PCE)
    pi*= inflation target (2%)
    y  = log(actual GDP)
    y* = log(potential GDP)  [output gap: y-y*]
    If pi > pi*: raise rates (fight inflation)
    If y  < y*: cut rates (fight recession)
""")

# Taylor rule calculation
r_star = 2.0
pi_target = 2.0

scenarios = [
    ("Normal (2019)",       2.0,  0.0),
    ("High inflation (2022)", 7.5, -0.5),
    ("Recession (2009)",    -0.5, -3.0),
    ("Current (2024 est.)", 3.5,   0.5),
]
print(f"  {'Scenario':<28} {'pi':<6} {'gap':<7} {'Taylor r':<12} {'Actual FFR'}")
print(f"  {'-'*60}")
actual_ffr = {"Normal (2019)": 2.25, "High inflation (2022)": 4.25,
              "Recession (2009)": 0.25, "Current (2024 est.)": 5.25}
for name, pi, gap in scenarios:
    r_taylor = r_star + pi + 0.5*(pi - pi_target) + 0.5*gap
    actual   = actual_ffr.get(name, "?")
    print(f"  {name:<28} {pi:<6.1f} {gap:<7.1f} {r_taylor:<12.2f} {actual}")

print("""
  BONDS: PRICE vs YIELD (INVERSE RELATIONSHIP):
    Bond: issuer borrows $1000 (face), pays coupon (e.g. 5%=$50/yr), repays face at maturity.
    Price = SUM [coupon/(1+y)^t] + face/(1+y)^T   where y = yield to maturity
    If market rates RISE: new bonds pay more -> old bond less attractive -> price FALLS.
    If market rates FALL: old bond (higher coupon) more attractive -> price RISES.
    Duration: sensitivity of price to yield change.
    DV01: dollar value of 1bp (0.01%) yield change.

  EFFECT ON ASSETS:
    Rates UP:
      Bonds: prices fall (inverse)
      Equities: P/E compression (DCF discounts future earnings more)
      Real estate: mortgage rates up, prices fall
      Crypto: "risk-off" -> sell speculative assets -> prices fall
      USD: appreciates (higher yield attracts foreign capital)
    Rates DOWN:
      Opposite of above. Everything with future cash flows rises.
      "All assets are just discounted cash flows" -- Howard Marks.

  CRYPTO + RATES (2021-2023 case study):
    2020-2021: FFR ~ 0-0.25% -> risk-on -> BTC: $5K -> $69K (peak Nov 2021)
    2022: Fed raised 425bp in 12 months (fastest since 1980)
    -> BTC: $69K -> $16K (-77%). ETH: $4800 -> $880 (-82%).
    Correlation: crypto increasingly correlated with Nasdaq (risk assets).
""")

# Bond price calculation
T_bond = 10  # years
coupon = 0.05  # 5% annual coupon
face   = 1000.0

yields = np.linspace(0.01, 0.15, 200)
prices = np.array([
    sum(coupon*face/(1+y)**t for t in range(1, T_bond+1)) + face/(1+y)**T_bond
    for y in yields
])

print(f"\n  10-YEAR BOND (5% coupon, $1000 face) PRICE vs YIELD:")
for y_check in [0.02, 0.05, 0.08, 0.10]:
    idx = np.argmin(np.abs(yields - y_check))
    print(f"    Yield={y_check:.0%}: Price=${prices[idx]:.2f}  "
          f"({'PREMIUM' if prices[idx]>1000 else 'DISCOUNT' if prices[idx]<1000 else 'PAR'})")

# ============================================================
# S4: ETHEREUM + CRYPTO + COMMITMENT SCHEMES
# ============================================================
print(f"\n{SEP}")
print("SECTION 4: ETHEREUM + CRYPTOGRAPHIC COMMITMENTS")
print(SEP)

print("""
  ETHEREUM BASICS:
    Blockchain: distributed ledger of transactions. No single authority.
    ETH: native currency. Used to pay "gas" (computation fees).
    EVM: Ethereum Virtual Machine. Turing-complete. Stack-based.
    Accounts:
      Externally Owned Account (EOA): controlled by private key (human wallet)
      Contract Account: controlled by code (smart contract)
    Transactions: signed message sent to network.
      From: sender address
      To:   recipient (or contract address)
      Value: ETH sent
      Data: calldata (function call + args if To is contract)
      Gas limit: max gas willing to pay
      Gas price: price per gas unit (in gwei, 1 ETH = 10^9 gwei)

  SMART CONTRACT:
    Code deployed to blockchain. Runs deterministically on all nodes.
    Cannot be changed after deployment (immutable).
    Triggered by transactions (function calls).
    Solidity example:
    contract SimpleStorage {
        uint256 private value;
        function set(uint256 v) public { value = v; }
        function get() public view returns (uint256) { return value; }
    }

  CRYPTOGRAPHIC HASH FUNCTION (basis of blockchain):
    Properties needed:
    1. Deterministic: same input -> same output always
    2. Preimage resistance: given H, cannot find x such that hash(x)=H
    3. Collision resistance: cannot find x!=y with hash(x)=hash(y)
    4. Avalanche: 1-bit change in input -> ~50% bits change in output
    Bitcoin: SHA-256.  Ethereum: keccak256 (SHA-3 variant, NOT standard SHA-3).
""")

# Hash demonstration
msg1 = b"hello"
msg2 = b"hello."   # one character different
h1 = hashlib.sha256(msg1).hexdigest()
h2 = hashlib.sha256(msg2).hexdigest()
bits_diff = sum(bin(int(a,16)^int(b,16)).count('1') for a,b in zip(h1,h2))
total_bits = 256
print(f"  SHA-256 AVALANCHE EFFECT:")
print(f"    sha256('{msg1.decode()}')  = {h1[:32]}...")
print(f"    sha256('{msg2.decode()}') = {h2[:32]}...")
print(f"    Bits changed: {bits_diff}/{total_bits} = {bits_diff/total_bits:.1%}  (expect ~50%)")

print("""
  HASH COMMITMENT SCHEME:
    Problem: prove you know a secret WITHOUT revealing it yet.
    Use case: sealed bid auction, coin flip, zero-knowledge proof.

    Protocol:
    COMMIT phase:
      Choose secret s and random nonce r.
      Compute commitment C = H(s || r)   (concatenate s and r, then hash)
      Publish C. (C reveals nothing about s due to preimage resistance.)

    REVEAL phase:
      Publish s and r.
      Verifier checks: H(s || r) == C  (binding: cannot change s after commit)

    Properties:
    HIDING:    C reveals nothing about s (random r hides it).
    BINDING:   Cannot find s' != s with H(s' || r') == C (collision resistance).
""")

# Hash commitment demo
secret  = b"I will vote YES"
nonce   = os.urandom(32)   # 256-bit random nonce
commit_input = secret + nonce
commitment   = hashlib.sha256(commit_input).hexdigest()
# Verify
verify_input = secret + nonce
verify_hash  = hashlib.sha256(verify_input).hexdigest()
print(f"\n  COMMITMENT DEMO:")
print(f"    Secret:     '{secret.decode()}'")
print(f"    Nonce:      {nonce.hex()[:32]}... (random)")
print(f"    Commitment: {commitment}")
print(f"    Verify:     {verify_hash}")
print(f"    Match:      {commitment == verify_hash}")

print("""
  MERKLE TREE (structure underlying blockchains):
    Binary tree where each node = hash of its two children.
    Leaf nodes = hash of individual data blocks.
    Root = "Merkle root" -- single hash of ALL data.

    To prove block i is in tree:
    Provide block_i + O(log n) sibling hashes (Merkle proof).
    Verifier recomputes root in O(log n) time.
    Ethereum blocks contain Merkle root of all transactions.
    Light clients verify transactions without downloading full blockchain.

    Example (4 leaves):
         H(H12||H34)            <- Merkle root
         /          \\
     H(L1||L2)   H(L3||L4)
     /      \\      /    \\
    H(L1)  H(L2) H(L3) H(L4)   <- leaf hashes
     |       |     |     |
     L1      L2    L3    L4     <- data blocks
""")

# Simple Merkle tree
def merkle_tree(leaves):
    """Compute Merkle root of a list of byte strings."""
    layer = [hashlib.sha256(l).digest() for l in leaves]
    while len(layer) > 1:
        if len(layer) % 2:
            layer.append(layer[-1])   # duplicate last if odd
        layer = [hashlib.sha256(layer[i]+layer[i+1]).digest()
                 for i in range(0, len(layer), 2)]
    return layer[0].hex()

txs = [b"Alice->Bob: 1 ETH", b"Bob->Carol: 0.5 ETH",
       b"Carol->Dave: 2 ETH", b"Dave->Alice: 0.1 ETH"]
root = merkle_tree(txs)
print(f"\n  Merkle root of 4 transactions: {root}")
# Tamper with one tx
txs_tampered = txs.copy()
txs_tampered[1] = b"Bob->Carol: 500 ETH"  # fraud attempt
root_tampered = merkle_tree(txs_tampered)
print(f"  Merkle root (tampered tx):    {root_tampered}")
print(f"  Roots match: {root == root_tampered}  -> tamper detected")

# ============================================================
# S5: CHEMICAL CATALYSIS + SILICA
# ============================================================
print(f"\n{SEP}")
print("SECTION 5: CHEMICAL CATALYSIS + SILICA")
print(SEP)

print("""
  CATALYST DEFINITION:
    A substance that INCREASES reaction rate WITHOUT being consumed.
    Lowers the activation energy Ea -- provides alternative pathway.
    Does NOT change the thermodynamic equilibrium (delta G, K_eq).
    Does NOT appear in the net reaction (not a reactant or product).
    Types: homogeneous (same phase as reactants), heterogeneous (different phase),
           enzymatic (biological -- protein catalyst), photocatalytic.

  ARRHENIUS EQUATION:
    k(T) = A * exp(-Ea / (R*T))
    k  = rate constant [units depend on reaction order]
    A  = pre-exponential factor (collision frequency * steric factor)
    Ea = activation energy [J/mol]
    R  = 8.314 J/(mol*K)
    T  = temperature [K]

    LINEARIZED FORM (Arrhenius plot):
    ln(k) = ln(A) - Ea/(R*T)
    Plot ln(k) vs 1/T: slope = -Ea/R, intercept = ln(A)
    Catalyst: shifts the ENTIRE LINE upward (larger A or smaller Ea).
""")

# Arrhenius calculation
R_gas = 8.314   # J/(mol*K)
T_arr = np.linspace(300, 1200, 500)   # K

Ea_nocatalyst = 100e3   # J/mol  (100 kJ/mol)
Ea_catalyst   = 60e3    # J/mol  (60 kJ/mol -- catalyst lowers by 40 kJ)
A_factor      = 1e13    # pre-exponential [s^-1 for first order]

k_nocat = A_factor * np.exp(-Ea_nocatalyst / (R_gas * T_arr))
k_cat   = A_factor * np.exp(-Ea_catalyst   / (R_gas * T_arr))

print(f"\n  RATE CONSTANT k(T) COMPARISON (Ea_nocat={Ea_nocatalyst/1e3:.0f} kJ, "
      f"Ea_cat={Ea_catalyst/1e3:.0f} kJ):")
print(f"  {'T (K)':<8} {'k (no cat)':<18} {'k (cat)':<18} {'speedup'}")
print(f"  {'-'*55}")
for T_val in [300, 400, 500, 700, 1000]:
    kn = A_factor * np.exp(-Ea_nocatalyst / (R_gas * T_val))
    kc = A_factor * np.exp(-Ea_catalyst   / (R_gas * T_val))
    print(f"  {T_val:<8} {kn:<18.3e} {kc:<18.3e} {kc/kn:.1f}x")

print("""
  MICHAELIS-MENTEN KINETICS (enzyme catalysis):
    Enzyme E + Substrate S <-> ES complex -> E + Product P
    Rate: v = v_max * [S] / (Km + [S])
    v_max = k_cat * [E_total]   (maximum rate when all enzyme occupied)
    Km    = (k_off + k_cat) / k_on  (Michaelis constant, ~ substrate affinity)
    At [S] = Km: v = v_max / 2  (half-saturation)
    Turnover number: k_cat = v_max / [E_total]  [reactions/enzyme/second]
    Catalytic efficiency: k_cat/Km  [L/(mol*s)]  (diffusion limit ~ 10^8-10^9)
""")

# MM kinetics
S_arr  = np.logspace(-3, 2, 300)  # uM
Km_val = 1.0    # uM
v_max_val = 100.0  # uM/s
v_MM   = v_max_val * S_arr / (Km_val + S_arr)

print(f"  MM curve: Km={Km_val} uM, v_max={v_max_val} uM/s")
print(f"    At [S]=0.1Km: v={v_max_val*0.1/(1+0.1):.2f} uM/s  ({0.1/1.1*100:.1f}% max)")
print(f"    At [S]=Km:    v={v_max_val*1/(1+1):.2f} uM/s  (50% max)")
print(f"    At [S]=10*Km: v={v_max_val*10/(1+10):.2f} uM/s  ({10/11*100:.1f}% max)")

print("""
  SILICA (SiO2):
    Molecular structure: Si tetrahedrally bonded to 4 O atoms.
    Network solid: no discrete molecules, extended 3D lattice.
    Melting point: ~1713 C (amorphous) or ~1600 C (quartz).
    Bandgap: ~9 eV (excellent insulator, UV transparent).
    Refractive index: n ~ 1.444 at 1550nm (optical fiber core).

  SILICA IN OPTICAL FIBER:
    Core: SiO2 doped with GeO2 (n ~ 1.468) -- higher n than cladding.
    Cladding: pure SiO2 (n = 1.444).
    Delta n = 0.024 -> NA = sqrt(n_core^2 - n_clad^2) ~ 0.26
    Attenuation: ~0.2 dB/km at 1550nm (Rayleigh scattering minimum).
    Dispersion: 17 ps/(nm*km) at 1550nm (standard SMF-28).
    GS connection: D in H(nu)=exp(i*pi*D*nu^2) comes from silica GVD.

  SILICA AS CATALYST SUPPORT:
    High surface area: 100-800 m^2/g (mesoporous silica: MCM-41, SBA-15).
    Silanol groups (Si-OH) on surface: anchor metal nanoparticles (Pt, Pd, Au).
    Chemically inert: does not poison catalyst.
    Thermally stable: survives high temperature reactions.
    Applications:
    - Pt/SiO2: hydrogenation catalysis (vegetable oil -> margarine)
    - Pd/SiO2: Suzuki coupling (pharmaceutical synthesis)
    - Cu/ZnO/Al2O3: methanol synthesis from CO2+H2
    - TiO2/SiO2: photocatalysis (water splitting, pollutant degradation)

  SILICA IN MICROFLUIDICS:
    Thermally oxidized Si wafers: SiO2 surface.
    Silanol Si-OH: negatively charged above pH 4 -> generates EOF.
    Zeta potential: -50 to -80 mV (strong electroosmotic flow).
    Bonding: SiO2 surfaces bond by O2 plasma activation + heating.
    Etching: HF for isotropic, DRIE for anisotropic (deep channels).
    Bio-compatibility: cells adhere well, protein-compatible.
""")

# RogueGuard silica dispersion connection
n_silica = 1.444
lambda0  = 1550e-9   # m
GVD      = 17e-3     # ps/(nm*km) -> convert to s/m^2
# D = -lambda/c * d^2n/dlambda^2 ~ GVD
c_light  = 3e8
GVD_SI   = GVD * 1e-12 / (1e-9 * 1e3)   # s/(m*m)
# beta_2 = -lambda^2/(2*pi*c) * GVD_raw
beta2    = -lambda0**2 / (2*np.pi*c_light) * (GVD * 1e-12/(1e-9*1e3))
print(f"\n  SILICA FIBER DISPERSION at 1550nm:")
print(f"    GVD  = {GVD} ps/(nm*km)")
print(f"    beta2 = {beta2*1e24:.4f} ps^2/km")
print(f"    For 1km fiber: D_GS = beta2*L = {beta2*1e3*1e24:.2f} ps^2")
print(f"    This D enters H(nu) = exp(i*pi*D*nu^2) as the GS diversity parameter.")

# ============================================================
# MATPLOTLIB -- 6-PANEL FIGURE
# ============================================================
print(f"\n{SEP}")
print("BUILDING FIGURE...")
print(SEP)

fig = plt.figure(figsize=(18, 13))
fig.patch.set_facecolor("#F8F8F0")
gs0 = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38,
                        top=0.93, bottom=0.06, left=0.06, right=0.97)

ax_mem   = fig.add_subplot(gs0[0, 0])
ax_data  = fig.add_subplot(gs0[0, 1])
ax_bond  = fig.add_subplot(gs0[0, 2])
ax_arrh  = fig.add_subplot(gs0[1, 0])
ax_mm    = fig.add_subplot(gs0[1, 1])
ax_rates = fig.add_subplot(gs0[1, 2])

fig.suptitle("C Memory + Data Scales + Fed Rates + ETH Commitments + Catalysis + Silica",
             fontsize=12, fontweight="bold", color="#1a1a2e")

# ---- AX_MEM: heap fragmentation visual ----
ax = ax_mem
ax.set_facecolor("#F0F4FF")
ax.set_xlim(0, 100); ax.set_ylim(0, 5); ax.axis("off")
ax.set_title("External Heap Fragmentation", fontsize=10)

# Initial state: mixed alloc/free
blocks_init = [
    (0, 15, "used", "#1f77b4", "A:15"),
    (15, 10, "free", "#eee", ""),
    (25, 20, "used", "#ff7f0e", "B:20"),
    (45, 8,  "free", "#eee", ""),
    (53, 18, "used", "#2ca02c", "C:18"),
    (71, 10, "free", "#eee", ""),
    (81, 12, "used", "#d62728", "D:12"),
    (93, 7,  "free", "#eee", ""),
]
for y_base, label in [(3.2, "Before free:"), (1.0, "After free A,C,D:")]:
    ax.text(-1, y_base+0.6, label, fontsize=8, va="bottom")
    for start, width, kind, color, text in blocks_init:
        rect = plt.Rectangle((start, y_base), width, 0.5,
                              facecolor=color, edgecolor="k", linewidth=0.8)
        ax.add_patch(rect)
        if text:
            ax.text(start + width/2, y_base+0.25, text,
                    ha="center", va="center", fontsize=6.5, color="white" if kind=="used" else "#888")

# After freeing A, C, D -- more fragmentation
blocks_after = [
    (0, 15,  "free", "#eee", "FREE"),
    (15, 10, "free", "#eee", ""),
    (25, 20, "used", "#ff7f0e", "B:20"),
    (45, 8,  "free", "#eee", ""),
    (53, 18, "free", "#eee", "FREE"),
    (71, 10, "free", "#eee", ""),
    (81, 12, "free", "#eee", "FREE"),
    (93, 7,  "free", "#eee", ""),
]
for start, width, kind, color, text in blocks_after:
    rect = plt.Rectangle((start, 1.0), width, 0.5,
                          facecolor=color, edgecolor="k", linewidth=0.8)
    ax.add_patch(rect)
    if text == "FREE":
        ax.text(start + width/2, 1.25, text,
                ha="center", va="center", fontsize=6, color="#d62728")

total_free = sum(w for _,w,k,_,_ in blocks_after if k=="free")
ax.text(0, 0.4, f"Total free: {total_free}B  but largest block = 15B  -> malloc(16) FAILS!",
        fontsize=8, color="#d62728")

# ---- AX_DATA: log-scale data sizes ----
ax = ax_data
ax.set_facecolor("#FFFFF0")
names_d = ["bit","byte","KB","MB","GB","TB","PB","EB","ZB","YB"]
vals_d  = [1, 8, 8e3, 8e6, 8e9, 8e12, 8e15, 8e18, 8e21, 8e24]
colors_d = plt.cm.viridis(np.linspace(0.1, 0.9, len(names_d)))
ax.bar(names_d, vals_d, color=colors_d, edgecolor="k", linewidth=0.5)
ax.set_yscale("log")
ax.set_ylabel("Bits", fontsize=9)
ax.set_title("Data Scale Hierarchy (log)", fontsize=10)
ax.grid(axis="y", alpha=0.3, which="both")
# Annotations
refs = [(8e15, "1 PB = LHC/year"), (8e21, "Global internet/day")]
for val, label in refs:
    ax.axhline(val, color="#d62728", lw=0.8, ls="--", alpha=0.5)
    ax.text(9.5, val*1.5, label, fontsize=7, color="#d62728", ha="right")
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, fontsize=8)

# ---- AX_BOND: bond price vs yield ----
ax = ax_bond
ax.set_facecolor("#F0FFF0")
ax.plot(yields*100, prices, "#1f77b4", lw=2.0)
ax.axhline(1000, color="#999", lw=0.8, ls="--")
ax.axvline(5, color="#d62728", lw=0.8, ls="--")
ax.scatter([5], [1000], color="#d62728", s=80, zorder=5, label="Par (yield=coupon=5%)")
ax.set_xlabel("Yield (%)", fontsize=9)
ax.set_ylabel("Bond Price ($)", fontsize=9)
ax.set_title("10yr Bond: Price vs Yield\n(5% coupon, $1000 face)", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.2)
ax.text(0.55, 0.85,
        "Rates UP -> Price DOWN\n(inverse relationship)",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(fc="#fff0f0", ec="#d62728", pad=3))

# ---- AX_ARRH: Arrhenius with/without catalyst ----
ax = ax_arrh
ax.set_facecolor("#FFF5F0")
ax.semilogy(T_arr, k_nocat, "#d62728", lw=2.0, label=f"No catalyst (Ea={Ea_nocatalyst/1e3:.0f} kJ)")
ax.semilogy(T_arr, k_cat,   "#2ca02c", lw=2.0, label=f"Catalyst (Ea={Ea_catalyst/1e3:.0f} kJ)")
ax.set_xlabel("Temperature (K)", fontsize=9)
ax.set_ylabel("Rate constant k (s$^{-1}$)", fontsize=9)
ax.set_title("Arrhenius: Catalyst Lowers Ea", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
ax.axvline(298, color="#888", lw=0.8, ls=":", label="298K")
speedup_RT = (A_factor*np.exp(-Ea_catalyst/(R_gas*298)) /
              A_factor*np.exp(-Ea_nocatalyst/(R_gas*298)))
ax.text(0.02, 0.97,
        f"At 298K: catalyst is\n{A_factor*np.exp(-Ea_catalyst/(R_gas*298)) / (A_factor*np.exp(-Ea_nocatalyst/(R_gas*298))):.0e}x faster",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(fc="white", ec="#bbb", pad=2))

# ---- AX_MM: Michaelis-Menten ----
ax = ax_mm
ax.set_facecolor("#F0F8FF")
ax.semilogx(S_arr, v_MM, "#1f77b4", lw=2.0)
ax.axhline(v_max_val, color="#999", lw=0.8, ls="--", label=f"v_max={v_max_val}")
ax.axhline(v_max_val/2, color="#ff7f0e", lw=0.8, ls="--", label=f"v_max/2")
ax.axvline(Km_val, color="#d62728", lw=0.8, ls="--", label=f"Km={Km_val} uM")
ax.set_xlabel("[S] (uM)", fontsize=9)
ax.set_ylabel("v (uM/s)", fontsize=9)
ax.set_title("Michaelis-Menten Enzyme Kinetics", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.2)
ax.text(0.55, 0.12,
        r"$v = \frac{v_{max}[S]}{K_m + [S]}$",
        transform=ax.transAxes, fontsize=11,
        bbox=dict(fc="white", ec="#bbb", pad=3))

# ---- AX_RATES: Taylor rule visualization ----
ax = ax_rates
ax.set_facecolor("#FFF0FF")
pi_range = np.linspace(-1, 10, 100)
for gap, col, lbl in [(-3, "#d62728","gap=-3 (recession)"),
                       (0,  "#1f77b4","gap=0 (neutral)"),
                       (2,  "#2ca02c","gap=+2 (overheating)")]:
    r_taylor_line = r_star + pi_range + 0.5*(pi_range - pi_target) + 0.5*gap
    ax.plot(pi_range, r_taylor_line, color=col, lw=1.8, label=lbl)

ax.axhline(0, color="k", lw=0.5)
ax.axvline(pi_target, color="#999", lw=0.8, ls="--")
ax.text(pi_target+0.1, -1, "pi*=2%", fontsize=8, color="#888")
ax.set_xlabel("Inflation pi (%)", fontsize=9)
ax.set_ylabel("Taylor Rule FFR (%)", fontsize=9)
ax.set_title("Taylor Rule: r = r*+pi+0.5(pi-pi*)+0.5*gap", fontsize=10)
ax.legend(fontsize=7.5)
ax.grid(alpha=0.2)
ax.set_ylim(-5, 20)

plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved: {OUT}")

print(f"\n{SEP}")
print("Done.")
print(SEP)
