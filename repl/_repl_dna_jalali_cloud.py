"""
_repl_dna_jalali_cloud.py

S1: 2 bits/base -- DNA information theory, genome size, Shannon entropy
S2: Jalali STEAM -- photonic time-stretch microscopy, dispersion-assisted
    imaging of 100K cells/sec -- DIRECT connection to our GS work
S3: Chemical inputs -- Hill equation, dose-response, receptor binding
S4: Cloud backup / flood resilience -- S3 durability, RAID, erasure coding
S5: Towers of Hanoi -- recursion, O(2^n), explicit solution, DP analogy
"""

import numpy as np
import sympy as sp
import math

SEP = "=" * 65

# ------------------------------------------------------------------ #
# S1: 2 BITS/BASE -- DNA INFORMATION THEORY
# ------------------------------------------------------------------ #
print(SEP)
print("SECTION 1: 2 BITS/BASE -- DNA INFORMATION THEORY")
print(SEP)

print("""
  DNA ALPHABET:  A  T  G  C   (4 bases)
  Information per base:  log2(4) = 2 bits/base

  HUMAN GENOME:
    3.2 billion base pairs (diploid: 6.4 Gbp)
    Information: 3.2e9 * 2 = 6.4 Gbits = 800 MB (haploid)
    Diploid: 1.6 GB -- fits on a USB stick
    But: only ~1.5% is protein-coding exons
    Exome: ~30 Mbp = 60 Mb = 7.5 MB

  SHANNON ENTROPY OF GENOME:
    H = -sum_b p(b) * log2(p(b))   [bits/base]
    If perfectly random (p(A)=p(T)=p(G)=p(C)=0.25): H = 2 bits/base
    Real genome: GC content varies by species, 40-60% in humans
    Local CpG islands: high GC, near promoters
    Repetitive elements (SINEs, LINEs): ~45% of genome -> lower entropy

  SEQUENCING AS CHANNEL CODING:
    Sequencer = noisy channel: base -> fluorescent signal -> base call
    Error model: Phred Q = -10 * log10(p_error)
    Q30: p_error = 0.001 (99.9% accurate)  <- Illumina short reads
    Q20: p_error = 0.01                    <- lower quality
    Shannon capacity: C = B * log2(1 + SNR)
    Same formula as ADC SNR = 6.02N + 1.76 dB (from Nyquist session)

  COMPRESSION:
    Human genome compressed (gzip): ~850 MB -> 200 MB
    Reference-based: store only differences (SNPs) -> ~4 MB
    1000 Genomes Project: 2504 people * 4MB = 10 GB for all variants
    CRAM format (EBI): 60% smaller than BAM
""")

# entropy calculation
bases = ['A', 'T', 'G', 'C']
print("  Shannon entropy vs GC content:")
print(f"  {'GC%':>6} {'p(G)=p(C)':>12} {'p(A)=p(T)':>12} {'H (bits/base)':>15}")
print("  " + "-" * 46)
for gc_pct in [20, 35, 40, 50, 60, 65, 80]:
    p_gc = gc_pct / 200.0   # each of G, C
    p_at = (100 - gc_pct) / 200.0  # each of A, T
    probs = [p_at, p_at, p_gc, p_gc]
    H = -sum(p * np.log2(p) for p in probs if p > 0)
    print(f"  {gc_pct:>6}% {p_gc:>12.4f} {p_at:>12.4f} {H:>15.6f}")

# genome sizes
print()
print("  Genome sizes and bit content:")
genomes = [
    ("E. coli",         4.6e6,  51),
    ("S. cerevisiae",   12e6,   38),
    ("C. elegans",      100e6,  36),
    ("D. melanogaster", 180e6,  42),
    ("H. sapiens",      3.2e9,  41),
    ("T. aestivum",     17e9,   43),   # wheat, largest known diploid
]
print(f"  {'Organism':<22} {'Genome (bp)':>14} {'GC%':>5} {'Info (MB)':>10}")
print("  " + "-" * 52)
for name, bp, gc in genomes:
    MB = bp * 2 / 8e6   # 2 bits/base -> bytes -> MB
    print(f"  {name:<22} {bp:>14.2e} {gc:>5}% {MB:>10.1f}")

# ------------------------------------------------------------------ #
# S2: JALALI STEAM -- PHOTONIC TIME STRETCH + RogueGuard CONNECTION
# ------------------------------------------------------------------ #
print()
print(SEP)
print("SECTION 2: JALALI STEAM -- DISPERSION-ASSISTED CELL IMAGING")
print(SEP)

print("""
  BAHRAM JALALI (UCLA):
    Inventor of photonic time stretch ADC (1999) and
    STEAM (Serial Time-Encoded Amplified Microscopy, 2009, Nature).
    His lab images cells using DISPERSION -- the same physics as
    our GS phase retrieval.

  STEAM PRINCIPLE:
    1. Broadband pulsed laser (modelocked, ~10 nm bandwidth at 1550 nm)
    2. SPACE -> TIME mapping:
       Diffract pulse off a spatial sample (e.g. cell flowing in microfluidic)
       -> each spatial frequency maps to a wavelength
       -> disperse with DCF (dispersion compensating fiber, D ~ -1000 ps/nm)
       -> wavelength differences become TIME differences
       -> spatial image encoded as a 1D time waveform
    3. Amplify with Raman amplifier (compensates fiber loss)
    4. Digitize with single 1 GHz photodetector + ADC
    5. Reconstruct 2D image from time series

  DISPERSION = LENS IN TIME:
    phi(omega) = (1/2) * beta2 * L * omega^2
    Same as our GS dispersion: H(nu) = exp(i*pi*D*nu^2)
    D = beta2 * L * c / lambda^2  [ps^2 -> our unitless D]
    The DISPERSED fiber acts as a Fourier transformer in time.

  WHY DISPERSION DOES SPACE-TO-TIME:
    Input: E(t) with spatial frequency content f_x
    After grating: lambda component -> angle theta_lambda
    -> lands at position x on cell: x ~ lambda (linear dispersion)
    -> cell transmits/reflects amplitude A(x) at that position
    After temporal disperser (DCF):
      wavelength lambda offset -> time delay tau = D * delta_lambda
      -> A(x) becomes A(t) -- spatial image encoded in time!
    This is exactly a Fourier transform: the disperser applies quadratic
    phase exp(i*pi*D*nu^2) which is our GS H(nu) kernel.

  THROUGHPUT:
    Pulse repetition rate: 36 MHz (modelocked fiber laser)
    Each pulse = one line scan of the cell
    Cell flows at ~1 m/s through microfluidic channel
    Cell width ~10 um -> 10 line scans per cell = 36MHz/10 = 3.6M cells/sec
    Actual reported: >100,000 cells/sec (limited by cell spacing, not optics)
    Compare: conventional flow cytometer: 10,000 cells/sec (scatter only)
    STEAM advantage: full 2D GREY-SCALE image at 10x flow cytometer rate

  PHASE-SENSITIVE STEAM (connects directly to our project):
    Interferometric STEAM: split beam, delay one arm, interfere.
    Measures PHASE of scattered field, not just intensity.
    Phase contrast image: reveals dry mass, refractive index of cell.
    This IS our GS problem: recover phi from I1, I2, D1, D2.
    Jalali group's 2017 paper: used GS-type algorithm for phase retrieval
    from two intensity measurements at different dispersions.

  PHOTONIC TIME STRETCH ADC:
    Stretch-and-digitize:
    1. Modulate signal onto optical carrier
    2. Disperse in fiber: time axis stretched by M = 1 + D2/D1
    3. Detect with slower ADC: apparent sample rate = f_ADC * M
    M = 100x stretch -> 1 GSPS ADC captures 100 GSPS signals
    Used for: single-shot oscilloscopes, radar, rogue wave detection

  CONNECTION TO ROGUEGUARD:
    Our GS phase retrieval: H(nu) = exp(i*pi*D*nu^2) <- SAME as Jalali
    RogueGuard input: single-shot optical waveform (rogue event)
    Time stretch: dispersed fiber stretches event in time
    -> slower ADC (400 MSPS) can capture 40 GHz optical bandwidth
    SBIR angle: cite Jalali's time-stretch for rogue wave detection (2013)
    Ref: Mahjoubfar et al., Nature Photonics 2017 "Time stretch and its
         applications" -- directly cites fiber rogue wave monitoring.
""")

# time stretch calculation
print("  Photonic time stretch parameters:")
D1_ts = -1000.0   # ps/nm  (pre-disperser DCF)
D2_ts = 10000.0   # ps/nm  (main stretch fiber)
M_stretch = 1 + abs(D2_ts / D1_ts)
f_ADC = 1.0        # GHz
f_eff = f_ADC * M_stretch

print(f"  D1 (pre-chirp) = {D1_ts} ps/nm")
print(f"  D2 (stretch)   = {D2_ts} ps/nm")
print(f"  Stretch factor M = 1 + |D2/D1| = {M_stretch:.0f}x")
print(f"  ADC rate {f_ADC} GHz -> effective capture rate {f_eff:.0f} GHz")
print()
print("  STEAM cell throughput:")
f_rep = 36e6      # Hz pulse rep rate
lines_per_cell = 10
cells_per_sec = f_rep / lines_per_cell
print(f"  Pulse rep rate: {f_rep/1e6:.0f} MHz")
print(f"  Lines per cell: {lines_per_cell}")
print(f"  Cells/sec: {cells_per_sec/1e3:.0f}K cells/sec")
print(f"  100K cells: {100_000/cells_per_sec*1e3:.2f} ms acquisition time")

# ------------------------------------------------------------------ #
# S3: CHEMICAL INPUTS -- HILL EQUATION, DOSE-RESPONSE
# ------------------------------------------------------------------ #
print()
print(SEP)
print("SECTION 3: CHEMICAL INPUTS -- HILL EQUATION, DOSE-RESPONSE")
print(SEP)

print("""
  LIGAND-RECEPTOR BINDING:
    Receptor R + Ligand L <-> RL   (reversible binding)
    K_d = [R][L] / [RL]            (dissociation constant, lower = tighter)
    Occupancy: theta = [L] / ([L] + K_d)

  HILL EQUATION (cooperative binding):
    theta = [L]^n / ([L]^n + K_d^n)
    n = Hill coefficient
    n=1: Michaelis-Menten (hyperbolic, non-cooperative)
    n>1: positive cooperativity (sigmoidal) -- e.g. hemoglobin O2 binding n=2.8
    n<1: negative cooperativity

  DOSE-RESPONSE (pharmacology / cell biology):
    E = E_max * [D]^n / (EC50^n + [D]^n)
    E_max  = maximum effect
    EC50   = concentration giving 50% of E_max
    n      = Hill slope (steepness)

  SIGMOID IN LOG SCALE:
    Plot E vs log10([D]) -> S-shaped curve
    Linear range: EC10 to EC90 (10% to 90% effect)
    EC50 is the inflection point of the log-dose curve.

  AMAZON / FLOOD = DOSE ABOVE EC100:
    Overdose: [D] >> EC50 -> E -> E_max (saturated, no more effect)
    Below threshold [D] << EC50: E ~ E_max * [D]^n / EC50^n (linear)
    Backup systems: redundancy when primary saturates

  MICHAELIS-MENTEN KINETICS (enzyme):
    v = V_max * [S] / (K_m + [S])
    V_max = maximum reaction rate   [mM/s]
    K_m   = Michaelis constant (substrate at half-max rate) [mM]
    Lineweaver-Burk: 1/v = (K_m/V_max)*(1/[S]) + 1/V_max  <- linear plot
    Same hyperbolic as Hill n=1.
""")

# Hill equation numerical
print("  Hill equation: E_max=1, EC50=1uM, varying Hill slope n:")
conc_uM = np.array([0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0])
print(f"  {'[D] (uM)':>10}", end="")
for n_hill in [1, 2, 4]:
    print(f"  {'n='+str(n_hill):>10}", end="")
print()
print("  " + "-" * 42)
for c in conc_uM:
    print(f"  {c:>10.3f}", end="")
    for n_hill in [1, 2, 4]:
        E = c**n_hill / (1.0**n_hill + c**n_hill)
        print(f"  {E:>10.4f}", end="")
    print()

# ------------------------------------------------------------------ #
# S4: CLOUD BACKUP / FLOOD RESILIENCE
# ------------------------------------------------------------------ #
print()
print(SEP)
print("SECTION 4: CLOUD BACKUP -- AWS S3, ERASURE CODING, RAID")
print(SEP)

print("""
  AWS S3 DURABILITY: 99.999999999% (11 nines) per object per year.
    P(loss) = 1 - 0.99999999999 = 1e-11 per year
    For 1 million objects: expected loss = 0.00001 objects/year
    Achieved by: replication across 3+ availability zones (AZs)
    + erasure coding within each AZ

  ERASURE CODING (the math of backup):
    Reed-Solomon (n, k) code:
      k = data chunks, n = total chunks (k + parity)
      Can recover from ANY n-k chunk failures.
      S3 uses (14, 10): 14 chunks, 10 data, 4 parity
        -> survive any 4 simultaneous disk failures
        -> overhead: 14/10 = 1.4x storage

    Compare to RAID:
      RAID-1 (mirror): 2x overhead, survive 1 failure
      RAID-5: (n-1)/n efficiency, survive 1 failure
      RAID-6: (n-2)/n efficiency, survive 2 failures
      RAID-10: 2x overhead, survive 1 per mirror pair

  HAMMING CODE (error correction basics):
    (7,4) Hamming: 4 data bits + 3 parity bits = 7 bits
    Detects and corrects 1-bit error.
    General (2^r - 1, 2^r - 1 - r): r parity bits protect 2^r-1-r data bits.

  FLOOD ANALOGY (disaster recovery):
    RTO (Recovery Time Objective): max downtime acceptable
    RPO (Recovery Point Objective): max data loss acceptable (time)
    S3 cross-region replication: async copy to second region
    -> RPO = minutes (replication lag), RTO = seconds (DNS failover)
    Multi-region active-active: RPO = 0, RTO = 0 (no downtime, no loss)

  FOR ROGUEGUARD DATA:
    Measurement data: raw I1, I2 from fiber sensors -> S3 standard
    Config / firmware:  git repo (already on differentiable-gs branch)
    Processed results:  _overnight_results.jsonl (gitignored, needs S3)
    3-2-1 rule: 3 copies, 2 different media, 1 offsite
      -> local SSD + external drive + S3 cross-region
""")

# erasure code overhead
print("  Erasure coding overhead vs protection:")
print(f"  {'Scheme':<20} {'k (data)':>10} {'n (total)':>10} {'Overhead':>10} {'Failures OK':>12}")
print("  " + "-" * 64)
schemes = [
    ("RAID-1",    1,  2,  1),
    ("RAID-5 (8+1)",7, 8, 1),
    ("RAID-6 (8+2)",6, 8, 2),
    ("RS(14,10)",10, 14, 4),
    ("RS(16, 10)",10, 16, 6),
    ("RS(n=3k)",  10, 30, 20),
]
for name, k, n, f in schemes:
    overhead = n / k
    print(f"  {name:<20} {k:>10} {n:>10} {overhead:>10.2f}x {f:>12}")

# probability of data loss (simplified)
print()
print("  Simplified probability of data loss:")
print("  (independent disk failure p=0.001/year, RS(14,10)=4 parity)")
p_disk = 0.001  # per year
for n_disks, k_data in [(14, 10), (16, 10), (3, 1), (2, 1)]:
    f_parity = n_disks - k_data
    # P(more than f_parity failures) -- binomial
    P_loss = sum(math.comb(n_disks, i) * p_disk**i * (1-p_disk)**(n_disks-i)
                 for i in range(f_parity+1, n_disks+1))
    print(f"  RS({n_disks},{k_data}): P(data loss) = {P_loss:.3e} per year")

# ------------------------------------------------------------------ #
# S5: TOWERS OF HANOI
# ------------------------------------------------------------------ #
print()
print(SEP)
print("SECTION 5: TOWERS OF HANOI -- RECURSION, O(2^n), DP CONNECTION")
print(SEP)

print("""
  PROBLEM: Move n disks from peg A to peg C using peg B.
  Rules:  (1) move one disk at a time
          (2) never place a larger disk on a smaller one

  RECURSIVE SOLUTION:
    hanoi(n, src, dst, aux):
      if n == 1: move disk from src to dst
      else:
        hanoi(n-1, src, aux, dst)   # move top n-1 to aux
        move disk n from src to dst  # move bottom disk
        hanoi(n-1, aux, dst, src)   # move n-1 from aux to dst

  RECURRENCE:
    T(n) = 2*T(n-1) + 1
    T(1) = 1
    Solution: T(n) = 2^n - 1

  EXPLICIT FORMULA:
    T(n) = 2^n - 1
    This is the MINIMUM number of moves (optimal).

  ITERATIVE SOLUTION (no recursion):
    For n disks, 2^n - 1 moves.
    Move k (1-indexed): move disk number = lowest set bit of k
      bit_k = k & (-k)   (isolate lowest set bit)
      disk  = log2(bit_k) + 1
    Peg of disk d depends on parity: pegs cycle in fixed order.

  CONNECTION TO 2^n (binary):
    Each move corresponds to one bit flip in Gray code counter.
    Gray code: 0,1,3,2,6,7,5,4,... (adjacent numbers differ by 1 bit)
    Tower state = Gray code value.
    Disk d moves when bit d-1 changes in Gray code.

  DP ANALOGY:
    Towers of Hanoi is NOT efficiently solved by DP (no overlapping subproblems).
    But the recurrence T(n) = 2*T(n-1) + 1 IS a DP recurrence.
    Memoized: T(n) = 2*T(n-1) + 1, T[1]=1 -> same as closed form.
    Compare Fibonacci: F(n) = F(n-1) + F(n-2) -- overlapping subproblems,
    DP gives O(n) vs naive O(2^n).

  LEGEND: Hindu priests at Benares move 64 golden disks.
    T(64) = 2^64 - 1 = 18,446,744,073,709,551,615 moves
    At 1 move/second: 585 billion years (42x age of universe)
""")

# hanoi moves and closed form
print("  Hanoi: moves required and time at 1 move/sec:")
print(f"  {'n':>4} {'T(n) = 2^n-1':>20} {'Time':>20}")
print("  " + "-" * 46)
units = [(1, "sec"), (60, "min"), (3600, "hr"), (86400, "day"),
         (3.156e7, "yr"), (3.156e9, "kyr"), (3.156e12, "Myr"),
         (3.156e16, "Gyr")]
for n in [1, 2, 3, 4, 8, 16, 32, 40, 64]:
    T = 2**n - 1
    secs = float(T)
    for threshold, label in units:
        if secs < threshold * 1000 or label == "Gyr":
            time_str = f"{secs/threshold:.2f} {label}"
            break
    print(f"  {n:>4} {T:>20,} {time_str:>20}")

# recursive generator for small n
print()
print("  Optimal moves for n=3 disks (A->C via B):")
moves = []
def hanoi(n, src, dst, aux):
    if n == 1:
        moves.append(f"  Move disk 1: {src} -> {dst}")
    else:
        hanoi(n-1, src, aux, dst)
        moves.append(f"  Move disk {n}: {src} -> {dst}")
        hanoi(n-1, aux, dst, src)

hanoi(3, 'A', 'C', 'B')
for i, m in enumerate(moves, 1):
    print(f"  Step {i:2d}: {m.strip()}")
print(f"  Total moves: {len(moves)} = 2^3 - 1 = 7  (confirmed)")

print()
print(SEP)
print("SECTION 6: THE FULL CONNECTION")
print(SEP)
print("""
  2 bits/base  (DNA)       -> Shannon: H = -sum p*log2(p)
  STEAM imaging (Jalali)   -> dispersion H(nu)=exp(i*pi*D*nu^2) encodes space->time
  100K cells/sec (STEAM)   -> same math as our GS: two intensities + dispersion -> phase
  Hill equation (chem)     -> sigmoid = softmax in neural networks (same curve)
  Erasure code (S3)        -> Reed-Solomon = same polynomial math as BCH codes / CRC
  Towers of Hanoi          -> T(n) = 2*T(n-1)+1 -> 2^n structure = binary recursion tree

  THE UNIFYING THREAD:
    DNA: 2 bits/base * 3.2e9 bases = 6.4 Gbits
    STEAM: 2 bits/pixel * (36MHz * 10px) / cell = 720 Mbits/sec imaging
    GS phase: 2 intensity planes -> 1 phase plane (information doubling at output)
    RS(14,10): 14 chunks encode 10 data -> 40% redundancy = 2/5 overhead
    Hanoi T(64): 2^64 steps, same as 64-bit address space
    ALL are manifestations of 2^n -- the fundamental combinatorial structure.
""")

print(SEP)
print("Done.")
print(SEP)
