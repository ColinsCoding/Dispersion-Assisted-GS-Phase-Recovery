"""
classifier.py — read recovered phase from C via stdin, output label to stdout

C sends: N=256 raw doubles (binary) via pipe
We send: "rogue\n" or "normal\n"

No external ML weights. Rule-based + statistical — auditable for OUSD/DoD.
"""

import sys
import struct
import math

N = 256

def read_phase():
    raw = sys.stdin.buffer.read(N * 8)   # 8 bytes per double
    if len(raw) < N * 8:
        return None
    return list(struct.unpack(f'{N}d', raw))

def classify(phase):
    # 1. Phase variance — rogue pulses have low phase variance (mode-locked spike)
    mean = sum(phase) / N
    var  = sum((p - mean)**2 for p in phase) / N

    # 2. Phase gradient — rapid phase jumps indicate coherent burst
    grad = [abs(phase[k+1] - phase[k]) for k in range(N-1)]
    max_grad = max(grad)
    mean_grad = sum(grad) / len(grad)

    # 3. Spectral flatness proxy — uniform phase = coherent rogue
    # Wrap phases to [-π, π] and check clustering
    wrapped = [((p + math.pi) % (2*math.pi)) - math.pi for p in phase]
    phase_range = max(wrapped) - min(wrapped)

    # Decision (threshold-based, no foreign weights)
    is_rogue = (var < 0.1) and (max_grad > 1.5) and (phase_range < 2.0)

    return "rogue" if is_rogue else "normal", {
        "variance": round(var, 4),
        "max_grad": round(max_grad, 4),
        "phase_range": round(phase_range, 4)
    }

def main():
    phase = read_phase()
    if phase is None:
        print("normal", flush=True)   # safe default
        return

    label, stats = classify(phase)
    print(label, flush=True)          # C reads this line
    print(f"# stats: {stats}", file=sys.stderr)   # debug only, not sent to C

if __name__ == "__main__":
    main()
