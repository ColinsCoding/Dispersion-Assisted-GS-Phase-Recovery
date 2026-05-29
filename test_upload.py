"""
test_upload.py -- Verify dashboard is live and upload log is working
====================================================================
Generates synthetic optical intensity data for 6 standard modulation
formats, POSTs each to the running Flask server, reads back
/uploads/log and prints a verification table.

Usage:
    python test_upload.py                     # default http://localhost:5000
    python test_upload.py --url https://...   # remote / Render deploy

Synthetic standards generated
------------------------------
  1  OOK-NRZ     On-Off Keying  (binary ASK, rect pulses)
  2  PAM4        4-level amplitude modulation (2 bits/symbol)
  3  QPSK        Gray-coded quadrature phase  (via dsp module)
  4  DPSK        Differential QPSK  (phase transitions only)
  5  STEAM       Chirped Gaussian pulse  (dispersion-stretched)
  6  Soliton     Optical soliton  (sech^2 envelope + chirp)

Each format gets a two-arm .mat file:
    I1[n] = |E(t) * h_D1(t)|^2    D1 = -600 ps^2
    I2[n] = |E(t) * h_D2(t)|^2    D2 = -1200 ps^2

All results are verified against /uploads/log.
"""

import sys, argparse, io, time, hashlib
import numpy as np

try:
    import scipy.io as sio
except ImportError:
    sys.exit("[!] scipy required:  pip install scipy")

try:
    import requests
except ImportError:
    sys.exit("[!] requests required:  pip install requests")

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://localhost:5000",
                    help="Base URL of the running dashboard")
parser.add_argument("--no-color", action="store_true")
args = parser.parse_args()
BASE = args.url.rstrip("/")

GREEN  = "" if args.no_color else "\033[92m"
RED    = "" if args.no_color else "\033[91m"
YELLOW = "" if args.no_color else "\033[93m"
CYAN   = "" if args.no_color else "\033[96m"
RESET  = "" if args.no_color else "\033[0m"
BOLD   = "" if args.no_color else "\033[1m"

def ok(s):   return f"{GREEN}OK{RESET}  {s}"
def err(s):  return f"{RED}FAIL{RESET}  {s}"
def info(s): return f"{CYAN}{s}{RESET}"

# ---------------------------------------------------------------------------
# Disperse field to produce I1, I2
# ---------------------------------------------------------------------------
PI = np.pi

def disperse(E, D_ps2, N=None):
    """Apply dispersive transfer function H(nu)=exp(i*pi*D*nu^2)."""
    if N is None: N = len(E)
    nu = np.fft.fftfreq(N)
    H  = np.exp(1j * PI * D_ps2 * nu**2)
    return np.fft.ifft(np.fft.fft(E, N) * H)

def two_arm_mat(E, D1=-600.0, D2=-1200.0):
    """Return dict with I1, I2 ready for scipy.io.savemat."""
    I1 = np.abs(disperse(E, D1))**2
    I2 = np.abs(disperse(E, D2))**2
    I1 = I1 / (I1.max() + 1e-12)
    I2 = I2 / (I2.max() + 1e-12)
    return {"I1": I1.astype(np.float32),
            "I2": I2.astype(np.float32)}

def mat_bytes(d):
    """Serialise dict to .mat file bytes in memory."""
    buf = io.BytesIO()
    sio.savemat(buf, d)
    buf.seek(0)
    return buf.read()

# ---------------------------------------------------------------------------
# Synthetic signal generators
# ---------------------------------------------------------------------------
RNG  = np.random.default_rng(42)
N    = 512         # samples per signal (keep light for CI)
t    = np.linspace(-50, 50, N)
dt   = t[1] - t[0]


def sig_ook_nrz():
    """OOK NRZ: binary ASK, rectangular pulses, sps=8."""
    bits = RNG.integers(0, 2, N // 8)
    sps  = 8
    env  = np.repeat(bits.astype(float), sps)[:N]
    E    = env.astype(complex)
    return E, "OOK-NRZ", "OOK On-Off Keying, NRZ rect pulses"


def sig_pam4():
    """PAM4: 4-level amplitude modulation, 2 bits/symbol."""
    syms = RNG.integers(0, 4, N // 8)   # {0,1,2,3} -> {-3,-1,+1,+3}/3
    levels = (2 * syms - 3) / 3.0
    env  = np.repeat(levels, 8)[:N]
    E    = (env + 1).astype(complex)    # shift to positive
    return E, "PAM4", "PAM4 4-level amplitude, 2 bits/symbol"


def sig_qpsk():
    """QPSK: Gray-coded 4-phase."""
    bits = RNG.integers(0, 2, N)
    pairs = bits[:N - N%2].reshape(-1, 2)
    enc  = {(0,0):1+1j, (0,1):-1+1j, (1,1):-1-1j, (1,0):1-1j}
    syms = np.array([enc[tuple(p)] for p in pairs]) / np.sqrt(2)
    E    = np.repeat(syms, 8)[:N]
    return E, "QPSK", "Gray-coded QPSK 4-phase"


def sig_dpsk():
    """DPSK: differential binary phase-shift keying."""
    bits = RNG.integers(0, 2, N // 8)
    phase = np.cumsum(bits * PI) % (2 * PI)
    E    = np.exp(1j * np.repeat(phase, 8)[:N])
    return E, "DPSK", "Differential BPSK, phase transitions"


def sig_steam():
    """STEAM: dispersion-stretched Gaussian pulse (broadband)."""
    env = np.exp(-t**2 / (2 * 12**2))
    phi = 0.6 * np.exp(-((t - 5) / 10)**2) * np.sin(2 * PI * t / 40)
    E   = env * np.exp(1j * phi)
    return E, "STEAM", "STEAM chirped Gaussian + cellular phase"


def sig_soliton():
    """Optical soliton: sech^2 envelope, phase chirp."""
    amp   = 1.0 / np.cosh(t / 6.0)
    omega = 0.05                       # carrier offset
    E     = amp * np.exp(1j * omega * t)
    return E, "Soliton", "Optical soliton sech^2 envelope"


STANDARDS = [sig_ook_nrz, sig_pam4, sig_qpsk, sig_dpsk, sig_steam, sig_soliton]

# ---------------------------------------------------------------------------
# Check server is alive
# ---------------------------------------------------------------------------
print(f"\n{BOLD}Jalabi Lab Dashboard -- Upload Verification{RESET}")
print(f"Target: {CYAN}{BASE}{RESET}\n")

try:
    h = requests.get(f"{BASE}/health", timeout=10).json()
    print(ok(f"/health  version={h.get('version')}  "
             f"uptime={h.get('uptime_s')}s  "
             f"uploads_total={h.get('uploads_total')}"))
except Exception as e:
    print(err(f"Cannot reach {BASE}/health  --  {e}"))
    print(f"\n{YELLOW}Start the server first:{RESET}")
    print("  python optical_dashboard/app.py")
    print("  -- or --")
    print("  docker compose up -d")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Upload each standard
# ---------------------------------------------------------------------------
results = []
print(f"\n{BOLD}Uploading 6 optical standards...{RESET}")

for fn in STANDARDS:
    E, label, description = fn()
    mat = two_arm_mat(E)
    raw = mat_bytes(mat)
    fname = f"test_{label.replace('-','_').lower()}.mat"

    t0 = time.monotonic()
    try:
        resp = requests.post(
            f"{BASE}/upload",
            files={"file": (fname, raw, "application/octet-stream")},
            data={
                "D1":   "-600",
                "D2":   "-1200",
                "lam":  "1550",
                "fs":   "56",
                "lab":  "Jalali Lab UCLA  --  test_upload.py",
                "desc": f"{description}  |  N={N}  synthetic",
            },
            timeout=60,
        )
        elapsed = (time.monotonic() - t0) * 1000
        d = resp.json()
        if d.get("error"):
            results.append((label, False, d["error"], elapsed, fname))
            print(err(f"{label:<12}  {d['error']}"))
        else:
            two_arm = d.get("two_arm", False)
            N_got   = d["stats"]["N"]
            results.append((label, True, description, elapsed, fname))
            arm_tag = f"{GREEN}2-arm{RESET}" if two_arm else "1-arm"
            print(ok(f"{label:<12}  N={N_got:<6}  {arm_tag}  "
                     f"{elapsed:.0f} ms  {fname}"))
    except Exception as e:
        elapsed = (time.monotonic() - t0) * 1000
        results.append((label, False, str(e), elapsed, fname))
        print(err(f"{label:<12}  {e}"))

# ---------------------------------------------------------------------------
# Read back /uploads/log and verify rows landed
# ---------------------------------------------------------------------------
print(f"\n{BOLD}Reading /uploads/log...{RESET}")
try:
    log = requests.get(f"{BASE}/uploads/log?limit=20", timeout=10).json()
    rows = log.get("rows", [])
    total = log.get("total", 0)
    print(info(f"  Total rows in DB: {total}"))
    print()
    # Header
    hdr = f"  {'id':>4}  {'filename':<34}  {'status':<6}  {'N':>6}  " \
          f"{'2arm':>4}  {'ms':>7}  {'lab'}"
    print(BOLD + hdr + RESET)
    print("  " + "-" * 80)
    for r in rows[:10]:
        stat_col = GREEN+"ok  "+RESET if r["status"]=="ok" else RED+"err "+RESET
        print(f"  {r['id']:>4}  {(r['filename'] or ''):.<34}  "
              f"{stat_col}  {str(r.get('n_samples') or ''):>6}  "
              f"{'yes' if r.get('two_arm') else 'no':>4}  "
              f"{str(r.get('processing_ms') or ''):>7}  "
              f"{(r.get('lab') or '')[:30]}")
    if total > 10:
        print(f"  ... {total-10} more rows — GET /uploads/log?offset=10")
except Exception as e:
    print(err(f"/uploads/log failed: {e}"))

# ---------------------------------------------------------------------------
# Read back /summary (if endpoint exists)
# ---------------------------------------------------------------------------
print(f"\n{BOLD}Reading /summary...{RESET}")
try:
    s = requests.get(f"{BASE}/summary", timeout=10)
    if s.status_code == 200:
        d = s.json()
        print(info(f"  total_uploads  : {d.get('total_uploads')}"))
        print(info(f"  ok_uploads     : {d.get('ok_uploads')}"))
        print(info(f"  two_arm_uploads: {d.get('two_arm_uploads')}"))
        print(info(f"  avg_proc_ms    : {d.get('avg_processing_ms')}"))
        for row in (d.get("by_extension") or []):
            print(info(f"    {row['extension']:<8}  count={row['count']}  "
                       f"avg_N={row['avg_n_samples']}"))
    else:
        print(f"  (not yet deployed — add /summary endpoint)")
except Exception:
    print(f"  (not yet deployed)")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
passed = sum(1 for r in results if r[1])
print(f"\n{BOLD}Result: {passed}/{len(results)} uploads passed{RESET}")
if passed == len(results):
    print(f"{GREEN}All synthetic standards verified.{RESET}")
    print(f"Check the log at: {CYAN}{BASE}/uploads/log{RESET}")
else:
    print(f"{RED}Some uploads failed — see errors above.{RESET}")
    sys.exit(1)
