"""
optical_dashboard/app.py — Jalabi Lab Optical Signal Dashboard
==============================================================
Run:  python app.py  →  http://localhost:5000
Docker: docker compose up -d  →  http://localhost:5000

Features
--------
  Signal Analysis : upload .mat/CSV/.npy → time-domain, PSD, spectrogram, TD-GS phase
  QPSK Modem      : Gray-coded QPSK TX/RX, RRC pulse shaping, BER vs SNR
  48-ch WDM       : ITU-T G.694.1 C-band, mux/demux, per-channel power
  Digital Logic   : 2:1 MUX, D-latch, D flip-flop, 8-bit shift register
  3-D Hash        : sparse voxel hash, energy minimisation, LSH retrieval
  Upload log      : GET /uploads/log  — SQLite audit table, all upload metadata

Security
--------
  Filename:  regex-whitelist  [A-Za-z0-9_-.]{1,64}  + extension check
  Content:   null-byte check, max line length guard, size cap (env MAX_UPLOAD_MB)
  Network:   Flask container runs on internal Docker network (no outbound internet)
  Container: read-only rootfs, cap_drop=ALL, no-new-privileges
  Uploads:   UUID-isolated dirs, daemon-cleaned after UPLOAD_TTL_S seconds
  IP privacy: client IP stored as SHA-256 hash only — never raw

Two-click upload: selecting a file auto-processes it immediately.
"""

import os, io, re, uuid, time, hashlib, sqlite3, datetime, threading, traceback, base64
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal as sig

from flask import Flask, render_template, request, jsonify, abort
import dsp as DSP

# ── App setup ────────────────────────────────────────────────────────────────
_MAX_MB      = int(os.environ.get("MAX_UPLOAD_MB", 64))
_UPLOAD_TTL  = int(os.environ.get("UPLOAD_TTL_S", 3600))
_START_TIME  = time.time()
_VERSION     = "1.2.0"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = _MAX_MB * 1024 * 1024

UPLOAD_ROOT = Path(__file__).parent / "uploads"
UPLOAD_ROOT.mkdir(exist_ok=True)
UPLOAD_TTL  = _UPLOAD_TTL

# ── SQLite upload audit log ───────────────────────────────────────────────────
DB_PATH   = UPLOAD_ROOT / "upload_log.db"
_DB_LOCK  = threading.Lock()          # SQLite WAL is fine but one writer at a time

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS uploads (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT,
    filename         TEXT,
    extension        TEXT,
    size_bytes       INTEGER,
    uploaded_at      TEXT,           -- ISO-8601 UTC timestamp
    processing_ms    REAL,           -- wall-clock seconds × 1000
    client_ip_hash   TEXT,           -- SHA-256(IP) — privacy-safe
    user_agent       TEXT,
    referer          TEXT,
    two_arm          INTEGER,        -- 1 if .mat with I1+I2, else 0
    n_samples        INTEGER,
    dt_s             REAL,
    fs_hz            REAL,
    f_peak_hz        REAL,
    rms              REAL,
    D1_ps2           REAL,           -- fiber dispersion arm 1 (from metadata form)
    D2_ps2           REAL,           -- fiber dispersion arm 2
    lambda_nm        REAL,           -- center wavelength
    fs_GSas          REAL,           -- ADC sample rate
    lab              TEXT,           -- lab attribution
    experiment_desc  TEXT,           -- free-text description
    gs_iters         INTEGER,        -- GS iterations run
    status           TEXT,           -- 'ok' or 'error'
    error_msg        TEXT            -- NULL on success
);
"""

def _init_db():
    with _DB_LOCK:
        con = sqlite3.connect(str(DB_PATH))
        con.execute(_CREATE_TABLE)
        con.commit()
        con.close()

_init_db()

def _log_upload(row: dict):
    """Insert one row into the uploads audit table.  Never raises."""
    cols = [
        "session_id","filename","extension","size_bytes","uploaded_at",
        "processing_ms","client_ip_hash","user_agent","referer",
        "two_arm","n_samples","dt_s","fs_hz","f_peak_hz","rms",
        "D1_ps2","D2_ps2","lambda_nm","fs_GSas","lab","experiment_desc",
        "gs_iters","status","error_msg",
    ]
    placeholders = ",".join("?" for _ in cols)
    values = tuple(row.get(c) for c in cols)
    try:
        with _DB_LOCK:
            con = sqlite3.connect(str(DB_PATH))
            con.execute(
                f"INSERT INTO uploads ({','.join(cols)}) VALUES ({placeholders})",
                values
            )
            con.commit()
            con.close()
    except Exception as exc:
        # Non-fatal — never let a logging failure kill the upload response
        app.logger.warning(f"_log_upload failed: {exc}")

# ── Input sanitisation ────────────────────────────────────────────────────────
# Filename: printable ASCII, no path separators, no shell metacharacters
_SAFE_FNAME = re.compile(
    r'^[A-Za-z0-9_\-][A-Za-z0-9_\-\.]{0,60}'
    r'\.(csv|npy|txt|dat|mat)$',
    re.IGNORECASE,
)
_NULL_BYTES    = re.compile(rb'\x00')
_MAX_LINE_LEN  = 8192    # bytes per text line

def _sanitize_filename(raw: str) -> str:
    """Reject filenames that could cause path traversal or shell injection."""
    name = Path(raw).name          # strip any directory component
    if not _SAFE_FNAME.match(name):
        raise ValueError(f"Filename rejected: {name!r} — use only A-Z 0-9 _ - .")
    return name

def _sanitize_bytes(data: bytes, ext: str) -> bytes:
    """Light content scan for text files; binary formats skip text checks."""
    if ext == ".npy":
        # NumPy magic bytes: \x93NUMPY
        if len(data) > 6 and data[:6] != b'\x93NUMPY':
            raise ValueError("File claims .npy but missing NumPy magic header")
        return data
    if ext == ".mat":
        # MATLAB v5 magic: first 116 bytes are ASCII header starting with "MATLAB 5.0"
        # MATLAB v4 and HDF5-based v7.3 are also binary — just ensure non-empty
        if len(data) < 4:
            raise ValueError(".mat file too small")
        return data
    if _NULL_BYTES.search(data):
        raise ValueError("File contains null bytes — binary content in text file?")
    for i, line in enumerate(data.split(b'\n')[:20]):
        if len(line) > _MAX_LINE_LEN:
            raise ValueError(f"Line {i+1} exceeds {_MAX_LINE_LEN} bytes — file looks malformed")
    return data

# ── Query parameter helpers (reject out-of-range / non-numeric kwargs) ────────
def _qfloat(name: str, default: float, lo: float, hi: float) -> float:
    raw = request.args.get(name, str(default))
    try:
        v = float(raw)
    except (ValueError, TypeError):
        abort(400, description=f"Parameter '{name}' must be a number, got {raw!r}")
    if not (lo <= v <= hi):
        abort(400, description=f"Parameter '{name}'={v} out of range [{lo}, {hi}]")
    return v

def _qint(name: str, default: int, lo: int, hi: int) -> int:
    raw = request.args.get(name, str(default))
    try:
        v = int(raw)
    except (ValueError, TypeError):
        abort(400, description=f"Parameter '{name}' must be an integer, got {raw!r}")
    if not (lo <= v <= hi):
        abort(400, description=f"Parameter '{name}'={v} out of range [{lo}, {hi}]")
    return v

# ── Periodic upload cleanup (daemon thread) ───────────────────────────────────
def _cleanup():
    while True:
        time.sleep(300)
        now = time.time()
        try:
            for d in UPLOAD_ROOT.iterdir():
                if d.is_dir() and (now - d.stat().st_mtime) > _UPLOAD_TTL:
                    import shutil; shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass

threading.Thread(target=_cleanup, daemon=True).start()

# ── Plot helpers ──────────────────────────────────────────────────────────────
FG  = (0.04, 0.04, 0.10)
BG  = (0.06, 0.06, 0.14)

def fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    b64 = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    return b64

def dark(ax, title="", xl="", yl=""):
    ax.set_facecolor(BG)
    for s in ax.spines.values(): s.set_color("#334466")
    ax.tick_params(colors="#99aabb", labelsize=8)
    if title: ax.set_title(title, color="white", fontsize=9, fontweight="bold", pad=5)
    if xl:    ax.set_xlabel(xl, color="#99aabb", fontsize=8)
    if yl:    ax.set_ylabel(yl, color="#99aabb", fontsize=8)

# ── Data loader ───────────────────────────────────────────────────────────────
def load_optical(path: Path):
    """
    Load optical intensity data from CSV / NPY / MAT / TXT / DAT.

    For .mat files with an I1/I2 pair, both arms are returned so the
    Signal Analysis tab can display them side-by-side and run TD-GS.
    For single-channel files, I2 is set to None.
    Returns (t, y, y2_or_None).
    """
    ext = path.suffix.lower()

    if ext == ".mat":
        t, y, y2 = DSP.load_mat(path)
        # normalise
        for arr in (y, y2) if y2 is not None else (y,):
            if np.ptp(arr) > 0:
                arr -= arr.min(); arr /= (np.ptp(arr) + 1e-30)
        return t.astype(float), y.astype(float), y2

    if ext == ".npy":
        arr = np.load(path)
    else:
        rows = []
        for line in path.read_text(errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#"): continue
            try:
                rows.append([float(x) for x in line.replace(",", " ").split()])
            except ValueError:
                continue
        arr = np.array(rows) if rows else np.zeros((2, 2))

    if arr.ndim == 2 and arr.shape[1] >= 2:
        t, y = arr[:, 0], arr[:, 1]
    else:
        y = arr.ravel()
        t = np.arange(len(y), dtype=float)

    y = y.astype(float)
    if np.ptp(y) > 0:
        y = (y - y.min()) / np.ptp(y)
    return t.astype(float), y, None

# ── Signal analysis plot ──────────────────────────────────────────────────────
def analyse_plot(t, y, y2=None):
    N  = len(y)
    dt = float(np.mean(np.diff(t))) if N > 1 else 1.0
    fs = 1.0 / (dt + 1e-30)

    freqs = np.fft.rfftfreq(N, d=dt)
    Y     = np.fft.rfft(y)
    psd   = np.abs(Y) ** 2
    f_pk  = freqs[np.argmax(psd[1:]) + 1] if len(psd) > 1 else 0.0

    env   = np.abs(sig.hilbert(y))
    npseg = min(max(64, N // 32), 512)
    f_sp, t_sp, Sxx = sig.spectrogram(y, fs=fs, nperseg=npseg, noverlap=npseg//2)

    D1, D2 = -600.0, -1200.0
    nu  = np.fft.fftfreq(N)
    H   = np.exp(1j * np.pi * (D2 - D1) * nu ** 2)
    E   = np.sqrt(np.maximum(y, 0)).astype(complex)
    if y2 is not None:
        # Full 2-arm TD-GS (50 iterations)
        rx = DSP.OpticalRxFSM(D1=D1, D2=D2, n_iter=50)
        phi, _ = rx.run(y, y2)
    else:
        phi = np.angle(np.fft.ifft(np.fft.fft(E) * H))

    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor(FG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)

    ax0 = fig.add_subplot(gs[0, 0])
    label_y = "I₁(t)" if y2 is not None else "signal"
    ax0.plot(t, y,   color="#50d8ff", lw=0.8, alpha=0.7, label=label_y)
    if y2 is not None:
        ax0.plot(t, y2, color="#ff3278", lw=0.8, alpha=0.7, label="I₂(t)  D₂")
    ax0.plot(t, env, color="#ffd040", lw=1.3, label="envelope")
    ax0.legend(fontsize=7, facecolor=BG, labelcolor="white")
    arm_note = "  (2-arm .mat)" if y2 is not None else ""
    dark(ax0, f"Time Domain  N={N}{arm_note}", "t", "amp")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.semilogy(freqs, np.maximum(psd, 1e-12), color="#00ff9f", lw=1.0)
    ax1.axvline(f_pk, color="#ff3278", lw=1, ls="--", label=f"f={f_pk:.3g}")
    ax1.legend(fontsize=7, facecolor=BG, labelcolor="white")
    dark(ax1, "Power Spectrum", "freq", "PSD")

    ax2 = fig.add_subplot(gs[0, 2])
    im  = ax2.pcolormesh(t_sp, f_sp,
                         10 * np.log10(np.maximum(Sxx, 1e-20)),
                         cmap="inferno", shading="auto")
    fig.colorbar(im, ax=ax2, label="dB").ax.tick_params(colors="#99aabb", labelsize=6)
    dark(ax2, "STFT Spectrogram", "t", "freq")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t, phi, color="#cc88ff", lw=0.8)
    dark(ax3, "TD-GS Phase (1 iter)\nD1=-600 D2=-1200 ps²", "t", "φ [rad]")

    ax4 = fig.add_subplot(gs[1, 1])
    f_i = np.diff(np.unwrap(np.angle(sig.hilbert(y)))) / (2 * np.pi * dt)
    ax4.plot(t[:-1], f_i, color="#ffd040", lw=0.8)
    dark(ax4, "Instantaneous Freq (Hilbert)", "t", "f_inst")

    ax5 = fig.add_subplot(gs[1, 2])
    lags = sig.correlation_lags(N, N, mode="full")
    ac   = np.correlate(y - y.mean(), y - y.mean(), mode="full")
    ac  /= ac.max() + 1e-12
    mid  = len(ac) // 2; half = min(N // 4, 400)
    ax5.plot(lags[mid-half:mid+half] * dt, ac[mid-half:mid+half],
             color="#50d8ff", lw=0.8)
    ax5.axhline(0, color="gray", lw=0.5, ls=":")
    dark(ax5, "Autocorrelation R(τ)", "lag", "R")

    fig.suptitle("Jalali Lab Optical Dashboard",
                 color="white", fontsize=11, fontweight="bold")
    stats = {
        "N": N, "dt": f"{dt:.4g}", "fs": f"{fs:.4g}",
        "f_peak": f"{f_pk:.4g}",
        "rms": f"{float(np.sqrt(np.mean(y**2))):.4f}",
        "two_arm": y2 is not None,
        "gs_iters": 50 if y2 is not None else 1,
    }
    return fig_b64(fig), stats

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

# ── 0. Health / boolean status ────────────────────────────────────────────────
@app.route("/health")
def health():
    """Docker HEALTHCHECK target — returns 200 + JSON while the app is alive."""
    # Quick upload stats from DB
    try:
        with _DB_LOCK:
            con = sqlite3.connect(str(DB_PATH))
            total_uploads = con.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]
            ok_uploads    = con.execute("SELECT COUNT(*) FROM uploads WHERE status='ok'").fetchone()[0]
            con.close()
    except Exception:
        total_uploads = ok_uploads = None

    return jsonify({
        "alive":         True,
        "version":       _VERSION,
        "uptime_s":      int(time.time() - _START_TIME),
        "upload_root":   str(UPLOAD_ROOT),
        "uploads_total": total_uploads,
        "uploads_ok":    ok_uploads,
        "log_endpoint":  "/uploads/log",
    })

# ── 0b. Aggregate summary (by extension, by lab, two-arm ratio, avg time) ────
@app.route("/summary")
def summary():
    """
    Aggregate statistics from the upload audit log.
    Useful for: Jalabi Lab aggregate study of optical standard uploads.

    Returns:
      total_uploads, ok_uploads, error_uploads, two_arm_uploads
      avg_processing_ms
      by_extension  : count, ok_count, avg_n_samples, avg_processing_ms
      by_lab        : count, ok_count, two_arm_count
      recent_formats: last 20 distinct experiment descriptions
    """
    try:
        with _DB_LOCK:
            con = sqlite3.connect(str(DB_PATH))
            con.row_factory = sqlite3.Row

            total    = con.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]
            ok_cnt   = con.execute("SELECT COUNT(*) FROM uploads WHERE status='ok'").fetchone()[0]
            err_cnt  = con.execute("SELECT COUNT(*) FROM uploads WHERE status='error'").fetchone()[0]
            two_arm  = con.execute("SELECT COUNT(*) FROM uploads WHERE two_arm=1").fetchone()[0]
            avg_ms   = con.execute(
                "SELECT AVG(processing_ms) FROM uploads WHERE status='ok'"
            ).fetchone()[0]

            by_ext = [dict(r) for r in con.execute("""
                SELECT extension,
                       COUNT(*)                         AS count,
                       SUM(CASE WHEN status='ok' THEN 1 ELSE 0 END) AS ok_count,
                       AVG(n_samples)                   AS avg_n_samples,
                       AVG(processing_ms)               AS avg_processing_ms,
                       SUM(CASE WHEN two_arm=1 THEN 1 ELSE 0 END)   AS two_arm_count
                FROM uploads
                GROUP BY extension
                ORDER BY count DESC
            """).fetchall()]

            by_lab = [dict(r) for r in con.execute("""
                SELECT COALESCE(NULLIF(lab,''), '(unset)') AS lab,
                       COUNT(*)                            AS count,
                       SUM(CASE WHEN status='ok' THEN 1 ELSE 0 END) AS ok_count,
                       SUM(CASE WHEN two_arm=1 THEN 1 ELSE 0 END)   AS two_arm_count,
                       AVG(n_samples)                      AS avg_n_samples
                FROM uploads
                GROUP BY lab
                ORDER BY count DESC
                LIMIT 20
            """).fetchall()]

            recent_formats = [dict(r) for r in con.execute("""
                SELECT DISTINCT experiment_desc, extension, D1_ps2, D2_ps2,
                       lambda_nm, fs_GSas, uploaded_at
                FROM uploads
                WHERE status='ok' AND experiment_desc != '' AND experiment_desc IS NOT NULL
                ORDER BY id DESC
                LIMIT 20
            """).fetchall()]

            con.close()

        return jsonify({
            "total_uploads":    total,
            "ok_uploads":       ok_cnt,
            "error_uploads":    err_cnt,
            "two_arm_uploads":  two_arm,
            "two_arm_pct":      round(100.0 * two_arm / max(ok_cnt, 1), 1),
            "avg_processing_ms": round(avg_ms or 0, 1),
            "by_extension":     by_ext,
            "by_lab":           by_lab,
            "recent_formats":   recent_formats,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── 0c. Alarm endpoint — 200 healthy, 503 degraded ────────────────────────────
@app.route("/alarm")
def alarm():
    """
    Simple health + threshold alarm endpoint.
    Returns HTTP 200 if healthy, HTTP 503 if any alarm condition triggers.
    UptimeRobot / cron / Cloudflare can monitor this URL.

    Alarm conditions (configurable via query params):
      max_error_pct   — max % of uploads that are errors   (default 50)
      min_ok          — min total ok uploads to arm alarm   (default 10)
    """
    max_err_pct = float(request.args.get("max_error_pct", 50))
    min_ok      = int(request.args.get("min_ok", 10))

    alarms = []
    try:
        with _DB_LOCK:
            con = sqlite3.connect(str(DB_PATH))
            total   = con.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]
            ok_cnt  = con.execute("SELECT COUNT(*) FROM uploads WHERE status='ok'").fetchone()[0]
            err_cnt = con.execute("SELECT COUNT(*) FROM uploads WHERE status='error'").fetchone()[0]
            # Last upload time
            last_row = con.execute(
                "SELECT uploaded_at, status FROM uploads ORDER BY id DESC LIMIT 1"
            ).fetchone()
            con.close()

        err_pct = 100.0 * err_cnt / max(total, 1)

        if ok_cnt >= min_ok and err_pct > max_err_pct:
            alarms.append(f"error_rate {err_pct:.1f}% > threshold {max_err_pct}%")

        status_code = 503 if alarms else 200
        return jsonify({
            "alarm":          bool(alarms),
            "conditions":     alarms,
            "total_uploads":  total,
            "ok_uploads":     ok_cnt,
            "error_uploads":  err_cnt,
            "error_pct":      round(err_pct, 1),
            "last_upload":    dict(last_row) if last_row else None,
            "uptime_s":       int(time.time() - _START_TIME),
            "version":        _VERSION,
        }), status_code

    except Exception as e:
        return jsonify({"alarm": True, "conditions": [str(e)]}), 503


# ── 0e. Upload audit log (read-only JSON view of SQLite) ─────────────────────
@app.route("/uploads/log")
def uploads_log():
    """
    Return the upload audit log as JSON.
    Query params:
      limit  — max rows returned  (default 100, max 1000)
      offset — row offset         (default 0)
      status — filter 'ok' / 'error' / 'all' (default 'all')
    """
    limit  = min(int(request.args.get("limit",  100)), 1000)
    offset = max(int(request.args.get("offset", 0)),   0)
    status_filter = request.args.get("status", "all")

    sql = "SELECT * FROM uploads"
    params: list = []
    if status_filter in ("ok", "error"):
        sql += " WHERE status = ?"
        params.append(status_filter)
    sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params += [limit, offset]

    try:
        with _DB_LOCK:
            con = sqlite3.connect(str(DB_PATH))
            con.row_factory = sqlite3.Row
            rows = [dict(r) for r in con.execute(sql, params).fetchall()]
            total = con.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]
            con.close()
        return jsonify({"total": total, "limit": limit, "offset": offset,
                        "rows": rows})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── 1. Optical file upload (two-click, UUID-isolated, sanitised) ──────────────
@app.route("/upload", methods=["POST"])
def upload():
    _t0        = time.monotonic()
    session_id = str(uuid.uuid4())

    # ── Shared metadata from form fields (sent alongside the file) ────────────
    def _form_float(name, default=None):
        v = request.form.get(name, "")
        try: return float(v)
        except (ValueError, TypeError): return default

    def _form_str(name, maxlen=200):
        return (request.form.get(name, "") or "")[:maxlen]

    meta_D1   = _form_float("D1",   -600.0)
    meta_D2   = _form_float("D2",  -1200.0)
    meta_lam  = _form_float("lam",  1550.0)
    meta_fs   = _form_float("fs",     56.0)
    meta_lab  = _form_str("lab")
    meta_desc = _form_str("desc", 500)

    # Privacy-safe client fingerprint
    raw_ip   = (request.headers.get("X-Forwarded-For") or
                request.remote_addr or "")
    ip_hash  = hashlib.sha256(raw_ip.encode()).hexdigest()
    ua       = (request.user_agent.string or "")[:300]
    referer  = (request.referrer or "")[:300]
    now_iso  = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    # Base log row — filled in further as we go
    log_row = dict(
        session_id=session_id, uploaded_at=now_iso,
        client_ip_hash=ip_hash, user_agent=ua, referer=referer,
        D1_ps2=meta_D1, D2_ps2=meta_D2, lambda_nm=meta_lam,
        fs_GSas=meta_fs, lab=meta_lab, experiment_desc=meta_desc,
        status="error",
    )

    try:
        if "file" not in request.files:
            log_row.update(error_msg="No file field", processing_ms=round((time.monotonic()-_t0)*1000,1))
            _log_upload(log_row)
            return jsonify({"error": "No file field"}), 400
        f = request.files["file"]
        if not f.filename:
            log_row.update(error_msg="Empty filename", processing_ms=round((time.monotonic()-_t0)*1000,1))
            _log_upload(log_row)
            return jsonify({"error": "Empty filename"}), 400

        # ── Regex filename whitelist ──────────────────────────────────────────
        try:
            safe_name = _sanitize_filename(f.filename)
        except ValueError as e:
            log_row.update(filename=f.filename[:120], error_msg=str(e),
                           processing_ms=round((time.monotonic()-_t0)*1000,1))
            _log_upload(log_row)
            return jsonify({"error": str(e)}), 400

        ext = Path(safe_name).suffix.lower()
        log_row.update(filename=safe_name, extension=ext)

        # ── UUID-isolated session directory ───────────────────────────────────
        sess_dir = UPLOAD_ROOT / session_id
        sess_dir.mkdir()

        # ── Read, sanitise bytes, save ─────────────────────────────────────────
        raw = f.read()
        log_row["size_bytes"] = len(raw)

        if len(raw) > _MAX_MB * 1024 * 1024:
            msg = f"File exceeds {_MAX_MB} MB limit"
            log_row.update(error_msg=msg, processing_ms=round((time.monotonic()-_t0)*1000,1))
            _log_upload(log_row)
            return jsonify({"error": msg}), 400
        try:
            raw = _sanitize_bytes(raw, ext)
        except ValueError as e:
            log_row.update(error_msg=str(e), processing_ms=round((time.monotonic()-_t0)*1000,1))
            _log_upload(log_row)
            return jsonify({"error": str(e)}), 400

        save_path = sess_dir / safe_name
        save_path.write_bytes(raw)

        t, y, y2 = load_optical(save_path)
        if len(y) < 16:
            msg = f"Too few samples: {len(y)}"
            log_row.update(error_msg=msg, n_samples=len(y),
                           processing_ms=round((time.monotonic()-_t0)*1000,1))
            _log_upload(log_row)
            return jsonify({"error": msg}), 400

        plot, stats = analyse_plot(t, y, y2=y2)
        has_two_arms = y2 is not None

        # ── Record DSP-extracted metadata ─────────────────────────────────────
        try:
            dt_val = float(stats["dt"])
            fs_val = float(stats["fs"])
        except Exception:
            dt_val = None; fs_val = None
        log_row.update(
            two_arm    = 1 if has_two_arms else 0,
            n_samples  = int(stats["N"]),
            dt_s       = dt_val,
            fs_hz      = fs_val,
            f_peak_hz  = float(stats.get("f_peak", 0) or 0),
            rms        = float(stats.get("rms", 0) or 0),
            gs_iters   = stats.get("gs_iters", 1),
            status     = "ok",
            error_msg  = None,
            processing_ms = round((time.monotonic()-_t0)*1000, 1),
        )
        _log_upload(log_row)

        return jsonify({"plot": plot, "stats": stats, "session": session_id,
                        "filename": safe_name,
                        "two_arm": has_two_arms})
    except Exception as e:
        log_row.update(error_msg=str(e)[:500],
                       processing_ms=round((time.monotonic()-_t0)*1000,1))
        _log_upload(log_row)
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

# ── 2. Demo (synthetic STEAM pulse) ───────────────────────────────────────────
@app.route("/demo")
def demo():
    rng = np.random.default_rng(42)
    N   = 4096
    t   = np.linspace(-200, 200, N)
    env = np.exp(-t**2 / (2 * 30**2))
    phi = 0.8 * np.exp(-((t-10)/15)**2) * np.sin(2*np.pi*t/60)
    y   = env**2 * (1 + 0.3*np.cos(phi)) + 0.02*rng.normal(size=N)
    y   = np.clip(y, 0, None)
    plot, stats = analyse_plot(t, y)
    return jsonify({"plot": plot, "stats": stats,
                    "source": "Synthetic STEAM: dispersion-stretched Gaussian + cell phase",
                    "two_arm": False})

# ── 3. QPSK modem ─────────────────────────────────────────────────────────────
@app.route("/qpsk")
def qpsk_route():
    snr  = _qfloat("snr",   12.0,   0.0,  30.0)
    n_b  = _qint  ("nbits", 2048,   64,   16384)
    data = DSP.simulate_link(n_bits=n_b, snr_db=snr)

    snr_range   = np.array(data["snr_range"])
    ber_theory  = np.array(data["ber_theory"])
    syms_i      = np.array(data["syms_i"])
    syms_q      = np.array(data["syms_q"])
    tx_real     = np.array(data["tx_real"])
    rx_real     = np.array(data["rx_real"])

    fig = plt.figure(figsize=(13, 8))
    fig.patch.set_facecolor(FG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.42)

    # Constellation
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.scatter(syms_i, syms_q, c="#50d8ff", s=4, alpha=0.5)
    for (i_ref, q_ref), col in zip(
            [( 1, 1),(-1, 1),(-1,-1),( 1,-1)],
            ["#ffd040","#00ff9f","#ff3278","#cc88ff"]):
        ax0.plot(i_ref/np.sqrt(2), q_ref/np.sqrt(2), "+", ms=14, mew=2, color=col)
    ax0.axhline(0, color="gray", lw=0.5); ax0.axvline(0, color="gray", lw=0.5)
    ax0.set_aspect("equal"); ax0.set_xlim(-2, 2); ax0.set_ylim(-2, 2)
    dark(ax0, f"QPSK Constellation\nSNR={snr:.0f}dB  BER={data['ber']:.2e}", "I", "Q")

    # BER curve
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.semilogy(snr_range, ber_theory, color="#ffd040", lw=2, label="theory")
    ax1.axvline(snr, color="#ff3278", lw=1, ls="--", label=f"current SNR")
    ax1.axhline(data["ber"], color="#50d8ff", lw=1, ls=":", label=f"meas BER={data['ber']:.1e}")
    ax1.set_xlim(0, 20); ax1.set_ylim(1e-6, 0.6)
    ax1.legend(fontsize=7, facecolor=BG, labelcolor="white")
    dark(ax1, "BER vs SNR (QPSK)", "SNR [dB]", "BER")

    # Waveform
    ax2 = fig.add_subplot(gs[0, 2])
    t_w = np.arange(len(tx_real))
    ax2.plot(t_w, tx_real, color="#00ff9f", lw=0.6, alpha=0.7, label="TX")
    ax2.plot(t_w, rx_real, color="#ff7040", lw=0.6, alpha=0.6, label="RX (AWGN)")
    ax2.legend(fontsize=7, facecolor=BG, labelcolor="white")
    dark(ax2, "TX vs RX Waveform\n(RRC pulse-shaped)", "sample", "amplitude")

    # I/Q eye diagram (real part)
    sps = 8
    ax3 = fig.add_subplot(gs[1, 0:2])
    eye = rx_real[:min(len(rx_real), sps*64)]
    for k in range(0, len(eye) - 2*sps, 2*sps):
        ax3.plot(np.arange(2*sps), eye[k:k+2*sps], color="#50d8ff",
                 lw=0.5, alpha=0.15)
    ax3.axvline(sps, color="#ffd040", lw=1, ls="--", label="decision point")
    ax3.legend(fontsize=7, facecolor=BG, labelcolor="white")
    dark(ax3, "Eye Diagram (I channel, RRC matched filter RX)",
         "sample within symbol", "amplitude")

    # Phase trellis
    ax4 = fig.add_subplot(gs[1, 2])
    phases_tx = np.angle(DSP.qpsk_modulate(
        np.random.default_rng(0).integers(0, 2, 64)
    )) * 180 / np.pi
    ax4.step(range(len(phases_tx)), phases_tx, color="#cc88ff", lw=1.5, where="post")
    ax4.set_yticks([45, 135, -135, -45])
    dark(ax4, "TX Symbol Phase Trellis\nGray-coded QPSK", "symbol idx", "phase [deg]")

    fig.suptitle(f"QPSK Modem — N={n_b} bits  SNR={snr:.0f} dB  BER={data['ber']:.2e}",
                 color="white", fontsize=10, fontweight="bold")
    plot = fig_b64(fig)
    return jsonify({"plot": plot, "ber": data["ber"], "snr_db": snr})

# ── 4. 48-channel WDM ─────────────────────────────────────────────────────────
@app.route("/wdm")
def wdm_route():
    n_ch = _qint  ("nch", 48,   4,  48)
    snr  = _qfloat("snr", 25.0, 5.0, 40.0)
    data = DSP.wdm_sim(n_ch=n_ch, snr_db=snr)

    freqs  = np.array(data["freqs_thz"])
    lams   = np.array(data["lams_nm"])
    powers = np.array(data["ch_powers_db"])

    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor(FG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.35)

    # Channel power bar chart
    ax0 = fig.add_subplot(gs[0, 0:2])
    colors_ch = plt.cm.plasma(np.linspace(0.1, 0.9, n_ch))
    ax0.bar(range(n_ch), powers, color=colors_ch, width=0.8)
    ax0.axhline(-3, color="white", lw=0.7, ls="--", label="-3 dB")
    ax0.set_xticks(range(0, n_ch, 6))
    ax0.set_xticklabels([f"ch{i+1}" for i in range(0, n_ch, 6)],
                        color="#99aabb", fontsize=7)
    ax0.legend(fontsize=7, facecolor=BG, labelcolor="white")
    dark(ax0,
         f"{n_ch}-ch DWDM C-band — Per-Channel Demux Power\n"
         f"ITU-T G.694.1, 100 GHz spacing, {lams[0]:.1f}–{lams[-1]:.1f} nm",
         "channel", "relative power [dB]")

    # Wavelength axis
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.scatter(lams, powers, c=colors_ch, s=30, zorder=3)
    ax1.stem(lams, powers,
             linefmt="gray", markerfmt=" ", basefmt="gray",
             orientation="vertical")
    dark(ax1, "Power vs Wavelength (C-band)", "lambda [nm]", "power [dB]")

    # Frequency axis
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.scatter(freqs, powers, c=colors_ch, s=30, zorder=3)
    # Add wideband PSD overlay (first 2000 points around band)
    psd_arr  = np.array(data["psd_plot_db"])
    freq_arr = np.array(data["freqs_plot_thz"])
    band_mask = (freq_arr > freqs.min() - 0.5) & (freq_arr < freqs.max() + 0.5)
    if band_mask.sum() > 2:
        ax2.plot(freq_arr[band_mask], psd_arr[band_mask],
                 color="white", lw=0.5, alpha=0.25, label="muxed PSD")
    ax2.legend(fontsize=7, facecolor=BG, labelcolor="white")
    dark(ax2, "Power vs Frequency (THz)", "freq [THz]", "power [dB]")

    fig.suptitle(
        f"{n_ch}-Channel DWDM C-band  |  100 GHz ITU Grid  |"
        f"  SNR={snr:.0f} dB  |  QPSK/channel",
        color="white", fontsize=10, fontweight="bold"
    )
    return jsonify({"plot": fig_b64(fig),
                    "n_ch": n_ch, "lam_min": float(lams.min()),
                    "lam_max": float(lams.max())})

# ── 5. Digital logic demo ──────────────────────────────────────────────────────
@app.route("/digital")
def digital_route():
    data_byte = _qint("byte",   0xB2, 0, 255)
    _         = _qint("cycles",   16, 4,  64)   # validate even if unused below
    log = DSP.digital_demo(data_byte, n_cycles=16)

    cycles = [e["cycle"]  for e in log]
    clks   = [e["CLK"]    for e in log]
    ens    = [e["EN"]      for e in log]
    bits_in= [e["bit_in"] for e in log]
    latches= [int(e["latch"], 2) for e in log]
    ffs    = [int(e["ff"],    2) for e in log]
    srs    = [int(e["sr"],    2) for e in log]
    muxes  = [int(e["mux"],   2) for e in log]

    # TDM demo
    rng  = np.random.default_rng(7)
    a_s  = rng.integers(0, 2, 12).astype(int)
    b_s  = rng.integers(0, 2, 12).astype(int)
    mux2 = DSP.tdm_mux(a_s, b_s)
    a_rx, b_rx = DSP.tdm_demux(mux2)
    tdm_ok = (a_rx[:len(a_s)] == a_s).all() and (b_rx[:len(b_s)] == b_s).all()

    fig = plt.figure(figsize=(13, 8))
    fig.patch.set_facecolor(FG)
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.60, wspace=0.35)

    # Register state over cycles
    ax0 = fig.add_subplot(gs[0, 0:2])
    ax0.step(cycles, latches, color="#ffd040", lw=1.5, label="D-latch Q", where="post")
    ax0.step(cycles, ffs,     color="#50d8ff", lw=1.5, label="D flip-flop Q", where="post", ls="--")
    ax0.step(cycles, srs,     color="#cc88ff", lw=1.2, label="Shift register", where="post", ls=":")
    ax0.step(cycles, muxes,   color="#00ff9f", lw=1.0, label="2:1 MUX out", where="post", ls="-.")
    ax0.legend(fontsize=8, facecolor=BG, labelcolor="white", ncol=4)
    dark(ax0, f"Digital Logic — 8-bit states vs clock cycle  (data=0x{data_byte:02X}={data_byte:08b}b)",
         "cycle", "register value (uint8)")

    # CLK and EN signals
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.step(cycles, clks,   color="#ffd040", lw=1.5, label="CLK",  where="post")
    ax1.step(cycles, ens,    color="#ff3278", lw=1.5, label="EN",   where="post", ls="--")
    ax1.step(cycles, bits_in, color="#00ff9f", lw=1.2, label="D_in", where="post", ls=":")
    ax1.set_ylim(-0.2, 1.5)
    ax1.legend(fontsize=7, facecolor=BG, labelcolor="white")
    dark(ax1, "Control Signals", "cycle", "")

    # Shift register bit waterfall
    sr_bits = np.array([[int(b) for b in e["sr"]] for e in log])
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.imshow(sr_bits.T, aspect="auto", cmap="Blues",
               vmin=0, vmax=1, origin="upper",
               extent=[0, len(log)-1, 7.5, -0.5])
    ax2.set_yticks(range(8)); ax2.set_yticklabels(range(7,-1,-1), color="#99aabb", fontsize=7)
    dark(ax2, "Shift Register — bit waterfall\n(row = bit position, col = cycle)",
         "cycle", "bit index")

    # TDM mux/demux
    ax3 = fig.add_subplot(gs[2, 0:2])
    x = np.arange(len(mux2))
    ax3.step(np.arange(len(a_s)), a_s, color="#50d8ff", lw=1.5, label="A (TX)", where="post")
    ax3.step(np.arange(len(b_s)), b_s, color="#ffd040", lw=1.5, label="B (TX)", where="post", ls="--")
    # Interleaved overlay
    ax3b = ax3.twinx()
    ax3b.step(x, mux2, color="#cc88ff", lw=1.0, alpha=0.6, label="MUX A,B", where="post")
    ax3b.set_ylabel("MUX output", color="#cc88ff", fontsize=8)
    ax3b.tick_params(colors="#cc88ff", labelsize=7)
    for sp in ax3b.spines.values(): sp.set_color("#334466")
    ax3.legend(fontsize=7, facecolor=BG, labelcolor="white")
    ax3b.legend(fontsize=7, facecolor=BG, labelcolor="white", loc="upper right")
    dark(ax3, f"TDM 2:1 Mux/Demux  |  round-trip {'OK' if tdm_ok else 'FAIL'} "
              f"({'identical' if tdm_ok else 'MISMATCH'})",
         "sample", "bit value")

    fig.suptitle(
        f"Digital Logic: D-Latch, D-FF, 8-bit Shift Register, 2:1 MUX, TDM  "
        f"|  data=0b{data_byte:08b}",
        color="white", fontsize=9.5, fontweight="bold"
    )
    return jsonify({"plot": fig_b64(fig), "tdm_ok": tdm_ok,
                    "log": log[:8]})   # first 8 cycles for debug

# ── 6. 3-D Optical Hash + Energy Minimisation ─────────────────────────────
@app.route("/hash3d")
def hash3d_route():
    n_pts = _qint("npts", 256, 32, 512)
    data  = DSP.optical_hash_demo(n_points=n_pts)

    x_um    = np.array(data["x_um"])
    y_um    = np.array(data["y_um"])
    lam_nm  = np.array(data["lam_nm"])
    amp     = np.array(data["amp"])
    phi     = np.array(data["phi_rad"])
    iters   = np.array(data["iters"])
    H_tv    = np.array(data["H_tv"])
    H_pc    = np.array(data["H_pc"])
    H_tot   = np.array(data["H_total"])
    bcounts = np.array(data["bucket_counts"])

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor(FG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.38)

    # ── (0,0) Spatial scatter: amplitude ──────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    sc0 = ax0.scatter(x_um, y_um, c=amp, cmap="plasma", s=10, alpha=0.7)
    fig.colorbar(sc0, ax=ax0, label="|E|").ax.tick_params(colors="#99aabb", labelsize=6)
    dark(ax0, f"Wavefront Amplitude  |E(x,y)|  N={n_pts}", "x [μm]", "y [μm]")

    # ── (0,1) λ-scatter: phase ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    sc1 = ax1.scatter(lam_nm, amp, c=phi, cmap="hsv", s=8, alpha=0.6, vmin=-PI, vmax=PI)
    fig.colorbar(sc1, ax=ax1, label="φ [rad]").ax.tick_params(colors="#99aabb", labelsize=6)
    dark(ax1, "Field vs Wavelength  |E|(λ)", "λ [nm]", "|E|")

    # ── (0,2) Hash bucket histogram ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.bar(range(len(bcounts)), bcounts,
            color=plt.cm.cool(np.linspace(0.1, 0.9, len(bcounts))), width=0.9)
    ax2.axhline(n_pts / len(bcounts), color="#ffd040", lw=1.2, ls="--",
                label=f"ideal={n_pts/len(bcounts):.1f}")
    ax2.legend(fontsize=7, facecolor=BG, labelcolor="white")
    dark(ax2, f"Polynomial Hash Bucket Distribution\n"
              f"collision rate={data['collision_rate']:.2%}",
         "bucket index", "voxel count")

    # ── (1,0:2) Energy convergence ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0:2])
    ax3.semilogy(iters, H_tot, color="#50d8ff", lw=2.2, label=f"H_total (init={data['H_init']:.2f}→{data['H_final']:.2f})")
    ax3.semilogy(iters, H_tv,  color="#ffd040", lw=1.4, ls="--", label="H_TV  (total variation)")
    ax3.semilogy(iters, H_pc,  color="#ff3278", lw=1.4, ls=":",  label="H_pc  (phase constraint)")
    ax3.legend(fontsize=7.5, facecolor=BG, labelcolor="white")
    ax3.set_xlim(0, len(iters)-1)
    dark(ax3,
         f"Energy Minimisation — Gradient Descent on H = 0.1·H_TV + 0.4·H_pc\n"
         f"Reduction {data['reduction']:.1%}   ({len(iters)} iterations, lr=0.035)",
         "iteration", "energy H")

    # ── (1,2) (x,y,λ) 3-D projection: two 2-D panels ─────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    sc4 = ax4.scatter(x_um, lam_nm, c=amp, cmap="viridis", s=7, alpha=0.6)
    fig.colorbar(sc4, ax=ax4, label="|E|").ax.tick_params(colors="#99aabb", labelsize=6)
    dark(ax4, "3-D Projection  (x, λ) plane", "x [μm]", "λ [nm]")

    fig.suptitle(
        f"Optical 3-D Voxel Hash  |  Energy Minimisation  |  LSH retrieval  "
        f"|  N={data['n_stored']} voxels",
        color="white", fontsize=10, fontweight="bold"
    )
    return jsonify({
        "plot":           fig_b64(fig),
        "n_stored":       data["n_stored"],
        "H_init":         data["H_init"],
        "H_final":        data["H_final"],
        "reduction":      data["reduction"],
        "collision_rate": data["collision_rate"],
    })


if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"Jalali Lab Optical Dashboard -> http://{host}:{port}")
    print(f"Upload folder : {UPLOAD_ROOT}")
    print(f"Max upload    : {_MAX_MB} MB   TTL: {_UPLOAD_TTL}s   version: {_VERSION}")
    app.run(host=host, port=port, debug=debug, use_reloader=False)
