# ══════════════════════════════════════════════════════════════════════════
#  Jalali Lab Optical Dashboard  —  Dockerfile
#  Build : docker build -t jalabi-dashboard .
#  Run   : docker compose up -d
# ══════════════════════════════════════════════════════════════════════════
FROM python:3.12-slim

# ── System deps ───────────────────────────────────────────────────────────
# wget: tiny (200 KB), used only for HEALTHCHECK; no curl needed
RUN apt-get update \
    && apt-get install -y --no-install-recommends wget \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user (security: container can't write outside /app) ──────────
RUN useradd -m -u 1001 appuser

WORKDIR /app

# ── Python dependencies (cached layer unless requirements change) ─────────
COPY optical_dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────
COPY optical_dashboard/ ./optical_dashboard/

# Uploads dir owned by appuser (mounted volume writes land here)
RUN mkdir -p /app/optical_dashboard/uploads \
    && chown -R appuser:appuser /app

USER appuser

WORKDIR /app/optical_dashboard

# ── Runtime env vars (overridable in docker-compose) ─────────────────────
ENV FLASK_HOST=0.0.0.0 \
    FLASK_PORT=5000 \
    MAX_UPLOAD_MB=64 \
    UPLOAD_TTL_S=3600 \
    PYTHONUNBUFFERED=1

EXPOSE 5000

# ── Healthcheck: /health must return 200 within 5 s ───────────────────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD wget -qO- http://localhost:5000/health || exit 1

CMD ["python", "app.py"]
