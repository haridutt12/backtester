# ─── Stage 1: build dependencies ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps required by some Python packages (e.g. matplotlib, scipy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libfreetype6-dev \
    libpng-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ─── Stage 2: runtime image ───────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN useradd --create-home appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY --chown=appuser:appuser . .

# Ensure the config.json is readable
RUN chmod 644 config.json 2>/dev/null || true

USER appuser

# Expose the API port
EXPOSE 8000

# Health-check so orchestrators can detect readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Default: run the FastAPI server with uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
