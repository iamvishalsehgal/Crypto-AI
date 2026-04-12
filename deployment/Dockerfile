# ---------------------------------------------------------------------------
# Stage 1 -- builder: install Python dependencies into a virtual-env
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Prevent Python from writing .pyc files and enable unbuffered output.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# Install build-time system dependencies (gcc is needed for some wheels).
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libffi-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy only requirements first to maximise Docker layer caching.
COPY requirements.txt .

# Install Python packages into a virtual-env so we can copy them cleanly.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2 -- runtime: lean image with only what we need
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Inherit the virtual-env from the builder stage.
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create a non-root user for security.
RUN groupadd --gid 1000 botuser && \
    useradd --uid 1000 --gid botuser --shell /bin/bash --create-home botuser

WORKDIR /app

# Copy the application source code.
COPY . .

# Ensure the non-root user owns the working directory.
RUN chown -R botuser:botuser /app

USER botuser

# Prometheus metrics endpoint.
EXPOSE 8000

# Simple health check using the metrics port.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

CMD ["python", "-m", "crypto_bot.main"]
