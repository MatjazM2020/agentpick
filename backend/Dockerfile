# Multi-stage build for FastAPI backend
# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY backend/requirements.txt .

# CPU-only PyTorch (~80 MB vs ~426 MB CUDA wheel on aarch64). Install before
# sentence-transformers so pip does not pull the default CUDA build from PyPI.
RUN pip install --no-cache-dir --default-timeout=300 --retries 5 \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies globally (not to user directory)
RUN pip install --no-cache-dir --default-timeout=300 --retries 5 -r requirements.txt


# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies (keep minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local /usr/local

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PORT=5000 \
    HOST=0.0.0.0 \
    ENV=production

# Copy application code
COPY backend/ /app/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Expose port
EXPOSE 5000

# Run FastAPI app with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
