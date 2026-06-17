# GitHub Copilot Gateway — Docker Image
# Multi-stage build for a minimal production image.

FROM python:3.12-slim AS builder

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Copy application code
COPY main.py config.py auth.py models.py proxy.py convert.py ./

# Create a non-root user
RUN useradd --create-home --shell /bin/bash gateway && \
    chown -R gateway:gateway /app

USER gateway

EXPOSE 9992

ENV GATEWAY_PORT=9992
ENV GATEWAY_HOST=0.0.0.0

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9992/health')" || exit 1

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9992"]
