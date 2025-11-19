# syntax=docker/dockerfile:1

########################################
# Builder stage
########################################
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential gcc \
 && rm -rf /var/lib/apt/lists/*

COPY ingestor/requirements.txt ./requirements.txt
RUN pip install --prefix=/install -r requirements.txt

COPY ingestor ./ingestor

########################################
# Runtime stage
########################################
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN useradd -u 1000 -m appuser && chown -R appuser /app
USER appuser

COPY --from=builder /install /usr/local
COPY --from=builder /app/ingestor ./ingestor

# Adjust to your ingestion entrypoint (module or script)
CMD ["python", "-m", "ingestor.main"]
