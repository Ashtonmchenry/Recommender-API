# syntax=docker/dockerfile:1

# -------------------------------------------------------------
# Builder stage: install dependencies into a virtual environment
# -------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    librdkafka-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install -r requirements.txt

COPY . .

# -------------------------------------------------------------
# Runtime stage: slim image with only runtime dependencies
# -------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    librdkafka1 \
 && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:${PATH}" \
    PORT=8080

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/service ./service
COPY --from=builder /app/recommender ./recommender
COPY --from=builder /app/model_registry ./model_registry
COPY --from=builder /app/requirements.txt ./requirements.txt

CMD ["sh", "-c", "uvicorn service.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
