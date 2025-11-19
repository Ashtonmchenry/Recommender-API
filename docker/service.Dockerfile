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

COPY service/requirements.txt ./requirements.txt
RUN pip install --prefix=/install -r requirements.txt

COPY service ./service

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
COPY --from=builder /app/service ./service

EXPOSE 8080

# Adjust module path if your service package is named differently
CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8080"]
