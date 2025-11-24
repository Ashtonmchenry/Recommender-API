# DEPRECATED

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

COPY recommender_api/requirements.txt ./requirements.txt
RUN pip install --prefix=/install -r requirements.txt

COPY recommender_api ./recommender_api

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
COPY --from=builder /app/recommender_api ./recommender_api

EXPOSE 8080

# If your module/path is different, tweak the module below.
CMD ["uvicorn", "recommender_api.main:app", "--host", "0.0.0.0", "--port", "8080"]
