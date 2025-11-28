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
COPY recommender ./recommender
COPY model_registry ./model_registry

########################################
# Runtime stage
########################################
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install libgomp for implicit/ALS
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN useradd -u 1000 -m appuser && chown -R appuser /app
USER appuser

COPY --from=builder /install /usr/local
COPY --from=builder /app/service ./service
COPY --from=builder /app/recommender ./recommender
COPY --from=builder /app/model_registry ./model_registry

EXPOSE 8080

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8080"]
