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

# Trainer-only dependencies
COPY trainer/requirements.trainer.txt .
RUN pip install --prefix=/install -r requirements.trainer.txt

# Copy code used by the trainer
COPY recommender ./recommender
COPY trainer ./trainer
COPY data ./data

########################################
# Runtime stage
########################################
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN useradd -u 1000 -m appuser && chown -R appuser /app
USER appuser

COPY --from=builder /install /usr/local
COPY --from=builder /app/recommender ./recommender
COPY --from=builder /app/trainer ./trainer
COPY --from=builder /app/data ./data

# Trainer entrypoint: run the retrain+publish script
CMD ["python", "trainer/train_and_publish.py"]
