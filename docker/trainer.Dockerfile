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

# System deps for building wheels if needed
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential gcc \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY trainer/requirements.txt ./requirements.txt
RUN pip install --prefix=/install -r requirements.txt

# Copy only the trainer code
COPY trainer ./trainer

########################################
# Runtime stage
########################################
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Non-root user
RUN useradd -u 1000 -m appuser && chown -R appuser /app
USER appuser

# Copy installed packages + trainer source
COPY --from=builder /install /usr/local
COPY --from=builder /app/trainer ./trainer

# Adjust the module/entrypoint to match your trainer
# e.g. trainer.main:main or trainer.train:main etc.
CMD ["python", "-m", "trainer.main"]
