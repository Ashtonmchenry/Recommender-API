# docker/service.Dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ---- OS deps for scipy/numpy/implicit (compiled) ----
# - build-essential, g++, gcc, make, cmake: toolchain
# - libopenblas-dev: BLAS/LAPACK
# - libgomp1: OpenMP runtime (implicit uses OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential g++ gcc make cmake \
    libopenblas-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---- Install Python deps first for better layer caching ----
COPY service/requirements.txt /tmp/service.requirements.txt
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r /tmp/service.requirements.txt

# ---- App code ----
COPY service/ /app/service
COPY recommender/ /app/recommender

# Let Python find your in-repo package "recommender"
ENV PYTHONPATH=/app

EXPOSE 8080
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8080"]
