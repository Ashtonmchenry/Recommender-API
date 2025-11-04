# Python runtime
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps needed by confluent-kafka/fastavro (and typical builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ librdkafka-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for better layer caching
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy the app
COPY . .

# Cloud Run expects the server to listen on $PORT (default 8080)
ENV PORT=8080
CMD ["sh", "-c", "uvicorn service.app:app --host 0.0.0.0 --port ${PORT:-8080}"]


