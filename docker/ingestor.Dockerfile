FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
COPY stream /app/stream
COPY recommender /app/recommender
COPY infra /app/infra
ENV PYTHONPATH=/app
CMD ["python", "stream/consumer.py"]
