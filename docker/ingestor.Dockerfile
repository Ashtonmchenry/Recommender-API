FROM python:3.12-slim
WORKDIR /app
COPY stream /app/stream
CMD ["python", "-c", "print('stream ingestor placeholder')"]
