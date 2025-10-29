FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY recommender/ recommender/
COPY model_registry/ model_registry/
CMD ["python", "-m", "recommender.train"]