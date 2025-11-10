FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
COPY trainer /app/trainer
COPY recommender /app/recommender
COPY model_registry /app/model_registry
ENV PYTHONPATH=/app
CMD ["python", "trainer/train_and_publish.py"]
