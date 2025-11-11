FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential gcc g++ cmake libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.trainer.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.trainer.txt

COPY trainer ./trainer

CMD ["python", "trainer/train_and_publish.py"]
