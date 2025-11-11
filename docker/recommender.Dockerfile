# build stage
FROM python:3.12-slim AS build
WORKDIR /app
COPY service /app/service
RUN pip install --no-cache-dir --upgrade pip \
 && pip wheel --wheel-dir=/wheels fastapi uvicorn prometheus-client pydantic

# runtime stage
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels
COPY service /app/service
EXPOSE 8080
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8080"]
