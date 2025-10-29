from __future__ import annotations
from pydantic import BaseSettings, Field, AnyHttpUrl
from typing import Optional

class Settings(BaseSettings):
    # Kafka / Confluent
    kafka_bootstrap: str = Field(..., env="KAFKA_BOOTSTRAP")
    kafka_security_protocol: str = Field("SASL_SSL", env="KAFKA_SECURITY_PROTOCOL")
    kafka_sasl_mechanism: str = Field("PLAIN", env="KAFKA_SASL_MECHANISM")
    kafka_api_key: Optional[str] = Field(default=None, env="KAFKA_API_KEY")
    kafka_api_secret: Optional[str] = Field(default=None, env="KAFKA_API_SECRET")
    reco_requests_topic: str = Field("aerosparks.requests", env="REQUESTS_TOPIC")
    reco_responses_topic: str = Field("aerosparks.reco_responses", env="RESPONSES_TOPIC")
    group_id: str = Field("reco-api", env="GROUP_ID")

    # Schema Registry
    schema_registry_url: Optional[str] = Field(default=None, env="SCHEMA_REGISTRY_URL")
    schema_registry_key: Optional[str] = Field(default=None, env="SCHEMA_REGISTRY_KEY")
    schema_registry_secret: Optional[str] = Field(default=None, env="SCHEMA_REGISTRY_SECRET")

    # Model
    model_path: str = Field("models/reco.joblib", env="MODEL_PATH")

    # Service
    service_name: str = Field("reco-api", env="SERVICE_NAME")
    env: str = Field("dev", env="APP_ENV")

    class Config:
        case_sensitive = False

settings = Settings()
