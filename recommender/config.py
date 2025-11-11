"""Minimal settings loader with optional Pydantic support."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

try:  # pragma: no cover - exercised indirectly in production
    from pydantic import Field
    from pydantic_settings import BaseSettings  # type: ignore
except Exception:  # pragma: no cover - fallback for local/unit tests
    BaseSettings = None  # type: ignore
    Field = None  # type: ignore


if BaseSettings:

    class Settings(BaseSettings):
        kafka_bootstrap: str = Field(default="", env="KAFKA_BOOTSTRAP")
        kafka_security_protocol: str = Field("SASL_SSL", env="KAFKA_SECURITY_PROTOCOL")
        kafka_sasl_mechanism: str = Field("PLAIN", env="KAFKA_SASL_MECHANISM")
        kafka_api_key: str | None = Field(default=None, env="KAFKA_API_KEY")
        kafka_api_secret: str | None = Field(default=None, env="KAFKA_API_SECRET")
        reco_requests_topic: str = Field("aerosparks.requests", env="REQUESTS_TOPIC")
        reco_responses_topic: str = Field("aerosparks.reco_responses", env="RESPONSES_TOPIC")
        group_id: str = Field("reco-api", env="GROUP_ID")

        schema_registry_url: str | None = Field(default=None, env="SCHEMA_REGISTRY_URL")
        schema_registry_key: str | None = Field(default=None, env="SCHEMA_REGISTRY_KEY")
        schema_registry_secret: str | None = Field(default=None, env="SCHEMA_REGISTRY_SECRET")

        model_path: str = Field("models/reco.joblib", env="MODEL_PATH")

        service_name: str = Field("reco-api", env="SERVICE_NAME")
        env: str = Field("dev", env="APP_ENV")

        class Config:
            case_sensitive = False

else:

    @dataclass
    class Settings:  # pragma: no cover - simple fallback for unit tests
        kafka_bootstrap: str = field(default_factory=lambda: os.getenv("KAFKA_BOOTSTRAP", ""))
        kafka_security_protocol: str = field(default_factory=lambda: os.getenv("KAFKA_SECURITY_PROTOCOL", "SASL_SSL"))
        kafka_sasl_mechanism: str = field(default_factory=lambda: os.getenv("KAFKA_SASL_MECHANISM", "PLAIN"))
        kafka_api_key: str | None = field(default_factory=lambda: os.getenv("KAFKA_API_KEY"))
        kafka_api_secret: str | None = field(default_factory=lambda: os.getenv("KAFKA_API_SECRET"))
        reco_requests_topic: str = field(default_factory=lambda: os.getenv("REQUESTS_TOPIC", "aerosparks.requests"))
        reco_responses_topic: str = field(
            default_factory=lambda: os.getenv("RESPONSES_TOPIC", "aerosparks.reco_responses")
        )
        group_id: str = field(default_factory=lambda: os.getenv("GROUP_ID", "reco-api"))

        schema_registry_url: str | None = field(default_factory=lambda: os.getenv("SCHEMA_REGISTRY_URL"))
        schema_registry_key: str | None = field(default_factory=lambda: os.getenv("SCHEMA_REGISTRY_KEY"))
        schema_registry_secret: str | None = field(default_factory=lambda: os.getenv("SCHEMA_REGISTRY_SECRET"))

        model_path: str = field(default_factory=lambda: os.getenv("MODEL_PATH", "models/reco.joblib"))

        service_name: str = field(default_factory=lambda: os.getenv("SERVICE_NAME", "reco-api"))
        env: str = field(default_factory=lambda: os.getenv("APP_ENV", "dev"))


settings = Settings()
