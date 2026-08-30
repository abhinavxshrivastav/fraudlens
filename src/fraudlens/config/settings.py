"""Application configuration.

Twelve-factor: every value is overridable by environment variable (prefix
``FRAUDLENS_``) or a local ``.env`` file, and nothing is required to boot.
Defaults are chosen so that ``uvicorn fraudlens.api.app:app`` works on a clean
checkout with no infrastructure running.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Profile(StrEnum):
    """Which infrastructure implementations to bind at startup.

    ``NATIVE`` keeps everything in-process so the platform runs on a laptop with
    no Docker daemon. ``DOCKER`` swaps in Kafka and Redis via ``docker-compose``.
    The application code never branches on this directly -- it is read once, in
    the dependency-injection layer, to choose an implementation of the
    ``StreamBroker`` and ``FeatureStore`` protocols.
    """

    NATIVE = "native"
    DOCKER = "docker"


class Environment(StrEnum):
    LOCAL = "local"
    DOCKER = "docker"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Root configuration object. Access via :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_prefix="FRAUDLENS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # -- runtime -----------------------------------------------------------
    env: Environment = Environment.LOCAL
    profile: Profile = Profile.NATIVE
    log_level: str = "INFO"
    log_json: bool = False

    # -- paths -------------------------------------------------------------
    data_dir: Path = Path("data")
    artifact_dir: Path = Path("artifacts")

    # -- serving -----------------------------------------------------------
    api_host: str = "0.0.0.0"  # noqa: S104 - binding all interfaces is intended in a container
    api_port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    # -- infrastructure (only read when profile is DOCKER) -----------------
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic: str = "transactions"
    redis_url: str = "redis://localhost:6379/0"

    # -- demo --------------------------------------------------------------
    demo_mode: bool = False
    replay_speed: float = 60.0

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_csv_origins(cls, value: object) -> object:
        """Allow ``FRAUDLENS_CORS_ORIGINS=a,b`` as well as a JSON list."""
        if isinstance(value, str) and not value.strip().startswith("["):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        level = value.upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if level not in allowed:
            msg = f"log_level must be one of {sorted(allowed)}, got {value!r}"
            raise ValueError(msg)
        return level

    # -- derived paths -----------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        """Where the downloaded Kaggle CSVs live. Git-ignored."""
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        """Parquet outputs of the ingestion + split stage. Git-ignored."""
        return self.data_dir / "processed"

    @property
    def sample_dir(self) -> Path:
        """Small committed sample so tests and CI run without the download."""
        return self.data_dir / "sample"

    @property
    def model_dir(self) -> Path:
        return self.artifact_dir / "models"

    @property
    def report_dir(self) -> Path:
        return self.artifact_dir / "reports"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so that importing configuration is free at call sites. Tests that
    need to vary configuration should call ``get_settings.cache_clear()``.
    """
    return Settings()
