"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the Foresight API.

    Values are read from environment variables and fall back to the
    defaults declared here.  A ``.env`` file in the project root is
    loaded automatically.
    """

    ANTHROPIC_API_KEY: str

    PLAID_CLIENT_ID: str = ""
    PLAID_SECRET: str = ""
    PLAID_ENV: str = "sandbox"

    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""

    NEO4J_URI: str = "bolt://neo4j:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"

    POSTGRES_URL: str = "postgresql://postgres:password@postgres:5432/foresight"
    REDIS_URL: str = "redis://redis:6379"
    QDRANT_URL: str = "http://qdrant:6333"

    APP_ENV: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance.

    Using ``lru_cache`` ensures environment variables are read only once
    and the same object is reused for the lifetime of the process.
    """
    return Settings()
