"""Configuration for the local Db2 + llama.cpp RAG pipeline.

Everything the pipeline needs comes from environment variables (see .env.example).
The tutorial used cloud Db2 + watsonx.ai; this reads a local Db2 instance and two
OpenAI-compatible llama.cpp endpoints instead.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from haystack.utils import Secret

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")


def _env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True)
class Db2Settings:
    database: str
    hostname: str
    port: int
    username: str
    password: str
    table_name: str
    embedding_dim: int
    distance_metric: str

    def store_kwargs(self) -> dict:
        """Keyword arguments for IBMDb2DocumentStore.

        The store reads credentials as Secrets; we hand it explicit tokens so the
        pipeline works whether or not DB2_USERNAME/DB2_PASSWORD are exported.
        """
        return {
            "database": self.database,
            "hostname": self.hostname,
            "port": self.port,
            "username": Secret.from_token(self.username),
            "password": Secret.from_token(self.password),
            "table_name": self.table_name,
            "embedding_dim": self.embedding_dim,
            "distance_metric": self.distance_metric,
            "use_ssl": False,
        }


@dataclass(frozen=True)
class EndpointSettings:
    """An OpenAI-compatible llama.cpp endpoint."""

    base_url: str
    model: str

    @property
    def api_key(self) -> Secret:
        # llama.cpp ignores the key, but the OpenAI client requires a non-empty one.
        return Secret.from_token(os.getenv("LLAMACPP_API_KEY", "sk-no-key-required"))


def db2_settings() -> Db2Settings:
    return Db2Settings(
        database=_env("DB2_DATABASE", "SAMPLE"),
        hostname=_env("DB2_HOSTNAME", "localhost"),
        port=int(_env("DB2_PORT", "50000")),
        username=_env("DB2_USERNAME"),
        password=_env("DB2_PASSWORD"),
        table_name=_env("DB2_TABLE_NAME", "HAYSTACK_DOCUMENTS"),
        embedding_dim=int(_env("EMBED_DIM", "384")),
        distance_metric=_env("DISTANCE_METRIC", "COSINE"),
    )


def embed_endpoint() -> EndpointSettings:
    return EndpointSettings(
        base_url=_env("EMBED_BASE_URL", "http://127.0.0.1:8081/v1"),
        model=_env("EMBED_MODEL", "bge-small-en-v1.5"),
    )


def chat_endpoint() -> EndpointSettings:
    return EndpointSettings(
        base_url=_env("CHAT_BASE_URL", "http://127.0.0.1:8080/v1"),
        model=_env("CHAT_MODEL", "qwen2.5-3b-instruct"),
    )
