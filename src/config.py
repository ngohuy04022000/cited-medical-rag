"""Centralized configuration and logging for the Cited Medical RAG system.

All tunables live here and are read from environment variables (with sane
defaults) so the same code runs unchanged in the CLI, the unit tests, and the
FastAPI service. Modules receive a `Settings` instance instead of reading
`os.environ` directly — this keeps them decoupled and easy to test.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

_DEFAULT_DOCS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "knowledge_base"
)
_DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# Generation backends:
#   "anthropic" — call the Claude API (needs ANTHROPIC_API_KEY)
#   "local"     — offline extractive answers from retrieved context (no API key)
#   "hf"        — a small LLM downloaded to disk, run fully on this machine
#                 via Hugging Face Transformers (no API key, no network)
_LOCAL_ALIASES = {"local", "offline", "extractive", "mock"}
_HF_ALIASES = {"hf", "huggingface", "local-llm", "qwen"}

_DEFAULT_HF_MODEL_DIR = os.path.join(
    os.path.dirname(__file__), "..", "models", "Qwen2.5-1.5B-Instruct"
)


def _normalize_backend(raw: str) -> str:
    value = (raw or "anthropic").strip().lower()
    if value in _LOCAL_ALIASES:
        return "local"
    if value in _HF_ALIASES:
        return "hf"
    if value != "anthropic":
        logging.getLogger("cmrag.config").warning(
            "Unrecognized RAG_BACKEND=%r, falling back to 'anthropic'. "
            "Valid values: 'anthropic', %s (offline extractive), or %s (local LLM).",
            raw,
            sorted(_LOCAL_ALIASES),
            sorted(_HF_ALIASES),
        )
    return "anthropic"


def _get_int(name: str, default: int, min_value: Optional[int] = None) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    log = logging.getLogger("cmrag.config")
    try:
        value = int(raw)
    except ValueError:
        log.warning("Invalid int for %s=%r, using default %s", name, raw, default)
        return default
    if min_value is not None and value < min_value:
        log.warning(
            "%s=%s is below the minimum %s, clamping to %s", name, value, min_value, min_value
        )
        return min_value
    return value


def _get_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logging.getLogger("cmrag.config").warning(
            "Invalid float for %s=%r, using default %s", name, raw, default
        )
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of runtime configuration."""

    anthropic_api_key: Optional[str]
    backend: str
    model: str
    max_tokens: int
    temperature: float
    request_timeout: float
    max_retries: int
    docs_dir: str
    top_k: int
    max_per_source: int
    min_score: float
    local_min_confidence: float
    local_min_term_coverage: float
    chunk_size: int
    log_level: str
    hf_model_dir: str
    hf_max_new_tokens: int


def get_settings() -> Settings:
    """Build a Settings snapshot from the current environment."""
    return Settings(
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        backend=_normalize_backend(os.environ.get("RAG_BACKEND", "anthropic")),
        model=os.environ.get("ANTHROPIC_MODEL", _DEFAULT_MODEL),
        max_tokens=_get_int("RAG_MAX_TOKENS", 600),
        temperature=_get_float("RAG_TEMPERATURE", 0.0),
        request_timeout=_get_float("RAG_REQUEST_TIMEOUT", 30.0),
        max_retries=_get_int("RAG_MAX_RETRIES", 3),
        docs_dir=os.environ.get("RAG_DOCS_DIR", _DEFAULT_DOCS_DIR),
        top_k=_get_int("RAG_TOP_K", 5, min_value=1),
        max_per_source=_get_int("RAG_MAX_PER_SOURCE", 2, min_value=1),
        min_score=_get_float("RAG_MIN_SCORE", 0.01),
        local_min_confidence=_get_float("RAG_LOCAL_MIN_CONFIDENCE", 0.12),
        local_min_term_coverage=_get_float("RAG_LOCAL_MIN_TERM_COVERAGE", 0.75),
        chunk_size=_get_int("RAG_CHUNK_SIZE", 700, min_value=50),
        log_level=os.environ.get("RAG_LOG_LEVEL", "INFO"),
        hf_model_dir=os.environ.get("RAG_HF_MODEL_DIR", _DEFAULT_HF_MODEL_DIR),
        hf_max_new_tokens=_get_int("RAG_HF_MAX_NEW_TOKENS", 400, min_value=32),
    )


_logging_configured = False


def setup_logging(level: Optional[str] = None) -> None:
    """Configure root logging once. Safe to call multiple times."""
    global _logging_configured
    if _logging_configured:
        return
    resolved = (level or os.environ.get("RAG_LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, resolved, logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    _logging_configured = True
