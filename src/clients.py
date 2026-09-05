"""Cached factories for external service clients."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import streamlit as st
from google import genai
from google.genai import types
from qdrant_client import QdrantClient

from src.config import AppSettings, ConfigurationError, StorageMode


def _close_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()


def _fingerprint(secret: str | None) -> str:
    if not secret:
        return "none"
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


@st.cache_resource(scope="session", on_release=_close_client)
def _cached_gemini_client(
    credential_fingerprint: str,
    timeout_ms: int,
    _api_key: str,
) -> genai.Client:
    del credential_fingerprint
    return genai.Client(
        api_key=_api_key,
        http_options=types.HttpOptions(timeout=timeout_ms),
    )


def get_gemini_client(api_key: str, timeout_ms: int) -> genai.Client:
    """Build or reuse a Gemini client without placing the key in the cache key."""

    normalized_key = api_key.strip()
    if not normalized_key:
        raise ConfigurationError("Hãy nhập Gemini API key để sử dụng tính năng AI.")
    return _cached_gemini_client(
        _fingerprint(normalized_key),
        timeout_ms,
        _api_key=normalized_key,
    )


@st.cache_resource(on_release=_close_client)
def _cached_local_qdrant_client(location: str) -> QdrantClient:
    """Return one process-wide local client to avoid duplicate folder locks."""

    path = Path(location)
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


@st.cache_resource(scope="session", on_release=_close_client)
def _cached_demo_qdrant_client(workspace_id: str) -> QdrantClient:
    """Return an in-memory client isolated to the current browser workspace."""

    del workspace_id
    return QdrantClient(location=":memory:")


@st.cache_resource(scope="session", on_release=_close_client)
def _cached_cloud_qdrant_client(
    location: str,
    credential_fingerprint: str,
    timeout_seconds: int,
    _api_key: str,
) -> QdrantClient:
    """Return a session client while keeping cloud credentials out of cache keys."""

    del credential_fingerprint
    return QdrantClient(url=location, api_key=_api_key, timeout=timeout_seconds)


def get_qdrant_client(settings: AppSettings, workspace_id: str) -> QdrantClient:
    """Build the vector client for the configured isolation mode."""

    if settings.storage_mode is StorageMode.LOCAL:
        return _cached_local_qdrant_client(str(settings.qdrant_path))
    if settings.storage_mode is StorageMode.DEMO:
        return _cached_demo_qdrant_client(workspace_id)

    if not settings.qdrant_url or not settings.qdrant_api_key:
        raise ConfigurationError("Chế độ cloud yêu cầu QDRANT_URL và QDRANT_API_KEY.")
    return _cached_cloud_qdrant_client(
        settings.qdrant_url,
        _fingerprint(settings.qdrant_api_key),
        max(1, settings.request_timeout_ms // 1_000),
        _api_key=settings.qdrant_api_key,
    )
