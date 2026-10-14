from __future__ import annotations

import base64
import os

from typing import Any, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised only when dependency is missing.
    OpenAI = None


def encode_image(image_content: bytes) -> str:
    return base64.b64encode(image_content).decode("utf-8")


def build_client(api_key: Optional[str] = None):
    if OpenAI is None:
        raise ImportError("openai package is required for the GPT backbone")
    return OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))


def model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [model_dump(item) for item in value]
    if isinstance(value, dict):
        return {key: model_dump(item) for key, item in value.items()}
    return value


def get_field(value: Any, field: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(field, default)
    return getattr(value, field, default)
