from __future__ import annotations

import json
from typing import Any

from litellm import completion
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    RateLimitError,
    Timeout,
)

from tokenweir.settings import ConfiguredProvider


class ModelCallError(RuntimeError):
    """Raised when a model call cannot produce usable content."""


class ModelResponseError(ModelCallError):
    """Raised when a provider response has no usable content."""


class JsonObjectError(ValueError):
    """Raised when a model response does not contain a JSON object."""


def call_json_model(
    provider: ConfiguredProvider,
    *,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    role: str = "model",
) -> str:
    """Call one configured provider and return its text content."""
    request: dict[str, Any] = {
        "model": provider.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": temperature,
        "timeout": provider.timeout_s,
        "num_retries": 0,
        "drop_params": True,
    }
    if provider.api_key is not None:
        request["api_key"] = provider.api_key.get_secret_value()
    if provider.base_url:
        request["api_base"] = provider.base_url

    try:
        response = completion(**request)
    except AuthenticationError as exc:
        raise ModelCallError(
            f"{role} authentication failed; verify the configured API key"
        ) from exc
    except RateLimitError as exc:
        raise ModelCallError(f"{role} provider rate-limited the request") from exc
    except Timeout as exc:
        raise ModelCallError(f"{role} call timed out") from exc
    except APIConnectionError as exc:
        raise ModelCallError(f"could not connect to the {role} provider") from exc
    except APIError as exc:
        raise ModelCallError(
            f"{role} provider returned an API error ({type(exc).__name__})"
        ) from exc
    except Exception as exc:
        raise ModelCallError(f"{role} call failed ({type(exc).__name__})") from exc

    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise ModelResponseError(
            f"{role} returned an unexpected response shape"
        ) from exc
    if not isinstance(content, str) or not content.strip():
        raise ModelResponseError(f"{role} returned an empty response")
    return content


def extract_json_object(raw: str) -> dict[str, Any]:
    """Extract a JSON object from a model response, tolerating fences and prose."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    data: Any
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise JsonObjectError("response was not valid JSON") from None
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            raise JsonObjectError("response was not valid JSON") from exc

    if not isinstance(data, dict):
        raise JsonObjectError("response must be a JSON object")
    return data
