from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from litellm import completion
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    RateLimitError,
    Timeout,
)
from pydantic import ValidationError

from tokenweir.schemas import BuilderOutput
from tokenweir.settings import ConfiguredProvider


class BuilderError(RuntimeError):
    """Raised when the builder call cannot produce a usable response."""


class BuilderResponseError(BuilderError):
    """Raised when the builder response is malformed or violates the contract."""


def load_builder_prompt() -> str:
    prompt_path = Path(__file__).with_name("prompts") / "builder.md"
    try:
        prompt = prompt_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise BuilderError(f"cannot read builder prompt {prompt_path}: {exc}") from exc
    if not prompt:
        raise BuilderError(f"builder prompt is empty: {prompt_path}")
    return prompt


def parse_builder_output(raw: str) -> BuilderOutput:
    """Parse JSON from a model response and validate the file contract."""
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
            raise BuilderResponseError("builder response was not valid JSON") from None
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            raise BuilderResponseError("builder response was not valid JSON") from exc

    if not isinstance(data, dict):
        raise BuilderResponseError("builder response must be a JSON object")

    try:
        return BuilderOutput.model_validate(data)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors(include_input=False)
        )
        raise BuilderResponseError(
            f"builder response did not match the file contract: {details}"
        ) from exc


def build_files(
    task: str,
    provider: ConfiguredProvider,
    *,
    prompt: str,
    feedback: str | None = None,
    temperature: float = 0.0,
) -> BuilderOutput:
    """Ask one configured model for a complete set of files."""
    user_message = f"Task:\n{task}\n"
    if feedback:
        user_message += (
            "\nThe previous attempt failed. Fix the problem below and return "
            f"every required file again.\n\n{feedback}\n"
        )
    user_message += "\nReturn only the JSON object described by the system prompt."

    request: dict[str, Any] = {
        "model": provider.model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message},
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
        raise BuilderError(
            "builder authentication failed; verify the configured API key"
        ) from exc
    except RateLimitError as exc:
        raise BuilderError("builder provider rate-limited the request") from exc
    except Timeout as exc:
        raise BuilderError("builder call timed out") from exc
    except APIConnectionError as exc:
        raise BuilderError("could not connect to the builder provider") from exc
    except APIError as exc:
        raise BuilderError(
            f"builder provider returned an API error ({type(exc).__name__})"
        ) from exc
    except Exception as exc:
        raise BuilderError(f"builder call failed ({type(exc).__name__})") from exc

    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise BuilderResponseError(
            "builder returned an unexpected response shape"
        ) from exc
    if not isinstance(content, str) or not content.strip():
        raise BuilderResponseError("builder returned an empty response")

    return parse_builder_output(content)
