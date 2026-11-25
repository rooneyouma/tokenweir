from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import litellm
from litellm import completion
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    RateLimitError,
    Timeout,
)

from tokenweir.budget import BudgetError, BudgetPolicy, LedgerEntry
from tokenweir.settings import ConfiguredProvider

_MIN_OUTPUT_TOKENS = 64
_SAFETY_FACTOR = 1.05
_MIN_CALL_COST_USD = 0.0005


class ModelCallError(RuntimeError):
    """Raised when a model call cannot produce usable content."""


class ModelResponseError(ModelCallError):
    """Raised when a provider response has no usable content."""


class JsonObjectError(ValueError):
    """Raised when a model response does not contain a JSON object."""


@dataclass
class ModelCallResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    cost_estimated: bool
    free: bool


def _count_prompt_tokens(
    provider: ConfiguredProvider, messages: list[dict[str, str]]
) -> int:
    try:
        return max(
            1, int(litellm.token_counter(model=provider.model, messages=messages))
        )
    except Exception:
        return max(1, len(json.dumps(messages)) // 4)


def _pricing(provider: ConfiguredProvider) -> tuple[float, float]:
    if (
        provider.input_cost_per_million is not None
        and provider.output_cost_per_million is not None
    ):
        return (
            provider.input_cost_per_million / 1_000_000,
            provider.output_cost_per_million / 1_000_000,
        )
    try:
        info = litellm.get_model_info(provider.model)
    except Exception as exc:
        raise BudgetError(
            f"cannot price paid model '{provider.model}' for provider "
            f"'{provider.name}'; set input_cost_per_million and "
            "output_cost_per_million in registry.json"
        ) from exc
    input_price = float(info.get("input_cost_per_token") or 0.0)
    output_price = float(info.get("output_cost_per_token") or 0.0)
    if input_price <= 0 and output_price <= 0:
        raise BudgetError(
            f"model '{provider.model}' has no pricing; mark the provider free "
            "or configure input/output costs in registry.json"
        )
    return input_price, output_price


def _estimate_cost(
    prompt_tokens: int,
    max_output_tokens: int,
    input_price: float,
    output_price: float,
) -> float:
    return (
        (prompt_tokens * input_price + max_output_tokens * output_price)
        * _SAFETY_FACTOR
        + _MIN_CALL_COST_USD
    )


def _affordable_output_tokens(
    remaining_usd: float,
    prompt_cost: float,
    output_price: float,
    limit: int,
) -> int:
    if output_price <= 0:
        return limit if prompt_cost <= remaining_usd else 0
    available = (
        (remaining_usd / _SAFETY_FACTOR) - _MIN_CALL_COST_USD - prompt_cost
    )
    if available <= 0:
        return 0
    return min(limit, int(available / output_price))


def _prepare_request(
    request: dict[str, Any],
    provider: ConfiguredProvider,
    budget_policy: BudgetPolicy | None,
    role: str,
) -> float:
    request["max_tokens"] = provider.max_output_tokens
    if provider.free or budget_policy is None:
        return 0.0

    input_price, output_price = _pricing(provider)
    prompt_tokens = _count_prompt_tokens(provider, request["messages"])
    prompt_cost = prompt_tokens * input_price

    def estimate(max_output: int) -> float:
        return _estimate_cost(
            prompt_tokens, max_output, input_price, output_price
        )

    max_output = _affordable_output_tokens(
        budget_policy.budget.remaining_usd,
        prompt_cost,
        output_price,
        provider.max_output_tokens,
    )
    if max_output < _MIN_OUTPUT_TOKENS:
        budget_policy.ensure(
            needed_usd=estimate(_MIN_OUTPUT_TOKENS),
            reason=f"paid {role} call",
        )
        max_output = _affordable_output_tokens(
            budget_policy.budget.remaining_usd,
            prompt_cost,
            output_price,
            provider.max_output_tokens,
        )
        if max_output < _MIN_OUTPUT_TOKENS:
            raise BudgetError(
                f"paid {role} call cannot fit within the remaining budget"
            )
    else:
        budget_policy.ensure(
            needed_usd=estimate(max_output),
            reason=f"paid {role} call",
        )

    request["max_tokens"] = max_output
    return estimate(max_output)


def _usage(response: object) -> tuple[int, int, int]:
    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    return prompt_tokens, completion_tokens, total_tokens


def _actual_cost(
    response: object,
    provider: ConfiguredProvider,
    prompt_tokens: int,
    completion_tokens: int,
    estimated_cost: float,
) -> tuple[float, bool]:
    if provider.free:
        return 0.0, False
    try:
        cost = float(litellm.completion_cost(completion_response=response))
        if cost > 0:
            return cost, False
    except Exception:
        pass
    try:
        input_price, output_price = _pricing(provider)
        return (
            prompt_tokens * input_price + completion_tokens * output_price,
            False,
        )
    except BudgetError:
        pass
    if estimated_cost > 0:
        return estimated_cost, True
    return 0.0, False


def _record_call(
    budget_policy: BudgetPolicy | None,
    provider: ConfiguredProvider,
    *,
    role: str,
    task: str,
    subtask: str | None,
    attempt: int,
    run_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    cost_usd: float,
    cost_estimated: bool,
) -> None:
    if budget_policy is None:
        return
    budget_policy.budget.record(
        LedgerEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            project=budget_policy.budget.project,
            run_id=run_id,
            role=role,
            provider=provider.name,
            model=provider.model,
            task=task,
            subtask=subtask,
            attempt=attempt,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            cost_estimated=cost_estimated,
            free=provider.free,
        )
    )


def call_json_model(
    provider: ConfiguredProvider,
    *,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    role: str = "model",
    budget_policy: BudgetPolicy | None = None,
    task: str = "",
    subtask: str | None = None,
    attempt: int = 1,
    run_id: str = "",
) -> ModelCallResult:
    """Call one configured provider and return content plus usage."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    request: dict[str, Any] = {
        "model": provider.model,
        "messages": messages,
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

    estimated_cost = _prepare_request(request, provider, budget_policy, role)

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

    prompt_tokens, completion_tokens, total_tokens = _usage(response)
    cost_usd, cost_estimated = _actual_cost(
        response,
        provider,
        prompt_tokens,
        completion_tokens,
        estimated_cost,
    )

    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        _record_call(
            budget_policy,
            provider,
            role=role,
            task=task,
            subtask=subtask,
            attempt=attempt,
            run_id=run_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            cost_estimated=cost_estimated,
        )
        raise ModelResponseError(
            f"{role} returned an unexpected response shape"
        ) from exc

    if not isinstance(content, str) or not content.strip():
        _record_call(
            budget_policy,
            provider,
            role=role,
            task=task,
            subtask=subtask,
            attempt=attempt,
            run_id=run_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            cost_estimated=cost_estimated,
        )
        raise ModelResponseError(f"{role} returned an empty response")

    _record_call(
        budget_policy,
        provider,
        role=role,
        task=task,
        subtask=subtask,
        attempt=attempt,
        run_id=run_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        cost_estimated=cost_estimated,
    )

    return ModelCallResult(
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        cost_estimated=cost_estimated,
        free=provider.free,
    )


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
