from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from tokenweir.budget import BudgetPolicy
from tokenweir.llm import (
    JsonObjectError,
    ModelCallError,
    ModelProviderUnavailableError,
    ModelResponseError,
    call_json_model,
    extract_json_object,
)
from tokenweir.schemas import BuilderOutput
from tokenweir.settings import ConfiguredProvider


class BuilderError(RuntimeError):
    """Raised when the builder call cannot produce a usable response."""


class BuilderResponseError(BuilderError):
    """Raised when the builder response is malformed or violates the contract."""


class BuilderProviderUnavailableError(BuilderError):
    """Raised when the builder provider is rate-limited or out of quota."""


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
    try:
        data = extract_json_object(raw)
    except JsonObjectError as exc:
        raise BuilderResponseError(f"builder {exc}") from exc

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
    budget_policy: BudgetPolicy | None = None,
    subtask: str | None = None,
    attempt: int = 1,
    run_id: str = "",
) -> BuilderOutput:
    """Ask one configured model for a complete set of files."""
    user_message = f"Task:\n{task}\n"
    if feedback:
        user_message += (
            "\nThe previous attempt failed. Fix the problem below and return "
            f"every required file again.\n\n{feedback}\n"
        )
    user_message += "\nReturn only the JSON object described by the system prompt."

    try:
        result = call_json_model(
            provider,
            system_prompt=prompt,
            user_prompt=user_message,
            temperature=temperature,
            role="builder",
            budget_policy=budget_policy,
            task=task,
            subtask=subtask,
            attempt=attempt,
            run_id=run_id,
        )
    except ModelResponseError as exc:
        raise BuilderResponseError(str(exc)) from exc
    except ModelProviderUnavailableError as exc:
        raise BuilderProviderUnavailableError(str(exc)) from exc
    except ModelCallError as exc:
        raise BuilderError(str(exc)) from exc

    return parse_builder_output(result.content)
