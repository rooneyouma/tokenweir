from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from tokenweir.budget import BudgetPolicy
from tokenweir.llm import (
    JsonObjectError,
    ModelCallError,
    ModelResponseError,
    call_json_model,
    extract_json_object,
)
from tokenweir.schemas import Plan
from tokenweir.settings import ConfiguredProvider


class ArchitectError(RuntimeError):
    """Raised when the architect call cannot produce a usable plan."""


class ArchitectResponseError(ArchitectError):
    """Raised when the architect response is malformed or violates the plan contract."""


def _read_prompt_file(path: Path, label: str) -> str:
    try:
        content = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ArchitectError(f"cannot read {label} {path}: {exc}") from exc
    if not content:
        raise ArchitectError(f"{label} is empty: {path}")
    return content


def load_architect_prompt() -> str:
    return _read_prompt_file(
        Path(__file__).with_name("prompts") / "architect.md",
        "architect prompt",
    )


def load_project_brief() -> str:
    return _read_prompt_file(Path(__file__).with_name("AGENTS.md"), "project brief")


def parse_plan(raw: str) -> Plan:
    """Parse JSON from the architect and validate the complete plan."""
    try:
        data = extract_json_object(raw)
    except JsonObjectError as exc:
        raise ArchitectResponseError(f"architect {exc}") from exc

    try:
        return Plan.model_validate(data)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors(include_input=False)
        )
        raise ArchitectResponseError(
            f"architect plan did not match the contract: {details}"
        ) from exc


def plan_task(
    task: str,
    provider: ConfiguredProvider,
    *,
    prompt: str,
    brief: str,
    temperature: float = 0.0,
    budget_policy: BudgetPolicy | None = None,
    run_id: str = "",
) -> Plan:
    """Make exactly one planning call and return a validated plan."""
    user_message = (
        f"Project brief:\n{brief}\n\n"
        f"Task:\n{task}\n\n"
        "Return only the JSON plan described by the system prompt."
    )
    try:
        result = call_json_model(
            provider,
            system_prompt=prompt,
            user_prompt=user_message,
            temperature=temperature,
            role="architect",
            budget_policy=budget_policy,
            task=task,
            attempt=1,
            run_id=run_id,
        )
    except ModelResponseError as exc:
        raise ArchitectResponseError(str(exc)) from exc
    except ModelCallError as exc:
        raise ArchitectError(str(exc)) from exc

    return parse_plan(result.content)
