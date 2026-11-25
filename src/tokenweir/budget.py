from __future__ import annotations

import json
import math
import re
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_CAP_USD = 2.0
DEFAULT_WARN_FRACTION = 0.8


class BudgetError(RuntimeError):
    """Raised when project budget state cannot be loaded or enforced."""


class BudgetExceededError(BudgetError):
    """Raised when a paid call would exceed the project's cap."""

    def __init__(
        self,
        *,
        project: str,
        cap_usd: float,
        spent_usd: float,
        needed_usd: float | None = None,
        reason: str | None = None,
    ) -> None:
        self.project = project
        self.cap_usd = cap_usd
        self.spent_usd = spent_usd
        self.needed_usd = needed_usd
        self.reason = reason
        remaining = max(0.0, cap_usd - spent_usd)
        message = (
            f"project '{project}' has reached its ${cap_usd:.2f} cap "
            f"(${spent_usd:.2f} spent, ${remaining:.2f} remaining)"
        )
        if needed_usd is not None:
            message += f"; this call needs about ${needed_usd:.4f}"
        if reason:
            message += f" ({reason})"
        super().__init__(message)


class LedgerEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: str
    project: str
    run_id: str
    role: str
    provider: str
    model: str
    task: str = ""
    subtask: str | None = None
    attempt: int = Field(ge=1)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    cost_estimated: bool = False
    free: bool = False


def project_slug(name: str) -> str:
    """Return a safe directory name for a project."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", name.strip()).strip("-._")
    if not slug:
        raise BudgetError("project name must contain a letter or digit")
    return slug[:80]


def _read_spent(ledger_path: Path) -> float:
    if not ledger_path.exists():
        return 0.0
    total = 0.0
    try:
        lines = ledger_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise BudgetError(f"cannot read project ledger {ledger_path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            cost = float(entry["cost_usd"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise BudgetError(
                f"invalid ledger entry at {ledger_path}:{line_number}"
            ) from exc
        if not math.isfinite(cost) or cost < 0:
            raise BudgetError(
                f"invalid ledger cost at {ledger_path}:{line_number}"
            )
        total += cost
    return round(total, 8)


class ProjectBudget:
    """Dynamic paid-spend cap backed by a per-project JSONL ledger."""

    def __init__(
        self,
        *,
        project: str,
        cap_usd: float,
        spent_usd: float,
        state_dir: Path,
        warn_fraction: float = DEFAULT_WARN_FRACTION,
    ) -> None:
        self.project = project
        self.cap_usd = cap_usd
        self.spent_usd = spent_usd
        self.state_dir = state_dir
        self.warn_fraction = warn_fraction
        self.project_dir = state_dir / project_slug(project)
        self.budget_path = self.project_dir / "budget.json"
        self.ledger_path = self.project_dir / "ledger.jsonl"
        self._warned = False
        self._over_warned = False

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.cap_usd - self.spent_usd)

    def ensure_can_spend(
        self, *, needed_usd: float = 0.0, reason: str = ""
    ) -> None:
        if not math.isfinite(needed_usd) or needed_usd < 0:
            raise BudgetError(
                "needed cost must be a finite non-negative number"
            )
        if self.spent_usd >= self.cap_usd or needed_usd > self.remaining_usd + 1e-9:
            raise BudgetExceededError(
                project=self.project,
                cap_usd=self.cap_usd,
                spent_usd=self.spent_usd,
                needed_usd=needed_usd if needed_usd > 0 else None,
                reason=reason or None,
            )
        if (
            not self._warned
            and self.cap_usd > 0
            and self.spent_usd / self.cap_usd >= self.warn_fraction
        ):
            print(
                f"tokenweir: project '{self.project}' has used "
                f"${self.spent_usd:.2f} of its ${self.cap_usd:.2f} cap.",
                file=sys.stderr,
            )
            self._warned = True

    def record(self, entry: LedgerEntry) -> None:
        if entry.project != self.project:
            raise BudgetError(
                f"ledger entry project {entry.project!r} does not match "
                f"{self.project!r}"
            )
        try:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger_path.open("a", encoding="utf-8") as handle:
                handle.write(entry.model_dump_json() + "\n")
        except OSError as exc:
            raise BudgetError(f"cannot append to project ledger: {exc}") from exc
        self.spent_usd = round(self.spent_usd + entry.cost_usd, 8)
        if self.spent_usd > self.cap_usd and not self._over_warned:
            print(
                f"tokenweir: project '{self.project}' is now over its "
                f"${self.cap_usd:.2f} cap at ${self.spent_usd:.2f}; "
                "further paid calls are blocked.",
                file=sys.stderr,
            )
            self._over_warned = True

    def extend(self, new_cap_usd: float) -> None:
        if not math.isfinite(new_cap_usd) or new_cap_usd < 0:
            raise BudgetError("cap must be a finite non-negative number")
        if new_cap_usd < self.spent_usd:
            raise BudgetError(
                f"new cap ${new_cap_usd:.2f} is below the "
                f"${self.spent_usd:.2f} already spent"
            )
        if new_cap_usd <= self.cap_usd:
            raise BudgetError("new cap must be greater than the current cap")
        self.cap_usd = new_cap_usd
        self._warned = False
        self._over_warned = False
        self._write_budget()

    def _write_budget(self) -> None:
        payload = {
            "project": self.project,
            "cap_usd": round(self.cap_usd, 8),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self.project_dir.mkdir(parents=True, exist_ok=True)
            self.budget_path.write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            raise BudgetError(f"cannot write project budget: {exc}") from exc


CapPrompt = Callable[[BudgetExceededError], float | None]


class BudgetPolicy:
    """Enforces a project cap and can ask for a new cap when it is depleted."""

    def __init__(
        self, budget: ProjectBudget, prompt: CapPrompt | None = None
    ) -> None:
        self.budget = budget
        self.prompt = prompt

    def ensure(self, *, needed_usd: float = 0.0, reason: str = "") -> None:
        while True:
            try:
                self.budget.ensure_can_spend(
                    needed_usd=needed_usd, reason=reason
                )
                return
            except BudgetExceededError as exc:
                if self.prompt is None:
                    raise
                new_cap = self.prompt(exc)
                if new_cap is None:
                    raise
                self.budget.extend(new_cap)


def load_project_budget(
    project: str,
    *,
    state_dir: Path = Path("projects"),
    cap_usd: float | None = None,
    default_cap_usd: float = DEFAULT_CAP_USD,
    warn_fraction: float = DEFAULT_WARN_FRACTION,
) -> ProjectBudget:
    """Load or create a project budget, optionally replacing its cap."""
    slug = project_slug(project)
    project_dir = state_dir / slug
    budget_path = project_dir / "budget.json"
    ledger_path = project_dir / "ledger.jsonl"
    spent = _read_spent(ledger_path)

    existing_cap: float | None = None
    if budget_path.exists():
        try:
            data = json.loads(budget_path.read_text(encoding="utf-8"))
            existing_cap = float(data["cap_usd"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise BudgetError(f"invalid project budget {budget_path}: {exc}") from exc
        if not math.isfinite(existing_cap) or existing_cap < 0:
            raise BudgetError(
                f"project cap must be a finite non-negative number: "
                f"{budget_path}"
            )

    if cap_usd is not None:
        if not math.isfinite(cap_usd) or cap_usd < 0:
            raise BudgetError("cap must be a finite non-negative number")
        selected_cap = cap_usd
    elif existing_cap is not None:
        selected_cap = existing_cap
    else:
        if not math.isfinite(default_cap_usd) or default_cap_usd < 0:
            raise BudgetError(
                "default cap must be a finite non-negative number"
            )
        selected_cap = default_cap_usd

    if not math.isfinite(warn_fraction) or not 0.0 <= warn_fraction <= 1.0:
        raise BudgetError("warn_fraction must be between 0 and 1")

    budget = ProjectBudget(
        project=project,
        cap_usd=selected_cap,
        spent_usd=spent,
        state_dir=state_dir,
        warn_fraction=warn_fraction,
    )
    if cap_usd is not None:
        budget._write_budget()
    return budget
