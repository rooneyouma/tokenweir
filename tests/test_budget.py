import json
from pathlib import Path

import pytest

from tokenweir.budget import (
    BudgetError,
    BudgetExceededError,
    BudgetPolicy,
    LedgerEntry,
    ProjectBudget,
    load_project_budget,
    project_slug,
)


def _entry(
    *,
    project: str = "demo",
    cost_usd: float = 0.0,
    attempt: int = 1,
    free: bool = False,
) -> LedgerEntry:
    return LedgerEntry(
        timestamp="2026-01-01T00:00:00+00:00",
        project=project,
        run_id="run-1",
        role="builder",
        provider="provider",
        model="model",
        attempt=attempt,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        cost_usd=cost_usd,
        free=free,
    )


def test_project_slug_sanitizes_and_rejects_empty_names() -> None:
    assert project_slug("My Project!") == "My-Project"
    assert project_slug("  a/b  ") == "a-b"

    with pytest.raises(BudgetError, match="letter or digit"):
        project_slug("---")


def test_load_project_budget_uses_default_without_writing(
    tmp_path: Path,
) -> None:
    budget = load_project_budget(
        "demo", state_dir=tmp_path / "projects", default_cap_usd=2.5
    )

    assert budget.cap_usd == 2.5
    assert budget.spent_usd == 0.0
    assert not budget.budget_path.exists()


def test_load_project_budget_writes_explicit_cap_and_reads_it_back(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "projects"
    budget = load_project_budget(
        "demo", state_dir=state_dir, cap_usd=3.0
    )

    assert budget.budget_path.exists()
    payload = json.loads(budget.budget_path.read_text(encoding="utf-8"))
    assert payload["project"] == "demo"
    assert payload["cap_usd"] == 3.0

    reloaded = load_project_budget("demo", state_dir=state_dir)
    assert reloaded.cap_usd == 3.0


def test_load_project_budget_reads_existing_ledger_spend(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "projects"
    budget = load_project_budget("demo", state_dir=state_dir, cap_usd=1.0)
    budget.record(_entry(cost_usd=0.25))
    budget.record(_entry(cost_usd=0.5))

    reloaded = load_project_budget("demo", state_dir=state_dir)

    assert reloaded.spent_usd == pytest.approx(0.75)
    assert reloaded.remaining_usd == pytest.approx(0.25)


def test_load_project_budget_rejects_invalid_budget_file(
    tmp_path: Path,
) -> None:
    budget_path = tmp_path / "projects" / "demo" / "budget.json"
    budget_path.parent.mkdir(parents=True)
    budget_path.write_text("{not json", encoding="utf-8")

    with pytest.raises(BudgetError, match="invalid project budget"):
        load_project_budget("demo", state_dir=tmp_path / "projects")


def test_load_project_budget_rejects_negative_cap(tmp_path: Path) -> None:
    with pytest.raises(BudgetError, match="finite non-negative"):
        load_project_budget(
            "demo", state_dir=tmp_path / "projects", cap_usd=-1.0
        )


def test_load_project_budget_rejects_non_finite_cap(tmp_path: Path) -> None:
    with pytest.raises(BudgetError, match="finite non-negative"):
        load_project_budget(
            "demo", state_dir=tmp_path / "projects", cap_usd=float("inf")
        )


def test_load_project_budget_rejects_invalid_ledger(tmp_path: Path) -> None:
    ledger = tmp_path / "projects" / "demo" / "ledger.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text("{not json\n", encoding="utf-8")

    with pytest.raises(BudgetError, match="invalid ledger entry"):
        load_project_budget("demo", state_dir=tmp_path / "projects")


def test_load_project_budget_rejects_negative_ledger_cost(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "projects" / "demo" / "ledger.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(json.dumps({"cost_usd": -0.1}) + "\n", encoding="utf-8")

    with pytest.raises(BudgetError, match="invalid ledger cost"):
        load_project_budget("demo", state_dir=tmp_path / "projects")


def test_ensure_can_spend_blocks_at_and_above_the_cap(
    tmp_path: Path,
) -> None:
    budget = load_project_budget(
        "demo", state_dir=tmp_path / "projects", cap_usd=1.0
    )
    budget.record(_entry(cost_usd=1.0))

    with pytest.raises(BudgetExceededError, match="cap"):
        budget.ensure_can_spend(needed_usd=0.01)


def test_ensure_can_spend_blocks_when_needed_exceeds_remaining(
    tmp_path: Path,
) -> None:
    budget = load_project_budget(
        "demo", state_dir=tmp_path / "projects", cap_usd=1.0
    )
    budget.record(_entry(cost_usd=0.2))

    with pytest.raises(BudgetExceededError, match="needs about"):
        budget.ensure_can_spend(needed_usd=0.9)


def test_ensure_can_spend_warns_once_at_the_warning_fraction(
    tmp_path: Path, capsys
) -> None:
    budget = load_project_budget(
        "demo", state_dir=tmp_path / "projects", cap_usd=1.0
    )
    budget.record(_entry(cost_usd=0.85))

    budget.ensure_can_spend(needed_usd=0.01)
    budget.ensure_can_spend(needed_usd=0.01)

    err = capsys.readouterr().err
    assert err.count("has used") == 1


def test_record_appends_and_warns_when_over_the_cap(
    tmp_path: Path, capsys
) -> None:
    budget = load_project_budget(
        "demo", state_dir=tmp_path / "projects", cap_usd=0.5
    )

    budget.record(_entry(cost_usd=0.75))

    lines = budget.ledger_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["cost_usd"] == 0.75
    assert budget.spent_usd == pytest.approx(0.75)
    assert "now over" in capsys.readouterr().err


def test_record_rejects_mismatched_project(tmp_path: Path) -> None:
    budget = load_project_budget(
        "demo", state_dir=tmp_path / "projects", cap_usd=1.0
    )

    with pytest.raises(BudgetError, match="does not match"):
        budget.record(_entry(project="other", cost_usd=0.1))


def test_extend_validates_and_writes_the_new_cap(tmp_path: Path) -> None:
    budget = load_project_budget(
        "demo", state_dir=tmp_path / "projects", cap_usd=1.0
    )
    budget.record(_entry(cost_usd=0.5))

    budget.extend(2.0)

    assert budget.cap_usd == 2.0
    assert json.loads(budget.budget_path.read_text(encoding="utf-8"))[
        "cap_usd"
    ] == 2.0

    with pytest.raises(BudgetError, match="below"):
        budget.extend(0.4)
    with pytest.raises(BudgetError, match="greater"):
        budget.extend(1.5)
    with pytest.raises(BudgetError, match="finite non-negative"):
        budget.extend(-1.0)


def test_budget_policy_prompts_for_a_new_cap(tmp_path: Path) -> None:
    budget = load_project_budget(
        "demo", state_dir=tmp_path / "projects", cap_usd=1.0
    )
    prompts: list[BudgetExceededError] = []

    def prompt(exc: BudgetExceededError) -> float:
        prompts.append(exc)
        return 2.0

    policy = BudgetPolicy(budget, prompt=prompt)
    budget.record(_entry(cost_usd=1.0))

    policy.ensure(needed_usd=0.1, reason="builder call")

    assert len(prompts) == 1
    assert budget.cap_usd == 2.0


def test_budget_policy_raises_when_the_prompt_declines(
    tmp_path: Path,
) -> None:
    budget = load_project_budget(
        "demo", state_dir=tmp_path / "projects", cap_usd=1.0
    )
    budget.record(_entry(cost_usd=1.0))
    policy = BudgetPolicy(budget, prompt=lambda exc: None)

    with pytest.raises(BudgetExceededError):
        policy.ensure(needed_usd=0.1)


def test_project_budget_can_be_constructed_directly(tmp_path: Path) -> None:
    budget = ProjectBudget(
        project="demo",
        cap_usd=1.0,
        spent_usd=0.25,
        state_dir=tmp_path / "projects",
    )

    assert budget.remaining_usd == pytest.approx(0.75)
    assert budget.ledger_path == (
        tmp_path / "projects" / "demo" / "ledger.jsonl"
    )
