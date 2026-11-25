from typing import Literal

import pytest

from tokenweir.router import RoutingError, candidates
from tokenweir.schemas import Subtask
from tokenweir.settings import ConfiguredProvider, LoadedSettings

Tier = Literal["local", "free_cloud", "paid"]
Difficulty = Literal["easy", "medium", "hard"]
RecommendedTier = Literal["local_small", "local_medium", "free_cloud"]


def _provider(name: str, tier: Tier) -> ConfiguredProvider:
    return ConfiguredProvider(
        name=name,
        model=f"model/{name}",
        tier=tier,
        free=tier != "paid",
    )


def _settings(providers: list[ConfiguredProvider]) -> LoadedSettings:
    return LoadedSettings(
        providers=providers,
        unconfigured=[],
        errors=[],
    )


def _subtask(
    *,
    difficulty: Difficulty = "easy",
    recommended_tier: RecommendedTier = "local_small",
) -> Subtask:
    return Subtask(
        id="impl",
        title="Implement",
        type="function",
        difficulty=difficulty,
        recommended_tier=recommended_tier,
        files=["impl.py"],
        spec="Write impl.py.",
        tests=["tests/test_impl.py"],
    )


def _names(providers: list[ConfiguredProvider]) -> list[str]:
    return [provider.name for provider in providers]


def test_easy_local_tier_starts_with_local_providers() -> None:
    settings = _settings(
        [
            _provider("paid-a", "paid"),
            _provider("free-b", "free_cloud"),
            _provider("local-b", "local"),
            _provider("local-a", "local"),
        ]
    )

    result = candidates(settings, _subtask())

    assert _names(result) == ["local-a", "local-b", "free-b", "paid-a"]


def test_hard_task_starts_with_free_cloud_providers() -> None:
    settings = _settings(
        [
            _provider("paid-a", "paid"),
            _provider("local-a", "local"),
            _provider("free-a", "free_cloud"),
        ]
    )

    result = candidates(
        settings,
        _subtask(difficulty="hard", recommended_tier="local_small"),
    )

    assert _names(result) == ["free-a", "local-a", "paid-a"]


def test_free_cloud_recommendation_starts_with_free_cloud() -> None:
    settings = _settings(
        [
            _provider("local-a", "local"),
            _provider("free-a", "free_cloud"),
        ]
    )

    result = candidates(
        settings,
        _subtask(difficulty="easy", recommended_tier="free_cloud"),
    )

    assert _names(result) == ["free-a", "local-a"]


def test_medium_difficulty_stays_on_the_local_rung() -> None:
    settings = _settings(
        [
            _provider("local-a", "local"),
            _provider("free-a", "free_cloud"),
        ]
    )

    result = candidates(
        settings,
        _subtask(difficulty="medium", recommended_tier="local_small"),
    )

    assert _names(result) == ["local-a", "free-a"]


def test_unavailable_budget_excludes_paid_providers() -> None:
    settings = _settings(
        [
            _provider("paid-a", "paid"),
            _provider("local-a", "local"),
            _provider("free-a", "free_cloud"),
        ]
    )

    result = candidates(
        settings, _subtask(), budget_available=False
    )

    assert _names(result) == ["local-a", "free-a"]


def test_no_affordable_candidates_raises() -> None:
    settings = _settings([_provider("paid-a", "paid")])

    with pytest.raises(RoutingError, match="within the current budget"):
        candidates(settings, _subtask(), budget_available=False)


def test_no_configured_candidates_raises() -> None:
    settings = _settings([])

    with pytest.raises(RoutingError, match="no configured provider"):
        candidates(settings, _subtask())
