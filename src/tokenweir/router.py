from __future__ import annotations

from tokenweir.schemas import Subtask
from tokenweir.settings import ConfiguredProvider, LoadedSettings

_TIER_ORDER = {"local_small": 0, "local_medium": 1, "free_cloud": 2}
_DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}


class RoutingError(RuntimeError):
    """Raised when no provider can run a subtask."""


def _start_tier(subtask: Subtask) -> int:
    score = max(
        _TIER_ORDER[subtask.recommended_tier],
        _DIFFICULTY_ORDER[subtask.difficulty],
    )
    return 0 if score <= 1 else 1


def _by_name(providers: list[ConfiguredProvider]) -> list[ConfiguredProvider]:
    return sorted(providers, key=lambda provider: provider.name)


def candidates(
    settings: LoadedSettings,
    subtask: Subtask,
    *,
    budget_available: bool = True,
) -> list[ConfiguredProvider]:
    """Return deterministic provider candidates for one subtask."""
    local = _by_name(
        [provider for provider in settings.providers if provider.tier == "local"]
    )
    free_cloud = _by_name(
        [
            provider
            for provider in settings.providers
            if provider.tier == "free_cloud"
        ]
    )
    paid = _by_name(
        [provider for provider in settings.providers if provider.tier == "paid"]
    )

    if _start_tier(subtask) == 0:
        ordered = local + free_cloud + paid
    else:
        ordered = free_cloud + local + paid

    if not budget_available:
        ordered = [
            provider
            for provider in ordered
            if provider.free or provider.tier != "paid"
        ]

    if not ordered:
        reason = (
            "within the current budget"
            if not budget_available
            else "for this task"
        )
        raise RoutingError(f"no configured provider can run this subtask {reason}")
    return ordered
