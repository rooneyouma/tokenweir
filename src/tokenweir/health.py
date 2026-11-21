from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
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

from tokenweir.settings import ConfiguredProvider, SettingsError, load_settings


def ping_provider(provider: ConfiguredProvider) -> tuple[bool, str]:
    request: dict[str, Any] = {
        "model": provider.model,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 1,
        "timeout": 2,
        "num_retries": 0,
    }
    if provider.api_key is not None:
        request["api_key"] = provider.api_key.get_secret_value()
    if provider.base_url:
        request["api_base"] = provider.base_url

    try:
        completion(**request)
    except AuthenticationError:
        return False, "authentication failed; verify the configured API key"
    except RateLimitError:
        return False, "provider is reachable but rate-limited this health check"
    except Timeout:
        return False, "health check timed out after 2 seconds"
    except APIConnectionError:
        return False, "could not connect to the provider endpoint"
    except APIError as exc:
        return False, f"provider returned an API error ({type(exc).__name__})"
    return True, "reachable"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ping_models",
        description="Check configured model endpoints with a short LiteLLM request.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help="Use a provider registry JSON file other than the packaged default.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(args.registry)
    except SettingsError as exc:
        print(f"ping_models: {exc}", file=sys.stderr)
        return 2

    for message in settings.unconfigured:
        print(f"SKIP {message}")
    for message in settings.errors:
        print(f"ERROR {message}")

    failures = bool(settings.errors)
    for provider in settings.providers:
        passed, message = ping_provider(provider)
        status = "PASS" if passed else "FAIL"
        print(f"{status} {provider.name}: {message}")
        failures = failures or not passed

    if not settings.providers:
        print("No providers are configured; set model environment variables to enable checks.")
        return 1
    return 1 if failures else 0
