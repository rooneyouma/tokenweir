# TokenWeir

TokenWeir is a budget-aware, test-driven coding-agent orchestration project.
The current implementation runs one task through one configured builder model
inside a disposable workspace, validates the returned files, runs pytest,
flake8, and mypy, and commits a passing result. Architect planning, model
routing, budget enforcement, and resume are not implemented yet.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
uv run flake8 --max-line-length=100 src/tokenweir tests scripts
uv run mypy --ignore-missing-imports src/tokenweir tests scripts
```

## Run a task

A task needs grading tests. The CLI looks for a `tests/` directory next to the
task file, or accepts `--tests PATH`.

```bash
uv run tokenweir bench/tasks/palindrome/task.md
uv run tokenweir bench/tasks/palindrome
uv run tokenweir --task "Create palindrome.py" --tests bench/tasks/palindrome/tests
```

The positional argument can be inline task text, a task file, or a directory
containing `task.md`. The explicit `--task TEXT` and `--task-file PATH` forms
also work. Inline tasks require `--tests` until the architect can write tests.

Each run creates `workspace/<run-id>/` with a fresh git repository and a copy
of the grading tests. The builder returns a JSON object shaped like
`{"files": [{"path": "relative/path.py", "content": "..."}]}`. Only those files
are written; paths under `tests/`, `.git/`, test configuration, or outside the
workspace are rejected. Passing checks are committed as one step. Run records
are written to `traces/<run-id>.json` without credentials.

Exit codes: `0` passed, `1` failed after the retry limit, `2` invalid input or
configuration.

## Builder configuration

`src/tokenweir/registry.json` selects the builder provider
(`builder.provider`, default `ollama`), the retry limit
(`builder.max_attempts`, 1-8), and temperature. Set that provider's model ID
and credentials in `.env`:

```bash
TOKENWEIR_OLLAMA_MODEL=ollama_chat/qwen2.5-coder:1.5b
# or, after changing builder.provider to deepseek:
TOKENWEIR_DEEPSEEK_MODEL=...
DEEPSEEK_API_KEY=...
```

The CLI does not choose or override models; the registry is the source of
truth. This phase has no router, ledger, or budget cap yet, so use free/local
providers or accept the cost of the configured builder.

## Provider health checks

Copy `.env.example` to `.env`, then set the model IDs and credentials for the
providers you want to check:

```bash
uv run python scripts/ping_models.py
```

Model values must be LiteLLM model IDs. The registry is
[`src/tokenweir/registry.json`](src/tokenweir/registry.json); model IDs are
intentionally supplied by the user rather than assuming a particular model
name or availability. Ollama defaults to `http://localhost:11434`; override it
with `TOKENWEIR_OLLAMA_BASE_URL` when needed. The health check sends a minimal
request with a two-second timeout. Cloud checks may incur a small provider
charge. API keys are read from environment variables and are not printed.
