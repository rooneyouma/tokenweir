# TokenWeir

TokenWeir is a budget-aware, test-driven coding-agent orchestration project.
The current implementation runs one task in a disposable workspace using one
architect planning call and one configured builder model. It validates the
plan and returned files, runs pytest, flake8, and mypy, and commits each passing
subtask. Model routing, budget enforcement, and resume are not implemented yet.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
uv run flake8 --max-line-length=100 src/tokenweir tests scripts
uv run mypy --ignore-missing-imports src/tokenweir tests scripts
```

## Run a task

TokenWeir supports two modes:

- **Direct mode** supplies grading tests with `--tests PATH` or by placing a
  `tests/` directory next to the task file. The builder implements the whole
  task in one bounded retry loop.
- **Architect mode** runs when no grading tests are supplied. The architect
  plans once, writes complete pytest files, and the builder implements each
  subtask in dependency order. Each passing subtask is committed, then a final
  full-suite check runs.

```bash
# Direct mode with supplied tests
uv run tokenweir bench/tasks/palindrome/task.md
uv run tokenweir bench/tasks/palindrome
uv run tokenweir --task "Create palindrome.py" --tests bench/tasks/palindrome/tests

# Architect mode: plan, write tests, and build subtasks
uv run tokenweir --task "Create greet.py with greet(name: str) -> str"

# Inspect the validated plan without creating a workspace
uv run tokenweir --task "Create greet.py with greet(name: str) -> str" --plan-only
```

The positional argument can be inline task text, a task file, or a directory
containing `task.md`. The explicit `--task TEXT` and `--task-file PATH` forms
also work. `--plan-only` prints the validated plan JSON and does not call the
builder.

Each run creates `workspace/<run-id>/` with a fresh git repository. Direct mode
copies the supplied grading tests. Architect mode starts without tests, writes
the plan's test files before each subtask, and rejects builder output that does
not exactly match that subtask's declared files. Builder paths under `tests/`,
`.git/`, test configuration, or outside the workspace are rejected; architect
test paths must be under `tests/` and named `test_*.py`.

Run records are written to `traces/<run-id>.json` without credentials. Direct
records contain the builder attempts; architect records also contain the plan,
per-subtask attempts, and the final full-suite check.

Exit codes: `0` passed, `1` failed after the retry limit, `2` invalid input or
configuration.

## Configuration

`src/tokenweir/registry.json` selects the builder provider
(`builder.provider`, default `ollama`), the retry limit
(`builder.max_attempts`, 1-8), and builder temperature. The architect provider
(`architect.provider`) defaults to the builder provider; `architect.temperature`
defaults to `0.0`. Set the selected provider's model ID and credentials in
`.env`:

```bash
TOKENWEIR_OLLAMA_MODEL=ollama_chat/qwen2.5-coder:1.5b
# or, after changing builder.provider to deepseek:
TOKENWEIR_DEEPSEEK_MODEL=...
DEEPSEEK_API_KEY=...
```

The CLI does not choose or override models; the registry is the source of
truth. The architect's `recommended_tier` is validated but is not used for
routing yet. This phase has no router, ledger, or budget cap, so use free/local
providers or accept the cost of the configured architect and builder. Small
local models can produce valid plans but still fail a subtask; choose a
stronger configured model if that happens.

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
