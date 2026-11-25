# TokenWeir

TokenWeir is a budget-aware, test-driven coding-agent orchestration project.
The current implementation runs one task in a disposable workspace using one
architect planning call and a configured builder. In architect mode a
deterministic router chooses providers for each subtask. It validates the plan
and returned files, runs pytest, flake8, and mypy, and commits each passing
subtask. Every model call is recorded in a per-project ledger, and paid calls
are checked against a project cap before they run. Paused or failed runs can be
resumed explicitly from their run record; resume is never automatic.

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
  plans once, writes complete pytest files, and the router chooses a provider
  for each subtask in dependency order. Each passing subtask is committed, then
  a final full-suite check runs.

```bash
# Direct mode with supplied tests
uv run tokenweir bench/tasks/palindrome/task.md
uv run tokenweir bench/tasks/palindrome
uv run tokenweir --task "Create palindrome.py" --tests bench/tasks/palindrome/tests

# Architect mode: plan, write tests, and build subtasks
uv run tokenweir --task "Create greet.py with greet(name: str) -> str"

# Set a project cap before running
uv run tokenweir --task "Create greet.py with greet(name: str) -> str" \
  --project greet --cap 1.50

# Inspect the validated plan without creating a workspace
uv run tokenweir --task "Create greet.py with greet(name: str) -> str" --plan-only

# Resume a paused or failed run from its preserved workspace
uv run tokenweir --resume traces/<run-id>.json --cap 3.00
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
per-subtask attempts, the chosen router configuration, and the final full-suite
check. Both include a budget summary and no API keys.

Paid calls are checked before they run. When a paid call would exceed the
project cap, free/local providers remain available. In an interactive terminal
TokenWeir asks for a new cap and continues if one is supplied. In a
non-interactive run it pauses, preserves the workspace and trace, and exits with
code `3`; rerun with `--cap USD` to extend the project.

To continue a paused or failed run, pass its record to `--resume`:

```bash
uv run tokenweir --resume traces/<run-id>.json --cap 3.00
```

Resume uses the project recorded in the trace unless `--project` is supplied,
resets the preserved workspace to its last committed subtask, and skips
subtasks that already passed. A new run record links back with `resumed_from`.
Run resume from the same working directory as the original run so the recorded
workspace path resolves. Resume is explicit; a paused run never continues on its
own.

Provider rate-limit and quota errors are detected at call time. The affected
provider is skipped for the rest of the run and the router tries the next
candidate; the architect also falls back to a free provider when planning
fails. Fixed per-provider request/token quotas are not tracked.

Exit codes: `0` passed, `1` failed after the retry limit, `2` invalid input or
configuration, `3` paused because the project budget is exhausted.

## Configuration

`src/tokenweir/registry.json` selects the builder provider
(`builder.provider`, default `ollama`), the direct-mode retry limit
(`builder.max_attempts`, 1-8), and builder temperature. The architect provider
(`architect.provider`) defaults to the builder provider; `architect.temperature`
defaults to `0.0`.

Router settings apply in architect mode. `router.attempts_per_provider` bounds
attempts on one provider and `router.max_total_attempts` bounds attempts across
providers for a subtask. Candidates are ordered deterministically from
`local`, `free_cloud`, and `paid` providers based on the subtask difficulty and
the architect's `recommended_tier`; paid providers are excluded when the project
cap is exhausted. Pass `--no-router` to use only `builder.provider` in architect
mode.

Budget settings are `budget.default_cap_usd`, `budget.warn_fraction`, and
`budget.state_dir`. Each project gets `budget.json` and `ledger.jsonl` under
`<state_dir>/<project-slug>/`; `--project` selects the group and `--cap` sets or
replaces its cap. `--state-dir` overrides the registry's state directory. The
ledger records timestamp, project, run, role, provider, model, task/subtask,
attempt, tokens, estimated cost, and whether the call was free.

Provider definitions may set `free`, `tier` (`local`, `free_cloud`, or `paid`),
`max_output_tokens`, and optional `input_cost_per_million` /
`output_cost_per_million` values. Set the selected provider's model ID and
credentials in `.env`:

```bash
TOKENWEIR_OLLAMA_MODEL=ollama_chat/qwen2.5-coder:1.5b
# or, after changing builder.provider to deepseek:
TOKENWEIR_DEEPSEEK_MODEL=...
DEEPSEEK_API_KEY=...
```

The CLI does not choose or override models; the registry and router are the
source of truth. Fixed per-provider request/token quotas are not tracked; quota
and rate-limit errors trigger provider fallback instead. Small local models can
produce valid plans but still fail a subtask; the router escalates within the
configured bounds, and you can choose a stronger configured model if that
happens.

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
