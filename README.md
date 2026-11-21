# TokenWeir

TokenWeir is a budget-aware, test-driven coding-agent orchestration project.
The current implementation provides a task-input CLI, validated provider
configuration, endpoint health checks, and a disposable test sandbox. The
multi-model build loop and task execution are not implemented yet.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
uv run flake8 --max-line-length=100 src/tokenweir tests scripts
uv run mypy --ignore-missing-imports src/tokenweir tests scripts
```

## CLI

The canonical invocation takes one positional argument: either inline task
text, or the path to an existing task file.

```bash
uv run tokenweir --help
uv run tokenweir "Add a palindrome checker"
uv run tokenweir bench/tasks/palindrome/task.md
```

If the argument names an existing file, its contents are loaded; otherwise it
is treated as inline task text. The explicit `--task TEXT` and
`--task-file PATH` forms also work when you want to be unambiguous.

The CLI validates and loads the task, but does not execute it until the build
loop is implemented.

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
