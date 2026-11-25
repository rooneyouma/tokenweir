# TokenWeir contributor guidance

TokenWeir is being built in phases. The current implementation runs one task in
a disposable workspace either directly with supplied tests or through one
architect planning call plus a configured builder. Architect mode uses a
deterministic router, and paid calls are enforced against a per-project cap and
recorded in a JSONL ledger. Do not imply that per-provider quota tracking or
automatic resume exists until those phases are implemented and tested.

Protect these product invariants as orchestration is added:

- The architect plans once; the plan is validated before any workspace is
  created. Application code, not the architect, selects models.
- Grading tests are supplied independently of the builder: from the task in
  direct mode, or from the validated plan in architect mode.
- Builders write only the files declared by their subtask and never tests.
- The architect only recommends a tier; the application router selects models.
- Prefer local/free providers; paid escalation is bounded and enforced in code.
- Account for every model call and enforce paid budgets before making a call.
- A paused budget run preserves its workspace and trace; resume is not automatic.
- Generated files belong only in an explicitly owned disposable workspace.
- Subprocess checks are not a security boundary; generated code and tests are
  untrusted.
- Never write API keys to output, source files, generated workspaces, or logs.

Use the documented `uv` commands in `README.md`. Run tests from any working
directory where practical, and add focused tests for behavior and failure cases.
