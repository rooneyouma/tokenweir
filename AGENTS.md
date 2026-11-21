# TokenWeir contributor guidance

TokenWeir is being built in phases. Keep the current CLI and provider checks
small; do not imply that task execution, model routing, budget enforcement, or
resume behavior exists until those phases are implemented and tested.

Protect these product invariants as orchestration is added:

- The architect plans once; application code, not the architect, selects models.
- Grading tests are supplied independently of the builder.
- Prefer local/free providers; paid escalation is bounded and enforced in code.
- Account for every model call and enforce paid budgets before making a call.
- Generated files belong only in an explicitly owned disposable workspace.
- Subprocess checks are not a security boundary; generated code is untrusted.
- Never write API keys to output, source files, generated workspaces, or logs.

Use the documented `uv` commands in `README.md`. Run tests from any working
directory where practical, and add focused tests for behavior and failure cases.
