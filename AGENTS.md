# TokenWeir contributor guidance

TokenWeir is being built in phases. The current implementation runs one task in
a disposable workspace either directly with supplied tests or through one
architect planning call plus one configured builder model. Do not imply that
model routing, quota tracking, budget enforcement, or resume behavior exists
until those phases are implemented and tested.

Protect these product invariants as orchestration is added:

- The architect plans once; the plan is validated before any workspace is
  created. Application code, not the architect, selects models.
- Grading tests are supplied independently of the builder: from the task in
  direct mode, or from the validated plan in architect mode.
- Builders write only the files declared by their subtask and never tests.
- The architect only recommends a tier; routing does not exist yet.
- Prefer local/free providers; paid escalation is bounded and enforced in code.
- Account for every model call and enforce paid budgets before making a call.
- Generated files belong only in an explicitly owned disposable workspace.
- Subprocess checks are not a security boundary; generated code and tests are
  untrusted.
- Never write API keys to output, source files, generated workspaces, or logs.

Use the documented `uv` commands in `README.md`. Run tests from any working
directory where practical, and add focused tests for behavior and failure cases.
