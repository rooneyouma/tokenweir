# TokenWeir project brief

Goal: build small, self-contained Python hobby projects from a short task. The
architect plans once; application code selects models. Grading tests are
supplied by the architect and are independent of the builder.

Stack and conventions:

- Python 3.12, standard library first.
- pytest for tests; flake8 (max line length 100) and mypy (ignore missing
  imports) must pass.
- Add type annotations to public functions and classes.
- Keep each file focused and small enough for a 1.5B to 3B builder model.

Constraints:

- Only Python unless the task explicitly requires otherwise.
- No network access, secrets, private data, or external services.
- The builder must not create or modify tests or test configuration.
- Use relative paths inside the generated workspace.

Model roles:

- architect: plans subtasks, files, specs, and tests. It does not write code.
- builder: writes only the files declared by one subtask.
- router: chooses the model for each subtask from difficulty and the
  recommended tier. The architect only recommends a free starting tier.

Return a validated plan; do not include commentary.
