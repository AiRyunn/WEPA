# AGENTS.md

This AGENTS.md provides persistent instructions for Codex when working in this repository.

Treat this file as always in force. Follow it before reading, editing, running, or proposing changes.

## Scope

This repository is a Python machine learning project focused on research code, experiments, and reproducible evaluation.

Optimize for:

- correctness
- reproducibility
- readability
- experiment traceability
- safe code modification

Do not optimize for speed of implementation at the expense of any of the above.

## Operating Principles

### Think Before Acting

Reason step-by-step before making code changes.

Before changing anything:

1. Understand the task and the expected behavior.
2. Identify the minimal set of affected files.
3. Check whether the change impacts training, generation, evaluation, caching, logging, or saved outputs.
4. Consider reproducibility risks, data leakage risks, and backward-compatibility risks.

Do not jump directly into implementation when the request is ambiguous or when a small misunderstanding could invalidate experiments.

### Plan First

For any non-trivial task, provide a short plan before editing files.

Include:

- files to change
- any new files to create
- expected behavior change
- key risks or assumptions

If the task is complex, potentially disruptive, or changes experimental behavior, wait for approval before implementing.

Examples of tasks that require a plan first:

- changing model behavior
- changing watermark logic
- altering experiment config schema
- modifying dataset loading or caching
- changing evaluation metrics
- adding dependencies
- restructuring package layout

Package layout changes require explicit approval and should preserve a single canonical package root.

### Prefer Minimal Safe Changes

Make the smallest change that fully solves the problem.

- Preserve existing APIs unless explicitly asked to change them.
- Preserve file layout and experiment entrypoints unless there is a strong reason to modify them.
- Do not mix refactors with behavioral changes unless the user asked for both.
- Avoid broad formatting-only diffs.

### Use Diff-Style Suggestions

When proposing code changes in prose, prefer unified diff style.

### Never Be Destructive Without Explicit Confirmation

Do not delete or overwrite user work without confirmation.

Never run destructive commands such as:

- `rm -rf`
- dataset or cache deletion
- bulk notebook output clearing
- `git reset --hard`
- `git checkout --`
- history rewriting

Do not remove files, cached results, checkpoints, or generated outputs unless the user explicitly asks for that exact action.
Do not remove duplicate package roots or stale source directories during packaging refactors unless the user explicitly approves that cleanup.

## Repository Structure

Understand the repo before changing it.

- `src/watermarker/`: core watermarking implementations and shared model logic
- `experiments/runs/`: runnable experiment modules with `run(...)` entrypoints
- `experiments/configs/`: YAML experiment configs
- `experiments/run.py`: config-driven experiment dispatcher
- `experiments/utils/`: shared experiment helper package for config loading, runtime setup, data loading, metrics, logging, and execution
- `experiments/notebooks/`: manual exploration notebooks; keep them thin and secondary to scripts
- `scripts/`: shell wrappers for running experiments
- `pyproject.toml`: package metadata and dependencies

`src/watermarker/` is the canonical library source root. Do not reintroduce a duplicate top-level `watermarker/` package.
Imports should work via package installation, not repository-relative path mutation.

Prefer changes in library code and runner modules over notebook-only logic.

## Python Code Standards

Write clean, readable, modular Python.

- Prefer small, composable functions.
- Keep responsibilities narrow.
- Use explicit names over clever ones.
- Prefer simple control flow over dense abstractions.
- Match the surrounding code style unless it is clearly harmful.

### Type Hints

Add or preserve type hints for public functions and non-trivial internal functions when practical.

Do not add noisy annotations that reduce readability.

### Docstrings and Comments

Add docstrings and comments when logic is non-obvious.

- Explain why, not what, when the code is already readable.
- Document assumptions, tensor shapes, expected input formats, and side effects when relevant.
- Avoid redundant comments.

## Documentation

- For detailed natural-language explanations of functions, read and follow `docs/function-reference.md`.
- When editing or explaining a function, first check if it has an entry there.
- If you add, remove, rename, or materially change a function or method, update `docs/function-reference.md` in the same change.
- Keep the reference aligned with actual code behavior; do not let it become aspirational or stale.

### Errors and Logging

Fail clearly.

- Raise informative exceptions.
- Validate required config fields early.
- Prefer explicit checks over silent fallback behavior.
- Use clear logging or print messages for long-running experiment steps, cache hits/misses, and artifact writes.

Do not swallow exceptions unless there is a strong recovery path.

## ML and Experimentation Rules

### Reproducibility Is Mandatory

Protect determinism unless the user explicitly asks otherwise.

- Preserve seed handling.
- Do not introduce hidden nondeterminism.
- If you add randomness, thread it through config and seed setup.
- Keep config-driven behavior explicit and inspectable.

When changing training, generation, or evaluation behavior, state whether results may change and why.

### Preserve Experiment Traceability

Every experiment should be attributable to:

- a config file
- code version
- model choice
- seed
- output path or artifact destination

Do not hardcode hidden parameters when they should live in config.

If adding a new experiment:

- add a dedicated config under `experiments/configs/`
- add a dedicated runner under `experiments/runs/`
- keep `experiments/run.py` behavior consistent with the existing dispatch model
- add a script under `scripts/` if the experiment is intended to be commonly run

### Config Discipline

Treat YAML configs as part of the public experiment interface.

- Prefer adding new config keys over embedding values directly in code.
- Validate required config fields before use.
- Keep config naming consistent with existing files.
- Do not silently ignore unknown keys if they indicate a likely user mistake.

If changing config schema, document the migration impact clearly.

### Datasets and Caches

Handle data carefully.

- Do not change dataset splits, filters, or tokenization semantics casually.
- Flag any change that can alter reported metrics.
- Preserve cache compatibility when possible.
- Avoid introducing duplicate cache formats without a reason.
- Never delete caches or downloaded data unless explicitly asked.

If a cache key depends on preprocessing behavior, update it carefully to avoid stale or misleading reuse.

### Metrics and Evaluation

Be precise with evaluation logic.

- Define positive and negative classes explicitly.
- Check for metric directionality mistakes.
- Keep naming aligned with actual semantics.
- Do not change evaluation thresholds or defaults without surfacing the consequence.

If touching metrics, verify behavior with a small deterministic example when possible.

## Working in This Repository

### Before Editing

Inspect the relevant files first.

At minimum:

- read the direct entrypoint
- read shared helpers it depends on
- check config files that drive the behavior
- look for matching patterns in sibling experiment modules

Do not invent new patterns if a suitable local pattern already exists.

### Packaging and Imports

Preserve installed-package workflows.

- Use `pip install -e .` as the standard installation path.
- Keep [`pyproject.toml`](/data/airyunn/WEPA/pyproject.toml) aligned with the actual package layout.
- Do not add `sys.path` mutation in runners, scripts, notebooks, or library code.
- Keep notebooks compatible with normal package imports after editable install.
- If packaging changes affect imports or entrypoints, update the relevant docs in [`README.md`](/data/airyunn/WEPA/README.md) in the same change.

### When Editing

Keep edits localized.

- avoid unrelated renames
- avoid opportunistic cleanup
- preserve user comments unless they are wrong
- preserve imports and ordering unless a real change requires adjustment

If you notice unrelated bugs, mention them separately instead of bundling them into the task.

### New Dependencies

Be conservative about adding packages.

- Prefer the standard library or already-installed dependencies.
- Add a dependency only when it materially improves correctness or maintainability.
- If you add one, update the appropriate dependency file and explain why it is needed.

### Notebooks

Treat notebooks as secondary interfaces.

- Prefer putting real logic in Python modules.
- Keep notebooks as thin wrappers around reusable functions or runners.
- Do not duplicate core logic only in notebooks.
- Avoid large output-heavy notebook diffs unless explicitly requested.

## Validation and Testing

Do not claim a change works without checking it.

Choose the lightest validation that gives real confidence.

Examples:

- run the relevant module help command
- execute the targeted script path
- run a small deterministic smoke test
- validate YAML loading
- import the changed module

When practical, verify:

- the code imports cleanly
- the config loads
- the entrypoint still runs
- outputs are written where expected
- seeds and device handling still behave correctly

If you cannot run validation, say so explicitly and explain why.

## Command Safety

Assume the repository may contain valuable outputs, caches, and intermediate experiment artifacts.

- Do not remove files to "start fresh".
- Do not rewrite git history.
- Do not modify large generated artifacts unless required.
- Do not edit lockfiles or dependency pins casually.

Prefer read-only inspection first, then minimal edits, then targeted validation.

## Communication Style

Be direct, concise, and technical.

When responding:

- state assumptions when they matter
- call out risks explicitly
- separate observed facts from inference
- do not overstate confidence

For substantial changes, summarize:

- what changed
- why it changed
- what you validated
- any residual risk

## Review Checklist

Before finishing, check:

- Is the change minimal?
- Is the behavior correct?
- Is reproducibility preserved?
- Is the config interface still clear?
- Are errors explicit?
- Are docs/comments sufficient for non-obvious logic?
- Did you avoid destructive actions?
- Did you validate the affected path?

## Preferred Workflow

Follow this default sequence:

1. Read the relevant files and config.
2. Summarize the intended change and risks.
3. Present a short plan for non-trivial work.
4. Implement the minimal safe edit.
5. Run targeted validation.
6. Report what changed and any remaining caveats.

## Project-Specific Guidance

This repository appears to use:

- config-driven experiment dispatch via `python -m experiments.run`
- shell wrappers in `scripts/`
- reusable helpers in `experiments/utils/`
- packaged watermarking logic in `src/watermarker/`

Respect that structure.

- Add new experiment behavior in runners and helpers, not in ad hoc scripts.
- Keep script wrappers minimal.
- Keep package code importable and isolated from notebook-only assumptions.
- Treat `experiments/utils/` as shared infrastructure; changes there have broad impact.

When in doubt, choose the approach that makes experiments easier to rerun, inspect, and compare later.
