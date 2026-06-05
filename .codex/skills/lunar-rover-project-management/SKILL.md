---
name: lunar-rover-project-management
description: Use for this repository when organizing documentation, summarizing experiment status, interpreting proxy/PhysX training results, managing outputs, or deciding whether a run passed strict acceptance. Triggers on requests about project progress, experiment docs, training result summaries, output layout, run management, or agent-readable project context.
---

# Lunar Rover Project Management

## Read First

When this skill triggers, read these project docs in order:

1. `docs/current_status.md`
2. `docs/experiments/README.md`
3. The relevant `docs/experiments/exp_*.md`
4. `docs/references/output_management.md` if paths or output layout matter

Use `docs/archive/` only for provenance. Do not treat archive logs as the current source of truth.

## Strict Acceptance

Do not infer success from GIFs, TensorBoard curves, or one favorable checkpoint. Strict proxy acceptance is:

```text
dmax_reduction_ratio <= 0.2
success_rate >= 0.9
collision_rate <= 0.02
timeout_rate == 0
```

Prefer independent `metrics/final_eval_proxy.json` and suite-level `_suite/metrics/strict_acceptance.json` over training-internal impressions.

## Documentation Rules

- Keep `docs/current_status.md` short and current.
- Add one file per experiment under `docs/experiments/`.
- Move long date-based progress logs to `docs/archive/`.
- Add commands to `docs/runbooks/`, not to every experiment note.
- Track configs, scripts, source, tests, and Markdown. Do not commit generated outputs.

## Output Rules

Canonical generated results live under:

```text
outputs/runs/<experiment_id>/<run_id>/
```

Suite-level results live under:

```text
outputs/runs/<experiment_id>/_suite/
```

`outputs/**` is ignored. If a result must be preserved in git, summarize it in Markdown and keep the raw artifact in `outputs/`.

## Training Interpretation

- exp008 is the current full 3-seed terrain-aware strict pass.
- exp009 strong terrain is a diagnostic failure: seed23 passed, seed31 failed success/timeout, seed47 was not run.
- Do not continue unbounded PPO just because one seed is close. Diagnose failure gates first.

