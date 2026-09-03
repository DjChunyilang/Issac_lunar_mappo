#!/usr/bin/env python3
"""Reject legacy or unbalanced math delimiters in current project documents."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_DOCUMENTS = (
    "docs/README.md",
    "docs/current_status.md",
    "docs/implementation_plan.md",
    "docs/roadmap.md",
    "docs/architecture/overall_plan_v3.md",
    "docs/experiments/exp_155_multiscale_network_ablation.md",
    "docs/experiments/exp_156_differential_multiscale_ablation.md",
    "docs/experiments/exp_157_site_belief_diagnostic.md",
    "docs/experiments/exp_158_dae_validation.md",
    "docs/experiments/exp_159_analytical_prd.md",
    "docs/references/marl_credit_assignment_technical_appendix.md",
    "docs/status/weekly_report_2026-08-10_to_2026-08-14.md",
    "docs/status/weekly_report_2026-08-17_to_2026-08-21.md",
    "docs/status/ppt_plan_weekly_2026-08-17_to_2026-08-21.md",
)
LEGACY = re.compile(r"\\[()\[\]]")
INLINE_CODE = re.compile(r"`[^`]*`")
UNESCAPED_DOLLAR = re.compile(r"(?<!\\)\$")


def validate(path: Path) -> list[str]:
    errors = []
    in_fence = False
    in_block_math = False
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = raw_line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        without_code = INLINE_CODE.sub("", raw_line)
        if LEGACY.search(without_code):
            errors.append(f"{path}:{line_number}: legacy math delimiter")
        if stripped == "$$":
            in_block_math = not in_block_math
            continue
        if in_block_math:
            continue
        if len(UNESCAPED_DOLLAR.findall(without_code)) % 2:
            errors.append(f"{path}:{line_number}: unbalanced inline dollar delimiter")
    if in_fence:
        errors.append(f"{path}: unclosed fenced code block")
    if in_block_math:
        errors.append(f"{path}: unclosed block math delimiter")
    return errors


def main() -> None:
    errors = []
    for relative in ACTIVE_DOCUMENTS:
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"{path}: missing active document")
            continue
        errors.extend(validate(path))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)
    print(f"Markdown math delimiter check passed for {len(ACTIVE_DOCUMENTS)} documents.")


if __name__ == "__main__":
    main()
