#!/usr/bin/env python
"""Check the small set of repository boundaries that keep agent context reliable."""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ENTRY_DOCS = (
    "README.md",
    "docs/README.md",
    "docs/current_status.md",
    "docs/design/README.md",
)
REQUIRED_PATHS = (
    "source/lunar_rover_tasks/setup.py",
    "scripts/train.py",
    "configs/experiment/exp_001_minimal.yaml",
    "docs/experiments/README.md",
    "docs/references/output_management.md",
)
LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")


def main() -> None:
    tracked_outputs = subprocess.check_output(
        ["git", "ls-files", "--", "outputs"], cwd=ROOT, text=True
    ).splitlines()
    if tracked_outputs:
        raise SystemExit(f"Generated outputs are tracked: {tracked_outputs}")

    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "outputs/runs/example/checkpoints/best.pt"],
        cwd=ROOT,
        check=False,
    )
    if ignored.returncode != 0:
        raise SystemExit("outputs/ must be ignored by Git")

    for relative in REQUIRED_PATHS:
        if not (ROOT / relative).is_file():
            raise SystemExit(f"Missing canonical path: {relative}")

    root_config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    if "project" in root_config or "build-system" in root_config:
        raise SystemExit("Root pyproject.toml must contain tool configuration only")

    for path in (ROOT / "configs/experiment").glob("*.yaml"):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        base = payload.get("extends")
        if base and (path.parent / base).resolve().is_relative_to(ROOT / "outputs"):
            raise SystemExit(f"Experiment config extends generated output: {path}")

    for relative in ENTRY_DOCS:
        path = ROOT / relative
        for target in LINK.findall(path.read_text(encoding="utf-8")):
            target = target.split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            if not (path.parent / target).exists():
                raise SystemExit(f"Broken entry link: {relative} -> {target}")

    print("Repository hygiene checks passed")


if __name__ == "__main__":
    main()
