#!/usr/bin/env python
"""Print compact training log summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", nargs="?", default="outputs/logs/exp_001_minimal/train_metrics.json")
    args = parser.parse_args()
    data = json.loads(Path(args.log).read_text(encoding="utf-8"))
    metrics = data.get("metrics", [])
    print(json.dumps({"updates": len(metrics), "last": metrics[-1] if metrics else None}, indent=2))


if __name__ == "__main__":
    main()

