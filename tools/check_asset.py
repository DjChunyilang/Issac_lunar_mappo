#!/usr/bin/env python
"""Check first-stage proxy asset status or a future USD/URDF path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="")
    args = parser.parse_args()
    if not args.asset:
        print(
            json.dumps(
                {
                    "asset_mode": "first_stage_proxy",
                    "status": "ok",
                    "note": "No real rover USD/URDF is configured for stage 1.",
                },
                indent=2,
            )
        )
        return
    path = Path(args.asset)
    print(json.dumps({"asset": str(path), "exists": path.exists(), "is_file": path.is_file()}, indent=2))


if __name__ == "__main__":
    main()

