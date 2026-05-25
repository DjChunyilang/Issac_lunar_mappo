#!/usr/bin/env python
"""Generate first-stage terrain metadata."""

from __future__ import annotations

import json


def main() -> None:
    print(json.dumps({"terrain": "flat_proxy", "height": 0.0, "status": "ok"}, indent=2))


if __name__ == "__main__":
    main()

