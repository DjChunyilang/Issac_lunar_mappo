#!/usr/bin/env python
"""Print TensorBoard scalar tags grouped by run directory."""

from __future__ import annotations

import argparse
from pathlib import Path


def _load_event_accumulator():
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as exc:  # pragma: no cover - depends on optional tensorboard package
        raise SystemExit(f"TensorBoard event accumulator is unavailable: {exc}") from exc
    return EventAccumulator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="outputs/logs/exp_006_ppo_selected")
    args = parser.parse_args()

    logdir = Path(args.logdir)
    event_dirs = sorted({path.parent for path in logdir.rglob("events.out.tfevents.*")})
    if not event_dirs:
        raise SystemExit(f"No TensorBoard event files found under {logdir}")

    event_accumulator = _load_event_accumulator()
    for event_dir in event_dirs:
        accumulator = event_accumulator(str(event_dir), size_guidance={"scalars": 0})
        accumulator.Reload()
        print(event_dir)
        for tag in sorted(accumulator.Tags().get("scalars", [])):
            events = accumulator.Scalars(tag)
            last_value = events[-1].value if events else None
            print(f"  {tag}: n={len(events)} last={last_value}")


if __name__ == "__main__":
    main()
