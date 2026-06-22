"""Lightweight SKRL run metadata helpers."""

from __future__ import annotations

import re
from pathlib import Path, PureWindowsPath
from typing import Any


DEFAULT_TRAINING_SEMANTICS = "skrl_mappo_smoke"


class CheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint does not match the active observation interface."""


def _section(raw_cfg: dict[str, Any], name: str) -> dict[str, Any]:
    values = raw_cfg.get(name, {})
    return values if isinstance(values, dict) else {}


def _slug(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("._-")


def resolve_training_semantics(raw_cfg: dict[str, Any]) -> str:
    algorithm = _section(raw_cfg, "algorithm")
    explicit = algorithm.get("training_semantics")
    if explicit is not None and str(explicit).strip():
        return _slug(explicit)
    mode = algorithm.get("mode")
    if mode is not None and str(mode).strip():
        return f"skrl_mappo_{_slug(mode)}"
    return DEFAULT_TRAINING_SEMANTICS


def sanitize_checkpoint_name(name: Any) -> str:
    raw = str(name).strip()
    if not raw:
        raise ValueError("Checkpoint name must not be empty.")
    if Path(raw).is_absolute() or PureWindowsPath(raw).is_absolute():
        raise ValueError("Checkpoint name must be a file name, not an absolute path.")
    if "/" in raw or "\\" in raw or ".." in raw:
        raise ValueError("Checkpoint name must not contain path separators or '..'.")

    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-")
    if not sanitized:
        raise ValueError("Checkpoint name must contain at least one safe character.")
    if not sanitized.endswith(".pt"):
        sanitized = f"{sanitized}.pt"
    return sanitized


def resolve_checkpoint_name(raw_cfg: dict[str, Any], config_path: str | Path) -> str:
    experiment = _section(raw_cfg, "experiment")
    explicit = experiment.get("checkpoint_name")
    if explicit is not None and str(explicit).strip():
        return sanitize_checkpoint_name(explicit)

    fallback_name = experiment.get("name") or Path(config_path).stem
    return sanitize_checkpoint_name(f"{fallback_name}_skrl_mappo.pt")


def observation_interface_metadata(cfg: Any) -> dict[str, Any]:
    return {
        "observation_schema_version": str(cfg.observation.schema_version),
        "actor_obs_dim": int(cfg.actor_obs_dim),
        "critic_state_dim": int(cfg.critic_state_dim),
    }


def validate_checkpoint_compatibility(checkpoint: dict[str, Any], cfg: Any) -> dict[str, Any]:
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise CheckpointCompatibilityError(
            "Checkpoint is missing metadata and is incompatible with "
            f"observation schema {cfg.observation.schema_version!r}."
        )

    expected = observation_interface_metadata(cfg)
    missing = sorted(key for key in expected if key not in metadata)
    if missing:
        raise CheckpointCompatibilityError(
            "Checkpoint is missing observation interface metadata: "
            f"{', '.join(missing)}. Old checkpoints are not auto-migrated."
        )

    mismatches = {
        key: (metadata[key], value)
        for key, value in expected.items()
        if metadata[key] != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} (expected {expected_value!r})"
            for key, (actual, expected_value) in mismatches.items()
        )
        raise CheckpointCompatibilityError(
            f"Checkpoint observation interface is incompatible: {details}."
        )
    return metadata
