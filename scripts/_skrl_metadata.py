"""Lightweight SKRL run metadata helpers."""

from __future__ import annotations

import re
from pathlib import Path, PureWindowsPath
from typing import Any


DEFAULT_TRAINING_SEMANTICS = "skrl_mappo_smoke"
DEFAULT_ACTOR_ARCHITECTURE = "mlp_v1"
DEFAULT_CRITIC_ARCHITECTURE = "mlp_v1"


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
    metadata = {
        "observation_schema_version": str(cfg.observation.schema_version),
        "actor_obs_dim": int(cfg.actor_obs_dim),
        "critic_state_dim": int(cfg.critic_state_dim),
    }
    if str(cfg.observation.schema_version) in {
        "ego_v9_multiscale_intent",
        "ego_v10_multiscale_diff_intent",
        "ego_v11_multiscale_site_belief",
    }:
        metadata.update(
            {
                "action_type": str(cfg.planner.action_type),
                "action_dim": int(cfg.planner.action_dim),
                "action_distribution": "categorical",
            }
        )
    return metadata


def _metadata_value_with_default(
    metadata: dict[str, Any],
    key: str,
    default: str,
) -> str:
    value = metadata.get(key, default)
    return str(value)


def validate_checkpoint_compatibility(
    checkpoint: dict[str, Any],
    cfg: Any,
    *,
    expected_actor_architecture: str | None = None,
    expected_critic_architecture: str | None = None,
) -> dict[str, Any]:
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
    if expected_actor_architecture is not None:
        actual_actor = _metadata_value_with_default(
            metadata,
            "actor_architecture",
            DEFAULT_ACTOR_ARCHITECTURE,
        )
        if actual_actor != expected_actor_architecture:
            raise CheckpointCompatibilityError(
                "Checkpoint actor architecture is incompatible: "
                f"actor_architecture={actual_actor!r} "
                f"(expected {expected_actor_architecture!r})."
            )
    if expected_critic_architecture is not None:
        actual_critic = _metadata_value_with_default(
            metadata,
            "critic_architecture",
            DEFAULT_CRITIC_ARCHITECTURE,
        )
        if actual_critic != expected_critic_architecture:
            raise CheckpointCompatibilityError(
                "Checkpoint critic architecture is incompatible: "
                f"critic_architecture={actual_critic!r} "
                f"(expected {expected_critic_architecture!r})."
            )
    return metadata
