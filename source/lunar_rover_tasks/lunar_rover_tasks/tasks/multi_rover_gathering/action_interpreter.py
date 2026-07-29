"""Map normalized actor actions to local and world subgoals."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import PlannerCfg
from lunar_rover_tasks.utils.math_utils import rotate_2d


@dataclass(slots=True)
class DecodedAction:
    clipped_normalized: torch.Tensor
    physical: torch.Tensor
    local_subgoal_xy: torch.Tensor
    world_subgoal: torch.Tensor


@dataclass(slots=True)
class FormationCenterCorrection:
    """Result of a bounded common formation-centre subgoal translation."""

    decoded: DecodedAction
    active: torch.Tensor
    offset_xy: torch.Tensor


@dataclass(slots=True)
class TerminalSlotCapture:
    """Result of blending a terminal subgoal toward fixed assigned slots."""

    decoded: DecodedAction
    active: torch.Tensor


@dataclass(slots=True)
class FlatGeometryCapture:
    """Result of contracting a non-terminal flat formation in place."""

    decoded: DecodedAction
    active: torch.Tensor


def clip_action(action: torch.Tensor) -> torch.Tensor:
    return torch.clamp(action, -1.0, 1.0)


def scale_action(action: torch.Tensor, cfg: PlannerCfg) -> torch.Tensor:
    clipped = clip_action(action)
    rho = 0.5 * (clipped[..., 0] + 1.0) * cfg.rho_max
    beta = clipped[..., 1] * cfg.beta_max
    return torch.stack((rho, beta), dim=-1)


def polar_to_local_subgoal(physical_action: torch.Tensor) -> torch.Tensor:
    rho = physical_action[..., 0]
    beta = physical_action[..., 1]
    return torch.stack((rho * torch.cos(beta), rho * torch.sin(beta)), dim=-1)


def local_to_world_subgoal(
    positions: torch.Tensor,
    yaws: torch.Tensor,
    local_subgoal_xy: torch.Tensor,
) -> torch.Tensor:
    world_delta = rotate_2d(local_subgoal_xy, yaws)
    z = torch.zeros_like(world_delta[..., :1])
    return positions + torch.cat((world_delta, z), dim=-1)


def decode_action(
    action: torch.Tensor,
    positions: torch.Tensor,
    yaws: torch.Tensor,
    cfg: PlannerCfg,
) -> DecodedAction:
    clipped = clip_action(action)
    physical = scale_action(clipped, cfg)
    local_subgoal_xy = polar_to_local_subgoal(physical)
    world_subgoal = local_to_world_subgoal(positions, yaws, local_subgoal_xy)
    return DecodedAction(
        clipped_normalized=clipped,
        physical=physical,
        local_subgoal_xy=local_subgoal_xy,
        world_subgoal=world_subgoal,
    )


def apply_formation_center_correction(
    decoded: DecodedAction,
    *,
    centroid_xy: torch.Tensor,
    dmax: torch.Tensor,
    dispersion: torch.Tensor,
    formation_center_xy: torch.Tensor,
    dmax_threshold: float,
    dispersion_threshold: float,
    enabled: bool,
    activation_dmax_multiplier: float,
    activation_dispersion_multiplier: float,
    max_offset: float,
    gain: float,
    flatness_ok: torch.Tensor | None = None,
    require_flatness_failure: bool = False,
) -> FormationCenterCorrection:
    """Bias all rover subgoals by a bounded shared centre translation.

    All rover subgoals receive the same world-frame offset, preserving the
    symmetric slot formation and its intended pairwise separation. The outer
    loop only activates near terminal geometry and does not affect actor input
    or the success predicate.
    """
    if centroid_xy.ndim != 2 or centroid_xy.shape[-1] != 2:
        raise ValueError("centroid_xy must have shape [num_envs, 2].")
    if formation_center_xy.shape != centroid_xy.shape:
        raise ValueError("formation_center_xy must match centroid_xy shape.")
    if dmax.shape != centroid_xy.shape[:1] or dispersion.shape != centroid_xy.shape[:1]:
        raise ValueError("dmax and dispersion must have shape [num_envs].")
    if flatness_ok is not None and flatness_ok.shape != dmax.shape:
        raise ValueError("flatness_ok must have shape [num_envs].")
    if decoded.world_subgoal.shape[0] != centroid_xy.shape[0]:
        raise ValueError("decoded subgoals must share the centroid batch dimension.")
    if max_offset < 0.0:
        raise ValueError("max_offset must be non-negative.")
    if not 0.0 <= gain <= 1.0:
        raise ValueError("gain must be in [0, 1].")

    active = torch.zeros_like(dmax, dtype=torch.bool)
    offset_xy = torch.zeros_like(centroid_xy)
    if not enabled or max_offset == 0.0 or gain == 0.0:
        return FormationCenterCorrection(decoded=decoded, active=active, offset_xy=offset_xy)

    active = (dmax <= float(dmax_threshold) * float(activation_dmax_multiplier)) & (
        dispersion
        <= float(dispersion_threshold) * float(activation_dispersion_multiplier)
    )
    if require_flatness_failure:
        if flatness_ok is None:
            raise ValueError("flatness_ok is required when flatness gating is enabled.")
        active = active & ~flatness_ok
    center_error = formation_center_xy - centroid_xy
    distance = torch.linalg.norm(center_error, dim=-1, keepdim=True)
    bounded_error = center_error * (
        distance.clamp(max=float(max_offset)) / distance.clamp_min(1.0e-6)
    )
    offset_xy = bounded_error * float(gain) * active[:, None]
    world_subgoal = decoded.world_subgoal.clone()
    world_subgoal[..., :2] += offset_xy[:, None, :]
    return FormationCenterCorrection(
        decoded=DecodedAction(
            clipped_normalized=decoded.clipped_normalized,
            physical=decoded.physical,
            local_subgoal_xy=decoded.local_subgoal_xy,
            world_subgoal=world_subgoal,
        ),
        active=active,
        offset_xy=offset_xy,
    )


def apply_terminal_slot_capture(
    decoded: DecodedAction,
    *,
    gather_slot_points: torch.Tensor,
    dmax: torch.Tensor,
    dispersion: torch.Tensor,
    dmax_threshold: float,
    dispersion_threshold: float,
    enabled: bool,
    activation_dmax_multiplier: float,
    activation_dispersion_multiplier: float,
    blend: float,
) -> TerminalSlotCapture:
    """Blend terminal subgoals to per-rover assigned formation slots.

    The output target stays per-rover: it is never replaced by a shared
    centroid. Safety projection still processes the resulting controls before
    integration, while the success gate remains an independent actual-centroid
    and flatness test.
    """
    if gather_slot_points.shape != decoded.world_subgoal.shape:
        raise ValueError("gather_slot_points must match decoded world subgoal shape.")
    if dmax.ndim != 1 or dispersion.shape != dmax.shape:
        raise ValueError("dmax and dispersion must have shape [num_envs].")
    if dmax.shape[0] != decoded.world_subgoal.shape[0]:
        raise ValueError("dmax must share decoded subgoal batch dimension.")
    if not 0.0 <= blend <= 1.0:
        raise ValueError("blend must be in [0, 1].")

    active = torch.zeros_like(dmax, dtype=torch.bool)
    if not enabled or blend == 0.0:
        return TerminalSlotCapture(decoded=decoded, active=active)
    active = (dmax <= float(dmax_threshold) * float(activation_dmax_multiplier)) & (
        dispersion
        <= float(dispersion_threshold) * float(activation_dispersion_multiplier)
    )
    captured_subgoal = torch.lerp(decoded.world_subgoal, gather_slot_points, float(blend))
    world_subgoal = torch.where(
        active[:, None, None],
        captured_subgoal,
        decoded.world_subgoal,
    )
    return TerminalSlotCapture(
        decoded=DecodedAction(
            clipped_normalized=decoded.clipped_normalized,
            physical=decoded.physical,
            local_subgoal_xy=decoded.local_subgoal_xy,
            world_subgoal=world_subgoal,
        ),
        active=active,
    )


def apply_flat_geometry_capture(
    decoded: DecodedAction,
    *,
    gather_slot_points: torch.Tensor,
    centroid_xy: torch.Tensor,
    dmax: torch.Tensor,
    dispersion: torch.Tensor,
    flatness_ok: torch.Tensor,
    dmax_threshold: float,
    dispersion_threshold: float,
    enabled: bool,
    activation_dmax_multiplier: float,
    activation_dispersion_multiplier: float,
    blend: float,
) -> FlatGeometryCapture:
    """Contract toward fixed slots anchored at the current flat centroid.

    This is deliberately different from ``apply_terminal_slot_capture``:
    the slot offsets are retained, but their centre is the *actual current*
    centroid instead of the terrain-search point.  It can therefore shrink
    pairwise spread without intentionally translating an already-flat
    footprint.  It only operates while flatness has passed and the geometric
    success condition has not yet passed.
    """
    if gather_slot_points.shape != decoded.world_subgoal.shape:
        raise ValueError("gather_slot_points must match decoded world subgoal shape.")
    if centroid_xy.shape != (decoded.world_subgoal.shape[0], 2):
        raise ValueError("centroid_xy must have shape [num_envs, 2].")
    if dmax.ndim != 1 or dispersion.shape != dmax.shape or flatness_ok.shape != dmax.shape:
        raise ValueError("dmax, dispersion, and flatness_ok must have shape [num_envs].")
    if not 0.0 <= blend <= 1.0:
        raise ValueError("blend must be in [0, 1].")

    active = torch.zeros_like(dmax, dtype=torch.bool)
    if not enabled or blend == 0.0:
        return FlatGeometryCapture(decoded=decoded, active=active)

    near_terminal = (dmax <= float(dmax_threshold) * float(activation_dmax_multiplier)) & (
        dispersion
        <= float(dispersion_threshold) * float(activation_dispersion_multiplier)
    )
    geometry_complete = (dmax <= float(dmax_threshold)) & (
        dispersion <= float(dispersion_threshold)
    )
    active = flatness_ok & near_terminal & ~geometry_complete
    slot_offsets = gather_slot_points[..., :2] - gather_slot_points[..., :2].mean(
        dim=1, keepdim=True
    )
    anchored_slots = decoded.world_subgoal.clone()
    anchored_slots[..., :2] = centroid_xy[:, None, :] + slot_offsets
    captured_subgoal = torch.lerp(decoded.world_subgoal, anchored_slots, float(blend))
    world_subgoal = torch.where(
        active[:, None, None],
        captured_subgoal,
        decoded.world_subgoal,
    )
    return FlatGeometryCapture(
        decoded=DecodedAction(
            clipped_normalized=decoded.clipped_normalized,
            physical=decoded.physical,
            local_subgoal_xy=decoded.local_subgoal_xy,
            world_subgoal=world_subgoal,
        ),
        active=active,
    )
