"""Pure local prediction and feasibility checks shared by versioned adapters."""

import time
import numpy as np
from .contracts import DT, N, TimedTrajectory


def dynamics(pose, control, feature, slope_scale):
    v, w = control
    scale = np.clip(
        feature[4]
        * np.exp(-abs(feature[1] * np.cos(pose[2]) + feature[2] * np.sin(pose[2])) * slope_scale),
        0.25,
        1.0,
    )
    v, w = v * scale, w * scale
    mid = pose[2] + 0.5 * DT * w
    return pose + DT * np.array([v * np.cos(mid), v * np.sin(mid), w]), np.array([v, w])


def rollout(pose, controls, memory, now, slope_scale, *, flat=False):
    states, velocities = [pose.copy()], []
    for u in controls:
        features, _, _ = memory.lookup(states[-1][:2])
        state, velocity = dynamics(
            states[-1], u, np.array([0, 0, 0, 0, 1]) if flat else features[0], slope_scale
        )
        states.append(state)
        velocities.append(velocity)
    return TimedTrajectory(
        np.array(states),
        np.array(controls),
        np.array(velocities),
        now + np.arange(N + 1) * DT,
        now,
    )


def validate_trajectory(plan, memory, neighbors, cap, previous_control):
    if not all(
        np.isfinite(a).all() for a in (plan.states, plan.controls, plan.velocities, plan.times)
    ):
        return "nonfinite"
    u = plan.controls
    if (abs(u[:, 0]) > cap + 1e-4).any() or (abs(u[:, 1]) > 2.4001).any():
        return "speed_limit"
    if (abs(u[:, 0, None] + u[:, 1, None] * np.array([-0.188, 0.188])) > 18 * 0.098 + 1e-4).any():
        return "wheel_limit"
    if (abs(np.diff(np.vstack([previous_control, u]), axis=0)) > np.array([0.3001, 0.6001])).any():
        return "slew_limit"
    # Swept check at 0.025 s (not just shooting nodes), also for neighbor crossing.
    for alpha in np.linspace(0, 1, 9):
        points = (1 - alpha) * plan.states[:-1, :2] + alpha * plan.states[1:, :2]
        if not memory.safe(points).all():
            return "unknown_or_impassable"
        for neighbor in neighbors:
            peer = (1 - alpha) * neighbor.positions[:-1] + alpha * neighbor.positions[1:]
            margin = (1 - alpha) * neighbor.radii[:-1] + alpha * neighbor.radii[1:]
            # Add a sub-sampling Lipschitz allowance for relative displacement.
            sweep = (
                np.linalg.norm(
                    np.diff(plan.states[:, :2], axis=0) - np.diff(neighbor.positions, axis=0),
                    axis=1,
                )
                / 16
            )
            if (np.linalg.norm(points - peer, axis=1) < margin + sweep - 1e-4).any():
                return "neighbor_swept_collision"
    return ""


def stopped_prefix(
    pose,
    controls,
    memory,
    neighbors,
    cap,
    previous_control,
    now,
    deadline,
    slope_scale,
    *,
    flat=False,
):
    """Retain optimized controls only; append a rate-limited, validated stop.

    This neither searches for a new goal nor silently substitutes a route.
    Validate the complete horizon including the stationary tail and peers.
    """
    for length in range(N - 1, 0, -1):
        if time.monotonic() >= deadline:
            break
        candidate = controls.copy()
        last = candidate[length - 1].copy()
        for k in range(length, N):
            last = last + np.clip(-last, [-0.3, -0.6], [0.3, 0.6])
            candidate[k] = last
        if np.max(abs(candidate[-1])) > 1e-6:
            continue
        trajectory = rollout(pose, candidate, memory, now, slope_scale, flat=flat)
        if not validate_trajectory(trajectory, memory, neighbors, cap, previous_control):
            return trajectory, length
    return None, 0


def brake_controls(previous):
    last = np.asarray(previous, dtype=float).copy()
    controls = []
    for _ in range(N):
        last = last + np.clip(-last, [-0.3, -0.6], [0.3, 0.6])
        controls.append(last.copy())
    return np.array(controls)


def limit_control(desired, previous, cap=1.15):
    """Convex actuator mapping; not a collision/terrain safety projection."""
    desired = np.clip(desired, [-cap, -2.4], [cap, 2.4])
    wheel = np.max(abs(desired[0] + np.array([-0.188, 0.188]) * desired[1]))
    desired = desired * min(1.0, 18 * 0.098 / max(wheel, 1e-12))
    delta = desired - previous
    alpha = min(1.0, *(np.array([0.3, 0.6]) / np.maximum(abs(delta), 1e-12)))
    return previous + alpha * delta


def shifted_sequence(controls, elapsed_steps):
    k = max(0, min(N, elapsed_steps))
    result = list(np.asarray(controls)[k:].copy())
    last = result[-1].copy() if result else np.asarray(controls)[-1].copy()
    for _ in range(k):
        last = last + np.clip(-last, [-0.3, -0.6], [0.3, 0.6])
        result.append(last.copy())
    return np.asarray(result)
