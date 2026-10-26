"""Profile-bound actuator and midpoint model; independent of legacy constants.

Actuator projection is not terrain or collision certification. The feasible set
is a union of three convex polygons, because low-speed yaw limits differ.
NumPy is the scalar reference; Torch evaluates all batch/sample members at once.
"""

from dataclasses import dataclass
import math

import numpy as np
import torch

from .contracts import TimedTrajectory
from .upstream_spec import UpstreamSpec


def _finite(value, shape, name):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} requires finite shape {shape}")
    return value


def wheel_rates(control, spec):
    control = np.asarray(control, dtype=float)
    if control.shape[-1:] != (2,) or not np.isfinite(control).all():
        raise ValueError("Invalid actuator command")
    return (control[..., :1] + control[..., 1:] * np.array(
        [-spec.vehicle.effective_track / 2, spec.vehicle.effective_track / 2]
    )) / spec.vehicle.wheel_radius


def control_error(control, previous, spec, *, tolerance=1e-8):
    u = _finite(control, (2,), "control")
    p = _finite(previous, (2,), "previous control")
    from .upstream_native import control_error_cpu
    return control_error_cpu(u, p, spec, tolerance)


def _control_error_reference(u, p, spec, tolerance=1e-8):
    v = spec.vehicle; dt = spec.grid.dt
    if u[0] < -v.reverse_limit - tolerance or u[0] > v.forward_limit + tolerance:
        return "directional_speed_limit"
    yaw_cap = v.low_speed_yaw_limit if abs(u[0]) <= v.low_speed_threshold else v.yaw_limit
    if abs(u[1]) > yaw_cap + tolerance:
        return "yaw_mode_limit"
    if np.max(abs(wheel_rates(u, spec))) > v.wheel_limit + tolerance:
        return "wheel_speed_limit"
    for index, (acc, dec) in enumerate(((v.linear_acceleration, v.linear_deceleration),
                                       (v.angular_acceleration, v.angular_deceleration))):
        lo, hi = _direction_bounds(p[index], acc * dt, dec * dt)
        if u[index] < lo - tolerance or u[index] > hi + tolerance:
            return "directional_rate_limit"
    if np.max(abs(wheel_rates(u, spec) - wheel_rates(p, spec))) > v.wheel_acceleration * dt + tolerance:
        return "wheel_acceleration_limit"
    return ""


def _direction_bounds(previous, acceleration, deceleration):
    if previous > 0:
        return max(0., previous - deceleration), previous + acceleration
    if previous < 0:
        return previous - acceleration, min(0., previous + deceleration)
    return -acceleration, acceleration


def _normals(spec):
    half_track = spec.vehicle.effective_track / 2
    wheels = np.array([[1., -half_track], [1., half_track]])
    return np.vstack([np.eye(2), -np.eye(2), wheels, -wheels, wheels, -wheels])


def _polygon(previous, spec, mode, epsilon):
    v = spec.vehicle; dt = spec.grid.dt
    vlo, vhi = _direction_bounds(previous[0], v.linear_acceleration * dt, v.linear_deceleration * dt)
    wlo, whi = _direction_bounds(previous[1], v.angular_acceleration * dt, v.angular_deceleration * dt)
    vlo, vhi = max(vlo, -v.reverse_limit), min(vhi, v.forward_limit)
    if mode == 0:
        vlo, vhi = max(vlo, -v.low_speed_threshold), min(vhi, v.low_speed_threshold)
        cap = v.low_speed_yaw_limit
    elif mode == 1:
        vlo = max(vlo, v.low_speed_threshold + epsilon); cap = v.yaw_limit
    else:
        vhi = min(vhi, -v.low_speed_threshold - epsilon); cap = v.yaw_limit
    wlo, whi = max(wlo, -cap), min(whi, cap)
    wheel = previous[0] + np.array([-1., 1.]) * (v.effective_track / 2) * previous[1]
    limit = v.wheel_radius * v.wheel_limit
    step = v.wheel_radius * v.wheel_acceleration * dt
    return _normals(spec), np.r_[vhi, whi, -vlo, -wlo, [limit] * 4, wheel + step, -wheel + step]


def limit_control_numpy(desired, previous, spec):
    """Closest normalized command across all three actuator polygons.

    A newly lowered local speed cap is handled by stop-corridor admission, not
    by instantly clipping an already moving vehicle's actuator state here.
    """
    if not isinstance(spec, UpstreamSpec):
        raise ValueError("Actuator requires an immutable UpstreamSpec")
    target = _finite(desired, (2,), "desired control")
    previous = _finite(previous, (2,), "previous control")
    if control_error(previous, previous, spec):
        raise ValueError("Previous command violates the physical actuator contract")
    from .upstream_native import project_cpu
    return project_cpu(target, previous, spec)


def _limit_control_numpy_reference(target, previous, spec):
    """Scalar reference retained for numerical regression of the native kernel."""
    scale = np.array([spec.vehicle.linear_acceleration, spec.vehicle.angular_acceleration]) * spec.grid.dt
    target = np.clip(target, [-spec.vehicle.reverse_limit, -spec.vehicle.yaw_limit],
                     [spec.vehicle.forward_limit, spec.vehicle.yaw_limit]) / scale
    # A feasible target has zero objective and is the exact global minimizer.
    # Use the original polygon tolerances, including the low-speed mode gap.
    polygons = []
    for mode in range(3):
        a, b = _polygon(previous, spec, mode, 1e-9)
        a = a * scale
        if (a @ target <= b + 1e-10).all():
            return target * scale
        polygons.append((a, b))
    candidates = []
    for a, b in polygons:
        points = [target]
        for i in range(len(b)):
            points.append(target - a[i] * ((a[i] @ target - b[i]) / (a[i] @ a[i])))
            for j in range(i):
                system = a[[j, i]]
                if abs(np.linalg.det(system)) > 1e-15:
                    points.append(np.linalg.solve(system, b[[j, i]]))
        for point in points:
            if (a @ point <= b + 1e-10).all():
                candidates.append((float(np.sum((point - target)**2)), point * scale))
    if not candidates:
        raise ValueError("Empty physical actuator feasible set")
    return min(candidates, key=lambda item: item[0])[1]


def limit_control_torch(desired, previous, spec):
    """Batched polygon projection. No loop over environments or samples."""
    if not isinstance(spec, UpstreamSpec):
        raise ValueError("Actuator requires an immutable UpstreamSpec")
    if desired.shape != previous.shape or desired.shape[-1:] != (2,):
        raise ValueError("Actuator tensors require matching [...,2] shapes")
    if desired.device != previous.device or desired.dtype != previous.dtype or desired.dtype not in (torch.float32, torch.float64):
        raise ValueError("Actuator dtype/device mismatch")
    if not torch.isfinite(desired).all() or not torch.isfinite(previous).all():
        raise ValueError("Nonfinite actuator tensors")
    if desired.is_cuda and desired.dtype == torch.float64 and not (desired.requires_grad or previous.requires_grad):
        from .upstream_native import project_cuda
        return project_cuda(desired, previous, spec)
    v = spec.vehicle; dt = spec.grid.dt
    def tensor(value):
        return torch.as_tensor(value, device=desired.device, dtype=desired.dtype)
    scale = tensor([v.linear_acceleration * dt, v.angular_acceleration * dt])
    acceleration = scale
    deceleration = tensor([v.linear_deceleration * dt, v.angular_deceleration * dt])
    zero = torch.zeros_like(previous)
    lo = torch.where(previous > 0, torch.maximum(zero, previous - deceleration), previous - acceleration)
    hi = torch.where(previous < 0, torch.minimum(zero, previous + deceleration), previous + acceleration)
    global_lo = tensor([-v.reverse_limit, -v.yaw_limit]); global_hi = tensor([v.forward_limit, v.yaw_limit])
    lo, hi = torch.maximum(lo, global_lo), torch.minimum(hi, global_hi)
    # Closed high-speed polygons are separated from the <= threshold low mode.
    epsilon = 2e-7 if desired.dtype == torch.float32 else 1e-9
    mode_lo = tensor([-v.low_speed_threshold, v.low_speed_threshold + epsilon, -v.reverse_limit])
    mode_hi = tensor([v.low_speed_threshold, v.forward_limit, -v.low_speed_threshold - epsilon])
    yaw_cap = tensor([v.low_speed_yaw_limit, v.yaw_limit, v.yaw_limit])
    vl = torch.maximum(lo[..., 0, None], mode_lo); vh = torch.minimum(hi[..., 0, None], mode_hi)
    wl = torch.maximum(lo[..., 1, None], -yaw_cap); wh = torch.minimum(hi[..., 1, None], yaw_cap)
    wheel = previous[..., :1] + previous[..., 1:] * tensor([-v.effective_track / 2, v.effective_track / 2])
    limit = v.wheel_radius * v.wheel_limit; step = v.wheel_radius * v.wheel_acceleration * dt
    bounds = torch.cat([torch.stack([vh, wh, -vl, -wl], dim=-1),
                        torch.full((*vl.shape, 4), limit, dtype=desired.dtype, device=desired.device),
                        (wheel + step).unsqueeze(-2).expand(*vl.shape, 2),
                        (-wheel + step).unsqueeze(-2).expand(*vl.shape, 2)], dim=-1)
    a = tensor(_normals(spec)) * scale
    target = torch.minimum(torch.maximum(desired, global_lo), global_hi) / scale
    target = target.unsqueeze(-2).expand(*vl.shape, 2)
    projection = target.unsqueeze(-2) - a * (
        ((target.unsqueeze(-2) * a).sum(-1) - bounds) / a.square().sum(-1)).unsqueeze(-1)
    pair = torch.triu_indices(len(a), len(a), offset=1, device=desired.device)
    aa, ab = a[pair[0]], a[pair[1]]
    determinant = aa[:, 0] * ab[:, 1] - aa[:, 1] * ab[:, 0]
    good = determinant.abs() > 1e-15
    aa, ab, determinant = aa[good], ab[good], determinant[good]
    ba, bb = bounds[..., pair[0][good]], bounds[..., pair[1][good]]
    intersections = torch.stack([(ba * ab[:, 1] - bb * aa[:, 1]) / determinant,
                                 (aa[:, 0] * bb - ab[:, 0] * ba) / determinant], dim=-1)
    candidates = torch.cat([target.unsqueeze(-2), projection, intersections], dim=-2)
    tolerance = 3e-7 if desired.dtype == torch.float32 else 1e-10
    feasible = ((candidates.unsqueeze(-2) * a).sum(-1) <= bounds.unsqueeze(-2) + tolerance).all(-1)
    cost = (candidates - target.unsqueeze(-2)).square().sum(-1)
    cost = torch.where(feasible, cost, torch.full_like(cost, float('inf')))
    costs = cost.flatten(-2); points = candidates.flatten(-3, -2)
    index = costs.argmin(-1)
    if not torch.isfinite(costs.gather(-1, index[..., None])).all():
        raise ValueError("Empty physical actuator feasible set")
    selected = points.gather(-2, index[..., None, None].expand(*index.shape, 1, 2)).squeeze(-2) * scale
    # Reject bad previous commands rather than silently using a different initial state.
    previous_cap = torch.where(previous[..., 0].abs() <= v.low_speed_threshold,
                               tensor(v.low_speed_yaw_limit), tensor(v.yaw_limit))
    if ((previous < global_lo - tolerance) | (previous > global_hi + tolerance)).any() or (
        previous[..., 1].abs() > previous_cap + tolerance).any() or (wheel.abs() > limit + tolerance).any():
        raise ValueError("Previous command violates the physical actuator contract")
    return selected


def midpoint_numpy(pose, control, spec, *, gain=1.):
    p = _finite(pose, (3,), "pose"); u = _finite(control, (2,), "control")
    if not np.isfinite(gain) or not .25 <= gain <= 1:
        raise ValueError("Terrain velocity gain outside [.25,1]")
    actual = u * gain
    mid = p[2] + .5 * spec.grid.dt * actual[1]
    result = p + spec.grid.dt * np.array([actual[0] * np.cos(mid), actual[0] * np.sin(mid), actual[1]])
    return result, actual


def terrain_gain_numpy(pose, features, spec):
    p = _finite(pose, (3,), "pose"); f = _finite(features, (5,), "terrain feature")
    along = f[1] * np.cos(p[2]) + f[2] * np.sin(p[2])
    return float(np.clip(f[4] * np.exp(-abs(along) * spec.terrain.slope_speed_scale), .25, 1.))


def terrain_gain_torch(pose, features, spec):
    if features.shape != (*pose.shape[:-1], 5) or not torch.isfinite(features).all():
        raise ValueError("Terrain features require finite [...,5]")
    if pose.dtype != features.dtype or pose.device != features.device:
        raise ValueError("Terrain feature device/dtype mismatch")
    return _terrain_gain_torch_values(pose, features, spec)


def _terrain_gain_torch_values(pose, features, spec):
    """Internal arithmetic shared with the checked public entry point."""
    along = features[..., 1] * pose[..., 2].cos() + features[..., 2] * pose[..., 2].sin()
    return (features[..., 4] * torch.exp(-along.abs() * spec.terrain.slope_speed_scale)).clamp(.25, 1.)


def midpoint_torch(pose, control, spec, *, gain=None):
    if pose.shape[-1:] != (3,) or control.shape != (*pose.shape[:-1], 2):
        raise ValueError("Midpoint tensors require [...,3] and [...,2]")
    if pose.dtype not in (torch.float32, torch.float64) or pose.dtype != control.dtype or pose.device != control.device or not torch.isfinite(pose).all() or not torch.isfinite(control).all():
        raise ValueError("Invalid midpoint tensor dtype/device/value")
    gain = torch.ones_like(control[..., 0]) if gain is None else gain
    if gain.shape != control.shape[:-1] or gain.dtype != pose.dtype or gain.device != pose.device or not torch.isfinite(gain).all() or ((gain < .25) | (gain > 1)).any():
        raise ValueError("Invalid terrain velocity gains")
    return _midpoint_torch_values(pose, control, spec, gain)


def _midpoint_torch_values(pose, control, spec, gain):
    """Internal arithmetic; callers retain finite/range validation ownership."""
    actual = control * gain[..., None]
    mid = pose[..., 2] + .5 * spec.grid.dt * actual[..., 1]
    delta = torch.stack([actual[..., 0] * mid.cos(), actual[..., 0] * mid.sin(), actual[..., 1]], dim=-1)
    return pose + spec.grid.dt * delta, actual


def brake_sequence(previous, spec, *, max_steps=512):
    """Actual applied-command stop, including a final stationary step."""
    last = _finite(previous, (2,), "previous control")
    if control_error(last,last,spec):raise ValueError("Previous command violates the physical actuator contract")
    from .upstream_native import brake_sequence_cpu
    return brake_sequence_cpu(last,spec,max_steps)


def rollout_numpy(pose, controls, spec, now=0., *, gain=1.):
    controls = np.asarray(controls, dtype=float)
    if controls.ndim != 2 or controls.shape[-1] != 2 or not np.isfinite(controls).all():
        raise ValueError("Controls require finite [T,2]")
    if not np.isfinite(now) or now < 0:
        raise ValueError("Invalid trajectory time origin")
    state = _finite(pose, (3,), "pose").copy(); states = [state]; velocities = []
    gains = None if callable(gain) else np.broadcast_to(gain, (len(controls),))
    if (len(controls) and not controls.any() and not np.signbit(controls).any()
            and getattr(gain, '_stationary_repeatable', False)):
        # Owned static terrain has the same gain at every identical pose.
        # Integrate once (including validation and signed-zero arithmetic),
        # then construct every original state/control/time sample.
        next_state, actual = midpoint_numpy(state, controls[0], spec, gain=gain(state))
        return TimedTrajectory(np.vstack([state, np.tile(next_state, (len(controls),1))]), controls.copy(),
            np.tile(actual, (len(controls),1)), now + np.arange(len(controls)+1)*spec.grid.dt, now)
    for i, command in enumerate(controls):
        local_gain = gain(state) if gains is None else gains[i]
        state, actual = midpoint_numpy(state, command, spec, gain=local_gain)
        states.append(state); velocities.append(actual)
    return TimedTrajectory(np.asarray(states), controls.copy(), np.asarray(velocities).reshape(-1, 2),
                           now + np.arange(len(controls) + 1) * spec.grid.dt, now)


def rollout_torch(pose, controls, previous, spec, *, gains=None, limit=True):
    if controls.ndim < 2 or controls.shape[-1] != 2 or controls.shape[:-2] != pose.shape[:-1]:
        raise ValueError("Rollout requires pose [...,3] and controls [...,T,2]")
    if pose.shape[-1:] != (3,) or previous.shape != (*pose.shape[:-1], 2):
        raise ValueError("Rollout pose/previous control shape mismatch")
    state = pose; last = previous; states = [state]; velocities = []; applied = []
    if gains is not None and gains.shape != controls.shape[:-1]:
        raise ValueError("Rollout gains require [...,T]")
    for t in range(controls.shape[-2]):
        command = limit_control_torch(controls[..., t, :], last, spec) if limit else controls[..., t, :]
        state, actual = midpoint_torch(state, command, spec, gain=None if gains is None else gains[..., t])
        states.append(state); velocities.append(actual); applied.append(command); last = command
    # Empty sequence remains defined for diagnostic batch boundaries.
    empty = controls[..., :0, :]
    return (torch.stack(states, dim=-2), torch.stack(applied, dim=-2) if applied else empty,
            torch.stack(velocities, dim=-2) if velocities else empty)


@dataclass(frozen=True)
class SafetyStopEnvelope:
    trajectory: TimedTrajectory
    delay_steps: int
    commitment_steps: int
    braking_steps: int
    stationary_steps: int
    fits_horizon: bool
    information_time: float
    activation_time: float
    conservative_center_distance: float


def stop_envelope(pose, previous, candidate, spec, *, information_time=0., latency=.5, commit=True, gain=1., pad_to_horizon=False):
    """Information-time reserve, never an instruction to add execution waiting.

    Each braking step holds its incoming command before the next decrement.
    The execution trajectory must instead start at actual activation time.
    """
    if not np.isfinite(latency) or latency < 0:
        raise ValueError("Invalid perception-to-execution latency")
    previous = _finite(previous, (2,), "previous control")
    candidate = _finite(candidate, (2,), "candidate control")
    if control_error(previous, previous, spec) or control_error(candidate, previous, spec):
        raise ValueError("Stop reserve requires a legal actuator transition")
    delay_steps = math.ceil(latency / spec.grid.dt - 1e-12)
    commitment = int(bool(commit))
    last = candidate.copy() if commit else previous.copy()
    actual_brake = brake_sequence(last, spec)
    # The last applied braking command is zero. Shift incoming commands for
    # the conservative coverage bound, then append the mandatory static tail.
    incoming = np.vstack([last, actual_brake[:-1]])
    if not last.any():
        incoming = np.empty((0, 2))
    controls = np.vstack([np.tile(previous, (delay_steps, 1)),
                          np.tile(candidate, (commitment, 1)), incoming, np.zeros((1, 2))])
    reserved_steps = len(controls)
    if pad_to_horizon and reserved_steps <= spec.grid.steps:
        controls = np.vstack([controls, np.zeros((spec.grid.steps-reserved_steps, 2))])
    trajectory = rollout_numpy(pose, controls, spec, information_time, gain=gain)
    distance = np.linalg.norm(np.diff(trajectory.states[:, :2], axis=0), axis=1).sum()
    return SafetyStopEnvelope(trajectory, delay_steps, commitment, len(incoming), 1,
                             reserved_steps <= spec.grid.steps, information_time,
                             information_time + delay_steps * spec.grid.dt, float(distance))
