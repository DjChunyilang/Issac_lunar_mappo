"""Native fixed-size kernels. Build failure is explicit; no device fallback."""
import ctypes
from functools import lru_cache
import hashlib
from pathlib import Path
import subprocess

import numpy as np
import torch


@lru_cache(maxsize=2)
def library(cuda=True):
    from torch.utils.cpp_extension import CUDA_HOME, get_default_build_root
    source = Path(__file__).with_suffix('.cu' if cuda else '.cpp')
    capability = torch.cuda.get_device_capability() if cuda else None
    arch = f'sm_{capability[0]}{capability[1]}' if cuda else 'cpu'
    identity = hashlib.sha256(source.read_bytes() + source.with_suffix('.h').read_bytes() + arch.encode()).hexdigest()[:20]
    build = Path(get_default_build_root()) / ('lunar_native_' + identity)
    build.mkdir(parents=True, exist_ok=True)
    target = build / 'kernels.so'
    if not target.exists():
        # FileBaton avoids two planner processes compiling the same library.
        from torch.utils.file_baton import FileBaton
        baton = FileBaton(str(build / 'lock'))
        if baton.try_acquire():
            try:
                command = ([str(Path(CUDA_HOME) / 'bin/nvcc'), str(source), '-o', str(target),
                                '-shared', '-Xcompiler', '-fPIC', '-O3', '--fmad=false',
                                '-std=c++17', '-arch=' + arch] if cuda else
                           ['c++', str(source), '-o', str(target), '-shared', '-fPIC', '-O3',
                            '-ffp-contract=off', '-std=c++17'])
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode:
                    raise RuntimeError('Native kernel build failed:\n' + result.stderr)
            finally:
                baton.release()
        else:
            baton.wait()
    lib = ctypes.CDLL(str(target))
    ptr, integer = ctypes.c_void_p, ctypes.c_int64
    if cuda:
        lib.project_cuda.argtypes = [ptr, ptr, ptr, integer, ptr, ptr]
        lib.rollout_cuda.argtypes = [ptr, ptr, ptr, ptr, integer, integer, ptr, ctypes.c_double, ctypes.c_double,
                                    ptr, ptr, ptr, integer, ptr, ptr, integer, integer, integer, integer,
                                    ctypes.c_double, ctypes.c_double, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr]
        lib.lookup_cuda.argtypes = [ptr, integer, integer, ptr, ptr, ptr, integer, ptr, ptr,
                               integer, integer, integer, integer, ctypes.c_double, ctypes.c_double,
                               ptr, ptr, ptr, ptr, ptr]
    else:
        lib.project_cpu.argtypes = [ptr, ptr, ptr, integer, ptr]
        lib.brake_sequence_cpu.argtypes = [ptr, ptr, integer, ptr]
        lib.control_error_cpu.argtypes = [ptr, ptr, ptr, ctypes.c_double, ctypes.c_double]
        lib.sequence_valid_cpu.argtypes = [ptr, ptr, integer, ptr, ctypes.c_double]
        lib.memory_merge_cpu.argtypes = [ptr, ptr, ptr, ptr, integer, ptr, ptr, ptr, integer,
                                        ctypes.c_double, integer, ptr, ptr, ptr, ptr, ptr]
    return lib


def memory_merge_cpu(previous, xy, features, timestamp, spec):
    """Own fresh output buffers; old snapshots remain immutable across ticks."""
    xy = np.ascontiguousarray(xy, dtype=np.float64)
    features = np.ascontiguousarray(features, dtype=np.float64)
    resolution = spec.perception.memory_resolution
    # Match np.rint's ties-to-even rather than C++ round's away-from-zero.
    if previous._memory_cells is None:
        previous._memory_cells = np.rint(previous._xy / resolution).astype(np.int64)
    oldcells = previous._memory_cells
    cells = np.rint(xy / resolution).astype(np.int64)
    capacity = spec.perception.capacity
    size = min(capacity, len(previous._xy) + len(xy))
    outxy = np.empty((size, 2)); outf = np.empty((size, 5)); outt = np.empty(size)
    counts = np.zeros(2, dtype=np.int64)
    t = spec.terrain
    limits = np.array([t.minimum_traversability, t.driving_slope, t.roughness_rms, t.step_height])
    status = library(False).memory_merge_cpu(previous._xy.ctypes.data, previous._features.ctypes.data,
        previous._timestamps.ctypes.data, oldcells.ctypes.data, len(oldcells), xy.ctypes.data,
        features.ctypes.data, cells.ctypes.data, len(xy), timestamp, capacity, limits.ctypes.data,
        outxy.ctypes.data, outf.ctypes.data, outt.ctypes.data, counts.ctypes.data)
    if status == 1:
        raise ValueError('Capacity cannot retain the complete direct frame')
    if status == 2:
        return None  # Scalar NumPy decides near-threshold/tied gradient comparisons.
    if status != 0:
        raise RuntimeError(f'Native memory merge failed: {status}')
    count, fresh = counts
    return outxy[:count], outf[:count], outt[:count], int(fresh)


@lru_cache(maxsize=8)
def parameters(spec):
    v = spec.vehicle; dt = spec.grid.dt
    return np.array([v.linear_acceleration*dt, v.angular_acceleration*dt,
                     v.linear_deceleration*dt, v.angular_deceleration*dt,
                     v.reverse_limit, v.forward_limit, v.yaw_limit, v.low_speed_threshold,
                     v.low_speed_yaw_limit, v.effective_track/2, v.wheel_radius*v.wheel_limit,
                     v.wheel_radius*v.wheel_acceleration*dt], dtype=np.float64)


def project_cuda(desired, previous, spec):
    result = _project_cuda_unchecked(desired, previous, spec)
    if not torch.isfinite(result).all():
        raise ValueError('Invalid previous command or empty physical actuator feasible set')
    return result


def _project_cuda_unchecked(desired, previous, spec):
    """Internal FP64 launch; the owning rollout must check all output values."""
    desired, previous = desired.contiguous(), previous.contiguous()
    result = torch.empty_like(desired)
    status = library().project_cuda(desired.data_ptr(), previous.data_ptr(), result.data_ptr(),
                                    desired.numel()//2, parameters(spec).ctypes.data,
                                    torch.cuda.current_stream(desired.device).cuda_stream)
    if status:
        raise RuntimeError(f'Native actuator CUDA launch failed: {status}')
    return result


def project_cpu(desired, previous, spec):
    desired = np.ascontiguousarray(desired, dtype=np.float64)
    previous = np.ascontiguousarray(previous, dtype=np.float64)
    result = np.empty_like(desired)
    status = library(False).project_cpu(desired.ctypes.data, previous.ctypes.data, result.ctypes.data,
                                        desired.size//2, parameters(spec).ctypes.data)
    if status:
        raise ValueError('Invalid previous command or empty physical actuator feasible set')
    return result


def control_error_cpu(control, previous, spec, tolerance):
    control = np.ascontiguousarray(control, dtype=np.float64)
    previous = np.ascontiguousarray(previous, dtype=np.float64)
    code = library(False).control_error_cpu(control.ctypes.data, previous.ctypes.data,
        parameters(spec).ctypes.data, spec.vehicle.wheel_radius, tolerance)
    return ('', 'directional_speed_limit', 'yaw_mode_limit', 'wheel_speed_limit',
            'directional_rate_limit', 'wheel_acceleration_limit')[code]


def sequence_valid_cpu(controls, previous, spec):
    controls = np.ascontiguousarray(controls, dtype=np.float64)
    previous = np.ascontiguousarray(previous, dtype=np.float64)
    return bool(library(False).sequence_valid_cpu(controls.ctypes.data,previous.ctypes.data,len(controls),
                                                parameters(spec).ctypes.data,spec.vehicle.wheel_radius))


def lookup_cuda(batch, flat, per_env):
    flat = flat.contiguous()
    features = torch.empty((len(flat), 5), dtype=torch.float64, device=flat.device)
    masks = [torch.empty(len(flat), dtype=torch.bool, device=flat.device) for _ in range(3)]
    status = library().lookup_cuda(flat.data_ptr(), len(flat), per_env,
        batch.keys.data_ptr(), batch.first.data_ptr(), batch.counts.data_ptr(), len(batch.keys),
        batch.xy.data_ptr(), batch.features.data_ptr(), int(batch.origin[0]), int(batch.origin[1]),
        batch.nx, batch.ny, batch.cell, batch.support, features.data_ptr(),
        *(mask.data_ptr() for mask in masks), torch.cuda.current_stream(flat.device).cuda_stream)
    if status:
        raise RuntimeError(f'Native terrain CUDA launch failed: {status}')
    return features, *masks


def rollout_cuda(requests, raw, terrain, spec):
    """All 40 physical steps in one launch for uniform-attribute owned views.

    Support still queries every actual local point set, including unknown holes.
    Nonuniform ties retain the authoritative CPU source-order query path.
    """
    raw = raw.double().contiguous()
    def tensor(value):
        return torch.as_tensor(value, device=raw.device, dtype=torch.float64)
    pose = tensor(np.stack([r.ego_pose for r in requests]))
    previous = tensor(np.stack([r.previous_control for r in requests]))
    caps = tensor([spec.vehicle.forward_limit if r.intent.speed_cap is None else r.intent.speed_cap for r in requests])
    b, k = raw.shape[:2]
    states = torch.empty((b,k,41,3), device=raw.device, dtype=torch.float64)
    controls = torch.empty_like(raw); velocity = torch.empty_like(raw)
    numeric = torch.empty((b,k), device=raw.device, dtype=torch.bool)
    known, ambiguous, boundary = [torch.empty((b,k,41), device=raw.device, dtype=torch.bool) for _ in range(3)]
    features = torch.empty((b,k,41,5), device=raw.device, dtype=torch.float64)
    status = library().rollout_cuda(raw.data_ptr(),pose.data_ptr(),previous.data_ptr(),caps.data_ptr(),b*k,k,
        parameters(spec).ctypes.data,spec.grid.dt,spec.terrain.slope_speed_scale,
        terrain.keys.data_ptr(),terrain.first.data_ptr(),terrain.counts.data_ptr(),len(terrain.keys),
        terrain.xy.data_ptr(),terrain.features.data_ptr(),int(terrain.origin[0]),int(terrain.origin[1]),
        terrain.nx,terrain.ny,terrain.cell,terrain.support,states.data_ptr(),controls.data_ptr(),velocity.data_ptr(),
        numeric.data_ptr(),known.data_ptr(),ambiguous.data_ptr(),boundary.data_ptr(),features.data_ptr(),
        torch.cuda.current_stream(raw.device).cuda_stream)
    if status:
        raise RuntimeError(f'Native rollout CUDA launch failed: {status}')
    if not torch.isfinite(states).all():
        raise ValueError('Native rollout violates physical actuator contract')
    if boundary.any():
        return None  # Exact CPU support-boundary handling in the stepwise path.
    terrain.ambiguous_queries += int(ambiguous.sum())
    terrain._rollout_query = (states, features, known, ambiguous, states._version, known._version)
    return states, controls, velocity, numeric, known


def brake_sequence_cpu(previous,spec,max_steps):
    previous=np.ascontiguousarray(previous,dtype=np.float64)
    result=np.empty((max_steps,2),dtype=np.float64)
    count=library(False).brake_sequence_cpu(previous.ctypes.data,result.ctypes.data,max_steps,parameters(spec).ctypes.data)
    if count<0:raise ValueError('Invalid previous command or empty physical actuator feasible set')
    if count==0:raise ValueError('Joint actuator stop exceeds diagnostic bound')
    return result[:count]
