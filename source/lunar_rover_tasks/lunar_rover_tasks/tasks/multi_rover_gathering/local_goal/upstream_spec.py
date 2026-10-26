"""Immutable upstream vehicle contracts; never change legacy defaults."""

from dataclasses import asdict, dataclass
from functools import cached_property
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType

import numpy as np


VERSIONS = MappingProxyType({
    "runtime_family": "upstream_scale_v1",
    "planner_protocol": "planner_protocol_v2",
    "task_contract_version": "finite_task_upstream_v1",
    "action_version": "polar_hold_upstream_v1",
    "observation_version": "local_intent_upstream_v1",
    "dynamics_version": "midpoint_upstream_v1",
    "geometry_version": "footprint_upstream_v1",
    "perception_version": "dense_local_upstream_v1",
    "communication_version": "distance_comm_upstream_v1",
    "cost_version": "normalized_upstream_v1",
    "safety_version": "discrete_cbf_qp_upstream_v1",
})


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def number(group, key, *, minimum=0., strict=True):
    value = group.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number")
    if value < minimum or (strict and value == minimum):
        raise ValueError(f"{key} outside its supported range")
    return float(value)


def integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class VehicleSpec:
    length: float
    width: float
    height: float
    wheel_radius: float
    track_width: float
    wheelbase: float
    track_multiplier: float
    forward_limit: float
    reverse_limit: float
    yaw_limit: float
    low_speed_yaw_limit: float
    low_speed_threshold: float
    linear_acceleration: float
    linear_deceleration: float
    angular_acceleration: float
    angular_deceleration: float
    wheel_acceleration: float
    clearance_margin: float
    contact_length: float
    contact_width: float
    ground_clearance: float
    sweep_spacing: float

    @property
    def physical_radius(self):
        return math.hypot(self.length, self.width) / 2

    @property
    def effective_radius(self):
        return self.physical_radius + self.clearance_margin

    @property
    def effective_track(self):
        return self.track_width * self.track_multiplier

    @property
    def wheel_limit(self):
        return self.forward_limit / self.wheel_radius

    @property
    def absolute_speed_bound(self):
        return max(self.forward_limit, self.reverse_limit)

    @property
    def contact_centers(self):
        return tuple((x * self.wheelbase / 2, y * self.track_width / 2)
                     for x in (-1, 1) for y in (-1, 1))


@dataclass(frozen=True)
class TimeGrid:
    dt: float
    steps: int
    high_level_period: float
    request_validity: float

    @property
    def horizon(self):
        return self.dt * self.steps

    def times(self, now):
        if isinstance(now, bool) or not np.isfinite(now) or now < 0:
            raise ValueError("Invalid episode time")
        values = float(now) + np.arange(self.steps + 1) * self.dt
        values.setflags(write=False)
        return values


@dataclass(frozen=True)
class TerrainSpec:
    driving_slope: float
    parking_slope: float
    roughness_rms: float
    step_height: float
    minimum_traversability: float = .25
    slope_speed_scale: float = .75


@dataclass(frozen=True)
class PerceptionSpec:
    forward: float
    reverse: float
    lateral: float
    spacing: float
    memory_resolution: float
    capacity: int
    support_radius: float

    @property
    def frame_shape(self):
        return (round((self.forward + self.reverse) / self.spacing) + 1,
                round(2 * self.lateral / self.spacing) + 1)

    @property
    def sample_count(self):
        return math.prod(self.frame_shape)


@dataclass(frozen=True)
class CommunicationSpec:
    full_radius: float
    sparse_radius: float
    full_period: float
    sparse_period: float
    state_validity: float
    trajectory_validity: float
    absolute_velocity_bound: float

    @property
    def communication_radius(self):
        # Generic config inventory access; not permission to use legacy execution.
        return self.full_radius


@dataclass(frozen=True)
class RendezvousSpec:
    n_agents: int
    spacing: float
    capture_radius: float
    exit_radius: float
    dmax: float
    dispersion: float
    worksite_radius: float
    linear_stop: float
    angular_stop: float
    boundary_stop: float
    hold_duration: float
    position_drift: float
    heading_drift: float
    spawn_tiers: tuple[tuple[float, float], ...]
    initial_distance: float
    episode_lengths: tuple[float, ...]

    @property
    def slots(self):
        return tuple((x * self.spacing / 2, y * self.spacing / 2)
                     for x in (-1, 1) for y in (-1, 1))


@dataclass(frozen=True)
class UpstreamSpec:
    vehicle: VehicleSpec
    grid: TimeGrid
    terrain: TerrainSpec
    perception: PerceptionSpec
    communication: CommunicationSpec
    task: RendezvousSpec
    rho_max: float
    latency_budget: float
    world_width: float
    world_height: float
    initial_test_speed: float
    profile_sha256: str
    profile_json: str

    @cached_property
    def fingerprint(self):
        return digest(asdict(self))

    @classmethod
    def from_design(cls, data, profile_sha256):
        if not isinstance(data, dict):
            raise ValueError("Upstream design must be a mapping")
        if (data.get("schema_version") != "upstream_vehicle_proxy_design_v2"
                or data.get("runtime_enabled") is not False
                or data.get("training_paused") is not True):
            raise ValueError("Expected isolated upstream design v2, not a runtime profile")
        dynamic = data.get("dynamic_speed_limit", {})
        if (dynamic.get("enabled") is not True or dynamic.get("latency_rule") != "by_clock_mode"
                or dynamic.get("next_control_commitment_steps") != 1
                or dynamic.get("minimum_stationary_tail_steps") != 1
                or dynamic.get("delay_rounding") != "ceil_latency_over_dt"):
            raise ValueError("Unsupported delayed-stop clock or commitment semantics")
        for key in ("upstream", "model", "motion", "geometry", "terrain", "perception",
                    "communication", "planning", "task", "physical_placeholders", "dynamic_speed_limit"):
            if not isinstance(data.get(key), dict):
                raise ValueError(f"Missing upstream design section {key}")
        u, model, m, g = (data[k] for k in ("upstream", "model", "motion", "geometry"))
        t, p, c, plan, task = (data[k] for k in ("terrain", "perception", "communication", "planning", "task"))
        if (model.get("type") != "differential_drive"
                or model.get("body_reference") != "planar_geometry_center"
                or model.get("integration") != "midpoint"
                or g.get("collision_envelope") != "circumscribed_circle"
                or g.get("terrain_footprint_envelope") != "oriented_body_rectangle_and_contact_patches"
                or task.get("certify_unused_common_area") is not False):
            raise ValueError("Unsupported upstream model/geometry/task semantics")
        forward = number(m, "default_forward_limit_mps")
        if not math.isclose(forward, number(u, "maximum_speed_kmh") / 3.6, abs_tol=1e-12, rel_tol=0):
            raise ValueError("Forward hard limit must equal upstream maximum speed / 3.6")
        v = VehicleSpec(
            *(number(u, k) for k in ("length_m", "width_m", "height_m")),
            number(u, "wheel_diameter_m") / 2, number(u, "track_width_m"),
            number(u, "wheelbase_m"), number(model, "effective_track_multiplier"),
            forward, number(m, "default_reverse_limit_mps"), number(m, "yaw_rate_limit_radps"),
            number(m, "in_place_yaw_rate_limit_radps"), number(m, "low_speed_turn_threshold_mps"),
            *(number(m, k) for k in ("linear_acceleration_mps2", "linear_deceleration_mps2",
                                     "angular_acceleration_radps2", "angular_deceleration_radps2",
                                     "wheel_acceleration_limit_radps2")),
            number(g, "clearance_margin_per_vehicle_m"), number(model, "wheel_contact_length_m"),
            number(model, "wheel_contact_width_m"), number(data["physical_placeholders"], "ground_clearance_m"),
            number(g, "max_swept_boundary_sample_spacing_m"),
        )
        if (v.low_speed_yaw_limit > v.yaw_limit or v.reverse_limit > v.forward_limit
                or v.wheelbase + v.contact_length > v.length
                or v.track_width + v.contact_width > v.width):
            raise ValueError("Inconsistent vehicle/contact geometry or direction limits")
        grid = TimeGrid(number(plan, "dt_s"), integer(plan.get("horizon_steps"), "horizon_steps"),
                        number(plan, "high_level_period_s"), number(plan, "request_validity_s"))
        if not math.isclose(grid.horizon, number(plan, "horizon_s"), abs_tol=1e-10):
            raise ValueError("Time grid and horizon disagree")
        ratio = grid.high_level_period / grid.dt
        if not math.isclose(ratio, round(ratio), abs_tol=1e-10):
            raise ValueError("High-level period must contain whole control steps")
        driving_deg, parking_deg = (number(t, k) for k in ("driving_slope_limit_deg", "parking_slope_limit_deg"))
        if not 0 < parking_deg <= driving_deg < 90:
            raise ValueError("Slope angles must satisfy 0 < parking <= driving < 90 degrees")
        terrain = TerrainSpec(math.tan(math.radians(driving_deg)),
                              math.tan(math.radians(parking_deg)),
                              number(t, "detrended_roughness_rms_limit_m"), number(t, "local_step_limit_m"))
        if terrain.parking_slope > terrain.driving_slope:
            raise ValueError("Parking slope cannot exceed driving slope")
        perception = PerceptionSpec(*(number(p, k) for k in (
            "forward_range_from_body_reference_m", "reverse_range_from_body_reference_m",
            "lateral_range_from_body_reference_m", "local_sample_spacing_m", "memory_key_resolution_m")),
            integer(p.get("memory_max_samples"), "memory_max_samples"),
            number(p, "nearest_neighbor_support_radius_m"))
        if perception.capacity < perception.sample_count:
            raise ValueError("Memory cannot preserve a complete direct sensor frame")
        for span in (perception.forward + perception.reverse, 2 * perception.lateral):
            if not math.isclose(span / perception.spacing, round(span / perception.spacing), abs_tol=1e-9):
                raise ValueError("Sensor extent must align with its direct sampling grid")
        comm = CommunicationSpec(*(number(c, k) for k in (
            "full_radius_m", "sparse_radius_m", "full_period_s", "sparse_period_s",
            "state_validity_s", "trajectory_validity_s")), v.absolute_speed_bound)
        if comm.sparse_radius < comm.full_radius or comm.trajectory_validity > comm.state_validity:
            raise ValueError("Inconsistent communication range or validity")
        if (c.get("future_prediction_is_control_commitment") is not False
                or not math.isclose(number(c, "unknown_sender_forward_speed_bound_mps"), forward,
                                    abs_tol=1e-12, rel_tol=0)):
            raise ValueError("Invalid sender conservative velocity/commitment semantics")
        tiers = tuple(tuple(float(x) for x in pair) for pair in task.get("spawn_radius_tiers_m", ()))
        durations = tuple(float(x) for x in task.get("diagnostic_episode_lengths_s", ()))
        if (len(tiers) != 3 or any(len(pair) != 2 or not all(math.isfinite(x) for x in pair)
                                   or not 0 < pair[0] <= pair[1] for pair in tiers)
                or not durations or any(not math.isfinite(x) or x <= 0 for x in durations)):
            raise ValueError("Invalid spawn tiers or episode budgets")
        rendezvous = RendezvousSpec(integer(task.get("n_agents"), "n_agents"),
            *(number(task, k) for k in ("square_grid_spacing_m", "slot_capture_radius_m",
              "slot_exit_radius_m", "max_center_distance_m", "dispersion_limit_m2",
              "common_worksite_radius_m", "parked_linear_speed_limit_mps",
              "parked_angular_speed_limit_radps", "parked_body_boundary_speed_limit_mps",
              "hold_duration_s", "max_position_drift_during_hold_m")),
            math.radians(number(task, "max_heading_drift_during_hold_deg")), tiers,
            number(task, "min_initial_center_distance_m"), durations)
        if rendezvous.n_agents != 4 or rendezvous.capture_radius >= rendezvous.exit_radius:
            raise ValueError("Expected four-agent square formation and capture hysteresis")
        error_budget = sum(number(g, k) for k in (
            "position_error_budget_m", "tracking_error_budget_m", "additional_geometry_allowance_m"))
        error_budget += 2 * v.physical_radius * math.sin(math.radians(number(g, "heading_error_budget_deg")) / 2)
        if error_budget > v.clearance_margin + 1e-12:
            raise ValueError("Geometry uncertainty exceeds per-vehicle margin")
        result = cls(v, grid, terrain, perception, comm, rendezvous, number(plan, "rho_max_m"),
                     number(m, "perception_to_execution_latency_budget_s"),
                     number(t, "map_window_width_m"), number(t, "map_window_height_m"),
                     number(m, "initial_test_speed_mps"), profile_sha256, canonical(data))
        result.geometry_audit()
        return result

    def geometry_audit(self):
        v, t = self.vehicle, self.task
        min_distance = t.spacing - 2 * t.capture_radius
        max_distance = math.sqrt(2) * t.spacing + 2 * t.capture_radius
        dispersion = (t.spacing / math.sqrt(2) + t.capture_radius) ** 2
        outer_radius = t.spacing / math.sqrt(2) + t.capture_radius + v.effective_radius
        if (min_distance < 2 * v.effective_radius or max_distance > t.dmax
                or dispersion > t.dispersion or outer_radius > t.worksite_radius
                or t.initial_distance < 2 * v.effective_radius):
            raise ValueError("Task success region has no robust square-formation witness")
        return {"minimum_captured_distance": min_distance, "maximum_captured_distance": max_distance,
                "maximum_captured_dispersion": dispersion, "captured_outer_radius": outer_radius,
                "physical_radius": v.physical_radius, "effective_radius": v.effective_radius,
                "safe_center_distance": 2 * v.effective_radius, "nonempty_success_region": True,
                "dense_sensor_shape": self.perception.frame_shape,
                "dense_sensor_samples": self.perception.sample_count,
                "geographic_anchor_buffer": max(x[1] for x in t.spawn_tiers)
                   + math.hypot(max(self.perception.forward, self.perception.reverse), self.perception.lateral)
                   + v.effective_radius}


@dataclass(frozen=True)
class SimulationSettings:
    num_envs: int
    device: str
    episode_length_s: float
    dt: float

    @property
    def max_episode_steps(self):
        return round(self.episode_length_s / self.dt)


@dataclass(frozen=True)
class UpstreamRuntimeConfig:
    spec: UpstreamSpec
    simulation: SimulationSettings
    seed: int
    planner_adapter: str
    planner_device: str
    mppi_samples: int
    mppi_iterations: int
    clock_mode: str
    simulated_latency: float
    configuration_json: str
    runtime_family: str = "upstream_scale_v1"
    training_paused: bool = True
    execution_review_version: str = "footprint_review_v1"
    terrain_query_version: str = "scalar_ties_v1"
    planner_execution_version: str = "not_connected"
    timing_version: str = "held_previous_control_v1"
    review_acceleration_version: str = "disabled"
    observation_execution_version: str = "not_connected"
    environment_execution_version: str = "not_connected"
    execution_safety_version: str = "projection_disabled"
    mppi_parameter_overrides: str = "{}"
    leader_execution_version: str = "leader_square_upstream_v1"
    training_session_version: str = ""
    motion_lease_version: str = "clearance_horizon_lease_v2"
    hazard_response_version: str = "strict_footprint_v1"
    obstacle_input: object = None

    @property
    def terrain_channels(self):
        return 3 if self.obstacle_input is not None else 2

    @property
    def actor_dim(self):
        return 523 if self.obstacle_input is not None else 411

    @property
    def critic_dim(self):
        return 1439 if self.obstacle_input is not None else 991

    @property
    def predictive_footprint_checks(self):
        return self.hazard_response_version == "strict_footprint_v1"

    @property
    def learning_runtime(self):
        return self.training_paused is False and self.training_session_version == "upstream_t11_learning_v1"

    @property
    def valid_execution_role(self):
        return (self.training_paused is True and not self.training_session_version) or self.learning_runtime

    @property
    def task(self):
        return self.spec.task

    @property
    def observation(self):
        return self.spec.communication

    @property
    def metadata(self):
        metadata = {**VERSIONS, "training_paused": self.training_paused, "profile_fingerprint": self.spec.fingerprint,
                "action_version": "primitive47_native_upstream_v1" if self.planner_adapter == "primitive47_upstream_v1" else VERSIONS["action_version"],
                "profile_file_sha256": self.spec.profile_sha256,
                "configuration_fingerprint": digest(json.loads(self.configuration_json)),
                "planner_adapter": self.planner_adapter, "planner_state_version": 4 if self.planner_adapter=='mppi_upstream_v4' else 3 if self.planner_adapter=='mppi_upstream_v3' else 2 if self.planner_adapter in {"mppi_upstream_v2","nmpc_upstream_v2"} else 1,
                "solver_cache_key": digest({"spec": self.spec.fingerprint, "adapter": self.planner_adapter, "versions": dict(VERSIONS)}),
                "time_grid": asdict(self.spec.grid), "vehicle": asdict(self.spec.vehicle),
                "planner_device": self.planner_device, "simulation_device": self.simulation.device,
                "mppi_samples": self.mppi_samples, "mppi_iterations": self.mppi_iterations,
                "clock_mode": self.clock_mode, "simulated_latency": self.simulated_latency,
                "rho_max": self.spec.rho_max, "beta_max": math.pi}
        if self.execution_review_version != "footprint_review_v1" or self.terrain_query_version != "scalar_ties_v1":
            metadata.update(execution_review_version=self.execution_review_version,
                            terrain_query_version=self.terrain_query_version,
                            geometry_version="footprint_upstream_v2")
            metadata["solver_cache_key"] = digest({"base": metadata["solver_cache_key"],
                "review": self.execution_review_version, "query": self.terrain_query_version})
        # Historical S00/S01 metadata remain byte-for-byte equivalent. Running
        # a planner is a distinct opt-in contract, never inferred from its ID.
        if self.planner_execution_version != "not_connected":
            metadata["planner_execution_version"] = self.planner_execution_version
            metadata["solver_cache_key"] = digest({"base": metadata["solver_cache_key"],
                "execution": self.planner_execution_version})
        if self.timing_version != "held_previous_control_v1":
            metadata["timing_version"] = self.timing_version
            metadata["solver_cache_key"] = digest({"base": metadata["solver_cache_key"], "timing": self.timing_version})
        if self.review_acceleration_version != "disabled":
            metadata["review_acceleration_version"] = self.review_acceleration_version
            metadata["solver_cache_key"] = digest({"base": metadata["solver_cache_key"], "review_acceleration": self.review_acceleration_version})
        if self.planner_adapter in {"mppi_upstream_v2","mppi_upstream_v3","mppi_upstream_v4"}:
            metadata["mppi_precision_version"] = "sampling_configured_actuator_fp64_v2"
        if self.planner_adapter in {'mppi_upstream_v3','mppi_upstream_v4'}:
            metadata['mppi_cost_version']='upstream_mppi_target_normalized_cost_v3' if self.planner_adapter=='mppi_upstream_v4' else 'upstream_mppi_speed_normalized_cost_v2'
            metadata['mppi_parameters']=json.loads(self.mppi_parameter_overrides)
        if self.observation_execution_version != "not_connected":
            metadata["observation_execution_version"] = self.observation_execution_version
            metadata["observation_version"] = self.observation_execution_version
        if self.environment_execution_version != "not_connected":
            metadata["environment_execution_version"] = self.environment_execution_version
            metadata["queue_version"] = ("accepted_request_lease_upstream_v2" if self.environment_execution_version == "actual_loop_upstream_v2"
                                         else "absolute_expiry_queue_upstream_v1")
            metadata["reward_version"] = "finite_reward_upstream_v1"
        if self.execution_safety_version != "projection_disabled":
            metadata['execution_safety_version']=self.execution_safety_version
        if self.leader_execution_version!="leader_square_upstream_v1":metadata["leader_execution_version"]=self.leader_execution_version
        if self.training_session_version:
            metadata["training_session_version"] = self.training_session_version
        if self.motion_lease_version != "clearance_horizon_lease_v2":
            metadata["motion_lease_version"] = self.motion_lease_version
        if not self.predictive_footprint_checks:
            metadata['hazard_response_version'] = self.hazard_response_version
            metadata['solver_cache_key'] = digest({'base': metadata['solver_cache_key'],
                'hazard_response': self.hazard_response_version})
        if self.obstacle_input is not None:
            metadata['obstacle_input'] = self.obstacle_input.metadata
            metadata['mppi_cost_version'] = 'upstream_grid_obstacle_soft_cost_v1'
            metadata['solver_cache_key'] = digest({'base': metadata['solver_cache_key'],
                'obstacle_input': self.obstacle_input.fingerprint})
        return metadata

    def validate_state_identity(self, state):
        if not isinstance(state, dict) or state.get("contracts") != self.metadata:
            raise ValueError("Upstream snapshot version/configuration/model mismatch")


def parse_upstream_experiment(data, root):
    """Separate schema dispatch; no legacy config can silently execute new physics."""
    if not isinstance(data, dict) or set(data) - {"experiment", "upstream_scale", "local_goal"}:
        raise ValueError("Unsupported upstream experiment root keys")
    e, u = data.get("experiment"), data.get("upstream_scale")
    if not isinstance(e, dict) or not isinstance(u, dict):
        raise ValueError("Upstream experiment requires experiment/upstream_scale mappings")
    session = u.get("training_session_version", "")
    paused = e.get("training_paused")
    if (not ((paused is True and session == "")
             or (paused is False and session == "upstream_t11_learning_v1"))
            or data.get("local_goal") != {"training_paused": paused}
            or u.get("schema_version") != "upstream_runtime_v1"):
        raise ValueError("Upstream runtime requires matching pause and explicit learning session version")
    allowed = {"schema_version", "design_profile", "design_profile_sha256", "planner_adapter",
               "planner_device", "mppi_samples", "mppi_iterations", "clock_mode", "simulated_latency_s",
               "execution_review_version", "terrain_query_version", "planner_execution_version", "timing_version",
               "review_acceleration_version", "observation_execution_version", "environment_execution_version", "execution_safety_version", "mppi_parameters", "leader_execution_version", "training_session_version", "motion_lease_version", "hazard_response_version", "obstacle_input_config"}
    if set(u) - allowed or set(e) - {"name", "seed", "num_envs", "device", "episode_length_s", "training_paused"}:
        raise ValueError("Unsupported upstream configuration keys")
    path = (Path(root) / u.get("design_profile", "")).resolve()
    if not path.is_file():
        raise ValueError("Missing upstream design profile")
    payload = path.read_bytes()
    checksum = hashlib.sha256(payload).hexdigest()
    if u.get("design_profile_sha256") != checksum:
        raise ValueError("Upstream design file fingerprint mismatch")
    import yaml
    spec = UpstreamSpec.from_design(yaml.safe_load(payload), checksum)
    adapter = u.get("planner_adapter")
    if adapter not in {"nmpc_upstream_v1", "nmpc_upstream_v2", "mppi_upstream_v1", "mppi_upstream_v2", "mppi_upstream_v3", "mppi_upstream_v4", "line_upstream_v1",
                        "quintic_upstream_v1", "primitive47_upstream_v1"}:
        raise ValueError("Unsupported upstream planner adapter")
    leader=u.get("leader_execution_version","leader_square_upstream_v1")
    if leader not in {"leader_square_upstream_v1","leader_square_visibility_upstream_v2","leader_square_visibility_upstream_v3","leader_square_visibility_upstream_v4"}:raise ValueError("Unsupported leader execution version")
    parameters=u.get('mppi_parameters',{})
    if (not isinstance(parameters,dict) or set(parameters)-{'goal_scale','terrain_scale','neighbor_scale','smooth_scale','noise_v','noise_w','temperature','footprint_margin_m','footprint_cover_spacing_m','large_crater_query'}
            or (parameters and adapter not in {'mppi_upstream_v3','mppi_upstream_v4'})):raise ValueError('Cost calibration requires explicit MPPI v3/v4 parameters')
    for name,value in parameters.items():
        if name=='large_crater_query':
            if value!='observed_grid_sat_v1' or adapter!='mppi_upstream_v4' or not u.get('obstacle_input_config'):
                raise ValueError('Observed-grid SAT requires v4 local obstacle grid input')
            continue
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0 or (value==0 and name!='footprint_margin_m'):
            raise ValueError('MPPI calibration parameters must be finite positive scalars (footprint margin may be zero)')
    if any(name in parameters for name in ('footprint_margin_m','footprint_cover_spacing_m')) and (adapter!='mppi_upstream_v4' or not u.get('obstacle_input_config')):
        raise ValueError('MPPI footprint tuning requires v4 local obstacle grid input')
    device = u.get("planner_device")
    if device not in {"cpu", "cuda"} or e.get("device") != "cpu":
        raise ValueError("Upstream host must be CPU; planner device must be explicit")
    latency = number(u, "simulated_latency_s", strict=False)
    if u.get("clock_mode") not in {"offline_simulated_latency", "realtime_measured_latency"}:
        raise ValueError("Invalid upstream clock mode")
    samples = integer(u.get("mppi_samples"), "mppi_samples")
    iterations = integer(u.get("mppi_iterations"), "mppi_iterations")
    num_envs = integer(e.get("num_envs"), "num_envs")
    seed = e.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or not e.get("name"):
        raise ValueError("Invalid upstream experiment identity/seed")
    duration = number(e, "episode_length_s")
    if not math.isclose(duration / spec.grid.dt, round(duration / spec.grid.dt), abs_tol=1e-9):
        raise ValueError("Episode budget must contain whole control steps")
    review = u.get("execution_review_version", "footprint_review_v1")
    query = u.get("terrain_query_version", "scalar_ties_v1")
    if review not in {"footprint_review_v1", "footprint_review_v2"} or query not in {"scalar_ties_v1", "batch_ties_v2"}:
        raise ValueError("Unsupported upstream review/query version")
    if query == "batch_ties_v2" and review != "footprint_review_v2":
        raise ValueError("Batch query requires the strict v2 review contract")
    execution = u.get("planner_execution_version", "not_connected")
    if execution not in {"not_connected", "upstream_planning_v1"}:
        raise ValueError("Unsupported upstream planner execution version")
    timing = u.get("timing_version", "held_previous_control_v1")
    if timing not in {"held_previous_control_v1", "actual_control_history_v1"}:
        raise ValueError("Unsupported upstream timing version")
    if timing == "actual_control_history_v1" and (execution != "upstream_planning_v1" or review != "footprint_review_v2"):
        raise ValueError("Actual control history requires the strict planning execution contract")
    acceleration=u.get("review_acceleration_version","disabled")
    if acceleration not in {"disabled","pose_scope_cache_v1","flat_grid_pose_cache_v2","flat_frame_pose_cache_v3","cuda_sweep_pose_v1"}:
        raise ValueError("Unsupported footprint review acceleration version")
    if acceleration != "disabled" and review != "footprint_review_v2":
        raise ValueError("Exact pose cache requires strict footprint review v2")
    observation=u.get("observation_execution_version","not_connected")
    if observation not in {"not_connected","dense_actor_clock_upstream_v1","obstacle_ground_known_actor_v1"}:
        raise ValueError("Unsupported upstream observation execution version")
    if observation != "not_connected" and timing != "actual_control_history_v1":
        raise ValueError("Dense actor observation requires explicit actual control history")
    environment=u.get("environment_execution_version","not_connected")
    if environment not in {"not_connected","actual_loop_upstream_v1","actual_loop_upstream_v2"}:raise ValueError("Unsupported upstream environment execution version")
    if environment != "not_connected" and observation not in {"dense_actor_clock_upstream_v1","obstacle_ground_known_actor_v1"}:
        raise ValueError("Actual environment requires explicit dense observation")
    safety=u.get('execution_safety_version','projection_disabled')
    hazard=u.get('hazard_response_version','strict_footprint_v1')
    if hazard not in {'strict_footprint_v1','cost_reward_only_v1'}:
        raise ValueError('Unsupported hazard response version')
    if hazard=='cost_reward_only_v1' and (safety!='projection_disabled' or adapter!='mppi_upstream_v4'):
        raise ValueError('Cost/reward-only experiment requires MPPI v4 and disabled projection')
    if safety not in {'projection_disabled','footprint_discrete_cbf_upstream_v1'}:raise ValueError('Unsupported upstream safety version')
    if safety!='projection_disabled' and environment=='not_connected':raise ValueError('Safety requires the actual environment')
    lease=u.get('motion_lease_version','clearance_horizon_lease_v2')
    if lease not in {'clearance_horizon_lease_v2','braking_clearance_lease_v3'}:raise ValueError('Unsupported motion lease policy')
    obstacle_input = None
    if 'obstacle_input_config' in u:
        from .upstream_obstacle_spec import ObstacleInputSpec
        obstacle_input = ObstacleInputSpec.load(Path(root)/u['obstacle_input_config'], root, spec)
        if observation != 'obstacle_ground_known_actor_v1' or hazard != 'cost_reward_only_v1':
            raise ValueError('Incremental obstacle input requires its observation and soft hazard contracts')
    elif observation == 'obstacle_ground_known_actor_v1':
        raise ValueError('Obstacle observations require an explicit input configuration')
    return UpstreamRuntimeConfig(spec, SimulationSettings(num_envs, "cpu", duration, spec.grid.dt),
        seed, adapter, device, samples, iterations, u["clock_mode"], latency, canonical(data),
        execution_review_version=review, terrain_query_version=query,
        planner_execution_version=execution, timing_version=timing, review_acceleration_version=acceleration,
        observation_execution_version=observation,environment_execution_version=environment,execution_safety_version=safety,
        mppi_parameter_overrides=canonical(parameters),leader_execution_version=leader,
        training_paused=paused,training_session_version=session,motion_lease_version=lease,hazard_response_version=hazard,
        obstacle_input=obstacle_input)
