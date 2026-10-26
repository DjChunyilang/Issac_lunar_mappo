"""Explicit S02 entry, lazy optional solvers and per-vehicle public lifecycle."""

from copy import deepcopy
from dataclasses import asdict
import json

from .upstream_planning import PlanningAdapter, UpstreamPlannerInput
from .upstream_spec import UpstreamRuntimeConfig, digest


EXECUTION_VERSION = "upstream_planning_v1"
POOL_STATE_VERSION = "upstream_planner_pool_v1"
ADAPTER_IDS = frozenset({"line_upstream_v1", "quintic_upstream_v1", "primitive47_upstream_v1",
                         "nmpc_upstream_v1", "nmpc_upstream_v2", "mppi_upstream_v1", "mppi_upstream_v2", "mppi_upstream_v3", "mppi_upstream_v4"})
MPPI_IDS = frozenset({"mppi_upstream_v1", "mppi_upstream_v2", "mppi_upstream_v3", "mppi_upstream_v4"})


def _configuration(cfg):
    if type(cfg) is not UpstreamRuntimeConfig or cfg.planner_execution_version != EXECUTION_VERSION:
        raise ValueError("Planner factory requires explicit upstream_planning_v1; S00/S01 are not connected")
    if cfg.execution_review_version != "footprint_review_v2" or cfg.terrain_query_version != "batch_ties_v2":
        raise ValueError("Planner factory requires strict footprint review and bounded batch query v2")
    if cfg.clock_mode != "offline_simulated_latency":
        raise ValueError("S02 realtime execution is not connected; use offline_simulated_latency")
    if cfg.planner_adapter not in ADAPTER_IDS or not cfg.valid_execution_role:
        raise ValueError("Invalid adapter, training pause or explicit learning runtime")
    if cfg.planner_adapter not in MPPI_IDS and cfg.planner_device != "cpu":
        raise ValueError("Only the MPPI adapter has a CUDA implementation")
    return cfg


def create_upstream_adapter(cfg, *, kernel_cache=None, mppi_settings=None, cache_dir=None, solver=None):
    """Construct only the requested backend. Sharing never shares per-car RNG."""
    _configuration(cfg)
    adapter = cfg.planner_adapter
    if adapter not in MPPI_IDS and mppi_settings is not None:
        raise ValueError("MPPI settings supplied to a different planner")
    if adapter not in {"nmpc_upstream_v1","nmpc_upstream_v2"} and solver is not None:
        raise ValueError("NMPC solver supplied to a different planner")
    if adapter in MPPI_IDS:
        from .upstream_mppi import UpstreamMPPIAdapter, UpstreamMPPIKernel, UpstreamMPPISettings
        kernel_type,adapter_type=UpstreamMPPIKernel,UpstreamMPPIAdapter
        settings_type=UpstreamMPPISettings
        if adapter == "mppi_upstream_v2":
            from .upstream_mppi_precision import UpstreamMPPIAdapterV2,UpstreamMPPIKernelV2
            kernel_type,adapter_type=UpstreamMPPIKernelV2,UpstreamMPPIAdapterV2
        if adapter == 'mppi_upstream_v3':
            from .upstream_mppi_calibration import UpstreamMPPISettingsV3,UpstreamMPPIKernelV3,UpstreamMPPIAdapterV3
            settings_type,kernel_type,adapter_type=UpstreamMPPISettingsV3,UpstreamMPPIKernelV3,UpstreamMPPIAdapterV3
        if adapter == 'mppi_upstream_v4':
            from .upstream_mppi_calibration import UpstreamMPPISettingsV3,UpstreamMPPIKernelV4,UpstreamMPPIAdapterV4
            settings_type,kernel_type,adapter_type=UpstreamMPPISettingsV3,UpstreamMPPIKernelV4,UpstreamMPPIAdapterV4
        overrides=json.loads(cfg.mppi_parameter_overrides) if adapter in {'mppi_upstream_v3','mppi_upstream_v4'} else {}
        settings = mppi_settings or settings_type(samples=cfg.mppi_samples,
            iterations=cfg.mppi_iterations, device=cfg.planner_device,**overrides)
        if type(settings) is not settings_type:
            raise ValueError("Expected versioned MPPI settings")
        if any(getattr(settings,k)!=v for k,v in overrides.items()):raise ValueError('MPPI settings disagree with configured calibration parameters')
        if (settings.samples, settings.iterations, settings.device) != (
                cfg.mppi_samples, cfg.mppi_iterations, cfg.planner_device):
            raise ValueError("MPPI settings disagree with resolved run configuration")
        key = digest({"profile": cfg.spec.fingerprint, "grid": asdict(cfg.spec.grid),
                      **({'obstacle_input':cfg.obstacle_input.metadata} if cfg.obstacle_input is not None else {}),
                      "settings": asdict(settings), "execution": EXECUTION_VERSION,
                      "kernel": kernel_type.version, "cost": kernel_type.cost_version})
        kernel = None if kernel_cache is None else kernel_cache.get(key)
        if kernel is None:
            kernel = kernel_type(cfg, settings)
            if kernel_cache is not None:kernel_cache[key] = kernel
        return adapter_type(cfg, kernel=kernel)
    if adapter in {"nmpc_upstream_v1","nmpc_upstream_v2"}:
        from .upstream_nmpc import UpstreamNmpcAdapter
        from .upstream_nmpc_recovery import UpstreamNmpcAdapterV2
        return (UpstreamNmpcAdapterV2 if adapter=='nmpc_upstream_v2' else UpstreamNmpcAdapter)(cfg, cache_dir=cache_dir, solver=solver)
    from .upstream_baselines import DeterministicAdapter, PrimitiveAdapter
    if adapter == "primitive47_upstream_v1":
        return PrimitiveAdapter(cfg)
    return DeterministicAdapter(cfg, "line" if adapter == "line_upstream_v1" else "quintic")


def _context(item):
    if type(item) is UpstreamPlannerInput:
        return item
    # Only native primitive requests use this wrapper, never arbitrary objects
    # that happen to expose a context/env attribute.
    from .upstream_baselines import UpstreamPrimitiveInput
    if type(item) is UpstreamPrimitiveInput:
        item.validate()
        return item.context
    raise ValueError("Expected an upstream planner input or native primitive payload")


def _validate_pairs(planners, inputs):
    if len(planners) != len(inputs) or len({id(p) for p in planners}) != len(planners):
        raise ValueError("Exactly one independent planner handle is required per input")
    contexts = [_context(item) for item in inputs]
    if len({(item.env_id, item.agent_id) for item in contexts}) != len(contexts):
        raise ValueError("Duplicate planner identity in batch")
    for planner, item, context in zip(planners, inputs, contexts, strict=True):
        if not isinstance(planner, PlanningAdapter):
            raise ValueError("Expected a public upstream planning adapter")
        _configuration(planner.cfg)
        if planner.adapter_id != planner.cfg.planner_adapter:
            raise ValueError("Planner implementation/configuration identity mismatch")
        native = type(item) is not UpstreamPlannerInput
        if native != (planner.adapter_id == "primitive47_upstream_v1"):
            raise ValueError("Native primitive payload and planner disagree")
        planner.validate_input(context)
    return contexts


def plan_batch(planners, inputs, *, record_samples=True):
    """Restore caller order after true tensor batches grouped by kernel identity.

    Input validation precedes any planner state changes. NMPC/geometric planners
    use explicit serial dispatch; MPPI uses its shared batched computation.
    """
    planners, inputs = list(planners), list(inputs)
    _validate_pairs(planners, inputs)
    outputs = [None] * len(inputs)
    groups = {}
    for index, planner in enumerate(planners):
        if planner.adapter_id in MPPI_IDS:
            key = digest(planner.batch_kernel.metadata)
            groups.setdefault(key, []).append(index)
        else:
            outputs[index] = planner.plan(inputs[index])
    for indices in groups.values():
        kernel = planners[indices[0]].batch_kernel
        results = kernel.plan_batch([planners[i] for i in indices], [inputs[i] for i in indices], record_samples=record_samples)
        for index, output in zip(indices, results, strict=True):
            outputs[index] = output
    return outputs


class UpstreamPlannerPool:
    """Lazy per-(environment,agent) handles with isolated reset and snapshots."""

    def __init__(self, cfg, *, mppi_settings=None, cache_dir=None):
        self.cfg = _configuration(cfg)
        self.mppi_settings = mppi_settings
        self.cache_dir = cache_dir
        # Diagnostic-only switch; never part of planner hot-state identity.
        self.record_samples = True
        self._kernels = {}
        self._planners = {}

    def _identity(self, item):
        context = _context(item)
        if context.env_id >= self.cfg.simulation.num_envs:
            raise ValueError("Planner environment ID is outside the configured pool")
        return context.env_id, context.agent_id

    def _new_adapter(self):
        return create_upstream_adapter(self.cfg, kernel_cache=self._kernels,
                                      mppi_settings=self.mppi_settings, cache_dir=self.cache_dir)

    def plan(self, item):
        return self.plan_batch([item])[0]

    def plan_batch(self, inputs):
        inputs = list(inputs)
        identities = [self._identity(item) for item in inputs]
        if len(set(identities)) != len(identities):
            raise ValueError("Duplicate planner identity in pool batch")
        # New handles remain unbound until the entire batch has been validated.
        pending = {key: self._new_adapter() for key in identities if key not in self._planners}
        planners = [self._planners[key] if key in self._planners else pending[key] for key in identities]
        _validate_pairs(planners, inputs)
        self._planners.update(pending)
        return plan_batch(planners, inputs, record_samples=self.record_samples)

    def reset(self, env_id=None):
        if env_id is not None and (isinstance(env_id, bool) or not isinstance(env_id, int)
                                   or not 0 <= env_id < self.cfg.simulation.num_envs):
            raise ValueError("Invalid asynchronous reset environment")
        for identity, planner in self._planners.items():
            if env_id is None or identity[0] == env_id:
                planner.reset()

    def state_dict(self):
        return deepcopy({"contracts": self.cfg.metadata, "pool_state_version": POOL_STATE_VERSION,
                         "planners": [{"identity": key, "state": planner.state_dict()}
                                      for key, planner in sorted(self._planners.items())]})

    def reset_agent(self,env_id,agent_id):
        """Explicit recovery of one planner; does not alter peer warm/RNG state."""
        if (type(env_id) is not int or type(agent_id) is not int or not 0<=env_id<self.cfg.simulation.num_envs
                or not 0<=agent_id<4):raise ValueError('Invalid per-agent planner reset identity')
        planner=self._planners.get((env_id,agent_id))
        if planner is not None:planner.reset()

    def export_diagnostics(self):
        """Detached CPU evidence; no solver objects or execution-state access.

        This is deliberately separate from hot-state snapshots. Sampling data
        describe only the recorded iteration, not every optimization iteration.
        """
        return deepcopy({"diagnostic_version":"upstream_pool_diagnostics_v1",
            "contracts":self.cfg.metadata,"planners":[{"identity":identity,
                "adapter_id":planner.adapter_id,"samples":getattr(planner,"last_samples",None),
                "diagnostics":getattr(planner,"last_diagnostics",{})}
                for identity,planner in sorted(self._planners.items())]})

    def load_state_dict(self, state):
        """Restore all handles atomically; a late bad entry cannot partly reset a pool."""
        incoming = deepcopy(state)
        self.cfg.validate_state_identity(incoming)
        if incoming.get("pool_state_version") != POOL_STATE_VERSION or not isinstance(incoming.get("planners"), list):
            raise ValueError("Planner pool state version/entries mismatch")
        trial = UpstreamPlannerPool(self.cfg, mppi_settings=self.mppi_settings, cache_dir=self.cache_dir)
        for record in incoming["planners"]:
            if not isinstance(record, dict) or set(record) != {"identity", "state"}:
                raise ValueError("Invalid planner pool record")
            identity = record["identity"]
            if (not isinstance(identity, tuple) or len(identity) != 2
                    or any(isinstance(x, bool) or not isinstance(x, int) for x in identity)
                    or not 0 <= identity[0] < self.cfg.simulation.num_envs or not 0 <= identity[1] < 4
                    or identity in trial._planners):
                raise ValueError("Invalid or duplicate restored planner identity")
            nested = record["state"]
            if not isinstance(nested, dict) or nested.get("identity") not in (None, identity):
                raise ValueError("Pool and nested planner identity disagree")
            planner = trial._new_adapter()
            planner.load_state_dict(nested)
            trial._planners[identity] = planner
        self._planners, self._kernels = trial._planners, trial._kernels
