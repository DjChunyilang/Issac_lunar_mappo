"""Explicit adapter registry. Optional solvers are constructed only by their factory."""

ADAPTERS = {
    "nmpc_bridge_v1": ("prefix_consistency_v2", 1),
    "nmpc_polar_v1": ("prefix_consistency_v2", 2),
    "nmpc_recovery_v3": ("merit_recovery_v3", 3),
    "nmpc_recovery_v4": ("merit_recovery_v4", 4),
    "deterministic_line_v1": ("deterministic_v1", 1),
    "deterministic_quintic_v1": ("deterministic_v1", 1),
    "primitive47_adapter_v1": ("exp156_primitive47_v1", 1),
    "mppi_base_v1": ("mppi_base_v1", 1),
    "mppi_terrain_neighbor_v1": ("mppi_terrain_neighbor_v1", 1),
    "mppi_terrain_neighbor_stable_v2": ("mppi_terrain_neighbor_stable_v2", 2),
}


def adapter_metadata(adapter, revision):
    if adapter not in ADAPTERS or ADAPTERS[adapter][0] != revision:
        raise ValueError("Unsupported planner revision/adapter combination")
    return ADAPTERS[adapter][1]


def create_adapters(count, cache_dir, slope_scale, contract, *, solver_factory=None):
    adapter_metadata(contract.planner_adapter, contract.planner_revision)
    adapter = contract.planner_adapter
    if adapter.startswith("nmpc_"):
        from .planner import NMPCPlanner, NMPCRecoveryPlanner, NMPCAdaptiveRecoveryPlanner, build_solver
        from .protocol import NMPCAdapter
        from .baselines import NMPCPolarAdapter, NMPCRecoveryAdapter, NMPCAdaptiveRecoveryAdapter

        factory = build_solver if solver_factory is None else solver_factory
        if adapter in ("nmpc_recovery_v3", "nmpc_recovery_v4"):
            cls, planner_cls = ((NMPCAdaptiveRecoveryAdapter, NMPCAdaptiveRecoveryPlanner)
                                if adapter == "nmpc_recovery_v4" else (NMPCRecoveryAdapter, NMPCRecoveryPlanner))
            return [cls(planner_cls(
                factory(cache_dir, slope_scale, build=(i == 0),
                        **({"revision": contract.planner_revision} if solver_factory is None else {})),
                slope_scale, deadline=contract.nmpc_deadline_s)) for i in range(count)]
        cls = NMPCPolarAdapter if adapter == "nmpc_polar_v1" else NMPCAdapter
        return [
            cls(NMPCPlanner(factory(cache_dir, slope_scale, build=(i == 0)), slope_scale,
                           deadline=contract.nmpc_deadline_s))
            for i in range(count)
        ]
    if adapter.startswith("deterministic_"):
        from .baselines import DeterministicAdapter

        return [
            DeterministicAdapter(
                "line" if adapter == "deterministic_line_v1" else "quintic", slope_scale
            )
            for _ in range(count)
        ]
    if adapter == "primitive47_adapter_v1":
        from .baselines import PrimitiveAdapter

        return [PrimitiveAdapter(slope_scale) for _ in range(count)]
    from .mppi import MPPIAdapter, MPPIKernel, MPPIConfig

    config = MPPIConfig.from_contract(contract, slope_scale)
    kernel = MPPIKernel(config)
    return [MPPIAdapter(kernel, seed=contract.planner_seed) for _ in range(count)]
