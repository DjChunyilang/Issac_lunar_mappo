"""Independent NMPC recovery for stationary SQP direction-mode locking."""
import numpy as np
from pathlib import Path
import time
import fcntl

from .upstream_nmpc import UpstreamNmpcAdapter,build_upstream_solver,solver_identity
from .upstream_spec import digest


class UpstreamNmpcAdapterV2(UpstreamNmpcAdapter):
    adapter_id='nmpc_upstream_v2'
    state_version=2

    def __init__(self,cfg,*,cache_dir=None,solver=None,deadline=10.):
        elapsed=0.
        if solver is None and cache_dir is not None:
            base=Path(cache_dir).resolve();base.mkdir(parents=True,exist_ok=True);key=digest(solver_identity(cfg.spec))
            with (base/(key+'.lock')).open('a+') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX);started=time.monotonic();folder=base/key
                library=folder/'code'/('libacados_ocp_solver_upstream_'+key[:12]+'.so')
                ready=library.is_file() and (folder/'upstream_model_identity.json').is_file()
                solver=build_upstream_solver(cfg.spec,base,build=not ready);elapsed=time.monotonic()-started
                fcntl.flock(lock,fcntl.LOCK_UN)
        super().__init__(cfg,cache_dir=cache_dir,solver=solver,deadline=deadline)
        if elapsed:self.engine.construction_seconds=elapsed

    def candidate_controls(self,inp):
        previous=self._previous_trajectory
        distance=float(np.linalg.norm(inp.intent.goal_xy-inp.ego_pose[:2]))
        stationary=(previous is not None and float(np.max(np.linalg.norm(
            previous.states[:,:2]-previous.states[0,:2],axis=-1)))<.01
            and float(np.max(np.abs(previous.controls)))<1e-8)
        reseed=stationary and not inp.intent.hold and distance>self.spec.task.capture_radius
        if reseed:self._previous_trajectory=None
        try:candidates=super().candidate_controls(inp)
        finally:self._previous_trajectory=previous
        for candidate in candidates:
            candidate.diagnostics['warm_seed_policy']='stationary_zero_mode_goal_reseed_v1'
            candidate.diagnostics['stationary_mode_reseed']=bool(reseed)
        if self.last_solver_diagnostics is not None:self.last_solver_diagnostics['stationary_mode_reseed']=bool(reseed)
        return candidates

    def _adapter_state(self):
        return {**super()._adapter_state(),'seed_recovery_version':'stationary_zero_mode_goal_reseed_v1'}

    def _validate_adapter_state(self,state):
        if state.get('seed_recovery_version')!='stationary_zero_mode_goal_reseed_v1':raise ValueError('NMPC recovery version mismatch')
        return super()._validate_adapter_state(state)
