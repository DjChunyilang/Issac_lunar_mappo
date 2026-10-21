"""Fixed, balanced near/far starts for the bounded combined-repair study."""
from copy import deepcopy
import gc

import numpy as np
import torch

COHORT = 'near_far_balanced_v1'


def near_poses():
    poses = np.zeros((4, 4, 3))
    poses[..., :2] = [
        [[-3.6, -3.6], [-3.6, 3.6], [3.6, -3.6], [3.6, 3.6]],
        [[-4., -3.5], [-4., 3.5], [4., -3.5], [4., 3.5]],
        [[-4., -3.4], [-3.4, 3.6], [3.6, -3.5], [4.1, 3.7]],
        [[-3.8, -3.4], [-3.3, 3.8], [3.5, -3.8], [4.2, 3.5]],
    ]
    poses[..., 2] = [[0., 1.2, -2.1, 2.8], [.8, -.6, 2.2, -2.6],
                     [-1., 2.5, .4, -2.], [2.9, -1.8, .9, -.2]]
    return poses


def mixed_initial_conditions(far, seed, cohort):
    if cohort != COHORT or not 23 <= seed <= 38:
        raise ValueError('Mixed cohort is fixed to seed23 and its 16 owner-local starts')
    poses = np.asarray(far).copy()
    controls = np.zeros((*poses.shape[:2], 2))
    near = near_poses()
    for local in range(len(poses)):
        index = seed - 23 + local
        if not 0 <= index < 16:
            raise ValueError('Mixed cohort has exactly 16 starts')
        if index % 2:
            poses[local] = near[(index // 2) % 4]
            # The second replica uses different headings without changing geometry.
            poses[local, :, 2] = (poses[local, :, 2] + (index // 8)*np.pi/2 + np.pi) % (2*np.pi) - np.pi
            controls[local, :, 0] = .1
    return poses, controls


def near_collector(base, policy, timing, *, samples=512):
    from run_upstream_t11 import make_collector
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_ablation_learning import UpstreamAblationRollout
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_environment import UpstreamEnvironment
    c = make_collector(base, 'A1', 23, samples=samples, num_envs=4, learning_time=timing,
                       policy=deepcopy(policy).cpu().eval())
    control = np.zeros((4, 4, 2)); control[..., 0] = .1
    env = UpstreamEnvironment(c.env.cfg, near_poses(), initial_control=control,
                              reward_settings=c.env.reward_tracker.settings)
    return UpstreamAblationRollout(env, c.policy, timing)


def evaluate_panels(base, checkpoint, policy, timing, *, final=False):
    from upstream_decision_evaluation import far_collector, panel
    results = {}
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        for name, constructor in (('far', far_collector), ('near', near_collector)):
            for deterministic in (False, True):
                kwargs = {'checkpoint': checkpoint} if name == 'far' else {}
                c = constructor(base, policy=policy, timing=timing, **kwargs)
                torch.manual_seed(20261019)
                results[f'{name}_{"mean" if deterministic else "sample"}'] = panel(c, deterministic=deterministic)
                del c; gc.collect()
    return results
