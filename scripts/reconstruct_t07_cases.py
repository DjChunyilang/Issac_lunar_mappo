"""Rebuild sensor snapshots by replaying recorded actual controls, not planners."""
import json
from pathlib import Path
import numpy as np
import torch
from _common import cfg_from_experiment
from planner_evidence import json_write
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.environment import LocalGoalEnvironment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.actions import PolicyAction
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.contracts import TimedTrajectory,DT
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.benchmark import VERSION,digest


def reconstruct(source,config_path,selected):
    data=json.loads(Path(source).read_text())
    if not config_path.exists():json_write(config_path,data['resolved_config'])
    cfg=cfg_from_experiment(config_path);cfg.simulation.num_envs=2
    cfg.local_goal_contract.mppi_samples=128
    env=LocalGoalEnvironment(cfg,config_path.parent/'unused_solver')
    max_error=0.
    try:
        for row in data['rows']:
            for record in row['records']:
                action=PolicyAction(np.array(record['request']['physical']),np.array(record['request']['hold'],bool))
                env.intents=action.decode(env.poses(),env.core.step_count.numpy()*DT,1.6,np.pi)
                env.plans=[None if plan is None else TimedTrajectory(np.asarray(plan['states']),
                    np.asarray(plan['controls']),np.asarray(plan['velocities']),np.asarray(plan['times']),plan['times'][0])
                    for plan in record['plans']]
                commands=np.asarray(record['commands']);env.commands=commands.copy()
                env._physics(commands,torch.as_tensor(record['active']))
                error=np.max(abs(env.core.positions.numpy()-np.asarray(record['positions'])))
                max_error=max(max_error,float(error))
                if error>1e-4:raise ValueError(f'T07 reconstruction state differs: {error}')
                env.observe()
        poses=env.poses();cases=[]
        for number,(e,a) in selected:
            view=env.memories[e*4+a];s=view;pose=poses[e,a]
            yaw=float(env.core.yaws[e,a]);velocity=[float(env.core.velocities_xy[e,a].numpy()@np.array([np.cos(yaw),np.sin(yaw)])),float(env.core.angular_velocities[e,a])]
            intent=env.intents[e*4+a];now=float(env.core.step_count[e])*DT
            neighbors=[{'sender_id':p.sender_id,'positions':p.prediction.positions.tolist(),
              'radii':p.prediction.radii.tolist(),'timestamp':p.prediction.timestamp,
              'expires_at':p.prediction.expires_at} for p in env.neighbors(e,a,identified=True)]
            case={'schema_version':VERSION,'case_id':f'regression_{number:02d}','family':'slope','seed':data['seed'],
              'env_id':e,'agent_id':a,
              'pose':pose.tolist(),'velocity':velocity,'previous_control':env.commands[e,a].tolist(),
              'goal':intent.goal_xy.tolist(),'hold':intent.hold,'now':now,'timestamp':intent.timestamp,
              'expires_at':intent.expires_at,'neighbors':neighbors,
              'terrain':{'xy':s.xy.tolist(),'features':s.features.tolist(),'observed_at':s._timestamps[:len(s.samples)].tolist(),
                         'support_radius':s.support_radius,'resolution':s.resolution},
              'map_identity':data['map_identity'],'solvability':'unknown_solvability','witness':None,
              'provenance':{'type':'control_reconstructed','source':str(source),'env_id':e,'agent_id':a,
                            'maximum_world_position_error':max_error,'original_memory_not_saved':True}}
            case['sha256']=digest(case);cases.append(case)
        return cases
    finally:env.close()
