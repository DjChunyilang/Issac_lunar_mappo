"""Optional shared GPU evaluation service for persistent environment owners.

CPU review stays with each owner. Changed terrain geometry is transferred once
per rover, then addressed by token. Requests ready together share one evaluate
launch; CPU/GPU transfer cost is intentionally exposed by end-to-end benchmarks.
"""
from dataclasses import fields
from multiprocessing.connection import wait
import time
import traceback

import numpy as np
import torch


def serve(connections):
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_footprint import FootprintTerrain
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_planning import UpstreamPlannerInput
    torch.set_num_threads(1)
    terrains={};kernels={}
    try:
        while connections:
            ready=wait(connections)
            # A bounded one millisecond coalescing window, never an environment
            # barrier: an owner can continue even when another is still on CPU.
            remaining=[pipe for pipe in connections if pipe not in ready]
            if remaining:ready+=wait(remaining,timeout=.001)
            groups={}
            for pipe in ready:
                message=pipe.recv()
                owner=connections.index(pipe)
                kernel_type,cfg,settings,items,changed,raw=message
                for key,xy,features,times in changed:
                    terrains[owner,key]=FootprintTerrain._from_owned_arrays(xy,features,times,cfg.spec)
                requests=[]
                for key,data in items:
                    requests.append(UpstreamPlannerInput(**data,terrain=terrains[owner,key]))
                identity=(kernel_type,settings,cfg.spec)
                if identity not in kernels:kernels[identity]=kernel_type(cfg,settings)
                groups.setdefault((identity,raw.shape[1:],raw.dtype.str),[]).append((pipe,requests,raw))
            for (identity,_,_),group in groups.items():
                kernel=kernels[identity];requests=sum([item[1] for item in group],[])
                raw=torch.as_tensor(np.concatenate([item[2] for item in group]),device=kernel.device)
                result=kernel.evaluate(requests,raw)
                def cpu(value):
                    if torch.is_tensor(value):return value.detach().cpu().numpy()
                    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
                    return tuple(cpu(v) for v in value)
                result=cpu(result)
                timing=kernel.last_evaluation_times
                if kernel.last_evaluation_events:
                    events=kernel.last_evaluation_events;events[-1].synchronize()
                    timing={'prediction_s':events[0].elapsed_time(events[1])/1000,'cost_s':events[1].elapsed_time(events[2])/1000}
                offset=0
                for pipe,items,_ in group:
                    end=offset+len(items)
                    def cut(value):
                        if isinstance(value,np.ndarray):return value[offset:end]
                        if isinstance(value,dict):return {k:cut(v) for k,v in value.items()}
                        return tuple(cut(v) for v in value)
                    pipe.send(('ok',(cut(result),timing,{'owners':len(group),'rows':len(requests)})));offset=end
    except (EOFError,BrokenPipeError):return
    except BaseException:
        message=traceback.format_exc()
        for pipe in connections:
            try:pipe.send(('error',message))
            except (BrokenPipeError,OSError):pass
        raise
    finally:
        for pipe in connections:pipe.close()


def install_client(pipe):
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_mppi import UpstreamMPPIKernel
    cache={}
    def evaluate(self,requests,raw,packed=None):
        items=[];changed=[]
        for request in requests:
            key=(request.env_id,request.agent_id);view=request.terrain
            old=cache.get(key)
            if old is None or old._xy is not view._xy or old._features is not view._features:
                changed.append((key,view._xy,view._features,view._timestamps));cache[key]=view
            items.append((key,{f.name:getattr(request,f.name) for f in fields(request) if f.name!='terrain'}))
        pipe.send((type(self),self.cfg,self.settings,items,changed,raw.detach().cpu().numpy()))
        status,result=pipe.recv()
        if status!='ok':raise RuntimeError('GPU service failed:\n'+result)
        result,timing,service=result
        stats=getattr(self,'service_stats',{'calls':0,'shared_calls':0,'maximum_batch_rows':0})
        stats['calls']+=1;stats['shared_calls']+=int(service['owners']>1)
        stats['maximum_batch_rows']=max(stats['maximum_batch_rows'],service['rows'])
        self.service_stats=stats
        def device(value):
            if isinstance(value,np.ndarray):return torch.as_tensor(value,device=self.device)
            if isinstance(value,dict):return {k:device(v) for k,v in value.items()}
            return tuple(device(v) for v in value)
        self.last_evaluation_events=None;self.last_evaluation_times=timing
        return device(result)
    UpstreamMPPIKernel.evaluate=evaluate
