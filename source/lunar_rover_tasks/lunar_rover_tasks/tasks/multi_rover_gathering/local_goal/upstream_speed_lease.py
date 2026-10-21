"""Research speed-lease proposal; execution still requires full stop review."""
import math

import numpy as np


def braking_clearance_limit(spec, clearance, latency):
    """Solve v*tau + v^2/(2*a) <= 0.8*clearance for a requested bound.

    This proposes an enforceable upper bound, not a collision certificate.
    Direction, terrain, peers and the discrete stopping tail remain the public
    execution review's responsibility. Existing lower leases are never revoked.
    """
    if math.isnan(clearance) or not math.isfinite(latency) or latency<0:
        raise ValueError('Invalid speed lease clearance/latency')
    if clearance==math.inf:return spec.vehicle.forward_limit
    margin=.8*max(0.,float(clearance))
    acceleration=spec.vehicle.linear_deceleration
    reaction=latency+spec.grid.request_validity
    # Stable quadratic root near zero clearance.
    denominator=math.sqrt((acceleration*reaction)**2+2*acceleration*margin)+acceleration*reaction
    speed=0. if denominator==0 else 2*acceleration*margin/denominator
    return min(spec.vehicle.forward_limit,speed)


def renew_braking_leases(env, requests=None):
    """Use own actuator state and legal received messages only.

    Learned actions are not inspected; scripted HOLD may request a zero lease
    once it can be enforced. The ledger retains every prior unexpired promise.
    """
    for e in range(env.e):
        if env._ended[e]:continue
        now=float(env.core.times[e])
        for a in range(4):
            pose=env.local_pose(e,a);snapshot=env.radio.snapshot(e,a,now,pose)
            gaps=[np.linalg.norm(pose[:2]-message.pose[:2])-2*env.spec.vehicle.effective_radius
                  for i,message in enumerate(snapshot.messages) if snapshot.state_valid[i]]
            hold=requests is not None and requests[e][a].hold
            requested=0. if hold else braking_clearance_limit(env.spec,min(gaps,default=np.inf),env.cfg.simulated_latency)
            cap=max(float(abs(env.core.previous_control[e,a,0])),float(requested))
            env.pledge_speed(e,a,cap,now+env.spec.grid.horizon+env.cfg.simulated_latency+2*env.spec.grid.high_level_period)
