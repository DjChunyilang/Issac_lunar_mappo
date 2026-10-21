"""Legal sender-circle visibility waypoints for the frozen square site.

The planner still certifies actual trajectories. This graph selects only a
rolling goal; it cannot certify terrain, dynamics or any issued control.
"""
from dataclasses import replace
import heapq

import numpy as np

from .upstream_gathering import LeaderSquareScript

SCRIPT_VERSION='leader_square_visibility_upstream_v2'


def visibility_waypoint(position,goal,circles,maximum=4.):
    position=np.asarray(position,float);goal=np.asarray(goal,float)
    circles=sorted(circles,key=lambda c:c[0])
    if position.shape!=(2,) or goal.shape!=(2,) or not np.isfinite([*position,*goal,maximum]).all() or maximum<=0:
        raise ValueError('Invalid local visibility goal')
    if not circles:
        delta=goal-position;return position+delta*min(1.,maximum/max(np.linalg.norm(delta),1e-12)),False
    centers=np.asarray([c[1] for c in circles]);radii=np.asarray([c[2] for c in circles])
    if centers.shape!=(len(circles),2) or not np.isfinite(centers).all() or not np.isfinite(radii).all() or (radii<=0).any():
        raise ValueError('Invalid legal sender circles')
    if (np.linalg.norm(position-centers,axis=-1)<radii-1e-8).any() or (np.linalg.norm(goal-centers,axis=-1)<radii-1e-8).any():return None,False
    angles=np.arange(64)*(2*np.pi/64)
    boundary=np.concatenate([center+(radius+.5)*np.c_[np.cos(angles),np.sin(angles)] for center,radius in zip(centers,radii)])
    feasible=(np.linalg.norm(boundary[:,None]-centers[None],axis=-1)>=radii[None]+1e-8).all(-1)
    nodes=np.vstack((position,goal,boundary[feasible]));delta=nodes[None,:,:]-nodes[:,None,:]
    length=np.linalg.norm(delta,axis=-1);square=np.maximum(length**2,1e-20)
    offset=centers[None,None,:,:]-nodes[:,None,None,:]
    fraction=np.clip(np.sum(offset*delta[:,:,None,:],axis=-1)/square[:,:,None],0.,1.)
    closest=nodes[:,None,None,:]+fraction[:,:,:,None]*delta[:,:,None,:]
    clear=(np.linalg.norm(closest-centers[None,None,:,:],axis=-1)>=radii[None,None,:]-1e-8).all(-1)
    if clear[0,1]:
        vector=goal-position;return position+vector*min(1.,maximum/max(length[0,1],1e-12)),False
    np.fill_diagonal(clear,False);distance=np.full(len(nodes),np.inf);previous=np.full(len(nodes),-1,int)
    distance[0]=0.;queue=[(0.,0)]
    while queue:
        cost,i=heapq.heappop(queue)
        if cost>distance[i]+1e-12:continue
        if i==1:break
        for j in np.flatnonzero(clear[i]):
            candidate=cost+length[i,j]
            if candidate<distance[j]-1e-12:
                distance[j]=candidate;previous[j]=i;heapq.heappush(queue,(candidate,int(j)))
    if not np.isfinite(distance[1]):return None,False
    path=[1]
    while path[-1]!=0:path.append(int(previous[path[-1]]))
    path.reverse()
    # Move to the farthest visible path point within the original polar range.
    # Truncation stays on a verified visibility segment, never a shortcut chord.
    for index in reversed(path[1:]):
        if clear[0,index]:
            vector=nodes[index]-position
            return position+vector*min(1.,maximum/max(np.linalg.norm(vector),1e-12)),True
    return None,False


class LeaderSquareVisibilityScript(LeaderSquareScript):
    @property
    def metadata(self):
        return {**super().metadata,'script_version':SCRIPT_VERSION,'routing_version':'sender_visibility_v1',
            'boundary_points':64,'navigation_buffer':.5}

    def decide(self,pose,actual_velocity,previous_control,terrain,snapshot,now,episode):
        decision=super().decide(pose,actual_velocity,previous_control,terrain,snapshot,now,episode)
        if decision.intent.hold or decision.slot_xy is None:return decision
        circles=[]
        for i,prediction in enumerate(snapshot.predictions):
            if not snapshot.state_valid[i]:continue
            message=snapshot.messages[i]
            # Future radius comes from the already validated communication
            # predictor, including forced bounds/age. HOLD supplies no shortcut.
            if not self._stopped(np.asarray(message.actual_velocity)):continue
            center,_=prediction.at(now);radius=max(prediction.at(now+t)[1] for t in (0.,self.spec.grid.horizon))
            circles.append((prediction.sender_id,center,radius))
        waypoint,detour=visibility_waypoint(pose[:2],decision.slot_xy,circles,self.spec.rho_max)
        if not detour or waypoint is None:return decision
        return replace(decision,intent=replace(decision.intent,goal_xy=waypoint),reason='sender_visibility_detour')


class LeaderSquareVisibilityHysteresisScript(LeaderSquareVisibilityScript):
    """Release followers when a legal parked peer remains inside HOLD hysteresis.

    The peer must still be fresh, FULL, actually stopped and broadcasting HOLD.
    Capture, exit and task-success thresholds themselves are unchanged.
    """
    @property
    def metadata(self):
        return {**super().metadata, 'script_version': 'leader_square_visibility_upstream_v3',
                'priority_version': 'parked_exit_radius_v1'}

    def _priority_release_radius(self):
        return self.spec.task.exit_radius


def clearance_waypoint(position, goal, circles, maximum=4.):
    """Avoid tangencies that collapse the legal clearance-based speed lease.

    A one-metre navigation margin tapers at the fixed destination. If the
    current pose is already inside that navigation margin, first move radially
    outward on a segment clear of every actual received obstacle circle.
    Physical radii and the subsequent trajectory reviewer are unchanged.
    """
    position = np.asarray(position, float); goal = np.asarray(goal, float)
    if not circles:return visibility_waypoint(position, goal, circles, maximum)
    circles = sorted(circles, key=lambda c: c[0])
    centers = np.asarray([c[1] for c in circles]); radii = np.asarray([c[2] for c in circles])
    distance = np.linalg.norm(position-centers, axis=1)
    if (distance < radii-1e-8).any():return None, False
    margins = np.minimum(1., np.maximum(0., np.linalg.norm(goal-centers, axis=1)-radii-.05))
    index = int(np.argmin(distance-radii-margins))
    if distance[index] < radii[index]+margins[index]-1e-8:
        direction = (position-centers[index])/max(distance[index], 1e-12)
        length = min(maximum, radii[index]+margins[index]+.1-distance[index])
        candidate = position+direction*length
        delta = candidate-position
        fraction = np.clip(((centers-position)*delta).sum(axis=1)/max(float(delta@delta),1e-20),0.,1.)
        if (np.linalg.norm(position+fraction[:,None]*delta-centers,axis=1)>=radii-1e-8).all():
            return candidate, True
    expanded = [(sender, center, radius+margin) for (sender, center, radius),margin in zip(circles,margins)]
    waypoint, detour = visibility_waypoint(position, goal, expanded, maximum)
    if waypoint is None:return visibility_waypoint(position, goal, circles, maximum)
    return waypoint, detour


class LeaderSquareClearanceScript(LeaderSquareVisibilityHysteresisScript):
    @property
    def metadata(self):
        return {**super().metadata, 'script_version': 'leader_square_visibility_upstream_v4',
                'routing_version': 'clearance_margin_v2', 'navigation_margin': 1.}

    def decide(self,pose,actual_velocity,previous_control,terrain,snapshot,now,episode):
        decision=super().decide(pose,actual_velocity,previous_control,terrain,snapshot,now,episode)
        if decision.intent.hold or decision.slot_xy is None:return decision
        circles=[]
        for i,prediction in enumerate(snapshot.predictions):
            if snapshot.state_valid[i] and self._stopped(np.asarray(snapshot.messages[i].actual_velocity)):
                center,_=prediction.at(now)
                radius=max(prediction.at(now+t)[1] for t in (0.,self.spec.grid.horizon))
                circles.append((prediction.sender_id,center,radius))
        waypoint,detour=clearance_waypoint(pose[:2],decision.slot_xy,circles,self.spec.rho_max)
        if waypoint is None:return decision
        return replace(decision,intent=replace(decision.intent,goal_xy=waypoint),
                       reason='clearance_navigation_detour' if detour else 'approach_slot')
