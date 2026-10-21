"""No-learning leader slots; only immutable permitted local observations."""
from dataclasses import dataclass
from copy import deepcopy
import numpy as np
from .actions import PolicyAction
from .contracts import rotation, wrap
from .protocol import frozen_array

VERSION='leader_slot_v1'


@dataclass(frozen=True)
class ScriptMessage:
    sender_id: int
    position: np.ndarray
    goal: np.ndarray | None
    yaw: float | None
    full: bool
    timestamp: float
    expires_at: float
    stop_intent: bool

    def __post_init__(self):
        object.__setattr__(self,'position',frozen_array(self.position))
        if self.goal is not None: object.__setattr__(self,'goal',frozen_array(self.goal))


@dataclass(frozen=True)
class ScriptObservation:
    env_id: int
    agent_id: int
    pose: np.ndarray
    velocity: np.ndarray
    now: float
    terrain: object
    messages: tuple
    neighbors: tuple

    def __post_init__(self):
        object.__setattr__(self,'pose',frozen_array(self.pose))
        object.__setattr__(self,'velocity',frozen_array(self.velocity))


class LeaderSlotPolicy:
    version = VERSION
    state_version = 1
    slot_radius = .8
    capture_radius = .2
    exit_radius = .3
    def __init__(self): self.reset()

    def reset(self,env_ids=None):
        if env_ids is None: self.states={}
        else:
            ids=set(int(e) for e in env_ids)
            self.states={k:v for k,v in self.states.items() if k[0] not in ids}

    def state_dict(self): return {'version':self.version,'state_version':self.state_version,'states':deepcopy(self.states)}

    def load_state_dict(self,state):
        if state.get('version')!=self.version or state.get('state_version')!=self.state_version:
            raise ValueError('Script version mismatch')
        states=deepcopy(state['states'])
        for key,value in states.items():
            if len(key)!=2 or not np.isfinite(value['heading']): raise ValueError('Invalid script state')
        self.states=states

    def action(self,observations,shape):
        physical=np.zeros((*shape,2));hold=np.ones(shape,bool);diagnostics=[]
        for observation in observations:
            e,a=observation.env_id,observation.agent_id
            key=(e,a);reason='leader_hold';slot=None
            if a:
                leader=next((m for m in observation.messages if m.sender_id==0),None)
                if leader is None or not leader.full or leader.goal is None or leader.yaw is None or observation.now>leader.expires_at:
                    reason='missing_or_expired_leader'
                else:
                    if key not in self.states: self.states[key]={'heading':leader.yaw,'holding':False}
                    state=self.states[key]
                    angle=state['heading']+2*np.pi*(a-1)/3
                    slot=leader.goal+self.slot_radius*np.array([np.cos(angle),np.sin(angle)])
                    delta=slot-observation.pose[:2];distance=np.linalg.norm(delta)
                    if state['holding'] and distance>self.exit_radius: state['holding']=False
                    if distance<=self.capture_radius: state['holding']=True
                    yielding=False
                    for peer in observation.neighbors:
                        message=next((m for m in observation.messages if m.sender_id==peer.sender_id),None)
                        if peer.sender_id<a and message is not None and message.full and not message.stop_intent and observation.now<=message.expires_at:
                            points=peer.prediction.positions[:6]
                            own=observation.pose[:2]+np.linspace(0,min(1.,distance),6)[:,None]*delta/max(distance,1e-12)
                            if np.any(np.linalg.norm(own-points,axis=1)<peer.prediction.radii[:6]): yielding=True
                    if state['holding']: reason='slot_hold'
                    elif yielding: reason='priority_yield'
                    else:
                        physical[e,a]=[min(1.6,distance),wrap(np.arctan2(delta[1],delta[0])-observation.pose[2])]
                        hold[e,a]=False;reason='move_to_slot'
            diagnostics.append({'env_id':e,'agent_id':a,'reason':reason,'slot':None if slot is None else slot.tolist()})
        return PolicyAction(physical,hold),diagnostics


class LocalSiteLeaderPolicy(LeaderSlotPolicy):
    """Find a supported local parking patch; no global-map or peer-truth access."""

    version = "leader_local_site_v2"
    state_version = 2
    slot_radius = .6
    capture_radius = .08
    exit_radius = .12
    height_margin = .02
    slope_margin = .02
    probe_capture_radius = .1
    footprint_probe = False
    probe_min_distance = .15
    probe_max_distance = .5

    def __init__(self, radius=.75, height_limit=.18, slope_limit=.25):
        self.radius, self.height_limit, self.slope_limit = radius, height_limit, slope_limit
        super().__init__()

    def state_dict(self):
        return {**super().state_dict(), "configuration": [self.radius, self.height_limit, self.slope_limit]}

    def load_state_dict(self, state):
        if state.get("configuration") != [self.radius, self.height_limit, self.slope_limit]:
            raise ValueError("Local site script configuration mismatch")
        for value in state.get("states", {}).values():
            for field in ("site", "probe"):
                if value.get(field) is not None and (
                    np.asarray(value[field]).shape != (2,) or not np.isfinite(value[field]).all()):
                    raise ValueError("Invalid local site script state")
        super().load_state_dict(state)

    def _select(self, observation):
        samples = observation.terrain.snapshot()
        if not len(samples.xy):
            return None, None, {"reason": "local_site_unknown"}
        delta = samples.xy - observation.pose[:2]
        distances = np.linalg.norm(delta, axis=-1)
        candidates = np.vstack([observation.pose[:2], samples.xy[distances <= 1.2]])
        angles = np.arange(12) * 2*np.pi/12
        offsets = np.vstack([np.zeros((1, 2)), *[
            self.radius*ring/3*np.stack([np.cos(angles), np.sin(angles)], -1) for ring in (1, 2, 3)]])
        points = candidates[:, None, :] + offsets
        feature, known, _ = observation.terrain.lookup(points.reshape(-1, 2))
        feature = feature.reshape(len(candidates), 37, 5)
        known = known.reshape(len(candidates), 37)
        height_range = np.ptp(feature[..., 0], axis=-1)
        slope = np.linalg.norm(feature[..., 1:3], axis=-1).max(axis=-1)
        legal = (known.all(axis=-1) & (feature[..., 4] >= .25).all(axis=-1)
                 & (height_range <= self.height_limit-self.height_margin)
                 & (slope <= self.slope_limit-self.slope_margin))
        site_indices = np.flatnonzero(legal)
        if len(site_indices):
            # Choose the nearest supported feasible patch, deterministic ties.
            chosen = min(site_indices, key=lambda i: (np.linalg.norm(candidates[i]-observation.pose[:2]),
                                                       slope[i], *candidates[i]))
            return candidates[chosen].copy(), None, {"reason": "local_site_selected",
                "height_range": float(height_range[chosen]), "max_slope": float(slope[chosen]),
                "known_nodes": 37, "source": "local_sensor_memory"}
        # Explore a SHORT known segment toward lower observed slope. A probe is
        # never labelled a feasible terminal site; unknown nodes stay unknown.
        eligible = np.flatnonzero((distances >= self.probe_min_distance) & (distances <= self.probe_max_distance))
        candidate_indices = {tuple(point): index for index, point in enumerate(candidates)}
        probes = []
        for index in eligible:
            point = samples.xy[index]
            if observation.terrain.safe(np.linspace(observation.pose[:2], point, 9)).all():
                score = np.linalg.norm(samples.features[index, 1:3])
                if self.footprint_probe:
                    candidate_index = candidate_indices[tuple(point)]
                    score = (height_range[candidate_index]/max(self.height_limit-self.height_margin, 1e-6)
                             + slope[candidate_index]/max(self.slope_limit-self.slope_margin, 1e-6)
                             + .3*(1-known[candidate_index].mean()))
                probes.append((score, distances[index], float(point[0]), float(point[1]), index))
        if probes:
            chosen = min(probes)[-1]
            return None, samples.xy[chosen].copy(), {"reason": "local_site_probe",
                "known_nodes": int(known.sum(axis=-1).max()), "source": "local_sensor_memory"}
        return None, None, {"reason": "local_site_unknown", "known_nodes": int(known.sum(axis=-1).max())}

    def action(self, observations, shape):
        action, diagnostics = super().action(observations, shape)
        physical, hold = action.physical.copy(), action.hold.copy()
        for index, observation in enumerate(observations):
            if observation.agent_id != 0:
                continue
            key = (observation.env_id, 0)
            state = self.states.setdefault(key, {"heading": float(observation.pose[2]),
                "holding": False, "site": None, "probe": None})
            if state["site"] is None:
                site, probe, detail = self._select(observation)
                if site is not None:
                    state["site"] = site
                    state["site_evidence"] = detail
                elif state["probe"] is None or np.linalg.norm(state["probe"]-observation.pose[:2]) <= self.probe_capture_radius:
                    state["probe"] = probe
                diagnostics[index].update(detail)
            else:
                diagnostics[index].update(state["site_evidence"])
            target = state["site"] if state["site"] is not None else state["probe"]
            if target is not None:
                delta = target-observation.pose[:2]
                distance = np.linalg.norm(delta)
                if distance > .05:
                    physical[observation.env_id, 0] = [min(1.6, distance),
                        wrap(np.arctan2(delta[1],delta[0])-observation.pose[2])]
                    hold[observation.env_id, 0] = False
                diagnostics[index]["target"] = target.tolist()
        return PolicyAction(physical, hold), diagnostics


class ConservativeLocalSitePolicy(LocalSiteLeaderPolicy):
    version = "leader_local_site_v3"
    state_version = 3
    slot_radius = .8
    capture_radius = .05
    exit_radius = .08
    height_margin = .04
    slope_margin = .05


class StagedLocalSitePolicy(ConservativeLocalSitePolicy):
    version = "leader_local_site_v4"
    state_version = 4
    probe_capture_radius = .2

    def action(self, observations, shape):
        action, diagnostics = super().action(observations, shape)
        physical, hold = action.physical.copy(), action.hold.copy()
        for index, observation in enumerate(observations):
            if observation.agent_id == 0:
                continue
            leader = next((message for message in observation.messages if message.sender_id == 0), None)
            if leader is not None and leader.full and observation.now <= leader.expires_at and not leader.stop_intent:
                physical[observation.env_id, observation.agent_id] = 0
                hold[observation.env_id, observation.agent_id] = True
                diagnostics[index]["reason"] = "leader_site_search_wait"
        return PolicyAction(physical, hold), diagnostics


class FootprintLocalSitePolicy(StagedLocalSitePolicy):
    version = "leader_local_site_v5"
    state_version = 5
    footprint_probe = True


class ValidatedArrivalSitePolicy(FootprintLocalSitePolicy):
    version = "leader_local_site_v6"
    state_version = 6
    probe_min_distance = .35
    probe_max_distance = .85

    def action(self, observations, shape):
        for observation in observations:
            if observation.agent_id != 0:
                continue
            state = self.states.get((observation.env_id, 0))
            if (state is not None and state["site"] is not None and not state["holding"]
                and np.linalg.norm(state["site"]-observation.pose[:2]) <= .2):
                site, _, evidence = self._select(observation)
                if site is not None and np.linalg.norm(site-observation.pose[:2]) <= 1e-5:
                    state["site"] = site
                    state["holding"] = True
                    state["site_evidence"] = {**evidence, "arrival_validated": True}
        return super().action(observations, shape)


class ConservativeArrivalSitePolicy(ValidatedArrivalSitePolicy):
    """Separate candidate reserves from supported actual-arrival verification."""

    version = "leader_local_site_v7"
    state_version = 7
    arrival_height_margin = .02
    arrival_slope_margin = .02

    def action(self, observations, shape):
        angles = np.arange(12)*2*np.pi/12
        offsets = np.vstack([np.zeros((1, 2)), *[
            self.radius*ring/3*np.stack([np.cos(angles), np.sin(angles)], -1) for ring in (1, 2, 3)]])
        for observation in observations:
            if observation.agent_id != 0:
                continue
            state = self.states.get((observation.env_id, 0))
            if (state is None or state["site"] is None or state["holding"]
                or np.linalg.norm(state["site"]-observation.pose[:2]) > .2):
                continue
            feature, known, _ = observation.terrain.lookup(observation.pose[:2]+offsets)
            height_range = float(np.ptp(feature[:, 0]))
            slope = float(np.linalg.norm(feature[:, 1:3], axis=-1).max())
            if (known.all() and (feature[:, 4] >= .25).all()
                and height_range <= self.height_limit-self.arrival_height_margin
                and slope <= self.slope_limit-self.arrival_slope_margin):
                state["site"] = observation.pose[:2].copy()
                state["holding"] = True
                state["site_evidence"] = {"reason": "local_site_selected",
                    "height_range": height_range, "max_slope": slope,
                    "known_nodes": 37, "source": "local_sensor_memory",
                    "arrival_validated": True, "arrival_reserve": [.02, .02]}
        return super().action(observations, shape)


class AdvancingProbeSitePolicy(ConservativeArrivalSitePolicy):
    """Renew exploratory targets before the near-goal terrain-cost standoff.

    Probe handoff is neither arrival at a feasible site nor permission to HOLD.
    Every new probe still needs a known, passable segment; parking needs its
    separate complete local-region verification.
    """

    version = "leader_local_site_v8"
    state_version = 8
    probe_capture_radius = .3


class LowReliefCaptureSitePolicy(AdvancingProbeSitePolicy):
    """Restore original capture only with complete, low-relief local evidence.

    The extra 0.15 m region accounts for three follower capture offsets in the
    team centroid. This remains a finite-point proxy check, not a safety proof.
    Terrain/control validation and actual task success are unchanged.
    """

    version = "leader_local_site_v9"
    state_version = 9

    def action(self, observations, shape):
        holding_before = {key: value["holding"] for key, value in self.states.items()}
        action, diagnostics = super().action(observations, shape)
        physical, hold = action.physical.copy(), action.hold.copy()
        angles = np.arange(12)*2*np.pi/12
        radius = self.radius + .15
        offsets = np.vstack([np.zeros((1, 2)), *[
            radius*ring/3*np.stack([np.cos(angles), np.sin(angles)], -1) for ring in (1, 2, 3)]])
        for observation, diagnostic in zip(observations, diagnostics):
            if not observation.agent_id or diagnostic["slot"] is None:
                continue
            leader = next((m for m in observation.messages if m.sender_id == 0), None)
            if leader is None or not leader.full or leader.goal is None or not leader.stop_intent or observation.now > leader.expires_at:
                continue
            key = observation.env_id, observation.agent_id
            distance = np.linalg.norm(np.asarray(diagnostic["slot"])-observation.pose[:2])
            capture = .3 if holding_before.get(key, False) else .2
            if distance > capture:
                continue
            feature, known, _ = observation.terrain.lookup(leader.goal+offsets)
            if (known.all() and (feature[:, 4] >= .25).all()
                and np.ptp(feature[:, 0]) <= .08
                and np.linalg.norm(feature[:, 1:3], axis=-1).max() <= .10):
                self.states[key]["holding"] = True
                physical[observation.env_id, observation.agent_id] = 0
                hold[observation.env_id, observation.agent_id] = True
                diagnostic.update(reason="low_relief_slot_hold", capture_radius=.2,
                                  exit_radius=.3, local_region_radius=radius, known_nodes=37)
        return PolicyAction(physical, hold), diagnostics


class ObservedStandstillSitePolicy(LowReliefCaptureSitePolicy):
    """Capture an already stopped follower on observed low-risk support.

    This requests HOLD at its actual position; it does not certify the unknown
    common parking region. Planner checks and original task gates decide that.
    """

    version = "leader_local_site_v10"
    state_version = 10

    def action(self, observations, shape):
        before = {key: value["holding"] for key, value in self.states.items()}
        action, diagnostics = super().action(observations, shape)
        physical, hold = action.physical.copy(), action.hold.copy()
        angles = np.arange(8)*2*np.pi/8
        offsets = np.vstack([np.zeros((1, 2)), .1*np.stack([np.cos(angles), np.sin(angles)], -1)])
        for observation, diagnostic in zip(observations, diagnostics):
            if not observation.agent_id or diagnostic["slot"] is None:
                continue
            leader = next((m for m in observation.messages if m.sender_id == 0), None)
            if leader is None or not leader.full or not leader.stop_intent or observation.now > leader.expires_at:
                continue
            key = observation.env_id, observation.agent_id
            distance = np.linalg.norm(np.asarray(diagnostic["slot"])-observation.pose[:2])
            if distance > (.3 if before.get(key, False) else .2) or abs(observation.velocity[0]) > .05 or abs(observation.velocity[1]) > .1:
                continue
            feature, known, _ = observation.terrain.lookup(observation.pose[:2]+offsets)
            if (known.all() and (feature[:, 4] >= .25).all() and np.ptp(feature[:, 0]) <= .05
                and np.linalg.norm(feature[:, 1:3], axis=-1).max() <= .1):
                self.states[key]["holding"] = True
                physical[observation.env_id, observation.agent_id] = 0
                hold[observation.env_id, observation.agent_id] = True
                diagnostic.update(reason="observed_standstill_slot_hold", capture_radius=.2,
                                  exit_radius=.3, local_region_radius=.1, known_nodes=9,
                                  common_site_certified=False)
        return PolicyAction(physical, hold), diagnostics


def create_script(cfg):
    version = cfg.local_goal_contract.no_learning_script
    if version == "leader_slot_v1":
        return LeaderSlotPolicy()
    if version == "leader_local_site_v2":
        site = cfg.gather_point
        return LocalSiteLeaderPolicy(site.flatness_radius, site.max_height_range, site.max_slope)
    if version == "leader_local_site_v3":
        site = cfg.gather_point
        return ConservativeLocalSitePolicy(site.flatness_radius, site.max_height_range, site.max_slope)
    if version == "leader_local_site_v4":
        site = cfg.gather_point
        return StagedLocalSitePolicy(site.flatness_radius, site.max_height_range, site.max_slope)
    if version == "leader_local_site_v5":
        site = cfg.gather_point
        return FootprintLocalSitePolicy(site.flatness_radius, site.max_height_range, site.max_slope)
    if version == "leader_local_site_v6":
        site = cfg.gather_point
        return ValidatedArrivalSitePolicy(site.flatness_radius, site.max_height_range, site.max_slope)
    if version == "leader_local_site_v7":
        site = cfg.gather_point
        return ConservativeArrivalSitePolicy(site.flatness_radius, site.max_height_range, site.max_slope)
    if version == "leader_local_site_v8":
        site = cfg.gather_point
        return AdvancingProbeSitePolicy(site.flatness_radius, site.max_height_range, site.max_slope)
    if version == "leader_local_site_v9":
        site = cfg.gather_point
        return LowReliefCaptureSitePolicy(site.flatness_radius, site.max_height_range, site.max_slope)
    if version == "leader_local_site_v10":
        site = cfg.gather_point
        return ObservedStandstillSitePolicy(site.flatness_radius, site.max_height_range, site.max_slope)
    raise ValueError("Unsupported diagnostic script")
