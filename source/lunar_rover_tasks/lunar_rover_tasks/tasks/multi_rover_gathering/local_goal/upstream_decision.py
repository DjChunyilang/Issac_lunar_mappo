"""Explicit policy time scale; planner/control clocks retain their old meaning."""
from dataclasses import asdict, dataclass, replace
import math


def action_can_affect_transition(terminated,duration,latency,dt):
    """A request first controls a tick AFTER its activation boundary.

    A true terminal reached before that tick has action-independent reward.
    Keep the transition for value learning and backward credit propagation.
    """
    delay=math.ceil(latency/dt-1e-12)
    return ~terminated | (duration>delay)

@dataclass(frozen=True)
class LearningTimeConfig:
    policy_decision_period_s: float = 1.0
    discount_reference_period_s: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95

    def __post_init__(self):
        if (self.policy_decision_period_s not in (1., 5.)
                or self.discount_reference_period_s not in (.2, 1.)
                or not 0 < self.gamma <= 1 or not 0 < self.lam <= 1):
            raise ValueError('Invalid decision/discount time configuration')

    def discounts(self, dt):
        return (self.gamma ** (dt / self.discount_reference_period_s),
                self.lam ** (dt / self.discount_reference_period_s))

    def steps(self, dt):
        count = round(self.policy_decision_period_s / dt)
        if not math.isclose(count * dt, self.policy_decision_period_s):
            raise ValueError('Decision duration must contain whole control steps')
        return count

    def metadata(self, dt):
        g, l = self.discounts(dt)
        return {'version': 't11_decision_time_v1', **asdict(self),
                'gamma_step': g, 'lambda_step': l, 'control_period_s': dt,
                'budget_unit': 'actual_environment_control_steps'}


def renew_fixed_targets(targets, times, validity):
    return [[replace(intent, timestamp=float(times[e]),
                     expires_at=float(times[e]) + validity) for intent in row]
            for e, row in enumerate(targets)]


def step_decision(bridge, payload, timing, renew_bounds):
    """One original action, fresh 1s tickets, no reset or policy call inside.

    Earlier terminal environments stay inactive while peers finish. Their last
    valid transition (including terminal observations) is retained verbatim.
    """
    env = bridge.env
    if timing is None or timing.policy_decision_period_s == 1.:
        return bridge.step(payload)
    targets, ids = bridge.decode(payload)
    total = None
    observations = None
    count = round(timing.policy_decision_period_s / env.spec.grid.high_level_period)
    for index in range(count):
        if env._ended.all():
            break
        if index:
            renew_bounds(env)
        requests = renew_fixed_targets(targets, env.core.times, bridge.contract.request_seconds)
        env.enqueue(requests, primitive_ids=ids)
        part = env._advance_intents(None)
        active = part['duration'] > 0
        # Preserve the original after-activation, 1s slice boundary for an
        # early terminal. Live rows need only the final decision observation.
        if index == count - 1 or env._ended.all() or (part['terminated'] & active).any():
            actor, critic = env.observe()
            if observations is None:
                observations = (actor, critic)
            else:
                observations[0][active] = actor[active]
                observations[1][active] = critic[active]
        # Capture each renewal's planner outcome before the next plan replaces it.
        for record in part['records']:
            if record.get('event') == 'request_activation':
                plan = env.plans[record['env_id']][record['agent_id']]
                if plan is not None:
                    record['planner_outcome'] = {'valid': bool(plan.valid), 'fallback_reason': plan.fallback_reason}
        if total is None:
            total = part
        else:
            weight = env.reward_tracker.settings.gamma ** total['duration']
            total['reward'] += weight * part['reward']
            for name, values in part['reward_components'].items():
                total['reward_components'][name] += weight * values
            total['duration'] += part['duration']
            for name in ('terminated', 'truncated', 'terminal_reason', 'bootstrap_allowed', 'learning_valid'):
                total[name][active] = part[name][active]
            total['episodes'].extend(part['episodes'])
            total['records'].extend(part['records'])
    return None if total is None else env._transition_from_facts(total,*observations)


def remaining_decisions(budget, used, num_envs, decision_steps, batch_decisions):
    """Reserve a full vector decision, including invalid learning samples."""
    return min(batch_decisions, max(0, (budget - used) // (num_envs * decision_steps)))
