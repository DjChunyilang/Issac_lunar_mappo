"""Two bounded closed-loop arms for the combined learning repairs.

Called by run_upstream_t11.py; shares its original sampling/update/checkpoint
driver, but has an explicit 20k physical-step admission and a balanced cohort.
"""
from copy import copy
import json
from pathlib import Path
import sys

import numpy as np

from _common import ROOT
from planner_evidence import file_sha256, json_write

ARMS = {
    'baseline': {'gradient_clipping': 'global', 'initial_continuous_std_scale': 1., 'terminal_actor_credit': 'all'},
    'combo': {'gradient_clipping': 'actor_critic', 'initial_continuous_std_scale': .25, 'terminal_actor_credit': 'effective'},
}


def specification(*, value_normalization='none'):
    result = {'version': 't11_learning_repair_validation_v1', 'representation': 'A1', 'seed': 23,
            'base_steps_per_arm': 20000, 'evaluation_base_step_limit_suite': 8000,
            'owners': 4, 'envs_per_owner': 4, 'initial_cohort': 'near_far_balanced_v1',
            'policy_decision_period_s': 5., 'discount_reference_period_s': 1.,
            'rollout_decisions_per_environment': 4, 'ppo_epochs_per_update': 5,
            'training_near_far_count': [8, 8], 'evaluation_phases': ['initial', 'final'],
            'near_evaluation_scope': 'four training geometries, first heading replica; acquisition probe',
            'arms': ARMS, 'learning_screen': {
                'combo_random_far_net_approach_cases_at_least': 3,
                'combo_random_far_improvement_over_baseline_m': 1.,
                'combo_random_far_improvement_over_own_initial_m': 1.,
                'combo_random_far_dispersion_reduction_at_least': 0.,
                'combo_far_hold_probability_no_higher_than_baseline': True,
                'combo_near_successes_no_lower_than_baseline_each_policy': True,
                'combo_near_successes_above_own_initial_for_at_least_one_policy': True,
                'statistical_significance_claim': False}}
    if value_normalization == 'running':
        result.update(version='t11_value_normalization_comparison_v1', evaluation_base_step_limit_suite=4000,
                      arms={'combo_value_norm': {**ARMS['combo'], 'value_normalization': 'running'}})
        result['learning_screen'] = {'random_far_three_of_four_approach': True,
            'random_far_improvement_over_own_initial_m': 1.,
            'random_far_improvement_over_reference_final_m': 1.,
            'random_far_dispersion_nonincreasing': True, 'far_hold_no_higher_than_reference': True,
            'near_success_no_lower_than_reference_each_policy': True, 'statistical_significance_claim': False}
    return result


def check_initial_reference(initial, reference):
    initial_ref = json.loads((reference/'metrics/evaluation_initial.json').read_text())['results']
    # Identical initial actors, states, RNG and physics must reproduce the panel.
    for panel in initial:
        for key in ('dmax_reduction', 'dispersion_reduction', 'hold_probability', 'successes', 'actual_base_steps'):
            if not np.array_equal(initial[panel][key], initial_ref[panel][key]):
                raise AssertionError(f'Initial reference behavior changed: {panel}/{key}')


def compare_normalization(initial, final, reference):
    initial_ref = json.loads((reference/'metrics/evaluation_initial.json').read_text())['results']
    final_ref = json.loads((reference/'metrics/evaluation_final.json').read_text())['results']
    check_initial_reference(initial, reference)
    result = compare({'baseline': initial_ref, 'combo': initial}, {'baseline': final_ref, 'combo': final})
    # This trial asks about approach learning with parking retained, not an
    # additional requirement that parking itself improve on this short budget.
    result['learning_screen'].pop('near_learns_parking')
    result['value_normalization_screen_passed'] = all(result['learning_screen'].values())
    result.pop('combined_learning_screen_passed')
    result['behavior'] = {'reference_combo': result['behavior']['baseline'],
                          'combo_value_norm': result['behavior']['combo']}
    return result


def check_normalization_reference(args):
    reference = Path(args.comparison_run).resolve()
    manifest = json.loads((reference/'run_manifest.json').read_text())['identity']
    if (manifest['version'] != 't11_learning_repair_validation_v1'
            or manifest['base_config_sha256'] != file_sha256(args.config)
            or manifest['far_checkpoint_sha256'] != file_sha256(args.far_checkpoint)
            or manifest.get('initial_continuous_std_scale') != .25
            or manifest['options'].get('gradient_clipping') != 'actor_critic'
            or manifest['options'].get('terminal_actor_credit') != 'effective'
            or 'value_normalization' in manifest['options']
            or manifest['specification'] != specification()):
        raise ValueError('Reference must be the completed, matching original combo')
    summaries = sorted((reference/'metrics').glob('summary_*.json'))
    if not summaries or not json.loads(summaries[-1].read_text())['budget_complete']:
        raise ValueError('Reference combo budget was not completed')
    return reference


def compare(initial, final):
    def compact(row):
        return {'mean_dmax_reduction_m': float(np.mean(row['dmax_reduction'])),
                'net_approach_cases': int(np.sum(np.asarray(row['dmax_reduction']) > 0)),
                'mean_dispersion_reduction_m2': float(np.mean(row['dispersion_reduction'])),
                'mean_hold_probability': float(np.mean(row['hold_probability'])),
                'successes': row['successes'], 'actual_base_steps': row['actual_base_steps']}
    table = {arm: {phase: {k: compact(v) for k, v in rows[arm].items()}
                    for phase, rows in (('initial', initial), ('final', final))} for arm in ARMS}
    b = table['baseline']['final']; c = table['combo']['final']; start = table['combo']['initial']
    f = c['far_sample']
    checks = {
        'random_far_three_of_four_approach': f['net_approach_cases'] >= 3,
        'random_far_beats_baseline_by_1m': f['mean_dmax_reduction_m'] >= b['far_sample']['mean_dmax_reduction_m'] + 1.,
        'random_far_learns_1m_beyond_initial': f['mean_dmax_reduction_m'] >= start['far_sample']['mean_dmax_reduction_m'] + 1.,
        'random_far_dispersion_nonincreasing': f['mean_dispersion_reduction_m2'] >= 0.,
        'random_far_hold_no_worse': f['mean_hold_probability'] <= b['far_sample']['mean_hold_probability'],
        'near_success_no_worse_each_policy': all(c[k]['successes'] >= b[k]['successes'] for k in ('near_sample', 'near_mean')),
        'near_learns_parking': any(c[k]['successes'] > start[k]['successes'] for k in ('near_sample', 'near_mean')),
    }
    return {'behavior': table, 'learning_screen': checks, 'combined_learning_screen_passed': all(checks.values()),
            'full_gathering_acceptance': False, 'physx_acceptance': False}


def run_experiment(args):
    from upstream_decision_experiment import run_arm
    if (args.budget != 20000 or args.representation != 'A1' or args.seed != 23 or args.samples != 512
            or args.env_owners != 4 or args.num_envs != 16 or args.owner_cpu_workers != 2
            or args.trainer_backend != 'skrl' or args.policy_device != 'cpu' or args.gpu_service
            or args.import_checkpoint or args.import_learner or args.resume_state
            or args.authorize_signal100k or args.updates_this_invocation is not None):
        raise ValueError('Combined repair validation is fresh A1/seed23/K512/4x4/CPU/skrl, 20000 base steps per arm')
    if not args.release_training_pause or not args.far_checkpoint:
        raise ValueError('Explicit research learning release and fixed far panel checkpoint are required')
    if (args.value_normalization == 'running') != bool(args.comparison_run):
        raise ValueError('Value normalization needs the completed combo comparison run; no inherited learning state')
    reference = check_normalization_reference(args) if args.comparison_run else None
    spec = specification(value_normalization=args.value_normalization)
    arms = spec['arms']
    if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in ('.', '..'):
        raise ValueError('Expected a new run name')
    suite = ROOT/'outputs/runs/upstream_scale_t11'/args.run_id
    if suite.exists() or any(suite.with_name(suite.name+'_'+arm).exists() for arm in arms):
        raise FileExistsError(suite)
    json_write(suite/'run_manifest.json', {'specification': spec, 'command': sys.argv,
        **({'comparison_run': str(reference), 'comparison_manifest_sha256': file_sha256(reference/'run_manifest.json')}
           if reference is not None else {}),
        'formal_training': False, 'bounded_research_learning': True, 'system_acceptance_complete': False})
    summaries = {}; initial = {}; final = {}; digests = []
    for arm, settings in arms.items():
        child = copy(args); child.decision_credit_arm = 'C'; child.run_id = args.run_id+'_'+arm
        for key, value in settings.items():
            setattr(child, key, value)
        summary, result = run_arm(child, repair_validation=True)
        path = suite.parent/child.run_id
        digests.append(json.loads((path/'metrics/initialization.json').read_text())['before_exploration_scale_sha256'])
        if len(set(digests)) != 1:
            raise AssertionError('Fresh actor/critic initializations differ before the intended std change')
        if not summary['budget_complete']:
            raise AssertionError('Arm did not complete its reserved budget')
        summaries[arm] = summary; final[arm] = result['results']
        initial[arm] = json.loads((path/'metrics/evaluation_initial.json').read_text())['results']
        if reference is not None:
            reference_digest = json.loads((reference/'metrics/initialization.json').read_text())['before_exploration_scale_sha256']
            if digests[-1] != reference_digest:
                raise AssertionError('Initial network weights differ from the reference combo')
    if sum(row['evaluation_reserved_base_steps'] for row in summaries.values()) > spec['evaluation_base_step_limit_suite']:
        raise AssertionError('Evaluation suite budget exceeded')
    comparison = (compare_normalization(initial['combo_value_norm'], final['combo_value_norm'], reference)
                  if reference is not None else compare(initial, final))
    json_write(suite/'metrics/summary.json', {'arms': summaries, **comparison})
