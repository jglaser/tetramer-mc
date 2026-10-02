#!/usr/bin/env python3
"""Independent density, geometry and accepted-history audit for guided SMC.

Reuses the checked SMC arithmetic auditor with a different independently
reconstructed proposal. No native labels, new poses or Poisson clouds are
generated. This is an implementation audit, not a convergence assessment.
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from collections import OrderedDict
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from scipy.special import logsumexp
import analyze_r4_smc_control as old
from physical_hard_free_line_vessel import PhysicalHardFreeLineGuide, audit_trace
from hard_free_line_physical_reference import audit_draw
from analyze_mobile_native_pocket import local_sources

require, close, read, sha = old.require, old.close, old.read, old.sha
SCHEMA = 'latent-region-smc-hard-free-initial-guide-v1'
SUMMARY_SCHEMA = 'latent-region-smc-hard-free-initial-guide-summary-v1'


def validate_options(opts):
    require(opts['bridge'] == 'proposal_density' and opts['initial_reference_region'] is None
        and opts['initial_current_probability'] == 1. and opts.get('exclude_native_entry') is None,
        'Guided SMC options changed')
    for key in ('initial_draws', 'population', 'cloud_replicates'):
        require(type(opts[key]) is int and opts[key] > 0, 'Invalid positive allocation: '+key)
    require(type(opts['sweeps_per_stage']) is int and opts['sweeps_per_stage'] >= 0,
        'Invalid mutation allocation')
    require(type(opts['seed']) is int and 0 <= opts['seed'] < 2**64, 'Invalid stream seed')
    require(type(opts['lambda_ratio']) in (int, float) and math.isfinite(opts['lambda_ratio'])
        and opts['lambda_ratio'] > 0, 'Invalid incremental intensity ratio')
    path = opts['schedule']
    require(isinstance(path, list) and len(path) >= 2
        and all(type(v) in (int, float) and math.isfinite(v) for v in path)
        and path[0] == 0 and path[-1] == 1 and all(a < b for a, b in zip(path, path[1:])),
        'Incomplete/nonmonotone bridge schedule')


def validate_zero_summary(summary, first, n):
    require(summary['zero_estimate'] and summary['log_Z'] is None and summary['Z'] == 0.
        and summary['terminal_particles'] == [] and first['zero_estimate']
        and first['log_Z'] is None and first['Z'] == 0.
        and first['particles'] == [] and first['parents'] == []
        and first['initial_draws'] == n and first['initial_hits'] == 0
        and first['activity'] == 0., 'Zero-hit population was refilled or lost')
    old.verify_ancestry([], summary['ancestry'])
    require(summary['ancestry']['largest_initial_family_fraction'] is None, 'Empty genealogy has a largest family')


def validate_positive_summary(summary, log_z, hits):
    close(summary['log_Z'], log_z, 'Terminal normalizer differs')
    close(summary['initial_weight_ESS'], hits, 'Summary initial ESS differs')
    try:
        linear = math.exp(log_z)
    except OverflowError:
        linear = math.inf
    represented = math.isfinite(linear) and linear > 0.
    require(summary['linear_Z_representable'] == represented, 'Linear normalizer flag differs')
    if represented:
        require(type(summary['Z']) in (int, float) and math.isfinite(summary['Z']) and summary['Z'] > 0,
            'Missing represented linear normalizer')
        close(math.log(summary['Z']), log_z, 'Linear normalizer differs')
    else:
        require(summary['Z'] is None, 'Unrepresentable normalizer was replaced by a number')


def validate_generation_branch(component, inside, trace, guide):
    require(type(trace['conditional']) is bool, 'Invalid conditioning flag')
    alpha, beta = guide['defensive_uniform_shell_probability'], guide['conditional_probability']
    if component is None:
        require(inside and not trace['conditional'], 'Uniform initialization left its R4 law')
    else:
        require(type(component) is int and 0 <= component < len(guide['gaussian_components'])
            and alpha < 1., 'Impossible Gaussian initialization branch')
        require((beta != 1. or trace['conditional']) and (beta != 0. or not trace['conditional']),
            'Impossible conditioning branch')


class Proposal:
    """Old audit interface, with bounded memoization of deterministic results."""
    def __init__(self, region, guide, config, shape, region_hash, shape_hash):
        self.current = old.Chart(region)
        self.guide = PhysicalHardFreeLineGuide(region, guide, config, shape,
            region_sha256=region_hash, expected_shape_sha256=shape_hash)
        self.config = config
        self.shape = shape
        self.cache = OrderedDict()
        self.evaluations = 0
        self.maximum_interval_error = 0.
        self.maximum_inverse_CDF_error = 0.
        self.analytic_sphere_cloud_bounds_checked = 0

    def details(self, pose):
        key = old.pose_key(pose)
        if key not in self.cache:
            density = self.guide.evaluate(pose)
            current = self.current.evaluate(pose)
            require(density.latent is not None and not density.structural_zero
                and math.isfinite(density.log_physical_density), 'Missing positive guide support')
            close(density.log_physical_jacobian, current['log_jacobian'], 'Independent Jacobians differ')
            require(np.max(abs(np.asarray(density.latent)-current['latent'])) < 2e-8,
                'Independent chart inverses differ')
            capture = math.dist(pose['position'], self.config['capture_center']) <= self.config['capture_radius']
            hard = bool(capture and current['inside'] and self.guide.recon.hard_valid(
                np.asarray(pose['position']), self.current.rotation(pose['orientation'])))
            self.cache[key] = (current, density, bool(capture), hard)
            self.evaluations += 1
            if len(self.cache) > 8192:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def evaluate(self, pose):
        current, density, _, _ = self.details(pose)
        return current, None, density.log_physical_density

    def check_density_record(self, pose, record):
        current, expected, _, _ = self.details(pose)
        require(record['structural_zero'] is False and record['in_reference_ball'] == current['inside'],
            'Density support certificate differs')
        require(np.max(abs(np.asarray(record['latent'])-expected.latent)) < 2e-8,
            'Density latent coordinates differ')
        for key in ('log_latent_density', 'log_physical_jacobian', 'log_physical_density'):
            close(record[key], getattr(expected, key), 'Complete guide density differs: '+key)
        self.maximum_interval_error = max(self.maximum_interval_error,
            audit_trace(record['hard_free_line_density'], expected, self.guide))
        return expected

    def check_cache(self, particle):
        cache = particle['guide_density_cache']
        require(cache['pose'] == particle['pose'], 'Retained proposal cache belongs to another pose')
        current, density, _, hard = self.details(particle['pose'])
        require(hard and current['inside'], 'Retained particle is hard invalid')
        require(np.max(abs(np.asarray(particle['latent'])-current['latent'])) < 2e-8,
            'Cached particle coordinates differ')
        close(cache['log_g'], density.log_physical_density, 'Retained density cache differs')

    def check_sphere_cloud_bounds(self, pose, clouds, delta_activity):
        """Independent exact lens bound for the one-sphere/one-neighbor limit."""
        if delta_activity == 0 or len(self.shape['atoms']) != 1 or len(self.config['fixed_poses']) != 1:
            return
        atom = self.shape['atoms'][0]; fixed = self.config['fixed_poses'][0]
        center = np.asarray(pose['position']) + self.current.rotation(pose['orientation']) @ atom['center']
        other = np.asarray(fixed['position']) + self.current.rotation(fixed['orientation']) @ atom['center']
        d = float(np.linalg.norm(center-other)); r = atom['radius']+self.config['depletant_radius']
        exact = math.pi*(4*r+d)*max(0.,2*r-d)**2/12
        for cloud in clouds:
            require(cloud['lower_volume'] <= exact+2e-8 and cloud['upper_volume'] >= exact-2e-8,
                'Certified Poisson envelope excludes the analytic sphere lens')
            self.analytic_sphere_cloud_bounds_checked += 1


def audit(directory, binary):
    require(sys.flags.optimize == 0, 'Assertions must remain enabled')
    root = Path(directory).resolve(); ledger = old.Ledger(); started = time.process_time()
    for p in local_sources(__file__).values(): ledger.bind(p)
    state = read(ledger.bind(root/'status.json'))
    require(state['complete'] and state['phase'] == 'complete', 'Incomplete guided SMC population')
    summary = read(ledger.bind(root/'summary.json', state['summary_sha256']))
    manifest = read(ledger.bind(root/'manifest.json', summary['manifest_sha256']))
    require(summary['complete'] and summary['schema'] == SUMMARY_SCHEMA and manifest['schema'] == SCHEMA,
        'Different or incomplete SMC mode')
    binary_path = ledger.bind(binary, manifest['executable_sha256'])
    opts = manifest['options']; binding = manifest['initial_guide']
    validate_options(opts)
    for name, key in [('config.json', 'config_sha256'), ('region.json', 'region_sha256'),
                      ('shape.json', 'shape_sha256'), ('source-bundle.json', 'source_bundle_sha256')]:
        ledger.bind(root/'provenance'/name, manifest[key])
    ledger.bind(root/'provenance/initial-guide.json', binding['sha256'])
    require((root/'provenance/source-bundle.json').read_bytes() in binary_path.read_bytes(),
        'Archived source bundle is not embedded in the provided executable')
    config, region, shape, guide = [read(root/'provenance'/name) for name in
        ('config.json', 'region.json', 'shape.json', 'initial-guide.json')]
    require(binding['schema'] == guide['schema'] == 'defensive-hard-free-line-guide-v1'
        and binding['region_sha256'] == guide['region_sha256'] == manifest['region_sha256']
        and binding['shape_sha256'] == region['shape_sha256'] == manifest['shape_sha256']
        and binding['density_measure'] == 'Lebesgue center volume times normalized SO(3) Haar measure',
        'Guide context or measure changed')
    require(region['activity'] == config['reservoir_density']
        and region['depletant_radius'] == config['depletant_radius']
        and region['capture_center'] == config['capture_center']
        and region['capture_radius'] == config['capture_radius']
        and region['physical_metric'] == config['metadata']
        and region.get('physical_fixed_neighbors', [region['fixed_neighbor']]) == config['fixed_poses']
        and config.get('target_region') is None, 'Physical target context differs')
    require(region['mahalanobis_radius'] == 4. and region.get('minimum_mahalanobis_radius', 0.) == 0.
        and region['minimum_original_q'] == 0. and region.get('minimum_original_q_inclusive', True) is True
        and region.get('maximum_original_q') is None, 'Guided SMC requires full R4 target')
    require(math.isfinite(config['reservoir_density']) and config['reservoir_density'] >= 0
        and math.isfinite(config['poisson_lambda_ratio']) and config['poisson_lambda_ratio'] > 0,
        'Invalid physical activity or mutation intensity')
    for key, expected in dict(uniform_probability=guide['defensive_uniform_shell_probability'],
        conditional_probability=guide['conditional_probability'], raw_translation_axes=guide['raw_translation_axes'],
        minimum_conditional_mass=guide['minimum_conditional_mass'],
        gaussian_component_count=len(guide['gaussian_components'])).items():
        require(binding[key] == expected, 'Manifest guide parameters differ: '+key)
    proposal = Proposal(region, guide, config, shape, manifest['region_sha256'], manifest['shape_sha256'])
    initial_path = ledger.bind(root/'initialization.jsonl', summary['initialization_sha256'])
    stages_path = ledger.bind(root/'stages.jsonl', summary['stages_sha256'])
    attempts_path = ledger.bind(root/'attempts.jsonl', summary['attempts_sha256'])
    attempts = iter(old.jsonlines(attempts_path)); attempt_count = 0
    def attempted(expected):
        nonlocal attempt_count
        require(next(attempts, None) == dict(expected, state='begin'), 'Attempt journal identity/order differs')
        attempt_count += 1
    initial = {}; n = 0; branch = {}
    for i, row in enumerate(old.jsonlines(initial_path)):
        attempted(dict(phase='initialization', draw=i))
        require(row['draw'] == i and i < opts['initial_draws'], 'Lost or reordered initial attempt')
        n += 1
        current, density, capture, hard = proposal.details(row['pose'])
        proposal.check_density_record(row['pose'], row['guide_density'])
        close(row['log_physical_proposal_density'], density.log_physical_density, 'Initial q/J differs')
        close(row['log_proposal_density'], density.log_latent_density, 'Initial latent q differs')
        close(row['log_physical_jacobian'], current['log_jacobian'], 'Initial Jacobian differs')
        close(row['latent_radius'], current['radius'], 'Initial radius differs')
        require(np.max(abs(np.asarray(row['latent'])-current['latent'])) < 2e-8,
            'Initial chart coordinates differ')
        require(row['selected_initial_chart'] == 'initial-guide' and row['current_ball_valid'] == current['inside']
            and row['capture_valid'] == capture and row['hard_valid'] == hard, 'Initial target support differs')
        require(row['log_initial_weight'] == (0. if hard else None), 'Beta-zero weights lost unconditional zeros')
        if hard:
            close(row['log_hard_weight'], -density.log_physical_density, 'Hard-volume diagnostic differs')
            initial[i] = dict(pose=row['pose'], latent=row['latent'], initial_ancestor=i)
        else:
            require(row['log_hard_weight'] is None, 'Invalid row has positive diagnostic mass')
        k = row['proposal_component']
        validate_generation_branch(k, current['inside'], row['hard_free_line_draw'], guide)
        require(np.max(abs(np.asarray(row['selected_latent'])-current['latent'])) < 2e-8,
            'Generated and inverted chart coordinates differ')
        close(row['selected_log_physical_jacobian'], current['log_jacobian'], 'Generated Jacobian differs')
        branch['uniform' if k is None else 'gaussian'] = branch.get('uniform' if k is None else 'gaussian', 0)+1
        adapter = dict(latent=row['selected_latent'], hard_free_line_draw=row['hard_free_line_draw'],
            proposal_branch='uniform-shell' if k is None else 'hard-free-line', proposal_component=k)
        generated = row['initial_guide_draw']
        require(generated['pose'] == row['pose'] and generated['latent'] == row['selected_latent']
            and generated['gaussian_component'] == k
            and generated['hard_free_line_draw'] == row['hard_free_line_draw'],
            'Duplicated generation metadata disagrees')
        close(generated['latent_radius'], float(np.linalg.norm(generated['latent'])), 'Generated radius differs')
        close(generated['log_physical_jacobian'], row['selected_log_physical_jacobian'], 'Generated J aliases differ')
        proposal.maximum_inverse_CDF_error = max(proposal.maximum_inverse_CDF_error,
            audit_draw(adapter, proposal.guide.recon, density.reconstruction))
    require(n == opts['initial_draws'] == summary['initial_draws'] and len(initial) == summary['initial_hits'],
        'Initial attempted denominator or successful count differs')
    stages = iter(old.jsonlines(stages_path)); first = next(stages)
    require(first['stage'] == 0, 'Missing initial resampling stage')
    if not initial:
        validate_zero_summary(summary, first, n)
        require(list(stages) == [], 'Zero-hit population has later stages')
        log_z = None; completed = 0; particles = []
    else:
        log_z = math.log(len(initial)/n)
        close(first['log_Z'], log_z, 'Initial mass is not hits/M')
        close(first['log_Z_increment'], log_z, 'Initial normalizer increment differs')
        require(first['initial_draws'] == n and first['initial_hits'] == len(initial)
            and first['activity'] == 0. and not first.get('zero_estimate', False), 'Initial stage accounting differs')
        close(first['initial_weight_ESS'], len(initial), 'Initial unit-weight ESS differs')
        ancestors = list(initial)
        parents = [ancestors[j] for j in old.systematic_parents([0.]*len(initial), opts['population'], first['resampling_offset'])]
        require(parents == first['parent_initial_draws'], 'Initial resampling differs')
        particles = first['particles']
        require(len(particles) == opts['population'], 'Initial population size differs')
        for p, index in zip(particles, parents):
            require(p['pose'] == initial[index]['pose'] and p['initial_ancestor'] == index,
                'Initial particle lineage differs')
            proposal.check_cache(p)
        old.verify_ancestry(particles, first['ancestry'])
        completed = 0
        for j, stage in enumerate(stages, 1):
            require(j < len(opts['schedule']), 'Unexpected extra annealing stage')
            for i in range(opts['population']):
                attempted(dict(phase='potential', stage=j, particle=i))
                row = stage['potentials'][i]
                proposal.check_sphere_cloud_bounds(row['pose'], row['clouds'], stage['delta_activity'])
            events = {(e['particle'], e['sweep']): e for e in stage['mutation_density_records']}
            for i, parent in enumerate(stage['parents']):
                retained = particles[parent]['pose']
                for sweep in range(opts['sweeps_per_stage']):
                    attempted(dict(phase='mutation', stage=j, particle=i, sweep=sweep, old_pose=retained))
                    event = events.get((i, sweep))
                    if event is not None and event['accepted']: retained = event['proposed_pose']
            for event in stage['mutation_density_records']:
                proposal.check_density_record(event['proposed_pose'], event['proposed_guide_density'])
                require(proposal.details(event['proposed_pose'])[3], 'Recorded valid mutation overlaps a core')
            log_z, _ = old.audit_stage(stage, particles, proposal, config, opts, j, log_z)
            for p in stage['particles']: proposal.check_cache(p)
            particles = stage['particles']; completed = j
        require(completed == len(opts['schedule'])-1 and not summary['zero_estimate'], 'Missing annealing stages')
        validate_positive_summary(summary, log_z, len(initial))
        require(summary['terminal_particles'] == particles, 'Summary lost rejected/retained endpoint')
        old.verify_ancestry(particles, summary['ancestry'])
    require(summary['completed_stage'] == state['completed_stage'] == completed, 'Completion stage differs')
    require(next(attempts, None) is None, 'Unexpected surplus attempted events')
    ledger.recheck()
    return dict(schema='hard-free-initial-guide-smc-independent-audit-v1', complete=True,
        directory=str(root), initial_draws=n, initial_hits=len(initial), initial_branches=branch,
        completed_stage=completed, log_Z=log_z, terminal_particles=len(particles),
        attempted_events=attempt_count,
        guide_evaluations=proposal.evaluations, maximum_interval_error=proposal.maximum_interval_error,
        maximum_inverse_CDF_error=proposal.maximum_inverse_CDF_error,
        analytic_sphere_cloud_bounds_checked=proposal.analytic_sphere_cloud_bounds_checked,
        input_sha256=ledger.files, audit_cpu_seconds=time.process_time()-started,
        new_pose_draws=0, new_Poisson_clouds=0, new_classifier_calls=0,
        scope='Independent full guide geometry/density and hard checks, incremental PGF arithmetic, '
            'resampling, cache binding and accepted history. Random-number independence, exact '
            'geometric Poisson thinning and floating-point execution remain separate implementation '
            'obligations. Geometrically rejected mutation candidates are represented by attempted '
            'identities/counters, not independently reconstructed proposed poses. A campaign controller '
            'must additionally match the options to its predeclared allocation. No physical convergence '
            'or finite-system conclusion.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    require(not args.out.exists(), 'Refuse to overwrite completed audit')
    result = audit(args.directory, args.binary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(complete=result['complete'], initial_draws=result['initial_draws'])))
