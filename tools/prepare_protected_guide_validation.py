#!/usr/bin/env python3
"""Freeze an already fitted guide and run only fresh proposal-volume controls.

No protein predicates, depletants, classifier calls, fitting or production launch.
The reusable validator reads the copied evidence and proposal-only records.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import sys
import numpy as np
import scipy
from analyze_latent_region import LatentImportanceGuide, shell_log_volume
from check_smc_guide_reference import draw_proposal
from prepare_contact_bank_guides import log_proposal, local_source_closure, sha

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'protected-contact-guide-preparation-v1'
REFERENCE_SCHEMA = 'protected-guide-analytic-volume-reference-v1'
N = 16_384
REFERENCE_SEEDS = tuple(148001010 + 1009*i for i in range(8))
RESERVED_PHYSICAL_SEEDS = tuple(148101010 + 1009*i for i in range(16))
PINS = dict(fit_analysis='4fae6ef241d87d155a07c6d72396d7be427bd03d63730e9e39d805ea762205c6',
    fit_freeze='0f116eb846bbadc5b374964c47be1f5952631d2328ec6c2b4a2b5c36a4e18ff5',
    bank='d0e62d1b23d627582ca5f393af515a1ca10b8ecc01ce1a0e75c40e3e18885929',
    source_guide='20cbf3cc6081d5ce2f434d4189dc890fa2f3c1d1a4fcad320317bea39e8142e8',
    region='924648f4d9db473045397c300b3b4af7ccde8cfda1b1703a6899239ec12e2f02',
    shape='c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9',
    config='340050d107c4e862f62cde293630daf5ae70aecfb5264bd5d8ac0676a589078e',
    reference_region='76ea65088e302d6b6478ac033af9b67b7cf21cb7cea1f854f70cdcd6db0473ae',
    sphere='89c75a30cabfb3e9db6965f6f44f5db52134caa99a495d2c7fbcda2b9da90d34',
    binary='b0e051638276de13336090c21b4f2c8def693d5b6037eac1e2591dba29225094',
    bundle='3de89630a7a797a8e79f13b5c094390dcb6c6a5a1e1368e0f8788b6c83d68e4e')
SCOPE = ('Fixed weights from the completed exploratory protected-stratum fit. All fitting and inspected '
    'holdout populations are prior design evidence, never fresh validation populations. The native-informed '
    '84-component geometry, repaired shape, R4 region, physical measure and classifier are unchanged. '
    'Only fresh proposal-volume controls are generated here; these do not establish protein weights, '
    'mixing, unseen-mode coverage or assembly. No old audits are replayed and no production is launched.')


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def inside(root, name):
    root = Path(root).resolve(); path = (root/name).resolve()
    require(path.is_relative_to(root) and path != root, 'Archive path escapes preparation')
    return path


def file_hashes(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(Path(root).rglob('*')) if p.is_file()}


def verify_freeze(root, expected=None):
    path = Path(root)/'freeze.json'
    require(expected is None or sha(path) == expected, 'Pinned evidence freeze changed')
    frozen = read(path)['files']
    for name, digest in frozen.items():
        require(sha(inside(root, name)) == digest, 'Frozen artifact changed: '+name)
    return frozen


def collect_seeds(value):
    result = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'seed' and type(item) is int:
                result.add(item)
            elif (key == 'seeds' or key.endswith('_seeds')) and isinstance(item, list):
                result.update(x for x in item if type(x) is int)
            result.update(collect_seeds(item))
    elif isinstance(value, list):
        for item in value:
            result.update(collect_seeds(item))
    return result


def seed_inventory(repository, extra_paths=(), excluded=None):
    paths = set(extra_paths)
    for pattern in ('*/protocol.json', '*/declaration.json', '*/proposal-reference/declaration.json',
                    '*/common/proposal-reference/declaration.json'):
        paths.update((Path(repository)/'runs').glob(pattern))
    seeds, sources = set(), {}
    for path in sorted(map(Path, paths)):
        if excluded is not None and path.resolve().is_relative_to(Path(excluded).resolve()):
            continue
        sources[str(path.resolve())] = sha(path); seeds.update(collect_seeds(read(path)))
    require(not set(REFERENCE_SEEDS) & seeds, 'Proposal reference seed collides with prior evidence')
    require(not set(REFERENCE_SEEDS) & set(RESERVED_PHYSICAL_SEEDS), 'Reference/physical seed collision')
    require(not set(RESERVED_PHYSICAL_SEEDS) & seeds, 'Reserved physical seed collides with prior evidence')
    return dict(seeds=sorted(seeds), sources=sources, fresh_reference_seeds=list(REFERENCE_SEEDS),
        reserved_physical_seeds=list(RESERVED_PHYSICAL_SEEDS),
        scope='Top-level campaign protocols/declarations, nested proposal-reference declarations and '
              'explicit guide-provenance plans; raw samples are neither replayed nor parsed.')


def build_protected(bank, source, fit):
    require(len(bank['gaussian_components']) == 80 and len(source['gaussian_components']) == 84,
            'Expected fixed80 baseline and fixed84 source geometry')
    require(bank['region_sha256'] == source['region_sha256'] and
            bank['defensive_uniform_shell_probability'] == source['defensive_uniform_shell_probability'] == .5,
            'Region or defensive probability changed')
    for old, new in zip(bank['gaussian_components'], source['gaussian_components']):
        require(old['mean'] == new['mean'] and old['covariance'] == new['covariance'], 'Old geometry changed')
    require(fit['complete'] is True and fit['optimizer']['success'] is True
            and not fit['missing_training_mandatory_groups'], 'Incomplete protected-stratum fit')
    weights = fit['candidate_weights']['protected_minimax']
    require(len(weights) == 84 and all(type(w) in (float, int) and math.isfinite(w) and w >= 1e-5 for w in weights)
            and abs(sum(weights)-1) < 2e-13, 'Invalid exact normalized fitted weights')
    guide = copy.deepcopy(source)
    for component, weight in zip(guide['gaussian_components'], weights):
        component['weight'] = weight
    parsed = LatentImportanceGuide(guide, bank['region_sha256'])
    require(parsed.count == 84 and parsed.alpha == .5, 'Protected proposal changed')
    return guide


def hoeffding_halfwidth(n, events=10, failure=1e-6):
    require(type(n) is int and n > 0 and events > 0 and 0 < failure < 1, 'Invalid reference bound')
    return math.sqrt(2*math.log(2*events/failure)/n)


def reference_population(arrays, n, seed, logvolume):
    require(set(arrays) == {'u', 'branch', 'component', 'log_q', 'support', 'volume_weight', 'draw'},
            'Incomplete proposal reference archive')
    require(arrays['u'].shape == (n, 6) and np.isfinite(arrays['u']).all()
            and np.array_equal(arrays['draw'], np.arange(n)), 'Original proposal attempts missing')
    support = np.sum(arrays['u']**2, axis=1) <= 16
    q, weights = arrays['log_q'], arrays['volume_weight']
    require(q.shape == weights.shape == (n,) and np.isfinite(q).all()
            and np.array_equal(support, arrays['support']), 'Proposal support changed')
    expected = np.where(support, np.exp(-logvolume-q), 0.)
    require(np.allclose(weights, expected, rtol=0, atol=2e-13) and np.isfinite(weights).all()
            and weights.min() >= 0 and weights.max() <= 2+2e-12, 'Invalid unconditional volume weights')
    require(arrays['branch'].shape == arrays['component'].shape == (n,)
            and arrays['branch'].dtype.kind == 'b' and arrays['component'].dtype.kind in ('i', 'u')
            and np.all(support[~arrays['branch']])
            and np.all(arrays['component'][~arrays['branch']] == -1)
            and np.all(arrays['component'][arrays['branch']] >= 0), 'Proposal branch records changed')
    mean = float(weights.mean()); bound = hoeffding_halfwidth(n)
    return dict(seed=seed, samples=n, mean=mean, maximum_weight=float(weights.max()),
        exterior_zeros=int((~support).sum()), familywise_Hoeffding_halfwidth=bound,
        passed=abs(mean-1) <= bound)


def run_reference(root, guides, code_sources):
    out = root/'proposal-reference'; out.mkdir()
    logvolume = float(shell_log_volume(read(root/'region.json')))
    declaration = dict(schema=REFERENCE_SCHEMA, samples_per_population=N, populations_per_arm=4,
        arms=['bank', 'protected'], alpha=.5, expected_mean=1., weight_bounds=[0., 2.],
        family_failure_probability=1e-6, familywise_events=10, seeds=list(REFERENCE_SEEDS),
        source_sha256={str((root/name).resolve()): sha(root/name) for name in
            ('declaration.json', 'guide-bank.json', 'guide-protected.json', 'region.json', 'seed-inventory.json')},
        code_sha256={str(path.resolve()): sha(path) for path in code_sources.values()},
        runtime=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__), scope=SCOPE)
    (out/'source').mkdir()
    for name, path in code_sources.items():
        shutil.copy2(path, out/'source'/name)
    write(out/'declaration.json', declaration); arms = {}
    for ai, (arm, guide) in enumerate(guides.items()):
        parsed = LatentImportanceGuide(guide, sha(root/'region.json')); populations = []
        for i in range(4):
            seed = REFERENCE_SEEDS[4*ai+i]
            u, branch, component = draw_proposal(np.random.default_rng(seed), guide, N)
            support = np.sum(u*u, axis=1) <= 16; q = parsed.log_density(u, support, logvolume)
            error = float(np.max(np.abs(q-log_proposal(u, guide))))
            require(error < 2e-10, 'Independent proposal densities differ')
            arrays = dict(u=u, branch=branch, component=component, log_q=q, support=support,
                volume_weight=np.where(support, np.exp(-logvolume-q), 0.), draw=np.arange(N))
            filename = f'{arm}-r{i:02d}.npz'; np.savez_compressed(out/filename, **arrays)
            populations.append(dict(reference_population(arrays, N, seed, logvolume), id=f'r{i:02d}',
                archive=filename, archive_sha256=sha(out/filename), maximum_density_difference=error))
        # Cover each center and both sides of every coordinate-axis support boundary.
        witnesses = [c['mean'] for c in guide['gaussian_components']]
        for axis in range(6):
            for sign in (-1, 1):
                for radius in (math.nextafter(4., 0.), 4., math.nextafter(4., math.inf), 8.):
                    u = np.zeros(6); u[axis] = sign*radius; witnesses.append(u.tolist())
        u = np.asarray(witnesses); inside_ball = np.sum(u*u, axis=1) <= 16
        error = float(np.max(np.abs(parsed.log_density(u, inside_ball, logvolume)-log_proposal(u, guide))))
        require(error < 2e-10, 'Boundary/center density witnesses differ')
        mean = float(np.mean([p['mean'] for p in populations])); bound = hoeffding_halfwidth(4*N)
        arms[arm] = dict(populations=populations, mean=mean, analytic_mean=1.,
            familywise_Hoeffding_halfwidth=bound, passed=abs(mean-1) <= bound,
            all_populations_passed=all(p['passed'] for p in populations),
            maximum_witness_density_difference=error, witness_count=len(witnesses))
    report = dict(schema=REFERENCE_SCHEMA, complete=True, arms=arms, total_proposal_only_draws=8*N,
        physical_draws=0, classifiers_rerun=0, audits_replayed=0,
        declaration_sha256=sha(out/'declaration.json'), scope=SCOPE)
    write(out/'validation.json', report); write(out/'freeze.json', dict(files=file_hashes(out)))
    require(all(a['passed'] and a['all_populations_passed'] for a in arms.values()), 'Proposal reference failed')
    return report


def validate_reference(root, plan):
    ref = root/plan['proposal_reference']; verify_freeze(ref)
    d, r = read(ref/'declaration.json'), read(ref/'validation.json')
    require(d['schema'] == r['schema'] == REFERENCE_SCHEMA and r['complete'] is True
        and d['samples_per_population'] == N and d['populations_per_arm'] == 4 and d['arms'] == ['bank', 'protected']
        and d['seeds'] == list(REFERENCE_SEEDS) and d['alpha'] == .5 and d['weight_bounds'] == [0., 2.]
        and d['expected_mean'] == 1. and d['family_failure_probability'] == 1e-6 and d['familywise_events'] == 10,
        'Proposal reference declaration changed')
    require(r['total_proposal_only_draws'] == 8*N and r['physical_draws'] == r['classifiers_rerun'] == r['audits_replayed'] == 0
        and r['declaration_sha256'] == sha(ref/'declaration.json'), 'Proposal reference provenance changed')
    for source, digest in d['code_sha256'].items():
        require(sha(ref/'source'/Path(source).name) == digest, 'Proposal reference source changed')
    required = {sha(root/name) for name in ('declaration.json', 'guide-bank.json', 'guide-protected.json', 'region.json', 'seed-inventory.json')}
    require(required <= set(d['source_sha256'].values()), 'Reference belongs to different prepared inputs')
    logv = float(shell_log_volume(read(root/'region.json')))
    require(set(r['arms']) == {'bank', 'protected'}, 'Reference arm changed')
    for ai, arm in enumerate(('bank', 'protected')):
        results = r['arms'][arm]; require(len(results['populations']) == 4, 'Reference population missing')
        for i, saved in enumerate(results['populations']):
            path = inside(ref, saved['archive']); require(sha(path) == saved['archive_sha256'], 'Reference archive changed')
            with np.load(path, allow_pickle=False) as archive:
                arrays = {k: archive[k] for k in archive.files}
            actual = reference_population(arrays, N, REFERENCE_SEEDS[4*ai+i], logv)
            require(np.all(arrays['component'][arrays['branch']] < (80 if arm == 'bank' else 84)),
                    'Reference selected a nonexistent component')
            require(saved['id'] == f'r{i:02d}' and all(saved[k] == v for k, v in actual.items()),
                    'Proposal reference population accounting changed')
            require(saved['passed'] is True and saved['maximum_density_difference'] < 2e-10, 'Reference population failed')
        mean = float(np.mean([p['mean'] for p in results['populations']]))
        require(results['mean'] == mean and results['familywise_Hoeffding_halfwidth'] == hoeffding_halfwidth(4*N)
            and abs(mean-1) <= hoeffding_halfwidth(4*N) and results['passed'] is results['all_populations_passed'] is True
            and results['maximum_witness_density_difference'] < 2e-10, 'Reference aggregate failed')
    return r


def validate_preparation(root):
    """Portable checked preparation contract; no physics or external raw-data reads."""
    root = Path(root).resolve(); frozen = verify_freeze(root)
    require({'plan.json', 'preparation.json', 'guide-bank.json', 'guide-protected.json',
        'proposal-reference/validation.json', 'sphere-reference.json'} <= set(frozen), 'Incomplete preparation freeze')
    plan, receipt = read(root/'plan.json'), read(root/'preparation.json')
    require(plan['schema'] == receipt['schema'] == SCHEMA and plan['complete'] is receipt['complete'] is True
            and receipt['plan_sha256'] == sha(root/'plan.json'), 'Incomplete preparation receipt')
    require(plan['fit_analysis_sha256'] == PINS['fit_analysis']
        and plan['fit_freeze_sha256'] == PINS['fit_freeze']
        and receipt['guide_sha256'] == plan['guide_sha256']
        and receipt['proposal_only_draws'] == 8*N
        and receipt['physical_draws_launched'] == receipt['native_classifier_calls'] == receipt['audits_replayed'] == 0,
        'Preparation evidence receipt changed')
    require(plan['physical_draws_launched'] == plan['native_classifier_calls'] == plan['audits_replayed'] == 0
            and plan['further_training'] is False and plan['prior_holdouts_are_exploratory'] is True,
            'Preparation scope changed')
    for name, pin, field in [('guide-bank.json', 'bank', None), ('region.json', 'region', 'region_sha256'),
        ('shape.json', 'shape', 'shape_sha256'), ('config.json', 'config', 'config_sha256'),
        ('old-r5-region.json', 'reference_region', 'reference_region_sha256'),
        ('source-guide.json', 'source_guide', None)]:
        require(sha(root/name) == PINS[pin] and (field is None or plan[field] == PINS[pin]), 'Physical/source identity changed: '+name)
    verify_freeze(root/'evidence/fit', PINS['fit_freeze'])
    require(sha(root/'evidence/fit/analysis.json') == PINS['fit_analysis'], 'Exploratory fitted weights changed')
    fit = read(root/'evidence/fit/analysis.json')
    expected = build_protected(read(root/'guide-bank.json'), read(root/'source-guide.json'), fit)
    require(read(root/'guide-protected.json') == expected, 'Prepared guide differs from exact fixed fitted weights')
    require(plan['guide_sha256'] == {a: sha(root/f'guide-{a}.json') for a in ('bank', 'protected')}, 'Prepared guide binding changed')
    inventory = read(root/'seed-inventory.json')
    require(plan['training_seeds'] == inventory['seeds'] and not set(REFERENCE_SEEDS)&set(plan['training_seeds'])
        and not set(RESERVED_PHYSICAL_SEEDS)&set(plan['training_seeds']), 'Reference/training seed collision')
    require(plan['proposal_reference'] == 'proposal-reference' and plan['sphere_reference'] == 'sphere-reference.json',
            'Reference location changed')
    reference = validate_reference(root, plan)
    require(plan['proposal_reference_validation_sha256'] == sha(root/'proposal-reference/validation.json')
        and plan['proposal_reference_freeze_sha256'] == sha(root/'proposal-reference/freeze.json'), 'Proposal evidence binding changed')
    sphere = read(root/'sphere-reference.json')
    require(sha(root/'sphere-reference.json') == PINS['sphere'] == plan['sphere_reference_sha256']
        and sphere['schema'] == 'conditional-ray-sphere-crosslanguage-v1' and sphere['complete'] is True
        and sphere['binary_sha256'] == plan['binary_sha256'] == PINS['binary']
        and sphere['source_bundle_sha256'] == plan['source_bundle_sha256'] == PINS['bundle']
        and [r['activity'] for r in sphere['results']] == [0., 2.]
        and all(r['physical_returncode'] == r['audit_returncode'] == 0 and all(c['passed'] is True for c in r['checks'])
                for r in sphere['results']), 'Reused sphere/executable reference changed')
    return plan


def prepare(out, repository=ROOT):
    out, repository = Path(out).resolve(), Path(repository).resolve()
    require(not out.exists(), 'Fresh preparation directory required; no overwrite or retry')
    fit = repository/'runs/contact-guide-stratum-reweighting-20260923'
    package = repository/'runs/refined-contact-bank-preparation-20260922'
    pilot = repository/'runs/smc-geometry-guide-pilot-20260923'
    geometry = repository/'runs/smc-geometry-guide-preparation-20260923'
    verify_freeze(fit, PINS['fit_freeze']); require(sha(fit/'analysis.json') == PINS['fit_analysis'], 'Fitted evidence changed')
    source = pilot/'smc/provenance/importance-guide.json'
    for name, key in [('region.json', 'region'), ('shape.json', 'shape'), ('config.json', 'config'),
                      ('old-r5-region.json', 'reference_region')]:
        require(sha(package/name) == PINS[key], 'Pinned target changed before reference: '+name)
    require(sha(pilot/'common/sphere-reference.json') == PINS['sphere'], 'Pinned sphere evidence changed')
    require(sha(source) == PINS['source_guide'] and sha(package/'guide-bank.json') == PINS['bank'], 'Source guide changed')
    protected = build_protected(read(package/'guide-bank.json'), read(source), read(fit/'analysis.json'))
    inventory = seed_inventory(repository, (package/'plan.json', geometry/'plan.json', fit/'plan.json'), excluded=out)
    out.mkdir(parents=True); sources = {}
    def copy_input(path, relative):
        path = Path(path); destination = inside(out, relative); destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination); sources[str(path.resolve())] = sha(path)
    for name in ('guide-bank.json', 'region.json', 'shape.json', 'config.json', 'old-r5-region.json'):
        copy_input(package/name, name)
    copy_input(source, 'source-guide.json')
    for name in [*read(fit/'freeze.json')['files'], 'freeze.json']:
        copy_input(inside(fit, name), 'evidence/fit/'+name)
    for path, name in [(package/'plan.json', 'old-bank-plan.json'), (geometry/'plan.json', 'smc-geometry-plan.json'),
                       (pilot/'protocol.json', 'fresh-pilot-protocol.json')]:
        copy_input(path, 'evidence/'+name)
    copy_input(pilot/'common/sphere-reference.json', 'sphere-reference.json')
    write(out/'guide-protected.json', protected); write(out/'seed-inventory.json', inventory)
    code = local_source_closure([Path(__file__), Path(__file__).with_name('test_prepare_protected_guide_validation.py')])
    (out/'source').mkdir()
    for name, path in code.items():
        shutil.copy2(path, out/'source'/name)
    declaration = dict(schema=SCHEMA, complete=False, scope=SCOPE, source_sha256=sources,
        code_sha256={str(path.resolve()): sha(path) for path in code.values()},
        fit_analysis_sha256=PINS['fit_analysis'], fit_freeze_sha256=PINS['fit_freeze'],
        exact_candidate='candidate_weights.protected_minimax', physical_draws_launched=0,
        native_classifier_calls=0, audits_replayed=0, further_training=False,
        prior_holdouts_are_exploratory=True, training_seeds=inventory['seeds'],
        training_seed_scope='Conservative exclusion of all inventoried prior seeds, including inspected holdouts; '
            'not a claim that every inventoried run trained these weights.', proposal_reference='proposal-reference',
        reserved_physical_seeds=list(RESERVED_PHYSICAL_SEEDS))
    write(out/'declaration.json', declaration)
    run_reference(out, {'bank': read(out/'guide-bank.json'), 'protected': protected}, code)
    plan = dict(declaration, complete=True, guide_sha256={a: sha(out/f'guide-{a}.json') for a in ('bank', 'protected')},
        region_sha256=sha(out/'region.json'), shape_sha256=sha(out/'shape.json'), config_sha256=sha(out/'config.json'),
        reference_region_sha256=sha(out/'old-r5-region.json'), sphere_reference='sphere-reference.json',
        sphere_reference_sha256=sha(out/'sphere-reference.json'), binary_sha256=PINS['binary'], source_bundle_sha256=PINS['bundle'],
        proposal_reference_validation_sha256=sha(out/'proposal-reference/validation.json'),
        proposal_reference_freeze_sha256=sha(out/'proposal-reference/freeze.json'),
        normalized_weight_sum=sum(c['weight'] for c in protected['gaussian_components']),
        minimum_component_weight=min(c['weight'] for c in protected['gaussian_components']))
    for path, digest in {**sources, **declaration['code_sha256'], **inventory['sources']}.items():
        require(sha(path) == digest, 'Source changed during preparation: '+path)
    write(out/'plan.json', plan)
    write(out/'preparation.json', dict(schema=SCHEMA, complete=True, plan_sha256=sha(out/'plan.json'),
        guide_sha256=plan['guide_sha256'], proposal_only_draws=8*N, physical_draws_launched=0,
        native_classifier_calls=0, audits_replayed=0, scope=SCOPE))
    write(out/'freeze.json', dict(files=file_hashes(out)))
    return validate_preparation(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    result = validate_preparation(args.out) if args.validate_only else prepare(args.out)
    print(json.dumps(dict(complete=result['complete'], guide_sha256=result['guide_sha256']), indent=2))
