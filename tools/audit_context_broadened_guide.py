"""Complete prospective broadened-guide audit and hard-only deterministic-mixture reduction.

Every frozen stratum draw remains in its arm's unconditional denominator. Only
fixed stride-64 poses receive independent atlas/atomic geometry reconstruction.
No cloud draws, fitting, replacement poses, or equilibrium inference.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp

# Reuse held, independently validated scalar/journal/decoder and geometry APIs.
from audit_context_covariance_guide import (GeometryReference, MapDensityReference,
    bind_map_source, canonical_tokens, close_log, coverage_summary, finite_log,
    require, scalar_row, audit_generation, audit_events, read, sha, SourceDensity)
from analyze_context_covariance_balance import (source_batch, log_component,
    importance_summary, across_populations, REGIONS)

ARMS = ('baseline', 'broadened')
COMPONENTS = ('full', 'diagonal', 'broad_full')
STRATA = {'baseline': (('full',2048),('diagonal',2048)),
          'broadened': (('full',1536),('diagonal',1536),('broad_full',1024))}
COEFFICIENTS = {'baseline': [.5,.25,.125,.125,0.],
                'broadened': [.5,.25,.09375,.09375,.0625]}
BINS = (0.,6.,12.,24.,48.,96.,math.inf)


def expected_identities():
    return {(arm,p,c) for arm in ARMS for p in range(4) for c,_ in STRATA[arm]}


def validate_contract(protocol, expected_sha):
    require(type(expected_sha) is str and len(expected_sha)==64
            and all(c in '0123456789abcdef' for c in expected_sha), 'Invalid allocation digest')
    require(protocol['schema']=='context-broadened-guide-audit-v1'
            and protocol['expected_strata']==20 and protocol['total_draws']==32768
            and protocol['panel_stride']==64 and protocol['expected_panel_size']==512
            and protocol['mahalanobis_squared_bins']==[0,6,12,24,48,96,None], 'Changed audit allocation')
    bindings=protocol['input_sha256']
    def asset(a):
        path=str(Path(a['path']).resolve())
        require(a['sha256']==bindings.get(path)==sha(path), 'Unbound/changed asset: '+path)
        return read(path)
    require(protocol['allocation']['sha256']==expected_sha, 'Wrong expected allocation')
    allocation=asset(protocol['allocation']); mixture=asset(protocol['mixture_manifest'])
    campaign=asset(protocol['campaign']); inventory=asset(protocol['inventory'])
    require(campaign['schema']=='context-broadened-guide-geometry-campaign-v1'
            and campaign['allocation']==protocol['allocation']
            and campaign['mixture_manifest']==protocol['mixture_manifest'], 'Changed campaign identity')
    require(allocation['schema']=='context-broadened-guide-geometry-allocation-v1'
            and allocation['comparison_arms']==list(ARMS) and allocation['populations_per_arm']==4
            and allocation['draws_per_population']==allocation['denominator']==4096
            and allocation['populations']==8 and allocation['stratum_jobs']==20
            and allocation['total_draws']==32768 and allocation['clouds']==0
            and allocation['mode']=='geometry' and allocation['mixture']==[.5,.25,.25]
            and allocation['panel_stride']==64 and allocation['panel_queries']==512
            and allocation['squared_mahalanobis_bin_edges']==[0,6,12,24,48,96,None]
            and allocation['mixture_manifest']==protocol['mixture_manifest'], 'Changed frozen allocation')
    require(mixture['schema']=='context-broadened-source-mixture-v1'
            and mixture['complete'] and mixture['passed']
            and mixture['total_future_attempts']==32768 and mixture['future_stratum_jobs']==20
            and mixture['runner_mixture']==[.5,.25,.25]
            and mixture['guides']==allocation['guides']==protocol['guides']
            and mixture['source_inputs']==allocation['inputs']
            and mixture['source_bundle']==allocation['source_bundle']
            and mixture['source_bundle']['path']==protocol['source_bundle'], 'Mixture provenance differs')
    require(mixture['effective_density_coefficients']==allocation['effective_density_coefficients']==COEFFICIENTS
            and mixture['density_component_order']==allocation['density_component_order']==['uniform','context',*COMPONENTS],
            'Changed complete mixture coefficients/order')
    require(allocation['strata']=={arm:[dict(component=c,draws=n,deterministic_fraction=n/4096)
                                      for c,n in STRATA[arm]] for arm in ARMS}, 'Changed stratum counts')
    require(inventory==protocol['strata'] and len(inventory)==20
            and len({r['id'] for r in inventory})==20
            and len({r['seed'] for r in inventory})==20, 'Incomplete/reused inventory')
    guides={name:dict(asset(protocol['guides'][name]),asset_sha256=protocol['guides'][name]['sha256'])
            for name in COMPONENTS}
    charts=[guides[name]['source_chart'] for name in COMPONENTS]
    require(all(set(c)=={'angular_length','covariance','explicit_gaussian'} for c in charts)
            and all(c['angular_length']==charts[0]['angular_length']
                    and c['explicit_gaussian']['mean']==charts[0]['explicit_gaussian']['mean']
                    and c['explicit_gaussian']['schema']=='source-gaussian-v1' for c in charts), 'Changed chart center/frame')
    f,d,b=(np.asarray(c['covariance'],float) for c in charts)
    require(f.shape==d.shape==b.shape==(6,6) and np.isfinite([f,d,b]).all()
            and np.array_equal(d,np.diag(np.diag(f))) and np.array_equal(b,4*f), 'Changed covariance rule')
    return allocation,guides,mixture


def validate_admitted_config(entry,cfg,digest,guides,identities,campaign_digest):
    arm,p,component=(entry[k] for k in ('comparison_arm','population_index','component'))
    require(arm in ARMS and type(p) is int and p in range(4)
            and (arm,p,component) in expected_identities()
            and (arm,p,component) not in identities, 'Missing/repeated stratum identity')
    index=[c for c,_ in STRATA[arm]].index(component)
    runner_index=5*p+(0 if arm=='baseline' else 2)+index
    n=dict(STRATA[arm])[component]; identity=cfg['identity']
    require(entry['component_index']==index and entry['population']==cfg['population']==runner_index
            and entry['draws']==cfg['draws']==n and entry['panel_ordinals']==list(range(0,n,64))
            and entry['seed']==cfg['seed'] and type(cfg['seed']) is int and 0<=cfg['seed']<2**64
            and entry['config_sha256']==digest
            and identity['kind']=='broadened-source-geometry'
            and identity['comparison_arm']==arm and identity['population_index']==p
            and identity['component']==component and identity['component_index']==index
            and identity['guide_sha256']==guides[component]['asset_sha256']
            and identity['campaign_sha256']==campaign_digest
            and identity['moving_label']==77 and identity['anchor_label']==16
            and cfg['source_chart']==guides[component]['source_chart'], 'Executed config differs from frozen stratum')
    identities.add((arm,p,component))
    return arm,p,component


def arm_log_q(component_logs,coefficients):
    values=np.asarray(component_logs,float); weights=np.asarray(coefficients,float)
    require(values.ndim==2 and values.shape[0]==5 and weights.shape==(5,)
            and np.isfinite(weights).all() and (weights>=0).all() and abs(weights.sum()-1)<1e-15
            and not np.isnan(values).any() and not np.isposinf(values).any(), 'Invalid complete mixture')
    active=weights>0
    return logsumexp(values[active]+np.log(weights[active,None]),axis=0)


def close_density_arrays(actual,expected):
    a,b=np.asarray(actual,float),np.asarray(expected,float)
    require(a.shape==b.shape and not np.isnan(a).any() and not np.isnan(b).any()
            and not np.isposinf(a).any() and not np.isposinf(b).any()
            and np.array_equal(np.isneginf(a),np.isneginf(b)), 'Density support/nonfinite mismatch')
    good=np.isfinite(a); e=np.abs(a[good]-b[good]); tol=2e-8+2e-10*(1+np.abs(a[good])+np.abs(b[good]))
    require(np.all(e<=tol), 'Independent complete source density differs')
    return float(np.max(e/tol)) if len(e) else 0.


def make_contribution(entry,row,source_rows,q,source_logs,radii):
    valid=row['actual']['physical_valid']; region=row['region'] if valid else 'hard_invalid'
    require(type(valid) is bool and region in REGIONS and math.isfinite(q), 'Invalid contribution')
    for name in COMPONENTS:
        require(not math.isnan(float(radii[name])) and radii[name]>=0
                and not math.isnan(float(source_logs[name])) and source_logs[name]!=math.inf, 'Invalid source coordinates')
    patches=row['patches']; full_t=valid and patches['source_fraction']==1.
    require(region!='A_patch_complete' or (full_t and patches['neighbor_labels']==[16,217]), 'Wrong A_T label')
    # Null + explicit seam status represent +infinite radius; no finite clipping.
    radius={c:float(radii[c]) if math.isfinite(radii[c]) else None for c in COMPONENTS}
    item=row['input']; canonical=json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return dict(comparison_arm=entry['comparison_arm'],population_index=entry['population_index'],
        component=entry['component'],stratum_id=entry['id'],runner_population=entry['population'],
        ordinal=item['ordinal'],branch=item['branch'],identity=item['identity'],
        row_id=dict(stratum_id=entry['id'],ordinal=item['ordinal']),source_rows=source_rows,
        canonical_source_row_sha256=hashlib.sha256(canonical).hexdigest(),
        proposed_pose=item['proposed_pose'],region=region,physical_valid=valid,physical_zero=not valid,
        log_q_arm=q,log_q_stratum=row['density']['log_q'],
        log_u=row['density']['log_u'],log_g=row['density']['log_g'],log_g_status=row['density']['log_g_status'],
        log_source={c:float(source_logs[c]) if math.isfinite(source_logs[c]) else None for c in COMPONENTS},
        mahalanobis_squared=radius,source_chart_seam={c:radius[c] is None for c in COMPONENTS},
        log_hard_contribution=-q if valid else None,is_A_T=region=='A_patch_complete',
        source_T_complete_all_regions=bool(full_t),source_fraction=patches['source_fraction'] if valid else None,
        neighbor_labels=patches['neighbor_labels'] if valid else None,
        patch_tokens=patches['tokens'] if valid else None,envelope=row['envelope'],
        physical_weight_status='not_estimated')


def radial_bin(value):
    if value is None:return 5 # Explicit Cayley seam has zero guide density, infinite radius.
    require(math.isfinite(value) and value>=0,'Invalid radial value')
    return min(int(np.searchsorted(BINS,value,side='right'))-1,5)


def radial_summary(records,denominator):
    target=[r for r in records if r['is_A_T']]
    total=importance_summary([r['log_hard_contribution'] for r in target],denominator)
    families=[]
    for chart in ('full','diagonal'):
        groups=[]
        for b in range(6):
            for component in COMPONENTS:
                for branch in ('uniform','context','source'):
                    selected=[r for r in target if r['component']==component and r['branch']==branch
                              and radial_bin(r['mahalanobis_squared'][chart])==b]
                    stats=importance_summary([r['log_hard_contribution'] for r in selected],denominator)
                    groups.append(dict(bin_index=b,lower=BINS[b],upper=BINS[b+1] if b<5 else None,
                        component=component,branch=branch,**stats,
                        fraction_of_A_T_weight=math.exp(stats['log_weight_sum']-total['log_weight_sum']) if stats['hits'] else 0.))
        require(sum(g['hits'] for g in groups)==total['hits'],'Radial partition lost a candidate')
        require(not total['hits'] or abs(sum(g['fraction_of_A_T_weight'] for g in groups)-1)<2e-12,'Radial weights do not close')
        families.append(dict(chart=chart,entries=groups))
    return families


def summarize_records(records,denominator):
    require(len(records)==denominator,'Conditional/omitted attempt denominator')
    partitions=[]
    for region in REGIONS:
        group=[r for r in records if r['region']==region]
        weights=[r['log_hard_contribution'] for r in group if r['physical_valid']]
        partitions.append(dict(region=region,attempted_count=len(group),**importance_summary(weights,denominator)))
    require(sum(p['attempted_count'] for p in partitions)==denominator,'Incomplete region partition')
    target=[r['log_hard_contribution'] for r in records if r['is_A_T']]
    any_t=[r['log_hard_contribution'] for r in records if r['source_T_complete_all_regions']]
    return dict(denominator=denominator,regions=partitions,A_T=importance_summary(target,denominator),
        T_any=importance_summary(any_t,denominator),radial_decomposition=radial_summary(records,denominator),
        all_valid=importance_summary([r['log_hard_contribution'] for r in records if r['physical_valid']],denominator))


def summarize_population(arm,p,records):
    require(len(records)==4096 and all(r['comparison_arm']==arm and r['population_index']==p for r in records),
            'Incomplete/mixed population')
    require({c:sum(r['component']==c for r in records) for c,_ in STRATA[arm]}==dict(STRATA[arm]), 'Wrong stratum fractions')
    return dict(comparison_arm=arm,population_index=p,**summarize_records(records,4096))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--expected-allocation-sha256', required=True)
    args = parser.parse_args()
    protocol = read(args.protocol)
    allocation, guides, mixture = validate_contract(protocol, args.expected_allocation_sha256)
    bindings = protocol['input_sha256']
    for path, digest in bindings.items(): require(sha(path) == digest, 'Changed input: '+path)
    def bound(path):
        key = str(Path(path).resolve())
        require(key in bindings, 'Unbound audit input: '+key)
        return read(key)
    entries = protocol['strata']
    require(bound(protocol['inventory']['path']) == entries, 'Inventory differs from protocol')
    require(len(entries) == 20 and len({r['id'] for r in entries}) == 20,
            'Incomplete population allocation')
    bound(protocol['source_bundle']); bind_map_source(Path(protocol['source_bundle']))
    for entry in entries:
        directory = Path(entry['result'])
        s, receipt = bound(directory/'summary.json'), bound(entry['execution_receipt'])
        require(s['complete'] and s['passed'] and s['mode'] == 'geometry'
                and s['attempted_records'] == s['completed_records'] == s['denominator'] == s['new_poses_generated'] == entry['draws']
                and s['clouds_begun'] == s['clouds_completed'] == s['raw_points'] == s['processed_points'] == 0,
                'Incomplete/nongeometry source population')
        require(receipt['success'] and receipt['child_drained'] and receipt['returncode'] == 0
                and Path(receipt['terminal']['path']).resolve() == (directory/'summary.json').resolve()
                and receipt['terminal']['sha256'] == sha(directory/'summary.json'), 'Population not drained')
    require(not args.out.exists(), 'Fresh audit output required')
    args.out.mkdir(); started = time.process_time()
    stratum_summaries, panel, checks, decodes = [], [], [], []
    records_by_population = {(arm, p): [] for arm in ARMS for p in range(4)}
    seams = {component: 0 for component in COMPONENTS}
    dense_checks = []
    atlas_cache, geometry_cache, identities = {}, {}, set()
    with (args.out/'events.jsonl').open('x') as journal, (args.out/'contributions.jsonl').open('x') as contributions:
        def emit(row):
            journal.write(json.dumps(row, allow_nan=False)+'\n'); journal.flush()
        try:
            for entry in entries:
                emit(dict(kind='population_begun', id=entry['id']))
                cfg = bound(entry['config']); inputs = cfg['inputs']; directory = Path(entry['result'])
                producer, manifest = bound(directory/'protocol.json'), bound(directory/'chart-manifest.json')
                digest = sha(entry['config'])
                require(cfg['schema'] == 'context-source-guide-v1' and cfg['mode'] == 'geometry'
                        and cfg['draws'] == entry['draws'] and cfg['mixture'] == [.5, .25, .25]
                        and producer['config'] == cfg and producer['config_sha256'] == digest
                        and producer['source_bundle_sha256'] == sha(protocol['source_bundle']), 'Changed producer law')
                for p, h in producer['input_sha256'].items():
                    require(bindings.get(str(Path(p).resolve())) == h, 'Missing direct physical/proposal pin')
                for name in ('invocation_config', 'model', 'prior', 'patch_map'):
                    asset = inputs[name]
                    require(bindings.get(str(Path(asset['path']).resolve())) == asset['sha256'], 'Changed input asset binding')
                arm, population_index, component = validate_admitted_config(entry, cfg, digest, guides, identities, protocol['campaign']['sha256'])
                require(cfg['inputs'] == mixture['source_inputs'] and cfg['envelope'] == mixture['envelope'], 'Changed physical/proposal inputs')
                invocation_path = Path(inputs['invocation_config']['path'])
                original = bound(invocation_path)
                require(sha(invocation_path) == inputs['invocation_config']['sha256'], 'Invocation pin differs')
                def physical_asset(name):
                    path = Path(original[name]); path = path if path.is_absolute() else invocation_path.parent/path
                    require(bindings.get(str(path.resolve())) == original['expected_sha256'][name], 'Physical asset pin differs')
                    return bound(path)
                shape, context, source_state = (physical_asset(k) for k in ('shape', 'fixed_context', 'source_state'))
                require(original['depletant_radius'] == 1.5 and original['reservoir_density'] == .035
                        and original['poisson_lambda_ratio'] == 64., 'Changed physical conditions')
                anchor = source_state['anchor_pose']
                require(context['excluded_moving_labels'] == [77]
                        and [b['label'] for b in context['bodies']] == [i for i in range(264) if i != 77]
                        and anchor == next(b['pose'] for b in context['bodies'] if b['label'] == 16), 'Changed anchor/context')
                model = bound(inputs['model']['path']); prior = bound(inputs['prior']['path'])['log_prior']
                key = sha(inputs['model']['path'])
                if key not in atlas_cache: atlas_cache[key] = MapDensityReference(model)
                atlas = atlas_cache[key]
                source = SourceDensity(cfg['source_chart'], source_state['pose'], anchor)
                source_manifest = manifest['source_chart']; params = source_manifest['parameters']
                require(manifest['anchor_pose'] == anchor and manifest['mixture'] == cfg['mixture']
                        and manifest['angular_length'] == source.ell == atlas.density.ell
                        and source_manifest['angular_length'] == source.ell
                        and source_manifest['explicit_gaussian'] == cfg['source_chart']['explicit_gaussian']
                        and np.allclose(params['anchor_position'], source.center['position'], atol=2e-12, rtol=2e-14)
                        and np.allclose(params['anchor_rotation'], source.center_rotation, atol=2e-13, rtol=2e-14)
                        and params['mean'] == cfg['source_chart']['explicit_gaussian']['mean'] and params['weight'] == 1.
                        and np.allclose(params['covariance'], source.covariance, atol=0, rtol=2e-14)
                        and np.allclose(source_manifest['reconstructed_map_lower'], source.lower, atol=0, rtol=2e-14),
                        'Source chart differs from declared guide')
                require(len(manifest['charts']) == len(atlas.branches) == 2048, 'Atlas inventory differs')
                for index, (chart, branch) in enumerate(zip(manifest['charts'], atlas.branches)):
                    require(chart['virtual_label'] == index and chart['component_index'] == branch['component_index']
                            and chart['inverted'] is branch['inverted']
                            and np.allclose(chart['reconstructed_map_lower'], atlas.density.lower[index], atol=2e-14, rtol=2e-14),
                            'Atlas map parameters differ')
                    checks.append(close_log(chart['effective_log_prior'], prior[index], 'context prior'))
                patch_map = bound(inputs['patch_map']['path'])
                key = tuple(original['expected_sha256'][k] for k in ('shape', 'fixed_context', 'source_state'))
                if key not in geometry_cache:
                    geometry_cache[key] = GeometryReference(shape, patch_map['atom_patch_ids'], context,
                        source_state['spherical_wall_radius'], 1.5)
                geometry = geometry_cache[key]; half = manifest['uniform_half_width']
                checks.append(close_log(half, source_state['spherical_wall_radius']+geometry.bound, 'uniform halfwidth'))
                regions = inputs['regions']; reference = regions['source_secondary_tokens']
                require(regions['a_neighbors'] == [16, 217] and regions['b_neighbors'] == [16, 56]
                        and regions['secondary_label'] == 217 and len(canonical_tokens(reference)) == 16
                        and regions['inclusion_boundaries'] == [0., .25, .5, .75, 1.], 'Changed regions')
                require(str((directory/'rows.jsonl').resolve()) in bindings, 'Unbound candidate rows')
                rows = [json.loads(line) for line in (directory/'rows.jsonl').read_text().splitlines()]
                require(len(rows) == entry['draws'], 'Incomplete candidate denominator')
                for ordinal, row in enumerate(rows):
                    checks.extend(scalar_row(row, ordinal, half, reference, source, cfg['mixture']))
                    decodes.append(audit_generation(row['input'], cfg, digest, ordinal, half, source, atlas, anchor))
                    require(row['clouds'] == [] and row['log_physical_contribution'] is None
                            and row['physical_weight_status'] == 'not_estimated', 'Unallocated physical weight')
                    hard = row['log_hard_only_contribution']
                    if row['actual']['physical_valid']: checks.append(close_log(hard, -row['density']['log_q'], 'hard contribution'))
                    else: require(hard is None, 'Hard-invalid contribution differs')
                require(str((directory/'events.jsonl').resolve()) in bindings, 'Unbound journal')
                count = audit_events(directory/'events.jsonl', rows, reference)
                summary = bound(directory/'summary.json'); aggregate = coverage_summary(rows)
                require(summary['identity'] == cfg['identity'] and summary['unconditional_denominator'] is True,
                        'Wrong summary identity/denominator convention')
                require(count == summary['journal_events'] and aggregate['hard_valid'] == summary['physical_valid_records'], 'Summary count differs')
                require(len(summary['regions']) == len(aggregate['partitions']) == 9, 'Incomplete mass partition')
                checks.append(close_log(summary['expected_complete_allocation_raw_points'], aggregate['expected_complete_allocation_raw_points'], 'prospective cost'))
                for group, reported in zip(sorted(aggregate['partitions'], key=lambda r:r['region']), summary['regions']):
                    require(group['region'] == reported['region'] and group['attempted_count'] == reported['count']
                            and reported['log_physical_mass'] is None, 'Wrong unconditional mass partition')
                    if group['attempted_count'] and group['region'] != 'hard_invalid':
                        checks.append(close_log(reported['log_hard_only_mass'], group['log_hard_only_importance_mass'], 'hard-only region mass'))
                    else: require(reported['log_hard_only_mass'] is None, 'Spurious region mass')
                source_rows = dict(path=str((directory/'rows.jsonl').resolve()), sha256=bindings[str((directory/'rows.jsonl').resolve())])
                stratum_summaries.append(dict(id=entry['id'], stratum_id=entry['id'], comparison_arm=arm,
                    population_index=population_index, component=component, source_rows=source_rows,
                    geometry_cpu_seconds=summary['cpu_seconds'], execution_receipt=entry['execution_receipt'], **aggregate))
                poses = [row['input']['proposed_pose'] for row in rows]
                logs_by_source, radii = {}, {}
                for name in COMPONENTS:
                    density = SourceDensity(guides[name]['source_chart'], source_state['pose'], anchor)
                    logs_by_source[name], radii[name], seam = source_batch(density, poses)
                    seams[name] += int(seam.sum())
                log_u = np.asarray([log_component(row['density']['log_u']) for row in rows])
                log_g = np.asarray([finite_log(row['density']['log_g'], row['density']['log_g_status']) for row in rows])
                source_recorded = [finite_log(row['density']['log_source'], row['density']['log_source_status']) for row in rows]
                dense_checks.append(close_density_arrays(source_recorded, logs_by_source[component]))
                matrix = np.stack((log_u, log_g, *(logs_by_source[name] for name in COMPONENTS)))
                q_arm = arm_log_q(matrix, mixture['effective_density_coefficients'][arm])
                q_own = np.asarray([row['density']['log_q'] for row in rows])
                require(np.isfinite(q_arm).all(), 'Zero/unrepresentable arm density at generated pose')
                fraction = entry['draws']/4096
                require(np.all(q_arm >= q_own+math.log(fraction)-2e-8), 'Deterministic-mixture lower bound failed')
                for ordinal, row in enumerate(rows):
                    record = make_contribution(entry, row, source_rows, float(q_arm[ordinal]),
                        {name: logs_by_source[name][ordinal] for name in COMPONENTS},
                        {name: radii[name][ordinal] for name in COMPONENTS})
                    contributions.write(json.dumps(record, allow_nan=False)+'\n')
                    records_by_population[(arm, population_index)].append(record)
                contributions.flush()
                emit(dict(kind='scalar_population_complete', id=entry['id']))
                selected = [rows[i] for i in range(0, entry['draws'], 64)]
                logs = atlas.evaluate([r['input']['proposed_pose'] for r in selected], anchor, prior)[0]
                for row, g in zip(selected, logs):
                    ordinal = row['input']['ordinal']
                    emit(dict(kind='panel_pose_begun', id=entry['id'], ordinal=ordinal))
                    numeric = close_log(finite_log(row['density']['log_g'], row['density']['log_g_status']), float(g), 'panel atlas density')
                    physical = geometry.classify(row['input']['proposed_pose'])
                    require(physical['wall_valid'] is row['actual']['wall_valid']
                            and physical['core_valid'] is row['actual']['core_valid'], 'Independent hard verdict differs')
                    if physical['core_valid'] is True:
                        require([list(t) for t in physical['patch_tokens']] == row['patches']['tokens'], 'Independent patch inventory differs')
                    result = dict(id=entry['id'], ordinal=ordinal, numeric=numeric, geometry=physical)
                    panel.append(result); emit(dict(kind='panel_pose_complete', **result))
            require(len(stratum_summaries) == 20 and len(panel) == 512 and len(decodes) == 32768,
                    'Incomplete audit')
            require(identities == expected_identities(), 'Missing stratum identity')
            populations = [summarize_population(arm, p, records_by_population[(arm,p)])
                           for arm in ARMS for p in range(4)]
            arm_summaries = {}
            for arm in ARMS:
                group = [p for p in populations if p['comparison_arm'] == arm]
                all_records = [r for p in range(4) for r in records_by_population[(arm,p)]]
                arm_summaries[arm] = dict(
                    regions={region: across_populations([next(r for r in p['regions'] if r['region']==region)
                                                        for p in group]) for region in REGIONS},
                    A_T=across_populations([p['A_T'] for p in group]),
                    T_any=across_populations([p['T_any'] for p in group]),
                    pooled=summarize_records(all_records, 16384))
            for path, digest in bindings.items(): require(sha(path) == digest, 'Input changed during audit')
            contributions.flush()
            report = dict(schema='context-broadened-guide-independent-audit-v1', complete=True, passed=True,
                protocol_sha256=sha(args.protocol), source_sha256=sha(__file__), input_sha256=bindings,
                allocation_sha256=args.expected_allocation_sha256, mixture_manifest=protocol['mixture_manifest'],
                guides=protocol['guides'], unconditional_denominator=True, all_attempts=32768,
                scalar_checks=len(checks), maximum_scalar_check_ratio=max(r['ratio'] for r in checks),
                maximum_cross_source_density_check_ratio=max(dense_checks), source_chart_seam_counts=seams,
                decoded_candidates=len(decodes), maximum_position_error=max(r['position_error'] for r in decodes),
                maximum_rotation_error=max(r['rotation_matrix_error'] for r in decodes),
                independent_panel_size=len(panel), panel=panel, stratum_summaries=stratum_summaries,
                populations=populations, arm_summaries=arm_summaries,
                contributions_sha256=sha(args.out/'contributions.jsonl'),
                physical_weight_status='not_estimated', clouds_generated=0, new_poses_generated=0,
                cpu_seconds=time.process_time()-started,
                expectation_identity='For fixed n_j and N=sum n_j, q_arm=sum(n_j/N) q_j; E[(1/N) sum_j sum_i H(X_ji)/q_arm(X_ji)]=integral H dmu. All hard-invalid attempts contribute zero and remain in N.',
                scope='Source-informed fixed-context hard-only region volumes, not physical depletion weights or equilibrium/assembly. Four independent population means determine uncertainty; weight ESS is concentration only. Hard-only tail instability need not imply physical-weight instability. Coarse patches do not define energy or native registry.')
            (args.out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        except BaseException as error:
            emit(dict(kind='fatal', error=repr(error), completed_strata=len(stratum_summaries), completed_panel_poses=len(panel)))
            (args.out/'failure.json').write_text(json.dumps(dict(complete=False, passed=False,
                error=repr(error), prefix_preserved=True), indent=2)+'\n')
            raise


if __name__ == '__main__':
    main()
