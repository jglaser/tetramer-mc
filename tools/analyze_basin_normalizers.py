#!/usr/bin/env python3
"""Independent fixed-N aggregation and paired-Poisson diagnostics for normalizers."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

QBINS = ['native_core', 'native_shell', 'shoulder', 'intermediate', 'distant']
REGIONS = [q + '_' + bound for q in QBINS for bound in ['bound', 'unbound']]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(x):
    return float(x) if np.isfinite(x) else None


def moments(logs):
    logs = np.asarray(logs)
    n = len(logs)
    nonzero = int(np.isfinite(logs).sum())
    if not nonzero:
        return dict(draws=n, nonzero=0, logQ=None, ess=0., relative_se=None,
                    max_fraction=None, top_one_percent_fraction=None,
                    coverage='Unresolved: no nonzero observations, not a physical zero or upper bound')
    total = logsumexp(logs)
    ess = float(np.exp(2*total-logsumexp(2*logs)))
    top = max(1, math.ceil(.01*n))
    return dict(draws=n, nonzero=nonzero, logQ=float(total-np.log(n)), ess=ess,
                relative_se=float(np.sqrt(max(0., (n/ess-1)/(n-1)))) if n > 1 else None,
                max_fraction=float(np.exp(max(logs)-total)),
                top_one_percent_fraction=float(np.exp(logsumexp(np.sort(logs)[-top:])-total)),
                top_one_percent_draw_count=top,
                coverage='Observed-sample uncertainty; unseen high-weight regions remain possible')


def paired_noise(logs, pairs):
    """Exact algebraic decomposition of observed per-draw variance, with zeros."""
    if len(logs) < 2 or not np.isfinite(logs).any():
        return None
    offset = max(np.max(logs), np.max(pairs))
    a, b = np.exp(pairs[:, 0]-offset), np.exp(pairs[:, 1]-offset)
    y = (a+b)/2
    total_var = float(np.var(y, ddof=1))
    # E[(W1-W2)^2 / 4] is the noise variance of the two-cloud average.
    cloud_var = float(np.mean((a-b)**2)/4)
    pose_var = total_var-cloud_var
    return dict(log_weight_offset=float(offset), scaled_total_variance=total_var,
        scaled_paired_cloud_variance=cloud_var, scaled_residual_pose_variance=pose_var,
        paired_cloud_variance_fraction=cloud_var/total_var if total_var > 0 else None,
        relative_variance_of_mean_cloud=cloud_var/len(y)/np.mean(y)**2,
        relative_variance_of_mean_pose=pose_var/len(y)/np.mean(y)**2,
        scaled_mean_W1W2=float(np.mean(a*b)), scaled_mean_squared_cloud_difference=float(np.mean((a-b)**2)),
        scope='Paired independent clouds at each pose, with zero weights for invalid draws. Residual pose variance may be negative due to finite sampling; it is not clipped. This diagnostic cannot certify unseen tails.')


def selection(row, name):
    if not row['hard_valid']:
        return False
    if name == 'total':
        return True
    if name == 'native':
        return row['q'] <= 1
    if name == 'other':
        return row['q'] > 1
    if name in QBINS:
        return row['region'].rsplit('_', 1)[0] == name
    if name in ['bound', 'unbound']:
        return row['depletion_contact'] == (name == 'bound')
    return row['region'] == name


def audit_pose_proposal(root, job, manifest, rows):
    """Independent schema2/3 proposal and metric reconstruction for every draw.

    Matrix/SciPy left-Cayley densities are independent of the Rust sampler.
    No sampled weight is changed or conditioned on capture/hard acceptance.
    """
    from prepare_smc_normalizer_atlas import Density, relative_poses, unwrap_proposal_model
    from prepare_deep_far_normalizer_atlas import registration
    assert manifest['schema'] in (2, 3, 4)
    schema = manifest.get('pose_proposal_schema', manifest['schema'])
    assert schema in (1, 2, 3)
    assert schema != 1 or manifest['schema'] == 4
    config=read(root/'config.json')
    raw_config=root/'provenance/input-config.json'
    model_path=root/'provenance/model.json'
    shape_path=root/'provenance/shape.json'
    assert sha(raw_config)==manifest['config_sha256']
    assert sha(model_path)==manifest['model_sha256']
    assert sha(shape_path)==manifest['shape_sha256']
    original_config=read(raw_config)
    for key in ('fixed_poses','capture_center','capture_radius','depletant_radius','metadata'):
        assert config[key]==original_config[key]
    assert config['reservoir_density']==manifest['activity']
    index=manifest['proposal_anchor_index']
    assert index==job.get('proposal_anchor_index')
    assert index is None or type(index) is int
    assert index is not None or schema in (1, 3)
    assert index is None or 0<=index<len(config['fixed_poses'])
    assert manifest['physical_fixed_neighbor_count']==len(config['fixed_poses'])
    raw_model=read(model_path)
    base, flags = unwrap_proposal_model(raw_model)
    assert base.get('dfs') is None or all(v is None for v in base['dfs']), 'Audit supports Gaussian components only'
    assert base['shape_sha256']==manifest['shape_sha256']
    if schema == 3:
        assert any(flags), 'Schema3 requires active reciprocal components'
        assert manifest['proposal_model_kind'] == 'reciprocal-pose-mixture-v1'
        assert type(manifest['base_component_count']) is int and manifest['base_component_count'] == len(flags)
        assert type(manifest['virtual_component_count']) is int and manifest['virtual_component_count'] == len(flags)+sum(flags)
        assert all(type(v) is bool for v in manifest['reciprocal_components'])
        assert manifest['reciprocal_components'] == flags
        assert manifest['covariance_scale'] == 1., 'Reciprocal density requires unchanged scale1 model'
        model = raw_model
        source_path = root/'provenance/source-bundle.json'
        assert sha(source_path) == manifest['source_bundle_sha256']
        source = read(source_path)
        assert source['files'], 'Source bundle must identify the executable sources'
        for entry in source['files'].values():
            assert hashlib.sha256(entry['text'].encode()).hexdigest() == entry['sha256']
    else:
        assert not any(flags), 'Active reciprocal model requires schema3'
        # An all-false envelope has exactly the legacy law, including scaling.
        model=dict(base,covariances=(np.asarray(base['covariances'])*manifest['covariance_scale']**2).tolist())
    actual=[r for r in rows if r.get('pose') is not None]
    assert len(actual)==len(rows), 'Open selected-anchor proposal has no intentional null: numerical failures require diagnosis'
    assert actual, 'Positive fixed sample budget is required'
    poses=[r['pose'] for r in actual]
    if schema == 3:
        from normalizer_proposal_density import NormalizerProposalDensity
        density = NormalizerProposalDensity(model, source_bundle=root/'provenance/source-bundle.json')
    else:
        density = Density(model)
    indices = [index] if index is not None else list(range(len(config['fixed_poses'])))
    assert indices
    log_gaussians = np.asarray([density.evaluate(relative_poses(poses,config['fixed_poses'][i]))[0]
                               for i in indices])
    displacement=np.asarray([p['position'] for p in poses])-config['capture_center']
    radius=config['capture_radius'];epsilon=manifest['uniform_probability']
    assert np.isfinite(radius) and radius > 0 and np.isfinite(epsilon) and 0 < epsilon <= 1
    cube=np.all((displacement>=-radius)&(displacement<radius),axis=1)
    uniform=np.where(cube,np.log(epsilon)-3*np.log(2*radius),-np.inf)
    anchor_densities=np.logaddexp(uniform[None,:],np.log1p(-epsilon)+log_gaussians) if epsilon<1 else np.broadcast_to(uniform,log_gaussians.shape)
    expected_density=logsumexp(anchor_densities,axis=0)-np.log(len(indices))
    recorded_density=np.asarray([r['log_proposal_density'] for r in actual])
    error=np.abs(recorded_density-expected_density)
    assert np.isfinite(error).all() and np.max(error)<2e-8
    capture=np.linalg.norm(displacement,axis=1)<=radius
    assert np.array_equal(capture,np.asarray([r['capture_valid'] for r in actual]))
    expected_q=registration(poses,config['metadata'])
    qerrors=[]
    for row,q,inside in zip(actual,expected_q,capture):
        if row['hard_valid']:
            assert inside and row['q'] is not None
            qerrors.append(abs(q-row['q']))
            assert qerrors[-1]<2e-8
        else:
            assert row['q'] is None and row['region'] is None
    label_counts = defaultdict(int)
    if schema == 3:
        initial_displacement = np.asarray(config['initial_pose']['position'])-config['capture_center']
        old_cube = np.all((initial_displacement >= -radius)&(initial_displacement < radius))
        old_uniform = np.log(epsilon)-3*np.log(2*radius) if old_cube else -np.inf
        old_gaussians = np.asarray([density.evaluate(relative_poses([config['initial_pose']],config['fixed_poses'][i]))[0][0]
                                    for i in indices])
        old_densities = np.logaddexp(old_uniform,np.log1p(-epsilon)+old_gaussians) if epsilon<1 else np.full(len(indices),old_uniform)
        for n, row in enumerate(actual):
            proposal = row['proposal']
            assert type(proposal['moving_index']) is int and proposal['moving_index'] == 0
            anchor = proposal['anchor_index']
            assert type(anchor) is int and 1 <= anchor <= len(indices), 'Invalid local anchor label'
            assert proposal['null_reason'] is None and proposal['candidate'] is not None
            candidate = proposal['candidate']
            assert np.allclose(candidate['position'],displacement[n],rtol=0,atol=2e-8)
            a, b = np.asarray(candidate['orientation']), np.asarray(row['pose']['orientation'])
            assert a.shape == b.shape == (4,) and min(np.linalg.norm(a-b),np.linalg.norm(a+b)) < 2e-8
            branch = proposal['branch']
            component = proposal['component_index']
            if branch == 'uniform':
                assert cube[n], 'Uniform draw must be in the half-open lab cube'
                assert component is None and 'component_inverted' not in proposal
                label = f'anchor{indices[anchor-1]}:uniform'
            else:
                assert branch == 'learned' and epsilon < 1
                assert type(component) is int and 0 <= component < len(flags), 'Label must be a stored base component index'
                inverted = proposal['component_inverted']
                assert type(inverted) is bool and (not inverted or flags[component]), 'Unsupported reciprocal branch label'
                label = f'anchor{indices[anchor-1]}:component{component}:inverted{inverted}'
            label_counts[label] += 1
            new, old = anchor_densities[anchor-1,n], old_densities[anchor-1]
            assert abs(proposal['new_log_density']-new) < 2e-8, 'Selected-anchor density mismatch'
            assert abs(proposal['old_log_density']-old) < 2e-8
            assert abs(proposal['log_reverse_forward']-(old-new)) < 2e-8
            if row['hard_valid']:
                assert type(row['depletion_contact']) is bool
                assert abs(row['log_hard_weight']+expected_density[n]) < 2e-8
            else:
                assert row['depletion_contact'] is None and row['log_hard_weight'] is None
                assert row['log_importance_weight'] is None and not row['clouds']
    # The normalizer's model list contains only one selected frame; logged
    # local anchor index1 refers to that list, while manifest stores the
    # physical-neighbor index. Do not silently reinterpret one as the other.
    result = dict(checked_actual_poses=len(actual),checked_q=len(qerrors),
        maximum_log_density_error=float(max(error)),maximum_q_error=max(qerrors,default=0.),
        proposal_anchor_index=index,physical_fixed_neighbor_count=len(config['fixed_poses']),
        oracle='Independent full Gaussian sum with exact left-Cayley/Haar Jacobian plus half-open lab cube; no retry or valid-only normalization')
    if schema == 3:
        result.update(proposal_model_kind=manifest['proposal_model_kind'],base_component_count=len(flags),
            virtual_component_count=len(flags)+sum(flags),reciprocal_components=flags,
            proposal_anchor_indices=indices,checked_proposal_labels=len(actual),proposal_label_counts=dict(label_counts),
            oracle='Independent exact reciprocal physical Gaussian sum, composed with each physical anchor, plus half-open lab cube/Haar; all selected anchors marginalized; no retry or valid-only normalization')
        result['factor_audit']=dict(algorithm=density.algorithm_metadata,components=density.factor_diagnostics)
    return result


def audit_wall_domain(root, manifest, rows, summary):
    """Independent transformed-atom wall check; never clip the ideal bath."""
    from scipy.spatial.transform import Rotation
    assert manifest['schema'] == 4 and manifest['pose_proposal_schema'] in (1, 2, 3)
    assert manifest['bath_wall_permeable'] is True
    source = root/'provenance/source-bundle.json'
    assert sha(source) == manifest['source_bundle_sha256']
    for entry in read(source)['files'].values():
        assert hashlib.sha256(entry['text'].encode()).hexdigest() == entry['sha256']
    config = read(root/'config.json')
    atoms = read(root/'provenance/shape.json')['atoms']
    centers = np.asarray([a['center'] for a in atoms], float)
    radii = np.asarray([a['radius'] for a in atoms], float)
    bound = np.max(np.linalg.norm(centers, axis=1)+radii)
    assert abs(manifest['shape_bound']-bound) < 2e-10
    wall = manifest['atomic_wall'];center = np.asarray(wall['center'],float);radius = wall['radius']
    assert center.shape == (3,) and np.isfinite(center).all()
    assert math.isfinite(radius) and radius > 0 and np.all(radii <= radius)
    required = np.linalg.norm(center-config['capture_center'])+radius+bound
    assert config['capture_radius'] >= required+256*np.finfo(float).eps*(1+required)
    def contains(pose):
        q = np.asarray(pose['orientation'])
        rotation = Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
        positions = centers@rotation.T+np.asarray(pose['position'])-center
        return bool(np.all(np.sum(positions**2,axis=1) <= (radius-radii)**2))
    for pose in config['fixed_poses']:
        assert contains(pose), 'Fixed scaffold leaves the atomic wall'
    wall_rejected=0
    for row in rows:
        assert row['pose'] is not None, 'Full-wall proposals must not censor numerical nulls'
        valid=contains(row['pose'])
        assert type(row['wall_valid']) is bool and row['wall_valid']==valid
        if valid:
            assert row['capture_valid'], 'Capture truncates a wall-valid pose'
        else:
            assert not row['hard_valid'] and row['log_importance_weight'] is None
            assert row['log_hard_weight'] is None and row['q'] is None
            assert row['region'] is None and row['depletion_contact'] is None and not row['clouds']
            wall_rejected += int(row['capture_valid'])
    assert summary['wall_rejected']==wall_rejected
    return dict(checked_poses=len(rows),wall_rejected_inside_capture=wall_rejected,
                atomic_wall=wall,bath_wall_permeable=True,
                oracle='Direct transformed atomic sphere squared distances; complete capture enclosure checked independently; no bath-wall clipping')


def audit_selected_anchor(root, job, manifest, rows):
    """Compatibility entry point for the unchanged schema2 selected-anchor audit."""
    assert manifest['schema'] == 2
    return audit_pose_proposal(root, job, manifest, rows)


def population(job):
    from prepare_smc_normalizer_atlas import unwrap_proposal_model
    root = Path(job['directory'])
    summary, manifest = read(root/'summary.json'), read(root/'manifest.json')
    assert ('atomic_wall' in manifest) == (manifest['schema'] == 4), 'Atomic wall requires its explicit domain schema'
    _, flags = unwrap_proposal_model(read(root/'provenance/model.json'))
    proposal_schema=manifest.get('pose_proposal_schema', manifest['schema'])
    assert any(flags) == (proposal_schema == 3), 'Active reciprocal model and population schema disagree'
    assert summary['complete'] and manifest['seed'] == job['seed']
    assert summary['manifest']==manifest
    assert manifest['covariance_scale'] == job['covariance_std_scale']
    rows = [json.loads(line) for line in (root/'samples.jsonl').read_text().splitlines()]
    assert [r['draw'] for r in rows] == list(range(job['samples']))
    assert manifest['samples'] == len(rows) and manifest['cloud_replicates'] == 2
    proposal_audit=None
    if manifest['schema'] in (2, 3, 4):
        assert summary['numerical_nulls']==0 and all(r.get('pose') is not None for r in rows)
        proposal_audit=audit_pose_proposal(root,job,manifest,rows)
        if manifest['schema'] == 4:
            proposal_audit['wall_domain']=audit_wall_domain(root,manifest,rows,summary)
    else:
        assert manifest['schema']==1
        assert manifest.get('proposal_anchor_index') is None and job.get('proposal_anchor_index') is None
    for row in rows:
        if row['hard_valid']:
            assert row['capture_valid'] and row['region'] in REGIONS
            expected = ('native_core' if row['q'] <= .8 else 'native_shell' if row['q'] <= 1
                        else 'shoulder' if row['q'] < 2 else 'intermediate' if row['q'] < 5 else 'distant')
            assert row['region'] == expected + ('_bound' if row['depletion_contact'] else '_unbound')
            assert len(row['clouds']) == 2
            lw = logsumexp([c['log_weight'] for c in row['clouds']])-np.log(2)-row['log_proposal_density']
            assert abs(lw-row['log_importance_weight']) < 1e-10
            for cloud in row['clouds']:
                expected_weight = manifest['activity']*cloud['lower_volume'] + cloud['overlap_points']*np.log1p(manifest['activity']/manifest['lambda'])
                assert abs(expected_weight-cloud['log_weight']) < 1e-10
        else:
            assert row['log_importance_weight'] is None and not row['clouds']
    names = ['total', 'native', 'other', 'bound', 'unbound'] + QBINS + REGIONS
    estimates = {}
    for name in names:
        logs = np.array([r['log_importance_weight'] if selection(r, name) else -np.inf for r in rows])
        pairs = np.array([[c['log_weight']-r['log_proposal_density'] for c in r['clouds']]
                          if selection(r, name) else [-np.inf, -np.inf] for r in rows])
        estimates[name] = moments(logs)
        estimates[name]['paired_noise'] = paired_noise(logs, pairs)
        if name in summary['estimates']:
            recorded = summary['estimates'][name]['log_normalizer']
            assert recorded is None if estimates[name]['logQ'] is None else abs(recorded-estimates[name]['logQ']) < 1e-10
    region_logs = [estimates[k]['logQ'] for k in REGIONS if estimates[k]['logQ'] is not None]
    if region_logs:
        assert abs(logsumexp(region_logs)-estimates['total']['logQ']) < 1e-10
    else:
        assert estimates['total']['logQ'] is None
    components = {}
    labels = sorted({str(r['proposal'].get('component_index')) for r in rows})
    for label in labels:
        selected = [r for r in rows if str(r['proposal'].get('component_index')) == label]
        logs = np.array([r['log_importance_weight'] if r['hard_valid'] and str(r['proposal'].get('component_index')) == label else -np.inf for r in rows])
        components['uniform' if label == 'None' else label] = dict(draws=len(selected),
            hard_valid=sum(r['hard_valid'] for r in selected), contribution=moments(logs))
    bounds = [manifest['activity']**2*c['uncertain_volume']/manifest['lambda'] for r in rows if r['hard_valid'] for c in r['clouds'][:1]]
    return dict(job=job, estimates=estimates, components=components, summary=summary,proposal_audit=proposal_audit,
                conditional_cloud_log_relative_second_moment_upper_quantiles=dict(zip(['min','median','p90','max'], np.quantile(bounds,[0,.5,.9,1]).tolist())) if bounds else None,
                rows=rows)


def audit_campaign_domain(physical, population_manifest):
    """An omitted wall CLI flag must not silently change the campaign target."""
    has_wall = 'atomic_wall' in physical
    assert (population_manifest['schema'] == 4) == has_wall, 'Campaign and population wall domains differ'
    assert ('atomic_wall' in population_manifest) == has_wall
    if has_wall:
        assert physical['atomic_wall'] == population_manifest['atomic_wall']
        assert physical['bath_wall_permeable'] is True
        assert population_manifest['bath_wall_permeable'] is True


def analyze(root, out=None):
    from prepare_smc_normalizer_atlas import unwrap_proposal_model
    manifest = read(root/'manifest.json')
    _, reciprocal_flags = unwrap_proposal_model(read(root/'provenance/model.json'))
    reciprocal = any(reciprocal_flags)
    if manifest.get('proposal_anchor_index') is not None or reciprocal or manifest['physical'].get('atomic_wall'):
        for name,digest in manifest['archive_sha256'].items():
            assert sha(root/'provenance'/name)==digest, f'Frozen campaign input changed: {name}'
    complete, pending = [], []
    for job in manifest['jobs']:
        path = Path(job['directory'])/'summary.json'
        (complete if path.exists() and read(path).get('complete') else pending).append(job)
    populations = [population(j) for j in complete]
    for p in populations:
        m = p['summary']['manifest']
        audit_campaign_domain(manifest['physical'], m)
        assert m['model_sha256'] == manifest['archive_sha256']['model.json']
        assert m['config_sha256'] == manifest['archive_sha256']['config.json']
        assert m['shape_sha256'] == manifest['archive_sha256']['shape.json']
        assert m['executable_sha256'] == manifest['archive_sha256']['basin-normalizer']
        assert m['activity'] == manifest['physical']['activity']
        assert m.get('proposal_anchor_index')==manifest.get('proposal_anchor_index')
        if m['schema'] in (2, 3, 4):
            cfg=read(Path(p['job']['directory'])/'config.json')
            assert cfg['fixed_poses']==manifest['physical']['fixed_poses']
            assert cfg['capture_center']==manifest['physical']['capture_center']
            assert cfg['capture_radius']==manifest['physical']['capture_radius']
            assert cfg['depletant_radius']==manifest['physical']['depletant_radius']
        if m['schema'] in (3, 4):
            assert m['uniform_probability'] == manifest['physical']['uniform_probability']
            assert m['physical_fixed_neighbor_count'] == len(manifest['physical']['fixed_poses'])
            if 'source-bundle.json' in manifest['archive_sha256']:
                assert m['source_bundle_sha256'] == manifest['archive_sha256']['source-bundle.json']
        assert math.isclose(m['lambda'], m['activity']*manifest['physical']['lambda_ratio'], rel_tol=1e-14)
    groups = {}
    for scale in sorted({j['covariance_std_scale'] for j in complete}):
        pops = [p for p in populations if p['job']['covariance_std_scale'] == scale]
        rows = sum([p['rows'] for p in pops], [])
        estimates = {}
        for name in pops[0]['estimates']:
            logs = np.array([r['log_importance_weight'] if selection(r,name) else -np.inf for r in rows])
            pairs = np.array([[c['log_weight']-r['log_proposal_density'] for c in r['clouds']]
                if selection(r,name) else [-np.inf,-np.inf] for r in rows])
            estimates[name] = moments(logs)
            estimates[name]['paired_noise'] = paired_noise(logs,pairs)
            population_logs = np.array([p['estimates'][name]['logQ'] if p['estimates'][name]['logQ'] is not None else -np.inf for p in pops])
            estimates[name]['independent_population_estimates'] = moments(population_logs)
            estimates[name]['independent_population_estimates']['logQ_values'] = [finite(x) for x in population_logs]
            assert estimates[name]['logQ'] is None if not np.isfinite(population_logs).any() else abs(logsumexp(population_logs)-np.log(len(pops))-estimates[name]['logQ']) < 1e-10
        total = estimates['total']['logQ']
        fractions = {name: np.exp(v['logQ']-total) if v['logQ'] is not None and total is not None else None for name,v in estimates.items()}
        def delta(a,b):
            return estimates[b]['logQ']-estimates[a]['logQ'] if estimates[b]['logQ'] is not None and estimates[a]['logQ'] is not None else None
        components = {}
        for label in sorted({str(r['proposal'].get('component_index')) for r in rows}):
            logs = np.array([r['log_importance_weight'] if r['hard_valid'] and str(r['proposal'].get('component_index')) == label else -np.inf for r in rows])
            c = moments(logs)
            components['uniform' if label == 'None' else label] = dict(estimates=c,
                fraction_of_observed_total=np.exp(c['logQ']-total) if c['logQ'] is not None else None,
                selected_draws=sum(str(r['proposal'].get('component_index')) == label for r in rows))
        groups[str(scale)] = dict(covariance_std_scale=scale, populations=len(pops), unconditional_draws=len(rows),
            model_sha256=pops[0]['summary']['manifest']['model_sha256'],
            config_sha256=pops[0]['summary']['manifest']['config_sha256'],
            poisson_lambda=pops[0]['summary']['manifest']['lambda'],
            poisson_lambda_ratio=manifest['physical']['lambda_ratio'], activity=manifest['physical']['activity'],
            estimates=estimates, fraction_of_observed_total=fractions, selected_component_contributions=components,
            F_native_minus_other_kBT=delta('native','other'), F_native_minus_distant_kBT=delta('native','distant'),
            sampler_cpu_seconds=sum(p['summary']['sampler_cpu_seconds'] for p in pops))
    for p in populations:
        p.pop('rows')
    if out is None:
        out = root/'assessment';out.mkdir(exist_ok=True)
    else:
        out = Path(out).resolve()
        assert not out.exists(), 'Use a fresh separate assessment output'
        out.mkdir(parents=True)
    provenance = {str(Path(j['directory'])/name):sha(Path(j['directory'])/name) for j in complete for name in ['summary.json','samples.jsonl','manifest.json']}
    provenance[str(Path(__file__).resolve())]=sha(__file__)
    provenance[str(root/'manifest.json')]=sha(root/'manifest.json')
    result = dict(groups=groups, populations=populations, pending=pending, provenance=provenance,
        limitation='Independent populations per scale are exploratory. Fixed-N invalid zeros are retained. Missing bins remain unresolved, ratios are plug-in estimates, and observed ESS cannot establish tail coverage. Selected proposal components are auxiliary labels, not physical basins.',
        cloud_bound='For one cloud, E[W^2|pose]/E[W|pose]^2 <= exp(z^2 uncertain_volume/lambda). Quantiles report the logarithm of this conditional geometric upper bound, not a measured RSE.')
    (out/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Independent normalizer pilot','',result['limitation'],'',f'Completed {len(complete)}/{len(manifest["jobs"])} populations.',
        f'Frozen model SHA-256: `{manifest["archive_sha256"]["model.json"]}`. Auxiliary lambda/z: {manifest["physical"]["lambda_ratio"]:g}. Physical activity: {manifest["physical"]["activity"]:g} Å⁻³. No pooling across models or auxiliary intensities.','',
        '| Std scale | Draws | log Q native | log Q other | F native−other / kBT | Native fraction | Shoulder fraction | Distant fraction | Total ESS | Total top 1% share | CPU s |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def f(value):return 'unresolved' if value is None else f'{value:.4g}'
    for key,g in groups.items():
        e=g['estimates'];r=g['fraction_of_observed_total']
        lines.append('| '+' | '.join([key,str(g['unconditional_draws']),f(e['native']['logQ']),f(e['other']['logQ']),f(g['F_native_minus_other_kBT']),f(r['native']),f(r['shoulder']),f(r['distant']),f(e['total']['ess']),f(e['total']['top_one_percent_fraction']),f(g['sampler_cpu_seconds'])])+' |')
    lines += ['', 'All ten exhaustive contact/registration region estimates, per-population values, paired-cloud variance decomposition, and selected proposal-component contributions are in [analysis.json](analysis.json).', '',
        'The q partitions are q≤0.8, 0.8<q≤1, 1<q<2, 2≤q<5, q≥5, each split by exact exclusion contact. The old SMC native partition is q≤1 and its other partition is all q>1. Zero observations provide no upper bound on missing physical mass.', '',
        '![Observed normalizers and cloud/pose variance](normalizer-pilot.png)']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    plot(result, out)
    print('\n'.join(lines))


def plot(result, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout='constrained')
    styles = [('native','#087f8c','o'), ('shoulder','#e9a038','s'), ('distant','#4c5a89','^')]
    for name,color,marker in styles:
        xy = [(g['covariance_std_scale'],g['estimates'][name]['logQ'])
              for g in result['groups'].values() if g['estimates'][name]['logQ'] is not None]
        if xy:
            x,y=zip(*xy);axes[0].plot(x,y,marker=marker,color=color,label=name)
    axes[0].set(xscale='log',xlabel='Gaussian standard-deviation multiplier',ylabel='Observed log Q',
                title='Proposal-width sensitivity')
    axes[0].set_xticks([g['covariance_std_scale'] for g in result['groups'].values()],
                      labels=[str(g['covariance_std_scale']) for g in result['groups'].values()])
    axes[0].legend()
    groups=list(result['groups'].values())
    x=np.arange(len(groups))
    cloud=[g['estimates']['total']['paired_noise']['paired_cloud_variance_fraction']
           if g['estimates']['total']['paired_noise'] else np.nan for g in groups]
    axes[1].bar(x,cloud,color='#87b8af',label='Paired-cloud noise')
    axes[1].bar(x,1-np.array(cloud),bottom=cloud,color='#ccd5e1',label='Residual pose variation')
    for i,g in enumerate(groups):
        axes[1].text(i,.5,f"ESS {g['estimates']['total']['ess']:.1f}",ha='center',fontsize=9)
    axes[1].set(xticks=x,xticklabels=[str(g['covariance_std_scale']) for g in groups],
        xlabel='Gaussian standard-deviation multiplier',ylabel='Fraction of observed weight variance',
        title='Separating observed cloud and pose variance')
    axes[1].legend(fontsize=8)
    sizes = {(g['populations'],g['unconditional_draws']//g['populations']) for g in groups}
    size = f'{next(iter(sizes))[0]} × {next(iter(sizes))[1]} draws per width' if len(sizes) == 1 else 'population sizes listed in report'
    fig.suptitle('Exploratory independent importance pilot: '+size+'\nZero observations are unresolved; these normalizers are not converged',fontsize=10)
    for extension in ['png','svg','pdf']:
        fig.savefig(out/f'normalizer-pilot.{extension}',dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out', type=Path, help='Fresh separate assessment directory; leaves the original campaign untouched')
    args=parser.parse_args()
    analyze(args.root.resolve(), args.out)
