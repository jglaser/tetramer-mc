#!/usr/bin/env python3
"""Classify completed entry-shell integration and compare the saved uniform R4.

This reads completed, hash-bound outputs. It neither launches sampling nor
replays the physical/proposal raw audit. Native and contact labels are reporting
masks; all proposal attempts, including exterior and hard-invalid zeros, remain
in the original denominator.
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
import numpy as np
from scipy.special import logsumexp
from analyze_mobile_native_pocket import require, read, sha, write, inside, load_classifier, local_sources, label_keys
from analyze_mobile_competing_reference import paired_moments, check_estimate
from analyze_mobile_full_capture import chart_radii
from analyze_mobile_threshold_reference import (
    REGION_SHA, DEFINITION_SHA, PRIMARY, ExclusionContact, partition_masks,
    add_native_masks, summarize_regions, validate_classifier_target,
    validate_campaign as validate_uniform_campaign)
from run_entry_shell_reference_campaign import (
    validate as validate_frozen_campaign, verify_assessment, verify_output)

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = 16384
SEEDS = [130101010 + 1009*j for j in range(4)]
GUIDE_SHA = '80df09a9a802dd4a7f2bd2a04de7f7745055067f09afc7b08cdf2eee2a6826de'
BINARY_SHA = '382518de841c5203bffe43d5b462b0d842f9bc72fbe9d76b479f3fb7b1551add'
BUNDLE_SHA = 'ccbf2e3286c63cb494e1a0066382d1a246248aeb2a1e0cd0db73ba72af8f906b'
UNIFORM_SHA = '276f68fa8b0f25e83b30d3f63a90923bed07bafe116082edd5b4c989d833372a'
SCOPE = ('Fresh entry-shell importance integration on the unchanged base146-at-B R4 physical region. '
         'Complete J/q weights and every attempted draw are retained. Native entry has precedence in the '
         'disjoint native / contact-without-entry / unbound-without-entry partition. No-entry is a threshold '
         'and catalogue label, not a distinct competing basin. The older equal-N uniform draws informed '
         'guide construction and remain a separate estimate; they are never pooled with these draws. '
         'Observed errors, ESS and cost-variance ratios do not establish tail coverage, equilibrium, '
         'whole-basin free energies, or a demonstrated speedup.')


def validate_allocation(jobs, terminal):
    ids = [f'r{j:02d}' for j in range(4)]
    require(len(jobs) == len(terminal) == 4 and sorted(j['id'] for j in jobs) == ids
            and sorted(j['id'] for j in terminal) == ids, 'Missing/repeated population')
    for job in jobs:
        index = int(job['id'][1:]); term = next(t for t in terminal if t['id'] == job['id'])
        require(job['seed'] == SEEDS[index] and job['samples'] == SAMPLES, 'Fresh seed or fixed N changed')
        require(term['status'] == 'complete' and term['returncode'] == 0
                and all(term[k] == job[k] for k in ('id', 'seed', 'samples')), 'Population not complete')


def read_weights(rows, samples, region):
    """Consume audited J/q values without conditioning on guide or target support."""
    require(len(rows) == samples and [r['draw'] for r in rows] == list(range(samples)),
            'Unconditional draws missing/repeated')
    z, h, pairs = [], [], []
    for row in rows:
        require(row['pose'] is not None and all(type(row[k]) is bool for k in
                ('hard_valid', 'region_valid', 'capture_valid', 'shell_valid')), 'Missing pose/support flag')
        radius = row['latent_radius']
        require(math.isfinite(radius) and radius >= 0. and
                row['shell_valid'] == (radius <= region['mahalanobis_radius']), 'Incorrect saved R4 support')
        require(math.isfinite(row['q']) and row['q'] >= 0. and row['region_valid'] is True,
                'Unexpected q filter in physical reference')
        require(all(math.isfinite(row[k]) for k in ('log_physical_jacobian', 'log_proposal_density')),
                'Missing proposal density/Jacobian')
        require(row['proposal_branch'] in ('entry-shell', 'uniform-shell'), 'Wrong proposal branch')
        if row['proposal_branch'] == 'uniform-shell':
            require(row['shell_valid'] and row['proposal_component'] is None, 'Invalid uniform branch')
        else:
            require(type(row['proposal_component']) is int and 0 <= row['proposal_component'] < 24,
                    'Invalid entry-shell component')
        require(not row['hard_valid'] or row['capture_valid'], 'Hard-valid pose outside capture')
        require(not row['shell_valid'] or row['capture_valid'], 'R4 capture enclosure contradicted')
        valid = row['hard_valid'] and row['shell_valid'] and row['region_valid']
        if not valid:
            require(row['log_hard_weight'] is None and row['log_importance_weight'] is None
                    and not row['clouds'], 'Invalid or exterior draw must retain zero')
            z.append(-np.inf); h.append(-np.inf); pairs.append([-np.inf, -np.inf]); continue
        require(len(row['clouds']) == 2 and math.isfinite(row['log_hard_weight'])
                and math.isfinite(row['log_importance_weight']), 'Invalid two-cloud weight')
        require(abs(row['log_hard_weight'] - (row['log_physical_jacobian'] - row['log_proposal_density'])) < 2e-10,
                'Saved hard weight is not J/q')
        pair = [row['log_hard_weight'] + c['log_weight'] for c in row['clouds']]
        require(all(math.isfinite(x) for x in pair) and
                abs(float(logsumexp(pair) - math.log(2)) - row['log_importance_weight']) < 2e-10,
                'Saved cloud mean differs')
        z.append(row['log_importance_weight']); h.append(row['log_hard_weight']); pairs.append(pair)
    return dict(z=np.asarray(z), h=np.asarray(h), pairs=np.asarray(pairs))


def validate_campaign(root):
    # These controller helpers only validate frozen files, saved output metadata
    # and source membership. No run(), subprocess, or raw auditor is invoked.
    protocol = validate_frozen_campaign(root)
    manifest, status = read(root/'manifest.json'), read(root/'status.json')
    require(status['schema'] == 'entry-shell-reference-status-v1' and status['complete'] is True
            and status['phase'] == 'complete' and not status.get('running', False), 'Campaign incomplete')
    require(status['protocol_sha256'] == sha(root/'protocol.json'), 'Terminal protocol binding changed')
    require(protocol['physical_executable_sha256'] == BINARY_SHA and protocol['source_bundle_sha256'] == BUNDLE_SHA,
            'Reviewed executable/source differs')
    require(manifest['region_sha256'] == REGION_SHA and manifest['importance_guide_sha256'] == GUIDE_SHA,
            'Frozen physical region or proposal differs')
    validate_allocation(manifest['jobs'], status['jobs'])
    for job in status['jobs']:
        require(verify_output(root, manifest, job) == job['output'], 'Terminal output binding changed')
    require(status['audit']['returncode'] == 0 and
            sha(root/'assessment/analysis.json') == status['audit']['analysis_sha256'], 'Raw audit failed/changed')
    assessment = read(root/'assessment/analysis.json')
    verify_assessment(root, manifest, assessment, status['jobs'])
    config, region = read(root/'provenance/config.json'), read(root/'provenance/region.json')
    definition = inside(root, protocol['native_definition'])
    require(sha(definition) == DEFINITION_SHA == protocol['native_definition_sha256'], 'Native definition differs')
    validate_classifier_target(config, manifest['shape_sha256'], read(definition), definition)
    return protocol, status, manifest, config, region, definition, assessment


def load_uniform_comparison(path, config, shape_sha):
    """Reuse the completed comparison; verify bytes without replaying its audit."""
    path = Path(path).resolve()
    require(sha(path/'analysis.json') == UNIFORM_SHA, 'Completed uniform comparison identity differs')
    for name, digest in read(path/'freeze.json').items():
        require(sha(inside(path, name)) == digest, 'Saved uniform comparison changed: '+name)
    data = read(path/'analysis.json'); root = Path(data['campaign']).resolve()
    require(data['complete'] and data['schema'] == 'mobile-threshold-reference-comparison-v1'
            and data['region_sha256'] == REGION_SHA and data['shape_sha256'] == shape_sha
            and data['native_definition']['definition_sha256'] == DEFINITION_SHA, 'Uniform target/classifier differs')
    _, status, manifest, old_config, _, _, _ = validate_uniform_campaign(root)
    normalized = copy.deepcopy(config); normalized['shape'] = old_config['shape']
    require(normalized == old_config, 'Uniform/entry-shell physical targets differ')
    for field, filename in [('protocol_sha256', 'protocol.json'), ('status_sha256', 'status.json'),
                            ('manifest_sha256', 'manifest.json'), ('assessment_sha256', 'assessment/analysis.json')]:
        require(data[field] == sha(root/filename), 'Uniform campaign binding changed: '+field)
    for name, digest in data['reference']['source_sha256'].items():
        require(sha(Path(name)) == digest, 'Saved uniform population changed: '+name)
    require({j['seed'] for j in manifest['jobs']}.isdisjoint(SEEDS), 'Uniform/reference seeds reused')
    require(all(e['row_uncertainty']['draws'] == 4*SAMPLES for e in data['reference']['estimates'].values()),
            'Uniform comparison discarded unconditional draws')
    alternative_path = path/'provenance/alternative-r32.json'
    require(sha(alternative_path) == data['exploratory_binding']['alternative_region_sha256'], 'Alternative diagnostic region changed')
    return data, read(alternative_path), dict(comparison=str(path), analysis_sha256=sha(path/'analysis.json'),
        campaign=str(root), status_sha256=sha(root/'status.json'), freeze_sha256=sha(path/'freeze.json'),
        alternative_region_sha256=sha(alternative_path), source_sha256=data['reference']['source_sha256'],
        interpretation='Completed uniform estimate reused without sampling or a repeated raw audit. Its discovery rows informed the guide; estimates are kept separate.')


def load_populations(root, manifest, status, assessment, config, region, classifier, alternative, out):
    contact = ExclusionContact(read(root/'provenance/shape.json'), config['fixed_poses'], config['depletant_radius'])
    populations, bindings, anomalies, dispositions = [], {}, [], []; cpu = 0.; negative = 0
    labels_path = out/'entry-shell-labels.jsonl'
    with labels_path.open('x') as stream:
        for job in manifest['jobs']:
            print('Classifying completed entry-shell poses: '+job['id'], flush=True)
            directory = Path(job['directory']).resolve()
            require(directory == root/'runs'/job['id'], 'Population path differs')
            term = next(t for t in status['jobs'] if t['id'] == job['id'])
            for name, key in [('samples.jsonl', 'samples_sha256'), ('manifest.json', 'manifest_sha256'), ('summary.json', 'summary_sha256')]:
                digest = sha(directory/name)
                require(digest == term['output'][key], 'Terminal population changed: '+name)
                bindings[str(directory/name)] = digest
            summary = read(directory/'summary.json')
            audit = next(p for p in assessment['populations'] if p['id'] == job['id'])
            rows = [json.loads(line) for line in (directory/'samples.jsonl').open()]
            arrays = read_weights(rows, SAMPLES, region); valid = np.isfinite(arrays['z']); indices = np.flatnonzero(valid)
            check_estimate(paired_moments(arrays['z'], arrays['h']), audit['estimate'], audit['hard_region'])
            require(int(valid.sum()) == summary['estimates']['region']['nonzero'] == summary['estimates']['hard_region']['nonzero'],
                    'Saved contributing counts disagree')
            radii = np.full(SAMPLES, np.inf)
            if len(indices): radii[indices], _ = chart_radii([rows[i]['pose'] for i in indices], alternative)
            native = np.zeros(SAMPLES, bool); anchors = np.zeros(SAMPLES, int)
            triangles = np.zeros(SAMPLES, bool); contacts = np.zeros(SAMPLES, bool); keys = []
            for i, row in enumerate(rows):
                label = classifier.classify(row['pose']) if valid[i] else None
                measured = contact.classify(row['pose']) if valid[i] else None
                keys.append(label_keys(label) if label is not None else set())
                if label is not None:
                    native[i] = label['native_any']; anchors[i] = label['native_anchor_count']
                    triangles[i] = label['registry_consistent_triangle']; contacts[i] = measured['exclusion_contact']
                    negative += int(measured['near_zero_negative_gap'])
                    if native[i] and not contacts[i]:
                        anomalies.append(dict(population=job['id'], draw=i, pose=row['pose'], classification=label, contact=measured))
                stream.write(json.dumps(dict(population=job['id'], seed=job['seed'], draw=i, applicable=bool(valid[i]),
                    classification=label, contact=measured, alternative_native_radius=float(radii[i]) if valid[i] else None),
                    separators=(',', ':'), allow_nan=False)+'\n')
            masks = partition_masks(valid, contacts, native, anchors, triangles, [r['q'] for r in rows], radii)
            branch_counts = {}
            for branch in ('uniform-shell', 'entry-shell'):
                selected = np.asarray([r['proposal_branch'] == branch for r in rows])
                branch_counts[branch] = dict(attempted=int(selected.sum()),
                    **{flag: int(sum(selected[i] and row[flag] for i, row in enumerate(rows)))
                       for flag in ('shell_valid', 'capture_valid', 'hard_valid')},
                    contributing=int((selected & valid).sum()),
                    **{name: int((selected & masks[name]).sum()) for name in PRIMARY})
                for name in ('total',)+PRIMARY:
                    masks['proposal_branch:'+branch+':'+name] = masks[name] & selected
            dispositions.append(dict(id=job['id'], seed=job['seed'], branches=branch_counts))
            populations.append(dict(id=job['id'], seed=job['seed'], masks=masks, label_keys=keys, **arrays))
            require(math.isfinite(summary['sampler_cpu_seconds']) and summary['sampler_cpu_seconds'] >= 0., 'Invalid physical CPU')
            cpu += summary['sampler_cpu_seconds']
    add_native_masks(populations)
    estimates, covariance, ratios = summarize_regions(populations)
    check_estimate(estimates['total']['row_uncertainty'], assessment['estimate'], assessment['hard_region'])
    return dict(estimates=estimates, primary_covariance=covariance, primary_ratios=ratios,
        sampler_cpu_seconds=cpu, source_sha256=bindings,
        native_labels=dict(path=labels_path.name, sha256=sha(labels_path)),
        native_entry_unbound_anomalies=anomalies, near_zero_negative_core_gap_count=negative,
        proposal_audit=assessment['importance_sampling'], proposal_dispositions=dispositions,
        branch_weight_contributions=branch_contributions(estimates),
        contact_definition='Exact minimum atom-surface gap to either fixed neighbor < 2*rd = 3 Å; independent of q and native labels.',
        hard_flag_validation='Completed frozen raw audit bound by hashes; contributing rows additionally pass independent atom-union gaps >= -1e-8 Å. Negative roundoff gaps are counted.')


def branch_contributions(estimates):
    """Sum component-class contributions with the full proposal q and total N."""
    result = {}
    for name in ('total',)+PRIMARY:
        result[name] = {}
        total = estimates[name]['row_uncertainty']
        for branch in ('uniform-shell', 'entry-shell'):
            key = 'proposal_branch:'+branch+':'+name
            row = estimates[key]['row_uncertainty']
            result[name][branch] = dict(draws=row['draws'], nonzero=row['nonzero'],
                **{kind: dict(log_contribution=row['log_'+kind],
                    observed_weight_fraction=None if total['log_'+kind] is None else (
                        0. if row['log_'+kind] is None else math.exp(row['log_'+kind]-total['log_'+kind])))
                   for kind in ('Q0', 'Qz')})
    return result


def compare_estimates(fresh, uniform):
    """Descriptive estimates; no pooling and no efficiency/convergence decision."""
    result = {}
    for name in ('total',)+PRIMARY:
        result[name] = {}
        for kind in ('Q0', 'Qz'):
            levels = {}
            for level in ('row_uncertainty', 'population_uncertainty'):
                left, right = fresh['estimates'][name][level], uniform['estimates'][name][level]
                if left['log_'+kind] is None or right['log_'+kind] is None:
                    levels[level] = dict(observed=False, reason='Unobserved is neither zero mass nor an upper bound.'); continue
                left_var, right_var = left[kind+'_relative_SE']**2, right[kind+'_relative_SE']**2
                fresh_cost = left_var*fresh['sampler_cpu_seconds']
                uniform_cost = right_var*uniform['sampler_cpu_seconds']
                levels[level] = dict(observed=True, log_fresh_minus_uniform=left['log_'+kind]-right['log_'+kind],
                    quadrature_observed_delta_SE=math.sqrt(left_var+right_var),
                    fresh_relative_variance_times_cpu=fresh_cost, uniform_relative_variance_times_cpu=uniform_cost,
                    uniform_over_fresh_cost_variance=uniform_cost/fresh_cost if fresh_cost > 0. else None)
            result[name][kind] = levels
    return result


def report(result):
    lines = ['# Entry-shell versus uniform integration on the same R4', '', SCOPE, '',
        '| Class | Proposal | log Qz | row RSE | population RSE | log Q0 | Qz ESS | largest weight |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for name in ('total',)+PRIMARY:
        for title, source in [('entry shells', result['reference']), ('previous uniform', result['uniform_reference'])]:
            row = source['estimates'][name]['row_uncertainty']; pop = source['estimates'][name]['population_uncertainty']
            if row['log_Qz'] is None:
                lines.append(f'| {name} | {title} | unobserved | — | — | — | — | — |'); continue
            lines.append(f"| {name} | {title} | {row['log_Qz']:.6f} | {row['Qz_relative_SE']:.2%} | {pop['Qz_relative_SE']:.2%} | {row['log_Q0']:.6f} | {row['Qz_ESS']:.1f} | {row['largest_Qz_fraction']:.2%} |")
    lines += ['', f"Each estimate retains four populations × {SAMPLES:,} unconditional draws. Physical CPU: entry shells {result['reference']['sampler_cpu_seconds']:.1f} s; uniform {result['uniform_reference']['sampler_cpu_seconds']:.1f} s.", '',
        'Per-population weights, paired-cloud variance, hard/depletion covariance, diagnostic q/R32 masks, and descriptive cost-variance ratios are retained in analysis.json. Ratios of observed error times CPU are not validated speedups; missed tails can make either estimate appear efficient.', '',
        f"Native-entry/unbound anomalies: {len(result['reference']['native_entry_unbound_anomalies'])}. Roundoff-negative contributing core gaps: {result['reference']['near_zero_negative_core_gap_count']}.", '',
        'No-entry boundary shells guide proposals only. The final classifier retains all original body, angle, patch, and residue criteria. Additional independent populations and wider coverage remain necessary before a physical mass verdict.', '',
        '| Proposal branch | Attempts | Inside R4 | Hard-valid anywhere | Contributing in R4 | Native | Contact, no entry | Unbound, no entry |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for branch in ('uniform-shell', 'entry-shell'):
        counts = [p['branches'][branch] for p in result['reference']['proposal_dispositions']]
        values = [sum(p[key] for p in counts) for key in ('attempted', 'shell_valid', 'hard_valid', 'contributing')+PRIMARY]
        lines.append('| '+branch+' | '+' | '.join(str(v) for v in values)+' |')
    lines += ['', 'Branch weight fractions in analysis.json use the full mixture density and the same unconditional denominator. They measure where this realized estimator obtained its weight; they do not renormalize separately by successful draws or by branch attempts.']
    return '\n'.join(lines)+'\n'


def plot(result, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ('total',)+PRIMARY
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for axis, kind in zip(axes, ('Qz', 'Q0')):
        for offset, (title, source) in enumerate([('entry shells', result['reference']), ('previous uniform', result['uniform_reference'])]):
            xs, ys, errors = [], [], []
            for index, name in enumerate(names):
                row = source['estimates'][name]['row_uncertainty']
                if row['log_'+kind] is not None:
                    xs.append(index+.18*(offset-.5)); ys.append(row['log_'+kind]); errors.append(row[kind+'_relative_SE'])
            axis.errorbar(xs, ys, yerr=errors, fmt='o', capsize=3, label=title, markersize=4)
        axis.set_xticks(range(len(names)), ['whole R4', 'native entry', 'contact, no entry', 'unbound, no entry'], rotation=20, ha='right')
        axis.set_ylabel('log '+kind+' (observed row delta SE)'); axis.grid(axis='y', alpha=.2)
    axes[0].legend(fontsize=8); fig.suptitle('Same finite physical target; separate unconditional estimates')
    fig.savefig(out/'entry-shell-reference.png', dpi=170); fig.savefig(out/'entry-shell-reference.svg'); plt.close(fig)


def analyze(campaign, out, uniform_comparison):
    root, out = Path(campaign).resolve(), Path(out).resolve()
    require(not out.exists(), 'Use fresh downstream output')
    protocol, status, manifest, config, region, definition, assessment = validate_campaign(root)
    classifier, binding = load_classifier(definition)
    uniform, alternative, uniform_binding = load_uniform_comparison(uniform_comparison, config, manifest['shape_sha256'])
    out.mkdir(parents=True)
    try:
        reference = load_populations(root, manifest, status, assessment, config, region, classifier, alternative, out)
        result = dict(schema='mobile-entry-shell-reference-comparison-v1', complete=True, campaign=str(root),
            protocol_sha256=sha(root/'protocol.json'), status_sha256=sha(root/'status.json'), freeze_sha256=sha(root/'freeze.json'),
            manifest_sha256=sha(root/'manifest.json'), assessment_sha256=sha(root/'assessment/analysis.json'),
            region_sha256=REGION_SHA, importance_guide_sha256=GUIDE_SHA, shape_sha256=manifest['shape_sha256'],
            native_definition=binding, primary_partition=list(PRIMARY), reference=reference,
            uniform_reference=uniform['reference'], uniform_binding=uniform_binding,
            comparison=compare_estimates(reference, uniform['reference']), scope=SCOPE)
        sources = out/'provenance'; sources.mkdir()
        for name, path in local_sources(__file__).items(): shutil.copy2(path, sources/name)
        for name, path in [('finite-region.json', root/'provenance/region.json'), ('importance-guide.json', root/'provenance/importance-guide.json'),
                           ('campaign-protocol.json', root/'protocol.json'), ('campaign-manifest.json', root/'manifest.json')]:
            shutil.copy2(path, sources/name)
        result['analyzer_source_sha256'] = {p.name: sha(p) for p in sources.iterdir()}
        write(out/'analysis.json', result); (out/'report.md').write_text(report(result)); plot(result, out)
        write(out/'freeze.json', {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
        print(json.dumps(dict(complete=True, out=str(out), analysis_sha256=sha(out/'analysis.json'))), flush=True)
        return result
    except Exception as error:
        write(out/'failure.json', dict(complete=False, error=str(error),
            scope='Saved-row downstream analysis only; no sampler or raw audit invoked. Partial outputs preserved.'))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--uniform-comparison', type=Path, default=ROOT/'runs/mobile-threshold-reference-comparison-20260921')
    args = parser.parse_args(); analyze(args.campaign, args.out, args.uniform_comparison)
