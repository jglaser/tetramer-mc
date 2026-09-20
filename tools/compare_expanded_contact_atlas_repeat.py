#!/usr/bin/env python3
"""Compare independent original/repeat campaigns without pooling their samples.

The repeat uses exactly the original frozen model bytes. Fixed-region and
width comparisons use independent-stream variance sums, whereas within-run
prefix diagnostics remain the correlated quantities saved by the source audit.
No empirical agreement, error estimate, or zero observation bounds unseen mass.
"""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

from analyze_contact_atlas import PHYSICAL_KEYS
from prepare_cayley_rms_cover import read, require, sha, write
from run_shoulder_mis_campaign import local_dependencies


def independent_difference(original, repeat):
    """Variance of two independent estimated means; no logarithmic unbiasedness."""
    variances = [row['log_variance_of_mean'] for row in (original, repeat)
                 if row['log_variance_of_mean'] is not None]
    if not variances:
        logvar = None
    else:
        offset = max(variances); logvar = offset+math.log(sum(math.exp(v-offset) for v in variances))
    a, b = original['logQ'], repeat['logQ']
    if a is None or b is None:
        logratio = ratio = difference = relative_se = logratio_se = standardized = None
    else:
        logratio = b-a; ratio = math.exp(logratio); difference = math.expm1(logratio)
        relative_se = math.exp(.5*logvar-a) if logvar is not None else 0.
        logratio_se = math.hypot(original['row_RSE'], repeat['row_RSE'])
        standardized = difference/relative_se if relative_se else None
    nvar_ratio = None
    if original['log_variance_of_mean'] is not None and repeat['log_variance_of_mean'] is not None:
        nvar_ratio = math.exp(repeat['log_variance_of_mean']-original['log_variance_of_mean']+
                             math.log(repeat['draws']/original['draws']))
    return dict(original=original, repeat=repeat, delta_logQ=logratio, repeat_over_original=ratio,
        difference_over_original_mean=difference, observed_SE_difference_over_original_mean=relative_se,
        observed_difference_in_SE_units=standardized, observed_SE_delta_logQ=logratio_se,
        estimated_log_variance_of_difference=logvar,
        unconditional_draw_count_ratio=repeat['draws']/original['draws'],
        N_scaled_observed_variance_ratio=nvar_ratio,
        qualification='Independent-stream variance sum. Log/ratio and standardized difference are descriptive, not unbiased estimators or tail-safe significance tests. Unobserved masks remain unresolved.')


def region_rows(campaign, kind):
    rows = dict(campaign[kind])
    rows.update({f'new_{key}': value for key, value in campaign['new_neighborhood'][kind].items()})
    rows.update({f'partition_{key}': value for key, value in campaign['disjoint_threeway'][kind].items()})
    return rows


def verified_audit(path):
    path = Path(path).resolve(); audit = read(path)
    require(audit['complete'], 'Need a completed independent audit')
    for name, digest in audit['archived_sha256'].items():
        require(sha(path.parent/'provenance'/name) == digest, 'Source audit archive changed')
    for name, digest in audit['input_sha256'].items():
        require(sha(Path(name)) == digest, 'Audited physical or proposal input changed')
    protocol_path = path.parent/'provenance/protocol.json'; protocol = read(protocol_path)
    freeze = read(path.parent/'provenance/freeze.json')
    require(freeze['protocol_sha256'] == sha(protocol_path), 'Archived frozen protocol differs')
    streams = set(); records = {}
    for campaign in audit['campaigns']:
        require(campaign['arm'] not in records, 'Repeated source arm'); records[campaign['arm']] = campaign
        root = Path(campaign['root']); original = campaign['original_full_window_audit']
        require(sha(root/'manifest.json') == original['manifest_sha256'], 'Source campaign manifest changed')
        for name, digest in campaign['input_sha256'].items():
            with Path(name).open('rb') as handle:
                require(hashlib.file_digest(handle, 'sha256').hexdigest() == digest, 'Audited source row/manifest bytes changed')
        model_hash = sha(root/'provenance/guide-model.json')
        require(model_hash == freeze['model_sha256'][campaign['arm']] == original['training_provenance']['model_sha256'],
                'Campaign frozen proposal model differs')
        for population in campaign['populations']:
            seed = population['seed']; require(seed not in streams, 'Repeated random stream inside audit'); streams.add(seed)
        require(original['original_integration_window'] == audit['original_q_window'] == protocol['q_window'],
                'Physical integration window differs')
        require(all(original['physical_signature'][key] == protocol['physical'][key] for key in PHYSICAL_KEYS),
                'Physical signature differs from audited protocol')
    return audit, protocol, freeze, records, streams


def compare(original_path, repeat_path, out):
    original_path, repeat_path, out = [Path(p).resolve() for p in (original_path, repeat_path, out)]
    require(not out.exists(), 'Use a fresh independent-comparison output directory')
    a, pa, fa, ca, sa = verified_audit(original_path)
    b, pb, fb, cb, sb = verified_audit(repeat_path)
    require(not sa.intersection(sb), 'Original and repeat share population streams')
    require(set(ca) == set(cb) and fa['model_sha256'] == fb['model_sha256'], 'Repeat changes model bytes or arm identities')
    for key in ('physical', 'q_window', 'analysis_charts', 'reference_audits', 'reference_totals',
                'executable_sha256', 'lambda_ratio', 'cloud_replicates'):
        require(pa[key] == pb[key], f'Repeat changed the physical domain or control: {key}')
    require(pb['repeat_source']['sha256'] == fa['protocol_sha256'] and
            pb['repeat_source']['model_sha256'] == fa['model_sha256'], 'Repeat lineage does not match original source')
    require(b['repeat_source_verification']['models_unchanged'] and b['repeat_source_verification']['disjoint_population_seeds'],
            'Repeat audit did not certify fixed models and independent streams')
    sources = {'original-audit.json': original_path, 'repeat-audit.json': repeat_path}
    inputs = {str(path): sha(path) for path in sources.values()}
    arms = []
    for arm in ca:
        original, repeat = ca[arm], cb[arm]
        comparison = {kind: {key: independent_difference(value, region_rows(repeat, kind)[key])
                             for key, value in region_rows(original, kind).items()}
                      for kind in ('physical', 'hard')}
        all_row = comparison['physical']['full']
        if all_row['N_scaled_observed_variance_ratio'] is not None:
            all_row['observed_variance_times_CPU_ratio'] = math.exp(
                repeat['physical']['full']['log_variance_of_mean']-original['physical']['full']['log_variance_of_mean']+
                math.log(repeat['CPU_seconds']/original['CPU_seconds']))
        arms.append(dict(arm=arm, **comparison,
            original_CPU_seconds=original['CPU_seconds'], repeat_CPU_seconds=repeat['CPU_seconds'],
            original_prefix_comparisons=original['prefix_comparisons'], repeat_prefix_comparisons=repeat['prefix_comparisons'],
            original_populations=original['populations'], repeat_populations=repeat['populations']))
    width_comparisons = {}
    if set(ca) == {'narrow', 'broad'}:
        for name, records in (('original', ca), ('repeat', cb)):
            width_comparisons[name] = {kind: {
                key: independent_difference(records['narrow']['disjoint_threeway'][kind][key],
                                             records['broad']['disjoint_threeway'][kind][key])
                for key in records['narrow']['disjoint_threeway'][kind]} for kind in ('physical', 'hard')}
    archive = out/'provenance'; archive.mkdir(parents=True)
    for name, path in {**local_dependencies([Path(__file__)]), **sources}.items():
        (archive/name).write_bytes(path.read_bytes())
    for path, digest in inputs.items(): require(sha(Path(path)) == digest, 'Source changed during comparison')
    result = dict(complete=True, arms=arms, width_comparisons=width_comparisons,
        input_sha256=inputs, archived_sha256={p.name: sha(p) for p in archive.iterdir()},
        original_prefix_counts=a['prefix_counts'], repeat_prefix_counts=b['prefix_counts'],
        original_total_draws=sum(c['physical']['full']['draws'] for c in ca.values()),
        repeat_total_draws=sum(c['physical']['full']['draws'] for c in cb.values()),
        same_model_sha256=fa['model_sha256'], unchanged_physical_signature=pa['physical'],
        original_shape_sha256={name: row['original_full_window_audit']['shape_sha256'] for name, row in ca.items()},
        repeat_shape_sha256={name: row['original_full_window_audit']['shape_sha256'] for name, row in cb.items()},
        independence='Different declared production seeds and identical frozen model bytes verified. Estimates stay separate. Prefixes within each source share rows and keep their original covariance-aware diagnostics.',
        uncertainty='Observed row variance and population dispersion only. Neither agreement nor N-scaled variance agreement certifies unseen-tail coverage; no fitted acceptance threshold or retrospective pooling.',
        scope='Complete fixed intermediate 2<=q<5 domain, with unchanged old/new geometry masks. Does not resolve other contact regions, neighbor formation costs, dynamical mixing, or crystallization.')
    require(result['original_shape_sha256'] == result['repeat_shape_sha256'], 'Physical shape changed between campaigns')
    write(out/'analysis.json', result)
    for arm in arms:
        row = arm['physical']['full']
        print(arm['arm'], 'original/repeat logQ', row['original']['logQ'], row['repeat']['logQ'],
              'difference / observed SE', row['observed_difference_in_SE_units'], flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--repeat', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); compare(args.original, args.repeat, args.out)
