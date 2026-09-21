#!/usr/bin/env python3
"""Carry forward audited AB estimates and replace only the full shoulder controls."""
import argparse
import copy
import itertools
import math
from pathlib import Path

from scipy.special import logsumexp

from prepare_cayley_rms_cover import read, write, sha, require
from summarize_ab_region_weights import WINDOWS
from run_shoulder_mis_campaign import local_dependencies


def verified_result(path, source_key):
    result = read(path)
    require(result['complete'], 'Completed source analysis required')
    for name, digest in result['archived_sha256'].items():
        require(sha(path.parent/'provenance'/name) == digest, 'Archived source changed')
    for name, digest in result[source_key].items():
        require(sha(name) == digest, 'Audited input changed')
    return result


def update(previous, shoulder_path, native_path, out):
    require(not out.exists(), 'Use a fresh ledger output')
    old = verified_result(previous, 'source_sha256')
    shoulder = verified_result(shoulder_path, 'source_sha256')
    native = verified_result(native_path, 'input_sha256')
    require(old['original_windows'] == WINDOWS and shoulder['original_q_window'] == WINDOWS['shoulder']
            and native['original_q_window'] == WINDOWS['native'], 'Original windows differ')
    physical = old['common_configuration_excluding_shape_path']; shape = old['shape_sha256']
    sources = {str(p): sha(p) for p in [previous, shoulder_path, native_path]}
    rows = copy.deepcopy(old['region_proposal_controls'])
    seeds = set()
    for region, controls in rows.items():
        if region == 'shoulder': continue
        for control in controls:
            manifest = read(Path(control['root'])/'manifest.json')
            for job in manifest['jobs']:
                require(job['seed'] not in seeds, 'Retained controls reuse a stream')
                seeds.add(job['seed'])
    rows['shoulder'] = []
    for campaign in shoulder['campaigns']:
        root = Path(campaign['root']); cfg = read(root/'provenance/config.json')
        require({k:v for k,v in cfg.items() if k != 'shape'} == physical,
                'New shoulder physical configuration differs')
        require(sha(root/'provenance/shape.json') == shape, 'New shoulder shape differs')
        value = campaign['physical']['full']
        master = read(root/'manifest.json'); status = read(root/'runner-status.json')
        require(status['complete'] and status['success'], 'New shoulder run not terminal')
        require(value['draws'] == master['total_unconditional_draws'] == campaign['original_unconditional_draws'], 'Changed full denominator')
        for population in campaign['populations']:
            require(population['seed'] not in seeds, 'New shoulder reuses a retained stream')
            seeds.add(population['seed'])
        rows['shoulder'].append(dict(proposal=campaign['arm'], root=str(root), draws=value['draws'],
            logQ=value['logQ'], observed_row_RSE=value['row_RSE'], original_statistics=value,
            limitation='Fresh complete-support weights with calibrated finite contacts. Width, prefix and complementary-region concentration still require inspection; no unseen-mass bound.'))
    require(len(rows['shoulder']) == 2, 'Keep separate width controls')
    native_root = Path(native['campaign']); cfg = read(native_root/'provenance/config.json')
    require({k:v for k,v in cfg.items() if k != 'shape'} == physical and
            sha(native_root/'provenance/shape.json') == shape, 'Native tail target differs')
    for population in native['populations']:
        require(population['seed'] not in seeds, 'Native tail reuses an existing control stream')
        seeds.add(population['seed'])
    combinations = []
    for chosen in itertools.product(*(rows[k] for k in WINDOWS)):
        other = float(logsumexp([v['logQ'] for v in chosen[1:]])); native_logq = chosen[0]['logQ']
        combinations.append(dict(proposals={k:v['proposal'] for k,v in zip(WINDOWS, chosen)},
            observed_logQ_other=other, observed_logQ_native=native_logq,
            observed_native_minus_other_log_weight=native_logq-other,
            observed_other_to_native_ratio=math.exp(other-native_logq)))
    tail = native['physical']['old_r_gt_12']
    tail_ratios = [dict(native_proposal=c['proposal'], observed_tail_to_full_ratio=
        None if tail['logQ'] is None else math.exp(tail['logQ']-c['logQ'])) for c in rows['native']]
    result = dict(complete=True, new_physical_draws=0, full_convergence_established=False,
        common_configuration_excluding_shape_path=physical, shape_sha256=shape, original_windows=WINDOWS,
        region_proposal_controls=rows, all_control_combinations=combinations,
        native_tail_diagnostic=dict(original_statistics=tail, comparisons_with_existing_full_native=tail_ratios,
            rule='Independent direct tail estimate only; never added to an existing full-native estimate, and not substituted for it.'),
        retained_source_ledger=dict(path=str(previous), sha256=sha(previous)),
        combination_scope='Ratios of existing independent importance point estimates; proposal controls remain separate. Their envelope is not a confidence interval. Native-tail uncertainty and complementary-region diagnostics remain qualifications.',
        physical_scope=old['physical_scope'], source_sha256=sources)
    archive = out/'provenance'; archive.mkdir(parents=True)
    files = {'previous-ledger.json': previous, 'shoulder-analysis.json': shoulder_path,
             'native-tail-analysis.json': native_path, **local_dependencies([Path(__file__)])}
    for name, path in files.items(): (archive/name).write_bytes(path.read_bytes())
    result['archived_sha256'] = {name: sha(archive/name) for name in files}
    write(out/'analysis.json', result)
    print(dict(complete=True, output=str(out/'analysis.json'), observed_gap_range=[
        min(c['observed_native_minus_other_log_weight'] for c in combinations),
        max(c['observed_native_minus_other_log_weight'] for c in combinations)], native_tail_ratios=tail_ratios))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--shoulder', type=Path, required=True)
    parser.add_argument('--native-tail', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    update(*(p.resolve() for p in [args.previous, args.shoulder, args.native_tail, args.out]))
