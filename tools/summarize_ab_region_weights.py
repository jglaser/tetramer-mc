#!/usr/bin/env python3
"""Compare already audited AB windows without pooling proposal controls.

This is an evidence ledger, not another estimator or convergence test.
Ratios formed from the reported estimates remain provisional.
"""
import argparse
import itertools
import math
from pathlib import Path
import shutil

from scipy.special import logsumexp

from prepare_cayley_rms_cover import read, write, sha, require

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = {
    'native': dict(minimum=0., maximum=1., lower_inclusive=True, upper_inclusive=True),
    'shoulder': dict(minimum=1., maximum=2., lower_inclusive=False, upper_inclusive=False),
    'intermediate': dict(minimum=2., maximum=5., lower_inclusive=True, upper_inclusive=False),
    'far': dict(minimum=5., maximum=37., lower_inclusive=True, upper_inclusive=False),
}
LIMITATIONS = {
    'native': 'Uniform 8–12 reference reproduces the wider guide, but precision of the 5–8 shell and mass beyond original-chart radius12 remain unresolved.',
    'shoulder': 'Independent inner-band cover is dominated by rare weights. Broader confirmation also reveals concentration; observed errors do not resolve normalizer convergence.',
    'intermediate': 'Unchanged-model repeat widths agree within one observed SE. The positive remainder is less precise; no unseen-tail certificate.',
    'far': 'Complete proposal support and independent repeats are diagnostic. Observed errors do not bound unknown narrow contacts.',
}


def summarize(far_path, out):
    out = out.resolve(); require(not out.exists(), 'Fresh status output required')
    paths = {
        'native': ROOT/'runs/native-ab-refined-comparison-20260920/comparison.json',
        'shoulder': ROOT/'runs/ab-shoulder-confirmation-with-cover-comparison-20260920/comparison.json',
        'intermediate': ROOT/'runs/ab-intermediate-expanded-atlas-repeat-audit-20260920/analysis.json',
        'far': far_path.resolve(),
    }
    data = {k: read(p) for k,p in paths.items()}
    require(all(data[k]['complete'] for k in ('native','intermediate','far')), 'Incomplete source audit')
    require(data['native']['same_physical_native_AB_target_verified'], 'Native AB identity not audited')
    for key in ('shoulder','intermediate','far'):
        window = data[key].get('q_window', data[key].get('original_q_window'))
        require(window == WINDOWS[key], 'Different original window: '+key)
    selected = {
        'native': [(k, data['native']['campaigns'][k]) for k in ('fresh_selected','fresh_wide')],
        'shoulder': [(k, data['shoulder']['campaigns'][k]) for k in ('mixture-confirmation','geometry-confirmation')],
        'intermediate': [(c['arm'], c) for c in data['intermediate']['campaigns']],
        'far': [(c['arm'], c) for c in data['far']['campaigns']],
    }
    sources = {str(p): sha(p) for p in paths.values()}; common = None; shape_hash = None
    rows = {}; all_seeds = set()
    for region, campaigns in selected.items():
        rows[region] = []
        require(len(campaigns) == 2, 'Need both proposal controls')
        for label, campaign in campaigns:
            root = Path(campaign['root']); manifest_path = root/'manifest.json'
            master = read(manifest_path); sources[str(manifest_path)] = sha(manifest_path)
            status_path = root/'runner-status.json'; status = read(status_path)
            require(status['complete'] and status['success'], 'Physical campaign unfinished')
            sources[str(status_path)] = sha(status_path)
            require(master.get('q_window', WINDOWS['native']) == WINDOWS[region], 'Actual window differs')
            actual_count = 0; seeds = set()
            for job in master['jobs']:
                manifest = Path(job['output'])/'manifest.json'; run = read(manifest)
                sources[str(manifest)] = sha(manifest)
                cfg_path = Path(job['command'][job['command'].index('--config')+1]); cfg = read(cfg_path)
                sources[str(cfg_path)] = sha(cfg_path)
                require(run['config_sha256'] == sources[str(cfg_path)], 'Runtime config changed')
                require(run.get('q_window', WINDOWS['native']) == WINDOWS[region], 'Runtime q window differs')
                require(run['activity'] == .035 and run['depletant_radius'] == 1.5 and run['lambda_ratio'] == 64. and run['cloud_replicates'] == 2, 'Runtime bath differs')
                require(run['metric'] == {k:cfg['metadata'][k] for k in ('member_error_scale','angle_error_scale_deg','native_poses','rigid_members')}, 'Runtime registration metric differs')
                shape = Path(cfg['shape']); shape = shape if shape.is_absolute() else cfg_path.parent/shape
                digest = sha(shape); sources[str(shape)] = digest
                physical = {k:v for k,v in cfg.items() if k != 'shape'}
                if common is None: common, shape_hash = physical, digest
                require(physical == common and digest == shape_hash == run['shape_sha256'], 'Different physical configuration or shape')
                require(job['seed'] == run['seed'] and run['seed'] not in seeds, 'Repeated stream in campaign')
                seeds.add(run['seed']); actual_count += run['samples']
            require(not all_seeds.intersection(seeds), 'Selected production streams overlap')
            all_seeds.update(seeds)
            if region == 'native':
                value = campaign['physical']; n, logq, rse = value['samples'], value['log_normalizer'], value['relative_SE']
            elif region == 'shoulder':
                value = campaign['physical']; n, logq, rse = value['draws'], value['logQ'], value['observed_RSE']
            else:
                value = campaign['physical']['full']; n, logq, rse = value['draws'], value['logQ'], value['row_RSE']
            require(actual_count == n == master['total_unconditional_draws'], 'Changed original denominator')
            rows[region].append(dict(proposal=label, root=str(root), draws=n, logQ=logq, observed_row_RSE=rse,
                original_statistics=value, limitation=LIMITATIONS[region]))
    require(len(common['fixed_poses']) == 2 and common['capture_radius'] == 18. and common['metadata']['member_error_scale'] == 2. and common['metadata']['angle_error_scale_deg'] == 15., 'Different AB problem')
    # Enumerate every combination of reported controls; these are not independent
    # replications and their numerical envelope is not a confidence interval.
    combinations = []
    for chosen in itertools.product(*(rows[k] for k in WINDOWS)):
        native = chosen[0]['logQ']; other = float(logsumexp([v['logQ'] for v in chosen[1:]]))
        combinations.append(dict(proposals={k:v['proposal'] for k,v in zip(WINDOWS,chosen)},
            observed_logQ_other=other, observed_logQ_native=native,
            observed_native_minus_other_log_weight=native-other,
            observed_other_to_native_ratio=math.exp(other-native)))
    archive = out/'provenance'; archive.mkdir(parents=True)
    for key,path in paths.items(): shutil.copy2(path, archive/f'{key}-analysis.json')
    shutil.copy2(Path(__file__), archive/'summarize_ab_region_weights.py')
    for path,digest in sources.items(): require(sha(Path(path)) == digest, 'Input changed during status audit')
    result = dict(complete=True, new_physical_draws=0, full_convergence_established=False,
        common_configuration_excluding_shape_path=common, shape_sha256=shape_hash, original_windows=WINDOWS,
        region_proposal_controls=rows, all_control_combinations=combinations, production_stream_count=len(all_seeds),
        combination_scope='Derived ratios of existing point estimates, not additional simulations, independent replications, certified probabilities or confidence bounds. No proposals or populations pooled.',
        physical_scope='One mobile rigid tetramer conditioned on two fixed AB neighbors and capture. Native intratetramer structure and AB neighborhood are supplied; this is not assembly or the fluid chemical potential.',
        source_sha256=sources, archived_sha256={p.name:sha(p) for p in archive.iterdir()})
    write(out/'analysis.json', result)
    lines = ['# Existing AB regional weights: provisional evidence ledger', '',
        '| Original region | Proposal control | N | log Q | Observed row RSE |',
        '| --- | --- | ---: | ---: | ---: |']
    for region, variants in rows.items():
        for v in variants: lines.append(f"| {region} | {v['proposal']} | {v['draws']:,} | {v['logQ']:.6f} | {100*v['observed_row_RSE']:.2f}% |")
    lines += ['', 'Configuration bytes agree apart from shape paths; all shape contents, bath, registration metric and actual runtime windows agree.', '']
    lines += [f'- {key}: {LIMITATIONS[key]}' for key in WINDOWS]
    lines += ['', result['combination_scope'], '', result['physical_scope'], '']
    (out/'report.md').write_text('\n'.join(lines))
    print(dict(complete=True, output=str(out/'analysis.json'), unresolved=True))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--far-analysis',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();summarize(args.far_analysis,args.out)
