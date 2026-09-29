#!/usr/bin/env python3
"""Trace coarse FFT repairs to archived full-scan contacts at matching stages.

Read-only provenance/score analysis; no new physical or geometric samples.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), digest(raw)


def pose_key(p):
    q = np.asarray(p['orientation'], float)
    if q[np.flatnonzero(q != 0)[0]] < 0:
        q = -q
    return tuple(np.round(np.r_[p['position'], q], 10))


def moments(clouds):
    if len(clouds) != 2 or clouds[0]['seed'] == clouds[1]['seed']:
        raise ValueError('expected two independently seeded validation clouds')
    return dict(volume=float(np.mean([r['volume'] for r in clouds])),
                standard_error=float(np.sqrt(sum(r['standard_error']**2 for r in clouds))/2),
                seeds=[r['seed'] for r in clouds], points_per_cloud=[r['draws'] for r in clouds])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recovery', type=Path, required=True)
    p.add_argument('--union-summary', type=Path, required=True)
    p.add_argument('--cpu-selected', type=Path, required=True)
    p.add_argument('--gpu-selected', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    plan, ph = read(args.recovery/'plan.json')
    recovery, rh = read(args.recovery/'report.json')
    union, uh = read(args.union_summary)
    if not recovery['complete'] or recovery['denominator'] != plan['all_slot_denominator']:
        raise ValueError('recovery run incomplete or allocation differs')
    provenance = dict(recovery_plan=dict(path=str(args.recovery/'plan.json'), sha256=ph),
                      recovery_report=dict(path=str(args.recovery/'report.json'), sha256=rh),
                      union_summary=dict(path=str(args.union_summary), sha256=uh), sources=[])
    for key, f in plan['frozen'].items():
        if digest(Path(f['path']).read_bytes()) != f['sha256']:
            raise ValueError('frozen recovery input changed: '+key)
    targets = {pose_key(r['reference_pose']):r for r in plan['slots']}
    if len(targets) != plan['all_slot_denominator']:
        raise ValueError('duplicate reference poses')
    selected = {}
    for origin, path in [('cpu', args.cpu_selected), ('gpu', args.gpu_selected)]:
        data, h = read(path)
        selected[origin] = {pose_key(r['pose']):dict(index=i, record=r) for i,r in enumerate(data)}
        provenance[origin+'_selected'] = dict(path=str(path), sha256=h)
        summary, h = read(path.parent/'summary.json')
        provenance[origin+'_selection_summary'] = dict(path=str(path.parent/'summary.json'), sha256=h)
        if summary['arguments']['rd'] != plan['rd'] or summary['shape_sha256'] != plan['frozen']['shape']['sha256']:
            raise ValueError('original selection shape/bath mismatch')
    matched = {}
    for index, directory in enumerate(union['sources']):
        origin = 'cpu' if index == 0 else 'gpu'
        for file in sorted(Path(directory).glob('populations/*/discovery.json')):
            data, h = read(file)
            manifest, mh = read(file.parent/'manifest.json')
            if not manifest['complete'] or manifest['outputs_sha256']['discovery.json'] != h:
                raise ValueError('source hash mismatch or incomplete: '+str(file))
            provenance['sources'].append(dict(path=str(file), sha256=h, manifest_sha256=mh))
            if data['shape_sha256'] != plan['frozen']['shape']['sha256'] or data['config']['rd'] != plan['rd']:
                raise ValueError('source shape/radius differs')
            for slot in data['slots']:
                key = pose_key(slot['optimized_pose'])
                if key not in targets:
                    continue
                global_slot = targets[key]['slot']
                if global_slot in matched:
                    raise ValueError('ambiguous lineage')
                selected_start = selected[origin].get(pose_key(slot['initial_pose']))
                if selected_start is None:
                    raise ValueError('source initial pose not one of recorded full-scan repaired starts')
                candidate = selected_start['record']['pose']
                if (not np.allclose(candidate['position'], slot['initial_pose']['position'], rtol=0, atol=1e-10)
                    or abs(np.dot(candidate['orientation'], slot['initial_pose']['orientation'])) < 1.-1e-12):
                    raise ValueError('source/selected pose differs beyond floating normalization')
                matched[global_slot] = dict(origin=origin, source=str(file), source_sha256=h,
                    local_slot=slot['slot'], fullscan_selected_index=selected_start['index'],
                    fullscan_grid_peak=selected_start['record']['scan_row'],
                    fullscan_repaired_start_pose=slot['initial_pose'],
                    fullscan_repaired_start_overlap=moments(slot['initial_validation']),
                    original_optimized_overlap=moments(slot['optimized_validation']),
                    original_search_steps=data['config']['search_steps'])
    if len(matched) != len(targets):
        raise ValueError(f'matched {len(matched)} of {len(targets)} source poses')
    rows = []
    for result in recovery['slots']:
        lineage = matched[result['slot']]
        value = result['recovered_overlap_A3']
        final_mean = result['archived_overlap_A3']
        initial_mean = lineage['fullscan_repaired_start_overlap']['volume']
        original_mean = lineage['original_optimized_overlap']['volume']
        rows.append(dict(slot=result['slot'], repaired=result['repair_succeeded'],
                    coarse_repaired_overlap_A3=value,
                    coarse_score_se_A3=result['score']['score']['standard_error'] if result['score'] else None,
                    lineage=lineage, final_union_optimized_overlap_A3=final_mean,
                    ratio_to_fullscan_repaired=value/initial_mean,
                    ratio_to_original_optimized=value/original_mean,
                    ratio_to_final_union_optimized=value/final_mean))
    groups = {}
    for name, group in [('all', rows), ('cpu_origin', [r for r in rows if r['lineage']['origin']=='cpu']),
                        ('gpu_origin', [r for r in rows if r['lineage']['origin']=='gpu'])]:
        if not group:
            continue
        groups[name] = dict(denominator=len(group), hard_valid=sum(r['repaired'] for r in group))
        for metric in ('fullscan_repaired', 'original_optimized', 'final_union_optimized'):
            vals = np.array([r['ratio_to_'+metric] for r in group])
            groups[name][metric] = dict(mean_ratio=float(vals.mean()), median_ratio=float(np.median(vals)),
                                        at_least_90pct=int(sum(vals >= .9)))
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.out/Path(__file__).name)
    report = dict(schema='fft-recovery-stage-audit-v1', complete=True,
                  all_slot_denominator=plan['all_slot_denominator'], native_information=False,
                  groups=groups, slots=rows, total_recovery_cpu_seconds=recovery['total_cpu_seconds'],
                  physical_jobs=0, source_provenance=provenance, script_sha256=digest(Path(__file__).read_bytes()),
                  limitations=['Full-scan repaired starts were selected from a ranked pool and NMS; coarse starts were conditioned on existing reference neighborhoods.',
                      'GPU-origin starts additionally received fine-grid rescoring and perturbation search before exact repair.',
                      'The coarse repair uses400 steps and no overlap optimization; original centers use2000 search steps, final union centers another2000.',
                      'Score ratio counts are descriptive point estimates, not confidence or basin-recovery guarantees.',
                      'Therefore neither the original comparison nor this lineage audit isolates angular resolution.'])
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(groups=groups, all_slot_denominator=plan['all_slot_denominator']), indent=2))


if __name__ == '__main__':
    main()
