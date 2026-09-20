#!/usr/bin/env python3
"""Combine independently audited dynamic/frozen/off memory controls.

Verifies identical physical preparations and identical initial frozen/dynamic
banks, then adds proposal-source rejection and total-cost diagnostics. This
never reruns simulations or changes the native classifier.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
from collections import Counter,defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from analyze_free_tetramer_campaign import plot

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def proposal_diagnostics(directory):
    counts=defaultdict(Counter);numbers=defaultdict(lambda:defaultdict(list))
    for line in (directory/'moves.jsonl').open():
        move=json.loads(line)
        if move['kind']!='global':continue
        proposal=move['proposal']
        source=('uniform' if proposal['branch']=='uniform' else 'memory' if proposal.get('contact_memory_slot') is not None else 'base')
        row=counts[source];row['attempted']+=1;row['hard_valid']+=int(move['hard_valid']);row['accepted']+=int(move['accepted'])
        row['null']+=int(move['proposed_pose'] is None)
        if move['hard_valid']:
            alpha=move['log_acceptance'];numbers[source]['log_acceptance'].append(alpha)
            numbers[source]['conditional_acceptance_probability'].append(math.exp(alpha))
            numbers[source]['log_reverse_forward'].append(proposal['log_reverse_forward'])
            numbers[source]['gate_log_weight'].append(move['gate']['log_weight'])
    result={}
    for source,count in counts.items():
        result[source]=dict(count)
        for name,values in numbers[source].items():
            result[source][name]=dict(mean=float(np.mean(values)),median=float(np.median(values)),minimum=min(values),maximum=max(values),samples=len(values))
        result[source]['expected_acceptances_given_sampled_gates']=sum(numbers[source]['conditional_acceptance_probability'])
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dynamic',type=Path,required=True)
    parser.add_argument('--frozen',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    paths=[args.dynamic.resolve(),args.frozen.resolve()];runs=[];manifests=[]
    for path in paths:
        assessment=read(path/'assessment/analysis.json');assert assessment['passed']
        manifests.append(read(path/'manifest.json'))
        for run in assessment['results']:
            assert run['passed'];directory=path/'runs'/run['id']
            for source,expected in run['source_sha256'].items():assert sha(source)==expected
            run=copy.deepcopy(run);run['proposal_source_diagnostics']=proposal_diagnostics(directory)
            run['campaign']=str(path);run['directory']=str(directory);runs.append(run)
    ordering={'memory-off':0,'memory-frozen':1,'memory-on':2};runs.sort(key=lambda r:(r['replicate'],ordering[r['variant']]))
    pair_checks=[]
    for replicate in sorted({r['replicate'] for r in runs}):
        selected=[r for r in runs if r['replicate']==replicate];assert len(selected)==3
        frames={r['variant']:json.loads((Path(r['directory'])/'trajectory.jsonl').read_text().splitlines()[0]) for r in selected}
        cfgs=[read(Path(r['directory'])/'config.json') for r in selected]
        assert all(cfg['seed']==cfgs[0]['seed'] and cfg['initial_poses']==cfgs[0]['initial_poses'] for cfg in cfgs)
        assert all(frame['poses']==frames['memory-off']['poses'] for frame in frames.values())
        assert frames['memory-on']['contact_memory_state']==frames['memory-frozen']['contact_memory_state']
        on=next(c for c in cfgs if c['resolved_contact_memory'] and c['resolved_contact_memory']['attempts_per_sweep']>0)
        frozen=next(c for c in cfgs if c['resolved_contact_memory'] and c['resolved_contact_memory']['attempts_per_sweep']==0)
        left=copy.deepcopy(on['resolved_contact_memory']);right=copy.deepcopy(frozen['resolved_contact_memory'])
        left.pop('attempts_per_sweep');right.pop('attempts_per_sweep');assert left==right
        pair_checks.append(dict(replicate=replicate,physical_initial_poses_equal=True,mc_seeds_equal=True,
            frozen_and_dynamic_initial_banks_equal=True,resolved_bank_parameters_only_differ_in_update_attempts=True))
    pooled={}
    for variant in ordering:
        selected=[r for r in runs if r['variant']==variant]
        cpu=sum(r['cpu_seconds'] for r in selected);counts=Counter()
        for r in selected:
            for source,row in r['proposal_source_diagnostics'].items():
                for field in ('attempted','hard_valid','accepted'):counts[f'{source}_{field}']+=row[field]
        pooled[variant]=dict(total_cpu_seconds=cpu,proposal_counts=dict(counts),
            accepted_global_per_cpu_second=sum(r['counts']['global']['accepted'] for r in selected)/cpu,
            observed_nonspecific_environment_returns=sum(r['observed_environment_returns']['nonspecific'] for r in selected),
            native_formations=sum(r['motif_formations'] for r in selected),native_breakages=sum(r['motif_breakages'] for r in selected),
            memory_native_slots_ever=sum(len(r['contact_memory']['native_slots_ever']) if r['contact_memory'] else 0 for r in selected),
            memory_update_cpu_seconds=sum(r['cost'].get('contact_memory_cpu_seconds',0.) for r in selected))
    result=dict(passed=True,results=runs,pooled=pooled,pair_checks=pair_checks,source_campaigns=[dict(path=str(p),manifest_sha256=sha(p/'manifest.json'),assessment_sha256=sha(p/'assessment/analysis.json')) for p in paths],
        binary_sha256=[m['binary_sha256'] for m in manifests],source_sha256=sha(Path(__file__)))
    save(args.out/'analysis.json',result)
    manifest=copy.deepcopy(manifests[0]);manifest['compare_memory']=True
    plot(args.out,runs,manifest)
    sweeps=manifests[0]['sweeps'];assert all(m['sweeps']==sweeps for m in manifests)
    bank_config=next(r['contact_memory']['resolved_config'] for r in runs if r['variant']=='memory-on')
    lines=['# Static and evolving contact-memory controls','',
        f"{len(pair_checks)} matched, genuinely dispersed preparations of {len(runs[0]['rows'][0]['native_neighbor_sets'])} mobile tetramers; {sweeps:,} sweeps each. Every arm uses the same geometry-only base atlas, physical hard/depletion target, local/GCA/shift schedule, RJ count prior, and conditional mean transport. The frozen and evolving banks start from exactly the same {bank_config['slots']} relative poses. A frozen bank has zero pair updates; an evolving bank has {bank_config['attempts_per_sweep']} per sweep. Neither supplies native docking charts.",'',
        '| Start | Arm | Largest registered / nonspecific component | Bank accepted | Global accepted (base/memory/uniform) | CPU s |',
        '|---|---|---:|---:|---:|---:|']
    for r in runs:
        bank=r['contact_memory'];count=bank['counts'].get('accepted',0) if bank else 0;sources=r['accepted_global_sources']
        lines.append(f"| {r['replicate']} | {r['variant']} | {r['final_native']['largest_component_size']} / {r['final_nonspecific']['largest_component_size']} | {count} | {sources.get('base',0)}/{sources.get('memory',0)}/{sources.get('uniform',0)} | {r['cpu_seconds']:.2f} |")
    lines+=['','## Interpretation','',
        'Native registration and bank-native discovery remain separate observables. Physical non-native contact changes establish movement between neighborhoods, not discovery of a native pose. Global acceptance, when dominated by uniform draws, cannot be credited to the learned memory charts. Total CPU includes the bank work. These four correlated, matched experiments do not establish equilibrium or a mixing-time advantage.','',
        '| Arm | Total CPU s | Bank update CPU s | Global accepted / CPU s | Native formations / breakages | Bank-native slots ever |',
        '|---|---:|---:|---:|---:|---:|']
    for variant,row in pooled.items():
        lines.append(f"| {variant} | {row['total_cpu_seconds']:.2f} | {row['memory_update_cpu_seconds']:.2f} | {row['accepted_global_per_cpu_second']:.3f} | {row['native_formations']}/{row['native_breakages']} | {row['memory_native_slots_ever']} |")
    lines+=['','## Proposal diagnostic','',
        'The table separates geometric rejection from acceptance after the exact conditional Poisson gate. Expected acceptance counts are the sum of final acceptance probabilities over sampled auxiliary clouds. They are descriptive Monte Carlo diagnostics, not independent free-energy measurements; actual accept/reject decisions are reported separately.','',
        '| Arm | Source | Proposed | Hard valid | Accepted |',
        '|---|---|---:|---:|---:|']
    for variant,row in pooled.items():
        for source in ('base','memory','uniform'):
            c=row['proposal_counts'];lines.append(f"| {variant} | {source} | {c.get(source+'_attempted',0)} | {c.get(source+'_hard_valid',0)} | {c.get(source+'_accepted',0)} |")
    lines+=['','For memory-chart draws that pass the hard test:','',
        '| Start | Arm | Hard valid / proposed | Median log(q reverse/q forward) | Median final log acceptance | Expected accepted draws |',
        '|---|---|---:|---:|---:|---:|']
    for r in runs:
        if 'memory' not in r['proposal_source_diagnostics']:continue
        d=r['proposal_source_diagnostics']['memory']
        lines.append(f"| {r['replicate']} | {r['variant']} | {d['hard_valid']}/{d['attempted']} | {d.get('log_reverse_forward',{}).get('median',float('nan')):.2f} | {d.get('log_acceptance',{}).get('median',float('nan')):.2f} | {d['expected_acceptances_given_sampled_gates']:.3g} |")
    lines+=['','## Independent verification','',
        f"The contributing audits reconstruct {sum(r['replay']['records'] for r in runs):,} logged updates to {sum(r['replay']['frames'] for r in runs):,} frames/checkpoints, including physical, RJ and contact-bank moves. Every saved physical state and every retained bank pose is checked independently for hard geometry; the physical wall is checked atom by atom. Native templates are post-hoc analysis only.",'',
        'The baseline/evolving campaign and later frozen control archive separate executable hashes because the frozen control required allowing a zero bank-update budget. The initial bank equality and all other resolved parameters are explicitly checked. Raw CPU differences should not be interpreted as a proven speedup.','',
        '\n'.join(f"- Campaign {path}: binary `{manifest['binary_sha256']}`" for path,manifest in zip(paths,manifests)),'']
    (args.out/'report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(passed=True,runs=len(runs),pooled=pooled)),flush=True)

if __name__=='__main__':main()
