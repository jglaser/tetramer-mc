#!/usr/bin/env python3
"""Combine disjoint shells from independent fixed-radius local references."""
import argparse
from pathlib import Path

from scipy.special import logsumexp

from analyze_intermediate_partition import total_independent
from compare_intermediate_reference import population_rse
from prepare_cayley_rms_cover import read, write, sha, require
from run_shoulder_mis_campaign import local_dependencies


def combine(source, out):
    require(not out.exists(), 'Use a fresh derived directory')
    audit = read(source); require(audit['complete'], 'Need completed independent audit')
    require(audit['radial_edges_A'] == [0., .5, 1., 2.], 'Changed radial partition')
    require(len(audit['campaigns']) == 3, 'Need three distinct reference campaigns')
    campaigns = {c['radius_A']:c for c in audit['campaigns']}
    require(set(campaigns) == {.5, 1., 2.}, 'Changed source radii')
    for name, digest in audit['archived_sha256'].items():
        require(sha(source.parent/'provenance'/name) == digest, 'Changed audited source archive')
    parts=[]; seeds=set(); signature=None
    for i, radius in enumerate((.5, 1., 2.)):
        campaign=campaigns[radius]; current=campaign['original_reference_audit']['physical_signature']
        if signature is None: signature=current
        require(current == signature, 'Different physical target')
        require(len(campaign['populations']) == 4, 'Expected four independent populations')
        for population in campaign['populations']:
            require(population['seed'] not in seeds, 'Duplicate random stream'); seeds.add(population['seed'])
        key=f'radial_{i}'
        parts.append(dict(minimum=(0., .5, 1.)[i], maximum=radius, source=campaign['root'],
                          key=key, physical=campaign['physical'][key], hard=campaign['hard'][key],
                          populations=campaign['populations'], CPU_seconds=campaign['CPU_seconds']))
    results=[]
    for count in (2, 3):
        chosen=parts[:count]; result=dict(radius_A=chosen[-1]['maximum'],
                                         CPU_seconds=sum(p['CPU_seconds'] for p in chosen))
        for kind in ('physical','hard'):
            result[kind]=total_independent([p[kind] for p in chosen])
            groups=[]
            for i in range(4):
                values=[p['populations'][i][kind][p['key']]['logQ'] for p in chosen]
                groups.append(float(logsumexp([v for v in values if v is not None])) if any(v is not None for v in values) else None)
            result[kind]['population_group_logQ']=groups
            result[kind]['population_group_RSE']=population_rse(groups)
        results.append(result)
    output=dict(complete=True, source=str(source), source_sha256=sha(source), pieces=parts, totals=results,
                allocation='Smallest reference ball covering each of the already-declared radial bands: [0,.5],(.5,1],(1,2]. Every piece uses its original full N.',
                scope='Derived disjoint sum of independently sampled regions, not pooling or summing nested full-ball estimates. Complete only inside the new radius-two chart ball intersected with original masks; whole intermediate complement remains unresolved. Observed uncertainty cannot bound unseen weight.')
    (out/'provenance').mkdir(parents=True)
    (out/'provenance/source-audit.json').write_bytes(source.read_bytes())
    for name,path in local_dependencies([Path(__file__)]).items():
        (out/'provenance'/name).write_bytes(path.read_bytes())
    output['archived_sha256']={p.name:sha(p) for p in (out/'provenance').iterdir()}
    write(out/'analysis.json',output)
    for result in results:
        print(result['radius_A'], result['physical'], 'CPU',result['CPU_seconds'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();combine(args.source.resolve(),args.out.resolve())
