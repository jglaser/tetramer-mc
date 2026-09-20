#!/usr/bin/env python3
"""Freeze disjoint outer-reference allocation with a complete geometric remainder."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from prepare_intermediate_local_region import ROOT, GUIDES, BINARY, BINARY_SHA, prepare, read, write, sha


def freeze(out, geometry):
    assert not out.exists(), 'Use a fresh output directory'
    diagnostic = read(geometry/'analysis.json')
    assert diagnostic['complete']
    assert sha(geometry/'protocol.json') == diagnostic['protocol_sha256']
    probe = read(geometry/'protocol.json')
    for name, h in probe['archive_sha256'].items(): assert sha(geometry/'provenance'/name) == h
    assert [(p['radius'], p['draws']) for p in diagnostic['probes']] == [(8.,2048),(16.,2048),(32.,2048)]
    for p in diagnostic['probes']: assert sha(geometry/f"r{int(p['radius'])}-geometry.jsonl") == p['samples_sha256']
    out.mkdir(parents=True)
    allocations = [(8,32768,100301010),(16,131072,100401010),(32,524288,100501010)]
    plans = []
    for radius, samples, seed in allocations:
        path = out/f'r{radius}-plan'
        prepare(path, float(radius), samples, 4, seed)
        plans.append(dict(radius=radius, samples_per_population=samples, populations=4,
                          seeds=[seed+1009*i for i in range(4)], workers=4,
                          config=str(path/'config.json'), region=str(path/'region.json'),
                          region_sha256=sha(path/'region.json'), local_plan_sha256=sha(path/'protocol.json'),
                          integrated_piece=dict(minimum=radius/2, maximum=radius, lower_inclusive=False, upper_inclusive=True),
                          campaign=str(ROOT/f'runs/ab-intermediate-weighted-r{radius}-reference-4x{samples}-l64-20260920')))
    complete = GUIDES/'provenance/source-region.json'
    assert sha(complete) == read(GUIDES/'freeze.json')['archived_sha256']['source-region.json']
    shutil.copy2(complete, out/'complete-cover-region.json')
    shutil.copy2(GUIDES/'config.json', out/'config.json')
    r4 = ROOT/'runs/ab-intermediate-weighted-r4-reference-4x16384-l64-20260920'
    r4_manifest = read(r4/'manifest.json')
    assert r4_manifest['region_sha256'] == '67e30f37418c382b89505309329190cad4a4983d61e393f67b903b943f8186b5'
    for job in r4_manifest['jobs']: assert read(Path(job['directory'])/'summary.json')['complete']
    archive = out/'provenance'; archive.mkdir()
    for source, name in [(Path(__file__),'prepare_intermediate_outer_campaign.py'),
                         (geometry/'analysis.json','geometry-analysis.json'),(geometry/'protocol.json','geometry-protocol.json')]:
        shutil.copy2(source, archive/name)
    protocol = dict(schema='intermediate-complete-radial-allocation-v1', created_utc=datetime.now(timezone.utc).isoformat(),
        physical=dict(activity=.035,depletant_radius=1.5,capture_radius=18.,original_q_minimum=2.,original_q_maximum=5.,
                      original_q_lower_inclusive=True,original_q_upper_inclusive=False),
        chart_model=str(GUIDES/'model-weighted.json'),chart_model_sha256=sha(GUIDES/'model-weighted.json'),
        executable=str(BINARY),executable_sha256=BINARY_SHA,lambda_ratio=64.,cloud_replicates=2,
        boundaries=[0.,4.,8.,16.,32.],existing_inner_reference=str(r4),
        existing_inner_manifest_sha256=sha(r4/'manifest.json'),local_references=plans,
        remainder=dict(samples_per_population=262144,populations=4,workers=4,seeds=[100601010+1009*i for i in range(4)],
                       config=str(out/'config.json'),region=str(out/'complete-cover-region.json'),
                       region_sha256=sha(out/'complete-cover-region.json'),
                       campaign=str(ROOT/'runs/ab-intermediate-complete-remainder-reference-4x262144-l64-20260920'),
                       mask='Weighted-chart radius >32, original [2,5), capture and AB hard support; direct zero-inclusive contribution, never a difference of estimated totals.'),
        complete_cover_bound=diagnostic['complete_cover_bound'],
        allocation_reason='Cloud-free geometric hit rates .251465, .053223, .013672 motivate increasing fixed budgets to roughly comparable valid counts; variance of physical weights is not inferred from geometric validity.',
        selection='Region chart and original inner boundary predate the inner reference. Outer partitions/counts use historical contacts and independent cloud-free probes. All new physical draws follow this frozen plan; no optional stopping or refitting. Historical guide restrictions remain exploratory.',
        sum_rule='Use existing r<=4 reference plus fresh 4<r<=8,8<r<=16,16<r<=32 and r>32 contributions from distinct populations. Every summand uses its own original unconditional N and proposal Jacobian. Never pool campaigns under a common density or subtract noisy totals.',
        coverage='Disjoint pieces cover original [2,5) with the exact q/capture/AB masks. The complete geometric reference covers every remaining pose. Proposal support alone does not prove finite-sample convergence.',
        variance='Sum independent regional estimator variances; report per-region row/population errors, maxima and paired-cloud noise. A four-group total may be formed by pairing independent population indices across regions.',
        archive_sha256={p.name:sha(p) for p in archive.iterdir()})
    assert sha(BINARY) == BINARY_SHA
    write(out/'protocol.json',protocol)
    write(out/'freeze.json',dict(protocol_sha256=sha(out/'protocol.json'),config_sha256=sha(out/'config.json'),
                                complete_cover_region_sha256=sha(out/'complete-cover-region.json')))
    print(json.dumps(dict(out=str(out),protocol_sha256=sha(out/'protocol.json'),campaigns=len(plans)+1)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--geometry',type=Path,default=ROOT/'runs/ab-intermediate-outer-geometry-v2-20260920')
    args=parser.parse_args();freeze(args.out.resolve(),args.geometry.resolve())
