#!/usr/bin/env python3
"""Extract every cached IID source from an authenticated completed reference.

No geometry, source generation, kernel calls or outcome-dependent selection.
The physical source law is inherited from the completed independent reference
and its separate full journal audit. The large kernel journal is read once;
the large source-point journal is not rescanned.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT/'results/flexible-surrogate-reference-20261004'
SCHEMA = 'partner-atlas-iid-source-cache-v1'
STREAMS, PER_STREAM, TOTAL = 4, 2048, 8192
OLD_ARMS = ('m1_guided','m8_guided','m8_zero')
MAX_ROW_BYTES = 16*1024**2
PINS = {
    'validation.json':'84d0b14661f5b137cd84b130add0b6cb5fac6c691eb35d57c4af30d158f2428f',
    'reference/receipt.json':'d5aac345bd46dabf525b59e7b5c309caf29dc1eb19734bfdc1f7529361c4801d',
    'audit-validation.json':'98e4657314153617ab1a1277c0f9827c295752112b65eafbafe40004dd722146',
    'journal-audit.json':'2f22d5e47fcf717299ca140b1b9beb15748c7eef5c424f8a899e0e5e46098c79',
    'reference/protocol.json':'15c9293a6d7aafdd92246eaef8b8c81d407060c1a4955bf84ace40b42ee78165',
    'reference/summary.json':'48ffa59bac8ef551184c5100213c42de2eecdda1b16ebff88170245ad02ac666',
}
JOURNAL_SHA = 'b8218bb4bb658587fa70540f2a423b620cdfd600a083d179b70a82a983d454c5'
SOURCE_JOURNAL_SHA = 'b126f2dcc3d9d9cc9139af16abd659e596513fbaf6e85bb0912e360c338e468d'
REFERENCE_SOURCE_SHA = '7f57221b5d5ac25695bc99185fa8f6c38f49ceec0aa02ce11d4ba69d95990875'
OBSERVABLES = [
    'x0','y0','z0','x1','y1','z1','radius0_squared','radius1_squared',
    'radius_squared_label_difference','internal_separation','analytic_internal_lens',
    'analytic_fixed_lens0','analytic_fixed_lens1','world_quadrature_spectator_coverage',
    'world_quadrature_triple_coverage','attraction_volume_diagnostic','internal_contact',
    *[f'R{body}_{row}{column}' for body in range(2) for column in range(3) for row in range(3)],
    'q0_scalar_squared','q1_scalar_squared','relative_quaternion_scalar_squared',
]
PHYSICAL = dict(core_radius=.1,rd=.9,inflated_radius=1.,wall_radius=1.6,center_radius=1.5,
    activity=.5,**{'lambda':2.},fixed_spectator=dict(position=[0.,0.,0.],orientation=[1.,0.,0.,0.]),
    members=[0,1],body_midpoint_grid_side=8,body_raw_count=512,body_retained_count=280,
    point_volume=.015625,world_observable_grid_side=9,translation_std=.25,rotation_std_degrees=30.,
    gate_options=dict(max_cells=31,max_depth=8,min_width=0.))
PROSPECTIVE = dict(arms=[dict(id='direct',candidates_per_source=1),
    dict(id='m1_guided',candidates_per_source=1),dict(id='m8_guided',candidates_per_source=8),
    dict(id='flat8',candidates_per_source=8)],kernel_calls=32768,fixed_candidates=147456,
    streams=4,sources_per_stream=2048,new_source_draws=0,retries=0,replacements=0,extensions=0)


def require(condition,message):
    if not condition: raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key,value in items:
            require(key not in result,'Duplicate JSON key '+key); result[key] = value
        return result
    def invalid(value): raise ValueError('Nonfinite JSON constant '+value)
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())


def validate_metadata(validation,receipt,audit_validation,audit,protocol,summary):
    require(validation['schema'] == 'flexible-surrogate-reference-validation-v1'
        and all(validation[k] is True for k in ('complete','passed','child_drained'))
        and validation['returncode'] == 0 and validation['error'] is None
        and validation['source_before'] == validation['source_after']
        and validation['production_before'] == validation['production_after'], 'Original validation incomplete or changed')
    expected = {name.removeprefix('reference/'):digest for name,digest in PINS.items() if name.startswith('reference/')}
    expected.update({'kernel-attempts.jsonl':JOURNAL_SHA,'source-attempts.jsonl':SOURCE_JOURNAL_SHA})
    require(validation['outputs'] == {'reference/'+name:digest for name,digest in expected.items()},
        'Original validation output bindings differ')
    require(receipt['schema'] == 'flexible-surrogate-independent-reference-receipt-v1'
        and all(receipt[k] is True for k in ('complete','passed','numerical_allocation_complete','scientific_checks_passed'))
        and receipt['error'] is None and receipt['retries'] == receipt['replacement_draws'] == 0
        and receipt['output_sha256'] == {name:digest for name,digest in expected.items() if name != 'receipt.json'},
        'Original scientific receipt incomplete or changed')
    require(all(audit_validation[k] is True for k in ('complete','passed','child_drained','source_unchanged'))
        and audit_validation['returncode'] == 0 and audit_validation['error'] is None
        and audit_validation['audit_sha256'] == PINS['journal-audit.json'], 'Independent audit did not drain successfully')
    require(audit['schema'] == 'flexible-surrogate-reference-journal-audit-v1'
        and audit['complete'] is True and audit['passed'] is True and audit['reference_scientific_checks_passed'] is True
        and audit['validation_sha256'] == PINS['validation.json'] and audit['input_sha256'] == expected
        and audit['inventory'] == {arm:TOTAL for arm in OLD_ARMS}, 'Independent audit inventory/bindings differ')
    require(protocol['schema'] == 'flexible-surrogate-independent-reference-protocol-v1'
        and protocol['streams'] == STREAMS and protocol['sources_per_stream'] == PER_STREAM
        and protocol['arms'] == [[1,1.,'m1_guided'],[8,1.,'m8_guided'],[8,0.,'m8_zero']]
        and protocol['source_sampler_exact'] is True
        and protocol['reference_source_sha256'] == REFERENCE_SOURCE_SHA
        and validation['source_before']['tests/flexible_surrogate_stationarity.rs'] == REFERENCE_SOURCE_SHA
        and all(protocol[k] == v for k,v in PHYSICAL.items()), 'Original physical reference law differs')
    require(protocol['observable_names'] == OBSERVABLES and len(OBSERVABLES) == 38
        and summary['observable_names'] == OBSERVABLES
        and summary['schema'] == 'flexible-surrogate-independent-reference-summary-v1'
        and summary['complete'] is True, 'Original 38-observable reference differs')
    counts = receipt['source_counts']
    require(len(counts) == STREAMS and summary['source_counts'] == audit['source_counts'] == counts
        and all(type(c['void_accepted']) is int and c['void_accepted'] == PER_STREAM
            and type(c['attempts']) is int and c['attempts'] >= PER_STREAM for c in counts), 'Incomplete IID source inventory')
    require(len(summary['stream_panel']) == STREAMS and all(p['stream'] == i and p['source_counts'] == counts[i]
        for i,p in enumerate(summary['stream_panel'])), 'Original source stream panels differ')


def authenticate(reference):
    """Small metadata/source files only: do not hash either large journal here."""
    reference = Path(reference).resolve(); values,authority = {},{}
    for name,digest in PINS.items():
        path = reference/name; require(sha(path) == digest,'Unauthenticated '+name)
        values[name] = strict_json(path.read_bytes()); authority[name] = dict(path=str(path),sha256=digest)
    validate_metadata(*(values[name] for name in ('validation.json','reference/receipt.json','audit-validation.json',
        'journal-audit.json','reference/protocol.json','reference/summary.json')))
    for name,digest in [('source/tests/flexible_surrogate_stationarity.rs',REFERENCE_SOURCE_SHA),
                        ('audit.py',values['journal-audit.json']['auditor_sha256'])]:
        path = reference/name; require(sha(path) == digest,'Archived reference source changed '+name)
        authority[name] = dict(path=str(path),sha256=digest)
    return values,authority


def validate_pair(pair):
    require(type(pair) is list and len(pair) == 2,'Invalid cached pair shape')
    for pose in pair:
        require(type(pose) is dict and set(pose) == {'position','orientation'},'Invalid cached pose fields')
        for key,size in [('position',3),('orientation',4)]:
            require(type(pose[key]) is list and len(pose[key]) == size
                and all(type(x) in (int,float) and math.isfinite(x) for x in pose[key]),'Invalid cached pose coordinates')
        require(abs(math.fsum(x*x for x in pose['orientation'])-1.) <= 2e-13,'Invalid cached quaternion norm')


def extract_journal(journal,output,expected_sha,streams,per_stream,expected_last_attempts):
    """Stream all begin/outcome pairs; selection reads only m1 begin.old.

    Smaller inventories are accepted by this helper solely for synthetic tests;
    the public preparer fixes streams=4 and per_stream=2048.
    """
    require(type(streams) is int and streams > 0 and type(per_stream) is int and per_stream > 0,
        'Invalid extraction inventory')
    require(len(expected_last_attempts) == streams,'Missing source attempt endpoints')
    journal,output = Path(journal),Path(output)
    before = journal.stat(); signature = lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    digest = hashlib.sha256(); output_digest = hashlib.sha256(); last_attempts = [0]*streams
    rows = sources = 0; pair = None; source_attempt = None
    with journal.open('rb') as source,output.open('xb') as destination:
        def next_row():
            nonlocal rows
            raw = source.readline(MAX_ROW_BYTES+1)
            require(raw and len(raw) <= MAX_ROW_BYTES and raw.endswith(b'\n'),'Missing, oversized or incomplete journal row')
            digest.update(raw); rows += 1; return strict_json(raw)
        for ordinal in range(streams*per_stream*len(OLD_ARMS)):
            stream,index = divmod(ordinal//len(OLD_ARMS),per_stream)
            arm_index = ordinal % len(OLD_ARMS); arm = OLD_ARMS[arm_index]
            begin,outcome = next_row(),next_row()
            require(begin['kind'] == 'kernel_begin' and outcome['kind'] == 'kernel_outcome','Wrong kernel row pairing')
            require(type(begin['stream']) is int and type(begin['source_index']) is int
                and (begin['stream'],begin['source_index'],begin['arm']) == (stream,index,arm),
                'Missing, duplicate or reordered kernel source/arm')
            require(type(begin['source_attempt']) is int and begin['source_attempt'] > 0,'Invalid source attempt')
            require(all(outcome[k] == begin[k] for k in ('stream','source_index','arm','source_attempt')),
                'Kernel begin/outcome identity differs')
            validate_pair(begin['old'])
            if arm_index == 0:
                require(begin['source_attempt'] > last_attempts[stream],'Source attempts not strictly increasing')
                pair,source_attempt = begin['old'],begin['source_attempt']; last_attempts[stream] = source_attempt
                row = dict(kind='cached_iid_source',stream=stream,source_index=index,
                    source_attempt=source_attempt,old=pair)
                encoded = (json.dumps(row,separators=(',',':'),allow_nan=False)+'\n').encode()
                destination.write(encoded); output_digest.update(encoded); sources += 1
            else:
                require(begin['old'] == pair and begin['source_attempt'] == source_attempt,
                    'Original arms do not share the same IID source')
            # Deliberately no reads of accepted/retained/score/observables here.
            # The separately authenticated journal audit covered old outcomes.
        require(source.read(1) == b'','Extra journal rows after fixed inventory')
        require(digest.hexdigest() == expected_sha,'Kernel journal SHA differs')
        require(last_attempts == expected_last_attempts,'Final source attempts differ from authenticated counts')
        require(signature(journal.stat()) == signature(before),'Kernel journal changed during extraction')
        destination.flush(); os.fsync(destination.fileno())
    return dict(total_sources=sources,streams=streams,sources_per_stream=per_stream,
        scanned_rows=rows,kernel_journal_sha256=digest.hexdigest(),sources_sha256=output_digest.hexdigest(),
        last_source_attempts=last_attempts,source_point_journal_scans=0,geometry_queries=0)


def prepare(out,reference=REFERENCE):
    out,reference = Path(out).resolve(),Path(reference).resolve()
    require(not out.exists(),'Fresh source cache directory required; no replacement')
    values,authority = authenticate(reference)
    out.mkdir(parents=True); copied = out/'authority'; copied.mkdir()
    for name,binding in authority.items():
        target = copied/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(binding['path'],target)
        require(sha(target) == binding['sha256'],'Copied authority changed')
    source_dir = out/'source'; source_dir.mkdir()
    for path in (Path(__file__).resolve(),Path(__file__).with_name('test_prepare_partner_atlas_reference_sources.py')):
        shutil.copyfile(path,source_dir/path.name)
    write(out/'plan.json',dict(schema='partner-atlas-iid-source-extraction-plan-v1',complete=True,launched=False,
        streams=STREAMS,sources_per_stream=PER_STREAM,total_sources=TOTAL,
        kernel_journal=dict(path=str(reference/'reference/kernel-attempts.jsonl'),sha256=JOURNAL_SHA),
        authority=authority,prospective_reference=PROSPECTIVE,max_row_bytes=MAX_ROW_BYTES,
        source_sha256={p.name:sha(p) for p in source_dir.iterdir()},
        source_selection='All m1_guided kernel_begin.old entries, in original stream/index order; no outcome conditioning.',
        new_source_draws=0,new_kernel_calls=0,new_geometry_queries=0,retries=0,replacements=0,extensions=0))
    try:
        inventory = extract_journal(reference/'reference/kernel-attempts.jsonl',out/'sources.jsonl',JOURNAL_SHA,
            STREAMS,PER_STREAM,[c['attempts'] for c in values['reference/receipt.json']['source_counts']])
        require(inventory['total_sources'] == TOTAL,'Incomplete full IID source cache')
        for name,binding in authority.items():
            require(sha(binding['path']) == binding['sha256'] and sha(copied/name) == binding['sha256'],
                'Authority changed during extraction')
        authority['reference/kernel-attempts.jsonl'] = dict(path=str(reference/'reference/kernel-attempts.jsonl'),sha256=JOURNAL_SHA)
        receipt = dict(schema=SCHEMA,complete=True,passed=True,streams=STREAMS,sources_per_stream=PER_STREAM,
            total_sources=TOTAL,sources=dict(path=str(out/'sources.jsonl'),sha256=inventory['sources_sha256']),
            original_protocol=dict(path=str(copied/'reference/protocol.json'),sha256=PINS['reference/protocol.json']),
            authority=authority,inventory=inventory,physical_target=PHYSICAL,observable_names=OBSERVABLES,
            extraction_plan=dict(path=str(out/'plan.json'),sha256=sha(out/'plan.json')),
            source_selection='Every m1_guided kernel_begin.old; all 4x2048 IID sources retained without acceptance, score, contact or native-label filtering.',
            prospective_reference=PROSPECTIVE,new_source_draws=0,new_kernel_calls=0,new_geometry_queries=0,
            source_point_journal_rescanned=False,retries=0,replacements=0,extensions=0,
            inherited_source_point_journal=dict(path=str(reference/'reference/source-attempts.jsonl'),
                sha256=SOURCE_JOURNAL_SHA,authority='Historical completed independent journal audit; not reread by extraction'),
            scope='Authenticated cached IID source reuse only. No stationarity or efficiency result for the new kernels is asserted.')
        write(out/'receipt.json',receipt); return receipt
    except BaseException as error:
        write(out/'failure.json',dict(schema=SCHEMA,complete=False,passed=False,error=f'{type(error).__name__}: {error}',
            partial_sources_preserved=(out/'sources.jsonl').exists(),new_source_draws=0,new_kernel_calls=0,
            retries=0,replacements=0,extensions=0))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--reference',type=Path,default=REFERENCE)
    args = parser.parse_args(); result = prepare(args.out,args.reference)
    print(json.dumps(dict(complete=True,passed=True,total_sources=result['total_sources'],new_source_draws=0,new_kernel_calls=0)))
