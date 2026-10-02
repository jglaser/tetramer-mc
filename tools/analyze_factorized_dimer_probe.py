#!/usr/bin/env python3
"""Independent passive factorized/whole-joint audit. No proposals or physical updates.

Reconstruct every raw edge from its saved random variates, including rejects;
score complete F only for assembled candidates. Strict independent atomic
predicates and recovered-frame predicates must agree; near-boundary ambiguity
is reported, never silently accepted as a retry. Timings include proposal source
checks but exclude serialization/contact diagnostics. Fixed contexts do not
constitute equilibrium samples or an assembly test.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='1'
import argparse
from collections import Counter
import hashlib
import math
from pathlib import Path
import time
import json
import numpy as np
from scipy.spatial import cKDTree
from analyze_capped_dimer_probe import (read,sha,bound,require,Checks,DimerDestinationDensity,DimerGeometry,
    bind_current_source,feasibility,is_feasible,verify_stopping,audit_unique_trials,decode_edge,
    relative_pose,compose_pose,fingerprints,quantiles,serial,REFERENCE_CONFIG_SHA256,PANEL_SHA256)

METHODS=['whole_joint','factorized']
CAPS=dict(root=32,internal=32,joint=1)
IDENTITY=dict(position=[0.,0.,0.],orientation=[1.,0.,0.,0.])


def stream_seed(master,atlas,case,attempt,role):
    return int(hashlib.sha256(f'factorized-dimer-probe-v1/{master}/{atlas}/{case}/{attempt}/{role}'.encode()).hexdigest()[:16],16)


def method_order(master,atlas,case,attempt):
    return METHODS if stream_seed(master,atlas,case,attempt,'execution_order')&1==0 else METHODS[::-1]


def strict_equal(actual,expected,checks,kind,tag):
    checks.exact(actual,expected,kind,tag)
    require(actual==expected,f'Fatal geometry/frame mismatch: {kind}, {tag}')


def root_geometry(oracle,case,pose):
    points=oracle.placed(pose)
    pairs=cKDTree(points).sparse_distance_matrix(oracle.tree,oracle.cutoff,output_type='ndarray')
    labels=pairs['j']//oracle.count
    keep=(labels!=case['root'])&(labels!=case['child'])
    ii,jj,labels=pairs['i'][keep],pairs['j'][keep],labels[keep]
    distances=np.linalg.norm(points[ii]-oracle.fixed[jj],axis=1)
    radii=oracle.radii[ii]+oracle.radii[jj%oracle.count]
    return dict(spectator_core_collisions=sorted(int(x) for x in np.unique(labels[distances<radii])),
        wall_valid=bool(np.all(np.sum(points*points,axis=1)<=(oracle.wall_radius-oracle.radii)**2)))


def internal_geometry(oracle,relative):
    points=oracle.placed(relative)
    pairs=cKDTree(oracle.centers).sparse_distance_matrix(cKDTree(points),oracle.cutoff,output_type='ndarray')
    ii,jj=pairs['i'],pairs['j'];d=np.linalg.norm(oracle.centers[ii]-points[jj],axis=1)
    radii=oracle.radii[ii]+oracle.radii[jj]
    return dict(internal_core_overlap=bool(np.any(d<radii)),internal_exclusion_contact=bool(np.any(d<radii+2*oracle.rd)))


def root_ok(value):return not value['spectator_core_collisions'] and value['wall_valid']
def internal_ok(value):return not value['internal_core_overlap'] and value['internal_exclusion_contact']


def audit_frame(oracle,case,anchor,members,world,frame,checks,tag):
    require(frame is not None,'Missing frame record')
    edges=[relative_pose(anchor,members[0]),relative_pose(members[0],members[1])]
    root=compose_pose(anchor,edges[0]);rebuilt=[root,compose_pose(root,edges[1])]
    for a,b in zip(frame['recovered_edges'],edges):checks.pose(a,b,'recovered frame edge',tag)
    for a,b in zip(frame['reconstructed_members'],rebuilt):checks.pose(a,b,'recomposed frame member',tag)
    # Evaluate predicates at recorded recomposed endpoints too; a pose tolerance
    # cannot excuse a different strict geometric branch.
    full=feasibility(oracle,case,frame['reconstructed_members'],oracle.fingerprint([case['root'],case['child']],frame['reconstructed_members']))
    r=root_geometry(oracle,case,frame['reconstructed_members'][0]);i=internal_geometry(oracle,frame['recovered_edges'][1])
    for actual,expected,kind in [(frame['reconstructed_feasibility'],full,'reconstructed full predicate'),
        (frame['reconstructed_root'],r,'reconstructed root predicate'),(frame['internal_relative'],i,'recovered internal predicate'),
        (full,world,'world/recomposed predicates'),
        (r,dict(spectator_core_collisions=world['spectator_core_collisions'][0],wall_valid=world['wall_valid'][0]),'world/root predicate'),
        (i,{k:world[k] for k in ['internal_core_overlap','internal_exclusion_contact']},'world/internal predicate')]:
        strict_equal(actual,expected,checks,kind,tag)


def audit_stage(helper,oracle,case,anchor,old_edge,records,stage,cap,checks,tag):
    require(len(records)<=cap,'Too many raw edge draws')
    passed=False
    for j,record in enumerate(records):
        where=f'{tag}/{stage}{j+1}'
        require(not passed,'Hidden retries after first passing edge')
        checks.exact(record['index'],j+1,'edge index',where)
        edge=record['draw'];checks.pose(edge['old_relative_pose'],old_edge,'old edge coordinates',where)
        relative=decode_edge(helper,edge,.5,160.,checks,where)
        if stage=='root':
            world=compose_pose(anchor,relative);checks.pose(record['world_pose'],world,'root world decoding',where)
            expected=root_geometry(oracle,case,record['world_pose']);passed=root_ok(expected)
        else:
            expected=internal_geometry(oracle,edge['proposed_relative_pose']);passed=internal_ok(expected)
        strict_equal(record['feasibility'],expected,checks,'independent '+stage+' filter',where)
    require(passed or len(records)==cap,'Premature edge-cap exhaustion')
    return passed


def audit_factorized(helper,oracle,case,old,anchor,outcome,raw_contacts,source,checks,tag):
    checks.exact(outcome['caps'],CAPS,'factorized caps',tag);checks.exact(outcome['order'],'root_first','factorized order',tag)
    strict_equal(outcome['source_feasibility'],source,checks,'source full predicates',tag)
    audit_frame(oracle,case,anchor,old,source,outcome['source_frame'],checks,tag+'/source')
    attempts=outcome['attempts'];require(len(attempts)<=1,'Unallocated joint retry')
    checks.exact(len(raw_contacts),len(attempts),'factorized contacts length',tag)
    if not source['internal_exclusion_contact']:
        require(outcome['status']=='source_outside_domain' and not attempts and outcome['candidate'] is None,'Ineligible source retried')
        return dict(root_draws=0,internal_draws=0,candidate=None,stage_status='source_outside_domain',assembled=0)
    require(len(attempts)==1,'Missing allocated attempt');a=attempts[0]
    checks.exact(a['index'],1,'joint index',tag)
    edges=[relative_pose(anchor,old[0]),relative_pose(old[0],old[1])]
    rp=audit_stage(helper,oracle,case,anchor,edges[0],a['root_draws'],'root',32,checks,tag)
    ip=False
    if not rp:require(not a['internal_draws'],'Internal stage ran after root exhaustion')
    else:ip=audit_stage(helper,oracle,case,anchor,edges[1],a['internal_draws'],'internal',32,checks,tag)
    candidate=None;assembled=int(rp and ip)
    if not assembled:
        expected='root_cap_exhausted' if not rp else 'internal_cap_exhausted'
        require(all(a[k] is None for k in ['proposed','final_feasibility','frame']),'Assembled after failed edge')
        require(raw_contacts[0] is None,'Contacts exist without endpoint')
    else:
        root=a['root_draws'][-1]['world_pose'];internal=a['internal_draws'][-1]['draw']['proposed_relative_pose']
        proposed=[root,compose_pose(root,internal)]
        require(a['proposed'] is not None,'Missing assembled endpoint')
        strict_equal(a['proposed'][0],root,checks,'selected root unchanged',tag)
        checks.pose(a['proposed'][1],proposed[1],'assembled child',tag)
        geometry=oracle.fingerprint([case['root'],case['child']],a['proposed'])
        full=feasibility(oracle,case,a['proposed'],geometry)
        strict_equal(a['final_feasibility'],full,checks,'independent final geometry',tag)
        strict_equal(raw_contacts[0],geometry['contacts'],checks,'final contacts',tag)
        audit_frame(oracle,case,anchor,a['proposed'],full,a['frame'],checks,tag+'/final')
        strict_equal(a['root_draws'][-1]['feasibility'],dict(spectator_core_collisions=full['spectator_core_collisions'][0],wall_valid=full['wall_valid'][0]),checks,'selected root filter',tag)
        strict_equal(a['internal_draws'][-1]['feasibility'],{k:full[k] for k in ['internal_core_overlap','internal_exclusion_contact']},checks,'selected internal filter',tag)
        expected='candidate' if is_feasible(full) else 'final_rejected'
        if expected=='candidate':
            require(outcome['candidate'] is not None,'Missing candidate density')
            c=outcome['candidate'];checks.pose(c['root'],a['proposed'][0],'candidate root',tag);checks.pose(c['child'],a['proposed'][1],'candidate child',tag)
            # Reuse independent complete-mixture audit through its ordinary raw
            # outcome interface. Conditioning constants cancel at fixed labels;
            # score F, not a selected branch or fitted conditional normalization.
            raw=dict(index=1,draw=dict(spectator=anchor,old_root=old[0],old_child=old[1],
                edges=[a['root_draws'][-1]['draw'],a['internal_draws'][-1]['draw']],candidate=c,null_reason=None),feasibility=full)
            candidate=audit_unique_trials(helper,oracle,case,old,anchor,[raw],[geometry['contacts']],.5,160.,checks,tag+'/density')[0]
    require(a['status']==expected,'Wrong attempt terminal status')
    require(outcome['status']==('candidate' if expected=='candidate' else 'cap_exhausted'),'Wrong outer terminal status')
    require((outcome['candidate'] is not None)==(expected=='candidate'),'Candidate after failed stage')
    return dict(root_draws=len(a['root_draws']),internal_draws=len(a['internal_draws']),candidate=candidate,stage_status=expected,assembled=assembled)


def summarize(rows):
    candidates=[r['candidate'] for r in rows if r['candidate'] is not None]
    raw=sum(r['raw_edge_draws'] for r in rows);cpu=sum(r['proposal_cpu_seconds'] for r in rows)
    return dict(outer_attempts=len(rows),raw_edge_draws=raw,root_draws=sum(r['root_draws'] for r in rows),
        internal_draws=sum(r['internal_draws'] for r in rows),assembled_endpoints=sum(r['assembled'] for r in rows),
        candidate_count=len(candidates),candidate_fraction=len(candidates)/len(rows) if rows else None,
        candidates_per_raw_edge=len(candidates)/raw if raw else None,candidates_per_proposal_cpu_second=len(candidates)/cpu if cpu else None,
        proposal_cpu_seconds=cpu,contact_diagnostic_cpu_seconds=sum(r['contact_diagnostic_cpu_seconds'] for r in rows),
        status_counts=dict(Counter(r['status'] for r in rows)),stage_status_counts=dict(Counter(r['stage_status'] for r in rows)),
        candidates_with_external_contacts=sum(c['external_contacts']>0 for c in candidates),
        both_intended_contacts=sum(c['both_intended_contacts'] for c in candidates),
        log_reverse_forward=quantiles([c['log_reverse_forward'] for c in candidates]))


def validate_contract(config, protocol):
    require(config['master_seed']==protocol['master_seed']==6100203101,'Changed master seed')
    require([config[k] for k in ['joint_cap','root_cap','internal_cap','factorized_joint_cap','attempts_per_context']]==[32,32,32,1,32] and config['factorized_order']=='root_first','Changed allocation')
    require(protocol['caps']==dict(whole_joint=32,factorized_root=32,factorized_internal=32,factorized_joint=1) and protocol['factorized_order']=='root_first','Changed protocol caps')
    require(protocol['allocation']==dict(atlases=3,contexts=8,outer_slots_per_context=32,methods=METHODS,total_method_trials=1536,independent_context_slots=768,maximum_raw_edge_draws_per_method_trial=64,maximum_executed_raw_edge_draws=98304,extension=False),'Changed protocol allocation')
    require(protocol['density_tolerance']==dict(absolute=2e-7,relative=2e-10),'Changed audit tolerance')
    require(protocol['physical_draws']==0 and protocol['state_updates']==0 and protocol['native_classification'] is False and protocol['native_label_filtering'] is False,'Wrong passive scope')
    seeds=[stream_seed(config['master_seed'],a,c,i,role) for a in range(3) for c in range(8) for i in range(32) for role in METHODS+['execution_order']]
    require(len(set(seeds))==2304,'RNG-domain seed collision')


def analyze(run):
    clock=time.process_time();run=Path(run)
    config,binding,terminal,protocol=[read(run/p) for p in ['config.json','binding.json','terminal.json','protocol.json']]
    require(config['schema']=='factorized-dimer-screen-v1' and binding['schema']=='factorized-dimer-probe-binding-v1','Unknown schema')
    for name,key in [('config.json','config_sha256'),('protocol.json','protocol_sha256'),('example.rs','example_source_sha256')]:require(sha(run/name)==binding[key],'Binding mismatch: '+name)
    require(sha(run/'attempts.jsonl')==terminal['attempts_sha256'],'Attempt ledger changed')
    require(terminal['summary']['complete'] is True,'Retained fatal run; no replacement attempts')
    require(config['density_law']=='map-factor-full-mixture-v1','Wrong density law')
    validate_contract(config,protocol)
    require(config['reference_config']['sha256']==REFERENCE_CONFIG_SHA256,'Changed reference')
    ref=bound(config['reference_config']);panel=bound(ref['panel'])
    require(ref['panel']['sha256']==PANEL_SHA256 and ref['cases']==panel['cases'] and len(ref['cases'])==8 and len(ref['atlases'])==3,'Changed panel')
    require(ref['uniform_probability']==.5 and ref['uniform_half_width']==160. and ref['depletant_radius']==1.4 and ref['activity']==.0275,'Changed model')
    source=bound(ref['source_config']);frame=bound(ref['source_frame']);archive=bound(ref['source_freeze_manifest']);shape=bound(ref['shape'])
    require(source['initial_poses']==frame['poses'] and archive['frame_sha256']==ref['source_frame']['sha256'] and archive['shape_sha256']==ref['shape']['sha256'],'Source mismatch')
    source_binding=bind_current_source(run,config,binding,protocol)
    require(protocol['factorized_source_sha256']==config['compiled_source_sha256']['src/factorized_dimer.rs'],'Factorized source mismatch')
    require(sha(__file__)==protocol['audit_files']['analyze_factorized_dimer_probe.py']['sha256'],'Unbound executing auditor')
    state=source['initial_poses'];oracle=DimerGeometry(shape,state,ref['depletant_radius'],ref['wall_radius'])
    checks=Checks();results=[];source_records=[]
    with (run/'attempts.jsonl').open() as stream:
        for ai,atlas in enumerate(ref['atlases']):
            helper=DimerDestinationDensity(bound(atlas['model']))
            for ci,case in enumerate(ref['cases']):
                old=[state[case['root']],state[case['child']]];anchor=state[case['anchor']]
                geometry=oracle.fingerprint([case['root'],case['child']],old);sourcef=feasibility(oracle,case,old,geometry)
                require(geometry['hard_valid'],'Hard-invalid source');source_records.append(dict(atlas=atlas['name'],case=case['name'],feasibility=sourcef))
                for attempt in range(32):
                    order=method_order(config['master_seed'],ai,ci,attempt)
                    for position,method in enumerate(order):
                        line=stream.readline();require(bool(line),'Missing allocated row');r=json.loads(line);tag=f'{ai}/{ci}/{attempt}/{method}'
                        require(r['status']=='completed','Fatal retained row')
                        for key,want in [('atlas',atlas['name']),('atlas_index',ai),('case',case),('case_index',ci),('attempt',attempt),('method',method),('execution_order',order),('execution_position',position),('seed',stream_seed(config['master_seed'],ai,ci,attempt,method)),('order_seed',stream_seed(config['master_seed'],ai,ci,attempt,'execution_order')),('old',old),('anchor_pose',anchor),('old_contacts',geometry['contacts'])]:checks.exact(r[key],want,'row '+key,tag)
                        require(len(r['rng_after_fingerprint'])==4 and all(type(v)is int and 0<=v<2**64 for v in r['rng_after_fingerprint']),'Invalid RNG fingerprint')
                        for k in ['proposal_cpu_seconds','contact_diagnostic_cpu_seconds']:require(math.isfinite(r[k]) and r[k]>=0,'Invalid CPU time')
                        out=r['outcome']
                        for key,want in [('members',[case['root'],case['child']]),('anchor_label',case['anchor']),('old',old)]:checks.exact(out[key],want,'outcome '+key,tag)
                        if method=='whole_joint':
                            verify_stopping(out,32,sourcef,checks,tag);trials=out['trials']
                            require(len(r['raw_contacts'])==len(trials),'Wrong contact ledger size')
                            audited=audit_unique_trials(helper,oracle,case,old,anchor,trials,r['raw_contacts'],.5,160.,checks,tag)
                            x=dict(root_draws=len(trials),internal_draws=len(trials),assembled=len(trials),stage_status=out['status'],candidate=audited[-1] if out['candidate'] is not None else None)
                        else:x=audit_factorized(helper,oracle,case,old,anchor,out,r['raw_contacts'],sourcef,checks,tag)
                        count=x['root_draws']+x['internal_draws'];checks.exact(r['raw_edge_draws'],count,'raw edge count',tag)
                        require(count<=64,'Exceeded raw edge budget')
                        result=dict(atlas=atlas['name'],case=case['name'],attempt=attempt,method=method,execution_position=position,status=out['status'],raw_edge_draws=count,proposal_cpu_seconds=r['proposal_cpu_seconds'],contact_diagnostic_cpu_seconds=r['contact_diagnostic_cpu_seconds'],**x)
                        if x['candidate']:result['contact_change']=fingerprints(geometry['contacts'],x['candidate']['contacts'])
                        results.append(result)
        require(not stream.readline(),'Unallocated extra row')
    total=summarize(results);require(len(results)==1536 and total['raw_edge_draws']<=98304,'Allocation violated')
    for k,want in [('outer_attempts',1536),('raw_edge_draws',total['raw_edge_draws']),('candidates',total['candidate_count']),('physical_draws',0),('state_updates',0)]:checks.exact(terminal['summary']['result'][k],want,'terminal '+k)
    comparisons=[dict(atlas=a['name'],method=m,summary=summarize([r for r in results if r['atlas']==a['name'] and r['method']==m]),contexts=[dict(case=c['name'],summary=summarize([r for r in results if r['atlas']==a['name'] and r['method']==m and r['case']==c['name']])) for c in ref['cases']]) for a in ref['atlases'] for m in METHODS]
    return dict(schema='factorized-dimer-independent-analysis-v1',complete=True,passed=not checks.failures,checks=checks.count,failures=checks.failures,
        maximum_absolute_errors=dict(checks.maximum_absolute_errors),source_binding={**source_binding,'src/factorized_dimer.rs':protocol['factorized_source_sha256']},
        input_hashes={p:sha(run/p) for p in ['config.json','binding.json','protocol.json','source-bundle.json','example.rs','attempts.jsonl','terminal.json']},
        summary=total,comparisons=comparisons,source_contexts=source_records,rows=results,analyzer_cpu_seconds=time.process_time()-clock,probe_cpu_seconds=terminal['cpu_seconds'],
        limitations=protocol['limitations'],density_tolerance=protocol['density_tolerance'],
        pairing=dict(independent_context_slots=768,method_streams_independent=True,raw_prefixes_shared=False,method_execution_order='frozen independent coin',statistical_intervals='No equilibrium uncertainty claim; fixed purposive contexts.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    with args.output.open('x') as out:
        try:report=analyze(args.run)
        except Exception as error:report=dict(schema='factorized-dimer-independent-analysis-v1',complete=False,passed=False,error=f'{type(error).__name__}: {error}')
        json.dump(serial(report),out,indent=2,allow_nan=False);out.write('\n')
    print(json.dumps({k:report.get(k) for k in ['complete','passed','checks','analyzer_cpu_seconds','error']}))
    raise SystemExit(0 if report.get('passed') else 1)
