#!/usr/bin/env python3
"""Independent fixed-cloud guidance audit; never draw proposals or physical baths.

Reconstruct every saved uniform coordinate, root-thinning decision, raw edge,
strict geometric branch, guidance count and complete mixture correction. Counts
use closed point membership; core/contact predicates remain strict. Geometric
or count disagreement is fatal, including frame disagreement. Seed/fingerprint
and common-prefix checks bind the saved RNG history; this is not a proof of the
floating-point predicates or of a pseudorandom generator. Candidate threshold
ambiguity diagnostics do not cover every rejected edge. No equilibrium, native
registry or physical acceptance conclusion follows from this passive screen.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='1'
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from analyze_capped_dimer_probe import (read,sha,bound,require,Checks,DimerDestinationDensity,DimerGeometry,
    bind_current_source,feasibility,is_feasible,audit_unique_trials,decode_edge,
    relative_pose,compose_pose,fingerprints,quantiles,serial,REFERENCE_CONFIG_SHA256,PANEL_SHA256)
from analyze_factorized_dimer_probe import (strict_equal,root_geometry,internal_geometry,root_ok,internal_ok,audit_frame,audit_stage,stream_seed as old_seed)

ALLOCATION_SHA='d768b70d07e5837218217d512a3266bb75b221916fb3ee74c7312aba15126b1f'
BASELINE_RECEIPT='738c91a6d4a588626ce62fe567238db1ebf46683a474ad3684738795d1a98a1a'
METHODS=['m1','m4'];CAPS=dict(root=32,internal=32,joint=1)
IDENTITY=dict(position=[0.,0.,0.],orientation=[1.,0.,0.,0.])


def seed(master,atlas,case,attempt,role):
    return int(hashlib.sha256(f'auxiliary-overlap-probe-v1/{master}/{atlas}/{case}/{attempt}/{role}'.encode()).hexdigest()[:16],16)


def method_order(atlas,case,attempt):
    return METHODS if seed(6100300101,atlas,case,attempt,'execution_order')&1==0 else METHODS[::-1]


def transform(points,pose):
    q=np.asarray(pose['orientation']);rotation=Rotation.from_quat(np.roll(q,-1))
    return rotation.apply(points)+np.asarray(pose['position'])


def point_hash(points):return hashlib.sha256(np.asarray(points,dtype='<f8').tobytes()).hexdigest()


class CountOracle:
    """KD candidate pruning followed by independently evaluated closed spheres."""
    def __init__(self,centers,radii):
        self.centers=np.asarray(centers,dtype=float);self.radii=np.asarray(radii,dtype=float)
        self.low=(self.centers-self.radii[:,None]).min(axis=0)
        self.high=(self.centers+self.radii[:,None]).max(axis=0)
        self.width=self.high-self.low

    def members(self,points,centers=None):
        points=np.asarray(points,dtype=float).reshape((-1,3));centers=self.centers if centers is None else np.asarray(centers)
        require(np.isfinite(points).all() and np.isfinite(centers).all(),'Nonfinite count inputs')
        if len(points)==0:return np.zeros(0,dtype=bool)
        # Only broad-phase pruning is enlarged; the final predicate is <= r^2.
        cutoff=float(self.radii.max())*(1+1e-12)+1e-12
        pairs=cKDTree(centers).sparse_distance_matrix(cKDTree(points),cutoff,output_type='ndarray')
        ii,jj=pairs['i'],pairs['j'];difference=centers[ii]-points[jj]
        keep=np.sum(difference*difference,axis=1)<=self.radii[ii]**2
        result=np.zeros(len(points),dtype=bool);result[jj[keep]]=True
        return result

    def relative(self,points,relative):return int(self.members(points,transform(self.centers,relative)).sum())
    def world(self,points,members):return int(self.members(transform(points,members[0]),transform(self.centers,members[1])).sum())


def audit_cloud(meta,raw,oracle,checks,tag,expected_count=16384):
    require(len(raw)==expected_count*3*8,'Truncated raw cloud')
    require(hashlib.sha256(raw).hexdigest()==meta['raw_uniform_sha256'],'Raw uniform hash differs')
    u=np.frombuffer(raw,dtype='<f8').reshape((expected_count,3))
    require(np.isfinite(u).all() and np.all((u>=0)&(u<1)),'Invalid raw uniform')
    strict_equal(meta['raw_count'],expected_count,checks,'raw cloud count',tag)
    for k,want in [('low',oracle.low),('high',oracle.high),('width',oracle.width)]:strict_equal(meta[k],want.tolist(),checks,'cloud '+k,tag)
    points=oracle.low+oracle.width*u
    require(point_hash(points)==meta['transformed_raw_sha256'],'Affine cloud transform differs')
    keep=np.flatnonzero(oracle.members(points));retained=points[keep]
    strict_equal(meta['kept_indices'],keep.tolist(),checks,'root-only retained indices',tag)
    strict_equal(meta['retained_count'],len(keep),checks,'retained count',tag)
    require(point_hash(retained)==meta['retained_points_sha256'],'Retained cloud hash differs')
    return retained


def count_frame(counter,points,raw_relative,members,frame,checks,tag):
    expected=dict(relative_count=counter.relative(points,raw_relative),
        recovered_count=counter.relative(points,frame['recovered_edges'][1]),
        world_count=counter.world(points,members),
        reconstructed_world_count=counter.world(points,frame['reconstructed_members']))
    strict_equal(frame.get('guidance'),expected,checks,'independent frame guidance',tag)
    require(len(set(expected.values()))==1,'Fatal guidance count frame disagreement')
    return expected['relative_count']


def audit_threshold(diagnostics,old_count,point_count,m,checks,tag):
    for k,want in [('m',m),('point_count',point_count),('old_count',old_count)]:strict_equal(diagnostics[k],want,checks,'guidance '+k,tag)
    draws=diagnostics['integer_draws']
    require(len(draws)==m and all(type(x)is int and 0<=x<=old_count for x in draws),'Invalid auxiliary integer trace')
    strict_equal(diagnostics['threshold'],max(draws),checks,'fixed maximum threshold',tag)
    return max(draws)


def audit_internal(helper,oracle,old_edge,records,counter,points,threshold,checks,tag):
    require(len(records)<=32,'Too many internal raw edges')
    passed=False;queries=0;threshold_failures=0;geometric_passes=0
    for j,record in enumerate(records):
        where=f'{tag}/internal{j+1}';require(not passed,'Hidden retry after guided success')
        strict_equal(record['index'],j+1,checks,'internal index',where)
        edge=record['draw'];checks.pose(edge['old_relative_pose'],old_edge,'old internal coordinate',where)
        decode_edge(helper,edge,.5,160.,checks,where)
        expected=internal_geometry(oracle,edge['proposed_relative_pose']);passed=internal_ok(expected)
        strict_equal(record['feasibility'],expected,checks,'independent internal filter',where)
        if passed:
            geometric_passes+=1;queries+=1
            k=counter.relative(points,edge['proposed_relative_pose'])
            strict_equal(record.get('guidance_count'),k,checks,'internal guidance count',where)
            passed=k>=threshold;threshold_failures+=int(not passed)
        else:require('guidance_count' not in record,'Measured guidance count on geometric rejection')
    require(passed or len(records)==32,'Premature internal cap exhaustion')
    return passed,queries,threshold_failures,geometric_passes


def audit_guided(helper,oracle,counter,points,case,old,anchor,outcome,raw_contacts,source,m,checks,tag):
    strict_equal(outcome['caps'],CAPS,checks,'caps',tag);strict_equal(outcome['order'],'root_first',checks,'order',tag)
    strict_equal(outcome['source_feasibility'],source,checks,'source predicates',tag)
    audit_frame(oracle,case,anchor,old,source,outcome['source_frame'],checks,tag+'/source')
    source_edge=outcome['source_frame']['recovered_edges'][1]
    old_count=count_frame(counter,points,source_edge,old,outcome['source_frame'],checks,tag+'/source')
    d=outcome['guidance'];threshold=audit_threshold(d,old_count,len(points),m,checks,tag)
    require(source['internal_exclusion_contact'],'Frozen panel includes ineligible source')
    attempts=outcome['attempts'];require(len(attempts)==1,'Wrong fixed joint attempt count')
    strict_equal(len(raw_contacts),1,checks,'contact record count',tag)
    a=attempts[0];strict_equal(a['index'],1,checks,'joint index',tag)
    edges=[relative_pose(anchor,old[0]),relative_pose(old[0],old[1])]
    rp=audit_stage(helper,oracle,case,anchor,edges[0],a['root_draws'],'root',32,checks,tag)
    ip=False;queries=4;tf=gp=0
    if not rp:require(not a['internal_draws'],'Internal stage after root exhaustion')
    else:
        ip,iq,tf,gp=audit_internal(helper,oracle,edges[1],a['internal_draws'],counter,points,threshold,checks,tag);queries+=iq
    candidate=None;assembled=int(rp and ip);new_count=None
    if not assembled:
        expected='root_cap_exhausted' if not rp else 'internal_cap_exhausted'
        require(all(a[k] is None for k in ['proposed','final_feasibility','frame']),'Assembled failed edge')
        require(raw_contacts[0] is None,'Contacts without endpoint')
    else:
        root=a['root_draws'][-1]['world_pose'];internal=a['internal_draws'][-1]['draw']['proposed_relative_pose']
        proposed=[root,compose_pose(root,internal)]
        require(a['proposed'] is not None,'Missing endpoint')
        strict_equal(a['proposed'][0],root,checks,'selected root unchanged',tag)
        checks.pose(a['proposed'][1],proposed[1],'assembled child',tag)
        geometry=oracle.fingerprint([case['root'],case['child']],a['proposed']);full=feasibility(oracle,case,a['proposed'],geometry)
        strict_equal(a['final_feasibility'],full,checks,'final geometry',tag)
        strict_equal(raw_contacts[0],geometry['contacts'],checks,'final contacts',tag)
        audit_frame(oracle,case,anchor,a['proposed'],full,a['frame'],checks,tag+'/final')
        endpoint_count=count_frame(counter,points,internal,a['proposed'],a['frame'],checks,tag+'/final');queries+=4
        strict_equal(endpoint_count,a['internal_draws'][-1]['guidance_count'],checks,'raw/final count',tag)
        require(endpoint_count>=threshold,'Accepted endpoint below threshold')
        strict_equal(a['root_draws'][-1]['feasibility'],dict(spectator_core_collisions=full['spectator_core_collisions'][0],wall_valid=full['wall_valid'][0]),checks,'selected root predicate',tag)
        strict_equal(a['internal_draws'][-1]['feasibility'],{k:full[k] for k in ['internal_core_overlap','internal_exclusion_contact']},checks,'selected internal predicate',tag)
        expected='candidate' if is_feasible(full) else 'final_rejected'
        if expected=='candidate':
            c=outcome['candidate'];require(c is not None,'Missing candidate density')
            checks.pose(c['root'],a['proposed'][0],'candidate root',tag);checks.pose(c['child'],a['proposed'][1],'candidate child',tag)
            raw=dict(index=1,draw=dict(spectator=anchor,old_root=old[0],old_child=old[1],edges=[a['root_draws'][-1]['draw'],a['internal_draws'][-1]['draw']],candidate=c,null_reason=None),feasibility=full)
            candidate=audit_unique_trials(helper,oracle,case,old,anchor,[raw],[geometry['contacts']],.5,160.,checks,tag+'/density')[0]
            new_count=endpoint_count;aux=m*(math.log1p(old_count)-math.log1p(new_count))
            checks.close(d['aux_log_correction'],aux,'auxiliary correction',tag)
            candidate.update(old_count=old_count,new_count=new_count,aux_log_correction=aux,
                complete_log_correction=candidate['log_reverse_forward']+aux,
                count_retention=new_count/old_count if old_count else None)
    require(a['status']==expected,'Wrong joint terminal status')
    require(outcome['status']==('candidate' if expected=='candidate' else 'cap_exhausted'),'Wrong outer status')
    require((outcome['candidate'] is not None)==(candidate is not None),'Candidate after failure')
    strict_equal(d['new_count'],new_count,checks,'destination count',tag)
    if candidate is None:require(d['aux_log_correction'] is None,'Auxiliary correction without candidate')
    strict_equal(d['count_queries'],queries,checks,'count queries',tag)
    strict_equal(d['point_tests'],queries*len(points),checks,'point tests',tag)
    return dict(root_draws=len(a['root_draws']),internal_draws=len(a['internal_draws']),candidate=candidate,stage_status=expected,
        assembled=assembled,old_count=old_count,threshold=threshold,threshold_failures=tf,internal_geometric_passes=gp,count_queries=queries,point_tests=queries*len(points))


def common_prefixes(baseline,rows,checks,tag):
    require(set(rows)==set(METHODS),'Missing matched guidance arm')
    b=baseline['outcome'];r1=rows['m1']['outcome'];r4=rows['m4']['outcome']
    d1,d4=r1['guidance'],r4['guidance']
    strict_equal(d1['old_count'],d4['old_count'],checks,'paired old count',tag)
    strict_equal(d1['integer_draws'],d4['integer_draws'][:1],checks,'paired threshold RNG prefix',tag)
    require(d4['threshold']>=d1['threshold'],'Nonnested thresholds')
    outcomes=[b,r1,r4]
    for out in outcomes:require(len(out['attempts'])==1,'Missing matched attempt')
    for out in outcomes[1:]:strict_equal(out['attempts'][0]['root_draws'],b['attempts'][0]['root_draws'],checks,'cached root prefix',tag)
    records=[out['attempts'][0]['internal_draws'] for out in outcomes]
    require(len(records[0])<=len(records[1])<=len(records[2]),'Guided caps violate nested first-success ordering')
    for short,long in zip(records,records[1:]):
        project=lambda rs:[{k:v for k,v in x.items() if k!='guidance_count'} for x in rs]
        strict_equal(project(short),project(long[:len(short)]),checks,'matched internal raw prefix',tag)
    allrows=[baseline,rows['m1'],rows['m4']]
    for j in range(2):
        if len(records[j])==len(records[j+1]):strict_equal(allrows[j]['rng_after_fingerprint'],allrows[j+1]['rng_after_fingerprint'],checks,'same-stop proposal RNG',tag)


def validate_contract(config,protocol):
    require(config['scientific_allocation']==protocol['scientific_allocation'] and config['scientific_allocation']['sha256']==ALLOCATION_SHA,'Allocation receipt changed')
    bound(config['scientific_allocation'])
    require(config['proposal_master_seed']==protocol['proposal_master_seed']==6100203101 and config['auxiliary_master_seed']==protocol['auxiliary_master_seed']==6100300101,'Master seed changed')
    require([config[k] for k in ['root_cap','internal_cap','factorized_joint_cap','attempts_per_context','cloud_raw_count']]==[32,32,1,32,16384] and config['factorized_order']=='root_first','Allocation changed')
    require(protocol['caps']==CAPS and protocol['order']=='root_first' and protocol['m_values']==[1,4],'Guide/caps changed')
    require(protocol['allocation']==dict(atlases=3,contexts=8,slots_per_context=32,guided_methods=METHODS,new_outer_attempts=1536,reused_baseline_attempts=768,clouds=768,raw_points_per_cloud=16384,raw_cloud_points=12582912,raw_cloud_bytes=301989888,maximum_raw_edge_draws=98304,extension=False),'Unallocated trial/cloud extension')
    require(protocol['density_tolerance']==dict(absolute=2e-7,relative=2e-10),'Audit tolerance changed')
    require(protocol['physical_bath_draws']==0 and protocol['state_updates']==0 and protocol['native_classification'] is False,'Not a passive allocation')
    require(protocol['cloud_law']['is_poisson'] is False and protocol['cloud_law']['raw_count']==16384,'Cloud law changed')
    require(config['baseline']==protocol['baseline_cache'],'Baseline cache changed')
    allseeds=[seed(6100300101,a,c,j,role) for a in range(3) for c in range(8) for j in range(32) for role in ['cloud','threshold','execution_order']]
    allseeds += [old_seed(6100203101,a,c,j,'factorized') for a in range(3) for c in range(8) for j in range(32)]
    require(len(set(allseeds))==3072,'RNG seed collision')


def baseline_cache(protocol):
    sources=protocol['baseline_sources'];require(sources['completed-review.json']['sha256']==BASELINE_RECEIPT,'Unreviewed baseline')
    for r in sources.values():require(sha(r['path'])==r['sha256'],'Changed historical baseline file')
    receipt=read(sources['completed-review.json']['path']);require(receipt['complete'] and receipt['passed'] and receipt['audit_exit_code']==0,'Historical baseline audit failed')
    analysis=read(sources['analysis.json']['path']);require(analysis['complete'] and analysis['passed'] and not analysis['failures'],'Historical analysis failed')
    require(sources['analysis.json']['sha256']=='3facf648519de1e32433c0d1514f66b47716bc65400c804b87414bba22e9a450','Historical analysis binding')
    for name,r in sources.items():
        if name!='completed-review.json':require(receipt['output_hashes'][name]==r['sha256'],'Receipt does not authenticate '+name)
    cache=protocol['baseline_cache'];require(sha(cache['path'])==cache['sha256'],'Cache bytes changed')
    original=[line for line in Path(sources['execution/attempts.jsonl']['path']).read_bytes().splitlines(keepends=True) if json.loads(line)['method']=='factorized']
    require(Path(cache['path']).read_bytes()==b''.join(original),'Cache is not exact baseline subset')
    rows=[json.loads(line) for line in original]
    require([(r['atlas_index'],r['case_index'],r['attempt']) for r in rows]==[(a,c,j) for a in range(3) for c in range(8) for j in range(32)],'Historical slot mismatch')
    bykey={(r['atlas'],r['case'],r['attempt']):r for r in analysis['rows'] if r['method']=='factorized'}
    return rows,bykey


def summarize(rows):
    candidates=[r['candidate'] for r in rows if r['candidate'] is not None];cpu=sum(r['standalone_proposal_cpu_seconds'] for r in rows)
    return dict(outer_attempts=len(rows),candidate_count=len(candidates),candidate_fraction=len(candidates)/len(rows) if rows else None,
        raw_edge_draws=sum(r['raw_edge_draws'] for r in rows),root_draws=sum(r['root_draws'] for r in rows),internal_draws=sum(r['internal_draws'] for r in rows),
        assembled_endpoints=sum(r['assembled'] for r in rows),threshold_failures=sum(r['threshold_failures'] for r in rows),
        internal_geometric_passes=sum(r['internal_geometric_passes'] for r in rows),count_queries=sum(r['count_queries'] for r in rows),point_tests=sum(r['point_tests'] for r in rows),
        proposal_cpu_seconds=sum(r['proposal_cpu_seconds'] for r in rows),cloud_construction_cpu_seconds=sum(r['cloud_construction_cpu_seconds'] for r in rows),
        guidance_setup_cpu_seconds=sum(r['guidance_setup_cpu_seconds'] for r in rows),standalone_proposal_cpu_seconds=cpu,
        contact_diagnostic_cpu_seconds=sum(r['contact_diagnostic_cpu_seconds'] for r in rows),candidates_per_standalone_cpu_second=len(candidates)/cpu if cpu else None,
        status_counts=dict(Counter(r['status'] for r in rows)),stage_status_counts=dict(Counter(r['stage_status'] for r in rows)),
        candidates_with_external_contacts=sum(c['external_contacts']>0 for c in candidates),both_intended_contacts=sum(c['both_intended_contacts'] for c in candidates),
        old_count=quantiles([r['old_count'] for r in rows]),threshold=quantiles([r['threshold'] for r in rows]),
        new_count=quantiles([c['new_count'] for c in candidates]),count_retention=quantiles([c['count_retention'] for c in candidates if c['count_retention'] is not None]),
        log_reverse_forward=quantiles([c['log_reverse_forward'] for c in candidates]),aux_log_correction=quantiles([c['aux_log_correction'] for c in candidates]),complete_log_correction=quantiles([c['complete_log_correction'] for c in candidates]))


def audit_rng(record,key):
    values=record[key];require(len(values)==4 and all(type(v)is int and 0<=v<2**64 for v in values),'Invalid RNG fingerprint')


def analyze(run):
    clock=time.process_time();run=Path(run);config,binding,terminal,protocol=[read(run/p) for p in ['config.json','binding.json','terminal.json','protocol.json']]
    require(config['schema']=='auxiliary-overlap-screen-v1' and binding['schema']=='auxiliary-overlap-probe-binding-v1','Unknown schema')
    for name,key in [('config.json','config_sha256'),('protocol.json','protocol_sha256'),('example.rs','example_source_sha256')]:require(sha(run/name)==binding[key],'Binding mismatch '+name)
    for name,key in [('attempts.jsonl','attempts_sha256'),('clouds.jsonl','clouds_sha256'),('cloud-uniforms.bin','cloud_uniforms_sha256')]:require(sha(run/name)==terminal[key],'Changed output '+name)
    require(terminal['summary']['complete'] is True,'Retained fatal run; no replacement allocation')
    require(config['density_law']=='map-factor-full-mixture-v1','Wrong density law');validate_contract(config,protocol)
    require(config['reference_config']['sha256']==REFERENCE_CONFIG_SHA256,'Changed reference');ref=bound(config['reference_config']);panel=bound(ref['panel'])
    require(ref['panel']['sha256']==PANEL_SHA256 and ref['cases']==panel['cases'] and len(ref['cases'])==8 and len(ref['atlases'])==3,'Changed panel')
    require(ref['uniform_probability']==.5 and ref['uniform_half_width']==160. and ref['depletant_radius']==1.4 and ref['activity']==.0275,'Changed model')
    source=bound(ref['source_config']);frame=bound(ref['source_frame']);archive=bound(ref['source_freeze_manifest']);shape=bound(ref['shape'])
    require(source['initial_poses']==frame['poses'] and archive['frame_sha256']==ref['source_frame']['sha256'] and archive['shape_sha256']==ref['shape']['sha256'],'Source mismatch')
    source_binding=bind_current_source(run,config,binding,protocol)
    for name,digest in protocol['guidance_source_sha256'].items():require(config['compiled_source_sha256'][name]==digest,'Changed guide implementation');source_binding[name]=digest
    require(set(protocol['guidance_source_sha256'])=={'src/factorized_dimer.rs','src/auxiliary_overlap_threshold.rs'},'Incomplete guidance source binding')
    require(sha(__file__)==protocol['audit_files']['analyze_auxiliary_overlap_probe.py']['sha256'],'Unbound executing auditor')
    state=source['initial_poses'];oracle=DimerGeometry(shape,state,ref['depletant_radius'],ref['wall_radius']);counter=CountOracle(oracle.centers,oracle.radii+oracle.rd)
    for key,want in [('low',counter.low.tolist()),('high',counter.high.tolist())]:require(protocol['cloud_law'][key]==want,'Cloud bounds differ')
    volume=float(np.prod(counter.width));require(protocol['cloud_law']['volume_A3']==volume and protocol['cloud_law']['effective_raw_intensity_Aminus3']==16384/volume,'Cloud intensity differs')
    baseline,baseresults=baseline_cache(protocol);checks=Checks();results=[];cached=[];retained_total=0;source_records=[]
    with (run/'attempts.jsonl').open() as stream,(run/'clouds.jsonl').open() as cloudstream,(run/'cloud-uniforms.bin').open('rb') as blob:
        for ai,atlas in enumerate(ref['atlases']):
            helper=DimerDestinationDensity(bound(atlas['model']))
            for ci,case in enumerate(ref['cases']):
                old=[state[case['root']],state[case['child']]];anchor=state[case['anchor']]
                geometry=oracle.fingerprint([case['root'],case['child']],old);sourcef=feasibility(oracle,case,old,geometry)
                require(geometry['hard_valid'] and sourcef['internal_exclusion_contact'],'Invalid frozen source');source_records.append(dict(atlas=atlas['name'],case=case['name'],feasibility=sourcef))
                for attempt in range(32):
                    index=(ai*8+ci)*32+attempt;tag=f'{ai}/{ci}/{attempt}';line=cloudstream.readline();require(bool(line),'Missing cloud');meta=json.loads(line)
                    for key,want in [('cloud_id',index),('atlas_index',ai),('case_index',ci),('attempt',attempt),('seed',seed(6100300101,ai,ci,attempt,'cloud')),('threshold_seed',seed(6100300101,ai,ci,attempt,'threshold')),('raw_byte_offset',index*16384*24),('raw_byte_length',16384*24)]:strict_equal(meta[key],want,checks,'cloud '+key,tag)
                    audit_rng(meta,'rng_after_fingerprint');points=audit_cloud(meta,blob.read(16384*24),counter,checks,tag);retained_total+=len(points)
                    require(math.isfinite(meta['cloud_construction_cpu_seconds']) and meta['cloud_construction_cpu_seconds']>=0,'Invalid cloud timing')
                    order=method_order(ai,ci,attempt);pair={}
                    for position,method in enumerate(order):
                        line=stream.readline();require(bool(line),'Missing outer row');r=json.loads(line);where=tag+'/'+method;pair[method]=r
                        require(r['status']=='completed','Fatal retained outer')
                        for key,want in [('atlas',atlas['name']),('atlas_index',ai),('case',case),('case_index',ci),('attempt',attempt),('method',method),('execution_order',order),('execution_position',position),('seed',old_seed(6100203101,ai,ci,attempt,'factorized')),('order_seed',seed(6100300101,ai,ci,attempt,'execution_order')),('old',old),('anchor_pose',anchor),('old_contacts',geometry['contacts']),('cloud_id',index),('cloud_seed',meta['seed']),('auxiliary_seed',meta['threshold_seed']),('m',1 if method=='m1' else 4),('cloud_construction_cpu_seconds',meta['cloud_construction_cpu_seconds'])]:strict_equal(r[key],want,checks,'row '+key,where)
                        for key in ['rng_after_fingerprint','auxiliary_rng_after_fingerprint']:audit_rng(r,key)
                        for key in ['proposal_cpu_seconds','cloud_construction_cpu_seconds','guidance_setup_cpu_seconds','standalone_proposal_cpu_seconds','contact_diagnostic_cpu_seconds']:require(math.isfinite(r[key]) and r[key]>=0,'Invalid CPU timing')
                        checks.close(r['standalone_proposal_cpu_seconds'],sum(r[k] for k in ['proposal_cpu_seconds','cloud_construction_cpu_seconds','guidance_setup_cpu_seconds']),'complete charged CPU',where)
                        out=r['outcome']
                        for key,want in [('members',[case['root'],case['child']]),('anchor_label',case['anchor']),('old',old)]:strict_equal(out[key],want,checks,'outcome '+key,where)
                        x=audit_guided(helper,oracle,counter,points,case,old,anchor,out,r['raw_contacts'],sourcef,r['m'],checks,where)
                        count=x['root_draws']+x['internal_draws'];strict_equal(r['raw_edge_draws'],count,checks,'raw edge count',where);require(count<=64,'Edge budget exceeded')
                        if x['candidate'] is None:require(r['complete_log_correction'] is None,'Combined correction without candidate')
                        else:checks.close(r['complete_log_correction'],x['candidate']['complete_log_correction'],'checked complete correction',where)
                        result=dict(atlas=atlas['name'],case=case['name'],attempt=attempt,method=method,execution_position=position,status=out['status'],raw_edge_draws=count,
                            **{k:r[k] for k in ['proposal_cpu_seconds','cloud_construction_cpu_seconds','guidance_setup_cpu_seconds','standalone_proposal_cpu_seconds','contact_diagnostic_cpu_seconds']},**x)
                        if x['candidate']:result['contact_change']=fingerprints(geometry['contacts'],x['candidate']['contacts'])
                        results.append(result)
                    common_prefixes(baseline[index],pair,checks,tag)
                    # Reused baseline density/geometry already has its own complete
                    # audit; only its count against this NEW shared cloud is added.
                    b=baseresults[(atlas['name'],case['name'],attempt)];copyrow=json.loads(json.dumps(b));kold=pair['m1']['outcome']['guidance']['old_count']
                    copyrow.update(method='baseline',old_count=kold,threshold=0,threshold_failures=0,internal_geometric_passes=0,count_queries=0,point_tests=0,
                        cloud_construction_cpu_seconds=0.,guidance_setup_cpu_seconds=0.,standalone_proposal_cpu_seconds=b['proposal_cpu_seconds'])
                    if copyrow['candidate'] is not None:
                        c=baseline[index]['outcome']['candidate'];new=[c['root'],c['child']];knew=counter.world(points,new)
                        copyrow['candidate'].update(old_count=kold,new_count=knew,count_retention=knew/kold if kold else None,aux_log_correction=0.,complete_log_correction=copyrow['candidate']['log_reverse_forward'])
                    cached.append(copyrow)
        require(not stream.readline() and not cloudstream.readline() and not blob.read(1),'Unallocated extra row/cloud/point')
    total=summarize(results);require(len(results)==1536 and len(cached)==768 and total['raw_edge_draws']<=98304,'Allocation violated')
    for key,want in [('outer_attempts',1536),('raw_edge_draws',total['raw_edge_draws']),('candidates',total['candidate_count']),('clouds',768),('raw_cloud_points',12582912),('retained_cloud_points',retained_total),('physical_bath_draws',0),('state_updates',0)]:strict_equal(terminal['summary']['result'][key],want,checks,'terminal '+key,'terminal')
    allrows=cached+results
    comparisons=[dict(atlas=a['name'],method=m,summary=summarize([r for r in allrows if r['atlas']==a['name'] and r['method']==m]),contexts=[dict(case=c['name'],summary=summarize([r for r in allrows if r['atlas']==a['name'] and r['method']==m and r['case']==c['name']])) for c in ref['cases']]) for a in ref['atlases'] for m in ['baseline']+METHODS]
    return dict(schema='auxiliary-overlap-independent-analysis-v1',complete=True,passed=not checks.failures,checks=checks.count,failures=checks.failures,maximum_absolute_errors=dict(checks.maximum_absolute_errors),source_binding=source_binding,
        input_hashes={p:sha(run/p) for p in ['config.json','binding.json','protocol.json','source-bundle.json','example.rs','attempts.jsonl','clouds.jsonl','cloud-uniforms.bin','terminal.json']},
        scientific_allocation_sha256=ALLOCATION_SHA,baseline_review_sha256=BASELINE_RECEIPT,summary=total,reused_baseline_summary=summarize(cached),comparisons=comparisons,source_contexts=source_records,rows=results,reused_baseline_rows=cached,
        analyzer_cpu_seconds=time.process_time()-clock,probe_cpu_seconds=terminal['cpu_seconds'],limitations=protocol['limitations'],density_tolerance=protocol['density_tolerance'],
        pairing=dict(independent_context_slots=768,new_guided_outers=1536,cached_baseline_outers=768,raw_prefixes_shared=True,threshold_prefix_shared=True,method_order='frozen independent coin',statistical_claim='dependent passive comparisons, no ESS or assembly inference'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    try:result=analyze(args.run)
    except Exception as error:result=dict(schema='auxiliary-overlap-independent-analysis-v1',complete=False,passed=False,fatal_error=f'{type(error).__name__}: {error}')
    with args.output.open('x') as out:json.dump(serial(result),out,indent=2,allow_nan=False);out.write('\n')
    print(json.dumps({k:result.get(k) for k in ['complete','passed','checks','fatal_error','summary','maximum_absolute_errors','analyzer_cpu_seconds']},indent=2))
    raise SystemExit(0 if result.get('passed') else 1)
