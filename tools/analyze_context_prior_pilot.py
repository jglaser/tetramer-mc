#!/usr/bin/env python3
"""Completion-gated one-mobile patch observation and full retained-state audit.

The proposal never sees these patch labels. Geometry is queried only for the
saved states, under a separately frozen observer execution allocation.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

from analyze_contact_efficiency import ContactObserver, pose_arrays
from context_prior_metrics import fingerprint, summarize

ALLOCATION_SHA='bfac82ddad223056e5a1d8ead1af2b6363f19dd020e1675bf2cbae2aeb12830f'
METRIC_SHA='f816cf0a58d2d59f95b284d1bad35b6310805037d1acfee2506e9eeb336abe94'
ADDENDUM_SHA='45d7ea67f57aff7cce8c6f0705b17e7405046437104b326a22626363248ac821'
PATCH_SHA='a1dee6d26d86852afadd21de231271ac73617bdfff884b07f129700e5c25e2bc'
STEP_SHA='f92df5be10a4480f2c4ede809bca4232f48a5e592f585d0609140dcc28a28fed'


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def close(actual,expected,scale=None):
    require(type(actual) in (int,float) and math.isfinite(actual) and math.isfinite(expected),'Nonfinite arithmetic')
    tolerance=2e-8+2e-10*(1+abs(expected) if scale is None else scale)
    require(abs(actual-expected)<=tolerance,'Saved acceptance/proposal arithmetic differs')


def counts_zero():
    return dict(attempted=0,completed=0,accepted=0,nulls=0,wall_rejected=0,core_rejected=0,bath_rejected=0)


def verify_shared_source(bundle):
    entry=bundle['files']['src/context_docking.rs'];text=entry['text']
    require(entry['sha256']==STEP_SHA==hashlib.sha256(text.encode()).hexdigest(),'Unreviewed physical-step journal schema')
    start=text.index('pub fn physical_step(')
    require(text.index('trace.log_proposal_reverse_forward = Some(correction);',start)
        <text.index('phase = "endpoint_geometry";',start),'Proposal ratio was not recorded before hard rejection')


def audit_step(step, global_slot, anchor_index, log_prior):
    require(type(step['accepted']) is bool,'Acceptance must be Boolean')
    info=step['proposal'];branch=info['branch'];q=info['log_reverse_forward']
    # The held f92df5be shared step records this before either hard predicate.
    close(q,q);close(step['log_proposal_reverse_forward'],q)
    require(step['proposed_pose'] is not None,'Unexpected uncertified proposal null in completed run')
    pose_arrays([step['old_pose'],step['proposed_pose'],step['retained_pose']])
    if not global_slot:
        require(branch=='local','Local schedule changed');close(q,0.);close(info['log_reverse_forward'],0.)
    else:
        require(info['anchor_index']==anchor_index,'Changed fixed proposal anchor')
        if branch=='uniform':close(q,0.);close(info['log_reverse_forward'],0.)
        else:
            require(branch=='involution' and info['source_law']=='posterior' and info['identity'] is False,'Wrong frozen learned branch')
            trace=info['trace'];i,j=trace['source'],trace['target']
            require(type(i) is int and type(j) is int and 0<=i<len(log_prior) and 0<=j<len(log_prior),'Invalid virtual labels')
            require(info['step']['inverse_trace']['source']==j and info['step']['inverse_trace']['target']==i,'Wrong inverse labels')
            gx,gy=info['full_old_gaussian_log_density'],info['full_new_gaussian_log_density']
            source,target=info['selected_source_log_density'],info['selected_target_log_density']
            scale=1+abs(gx)+abs(gy)+abs(source)+abs(target)+abs(log_prior[i])+abs(log_prior[j])
            fs=log_prior[i]+source-gx;rs=log_prior[j]+target-gy
            labels=rs+log_prior[i]-fs-log_prior[j]
            close(info['source_log_probability'],fs,scale);close(info['inverse_source_log_probability'],rs,scale)
            close(info['label_log_reverse_forward'],labels,2*scale)
            expanded=info['step']['log_correction']+labels
            close(info['expanded_log_reverse_forward'],expanded,2*scale+abs(expanded))
            close(expanded,gx-gy,2*scale+abs(expanded));close(q,gx-gy,scale);close(info['log_reverse_forward'],q,scale)
    require(step['retained_pose']==(step['proposed_pose'] if step['accepted'] else step['old_pose']),'Rejected residence lost')
    status=step['status'];gate=step['gate']
    if status in ('wall_rejected','core_rejected'):
        require(not step['accepted'] and gate is None and all(step[k] is None for k in
            ('raw_log_acceptance','log_acceptance','acceptance_uniform','log_uniform')),'Hard rejection has a physical gate')
        require(step['wall_valid'] is (status!='wall_rejected'),'Wrong saved wall rejection')
        require(step['core_valid'] is (False if status=='core_rejected' else None),'Wrong saved core rejection')
    else:
        require(status in ('accepted','bath_rejected') and step['wall_valid'] is True and step['core_valid'] is True and isinstance(gate,dict),'Unknown completed physical disposition')
        close(step['log_proposal_reverse_forward'],q)
        for key in ('gained','lost','raw_points','retained_points','retained_cells','created_cells'):
            require(type(gate[key]) is int and gate[key]>=0,'Invalid bath counter')
        require(gate['gained']+gate['lost']==gate['retained_points']<=gate['raw_points'],'Cloud count mismatch')
        weight=math.log1p(1/64)*(gate['gained']-gate['lost'])
        close(gate['log_weight'],weight);raw=q+weight;scale=1+abs(q)+abs(weight)
        close(step['raw_log_acceptance'],raw,scale);close(step['log_acceptance'],min(0.,raw),scale)
        u=step['acceptance_uniform'];require(type(u) in (int,float) and 0<=u<1,'Invalid acceptance uniform')
        if u:close(step['log_uniform'],math.log(u))
        else:require(step['log_uniform'] is None,'Missing zero-uniform convention')
        take=(math.log(u) if u else -math.inf)<min(0.,raw)
        require(step['accepted'] is take and (status=='accepted') is take,'Wrong physical MH decision')
    return dict(branch=branch,status=status,logq=q,bath_log_weight=None if gate is None else gate['log_weight'],
        raw_log_acceptance=step['raw_log_acceptance'])


def replay(directory, summary, cfg, context, source, model, prior):
    """No geometry: validate every begun/decision/result and checkpoint/frame."""
    checkpoint=read(directory/'checkpoint.json')
    require(sha(directory/'checkpoint.json')==summary['checkpoint_sha256'],'Changed terminal checkpoint')
    require(summary['complete'] is True and summary['passed'] is True and summary['completed_cycles']==2304
        and summary['completed_attempts']==11520,'Incomplete fixed chain')
    require(cfg['warmup_cycles']==256 and cfg['local_attempts_per_cycle']==4 and cfg['depletant_radius']==1.5
        and cfg['reservoir_density']==.035 and cfg['poisson_lambda_ratio']==64.
        and cfg['translation_steps']==[.2] and cfg['rotation_steps_deg']==[1.]
        and cfg['rotation_probability']==.5 and cfg['uniform_probability']==.5,'Changed physical/proposal allocation')
    require(cfg['endpoint_gate']==dict(max_cells=255,max_depth=8,min_width=0.),'Changed envelope rule')
    require(summary['correlation']==0. and summary['identity']==cfg['identity'] and summary['method'] in ('local','posterior_involution'),'Changed method/job')
    labels=[b['label'] for b in context['bodies']]
    require(labels==[i for i in range(264) if i!=77] and context['excluded_moving_labels']==[77]
        and context['anchor_label']==source['anchor_label']==16,'Changed conditional label/context inventory')
    require(source['coordinate_frame']=='Saved spherical-center frame; no display offset, wrapping or pose transform.'
        and source['moving_label']==77 and source['boundary']=='spherical','Wrong coordinate frame')
    anchor=labels.index(16)
    base=model['base_model'];flags=model['reciprocal_components']
    require(len(base['weights'])==1024 and flags==[True]*1024,'Changed complete reciprocal atlas')
    total=0.
    for w in base['weights']:total+=w
    log_prior=[math.log(w/total/2) for w in base['weights'] for _ in range(2)]
    if prior is not None:
        require(summary['method']=='posterior_involution' and prior['moving_labels']==[77]
            and prior['anchor_label']==16 and len(prior['log_prior'])==2048,'Changed context prior')
        require(prior['fixed_context_sha256']==summary['bindings']['fixed_context']
            and prior['source_state_sha256']==summary['bindings']['source_state'],'Prior no longer matches fixed context')
        log_prior=prior['log_prior']
    require(all(math.isfinite(p) for p in log_prior),'Missing complete virtual prior')
    rows=[];cycles=[];counts=counts_zero();raw=retained=0;event_index=0;last_cpu=-1.;last_invocation=-1.
    with (directory/'events.jsonl').open() as stream:
        def event(kind):
            nonlocal event_index
            line=next(stream,None);require(line is not None,'Truncated chain journal')
            row=json.loads(line);require(row['kind']==kind and row['event_index']==event_index,'Wrong journal inventory/order')
            event_index+=1;return row
        invocation=event('invocation_begun')
        require(invocation['resume'] is None and invocation['certify'] is None and invocation['method']==summary['method'],'Pilot must use fresh unfiltered trajectories')
        prepared=event('prepared');fixed=event('fixed_context_audit')
        require(prepared['bindings']==summary['bindings'] and prepared['identity']==cfg['identity']
            and prepared['fixed_labels']==labels and prepared['moving_label']==77 and prepared['moving_state_index']==263
            and prepared['anchor_label']==16 and prepared['anchor_fixed_index']==anchor,'Prepared global/compact label mapping mismatch')
        require(fixed['audit']['physical_context_valid'] is True and not fixed['audit']['fixed_fixed_core_overlaps']
            and not fixed['audit']['fixed_wall_invalid_labels'],'Invalid fixed physical domain')
        initial=event('initial_state');pose=cfg['initial_pose'] if cfg['initial_pose'] is not None else source['pose']
        require(initial['pose']==pose and initial['counts']==counts and initial['completed_cycles']==initial['completed_attempts']==0
            and initial['resume_sha256'] is None and initial['identity']==cfg['identity'],'Changed initial state')
        previous_neighbors=initial['observation']['exclusion_contact_labels']
        for cycle in range(1,2305):
            for slot in range(5):
                index=(cycle-1)*5+slot;global_slot=slot==4 and summary['method']=='posterior_involution'
                begun=event('attempt_begun');decision=event('attempt_decision');complete=event('attempt_complete')
                for item in (begun,decision,complete):
                    require((item['cycle'],item['slot'],item['attempt_index'])==(cycle,slot,index),'Missing/reordered attempt')
                require(begun['old_pose']==pose and begun['global'] is global_slot and begun['identity']==cfg['identity'],'Wrong proposal starting state')
                step=decision['step'];require(step['old_pose']==pose,'Wrong decision source')
                factors=audit_step(step,global_slot,anchor,log_prior)
                for key in ('old_pose','proposed_pose','retained_pose','proposal','accepted','wall_valid','core_valid','gate',
                    'log_proposal_reverse_forward','raw_log_acceptance','log_acceptance','acceptance_uniform','log_uniform'):
                    require(complete[key]==step[key],'Decision/outcome arithmetic changed: '+key)
                pose=step['retained_pose'];counts['attempted']+=1;counts['completed']+=1
                counts['accepted']+=int(step['accepted'])
                if step['status']!='accepted':counts[step['status']]+=1
                if step['gate'] is not None:
                    raw+=step['gate']['raw_points'];retained+=step['gate']['retained_points']
                require(complete['counts']==counts and complete['global'] is global_slot and complete['production'] is (cycle>256)
                    and complete['identity']==cfg['identity'] and complete['exclusion_contact_labels_before']==previous_neighbors,'Lost state residence or counter')
                neighbors=complete['exclusion_contact_labels']
                require(neighbors==sorted(set(neighbors)) and set(neighbors)<=set(labels),'Invalid instantaneous neighbor labels')
                if not step['accepted']:require(neighbors==previous_neighbors,'Rejected state changed contact observation')
                cpu,inv=complete['sampler_cpu_seconds'],complete['invocation_cpu_seconds']
                require(math.isfinite(cpu) and cpu>=last_cpu and math.isfinite(inv) and inv>=last_invocation and inv>=cpu,'Invalid cumulative CPU')
                last_cpu,last_invocation=cpu,inv;previous_neighbors=neighbors
                rows.append(dict(attempt_index=index,cycle=cycle,slot=slot,production=cycle>256,pose=copy.deepcopy(pose),
                    accepted=step['accepted'],neighbor_labels=neighbors,sampler_cpu_seconds=cpu,invocation_cpu_seconds=inv,**factors))
            end=event('cycle_complete')
            require(end['cycle']==cycle and end['pose']==pose and end['counts']==counts and end['completed_attempts']==cycle*5
                and end['observation']['pose']==pose and end['observation']['exclusion_contact_labels']==previous_neighbors,'Cycle closure mismatch')
            cycles.append(end)
        require(next(stream,None) is None,'Unexpected physical journal tail')
    require(event_index==36868 and summary['counts']==counts and checkpoint['counts']==counts and
        checkpoint['pose']==summary['pose']==pose and summary['raw_points']==checkpoint['raw_points']==raw
        and summary['retained_points']==checkpoint['retained_points']==retained,'Terminal closure mismatch')
    require(summary['bindings']==checkpoint['bindings'] and checkpoint['method']==summary['method']
        and checkpoint['correlation']==0. and checkpoint['completed_cycles']==2304 and checkpoint['completed_attempts']==11520,'Checkpoint provenance changed')
    require(summary['sampler_cpu_seconds']>=last_cpu and summary['invocation_cpu_seconds']>=last_invocation,'Summary CPU precedes final state')
    frames=[json.loads(line) for line in (directory/'trajectory.jsonl').read_text().splitlines()]
    cadence=checkpoint['sample_every'];require(type(cadence) is int and cadence>0,'Invalid frame cadence')
    expected=[c for c in range(1,2305) if c%cadence==0 or c==2304]
    require([f['cycle'] for f in frames]==expected,'Accepted-only/missing trajectory frame')
    for f in frames:
        e=cycles[f['cycle']-1]
        require(f['pose']==e['pose'] and f['counts']==e['counts'] and f['exclusion_contact_labels']==e['observation']['exclusion_contact_labels'],'Trajectory differs from retained journal')
    return rows,initial,prepared,dict(events=event_index,attempts=len(rows),raw_points=raw,retained_points=retained)


class SingleMovingPatchObserver:
    def __init__(self,shape,patch_map,context,wall_radius,rd):
        self.context=context;self.moving=77;self.fixed={b['label']:b['pose'] for b in context['bodies']}
        self.labels=sorted(self.fixed);self.positions=np.asarray([self.fixed[k]['position'] for k in self.labels])
        bound=max(np.linalg.norm(a['center'])+a['radius'] for a in shape['atoms'])
        cfg=dict(depletant_radius=rd,boundary=dict(kind='spherical',radius=wall_radius),box_lengths=[2*(wall_radius+bound)]*3)
        self.observer=ContactObserver(shape,patch_map,cfg,native=None)
        self.cache={};self.calls=self.hits=0

    def classify(self,pose):
        key=canonical(pose)
        if key in self.cache:self.hits+=1;return self.cache[key]
        pos,rot=pose_arrays([pose]);obs=self.observer
        atoms=np.einsum('ij,aj->ai',rot[0],obs.atoms)+pos[0]
        require(np.min(obs.boundary['radius']-np.linalg.norm(atoms,axis=1)-obs.radii)>=-2e-8,'Saved mobile pose violates atomic wall')
        guard=512*np.finfo(float).eps*(1+np.linalg.norm(pos[0])+np.linalg.norm(self.positions,axis=1)+obs.bound+obs.rd)
        candidates=np.flatnonzero(np.linalg.norm(self.positions-pos[0],axis=1)<=2*(obs.bound+obs.rd)+guard)
        tokens=set()
        for index in candidates:
            label=self.labels[int(index)]
            # Sort GLOBAL labels before the two-pose observer, so patch sides
            # remain correct even when compact fixed indices differ (217->216).
            pair=sorted([(77,pose),(label,self.fixed[label])],key=lambda item:item[0])
            classified=obs.classify([item[1] for item in pair]);self.calls+=1
            require(not classified['instantaneous_native_keys'] and classified['native_cycle'] is None,'Native observer must be disabled')
            for a,b,pa,pb in classified['tokens']:
                require((a,b)==(0,1),'Unexpected pair-local contact label')
                tokens.add((pair[0][0],pair[1][0],pa,pb))
        neighbors=sorted({b if a==77 else a for a,b,_,_ in tokens})
        result=dict(patch_tokens=sorted(tokens),neighbor_labels=neighbors,fingerprint=fingerprint(tokens))
        self.cache[key]=result;return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('run-dir','patch-map','allocation','metric-spec','exchange-addendum','execution-receipt','out'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();started=time.process_time();clock=time.monotonic()
    require(sha(args.allocation)==ALLOCATION_SHA and sha(args.metric_spec)==METRIC_SHA and
        sha(args.exchange_addendum)==ADDENDUM_SHA and sha(args.patch_map)==PATCH_SHA,'Changed frozen measurement contract')
    require(not args.out.exists(),'Fresh observer output required')
    summary=read(args.run_dir/'summary.json');receipt=read(args.execution_receipt)
    require(receipt['success'] is True and receipt['child_drained'] is True and receipt['returncode']==0
        and Path(receipt['terminal']['path']).resolve()==(args.run_dir/'summary.json').resolve()
        and receipt['terminal']['sha256']==sha(args.run_dir/'summary.json'),'Missing completed/drained run authority')
    prov=args.run_dir/'provenance';inputs={}
    for key in ('config','model','shape','fixed_context','source_state'):
        path=prov/(key+'.json');require(sha(path)==summary['bindings'][key],'Changed frozen '+key);inputs[key]=read(path)
    require(sha(prov/'source-bundle.json')==summary['bindings']['source_bundle'],'Compiled source authority changed')
    verify_shared_source(read(prov/'source-bundle.json'))
    prior=None
    if 'prior' in summary['bindings']:
        require(sha(prov/'prior.json')==summary['bindings']['prior'],'Changed frozen prior');prior=read(prov/'prior.json')
    allocation=read(args.allocation)
    for key in ('model','shape'):
        require(summary['bindings'][key]==allocation['asset_pins'][key]['sha256'],'Wrong pilot '+key)
    for key,asset_key in [('fixed_context','fixed_context'),('source_state','source')]:
        require(summary['bindings'][key]==allocation['asset_pins'][asset_key]['sha256'],'Wrong pilot context/source')
    if prior is not None:require(summary['bindings']['prior']==allocation['asset_pins']['prior']['sha256'],'Wrong pilot prior')
    patch=read(args.patch_map)
    require(patch['shape_sha256']==summary['bindings']['shape'] and len(patch['atom_patch_ids'])==len(inputs['shape']['atoms']), 'Wrong complete atom patch map')
    rows,initial,prepared,journal_audit=replay(args.run_dir,summary,inputs['config'],inputs['fixed_context'],inputs['source_state'],inputs['model'],prior)
    args.out.mkdir()
    observer=SingleMovingPatchObserver(inputs['shape'],patch['atom_patch_ids'],inputs['fixed_context'],inputs['source_state']['spherical_wall_radius'],1.5)
    with (args.out/'observations.jsonl').open('x') as output:
        def emit(value):output.write(json.dumps(value,allow_nan=False)+'\n');output.flush()
        try:
            emit(dict(kind='initial_begun',pose=initial['pose']))
            initial_observation=observer.classify(initial['pose'])
            require(initial_observation['neighbor_labels']==initial['observation']['exclusion_contact_labels'],'Independent initial neighbor set mismatch')
            emit(dict(kind='initial_complete',**initial_observation))
            for row in rows:
                require(time.process_time()-started<300 and time.monotonic()-clock<600,'Observer resource cap reached')
                emit(dict(kind='observation_begun',attempt_index=row['attempt_index'],pose=row['pose']))
                found=observer.classify(row['pose'])
                require(found['neighbor_labels']==row['neighbor_labels'],'Independent global neighbor identities disagree')
                row.update(found)
                emit(dict(kind='observation_complete',**row))
            metrics=summarize(rows,summary['invocation_cpu_seconds'])
            report=dict(schema='context-prior-pilot-contact-analysis-v1',complete=True,passed=True,
                identity=summary['identity'],method=summary['method'],prior_active=prior is not None,
                input_sha256={str(p):sha(p) for p in (args.allocation,args.metric_spec,args.exchange_addendum,args.patch_map,args.execution_receipt,
                    args.run_dir/'summary.json',args.run_dir/'events.jsonl',args.run_dir/'checkpoint.json')},
                bindings=summary['bindings'],journal_audit=journal_audit,initial_observation=initial_observation,
                metrics=metrics,costs={k:summary[k] for k in ('preparation_cpu_seconds','sampler_cpu_seconds','observer_cpu_seconds',
                    'proposal_cpu_seconds','geometry_cpu_seconds','gate_cpu_seconds','invocation_cpu_seconds','invocation_wall_seconds')},
                offline_observer_cpu_seconds=time.process_time()-started,pair_classifications=observer.calls,
                exact_pose_cache_hits=observer.hits,unique_saved_poses=len(observer.cache),
                native_classifier_calls=0,new_poses=0,physical_draws=0,
                scope='Instantaneous all-state patch diagnostics and secondary persistent exchanges in one fixed historical500uM neighborhood; no finite-system106.8uM assembly or equilibrium-weight claim.')
            with (args.out/'report.json').open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
        except BaseException as error:
            emit(dict(kind='observer_failed',error=repr(error),complete=False));raise


if __name__=='__main__':main()
