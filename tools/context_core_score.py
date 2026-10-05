#!/usr/bin/env python3
"""Bounded actual-pose core/hinge scoring; never optimize or sample a pose."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from context_relaxed_atlas import OutsideContext, branches, pose, require
import context_relaxed_atlas as context_geometry


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def pose_key(center):
    return json.dumps(center,sort_keys=True,separators=(',',':'),allow_nan=False)


class ExactPointScorer:
    """Actual-pose squared hinge, with bounded materialized KD candidates.

    The gradient uses Cartesian translation and a left infinitesimal rotation
    about the moving body origin, in radians. At coincident atomic centers the
    hinge is nondifferentiable: return the declared zero-direction convention
    and an explicit coincidence count. No completeness claim survives a cap.
    """
    def __init__(self,context,*,margin=.002,max_candidates=65536,batch_size=64,attribute_bodies=False):
        require(math.isfinite(margin) and margin>0,'Positive finite margin required')
        require(type(max_candidates) is int and max_candidates>=0,'Invalid candidate cap')
        require(type(batch_size) is int and batch_size>0,'Invalid batch size')
        self.context=context;self.margin=margin;self.cap=max_candidates;self.batch=batch_size
        require(type(attribute_bodies) is bool,'Body attribution must be boolean')
        self.body_labels=None
        if attribute_bodies:
            labels=context.identity.get('outside_labels')
            require(type(labels) is list and len(labels)>0 and all(type(k) is int and k>=0 for k in labels),
                    'Attribution requires explicit outside body labels')
            require(labels==sorted(set(labels)),'Outside labels must match sorted unique contiguous body blocks')
            require(len(context.fixed)==len(labels)*len(context.atoms), 'Fixed atom blocks differ from body inventory')
            require(np.array_equal(context.fixed_radii,np.tile(context.radii,len(labels))),
                    'Fixed atom blocks must preserve the moving shape radius order')
            self.body_labels=list(labels)

    def score(self,center):
        started=time.process_time();ctx=self.context;r,t=pose(center)
        rotated=ctx.atoms@r.T;points=rotated+t
        guard=float(1024*np.finfo(float).eps*(1+np.max(np.linalg.norm(rotated,axis=1))+
                    np.linalg.norm(t)+ctx.maximum_fixed_center_norm))
        candidate_count=distance_tests=count_query_points=list_query_points=0
        exact_pairs=core=active=coincident=0;objective=0.;gradient=np.zeros(6);minimum=None
        query_cpu=filter_cpu=0.;complete=True;cap_batch=None;processed_atoms=0
        if self.body_labels is not None:
            body_count=len(self.body_labels)
            body_core=np.zeros(body_count,dtype=np.int64);body_active=np.zeros(body_count,dtype=np.int64)
            body_objective=np.zeros(body_count);body_minimum=np.full(body_count,np.inf)
        for first in range(0,len(points),self.batch):
            stop=min(first+self.batch,len(points));p=points[first:stop]
            radii=np.nextafter(ctx.radii[first:stop]+ctx.maximum_fixed_radius+self.margin+guard,np.inf)
            tick=time.process_time()
            counts=np.asarray(ctx.tree.query_ball_point(p,radii,workers=1,return_length=True),dtype=np.int64)
            count_query_points+=len(p);needed=int(counts.sum());query_cpu+=time.process_time()-tick
            require(counts.shape==(len(p),) and np.all(counts>=0),'Invalid KD count result')
            if needed>self.cap-candidate_count:
                complete=False;cap_batch=dict(first_atom=first,stop_atom=stop,
                    batch_candidates=needed,remaining_candidate_budget=self.cap-candidate_count);break
            tick=time.process_time()
            lists=ctx.tree.query_ball_point(p,radii,workers=1,return_sorted=True)
            list_query_points+=len(p);query_cpu+=time.process_time()-tick
            require(len(lists)==len(counts) and all(len(js)==int(count) for js,count in zip(lists,counts)),'KD count/list disagreement')
            candidate_count+=needed
            tick=time.process_time()
            if needed:
                mi=np.repeat(np.arange(first,stop),counts)
                fi=np.concatenate([np.asarray(js,dtype=np.int64) for js in lists])
                delta=points[mi]-ctx.fixed[fi];d2=np.einsum('ij,ij->i',delta,delta)
                distance_tests+=len(fi);radius_sum=ctx.radii[mi]+ctx.fixed_radii[fi]
                # The guard enlarges only the tree query; never the predicates.
                d=np.sqrt(d2);gap=d-radius_sum;within=d<=radius_sum+self.margin
                exact_pairs+=int(np.count_nonzero(within))
                if np.any(within):
                    smallest=float(np.min(gap[within]));minimum=smallest if minimum is None else min(minimum,smallest)
                core+=int(np.count_nonzero(d2<radius_sum*radius_sum))
                hinges=np.maximum(self.margin-gap,0.);mask=hinges>0
                active+=int(np.count_nonzero(mask));coincident+=int(np.count_nonzero(mask & (d==0)))
                unit=np.divide(delta[mask],d[mask,None],out=np.zeros_like(delta[mask]),where=d[mask,None]>0)
                force=-2*hinges[mask,None]*unit
                objective+=float(hinges@hinges)
                gradient[:3]+=np.sum(force,axis=0)
                gradient[3:]+=np.sum(np.cross(rotated[mi[mask]],force),axis=0)
                if self.body_labels is not None:
                    body=fi//len(ctx.atoms)
                    body_core+=np.bincount(body[d2<radius_sum*radius_sum],minlength=body_count)
                    body_active+=np.bincount(body[mask],minlength=body_count)
                    body_objective+=np.bincount(body,weights=hinges*hinges,minlength=body_count)
                    np.minimum.at(body_minimum,body[within],gap[within])
            processed_atoms=stop;filter_cpu+=time.process_time()-tick
        require(math.isfinite(objective) and np.isfinite(gradient).all(),'Nonfinite hinge result')
        fields=dict(core_overlap_pairs=core,active_hinge_pairs=active,within_individual_margin_pairs=exact_pairs,
                    objective_A2=objective,gradient_translation_A=gradient[:3].tolist(),
                    gradient_left_rotation_A2_per_rad=gradient[3:].tolist(),minimum_local_gap_A=minimum,
                    coincident_active_pairs=coincident)
        result=dict(status='complete' if complete else 'candidate_cap',complete_geometry=complete,
            **(fields if complete else {k:None for k in fields}),partial_values=None if complete else fields,
            minimum_gap_censored=complete and minimum is None,
            minimum_gap_censor_threshold_A=self.margin if complete and minimum is None else None,
            hard_clear=(core==0) if complete else None,margin_A=self.margin,query_guard_A=guard,
            candidates_materialized=candidate_count,pair_distance_tests=distance_tests,
            count_query_atom_points=count_query_points,list_query_atom_points=list_query_points,
            atoms_processed=processed_atoms,total_moving_atoms=len(points),cap_batch=cap_batch,
            cpu_seconds=time.process_time()-started,query_cpu_seconds=query_cpu,filter_cpu_seconds=filter_cpu)
        if self.body_labels is not None:
            require(int(body_core.sum())==core and int(body_active.sum())==active,'Body count partition differs')
            require(np.isfinite(body_objective).all(),'Nonfinite per-body objective')
            censored=[complete and math.isinf(float(v)) for v in body_minimum]
            attribution=dict(labels=self.body_labels,complete_geometry=complete,
                fixed_atom_block_size=len(ctx.atoms),core_overlap_pairs=body_core.tolist(),
                active_hinge_pairs=body_active.tolist(),objective_A2=body_objective.tolist(),
                minimum_local_gap_A=[None if math.isinf(float(v)) else float(v) for v in body_minimum],
                minimum_gap_censored=censored,
                minimum_gap_censor_threshold_A=[self.margin if c else None for c in censored])
            result.update(body_attribution=attribution if complete else None,
                          partial_body_attribution=None if complete else attribution)
            # Include attribution serialization in total score cost. The query
            # and filter timers retain their original meanings.
            result['cpu_seconds']=time.process_time()-started
        return result


def identities(source,exported):
    parents=branches(source);n=len(source['base_model']['weights']);b=exported['base_model']
    require(len(b['weights'])==n+len(parents),'Incomplete child table')
    records=[]
    for k,p in enumerate(parents):
        records.append(dict(identity=f'parent-{k:04d}',arm='parent',parent_virtual_label=k,
                            source_weight=p['weight'],center=p['center'],cache_of=None))
    for k,p in enumerate(parents):
        require(b['means'][n+k]==[0.]*6 and exported['reciprocal_components'][n+k] is False,'Child chart differs')
        center=b['anchors'][n+k]
        records.append(dict(identity=f'child-{k:04d}',arm='child',parent_virtual_label=k,
                            source_weight=p['weight'],center=center,
                            cache_of=f'parent-{k:04d}' if pose_key(center)==pose_key(p['center']) else None))
    return records


def evaluate_records(records,scorer,emit):
    """Every identity begins before work, and completes before the next begins."""
    results=[];cache={}
    for record in records:
        emit(dict(kind='pose_begun',**record))
        if record['cache_of'] is None:
            value=scorer.score(record['center']);cache[record['identity']]=copy.deepcopy(value)
        else:
            require(record['cache_of'] in cache,'Cache predecessor unavailable')
            value=copy.deepcopy(cache[record['cache_of']])
        result=dict(**record,score=value,performed_geometry=record['cache_of'] is None)
        emit(dict(kind='pose_complete',**result));results.append(result)
    return results


def summarize(records):
    result={}
    for arm in ('parent','child'):
        rows=[r for r in records if r['arm']==arm];weights=math.fsum(r['source_weight'] for r in rows)
        def tally(predicate):
            selected=[r for r in rows if predicate(r['score'])]
            return dict(count=len(selected),source_probability=math.fsum(r['source_weight'] for r in selected)/weights)
        result[arm]=dict(identities=len(rows),complete=tally(lambda s:s['complete_geometry']),
            capped=tally(lambda s:not s['complete_geometry']),hard_clear=tally(lambda s:s['hard_clear'] is True),
            clashing=tally(lambda s:s['hard_clear'] is False),
            hinge_positive=tally(lambda s:s['objective_A2'] is not None and s['objective_A2']>0))
    actual=[r['score'] for r in records if r['performed_geometry']]
    return dict(arms=result,actual_pose_queries=len(actual),cache_reuses=len(records)-len(actual),
        actual_cost={k:sum(s[k] for s in actual) for k in ('candidates_materialized','pair_distance_tests',
            'count_query_atom_points','list_query_atom_points','cpu_seconds','query_cpu_seconds','filter_cpu_seconds')})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-model','exported-model','shape','context','allocation','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();allocation=json.loads(args.allocation.read_bytes())
    require(allocation['schema']=='actual-center-core-score-allocation-v1','Wrong allocation')
    require(not args.out.exists(),'Fresh output required')
    for name in ('source_model','exported_model','shape','context'):
        require(sha(getattr(args,name))==allocation['inputs'][name],'Input changed: '+name)
    require(sha(__file__)==allocation['scorer_sha256'],'Scorer changed')
    require(sha(context_geometry.__file__)==allocation['geometry_sha256'],'Geometry helper changed')
    source=json.loads(args.source_model.read_bytes());exported=json.loads(args.exported_model.read_bytes())
    require(source['base_model']['shape_sha256']==allocation['inputs']['shape'] and
        exported['base_model']['shape_sha256']==allocation['inputs']['shape'],'Model/shape binding differs')
    records=identities(source,exported)
    require(len(records)==allocation['identities'] and
        sum(r['cache_of'] is None for r in records)==allocation['actual_pose_queries'],'Fixed identity/cache schedule differs')
    args.out.mkdir();start=time.monotonic();cpu=time.process_time()
    with (args.out/'attempts.jsonl').open('x') as journal:
        def emit(value):
            journal.write(json.dumps(value,allow_nan=False)+'\n');journal.flush()
        emit(dict(kind='context_begun'))
        context=OutsideContext.from_records(json.loads(args.shape.read_bytes()),json.loads(args.context.read_bytes()))
        require(len(context.atoms)==allocation['moving_atoms'] and len(context.fixed)==allocation['fixed_atoms'],'Atom allocation differs')
        emit(dict(kind='context_complete',moving_atoms=len(context.atoms),fixed_atoms=len(context.fixed),cpu_seconds=time.process_time()-cpu))
        scorer=ExactPointScorer(context,**allocation['settings'])
        results=evaluate_records(records,scorer,emit)
    for name in ('source_model','exported_model','shape','context'):
        require(sha(getattr(args,name))==allocation['inputs'][name],'Input changed during scoring: '+name)
    report=dict(schema='actual-center-core-score-v1',complete=True,passed=True,summary=summarize(results),
        records=results,cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-start,
        input_sha256=allocation['inputs'],allocation_sha256=sha(args.allocation),
        attempts_sha256=sha(args.out/'attempts.jsonl'),all_geometry_complete=all(r['score']['complete_geometry'] for r in results),
        scope='Actual-center core/hinge diagnostic only; capped geometry unresolved; no optimization, wall, native, depletion, proposal acceptance or equilibrium claim.')
    with (args.out/'report.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')

if __name__=='__main__':main()
