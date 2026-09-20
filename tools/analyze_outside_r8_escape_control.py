#!/usr/bin/env python3
"""Replay the matched contact-escape pilot and classify every retained state."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree
from prepare_smc_normalizer_atlas import ROOT,Density,arrays,read,relative_poses,sha,write
from analyze_trapped_docking_moves import classify,describe
from analyze_posterior_docking_pilot import one as standard_audit
from analyze_involution_docking_campaign import ChartAudit


def full_map_audit(root,job,result):
    """Complete the map/Jacobian/noise audit without repeating full-K densities."""
    began=time.process_time();cfg=read(job['config']);audit=ChartAudit(read(job['model']))
    rows=[json.loads(l) for l in (Path(job['directory'])/'moves.jsonl').open()]
    selected=[r for r in rows if (r.get('proposal') or {}).get('step') is not None]
    relative=relative_poses([r['old_pose'] for r in selected],cfg['fixed_poses'][0])
    for r,pose in zip(selected,relative):
        proposal=copy.deepcopy(r['proposal'])
        proposal.pop('full_old_gaussian_log_density',None);proposal.pop('full_new_gaussian_log_density',None)
        actual=audit.involution(pose,proposal,job['correlation'])
        audit.compare_pose(audit.absolute(actual,cfg['fixed_poses'][0]),r['proposed_pose'],'candidate_lab')
    elapsed=time.process_time()-began
    result['full_latent_map_audit']=dict(all_learned_maps_checked=len(selected),checks=dict(audit.checks),
        maximum_errors=dict(audit.max_errors),process_cpu_seconds=elapsed,
        scope='Every learned source/target latent, forward/inverse noise, selected component density, Haar Jacobian, and mapped endpoint independently reconstructed. Full-mixture densities are separately checked for every learned move by the primary replay.')
    result['complete_analysis_cpu_seconds']+=elapsed
    write(Path(root)/'assessment/runs'/job['id']/'analysis.json',result)
    return result


def only_maps(task):
    root,job=task
    result=read(Path(root)/'assessment/runs'/job['id']/'analysis.json')
    if 'full_latent_map_audit' in result:
        raise ValueError('Refusing to double-count an existing completed map audit')
    return full_map_audit(root,job,result)


class GeometryAudit:
    """Independent atom-pair geometry, cached only for identical input poses."""
    def __init__(self,cfg,shape):
        self.centers=np.asarray([a['center'] for a in shape['atoms']]);self.radii=np.asarray([a['radius'] for a in shape['atoms']])
        ft,_,fr=arrays(cfg['fixed_poses']);assert len(ft)==1
        self.fixed=self.centers@fr[0].T+ft[0];self.tree=cKDTree(self.fixed)
        self.lo=self.fixed.min(axis=0);self.hi=self.fixed.max(axis=0)
        self.rd=cfg['depletant_radius'];self.cache={};self.nearest_abs_gap=float('inf')

    def evaluate(self,pose):
        key=tuple(pose['position'])+tuple(pose['orientation'])
        if key in self.cache:return self.cache[key]
        t,_,r=arrays([pose]);moved=self.centers@r[0].T+t[0]
        reach=self.radii+self.radii.max()+2*self.rd
        # Atomwise AABB culling is conservative for both core and inflated tests.
        ids=np.flatnonzero(np.all((moved>=self.lo-reach[:,None])&(moved<=self.hi+reach[:,None]),axis=1))
        candidates=self.tree.query_ball_point(moved[ids],reach[ids],workers=1)
        bits=np.zeros(2*len(self.radii),dtype=bool)
        for i,js in zip(ids,candidates):
            if not js:continue
            js=np.asarray(js)
            gaps=np.linalg.norm(self.fixed[js]-moved[i],axis=1)-self.radii[i]-self.radii[js]
            self.nearest_abs_gap=min(self.nearest_abs_gap,float(np.min(np.abs(gaps))))
            if (gaps<0).any():
                self.cache[key]=(False,None);return self.cache[key]
            near=js[gaps<2*self.rd]
            if len(near):bits[i]=True;bits[len(self.radii)+near]=True
        self.cache[key]=(True,bits);return self.cache[key]


def analyze_job(task):
    root,job=task;root=Path(root);out=root/'assessment';start=time.process_time()
    # Every proposal-density, posterior-label, acceptance, and state transition
    # is checked by the independently validated arbitrary-K analyzer.
    result=standard_audit((job['campaign'],out,job,root/'provenance/old-region.json'))
    cfg=read(job['config']);shape=read(root/'provenance/shape.json')
    rows=[json.loads(l) for l in (Path(job['directory'])/'moves.jsonl').open()]
    frames=[json.loads(l) for l in (Path(job['directory'])/'trajectory.jsonl').open()]
    old=read(root/'provenance/old-region.json');new=read(root/'provenance/new-region.json')
    od,nd=Density(old['gaussian_chart']),Density(new['gaussian_chart'])
    q,ar,br,labels=classify([f['pose'] for f in frames],old['fixed_neighbor'],old['physical_metric'],od,nd)
    rq,ra,rb,rl=classify([frames[0]['pose']]+[r['retained_pose'] for r in rows],old['fixed_neighbor'],old['physical_metric'],od,nd)
    pq,pa,pb,pl=classify([r['proposed_pose'] for r in rows],old['fixed_neighbor'],old['physical_metric'],od,nd)
    a=(rq>=5)&(ra<=8);b=(rq>=5)&(ra>8)&(rb<=5)
    assert b[0]
    crossings=[];first_a=None;first_exit_b=None;returns_b=0
    for i,r in enumerate(rows):
        if b[i] and not b[i+1] and first_exit_b is None:first_exit_b=dict(cycle=r['cycle'],attempt=r['attempt'],branch=r['branch'])
        if a[i+1] and first_a is None:first_a=dict(cycle=r['cycle'],attempt=r['attempt'],branch=r['branch'])
        if not b[i] and b[i+1]:returns_b+=1
        if (a[i]!=a[i+1]) or (b[i]!=b[i+1]):
            assert r['accepted']
            crossings.append(dict(cycle=r['cycle'],attempt=r['attempt'],branch=r['branch'],old_label=rl[i],new_label=rl[i+1],
                source=(r.get('proposal',{}).get('trace') or {}).get('source'),target=(r.get('proposal',{}).get('trace') or {}).get('target'),
                proposal_correction=r['proposal']['log_reverse_forward'],gate_log=r['gate']['log_weight']))
    geometry=GeometryAudit(cfg,shape);geometry_start=time.process_time()
    valid,previous_bits=geometry.evaluate(frames[0]['pose']);assert valid
    checked=0;contact_changes=[];gate_max_error=0.
    for r in rows:
        if r['capture_valid']:
            valid,bits=geometry.evaluate(r['proposed_pose']);checked+=1
            assert valid==r['hard_valid'],(job['id'],r['cycle'],r['attempt'],valid,r['hard_valid'])
        if r['gate'] is not None:
            g=r['gate'];expected=np.log1p(1/cfg['poisson_lambda_ratio'])*(g['gained']-g['lost'])
            gate_max_error=max(gate_max_error,abs(g['log_weight']-expected))
        if r['accepted']:
            assert valid
            assert bool(bits.any())==r['depletion_contact']
            union=np.count_nonzero(bits|previous_bits);intersection=np.count_nonzero(bits&previous_bits)
            contact_changes.append(dict(cycle=r['cycle'],attempt=r['attempt'],branch=r['branch'],
                changed=bool(np.any(bits!=previous_bits)),jaccard_distance=1-intersection/union if union else 0.))
            previous_bits=bits
        assert bool(previous_bits.any())==r['depletion_contact']
    assert gate_max_error<1e-10
    proposal_groups={}
    for branch in ('local','involution','uniform'):
        mask=np.asarray([r['branch']==branch for r in rows])&b[:-1]&(pl=='old_R8')
        chosen=[r for r,yes in zip(rows,mask) if yes];gated=[r for r in chosen if r['gate'] is not None]
        proposal_groups[branch]=dict(proposed=len(chosen),hard_valid=sum(r['hard_valid'] for r in chosen),
            accepted=sum(r['accepted'] for r in chosen),gate_log=describe([r['gate']['log_weight'] for r in gated]),
            correction=describe([r['proposal']['log_reverse_forward'] for r in gated]))
    def fraction(x):return float(np.mean(x))
    fa=(q>=5)&(ar<=8);fb=(q>=5)&(ar>8)&(br<=5)
    result.update(model_variant=job['model_variant'],cohort_index=job['cohort_index'],
        region_escape=dict(first_B5_exit=first_exit_b,first_old_R8_entry=first_a,
            censored_at_cycle=2000,old_R8_entry_observed=first_a is not None,returns_to_B5=returns_b,crossings=crossings,
            B5_fraction_all_frames=fraction(fb),old_R8_fraction_all_frames=fraction(fa),
            other_fraction_all_frames=fraction(~(fa|fb)),B5_fraction_excluding_initial=fraction(fb[1:]),
            old_R8_fraction_excluding_initial=fraction(fa[1:]),first_half_B5_fraction=fraction(fb[1:1001]),
            last_half_B5_fraction=fraction(fb[1001:]),proposed_B5_to_old_R8=proposal_groups),
        contact_move_changes=dict(accepted_moves=len(contact_changes),changed=sum(c['changed'] for c in contact_changes),
            jaccard_distance=describe([c['jaccard_distance'] for c in contact_changes]),events=contact_changes,
            scope='Exact change of mobile/fixed atom-incidence descriptor on each accepted endpoint; no equilibrium-independence claim.'),
        full_geometry_audit=dict(captured_proposals_checked=checked,unique_pose_checks=len(geometry.cache),
            every_accepted_pose_checked=True,every_retained_contact_replayed=True,smallest_abs_tested_core_gap_A=geometry.nearest_abs_gap,
            maximum_gate_count_log_error=gate_max_error,geometry_process_cpu_seconds=time.process_time()-geometry_start,
            scope='Independent cKDTree candidate search with exact sphere-pair distances for every captured proposed endpoint and initial pose. Noncaptured endpoints have zero target weight and their hard checks are skipped, matching the runner.'),
        region_rows=[dict(cycle=i,old_radius=float(ar[i]),new_radius=float(br[i]),q=float(q[i]),label=str(labels[i])) for i in range(len(frames))])
    result['complete_analysis_cpu_seconds']=time.process_time()-start
    write(out/'runs'/job['id']/'analysis.json',result)
    print(json.dumps(dict(id=job['id'],complete=True,first_old_R8=first_a,residence_B5=result['region_escape']['B5_fraction_all_frames'],
        contact_changes=result['contact_move_changes']['changed'],checked=checked,analysis_cpu=result['complete_analysis_cpu_seconds'])),flush=True)
    return full_map_audit(root,job,result)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=ROOT/'runs/outside-r8-escape-2000');ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--complete-map-audit-only',action='store_true',help='Complete maps on existing validated analyses without rerunning geometry')
    args=ap.parse_args();manifest=read(args.root/'manifest.json');summary=read(args.root/'summary.json')
    assert summary['complete'] and len(manifest['jobs'])==8
    out=args.root/'assessment';out.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:results=list(pool.map(only_maps if args.complete_map_audit_only else analyze_job,[(str(args.root),j) for j in manifest['jobs']]))
    compact=[]
    for r in results:
        compact.append(dict(id=r['id'],model=r['model_variant'],correlation=r['correlation'],cohort=r['cohort_index'],
            first_old_R8=r['region_escape']['first_old_R8_entry'],first_B5_exit=r['region_escape']['first_B5_exit'],
            returns_to_B5=r['region_escape']['returns_to_B5'],B5_fraction=r['region_escape']['B5_fraction_all_frames'],
            old_R8_fraction=r['region_escape']['old_R8_fraction_all_frames'],contact_changes=r['contact_move_changes']['changed'],
            sampler_cpu_seconds=r['cpu_seconds'],analysis_cpu_seconds=r['complete_analysis_cpu_seconds']))
    report=dict(complete=True,rows=compact,all_jobs_passed=True,manifest_sha256=sha(args.root/'manifest.json'),
        analyzer_sha256=sha(__file__),total_sampler_cpu_seconds=summary['total_sampler_cpu_seconds'],
        total_analysis_cpu_seconds=sum(r['complete_analysis_cpu_seconds'] for r in results),
        total_learned_maps_checked=sum(r['full_latent_map_audit']['all_learned_maps_checked'] for r in results),
        total_captured_proposals_geometry_checked=sum(r['full_geometry_audit']['captured_proposals_checked'] for r in results),
        scope='Fixed-budget nonstationary matched escape diagnostic; no equilibrium ESS, independent event-rate, assembly, or generalization claim.')
    write(out/'analysis.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    image=np.asarray([[0 if row['label'] in ('new_B3','new_B5_shell') else 1 if row['label']=='old_R8' else 2 for row in r['region_rows']] for r in results])
    fig,ax=plt.subplots(figsize=(11,4.5));ax.imshow(image,aspect='auto',interpolation='nearest',extent=[0,2000,7.5,-.5],
        cmap=ListedColormap(['#de9754','#4699a7','#a0a0a0']),vmin=0,vmax=2)
    ax.set_yticks(range(8),[f"{r['model_variant']}, cohort{r['cohort_index']}, c={r['correlation']}" for r in results]);ax.set_xlabel('Cycle')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c,label=t) for c,t in [('#de9754','B5 contact extension'),('#4699a7','Old R8'),('#a0a0a0','Other')]],loc='upper center',bbox_to_anchor=(.5,1.18),ncol=3)
    fig.tight_layout();fig.savefig(out/'matched-escape-trajectories.png',dpi=170);fig.savefig(out/'matched-escape-trajectories.svg');plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
