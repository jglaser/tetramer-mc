#!/usr/bin/env python3
"""Independent registered/native and nonspecific assembly audit of free starts.

Native templates are used only here, after simulation. The production model's
prior information is recorded separately and must not be inferred from labels.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def one(task):
    reference,campaign,out,job=task;reference,campaign,out=map(Path,(reference,campaign,out))
    sys.path.insert(0,str(reference/'scripts'))
    import numpy as np
    from scipy.spatial.transform import Rotation
    from audit_tetramer_assembly import AtomicAssembly,pose_arrays,rotations
    from tetramer_order import TetramerOrder,graph_summary
    from analyze_precursor_exchange import prepare_templates
    directory=Path(job['directory']);cfg=read(directory/'config.json');summary=read(directory/'summary.json')
    manifest=read(directory/'manifest.json');campaign_manifest=read(campaign/'manifest.json')
    assert summary['complete'] and summary['completed_sweeps']==campaign_manifest['sweeps']
    assert manifest['executable_sha256']==campaign_manifest['binary_sha256']
    expected_model=job.get('model_sha256') or campaign_manifest['model_sha256']
    assert manifest['model_sha256']==expected_model==sha(directory/'provenance/frozen-relative-model.json')
    assert manifest['shape_sha256']==sha(directory/'provenance/shape.json')
    assert manifest['source_bundle_sha256']==sha(directory/'provenance/source-bundle.json')
    assert manifest['config_sha256']==sha(job['config'])==job['config_sha256']
    shape=read(directory/'provenance/shape.json');atoms=np.asarray([a['center'] for a in shape['atoms']]);radii=np.asarray([a['radius'] for a in shape['atoms']])
    bound=float(np.max(np.linalg.norm(atoms,axis=1)+radii));radius=cfg['boundary']['radius'];rd=cfg['depletant_radius']
    analysis_cfg=copy.deepcopy(cfg);analysis_cfg['shape']=str(directory/'provenance/shape.json')
    analysis_cfg['rigid_members']=shape['rigid_members'];analysis_cfg['box_lengths']=[8*(radius+bound)]*3
    templates=prepare_templates(cfg['monomer_shape'],reference/'results/c1c3-scaffold/motifs.json',reference/'results/native-neighbor-classes/classification.json')
    atomic=AtomicAssembly(analysis_cfg,directory);order=TetramerOrder(analysis_cfg,directory,templates=templates)
    frames=[json.loads(line) for line in (directory/'trajectory.jsonl').read_text().splitlines()];n=len(frames[0]['poses'])
    p,q=pose_arrays(frames[0]);distances=np.linalg.norm(p[:,None]-p[None,:],axis=2)
    assert np.min(distances[np.triu_indices(n,1)])>2*(bound+rd)
    assert not cfg['seed_labels'] and not cfg['fixed_body_indices']
    assert len(frames)==summary['completed_sweeps']//campaign_manifest['sample_every']+1
    state=copy.deepcopy(frames[0]['poses']);all_pairs=list(itertools.combinations(range(n),2))
    def classify(pairs=None):
        result=order.classify({'poses':state},pair_filter=pairs)
        entry={(*r['bodies'],r['motif_id']) for r in result['registered_tetramer_motifs'] if r['entry']}
        stay={(*r['bodies'],r['motif_id']) for r in result['registered_tetramer_motifs']}
        return entry,stay
    initial,_=classify();assert not initial;active=set();changes=[];gca_flips=Counter();gca_partitions=Counter()
    branch=Counter();counts=Counter();accepted=Counter();maximum_position_error=maximum_rotation_error=0.;rows=[]
    saved={f['sweep']:f for f in frames};current_sweep=0;selected=set();sweep_moves=0
    rj_cfg=cfg.get('reversible_jump');rj_labels=list(frames[0]['rj_state']['labels']) if rj_cfg else None
    rj_jumps=[];rj_counts=Counter();rj_k_hist=Counter()
    model_data=read(directory/'provenance/frozen-relative-model.json')
    dictionary_size=len(model_data['anchors'])
    def compare(expected):
        nonlocal maximum_position_error,maximum_rotation_error
        p,q=pose_arrays({'poses':state});ep,eq=pose_arrays({'poses':expected})
        pe=float(np.max(np.abs(p-ep)));re=float(np.max(np.abs(rotations(q)-rotations(eq))))
        maximum_position_error=max(maximum_position_error,pe);maximum_rotation_error=max(maximum_rotation_error,re)
        assert pe<5e-8 and re<5e-10,(job['id'],pe,re)
    def observe(frame):
        nonlocal state
        compare(frame['poses']);entry,stay=classify();assert active==(active&stay)|entry
        p,q=pose_arrays(frame);matrix=rotations(q);world=np.einsum('bij,aj->bai',matrix,atoms)+p[:,None,:]
        wall_clearance=float(np.min(radius-np.linalg.norm(world,axis=2)-radii[None,:]));assert wall_clearance>=-2e-8
        result=atomic.frame(p,q);assert result['hard_valid'],(job['id'],frame['sweep'],result)
        near={tuple(e) for e in result['depletion_edges']};native={k[:2] for k in active}
        assert native<=near
        if frame['sweep']==0:assert not near and not native
        rows.append(dict(sweep=frame['sweep'],native=graph_summary(n,native),
            nonspecific=graph_summary(n,near),core_contact=graph_summary(n,{tuple(e) for e in result['contact_edges']}),
            registered_keys=sorted(active),minimum_interbody_gap_A=result['minimum_interbody_gap_A'],
            minimum_atomic_wall_clearance_A=wall_clearance,hard_valid=True,
            native_neighbor_sets=[sorted(j if i==a else a for a,j in native if i in (a,j)) for i in range(n)],
            nonspecific_neighbor_sets=[sorted(j if i==a else a for a,j in near if i in (a,j)) for i in range(n)]))
        if rj_cfg:
            rj=frame['rj_state'];assert rj_labels==rj['labels']
            eta=np.asarray(rj['eta']);assert eta.shape==(len(rj_labels),6) and np.isfinite(eta).all()
            assert rj_cfg['min_components']<=len(rj_labels)<=rj_cfg['max_components']
            assert all(0<=label<dictionary_size for label in rj_labels)
            rows[-1]['rj']=dict(k=len(rj_labels),labels=list(rj_labels),distinct_labels=len(set(rj_labels)),eta_square_mean=float(np.mean(eta**2)))
        state=copy.deepcopy(frame['poses'])
    observe(frames[0]);move_count=0
    with (directory/'moves.jsonl').open() as stream:
        for serial,line in enumerate(stream):
            move=json.loads(line);sweep=move['sweep'];kind=move['kind'];move_count+=1
            if sweep!=current_sweep:
                if current_sweep:
                    assert selected==set(range(n)) and sweep_moves==n
                    if current_sweep in saved:observe(saved[current_sweep])
                current_sweep=sweep;selected=set();sweep_moves=0
            changed_pairs=[];previous=set(active);counts[kind]+=1
            if kind=='model_jump':
                assert rj_cfg is not None
                result=move['result'];before=len(rj_labels);assert result['before']==before
                birth=bool(result['birth']);rj_counts['attempted_births' if birth else 'attempted_deaths']+=1
                if result['boundary_null']:
                    assert not result['accepted']
                    assert before==(rj_cfg['max_components'] if birth else rj_cfg['min_components'])
                    rj_counts['boundary_nulls']+=1
                else:
                    expected=math.log(rj_cfg['poisson_mean'])-math.log(before+1) if birth else math.log(before)-math.log(rj_cfg['poisson_mean'])
                    assert abs(result['log_ratio']-expected)<1e-12
                    if result['accepted']:
                        slot=result['slot'];label=result['label'];assert 0<=label<dictionary_size
                        if birth:rj_labels.insert(slot,label)
                        else:assert rj_labels.pop(slot)==label
                        rj_counts['accepted_births' if birth else 'accepted_deaths']+=1
                assert len(rj_labels)==result['after']
                rj_k_hist[len(rj_labels)]+=1;rj_jumps.append(dict(sweep=sweep,**result))
            elif kind in ('local','global'):
                i=move['moving_index'];assert i not in selected;selected.add(i);sweep_moves+=1
                old=move['old_pose'];p,q=pose_arrays({'poses':[state[i]]});ep,eq=pose_arrays({'poses':[old]})
                assert np.max(np.abs(p-ep))<5e-8 and np.max(np.abs(rotations(q)-rotations(eq)))<5e-10
                if move['accepted']:
                    accepted[kind]+=1;state[i]=copy.deepcopy(move['retained_pose'])
                    assert state[i]==move['proposed_pose'];changed_pairs=[pair for pair in all_pairs if i in pair]
                    if kind=='global':branch[move['proposal']['branch']]+=1
            elif kind=='gca':
                result=move['result'];flipped=set(result['flipped_indices']);gca_partitions[str(sorted(result['component_sizes'],reverse=True))]+=1
                gca_flips['none' if not flipped else 'all' if len(flipped)==n else 'partial']+=1
                u=np.asarray(move['axis']);u/=np.linalg.norm(u);transform=2*np.outer(u,u)-np.eye(3)
                for i in flipped:
                    p,q=pose_arrays({'poses':[state[i]]});matrix=transform@rotations(q)[0]
                    state[i]=dict(position=(transform@p[0]).tolist(),orientation=Rotation.from_matrix(matrix).as_quat()[[3,0,1,2]].tolist())
                changed_pairs=[(a,b) for a,b in all_pairs if (a in flipped)!=(b in flipped)]
            elif kind=='center_shift':
                for pose in state:pose['position']=(np.asarray(pose['position'])+move['result']['displacement']).tolist()
            else:raise AssertionError(kind)
            if changed_pairs:
                entry,stay=classify(changed_pairs);subset=set(changed_pairs)
                active={k for k in active if k[:2] not in subset or k in stay}|entry
            if active!=previous:
                selected_component=(move.get('proposal') or {}).get('component_index')
                atlas_label=(rj_labels[selected_component] if rj_cfg and selected_component is not None else selected_component)
                changes.append(dict(sweep=sweep,update=move.get('update_in_sweep'),kind=kind,moving_index=move.get('moving_index'),
                    formed=sorted(active-previous),broken=sorted(previous-active),proposal=move.get('proposal'),atlas_label=atlas_label))
    assert selected==set(range(n)) and sweep_moves==n
    observe(saved[current_sweep]);checkpoint=read(directory/'checkpoint.json');compare(checkpoint['poses'])
    if rj_cfg:
        assert checkpoint['rj_state']['labels']==rj_labels
        assert len(rj_jumps)==summary['completed_sweeps']*rj_cfg['attempts_per_sweep']
        recorded=summary['counts']['model_jumps']
        for key,actual in [('births','attempted_births'),('deaths','attempted_deaths'),
            ('accepted_births','accepted_births'),('accepted_deaths','accepted_deaths'),('boundary_nulls','boundary_nulls')]:
            assert recorded[key]==rj_counts[actual]
    assert len(rows)==len(frames)
    for kind in ('local','global'):
        assert counts[kind]==summary['counts'][kind]['attempted']
        assert accepted[kind]==summary['counts'][kind]['accepted']
    for kind in ('gca','center_shift'):
        assert counts[kind]==summary['counts'][kind]['completed']==summary['counts'][kind]['attempted']
    # Episodes are observed at saved frames. No claimed continuous residence.
    episodes={};roundtrips={}
    for kind in ('native','nonspecific'):
        records=[];trip_count=0
        for body in range(n):
            previous=();begin=0;seen=set();changes_env=0
            for row in rows:
                env=tuple(row[kind+'_neighbor_sets'][body])
                if env!=previous:
                    records.append(dict(body=body,neighbors=list(previous),first_observed_sweep=begin,
                        last_observed_sweep=row['sweep']-campaign_manifest['sample_every'],
                        next_observation_sweep=row['sweep'],right_censored=False))
                    if env in seen:trip_count+=1
                    seen.add(previous);previous=env;begin=row['sweep'];changes_env+=1
            records.append(dict(body=body,neighbors=list(previous),first_observed_sweep=begin,last_observed_sweep=rows[-1]['sweep'],
                right_censored=True))
        episodes[kind]=records;roundtrips[kind]=trip_count
    result=dict(id=job['id'],replicate=job['replicate'],mode=job['mode'],variant=job.get('variant',job['mode']),
        model_label=job.get('model_label',cfg['metadata']['model_label']),passed=True,
        initial_native_component=rows[0]['native']['largest_component_size'],final_native=rows[-1]['native'],
        final_nonspecific=rows[-1]['nonspecific'],maximum_native_component=max(r['native']['largest_component_size'] for r in rows),
        first_native_association_sweep=min((e['sweep'] for e in changes if e['formed']),default=None),
        motif_formations=sum(len(e['formed']) for e in changes),motif_breakages=sum(len(e['broken']) for e in changes),
        changes=changes,rows=rows,observed_contact_episodes=episodes,observed_environment_returns=roundtrips,
        gca_flips=dict(gca_flips),gca_partitions=dict(gca_partitions),accepted_global_branches=dict(branch),
        counts=summary['counts'],cost=summary['cost'],cpu_seconds=summary['sampler_cpu_seconds'],
        reversible_jump=(dict(config=rj_cfg,counts=dict(rj_counts),post_attempt_k_histogram=dict(rj_k_hist),
            mean_k_after_attempt=sum(k*v for k,v in rj_k_hist.items())/len(rj_jumps),
            minimum_k=min(rj_k_hist),maximum_k=max(rj_k_hist),jumps=rj_jumps,
            labels_replay_to_every_frame_and_checkpoint=True,summary_jump_counts_match=True) if rj_cfg else None),
        preparation=cfg['metadata'],replay=dict(passed=True,records=move_count,frames=len(frames),
            maximum_position_error_A=maximum_position_error,maximum_rotation_error=maximum_rotation_error),
        provenance=dict(model_sha256=manifest['model_sha256'],binary_sha256=manifest['executable_sha256'],
            source_bundle_sha256=manifest['source_bundle_sha256']),protocol=order.protocol,
        source_sha256={str(directory/name):sha(directory/name) for name in ('config.json','manifest.json','summary.json','trajectory.jsonl','moves.jsonl')})
    save(out/'runs'/job['id']/'analysis.json',result)
    print(json.dumps(dict(id=job['id'],passed=True,native=result['final_native']['components'],
        nonspecific=result['final_nonspecific']['components'],formations=result['motif_formations'],breakages=result['motif_breakages'])),flush=True)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--out',type=Path);p.add_argument('--reference',type=Path,default=Path('/home/xvg/protein-nucleation'))
    p.add_argument('--workers',type=int,default=4);a=p.parse_args();a.campaign=a.campaign.resolve()
    a.out=(a.out or a.campaign/'assessment').resolve();a.out.mkdir(parents=True,exist_ok=True)
    manifest=read(a.campaign/'manifest.json');assert read(a.campaign/'summary.json')['complete']
    tasks=[(str(a.reference),str(a.campaign),str(a.out),job) for job in manifest['jobs']]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(one,tasks))
    save(a.out/'analysis.json',dict(passed=True,results=results,manifest=manifest,source_sha256=sha(Path(__file__))))
    model_scope=(
        'Production proposals compare a geometry-only atlas (native-off) with a 50/50 blend of that atlas and native docking charts (native-on). Both use the same labelled variable-K RJ law and instantaneous conditional-mean transport. Native-off removes external intertetramer native docking charts; both arms retain the same rigid tetramer shape. The base atlas is not retrained.'
        if manifest.get('reversible_jump') else
        f"Production model: **{manifest['model_label']}**. The default atlas includes native examples; these runs test assembly accessibility with that prior, not prior-free discovery. Transport changes instantaneous conditional means; the base atlas is not retrained.")
    lines=['# Assembly from dispersed tetramers','',
        f"Four independent random preparations, paired proposal controls, {manifest['sweeps']} sweeps each. All 12 tetramers are mobile; no seed, no initial exclusion contact, independent Haar orientations. R=354.5082 Å, rd=1.5 Å, z=0.035 Å⁻³. GCA and common shifts follow every sweep.",'',
        model_scope,'',
        '| Start | Mode | Final registered components | Native bonds | Nonspecific bonds | Form/break motifs | First native sweep | CPU s |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['replicate']} | {r['variant']} | {list(map(len,r['final_native']['components']))} | {r['final_native']['bonds']} | {r['final_nonspecific']['bonds']} | {r['motif_formations']}/{r['motif_breakages']} | {r['first_native_association_sweep']} | {r['cpu_seconds']:.2f} |")
    event_kinds=Counter()
    for result in results:
        for event in result['changes']:event_kinds[event['kind']]+=len(event['formed'])
    largest_by_variant={v:[r['final_native']['largest_component_size'] for r in results if r['variant']==v] for v in dict.fromkeys(r['variant'] for r in results)}
    lines+=['',f"Final largest registered component sizes by variant: {largest_by_variant}. Motif formations by update type: {dict(event_kinds)}; total motif breakages: {sum(r['motif_breakages'] for r in results)}. Sampled native environment returns: {sum(r['observed_environment_returns']['native'] for r in results)}; nonspecific returns: {sum(r['observed_environment_returns']['nonspecific'] for r in results)}.",'',
        'All starts pass the explicit exclusion-bound separation certificate and the independent initial native/contact classification. Every saved frame passes interbody atom-union overlap and atomic-wall checks. Every accepted local/global update, GCA half-turn, and common shift is reconstructed from the move log to every stored frame and final checkpoint. Native bonds use the unchanged registered tetramer motif plus external residue-patch criterion, excluding intrinsic contacts. Nonspecific bonds use minimum interbody atom gap ≤2rd, allowing exclusion overlap without registry.','',
        'Registered oligomer assembly, when observed, does not establish crystal growth, equilibrium, or a reliable sampling speedup. The four independent starts share paired random streams across algorithms; reusing these starts extends the same experiment rather than adding independent preparations. Environment returns and residence are sampled-frame observations, not proofs of continuous residence. The native analysis does not certify a defect-free global crystal lattice.','']
    if manifest.get('reversible_jump'):
        lines+=['## Reversible-jump auxiliary diagnostics','',
            'Model-jump records change labelled component slots and Gaussian residuals, not physical poses. Every logged label insertion/deletion and Poisson count ratio is independently replayed to stored frames/checkpoint. K occupancy is correlated trajectory data; agreement with its prior alone would not validate the physical marginal.','',
            '| Start | Variant | K range | Mean K after attempt | Accepted births/deaths | Boundary nulls |',
            '|---|---|---:|---:|---:|---:|']
        for r in results:
            jump=r['reversible_jump'];c=jump['counts']
            lines.append(f"| {r['replicate']} | {r['variant']} | {jump['minimum_k']}–{jump['maximum_k']} | {jump['mean_k_after_attempt']:.2f} | {c.get('accepted_births',0)}/{c.get('accepted_deaths',0)} | {c.get('boundary_nulls',0)} |")
        lines+=['','The native-off atlas uses shape geometry only; native-on mixes that same geometry atlas with explicit native charts. The native template catalogue is read by this post-hoc classifier in both arms. It does not enter the native-off production proposal. These are comparisons of supplied proposal priors, not changes to the physical equilibrium target.','']
    (a.out/'report.md').write_text('\n'.join(lines))
    plot(a.out,results,manifest)
    print(json.dumps(dict(passed=True,out=str(a.out))),flush=True)

def plot(out,results,manifest):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(2,4,figsize=(12,5.7),sharex=True,sharey='row')
    variants=list(dict.fromkeys(r.get('variant',r['mode']) for r in results))
    colors={v:['#355C9B','#BB4C31','#418A59','#795D9B'][i%4] for i,v in enumerate(variants)}
    for result in results:
        x=[r['sweep'] for r in result['rows']];column=result['replicate']
        for row,kind in enumerate(('native','nonspecific')):
            variant=result.get('variant',result['mode'])
            axes[row,column].step(x,[r[kind]['largest_component_size'] for r in result['rows']],where='post',
                label=variant,color=colors[variant],lw=2.8 if variant==variants[0] else 1.4,alpha=.9)
            axes[row,column].set_ylim(.5,12.5);axes[row,column].grid(alpha=.2)
    for column in range(4):axes[0,column].set_title(f'Independent start {column}');axes[1,column].set_xlabel('MC sweeps')
    axes[0,0].set_ylabel('Largest registered\ncomponent');axes[1,0].set_ylabel('Largest exclusion-contact\ncomponent')
    axes[0,0].legend(frameon=False,fontsize=9)
    title='Assembly from 12 separated tetramers — '+('variable-K RJ, matched atlas controls' if manifest.get('reversible_jump') else 'native-informed atlas')
    figure.suptitle(title,fontsize=14)
    figure.text(.5,.01,'Four paired starts; no initial contacts. Finite assembly pilot, not an equilibrium or mixing-speedup estimate.',ha='center',fontsize=9)
    figure.tight_layout(rect=(0,.035,1,.94))
    for ext in ('png','svg','pdf'):figure.savefig(out/f'free-assembly-components.{ext}',dpi=180)
    plt.close(figure)
    if manifest.get('reversible_jump'):
        import numpy as np
        from scipy.special import gammaln
        figure,axis=plt.subplots(figsize=(8,4.4));config=manifest['reversible_jump']
        k=np.arange(config['min_components'],config['max_components']+1)
        logw=k*np.log(config['poisson_mean'])-gammaln(k+1);expected=np.exp(logw-np.max(logw));expected/=expected.sum()
        for variant in variants:
            histogram=Counter()
            for result in results:
                if result['variant']==variant:
                    histogram.update({int(x):v for x,v in result['reversible_jump']['post_attempt_k_histogram'].items()})
            total=sum(histogram.values());axis.step(k,[histogram[int(x)]/total for x in k],where='mid',
                color=colors[variant],label=variant,lw=2)
        axis.plot(k,expected,'k--',label='Truncated Poisson prior',lw=1.5)
        axis.set(xlabel='Active labelled components K',ylabel='Fraction of RJ attempts',title='Auxiliary component-count occupancy')
        axis.legend(frameon=False);axis.grid(alpha=.2)
        figure.text(.5,.02,'Pooled correlated trajectories; descriptive prior check, not physical-marginal validation.',ha='center',fontsize=9)
        figure.tight_layout(rect=(0,.05,1,1))
        for ext in ('png','svg','pdf'):figure.savefig(out/f'rj-k-occupancy.{ext}',dpi=180)
        plt.close(figure)

if __name__=='__main__':main()
