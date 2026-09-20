#!/usr/bin/env python3
"""Read-only endpoint/gate audit of the recorded outside-R8 docking excursion.

No new clouds, trajectory steps, fitted proposals, or equilibrium estimates.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from prepare_smc_normalizer_atlas import ROOT, Density, arrays, read, relative_poses, sha, write
from prepare_deep_far_normalizer_atlas import registration


def describe(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return None
    assert np.isfinite(a).all()
    return dict(count=len(a), mean=float(a.mean()), **dict(zip(
        ('min', 'p05', 'median', 'p95', 'max'), np.quantile(a, [0, .05, .5, .95, 1]).tolist())))


def classify(poses, fixed, metadata, old_density, new_density):
    relative = relative_poses(poses, fixed)
    q = registration(poses, metadata)
    old_r = old_density.evaluate(relative)[1][:, 0]
    new_r = new_density.evaluate(relative)[1][:, 0]
    labels = np.full(len(poses), 'far_outside_both', dtype=object)
    labels[(q >= 5) & (old_r > 8) & (new_r <= 5)] = 'new_B5_shell'
    labels[(q >= 5) & (old_r > 8) & (new_r <= 3)] = 'new_B3'
    labels[(q >= 5) & (old_r <= 8)] = 'old_R8'
    labels[(q >= 2) & (q < 5)] = 'intermediate'
    labels[(q > 1) & (q < 2)] = 'shoulder'
    labels[q <= 1] = 'native'
    return q, old_r, new_r, labels


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--campaign', type=Path, default=ROOT/'runs/posterior-docking-mis-5000')
    ap.add_argument('--plan', type=Path, default=ROOT/'runs/outside-r8-local-region-20260920/site0/validation-plan.json')
    ap.add_argument('--out', type=Path, default=ROOT/'runs/outside-r8-production-20260920/assessment/attempt-audit.json')
    args = ap.parse_args()
    plan = read(args.plan)
    old = read(plan['poststratification']['old_region'])
    new = read(plan['regions']['3']['path'])
    old_density, new_density = Density(old['gaussian_chart']), Density(new['gaussian_chart'])
    model_path = args.campaign/'provenance/model.json'
    model = read(model_path)
    model_density = Density(model)
    reports = []
    for name in ('site0-m1-deep-r01-c09', 'site0-m1-deep-r01-c0'):
        directory = args.campaign/'runs'/name
        cfg = read(directory/'config.json')
        assert cfg['fixed_poses'] == [old['fixed_neighbor']]
        assert cfg['reservoir_density'] == old['activity'] and cfg['depletant_radius'] == old['depletant_radius']
        rows = [json.loads(line) for line in (directory/'moves.jsonl').open()]
        frames = [json.loads(line) for line in (directory/'trajectory.jsonl').open()]
        assert len(rows) == 3*(len(frames)-1)
        assert [(r['cycle'], r['attempt']) for r in rows] == [(c,a) for c in range(1,len(frames)) for a in range(3)]
        assert all(r['proposed_pose'] is not None for r in rows), 'Null endpoints require explicit separate classification'
        old_poses, new_poses = [r['old_pose'] for r in rows], [r['proposed_pose'] for r in rows]
        oq, ora, orb, ol = classify(old_poses, old['fixed_neighbor'], old['physical_metric'], old_density, new_density)
        nq, nra, nrb, nl = classify(new_poses, old['fixed_neighbor'], old['physical_metric'], old_density, new_density)
        ot, oquat, _ = arrays(old_poses); nt, nquat, _ = arrays(new_poses)
        translation = np.linalg.norm(nt-ot,axis=1)
        angle = np.degrees(2*np.arccos(np.clip(np.abs(np.sum(oquat*nquat,axis=1)),0,1)))
        branch = np.asarray([r['branch'] for r in rows])
        accepted = np.asarray([r['accepted'] for r in rows])
        capture = np.asarray([r['capture_valid'] for r in rows])
        hard = np.asarray([r['hard_valid'] for r in rows])
        gated = np.asarray([r['gate'] is not None for r in rows])
        assert np.array_equal(gated, capture & hard)
        assert not (accepted & ~gated).any()
        correction = np.asarray([r['proposal']['log_reverse_forward'] for r in rows])
        gate_log = np.asarray([r['gate']['log_weight'] if r['gate'] else np.nan for r in rows])
        source = np.asarray([r['proposal']['trace']['source'] if r['branch']=='involution' else -1 for r in rows])
        target = np.asarray([r['proposal']['trace']['target'] if r['branch']=='involution' else -1 for r in rows])
        old_G = np.asarray([r['proposal'].get('full_old_gaussian_log_density',np.nan) for r in rows])
        new_G = np.asarray([r['proposal'].get('full_new_gaussian_log_density',np.nan) for r in rows])
        coefficient = np.log1p(1/cfg['poisson_lambda_ratio'])
        gate_error, acceptance_error, correction_error = 0.,0.,0.
        for i,r in enumerate(rows):
            if gated[i]:
                gate_error = max(gate_error,abs(gate_log[i]-coefficient*(r['gate']['gained']-r['gate']['lost'])))
                acceptance_error = max(acceptance_error,abs(r['log_acceptance']-min(0.,correction[i]+gate_log[i])))
            if branch[i]=='involution':
                correction_error = max(correction_error,abs(correction[i]-(old_G[i]-new_G[i])))
            retained = r['proposed_pose'] if accepted[i] else r['old_pose']
            assert retained == r['retained_pose']
            if i+1<len(rows): assert r['retained_pose']==rows[i+1]['old_pose']
            if r['attempt']==2: assert r['retained_pose']==frames[r['cycle']]['pose']
        # A deterministic independent full-density reconstruction audit, without
        # recomputing all K components for every attempted endpoint.
        eligible = np.flatnonzero(branch=='involution')
        audit_ids = eligible[np.unique(np.linspace(0,len(eligible)-1,256,dtype=int))]
        go = model_density.evaluate(relative_poses([old_poses[i] for i in audit_ids],old['fixed_neighbor']))[0]
        gn = model_density.evaluate(relative_poses([new_poses[i] for i in audit_ids],old['fixed_neighbor']))[0]
        density_error = float(max(np.max(np.abs(go-old_G[audit_ids])), np.max(np.abs(gn-new_G[audit_ids]))))
        assert max(gate_error,acceptance_error,correction_error)<1e-10 and density_error<1e-7

        def summarize(mask):
            ids = np.flatnonzero(mask); gi=ids[gated[ids]]; ai=ids[accepted[ids]]
            learned = ids[branch[ids]=='involution']
            def counts(x): return dict(Counter(map(str,x)))
            def per_component(values):
                result = []
                for k, n in Counter(values[learned]).most_common():
                    ki = learned[values[learned]==k]
                    result.append(dict(component=int(k),attempts=n,hard_valid=int(hard[ki].sum()),
                        accepted=int(accepted[ki].sum()),model_weight=model['weights'][k],
                        destinations=counts(nl[ki]),
                        gated_log_poisson_factor=describe(gate_log[ki[gated[ki]]]),
                        gated_log_proposal_ratio=describe(correction[ki[gated[ki]]]),
                        summed_recorded_acceptance_probabilities=float(np.exp(np.minimum(0,
                            gate_log[ki[gated[ki]]]+correction[ki[gated[ki]]])).sum())))
                return result
            return dict(attempts=len(ids), capture_rejected=int((~capture[ids]).sum()),
                hard_rejected=int((capture[ids]&~hard[ids]).sum()),hard_valid=len(gi),
                endpoint_rejected=int((~accepted[gi]).sum()), accepted=len(ai),
                accepted_pose_changes=int(np.count_nonzero((translation[ai]>1e-10)|(angle[ai]>1e-6))),
                destination_counts=counts(nl[ids]),accepted_destination_counts=counts(nl[ai]),
                translation_A=describe(translation[ids]),rotation_deg=describe(angle[ids]),
                accepted_translation_A=describe(translation[ai]),accepted_rotation_deg=describe(angle[ai]),
                gated_log_poisson_factor=describe(gate_log[gi]),gated_log_proposal_ratio=describe(correction[gi]),
                gated_log_acceptance=describe([rows[i]['log_acceptance'] for i in gi]),
                summed_recorded_acceptance_probabilities=float(np.exp(np.minimum(0,gate_log[gi]+correction[gi])).sum()),
                gated_counts_gained=describe([rows[i]['gate']['gained'] for i in gi]),
                gated_counts_lost=describe([rows[i]['gate']['lost'] for i in gi]),
                proposal_old_G=describe(old_G[learned]),proposal_new_G=describe(new_G[learned]),
                source_components=per_component(source),target_components=per_component(target),
                same_component_attempts=int((source[learned]==target[learned]).sum()),
                same_component_accepted=int(np.sum((source[learned]==target[learned])&accepted[learned])))

        b3=(oq>=5)&(ora>8)&(orb<=3); b5=(oq>=5)&(ora>8)&(orb<=5)
        nb3=(nq>=5)&(nra>8)&(nrb<=3); nb5=(nq>=5)&(nra>8)&(nrb<=5)
        masks={'all':np.ones(len(rows),dtype=bool),'old_R8':ol=='old_R8','old_B3':b3,'old_B5':b5,
               'old_B5_proposes_escape':b5&~nb5,'old_B5_proposes_old_R8':b5&(nl=='old_R8'),
               'old_B3_proposes_escape':b3&~nb3,
               'final_dwell_cycles_2770_to_5000':np.asarray([r['cycle']>=2770 for r in rows])}
        masks['final_dwell_proposes_old_R8']=masks['final_dwell_cycles_2770_to_5000']&b5&(nl=='old_R8')
        groups = {}
        for label,mask in masks.items():
            groups[label] = dict(all=summarize(mask),by_branch={str(b):summarize(mask&(branch==b)) for b in np.unique(branch)})
        groups['B5_destinations']={str(label):{str(b):summarize(b5&(nl==label)&(branch==b))
            for b in np.unique(branch)} for label in np.unique(nl[b5])}
        # Every accepted boundary crossing is recorded, including short exits
        # that the once-per-cycle retained trajectory might not show.
        crossings=[]
        for i in np.flatnonzero(accepted & (b5 != nb5)):
            crossings.append(dict(cycle=rows[i]['cycle'],attempt=rows[i]['attempt'],branch=branch[i],
                source=int(source[i]),target=int(target[i]),entering=bool(nb5[i]),old_label=ol[i],new_label=nl[i],
                old_q=float(oq[i]),new_q=float(nq[i]),old_R8_radius=float(ora[i]),new_R8_radius=float(nra[i]),
                old_extension_radius=float(orb[i]),new_extension_radius=float(nrb[i]),
                gate_log=float(gate_log[i]),correction=float(correction[i]),log_acceptance=rows[i]['log_acceptance']))
        # Runs of unchanged retained poses are different from basin residence.
        stuck=[];start=0
        for c in range(1,len(frames)+1):
            if c==len(frames) or frames[c]['pose']!=frames[start]['pose']:
                stuck.append(dict(first_cycle=start,last_cycle=c-1,frames=c-start));start=c
        fq, fra, frb, fl = classify([f['pose'] for f in frames],old['fixed_neighbor'],old['physical_metric'],old_density,new_density)
        pose_runs_in_b5=[s for s in stuck if fl[s['first_cycle']] in ('new_B3','new_B5_shell')]
        reports.append(dict(id=name,config_sha256=sha(directory/'config.json'),moves_sha256=sha(directory/'moves.jsonl'),
            trajectory_sha256=sha(directory/'trajectory.jsonl'),groups=groups,B5_crossings=crossings,
            unchanged_pose_runs=dict(total=len(stuck),maximum=max(s['frames'] for s in stuck),
                B5_runs=len(pose_runs_in_b5),B5_maximum=max([s['frames'] for s in pose_runs_in_b5],default=0),
                B5_lengths=describe([s['frames'] for s in pose_runs_in_b5]),
                B5_longest=sorted(pose_runs_in_b5,key=lambda s:-s['frames'])[:10]),
            audits=dict(gate_log_max_error=gate_error,acceptance_log_max_error=acceptance_error,
                correction_max_error=correction_error,independent_full_density_max_error=density_error,
                independent_density_endpoint_pairs=len(audit_ids),all_retained_states_replayed=True),
            frozen_configuration=dict(activity=cfg['reservoir_density'],lambda_ratio=cfg['poisson_lambda_ratio'],
                uniform_probability=cfg['uniform_probability'],local_attempts_per_cycle=cfg['local_attempts_per_cycle'])))
        if name.endswith('c09'):
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(1,2,figsize=(11,4.1))
            returning=b5&(nl=='old_R8')&(branch=='involution')
            stages=[int(returning.sum()),int((returning&gated).sum()),int((returning&accepted).sum())]
            axes[0].bar(['Proposed','Hard-valid','Accepted'],stages,color=['#81939b','#439ca8','#df9555'])
            for i,v in enumerate(stages):axes[0].text(i,v+20,str(v),ha='center')
            axes[0].set_ylim(0,1150);axes[0].set_ylabel('Recorded learned return attempts')
            axes[0].set_title('New B5 → old R8, c=0.9')
            for k,color in [(113,'#607d8b'),(114,'#439ca8'),(115,'#df9555')]:
                ids=np.flatnonzero(returning&gated&(target==k))
                axes[1].scatter(correction[ids],gate_log[ids],s=14,alpha=.65,color=color,label=f'Target component {k}')
            ids=np.flatnonzero(returning&accepted)
            axes[1].scatter(correction[ids],gate_log[ids],s=100,facecolors='none',edgecolors='black',label='Accepted')
            xx=np.linspace(-17,1,100);axes[1].plot(xx,-xx,'k--',lw=1,label='log acceptance = 0')
            axes[1].set_xlim(-17,1);axes[1].set_ylim(-6,7)
            axes[1].set_xlabel('Log proposal correction: log G(old) − log G(new)')
            axes[1].set_ylabel('Fresh Poisson log factor')
            axes[1].set_title('261 geometrically valid return proposals')
            axes[1].legend(fontsize=7,loc='lower left');fig.tight_layout()
            args.out.parent.mkdir(parents=True,exist_ok=True)
            for suffix in ('png','svg'):fig.savefig(args.out.parent/f'escape-proposal-factors.{suffix}',dpi=170)
            plt.close(fig)
    result=dict(schema=1,scope='Read-only diagnostic of existing attempts; no fresh draws or physical-mass estimates.',
        model_sha256=sha(model_path),plan_sha256=sha(args.plan),analyzer_sha256=sha(__file__),
        source_sha256={name:sha(args.campaign/'provenance'/name) for name in ('docking.rs','depletion.rs')},
        auxiliary_state='No cloud or Poisson weight persists in the Markov state. Every geometrically valid attempt draws a fresh conditional pair cloud, uses log(1+z/lambda)*(gained-lost), and discards it. The retained state contains pose/contact only. This is not a persistent pseudo-marginal likelihood estimator.',
        classification='Native q<=1, shoulder1<q<2, intermediate2<=q<5; far q>=5 split into oldR8, newR3 outsideoldR8, newR5 shell outsideoldR8, and neither. Candidate classifications are geometric and include hard-invalid endpoints.',
        limitations='One realized cloud per candidate cannot separate stochastic gate loss from deterministic free-energy and proposal mismatch. Accepted moves and expected acceptance conditional on sampled clouds are not independent equilibrium observations. Data-dependent region and finite trajectory selection prevent a calibrated mechanism claim.',
        reports=reports)
    args.out.parent.mkdir(parents=True,exist_ok=True);write(args.out,result)
    for r in reports:
        print(r['id'])
        for key in ('old_R8','old_B5','old_B5_proposes_escape','old_B5_proposes_old_R8'):
            print(key,json.dumps({b:{k:v[k] for k in ('attempts','hard_valid','endpoint_rejected','accepted','summed_recorded_acceptance_probabilities','gated_log_poisson_factor','gated_log_proposal_ratio')} for b,v in r['groups'][key]['by_branch'].items()}))
        print('unchanged',r['unchanged_pose_runs'])


if __name__=='__main__':main()
