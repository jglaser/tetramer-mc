#!/usr/bin/env python3
"""Passive fixed-patch changes for accepted singleton moves in reset pilots.

Replay every phase and retain rejected events as zero changes. Only pairs
involving the accepted moving body and conservative old/new neighboring centers
are classified, using the existing ContactObserver with native=None.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import numpy as np
from analyze_contact_efficiency import ContactObserver
from compare_singleton_retry_caps import read, sha, require, quantiles


def neighbors(observer, state, moving, proposed):
    """Conservative union of center-bound candidates at both endpoints."""
    positions=np.asarray([p['position'] for p in state],float)
    result=set();reach=2*(observer.bound+observer.rd)
    for endpoint in (state[moving],proposed):
        center=np.asarray(endpoint['position'],float)
        delta=positions-center
        imaged=positions
        if observer.boundary['kind']=='periodic':
            imaged=positions-observer.lengths*np.floor(delta/observer.lengths+.5)
            delta=imaged-center
        guard=512*np.finfo(float).eps*(1+np.linalg.norm(center)+np.linalg.norm(imaged,axis=1)+observer.bound+observer.rd)
        result.update(int(j) for j in np.flatnonzero(np.linalg.norm(delta,axis=1)<=reach+guard) if j!=moving)
    return sorted(result)


def tokens(observer,state,moving,pose,candidates):
    found=set()
    for j in candidates:
        i,k=sorted((moving,j))
        a=pose if i==moving else state[i]
        b=pose if k==moving else state[k]
        observation=observer.classify([a,b])
        require(not observation['instantaneous_native_keys'] and observation['native_cycle'] is None,'Native observer unexpectedly invoked')
        found.update((i,k,pa,pb) for _,_,pa,pb in observation['tokens'])
    return found


def changed_patches(observer,state,moving,new,expected_old=None,expected_new=None):
    candidates=neighbors(observer,state,moving,new)
    if expected_old is not None and expected_new is not None:
        require((set(expected_old)|set(expected_new))<=set(candidates),'Conservative center bound omitted a recorded partner')
    before=tokens(observer,state,moving,state[moving],candidates)
    after=tokens(observer,state,moving,new,candidates)
    partners=lambda ts:{j if i==moving else i for i,j,_,_ in ts}
    old_partners,new_partners=partners(before),partners(after)
    if expected_old is not None:require(old_partners==set(expected_old),'Patch observer/source contact graph disagree')
    if expected_new is not None:require(new_partners==set(expected_new),'Patch observer/retained contact graph disagree')
    union=before|after;jaccard=1-len(before&after)/len(union) if union else 0.
    q=np.asarray(state[moving]['orientation'],float);r=np.asarray(new['orientation'],float)
    q/=np.linalg.norm(q);r/=np.linalg.norm(r)
    angle=math.degrees(2*math.acos(min(1.,abs(float(q@r)))))
    delta=np.asarray(new['position'])-np.asarray(state[moving]['position'])
    if observer.boundary['kind']=='periodic':delta-=observer.lengths*np.floor(delta/observer.lengths+.5)
    return dict(candidate_neighbors=candidates,classified_pair_endpoints=2*len(candidates),
        old_partners=sorted(old_partners),new_partners=sorted(new_partners),
        old_patch_tokens=sorted(before),new_patch_tokens=sorted(after),
        gained_patch_tokens=sorted(after-before),lost_patch_tokens=sorted(before-after),
        jaccard_distance=jaccard,patch_changed=before!=after,
        same_partners=old_partners==new_partners,partner_changed=old_partners!=new_partners,
        displacement_A=float(np.linalg.norm(delta)),rotation_degrees=angle)


def aggregate(events,attempted,cpu):
    changed=sum(e['patch_changed'] for e in events)
    unchanged_partner=sum(e['patch_changed'] and e['same_partners'] for e in events)
    count=Counter(attempted=attempted,accepted=len(events),rejected=attempted-len(events),
        patch_changes=changed,same_partner_patch_changes=unchanged_partner,
        partner_changes=sum(e['partner_changed'] for e in events),
        gained_patch_tokens=sum(len(e['gained_patch_tokens']) for e in events),
        lost_patch_tokens=sum(len(e['lost_patch_tokens']) for e in events))
    total=math.fsum(e['jaccard_distance'] for e in events)
    return dict(counts=dict(count),kernel_cpu_seconds=cpu,
        patch_changes_per_all_event=changed/attempted if attempted else None,
        same_partner_patch_changes_per_all_event=unchanged_partner/attempted if attempted else None,
        mean_jaccard_per_all_event=total/attempted if attempted else None,
        patch_changes_per_kernel_cpu_second=changed/cpu if cpu else None,
        jaccard_sum_per_kernel_cpu_second=total/cpu if cpu else None,
        accepted_jaccard=quantiles([e['jaccard_distance'] for e in events]),
        accepted_displacement_A=quantiles([e['displacement_A'] for e in events]),
        accepted_rotation_degrees=quantiles([e['rotation_degrees'] for e in events]))


def analyze(pilot,patch_path,out):
    pilot=Path(pilot).resolve();patch_path=Path(patch_path).resolve();out=Path(out).resolve()
    protocol,status=read(pilot/'protocol.json'),read(pilot/'status.json')
    require(status['complete'] and not status['running'],'Physical pilot must be complete')
    for name,digest in read(pilot/'freeze.json').items():require(sha(pilot/name)==digest,'Frozen pilot input changed')
    patch=read(patch_path);shape_path=pilot/'provenance/shape.json';shape=read(shape_path)
    require(patch['shape_sha256']==sha(shape_path)==protocol['shape_sha256'],'Patch map/shape identity mismatch')
    require(set(patch['atom_patch_ids'])==set(patch['patch_dictionary']),'Patch dictionary mismatch or filtered atom map')
    populations=[];event_rows=[];hashes={};by_arm={}
    for job in protocol['jobs']:
        directory=Path(job['directory']);config=read(job['config']);summary=read(directory/'summary.json')
        require(summary['complete'] and summary['all_attempted_events_retained'] and summary['all_phase_endpoints_replayed'],'Missing original exact replay receipt')
        for name in ('events','phases'):
            p=directory/(name+'.jsonl');digest=sha(p);require(digest==summary[name+'_sha256'],'Physical output changed');hashes[str(p)]=digest
        receipt={r['phase']:r for r in map(json.loads,(directory/'phases.jsonl').read_text().splitlines())}
        observer=ContactObserver(shape,patch['atom_patch_ids'],config,native=None)
        state=None;phase=None;attempted=0;accepted=[];phase_counts=Counter();seen=set()
        for row in map(json.loads,(directory/'events.jsonl').read_text().splitlines()):
            number=row['phase']
            if number!=phase:
                if phase is not None:
                    require(phase_counts['attempted']==receipt[phase]['counts']['events'] and phase_counts['accepted']==receipt[phase]['counts']['accepted'],'Phase disposition mismatch')
                require(number not in seen,'Noncontiguous or duplicate reset phase');seen.add(number)
                require(number in receipt and receipt[number]['all_attempts_replayed'],'Missing phase endpoint audit')
                phase=number;state=copy.deepcopy(config['initial_poses']);phase_counts=Counter()
            if row['kind']!='cluster_event':continue
            require(len(row['members'])==1,'Only singleton patch replay supported')
            moving=row['members'][0];require(state[moving]==row['old_poses'][0],'Old pose does not match exact reset replay')
            attempted+=1;phase_counts['attempted']+=1
            if row['accepted']:
                require(row['hard_valid'] and row['proposed_poses']==row['retained_poses'],'Invalid accepted state')
                old=set(row['diagnostic']['external_partners'])
                other=lambda e:e[1] if e[0]==moving else e[0]
                lost={other(e) for e in row['proposed_lost_contacts']};gained={other(e) for e in row['proposed_gained_contacts']}
                new=(old-lost)|gained
                event=changed_patches(observer,state,moving,row['retained_poses'][0],old,new)
                event.update(job=job['id'],arm=job['arm'],population=job['population'],phase=phase,event=row['event'],member=moving,
                    event_time=row['event_time'],branch=row['proposal']['branch'])
                accepted.append(event);event_rows.append(event);phase_counts['accepted']+=1
            else:require(row['old_poses']==row['retained_poses'],'Rejected state changed')
            state[moving]=row['retained_poses'][0]
        if phase is not None:require(phase_counts['attempted']==receipt[phase]['counts']['events'] and phase_counts['accepted']==receipt[phase]['counts']['accepted'],'Terminal phase disposition mismatch')
        require(seen==set(receipt)==set(range(1,protocol['phases']+1)),'Missing reset phases')
        require(attempted==summary['counts']['events'] and len(accepted)==summary['counts']['accepted'],'Attempt/acceptance totals mismatch')
        result=aggregate(accepted,attempted,summary['kernel_cpu_seconds'])
        result.update(id=job['id'],arm=job['arm'],population=job['population'],seed=job['seed'],phases_replayed=len(seen))
        populations.append(result);by_arm.setdefault(job['arm'],[]).extend(accepted)
    arms={}
    for arm,events in by_arm.items():
        ps=[p for p in populations if p['arm']==arm]
        arms[arm]=aggregate(events,sum(p['counts']['attempted'] for p in ps),sum(p['kernel_cpu_seconds'] for p in ps))
    output=dict(schema='singleton-fixed-patch-changes-v1',scope='Passive reset-event geometry; no native observer or equilibrium/ESS inference',
        pilot=str(pilot),protocol_sha256=sha(pilot/'protocol.json'),shape_sha256=sha(shape_path),patch_map_sha256=sha(patch_path),
        patch_count=len(patch['patch_dictionary']),patch_definition=patch['construction'],populations=populations,arms=arms,
        accepted_events=event_rows,input_sha256=hashes,
        observer_sha256=sha(Path(__file__).with_name('analyze_contact_efficiency.py')),script_sha256=sha(__file__),
        broadphase='Full spectator center scan at old and new poses, conservative 2*(body bound+rd)+observer guard; pairwise ContactObserver(native=None) only within union.',
        replay='Every old/retained pose and phase disposition replayed; original hashed Rust phase receipts independently certify full endpoint poses.',
        limitations='Fixed 32-patch map is coarse. A token change can be a threshold crossing or within-basin fluctuation, not necessarily a new basin or improved native registry. Rejected events contribute zero; no conditional accepted-only denominator is substituted.')
    out.mkdir(parents=True,exist_ok=True);(out/'patch-changes.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    archive=out/'analysis-source';archive.mkdir(exist_ok=True)
    for p in (Path(__file__),Path(__file__).with_name('analyze_contact_efficiency.py'),Path(__file__).with_name('compare_singleton_retry_caps.py'),patch_path):shutil.copy2(p,archive/p.name)
    lines=['# Fixed-patch reorganization in singleton reset controls','',output['limitations'],'',
        '| Arm | Attempted | Accepted | Partner changes | Patch changes | Patch changes with same partners | Mean patch Jaccard / all events | Patch changes / CPU s |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,r in arms.items():
        c=r['counts'];lines.append(f'| {arm} | {c["attempted"]} | {c["accepted"]} | {c["partner_changes"]} | {c["patch_changes"]} | {c["same_partner_patch_changes"]} | {r["mean_jaccard_per_all_event"]:.6f} | {r["patch_changes_per_kernel_cpu_second"]:.4f} |')
    lines+=['','Every accepted event records gained/lost patch tokens, Jaccard distance, translation and rotation in [patch-changes.json](patch-changes.json). '
        'All rejected events enter denominators with zero changes. Only the moving body is classified against conservative neighboring candidates; '
        'the resulting partner sets must exactly match the independently recorded exclusion-contact graph at both endpoints. '
        'No native classifier is instantiated. This is not a trajectory ESS measurement.','']
    (out/'report.md').write_text('\n'.join(lines))
    paths=[p for p in out.rglob('*') if p.is_file() and p.name!='manifest.json']
    (out/'manifest.json').write_text(json.dumps({'complete':True,'files':{str(p.relative_to(out)):sha(p) for p in paths}},indent=2)+'\n')
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pilot',type=Path,required=True)
    parser.add_argument('--patch-map',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=analyze(args.pilot,args.patch_map,args.out)
    print(json.dumps(result['arms'],indent=2))
if __name__=='__main__':main()
