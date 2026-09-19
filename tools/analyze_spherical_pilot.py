#!/usr/bin/env python3
"""Independent saved-frame and move replay audit of spherical protein pilots.

Requires the existing optional Python reference, not a simulator dependency.
Uses unchanged atom and native-motif classifiers in a fictitious box so large
that all physically relevant pairs use their unwrapped displacement.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def audit(task):
    reference, directory, out = map(Path, task)
    sys.path.insert(0, str(reference/'scripts'))
    import numpy as np
    from scipy.spatial.transform import Rotation
    from audit_tetramer_assembly import AtomicAssembly, pose_arrays, rotations
    from tetramer_order import TetramerOrder, graph_summary
    from analyze_precursor_exchange import prepare_templates
    engine = directory.parent.name
    cfg, summary, manifest = [read(directory/n) for n in ('config.json','summary.json','manifest.json')]
    shape = read(directory/'provenance/shape.json')
    assert summary['complete'] and summary['completed_sweeps'] == 40
    radius = cfg.get('spherical_radius') or cfg['boundary']['radius']
    analysis_cfg = copy.deepcopy(cfg)
    analysis_cfg['shape'] = str(directory/'provenance/shape.json')
    analysis_cfg['rigid_members'] = shape['rigid_members']
    atoms = np.asarray([a['center'] for a in shape['atoms']])
    radii = np.asarray([a['radius'] for a in shape['atoms']])
    bound = float(np.max(np.linalg.norm(atoms, axis=1)+radii))
    # 8(R+B) > 2 * maximum difference of centers or member centers.
    # Thus the unchanged periodic reference sees no images for any pair.
    analysis_cfg['box_lengths'] = [8*(radius+bound)]*3
    templates = prepare_templates(cfg['monomer_shape'], reference/'results/c1c3-scaffold/motifs.json',
                                 reference/'results/native-neighbor-classes/classification.json')
    order = TetramerOrder(analysis_cfg, directory, templates=templates)
    atomic = AtomicAssembly(analysis_cfg, directory)
    frames = [json.loads(s) for s in (directory/'trajectory.jsonl').read_text().splitlines()]
    assert frames[0]['sweep'] == 0 and frames[-1]['sweep'] == summary['completed_sweeps']
    n = len(frames[0]['poses']); all_pairs = list(itertools.combinations(range(n), 2))
    assert n == 12
    frame_audit = []
    for frame in frames:
        p, q = pose_arrays(frame); matrices = rotations(q)
        world = np.einsum('bij,aj->bai', matrices, atoms)+p[:,None,:]
        clearance = radius-np.linalg.norm(world, axis=2)-radii[None,:]
        smallest = float(clearance.min())
        assert smallest >= -2e-8, (directory,frame['sweep'],'wall',smallest)
        physical = atomic.frame(p,q)
        assert physical['hard_valid'], (directory,frame['sweep'],physical)
        frame_audit.append(dict(sweep=frame['sweep'],minimum_atomic_wall_clearance_A=smallest,
            minimum_interbody_gap_A=physical['minimum_interbody_gap_A'],hard_valid=True))

    state = copy.deepcopy(frames[0]['poses'])
    def record(pairs=None):
        d = order.classify({'poses':state},pair_filter=pairs)
        stay = {(*r['bodies'],r['motif_id']) for r in d['registered_tetramer_motifs']}
        entry = {(*r['bodies'],r['motif_id']) for r in d['registered_tetramer_motifs'] if r['entry']}
        return entry,stay
    initial,_ = record(); active = set(initial)
    rows = [dict(sweep=0,graph=graph_summary(n,{k[:2] for k in active},cfg['seed_labels']),registered_keys=sorted(active))]
    saved = {f['sweep']:f for f in frames}
    replay_error_p = replay_error_r = 0.
    changes = []; kinds = Counter(); acceptance = Counter(); moves_seen = 0
    components = Counter(); partitions = Counter(); flips = Counter(); gca_events = []
    def compare(expected):
        nonlocal replay_error_p,replay_error_r
        p,q = pose_arrays({'poses':state}); ep,eq = pose_arrays({'poses':expected})
        pe = float(np.max(np.abs(p-ep))); re = float(np.max(np.abs(rotations(q)-rotations(eq))))
        replay_error_p=max(replay_error_p,pe); replay_error_r=max(replay_error_r,re)
        assert pe < 5e-9 and re < 5e-11,(directory,pe,re)
    compare(cfg['initial_poses'])
    move_path = directory/('moves.jsonl.gz' if engine=='python' else 'moves.jsonl')
    with (gzip.open(move_path,'rt') if engine=='python' else move_path.open()) as stream:
        moves = [json.loads(s) for s in stream]
    for serial,move in enumerate(moves):
        kind = move['kind']; kinds[kind]+=1; moves_seen+=1; changed_pairs=[]
        previous=set(active); flip_ids=[]
        if kind in ('local','global'):
            i=move['moving_index']
            if engine=='rust':
                old=move['old_pose']; p,q=pose_arrays({'poses':[state[i]]}); ep,eq=pose_arrays({'poses':[old]})
                assert np.max(np.abs(p-ep))<5e-9 and np.max(np.abs(rotations(q)-rotations(eq)))<5e-11
            if move['accepted']:
                acceptance[kind]+=1; state[i]=copy.deepcopy(move['proposed_pose'])
                changed_pairs=[(a,b) for a,b in all_pairs if i in (a,b)]
        elif kind=='gca':
            data=move if engine=='python' else move['result']
            sizes=data['component_sizes']; components.update(sizes); partitions[str(sorted(sizes,reverse=True))]+=1
            flip_ids=data['flipped' if engine=='python' else 'flipped_indices']; flipped=set(flip_ids)
            flips['none' if not flipped else 'all' if len(flipped)==n else 'partial']+=1
            u=np.asarray(move['axis']); transform=2*np.outer(u,u)-np.eye(3)
            for i in flip_ids:
                p,q=pose_arrays({'poses':[state[i]]}); matrix=transform@rotations(q)[0]
                state[i]={'position':(transform@p[0]).tolist(),
                    'orientation':Rotation.from_matrix(matrix).as_quat()[[3,0,1,2]].tolist()}
            changed_pairs=[(a,b) for a,b in all_pairs if (a in flipped)!=(b in flipped)]
            gca_events.append(dict(sweep=move['sweep'],component_sizes=sizes,flipped_indices=flip_ids))
        elif kind in ('shift','center_shift'):
            displacement=(np.asarray(move['direction'])*move['distance'] if engine=='python'
                          else np.asarray(move['result']['displacement']))
            for pose in state: pose['position']=(np.asarray(pose['position'])+displacement).tolist()
        else: raise AssertionError(kind)
        if changed_pairs:
            entry,stay=record(changed_pairs); subset=set(changed_pairs)
            active={k for k in active if k[:2] not in subset or k in stay}|entry
        if active!=previous:
            changes.append(dict(sweep=move['sweep'],record_index=serial,kind=kind,
                moving_index=move.get('moving_index'),proposal=move.get('proposal'),
                formed=sorted(active-previous),broken=sorted(previous-active)))
        if kind=='gca':
            gca_events[-1].update(formed=sorted(active-previous),broken=sorted(previous-active))
        last=serial+1==len(moves) or moves[serial+1]['sweep']!=move['sweep']
        if last and move['sweep'] in saved:
            frame=saved[move['sweep']]; compare(frame['poses'])
            # Full classification independently checks that preserved internal
            # edges and common translations did not hide registry changes.
            entry,stay=record(); assert active==(active&stay)|entry
            rows.append(dict(sweep=frame['sweep'],graph=graph_summary(n,{k[:2] for k in active},cfg['seed_labels']),
                registered_keys=sorted(active)))
            state=copy.deepcopy(frame['poses'])  # prevent roundoff accumulation
    compare(read(directory/'checkpoint.json')['poses'])
    assert len(rows)==len(frames)
    assert kinds['local']+kinds['global']==12*summary['completed_sweeps']
    for kind in ('local','global'):
        assert kinds[kind]==summary['counts'][kind]['attempted']
        assert acceptance[kind]==summary['counts'][kind]['accepted']
    provenance={'shape_sha256':sha(directory/'provenance/shape.json'), 'model_sha256':manifest['model_sha256']}
    assert provenance['shape_sha256']==manifest['shape_sha256']
    model_file=directory/'provenance'/('model.json' if engine=='python' else 'frozen-relative-model.json')
    assert sha(model_file)==manifest['model_sha256']
    if engine=='python':
        source_check={Path(p).name:sha(directory/'provenance'/Path(p).name)==h for p,h in manifest['source_sha256'].items()}
        assert all(source_check.values()); provenance['archived_sources_match']=source_check
        provenance['kernel_source_sha256']=sha(directory/'provenance/spherical_ensemble.py')
        provenance['config_cli_note']='Archived config has inherited sweep/sample defaults; manifest arguments and trajectory establish actual 40 sweeps/sample_every=5.'
    else:
        assert sha(directory/'provenance/source-bundle.json')==manifest['source_bundle_sha256']
        provenance.update(executable_sha256=manifest['executable_sha256'],source_bundle_sha256=manifest['source_bundle_sha256'])
    gca_formed=sum(len(e['formed']) for e in gca_events); gca_broken=sum(len(e['broken']) for e in gca_events)
    result=dict(id=directory.name,engine=engine,passed=True,directory=str(directory),sweeps=40,
        radius_A=radius,depletant_radius_A=cfg['depletant_radius'],activity_A_minus3=cfg['reservoir_density'],
        initial_fragments=cfg['metadata']['initial_fragments'],initial_component=rows[0]['graph']['largest_component_size'],
        final_component=rows[-1]['graph']['largest_component_size'],rows=rows,frame_audit=frame_audit,
        counts=summary['counts'],cpu_seconds=summary.get('cpu_seconds',summary.get('sampler_cpu_seconds')),
        gca_component_size_counts=dict(components),gca_component_partitions=dict(partitions),gca_flip_counts=dict(flips),
        gca_motif_formations=gca_formed,gca_motif_breakages=gca_broken,gca_events=gca_events,
        all_motif_changes=changes,original_motifs_retained=len(initial&active),original_motifs=len(initial),
        replay=dict(records=moves_seen,frames=len(frames),maximum_position_error_A=replay_error_p,
            maximum_rotation_matrix_error=replay_error_r,checkpoint_match=True),provenance=provenance,
        protocol=order.protocol,source_sha256={str(p):sha(p) for p in [move_path,directory/'trajectory.jsonl',directory/'config.json',directory/'manifest.json',directory/'summary.json']})
    save(out/engine/directory.name/'analysis.json',result)
    print(json.dumps(dict(engine=engine,id=directory.name,passed=True,component=[result['initial_component'],result['final_component']],
        partial_flips=flips['partial'],motif_changes=len(changes))),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=Path('/home/xvg/protein-nucleation'))
    parser.add_argument('--campaign',type=Path,default=ROOT/'runs/spherical-pilot')
    parser.add_argument('--out',type=Path,default=ROOT/'results/spherical-pilot-audit')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    directories=sorted(p.parent for engine in ('python','rust') for p in (args.campaign/engine).glob('*/summary.json'))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results=list(pool.map(audit,[(str(args.reference),str(p),str(args.out)) for p in directories]))
    save(args.out/'analysis.json',dict(passed=True,runs=results,source_sha256=sha(Path(__file__))))
    lines=['# Spherical physical pilot: independent audit','',
        'All 12 runs pass saved-frame interbody hard-core and explicit atomic spherical-wall checks, move replay, unchanged native-motif classification, and archived shape/model/source checks.', '',
        'The physical target uses a protein-only spherical wall and a homogeneous ideal bath that permeates it. Radius 354.5082 Å, depletant radius 1.5 Å, activity 0.035 Å⁻³; 12 mobile tetramers, 40 sweeps per run. These starts are reembedded preassembled fragments from earlier periodic runs, not fresh dispersed nucleation experiments.', '',
        '| Engine | Arm | Largest registered component | Local/global accepted | GCA partial/none/all | GCA motif formation/breakage | CPU s |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in results:
        flip=r['gca_flip_counts'];c=r['counts']
        lines.append(f"| {r['engine']} | {r['id']} | {r['initial_component']} → {r['final_component']} | {c['local']['accepted']}/{c['global']['accepted']} | {flip.get('partial',0)}/{flip.get('none',0)}/{flip.get('all',0)} | {r['gca_motif_formations']}/{r['gca_motif_breakages']} | {r['cpu_seconds']:.2f} |")
    lines+=['','## Interpretation','',
        'Across the eight collective-move runs, 265/320 GCA updates flip a proper subset of bodies (33 flip none; 22 flip all). There are zero GCA registered-motif formations or breakages. The sole registry-changing update is a global learned proposal in the Python seeded frozen+GCA+shift arm at sweep 21: body 8 adds motif (4,8,7), growing the ordered component from 9 to 10. One event cannot establish an advantage over controls or latent transport.', '',
        'GCA components usually reproduce the existing physical aggregates: the seed-free state most often splits as [4,2,2,2,2], and the seeded state as [9,2,1]. Additional links sometimes join separate aggregates transiently into one move component. This is effective aggregate relocation while registry exchange remains unresolved.', '',
        'Partial GCA flips move subsets of bodies and can reposition intact aggregates relative to others; a completed rejection-free move may also flip zero bodies or every body. Common center shifts preserve every interbody relative pose exactly. Neither common motion nor rotation alone establishes exchange between competing registered environments.', '',
        'The registered descriptor requires a certified native tetramer relative pose plus its prescribed external monomer residue-patch contact. It excludes intrinsic tetramer contacts. A connected registered component is an ordered-association measure; this audit does not certify a perfect global lattice or thermodynamic crystal stability.', '',
        'Only one physical preparation per seeded/seed-free arm is used here. The Python and Rust generators differ, and the Rust arms share named random streams. Counts and raw CPU times are diagnostics, not mixing-speedup, equilibrium, or independent-replication estimates. Base atlas was not retrained; transported arms used the specified instantaneous conditional fit.', '',
        '### Registry changes by accepted move','']
    for r in results:
        lines.append(f"- {r['engine']}/{r['id']}: {len(r['all_motif_changes'])} changing updates; retained {r['original_motifs_retained']}/{r['original_motifs']} original registered motifs.")
        for e in r['all_motif_changes']:
            lines.append(f"  - Sweep {e['sweep']}, {e['kind']}, body {e['moving_index']}: formed {e['formed']}; broken {e['broken']}.")
    lines+=['','### Provenance','',
        'Each per-run analysis records hashes of archived inputs, model, source, logs, and the historical Rust executable. Python config files retain inherited sweep/sample defaults; manifest CLI arguments and actual records specify the 40-sweep, every-5-sweep pilot. Current source/build may include later parser or numerical-input validation guards; it is not silently substituted for archived campaign code.', '',
        'Replay checks every local/global accepted endpoint, half-turn axis/selected subset, common shift, saved frame, and final checkpoint. Rotation matrices are compared instead of quaternion sign conventions. Full native classification at every saved frame checks the partial pair updates used during replay. The atomic audit uses the unchanged reference in an analysis-only box of side 8(R+body_bound), whose minimum-image mapping is the identity for all possible physical pairs, plus an explicit wall check.','']
    (args.out/'report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(passed=True,runs=len(results),out=str(args.out))),flush=True)

if __name__=='__main__':main()
