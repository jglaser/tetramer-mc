#!/usr/bin/env python3
"""Read-only geometry diagnostic for the completed mobile-posterior trap.

Output is a snapshot-conditioned diagnostic, not a population or normalizer.
The AB gauge changes only a common proper isometry and body labels.
"""
from __future__ import annotations
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from prepare_smc_normalizer_atlas import Density

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays(pose):
    return np.asarray(pose['position']), Rotation.from_quat(np.asarray(pose['orientation'])[[1, 2, 3, 0]]).as_matrix()


def pose(position, rotation):
    return dict(position=position.tolist(), orientation=Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]].tolist())


def relative(value, anchor):
    p, r = arrays(value)
    a, s = arrays(anchor)
    return pose(s.T@(p-a), s.T@r)


def absolute(value, anchor):
    p, r = arrays(value)
    a, s = arrays(anchor)
    return pose(a+s@p, s@r)


def differences(value, target, members):
    p, r = arrays(value)
    a, s = arrays(target)
    errors = np.linalg.norm(members@r.T+p-members@s.T-a, axis=1)
    return dict(member_rms_A=float(np.sqrt(np.mean(errors**2))),
        maximum_member_error_A=float(max(errors)), center_error_A=float(np.linalg.norm(p-a)),
        rotation_error_degrees=float(np.degrees(Rotation.from_matrix(s.T@r).magnitude())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    assert not output.exists(), 'Refusing to overwrite a geometry diagnostic'
    campaign = ROOT/'runs/mobile-posterior-pilot-12x2000-20260921'
    directory = campaign/'runs/mobile3-dispersed-r01-c09'
    reference = ROOT/'runs/mobile-posterior-reference-recovery-20260921/reference'
    audit_path = ROOT/'runs/mobile-posterior-pilot-assessment-v2-20260921/runs/mobile3-dispersed-r01-c09/analysis.json'
    ab_path = ROOT/'runs/mobile-posterior-pilot-preparation-20260921/provenance/AB-source-config.json'
    paths = dict(trajectory=directory/'trajectory.jsonl', moves=directory/'moves.jsonl',
        config=directory/'config.json', model=directory/'provenance/frozen-relative-model.json',
        shape=directory/'provenance/shape.json', monomer=campaign/'provenance/monomer-shape.json',
        motifs=campaign/'provenance/native-pair-motifs.json', original_ab=ab_path,
        source_native_model=ROOT/'runs/mobile-posterior-pilot-preparation-20260921/provenance/model-native.json',
        audit=audit_path, campaign_manifest=campaign/'manifest.json')
    audit, config, shape, ab = [read(paths[key]) for key in ('audit', 'config', 'shape', 'original_ab')]
    assert audit['passed'] and audit['seed'] == config['seed'] == 115506055
    assert sha(paths['trajectory']) == audit['source_sha256']['trajectory.jsonl']
    assert sha(paths['moves']) == audit['source_sha256']['moves.jsonl']
    assert sha(paths['shape']) == read(campaign/'manifest.json')['shape_sha256']
    from analyze_mobile_posterior_pilot import validate_reference
    validated_reference = validate_reference(campaign, read(campaign/'manifest.json'), reference)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(reference/'scripts'))
    from audit_tetramer_assembly import AtomicAssembly
    from analyze_precursor_exchange import prepare_templates
    from tetramer_order import TetramerOrder
    observer_cfg = copy.deepcopy(config)
    observer_cfg.update(shape=str(paths['shape']), rigid_members=shape['rigid_members'], box_lengths=[2000.]*3)
    templates = prepare_templates(paths['monomer'], reference/'results/c1c3-scaffold/motifs.json',
        reference/'results/native-neighbor-classes/classification.json')
    order, atomic = TetramerOrder(observer_cfg, directory, templates), AtomicAssembly(observer_cfg, directory)
    model, native_model = read(paths['model']), read(paths['source_native_model'])
    density = Density(model)
    frames = [json.loads(line) for line in paths['trajectory'].open()]
    assert [f['sweep'] for f in frames] == list(range(2001))
    members = np.asarray([m['position'] for m in shape['rigid_members']])
    atoms = np.asarray([a['center'] for a in shape['atoms']])
    radii = np.asarray([a['radius'] for a in shape['atoms']])
    motifs = read(paths['motifs'])['motifs']
    motif_poses = [dict(position=m['relative_position'], orientation=m['relative_orientation']) for m in motifs]
    identity = dict(position=[0., 0., 0.], orientation=[1., 0., 0., 0.])
    original_native_relative_a = relative(identity, ab['fixed_poses'][0])
    assert differences(original_native_relative_a, motif_poses[4], members)['maximum_member_error_A'] < 1e-10
    assert differences(relative(ab['fixed_poses'][1], ab['fixed_poses'][0]), motif_poses[3], members)['maximum_member_error_A'] < 1e-10
    assert differences(relative(ab['fixed_poses'][0], ab['fixed_poses'][1]), motif_poses[8], members)['maximum_member_error_A'] < 1e-10

    def gap(value, anchor):
        p, r = arrays(relative(value, anchor))
        return atomic.exact_gap(p, r)

    def nearest(value):
        ds = [differences(value, target, members) for target in motif_poses]
        index = min(range(len(ds)), key=lambda i: ds[i]['member_rms_A'])
        return dict(motif_id=motifs[index]['id'], **ds[index])

    def wall(value):
        p, r = arrays(value)
        return dict(atom_wall_clearance_A=float(np.min(config['boundary']['radius']-np.linalg.norm(atoms@r.T+p, axis=1)-radii)),
            all_orientation_center_displacement_margin_A=float(config['boundary']['radius']-np.linalg.norm(p)-atomic.bound))

    snapshots = []
    for sweep in (1, 41, 400, 2000):
        frame, observed = frames[sweep], order.classify(frames[sweep])
        poses = frame['poses']
        # Map actual body2 exactly to original A; body1 then lies near original B.
        ap, ar = arrays(ab['fixed_poses'][0])
        p2, r2 = arrays(poses[2])
        rotation, translation = ar@r2.T, ap-ar@r2.T@p2
        transformed = [absolute(relative(p, poses[2]), ab['fixed_poses'][0]) for p in poses]
        roundtrip = []
        for source, mapped in zip(poses, transformed):
            p, r = arrays(mapped)
            roundtrip.append(differences(pose(rotation.T@(p-translation), rotation.T@r), source, members))
        assert max(v['maximum_member_error_A'] for v in roundtrip) < 1e-10
        mapped_center = rotation@np.zeros(3)+translation
        gaussian = {}
        for anchor_index in (1, 2):
            value = relative(poses[0], poses[anchor_index])
            total, norms, logs = density.evaluate([value])
            indices = np.argsort(logs[0])[::-1][:5]
            gaussian[str(anchor_index)] = dict(log_gaussian=float(total[0]),
                top_components=[dict(index=int(i), mahalanobis_norm=float(norms[0, i]),
                    log_weighted_density=float(logs[0, i]), posterior_responsibility=float(np.exp(logs[0, i]-total[0])),
                    native_source_component=native_model['components'][i] if i < 28 else None) for i in indices])
        native = absolute(original_native_relative_a, poses[2])
        original_q = differences(transformed[0], identity, members)
        q = max(original_q['maximum_member_error_A']/2., original_q['rotation_error_degrees']/15.)
        snapshots.append(dict(sweep=sweep, original_poses=poses, body2_frame=[relative(p, poses[2]) for p in poses],
            ab_gauge=dict(poses=transformed, common_rotation=rotation.tolist(), common_translation=translation.tolist(),
                inverse_rotation=rotation.T.tolist(), inverse_translation=(-rotation.T@translation).tolist(),
                mapped_physical_wall_center=mapped_center.tolist(), wall_radius=config['boundary']['radius'],
                body_assignment=dict(A=2, B=1, mobile=0),
                roundtrip_maximum_member_error_A=max(v['maximum_member_error_A'] for v in roundtrip),
                scaffold_B_deviation=differences(transformed[1], ab['fixed_poses'][1], members),
                original_capture_center=ab['capture_center'], original_capture_radius_A=ab['capture_radius'],
                trapped_distance_from_original_capture_A=float(np.linalg.norm(np.asarray(transformed[0]['position'])-ab['capture_center'])),
                trapped_original_q=q),
            pair_gaps_A={f'{i}-{j}':gap(poses[i],poses[j]) for i,j in ((0,1),(0,2),(1,2))},
            nearest_motifs={f'{i}-{j}':nearest(relative(poses[j],poses[i])) for i,j in ((0,1),(0,2),(1,2))},
            registered_motifs=observed['registered_tetramer_motifs'],
            native_monomer_bonds=observed['native_entry_monomer_edges'],
            trapped_proposal_density=gaussian, trapped_wall=wall(poses[0]),
            original_native_alternative=dict(pose=native, body2_frame=original_native_relative_a,
                gaps_to_body1_and_2_A=[gap(native,poses[j]) for j in (1,2)], **wall(native))))
    final = snapshots[-1]
    assert final['ab_gauge']['trapped_distance_from_original_capture_A'] > ab['capture_radius']
    assert min(final['original_native_alternative']['gaps_to_body1_and_2_A']) >= 0
    assert final['original_native_alternative']['atom_wall_clearance_A'] > 0
    assert [(r['bodies'],r['motif_id']) for r in final['registered_motifs']] == [([1,2],8)]
    diversity = {}
    for moving, anchor in ((0,2),(1,2)):
        baseline = relative(frames[400]['poses'][moving],frames[400]['poses'][anchor])
        values = [relative(f['poses'][moving],f['poses'][anchor]) for f in frames[400:]]
        ds = [differences(value,baseline,members) for value in values]
        diversity[f'{moving}-relative-{anchor}'] = dict(inclusive_sweeps=[400,2000], samples=len(values),
            unique_translations_rounded_to_1e_minus7_A=len({tuple(np.round(v['position'],7)) for v in values}),
            maximum_member_rms_excursion_A=max(d['member_rms_A'] for d in ds),
            maximum_rotation_excursion_degrees=max(d['rotation_error_degrees'] for d in ds),
            interpretation='Common GCA isometries and center shifts removed; counts are observed distinct rounded states, not independent samples.')
    result = dict(schema='mobile-competing-geometry-v1', run_id=directory.name,
        source={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()},
        implementation_sha256={str(Path(__file__).resolve()):sha(__file__),
            str(ROOT/'tools/prepare_smc_normalizer_atlas.py'):sha(ROOT/'tools/prepare_smc_normalizer_atlas.py'),
            str(ROOT/'tools/analyze_mobile_posterior_pilot.py'):sha(ROOT/'tools/analyze_mobile_posterior_pilot.py')},
        reference_validation=validated_reference, body_bound_A=atomic.bound,
        snapshots=snapshots, postburn_relative_diversity=diversity,
        conclusions=[
            'Mobile body0 contacts only body2 nonspecifically. Bodies1 and2 form the original AB scaffold with the body labels exchanged.',
            'The exact observed scaffold differs slightly from the ideal original AB scaffold; preserving it defines a new snapshot-conditioned weight calculation.',
            'The trapped site is outside the original capture18 domain. Earlier AB regional weights therefore do not compare it against native binding.',
            'The trapped pose is supported mainly by explicit broad defensive Gaussian components, not a fitted contact basin.',
            'The original native alternative remains geometrically possible on this observed scaffold; geometry alone gives no statistical weight.',
            'Wall margins certify only neighborhoods explicitly covered by their displacement bound. They do not justify removing the spherical wall for global assembly or arbitrary native sites.'
        ])
    output.mkdir(parents=True)
    (output/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (output/'diagnose_mobile_competing_geometry.py').write_text(Path(__file__).read_text())
    # A convenient exact fixed-snapshot record, intentionally not a sampler config.
    fixed = dict(schema='mobile-fixed-snapshot-geometry-v1', source=result['source'], sweep=2000,
        body_assignment=dict(A=2,B=1,mobile=0),
        original_global=dict(poses=final['original_poses'],wall_center=[0.,0.,0.],wall_radius=config['boundary']['radius']),
        ab_gauge=final['ab_gauge'], trapped_pose=final['ab_gauge']['poses'][0],
        physical_fixed_neighbors=[final['ab_gauge']['poses'][2],final['ab_gauge']['poses'][1]],
        native_candidate=identity, scope='Exact snapshot geometry and common proper isometry only; no fitted model, capture region, statistical weight or campaign is implied.')
    (output/'fixed-snapshot.json').write_text(json.dumps(fixed,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output), final=final['ab_gauge'], diversity=diversity),indent=2))


if __name__ == '__main__':
    main()
