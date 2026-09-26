#!/usr/bin/env python3
"""Post-hoc native recall of shape-only contact poses (read-only diagnostic).

Native motifs are read only here, after candidates are frozen. Nothing is
selected, reweighted or refit. For each pose: complete native entry
(NativeContactRegions.classify_pair) and the body scaled error
min_motif max(max member error/2 A, rotation error/15 deg); <=1 is body entry.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from native_contact_regions import NativeContactRegions, pose_arrays

DEFINITION = ROOT/'runs/mobile-native-region-definition-20260921/definition.json'


def recall(poses, classifier):
    identity = dict(position=[0., 0., 0.], orientation=[1., 0., 0., 0.])
    rows = []
    for pose in poses:
        t, R = pose_arrays(pose)
        observed = classifier.member_positions@R.T+t
        best = None
        for k, motif in enumerate(classifier.motifs):
            expected = classifier.member_positions@classifier.motif_rotations[k].T+classifier.motif_positions[k]
            err = float(np.linalg.norm(observed-expected, axis=1).max())
            ang = float(np.degrees(Rotation.from_matrix(classifier.motif_rotations[k].T@R).magnitude()))
            scaled = max(err/2., ang/15.)
            if best is None or scaled < best['scaled']:
                best = dict(motif_id=motif['id'], scaled=scaled, member_error_A=err, angle_deg=ang)
        matches = classifier.classify_pair(identity, pose)
        rows.append(dict(complete_native_entry=bool(matches),
                         matched_motif_ids=[m['motif_id'] for m in matches], nearest=best))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('poses', help='JSON list of poses, or selected.json/discovery.json')
    p.add_argument('--stage', default='optimized_pose', help='slot key when reading discovery.json')
    a = p.parse_args()
    data = json.loads(Path(a.poses).read_text())
    if isinstance(data, dict) and 'slots' in data:
        poses = [s[a.stage] for s in data['slots']]
    else:
        poses = [d['pose'] if 'pose' in d else d for d in data]
    rows = recall(poses, NativeContactRegions(DEFINITION))
    for i, r in enumerate(rows):
        n = r['nearest']
        print(f"{i:3d} native={int(r['complete_native_entry'])} motif={n['motif_id']:2d} "
              f"scaled={n['scaled']:.2f} err={n['member_error_A']:.1f}A ang={n['angle_deg']:.1f}")
    motifs = sorted({m for r in rows for m in r['matched_motif_ids']})
    print(f'complete native entries {sum(r["complete_native_entry"] for r in rows)}/{len(rows)}; '
          f'body entry {sum(r["nearest"]["scaled"] <= 1 for r in rows)}; distinct motifs {motifs}')


if __name__ == '__main__':
    main()
