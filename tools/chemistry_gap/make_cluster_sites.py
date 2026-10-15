"""DFT jobs for small-cluster sites along the greedy native growth path, in the crystal-site
format ({label, poses}: environment poses relative to the inserted molecule at the origin).
Sites (cluster order from greedy_path_steps.json):
  trimer:   the m6 dimer + the third molecule (m1 + m3)       [bottleneck]
  5-mer:    4-molecule cluster + the fifth molecule (m1 + m2)
  9-mer:    8-molecule cluster + the ninth (m0 m3 m5 m6; favourable in the model)
Morphometric (White Bear II, exact measures) site insertion values from many_body_solvation.json
(equal to the pair sums) are recorded as blind predictions."""
import json

import numpy as np
from scipy.spatial.transform import Rotation

D = '/vast/xvg/tetramer-mc-runs/morph-growth-20261012/b22'
P = json.load(open('/home/xvg/tetramer-mc/examples/spherical-cluster-monomer.json'))['initial_poses'][:32]
order = json.load(open(f'{D}/greedy_path_steps.json'))['cluster']
mb = {r['step']: r for r in json.load(open(f'{D}/many_body_solvation.json'))}


def arr(p):
    w, x, y, z = p['orientation']
    return np.array(p['position']), Rotation.from_quat([x, y, z, w]).as_matrix()


def rel(c, i):
    (tc, Rc), (ti, Ri) = arr(P[c]), arr(P[i])
    x, y, z, w = Rotation.from_matrix(Rc.T @ Ri).as_quat()
    q = [w, x, y, z] if w >= 0 else [-w, -x, -y, -z]
    return dict(position=[float(v) for v in Rc.T @ (ti - tc)], orientation=[float(v) for v in q])


jobs, pred = [], {}
for name, size in (('trimer site', 3), ('5-mer site', 5), ('9-mer site', 9)):
    c, env = order[size - 1], order[:size - 1]
    label = f'cluster {name} (monomer)'
    jobs.append(dict(label=label, poses=[rel(c, i) for i in env]))
    r = mb[size]
    pred[label] = dict(environment_size=len(env), added=c, morphometry_site_kT=round(r['site'], 2),
                       morphometry_pair_sum_kT=round(r['pairs'], 2), contacts=None)
steps = {s['added']: s['contacts'] for s in json.load(open(f'{D}/greedy_path_steps.json'))['steps']}
for lab, p in pred.items():
    p['contacts'] = steps[p['added']]
json.dump(jobs, open(f'{D}/env_cluster_sites.json', 'w'), indent=1)
json.dump(dict(note='blind morphometry predictions (White Bear II, eta 0.383, R 1.4, exact measures, heavy-atom '
               'monomer-shape.json); site dF = F(S+c) - F(S) - F(c); equal to the sum of native pair terms',
               predictions=pred), open(f'{D}/cluster_sites_predictions.json', 'w'), indent=1)
print(json.dumps(pred, indent=1))
