#!/usr/bin/env python3
"""Spherical hard sphere-union MC: learned moves, implicit GCA, common shifts.

The homogeneous ideal bath permeates the protein-only wall. Production uses
exact union predicates and conditional Poisson gates, never estimated energies.
Auxiliary mode uses a normalized Gaussian law over the fixed-K mixture means;
it transports the same normal residual and evaluates the NEW reverse model.
The deterministic fitter is history-free; this is not accumulating adaptation.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp

REFERENCE = Path(os.environ.get('PROTEIN_NUCLEATION_ROOT', '/home/xvg/protein-nucleation'))
sys.path.insert(0, str(REFERENCE/'scripts'))
from contact_triangle_cloud import SphereUnion, BodyCloudEnvironment
from endpoint_cloud_gate import EndpointCloudGate
from moment_pose_transport import FrozenMomentMixture, NumericalPoseProposalError
from contact_metric_proposal import cayley_matrix
from run_learned_tetramer_pilot import pose, serial, sha, resolve_path


def save(path, value):
    Path(path).write_text(json.dumps(serial(value), indent=2, allow_nan=False)+'\n')


class SphericalRelativeProposal:
    """Unconditioned Gaussian mixture + cube/Haar floor, wall rejection outside.

    The declared enclosing support cube has no periodic images or retries.
    The runner uses radius R+body_bound, valid for arbitrary body-frame origins.
    """
    def __init__(self, data, wall_radius, uniform_weight=.1, support_radius=None):
        self.data = copy.deepcopy(data)
        self.radius = float(wall_radius)
        self.uniform_weight = float(uniform_weight)
        self.support_radius = self.radius if support_radius is None else float(support_radius)
        if not math.isfinite(self.support_radius) or self.support_radius < self.radius:
            raise ValueError('Invalid enclosing support cube')
        if not math.isfinite(self.radius) or self.radius <= 0 or not 0 < uniform_weight <= 1:
            raise ValueError('Invalid sphere or support floor')
        self.mixture = FrozenMomentMixture(data)
        self.log_uniform = math.log(uniform_weight)-3*math.log(2*self.support_radius)
        self.log_learned = math.log1p(-uniform_weight) if uniform_weight < 1 else -math.inf

    def logpdf(self, t, R, anchor_t, anchor_R):
        t = np.asarray(t)
        if np.any(t < -self.support_radius) or np.any(t >= self.support_radius):
            return -math.inf
        return float(logsumexp([self.log_uniform, self.log_learned+
            self.mixture.logpdf(anchor_R.T@(t-anchor_t), anchor_R.T@R)]))

    def sample(self, rng, positions, rotations, moving_index):
        i = int(moving_index)
        selected = int(rng.integers(len(positions)-1))
        j = selected + (selected >= i)
        at, aR = positions[j], rotations[j]
        out = dict(anchor_index=int(j), branch='uniform', null=False)
        if rng.random() < self.uniform_weight:
            t = rng.uniform(-self.support_radius, self.support_radius, 3)
            R = Rotation.random(random_state=rng).as_matrix()
        else:
            out['branch'] = 'learned'
            try:
                t0, R0, k, _ = self.mixture.sample(rng)
                t, R = at+aR@t0, aR@R0
                out['component_index'] = int(k)
            except NumericalPoseProposalError:
                return dict(out, null=True, reason='numerical_null')
            if np.any(t < -self.support_radius) or np.any(t >= self.support_radius):
                return dict(out, null=True, reason='outside_support_cube')
        return dict(out, position=t, rotation=R,
            forward_log_density=self.logpdf(t, R, at, aR),
            reverse_log_density=self.logpdf(positions[i], rotations[i], at, aR))


class AuxiliaryMeans:
    """Generative fixed-K normalized conditional Gaussian mixture-mean law.

    Ordered current relative poses assign to their nearest frozen Mahalanobis
    chart if residual norm <= cutoff. Bounded, shrunk residual averages define
    F(X); s=noise/sqrt(shrinkage+count) defines conditional uncertainty. In
    whitened mean coordinates a_k=f_k(X)+s_k(X)*eta_k, eta iid N(0,1).
    Component labels, covariances and weights do not change. No sample archive.
    """
    def __init__(self, proposal, gain=.25, noise=.1, cutoff=6., clip=3., shrinkage=1.):
        self.base = proposal
        self.gain, self.noise, self.cutoff, self.clip, self.shrinkage = map(
            float, (gain, noise, cutoff, clip, shrinkage))
        if not all(math.isfinite(v) for v in (gain, noise, cutoff, clip, shrinkage)) or min(noise, cutoff, clip, shrinkage) <= 0 or gain < 0:
            raise ValueError('Invalid auxiliary conditional settings')
        if any(df is not None for df in proposal.mixture.dfs):
            raise ValueError('Auxiliary prototype requires Gaussian components')
        self.count = len(proposal.mixture.components)
        self.lower = np.array([c.cholesky for c in proposal.mixture.components])
        self.means = np.array(proposal.mixture.means)
        self.inverse = np.linalg.inv(self.lower)

    def fit(self, positions, rotations):
        sums = np.zeros((self.count, 6)); counts = np.zeros(self.count, dtype=int)
        for i in range(len(positions)):
            for j in range(len(positions)):
                if i == j:
                    continue
                t = rotations[j].T@(positions[i]-positions[j])
                R = rotations[j].T@rotations[i]
                best, residual, minimum = None, None, math.inf
                for k in range(self.count):
                    try:
                        v, _ = self.base.mixture.encode(t, R, k)
                    except ValueError as error:
                        if 'pi rotation' not in str(error): raise
                        continue
                    r = self.inverse[k]@(v-self.means[k])
                    d2 = float(r@r)
                    if d2 < minimum:
                        best, residual, minimum = k, r, d2
                if best is not None and minimum <= self.cutoff**2:
                    counts[best] += 1
                    sums[best] += residual*min(1., self.clip/max(math.sqrt(minimum), 1e-300))
        f = self.gain*sums/(self.shrinkage+counts[:, None])
        s = self.noise/np.sqrt(self.shrinkage+counts)
        if not np.all(np.isfinite(f)) or not np.all(np.isfinite(s)) or np.any(s <= 0):
            raise FloatingPointError('Unrepresentable auxiliary conditional fit')
        return f, s, counts

    def coordinates(self, positions, rotations, eta):
        eta = np.asarray(eta)
        if eta.shape != (self.count, 6) or not np.all(np.isfinite(eta)):
            raise ValueError('Invalid Gaussian residual')
        f, s, _ = self.fit(positions, rotations)
        return f+s[:, None]*eta

    def model(self, positions, rotations, eta):
        a = self.coordinates(positions, rotations, eta)
        data = copy.deepcopy(self.base.data)
        data['means'] = (self.means+np.einsum('kij,kj->ki', self.lower, a)).tolist()
        return SphericalRelativeProposal(data, self.base.radius, self.base.uniform_weight, self.base.support_radius)

    def log_density(self, positions, rotations, a):
        """Density in whitened mean coordinates, not physical pose coordinates."""
        f, s, _ = self.fit(positions, rotations)
        eta = (np.asarray(a)-f)/s[:, None]
        return float(-.5*np.sum(eta**2)-3*self.count*math.log(2*math.pi)-6*np.log(s).sum())


class SphericalSystem:
    def __init__(self, body, radii, wall_radius, rd, z):
        self.body = np.asarray(body, dtype=float)
        self.radii = np.asarray(radii, dtype=float)
        self.radius, self.rd, self.z = map(float, (wall_radius, rd, z))
        if not all(math.isfinite(v) for v in (self.radius,self.rd,self.z)) or self.radius <= 0 or min(self.rd,self.z) < 0:
            raise ValueError('Invalid wall or bath')
        self.hard = SphereUnion(self.body, self.radii)
        self.exclusion = SphereUnion(self.body, self.radii+self.rd)
        self.bound = float(np.max(np.linalg.norm(self.body, axis=1)+self.radii))
        if np.max(self.radii) >= self.radius:
            raise ValueError('Wall too small')
        self.ebound = self.bound+self.rd+256*np.finfo(float).eps*(1+self.bound+self.rd+self.radius)

    def inside(self, t, R):
        return bool(np.all(np.linalg.norm(self.body@R.T+t, axis=1)+self.radii <= self.radius))

    def overlap(self, t, R, u, Q):
        if np.linalg.norm(t-u) > 2*self.bound+1e-10:
            return False
        centers = (self.body@R.T+t-u)@Q
        if self.hard.groups is not None:
            return any(np.any(tree.query(centers, workers=1)[0] < self.radii+r)
                       for r, tree in self.hard.groups)
        for p, r in zip(centers, self.radii):
            ids = self.hard.tree.query_ball_point(p, r+self.hard.maximum_radius)
            if ids and np.any(np.sum((self.body[ids]-p)**2,axis=1) < (self.radii[ids]+r)**2):
                return True
        return False

    def valid(self, positions, rotations):
        return (all(self.inside(t,R) for t,R in zip(positions,rotations)) and
                not any(self.overlap(positions[i],rotations[i],positions[j],rotations[j])
                    for i in range(len(positions)) for j in range(i)))

    def chord(self, positions, rotations, direction):
        direction = np.asarray(direction, dtype=float)
        if not np.all(np.isfinite(direction)) or np.linalg.norm(direction) == 0:
            raise ValueError('Invalid direction')
        direction = direction/np.linalg.norm(direction)
        xyz = np.array([self.body@R.T+t for t,R in zip(positions,rotations)])
        axial = xyz@direction
        transverse2 = np.maximum(0., np.sum(xyz*xyz,axis=2)-axial*axial)
        discriminant = (self.radius-self.radii)**2-transverse2
        if np.any(discriminant < -1e-9): raise ValueError('Infeasible chord')
        root = np.sqrt(np.maximum(0.,discriminant))
        return float(np.max(-axial-root)), float(np.min(-axial+root))

    def shift(self, rng, positions, rotations, direction=None):
        start = time.process_time()
        u = rng.normal(size=3) if direction is None else np.asarray(direction,dtype=float)
        u = u/np.linalg.norm(u)
        lo, hi = self.chord(positions,rotations,u)
        if lo > hi or lo > 1e-9 or hi < -1e-9: raise ValueError('Current state not on chord')
        amount = float(rng.uniform(lo,hi))
        result = positions+amount*u
        if not all(self.inside(t,R) for t,R in zip(result,rotations)):
            raise FloatingPointError('Shift wall roundoff; no silent acceptance veto')
        return result, rotations.copy(), dict(kind='shift',accepted=True,direction=u,
            distance=amount,chord=[lo,hi],cpu_seconds=time.process_time()-start)

    def gca(self, rng, positions, rotations, axis=None):
        start = time.process_time(); n = len(positions)
        u = rng.normal(size=3) if axis is None else np.asarray(axis,dtype=float)
        u = u/np.linalg.norm(u)
        T = 2*np.outer(u,u)-np.eye(3)
        shadow_t, shadow_R = positions@T.T, T@rotations
        parent = list(range(n))
        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]; i = parent[i]
            return i
        def join(ids):
            if len(ids):
                a = root(int(ids[0]))
                for j in ids[1:]: parent[root(int(j))] = a
        for i in range(n):
            for j in range(i):
                if (self.overlap(positions[i],rotations[i],shadow_t[j],shadow_R[j]) or
                        self.overlap(positions[j],rotations[j],shadow_t[i],shadow_R[i])):
                    join([i,j])
        stats = dict(kind='gca',accepted=True,axis=u,poisson_probes=0,owned_probes=0,
                     hyperedges=0,pair_envelopes=0,skipped_envelopes=0)
        B = self.ebound
        # One PPP on the union of pair lenses. First bounding-owner pair makes
        # the independently sampled cylinder envelopes a disjoint thinning.
        if self.z > 0:
            for i in range(n):
                for j in range(i+1,n):
                    d = positions[j]-positions[i]; distance = float(np.linalg.norm(d))
                    if distance >= 2*B: continue
                    if len({root(k) for k in range(n)}) == 1:
                        stats['skipped_envelopes'] += 1; continue
                    stats['pair_envelopes'] += 1
                    axis0 = d/distance if distance else np.array([1.,0.,0.])
                    ref = np.array([1.,0.,0.]) if abs(axis0[0]) < .8 else np.array([0.,1.,0.])
                    v = np.cross(axis0,ref); v /= np.linalg.norm(v)
                    w = np.cross(axis0,v)
                    half = B-distance/2; transverse2 = B*B-distance*distance/4
                    volume = 2*half*math.pi*transverse2
                    count = int(rng.poisson(self.z*volume)); stats['poisson_probes'] += count
                    for begin in range(0,count,8192):
                        m = min(8192,count-begin)
                        along = rng.uniform(-half,half,m)
                        radius = np.sqrt(rng.random(m)*transverse2)
                        theta = rng.uniform(0,2*math.pi,m)
                        points = ((positions[i]+positions[j])/2+along[:,None]*axis0+
                                  (radius*np.cos(theta))[:,None]*v+(radius*np.sin(theta))[:,None]*w)
                        bounds = np.sum((points[:,None,:]-positions[None,:,:])**2,axis=2) <= B*B
                        own = bounds[:,i]&bounds[:,j]&(~np.any(bounds[:,:i],axis=1))&(~np.any(bounds[:,i+1:j],axis=1))
                        points = points[own]; bounds = bounds[own]
                        stats['owned_probes'] += len(points)
                        if not len(points): continue
                        covered = np.zeros(len(points),dtype=bool)
                        for k in range(n):
                            ids = np.flatnonzero(~covered & (np.sum((points-shadow_t[k])**2,axis=1)<=B*B))
                            covered[ids] |= self.exclusion.contains((points[ids]-shadow_t[k])@shadow_R[k])
                        points, bounds = points[~covered],bounds[~covered]
                        owners = np.zeros((len(points),n),dtype=bool)
                        for k in range(n):
                            ids = np.flatnonzero(bounds[:,k])
                            owners[ids,k] = self.exclusion.contains((points[ids]-positions[k])@rotations[k])
                        for row in owners:
                            ids = np.flatnonzero(row)
                            if len(ids) >= 2: stats['hyperedges'] += 1; join(ids)
        components = {}
        for i in range(n): components.setdefault(root(i),[]).append(i)
        coins = rng.random(n) < .5
        result_t, result_R = positions.copy(), rotations.copy()
        flipped = []
        for ids in components.values():
            if coins[ids[0]]:
                result_t[ids],result_R[ids] = shadow_t[ids],shadow_R[ids]; flipped += ids
        if not self.valid(result_t,result_R):
            raise FloatingPointError('GCA hard/wall failure; no endpoint MH veto')
        stats.update(component_sizes=[len(ids) for ids in components.values()],flipped=sorted(flipped),
                     cpu_seconds=time.process_time()-start)
        return result_t,result_R,stats

    def environment(self, positions, rotations, i, new_t):
        ids = [j for j in range(len(positions)) if j != i and
               min(np.linalg.norm(positions[j]-positions[i]),np.linalg.norm(positions[j]-new_t)) <= 2*self.ebound]
        fixed = np.concatenate([self.body@rotations[j].T+positions[j] for j in ids]) if ids else np.empty((0,3))
        return BodyCloudEnvironment(self.body,self.radii,fixed,np.tile(self.radii,len(ids)),self.rd)

    def step(self, rng, positions, rotations, i, proposal=None, auxiliary=None, eta=None,
             kind='local', translation_std=.2, angle_std_deg=1.):
        start = time.process_time()
        out = dict(kind=kind,accepted=False,moving_index=int(i),log_reverse_forward=0.)
        old_t,old_R = positions[i],rotations[i]
        if kind == 'local':
            t = old_t+translation_std*rng.normal(size=3)
            R = cayley_matrix(math.radians(angle_std_deg)/2*rng.normal(size=3))@old_R
        elif kind == 'global':
            forward = auxiliary.model(positions,rotations,eta) if auxiliary is not None else proposal
            if forward is None:
                t = rng.uniform(-self.radius-self.bound,self.radius+self.bound,3); R = Rotation.random(random_state=rng).as_matrix()
            else:
                draw = forward.sample(rng,positions,rotations,i)
                out['proposal'] = {k:v for k,v in draw.items() if k not in ('position','rotation')}
                if draw['null']: return positions,rotations,dict(out,cpu_seconds=time.process_time()-start)
                t,R = draw['position'],draw['rotation']
                reverse_log = draw['reverse_log_density']
                if auxiliary is not None:
                    xt,xR = positions.copy(),rotations.copy(); xt[i],xR[i] = t,R
                    reverse = auxiliary.model(xt,xR,eta)
                    j = draw['anchor_index']
                    reverse_log = reverse.logpdf(old_t,old_R,positions[j],rotations[j])
                    out['transported_reverse_log_density'] = reverse_log
                out['log_reverse_forward'] = reverse_log-draw['forward_log_density']
        else: raise ValueError('Unknown single-body kernel')
        out['proposed_pose'] = pose(t,R)
        if not self.inside(t,R):
            return positions,rotations,dict(out,wall_rejected=True,cpu_seconds=time.process_time()-start)
        env = self.environment(positions,rotations,i,t)
        if not env.hard_valid(t,R):
            return positions,rotations,dict(out,hard_rejected=True,cpu_seconds=time.process_time()-start)
        sample = EndpointCloudGate(env).sample(rng,16*self.z if self.z else 1.,self.z,old_t,old_R,t,R)
        loga = min(0.,sample['log_weight']+out['log_reverse_forward'])
        accepted = math.log(max(float(rng.random()),np.finfo(float).tiny)) < loga
        out.update(accepted=accepted,gate=sample,log_acceptance=loga)
        if accepted:
            positions,rotations = positions.copy(),rotations.copy(); positions[i],rotations[i] = t,R
        return positions,rotations,dict(out,cpu_seconds=time.process_time()-start)


def run(args):
    config = json.loads(args.config.read_text())
    if config.get('fixed_body_indices'): raise ValueError('All-mobile prototype only')
    path = resolve_path(config['shape'],args.config.parent)
    shape = json.loads(path.read_text())
    boundary = config.get('boundary', {})
    radius = float(config.get('spherical_radius', boundary.get('radius') if isinstance(boundary, dict) else None))
    system = SphericalSystem([a['center'] for a in shape['atoms']],[a['radius'] for a in shape['atoms']],
                            radius,config['depletant_radius'],config['reservoir_density'])
    positions = np.array([p['position'] for p in config['initial_poses']],dtype=float)
    rotations = Rotation.from_quat(np.array([p['orientation'] for p in config['initial_poses']])[:,[1,2,3,0]]).as_matrix()
    if len(positions) < 2 or not system.valid(positions,rotations): raise ValueError('Invalid spherical initial state')
    proposal = None
    if args.model:
        data = json.loads(args.model.read_text())
        if data['shape_sha256'] != sha(path) or data['coordinate_convention'] != 'anchor-body-relative':
            raise ValueError('Model geometry or coordinate convention mismatch')
        proposal = SphericalRelativeProposal(data,radius,config.get('learned_uniform_weight',.1),radius+system.bound)
    if args.auxiliary and proposal is None: raise ValueError('Auxiliary transport needs a model')
    auxiliary = AuxiliaryMeans(proposal,**config.get('auxiliary_transport',{})) if args.auxiliary else None
    rng = np.random.default_rng(config['seed'])
    eta = rng.normal(size=(auxiliary.count,6)) if auxiliary else None
    if args.out.exists() and any(args.out.iterdir()): raise ValueError('Output must be empty')
    args.out.mkdir(parents=True,exist_ok=True)
    archive = args.out/'provenance'; archive.mkdir()
    shutil.copy2(args.config,archive/'input-config.json'); shutil.copy2(path,archive/'shape.json')
    if args.model: shutil.copy2(args.model,archive/'model.json')
    hashes = {}
    for module in list(sys.modules.values()):
        filename = getattr(module,'__file__',None)
        if filename and str(filename).endswith('.py'):
            f = Path(filename).resolve()
            if f.parent in (Path(__file__).resolve().parent,REFERENCE/'scripts'):
                hashes[str(f)] = sha(f); shutil.copy2(f,archive/f.name)
    save(args.out/'config.json',dict(config,boundary='spherical',box_lengths=[2*radius]*3,
        sweeps=args.sweeps,sample_every=args.sample_every,gca_every=args.gca_every,
        shift_every=args.shift_every,auxiliary_enabled=bool(args.auxiliary)))
    save(args.out/'manifest.json',dict(source_sha256=hashes,shape_sha256=sha(path),
        model_sha256=sha(args.model) if args.model else None,arguments={k:str(v) for k,v in vars(args).items()},
        target='hard protein wall; homogeneous ideal bath permeates wall; exp(-z union volume)',
        auxiliary='normalized fixed-K conditional means; latent refresh per sweep; new-model reverse' if auxiliary else 'frozen',
        schedule='N single-body attempts (uniform random permutation), then GCA and/or shift every configured sweep; fixed alternation invariant, not necessarily reversible'))
    start = time.process_time(); counts = {}; completed = 0
    def frame(sweep):
        return dict(sweep=sweep,poses=[pose(t,R) for t,R in zip(positions,rotations)],
                    boundary='spherical',spherical_radius=radius,counts=counts,
                    auxiliary_eta=eta,sampler_cpu_seconds=time.process_time()-start)
    def record(row):
        k = row['kind']; c = counts.setdefault(k,dict(attempted=0,accepted=0))
        c['attempted'] += 1; c['accepted'] += int(row['accepted'])
        moves.write(json.dumps(serial(dict(row,sweep=sweep)),allow_nan=False)+'\n')
    with (args.out/'trajectory.jsonl').open('w') as trajectory,gzip.open(args.out/'moves.jsonl.gz','wt') as moves:
        trajectory.write(json.dumps(serial(frame(0)),allow_nan=False)+'\n')
        for sweep in range(1,args.sweeps+1):
            if auxiliary: eta = rng.normal(size=(auxiliary.count,6))
            for i in rng.permutation(len(positions)):
                kind = 'global' if rng.random() < config.get('global_probability',.5) else 'local'
                positions,rotations,row = system.step(rng,positions,rotations,int(i),proposal,auxiliary,eta,kind,
                    config.get('local_translation_std_A',.2),config.get('local_small_angle_std_degrees',1.))
                record(row)
            if args.gca_every and sweep % args.gca_every == 0:
                positions,rotations,row = system.gca(rng,positions,rotations); record(row)
            if args.shift_every and sweep % args.shift_every == 0:
                positions,rotations,row = system.shift(rng,positions,rotations); record(row)
            completed = sweep
            if sweep % args.sample_every == 0 or sweep == args.sweeps:
                trajectory.write(json.dumps(serial(frame(sweep)),allow_nan=False)+'\n'); trajectory.flush()
    save(args.out/'checkpoint.json',dict(frame(completed),rng_state=rng.bit_generator.state,
        note='State archived; Python prototype does not expose resume'))
    summary = dict(complete=True,completed_sweeps=completed,counts=counts,boundary='spherical',
                   auxiliary_transport=auxiliary is not None,cpu_seconds=time.process_time()-start)
    save(args.out/'summary.json',summary); print(json.dumps(summary),flush=True)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True); p.add_argument('--model',type=Path)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--sweeps',type=int,default=100)
    p.add_argument('--sample-every',type=int,default=5); p.add_argument('--gca-every',type=int,default=1)
    p.add_argument('--shift-every',type=int,default=1); p.add_argument('--auxiliary',action='store_true')
    a = p.parse_args()
    if min(a.sweeps,a.sample_every) < 1 or min(a.gca_every,a.shift_every) < 0: p.error('Invalid cadence')
    run(a)
