#!/usr/bin/env python3
"""Native-blind exhaustive FFT scan of rigid pair depletion contacts.

For every rotation on a super-Fibonacci SO(3) grid, the ideal-depletant union
overlap C(t)=|E_0 cap E_A(t)| and a shrunk-core clash volume are evaluated for
all grid translations at once by FFT cross-correlation. Only the particle's own
sphere-union shape enters; no native motif, label or production pose is read.
The scan is an offline optimizer: grid scores are approximate and every kept
pose must pass the exact hard-core check and exact overlap refinement later.
"""
from __future__ import annotations
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import scipy.fft as sfft
from scipy.ndimage import maximum_filter
from scipy.spatial.transform import Rotation

PHI, PSI = np.sqrt(2.), 1.533751168755204288118041


def super_fibonacci(n, indices=None):
    """Alexa (2022) super-Fibonacci quaternions (w,x,y,z), near-uniform on SO(3)."""
    s = (np.arange(n) if indices is None else np.asarray(indices)) + 0.5
    r, R = np.sqrt(s/n), np.sqrt(1.-s/n)
    a, b = 2*np.pi*s/PHI, 2*np.pi*s/PSI
    q = np.stack([r*np.sin(a), r*np.cos(a), R*np.sin(b), R*np.cos(b)], 1)
    return q/np.linalg.norm(q, axis=1)[:, None]


def quat_matrix(q):
    return Rotation.from_quat(np.asarray(q)[..., [1, 2, 3, 0]]).as_matrix()


class Scanner:
    def __init__(self, shape, rd, spacing, core_shrink, size=None):
        self.X = np.array([a['center'] for a in shape['atoms']], float)
        self.r = np.array([a['radius'] for a in shape['atoms']], float)
        self.rd, self.h, self.shrink = rd, spacing, core_shrink
        reach = float(np.max(np.linalg.norm(self.X, axis=1)+self.r))+rd
        half = int(np.ceil(reach/spacing))+2
        n = size or sfft.next_fast_len(2*(2*half+1))
        self.N, self.o = n, half   # grid index i <-> coordinate (i-half)*h
        self.cell = spacing**3
        E0, K0 = self.voxelize(np.eye(3))
        self.F_E0, self.F_K0 = sfft.rfftn(E0), sfft.rfftn(K0)
        self._offsets = {}

    def voxelize(self, R):
        """Indicator grids of the inflated union (r+rd) and shrunk core (r-shrink)."""
        N, h, o = self.N, self.h, self.o
        E = np.zeros((N, N, N), np.float32); K = np.zeros((N, N, N), np.float32)
        Y = self.X@R.T
        for grid, radii in ((E, self.r+self.rd), (K, np.maximum(self.r-self.shrink, 0.))):
            for rad in np.unique(radii):
                if rad <= 0: continue
                P = Y[radii == rad]/h
                m = int(np.ceil(rad/h))+1
                off = np.stack(np.meshgrid(*[np.arange(-m, m+1)]*3, indexing='ij'), -1).reshape(-1, 3)
                base = np.floor(P).astype(int)
                pts = base[:, None, :]+off[None]
                d2 = np.sum((pts-P[:, None, :])**2, -1)*h*h
                sel = pts[d2 < rad*rad]+o
                grid[sel[:, 0], sel[:, 1], sel[:, 2]] = 1.
        return E, K

    def maps(self, R):
        """C(t) and clash(t) in A^3 for all grid shifts t (circular index)."""
        EA, KA = self.voxelize(R)
        shape = (self.N,)*3
        C = sfft.irfftn(self.F_E0*np.conj(sfft.rfftn(EA)), shape)*self.cell
        K = sfft.irfftn(self.F_K0*np.conj(sfft.rfftn(KA)), shape)*self.cell
        return C, K

    def shift(self, idx):
        idx = np.asarray(idx)
        return np.where(idx > self.N//2, idx-self.N, idx)*self.h

    def peaks(self, R, clash_max, top, separation):
        C, K = self.maps(R)
        S = np.where(K <= clash_max, C, -np.inf)
        size = max(1, int(round(separation/self.h)))
        local = (S == maximum_filter(S, size=2*size+1, mode='wrap')) & np.isfinite(S) & (S > 0)
        idx = np.argwhere(local)
        vals = S[local]
        order = np.argsort(-vals)[:top]
        return [(self.shift(idx[i]), float(vals[i]), float(K[tuple(idx[i])])) for i in order]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scan_shard(job):
    """Scan rotation indices worker::workers; write one immutable shard file."""
    args, worker = job
    shape = json.loads(Path(args['shape']).read_text())
    S = Scanner(shape, args['rd'], args['spacing'], args['core_shrink'])
    indices = np.arange(worker, args['rotations'], args['workers'])
    rows = []
    started = time.process_time()
    for i, q in zip(indices, super_fibonacci(args['rotations'], indices)):
        R = quat_matrix(q)
        C, K = S.maps(R)
        size = max(1, int(round(args['separation']/S.h)))
        for tol_class, tol in enumerate(args['clash_tolerances']):
            Sc = np.where(K <= tol, C, -np.inf)
            local = (Sc == maximum_filter(Sc, size=2*size+1, mode='wrap')) & (Sc > 0)
            idx = np.argwhere(local); vals = Sc[local]
            for j in np.argsort(-vals)[:args['peaks']]:
                t = S.shift(idx[j])
                rows.append((i, tol_class, *q, *t, vals[j], K[tuple(idx[j])]))
    out = Path(args['out'])/f'shard-{worker:04d}.npy'
    np.save(out, np.asarray(rows, float).reshape(-1, 11))
    return dict(worker=worker, rotations=len(indices), rows=len(rows),
                cpu_seconds=time.process_time()-started, sha256=sha(out))


def main():
    from multiprocessing import Pool
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shape', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--rd', type=float, default=1.4)
    p.add_argument('--spacing', type=float, default=1.0)
    p.add_argument('--core-shrink', type=float, default=0.5)
    p.add_argument('--rotations', type=int, default=300000)
    p.add_argument('--clash-tolerances', type=float, nargs='+', default=[2., 10.])
    p.add_argument('--peaks', type=int, default=4)
    p.add_argument('--separation', type=float, default=3.)
    p.add_argument('--workers', type=int, default=100)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True)   # refuse to reuse an existing scan
    args = dict(vars(a), shape=str(Path(a.shape).resolve()), shape_sha256=sha(a.shape),
                script_sha256=sha(__file__), native_information=False,
                columns=['rotation_index', 'tol_class', 'qw', 'qx', 'qy', 'qz',
                         'tx', 'ty', 'tz', 'grid_overlap_A3', 'grid_clash_A3'],
                scope='Grid correlation scores; approximate. Poses are shape-only candidates for exact refinement.')
    (out/'plan.json').write_text(json.dumps(args, indent=2)+'\n')
    started = time.time()
    with Pool(a.workers) as pool:
        shards = []
        for r in pool.imap_unordered(scan_shard, [(args, w) for w in range(a.workers)]):
            shards.append(r); print(json.dumps(r), flush=True)
    manifest = dict(complete=True, wall_seconds=time.time()-started,
                    shards=sorted(shards, key=lambda r: r['worker']), plan_sha256=sha(out/'plan.json'))
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
