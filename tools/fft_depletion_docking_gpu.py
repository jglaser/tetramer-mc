#!/usr/bin/env python3
"""GPU (CuPy/cuFFT) version of the native-blind FFT depletion pose scan.

Same grids, rotation set and scores as fft_depletion_docking.py, but batched
on the device and keeping *every* local maximum with grid overlap >= --min-overlap
(per clash-tolerance class) instead of a fixed top-k per rotation, so weaker
contacts that share a rotation with stronger ones are retained. Only the
particle's sphere-union shape enters; no native motif, label or production
pose is read.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import cupy as cp
from cupyx.scipy import ndimage as cnd
from scipy.spatial.transform import Rotation
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fft_depletion_docking import super_fibonacci

VOXELIZE = cp.RawKernel(r'''
extern "C" __global__ void voxelize(const float* centers, const float* radii, int natoms,
        int m, int n, int origin, float h, float* grid) {
    // one thread per (batch, atom, offset in (2m+1)^3 cube)
    long side = 2*m+1, cube = side*side*side;
    long tid = (long)blockIdx.x*blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (tid >= (long)natoms*cube) return;
    int a = tid / cube; long o = tid % cube;
    int ox = o/(side*side) - m, oy = (o/side)%side - m, oz = o%side - m;
    const float* c = centers + ((long)b*natoms + a)*3;
    float r = radii[a];
    if (r <= 0.f) return;
    int ix = (int)floorf(c[0]/h) + ox, iy = (int)floorf(c[1]/h) + oy, iz = (int)floorf(c[2]/h) + oz;
    float dx = ix*h - c[0], dy = iy*h - c[1], dz = iz*h - c[2];
    if (dx*dx + dy*dy + dz*dz >= r*r) return;
    ix += origin; iy += origin; iz += origin;
    if (ix < 0 || iy < 0 || iz < 0 || ix >= n || iy >= n || iz >= n) return;
    grid[(((long)b*n + ix)*n + iy)*n + iz] = 1.f;
}
''', 'voxelize')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class GpuScanner:
    def __init__(self, shape, rd, spacing, core_shrink, size=None):
        X = np.array([a['center'] for a in shape['atoms']], float)
        r = np.array([a['radius'] for a in shape['atoms']], float)
        reach = float(np.max(np.linalg.norm(X, axis=1)+r))+rd
        half = int(np.ceil(reach/spacing))+2
        import scipy.fft as sfft
        self.N, self.o, self.h = size or sfft.next_fast_len(2*(2*half+1)), half, spacing
        self.X = cp.asarray(X, cp.float32)
        self.rE = cp.asarray(r+rd, cp.float32)
        self.rK = cp.asarray(np.maximum(r-core_shrink, 0.), cp.float32)
        self.mE = int(np.ceil((r+rd).max()/spacing))+1
        self.mK = int(np.ceil(np.maximum(r-core_shrink, 0).max()/spacing))+1
        E0, K0 = self.voxelize(cp.eye(3, dtype=cp.float32)[None])
        self.F_E0, self.F_K0 = cp.fft.rfftn(E0[0]), cp.fft.rfftn(K0[0])

    def voxelize(self, R):
        B, n, na = R.shape[0], self.N, self.X.shape[0]
        Y = cp.ascontiguousarray(cp.einsum('bij,aj->bai', R, self.X), dtype=cp.float32)
        out = []
        for radii, m in ((self.rE, self.mE), (self.rK, self.mK)):
            grid = cp.zeros((B, n, n, n), cp.float32)
            total = na*(2*m+1)**3
            VOXELIZE(((total+255)//256, B), (256,), (Y, radii, np.int32(na), np.int32(m), np.int32(n),
                     np.int32(self.o), np.float32(self.h), grid))
            out.append(grid)
        return out

    def maps(self, R):
        E, K = self.voxelize(R)
        shape = (self.N,)*3; cell = self.h**3
        C = cp.fft.irfftn(self.F_E0[None]*cp.conj(cp.fft.rfftn(E, axes=(1, 2, 3))), shape, axes=(1, 2, 3))*cell
        del E
        Kc = cp.fft.irfftn(self.F_K0[None]*cp.conj(cp.fft.rfftn(K, axes=(1, 2, 3))), shape, axes=(1, 2, 3))*cell
        return C, Kc

    def peaks(self, R, tolerances, min_overlap, separation):
        C, K = self.maps(R)
        size = 2*max(1, int(round(separation/self.h)))+1
        rows = []
        for cls, tol in enumerate(tolerances):
            S = cp.where(K <= tol, C, -cp.inf)
            mx = cnd.maximum_filter(S, size=(1, size, size, size), mode='wrap')
            b, i, j, k = cp.nonzero((S == mx) & (S >= min_overlap))
            vals, clash = S[b, i, j, k], K[b, i, j, k]
            idx = cp.stack([i, j, k], 1)
            t = cp.where(idx > self.N//2, idx-self.N, idx).astype(cp.float32)*self.h
            rows.append((b.get(), cls, t.get(), vals.get(), clash.get()))
        return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shape', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--rd', type=float, default=1.4)
    p.add_argument('--spacing', type=float, default=1.0)
    p.add_argument('--core-shrink', type=float, default=0.5)
    p.add_argument('--rotations', type=int, default=300000)
    p.add_argument('--clash-tolerances', type=float, nargs='+', default=[2., 10.])
    p.add_argument('--min-overlap', type=float, default=500.)
    p.add_argument('--separation', type=float, default=3.)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--devices', type=int, default=1, help='total devices; this process takes indices device::devices')
    a = p.parse_args()
    cp.cuda.Device(a.device).use()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    shard = out/f'shard-gpu{a.device}.npy'
    if shard.exists():
        raise SystemExit('refusing to overwrite '+str(shard))
    plan = dict(vars(a), shape=str(Path(a.shape).resolve()), shape_sha256=sha(a.shape), script_sha256=sha(__file__),
                native_information=False,
                columns=['rotation_index', 'tol_class', 'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz', 'grid_overlap_A3', 'grid_clash_A3'])
    (out/f'plan-gpu{a.device}.json').write_text(json.dumps(plan, indent=2)+'\n')
    S = GpuScanner(json.loads(Path(a.shape).read_text()), a.rd, a.spacing, a.core_shrink)
    indices = np.arange(a.device, a.rotations, a.devices)
    started, chunks = time.time(), []
    for s in range(0, len(indices), a.batch):
        ids = indices[s:s+a.batch]
        q = super_fibonacci(a.rotations, ids)
        R = cp.asarray(Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix(), cp.float32)
        for b, cls, t, vals, clash in S.peaks(R, a.clash_tolerances, a.min_overlap, a.separation):
            chunks.append(np.column_stack([ids[b], np.full(len(b), cls), q[b], t, vals, clash]).astype(np.float32))
        if (s//a.batch) % 500 == 0:
            done = s+len(ids)
            print(json.dumps(dict(device=a.device, rotations=done, of=len(indices), rows=int(sum(len(c) for c in chunks)),
                                  elapsed=time.time()-started)), flush=True)
    rows = np.concatenate(chunks) if chunks else np.zeros((0, 11), np.float32)
    np.save(shard, rows)
    (out/f'manifest-gpu{a.device}.json').write_text(json.dumps(dict(complete=True, rotations=len(indices), rows=len(rows),
        wall_seconds=time.time()-started, shard_sha256=sha(shard)), indent=2)+'\n')


if __name__ == '__main__':
    main()
