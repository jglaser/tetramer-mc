#!/usr/bin/env python3
"""GPU re-ranking of FFT scan basins with fine-grid overlap and full-core clash.

For a hard-valid pair pose, E_0 cap E_A lies in the fixed body's depletion
shell {x in E_0 : x outside the union of cores shrunk by r_d}. The kernel sums,
over a 0.5 A shell voxel list of the fixed body, membership of R^T(x-t) in the
moving body's 0.5 A inflated-union grid (overlap), and over a core-shell list,
membership in its unshrunk core grid (clash). Each candidate is also tried at
outward radial pushes; the score is the best overlap among pushes whose clash
volume is <= --clash-max. Approximate (nearest-voxel) ranking only; selected
poses are repaired and scored exactly afterwards. Shape-only, no native data.
"""
from __future__ import annotations
import argparse, glob, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import cupy as cp
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fft_depletion_docking_gpu import VOXELIZE
from select_fft_contact_starts import basin_maxima

SCORE = cp.RawKernel(r'''
extern "C" __global__ void score(const float* poses, int npose, int npert, const float* perts,
        const float* shell, int nshell, const float* cshell, int ncshell,
        const unsigned char* E, const unsigned char* K, int n, int origin, float h, float reach,
        float* overlap, float* clash) {
    // block = (pose, perturbation). pose: row-major R (9) + t (3).
    // perturbation: row-major dR (9) + radial push (1), applied as
    // R' = dR R, t' = c + dR (t - c) with pivot c = t/2, then t' += push * t'/|t'|.
    int p = blockIdx.x, s = blockIdx.y;
    if (p >= npose) return;
    const float* P0 = poses + 12*(long)p;
    const float* D = perts + 10*(long)s;
    float P[12];
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j)
        P[3*i+j] = D[3*i]*P0[j] + D[3*i+1]*P0[3+j] + D[3*i+2]*P0[6+j];
    float hx = 0.5f*P0[9], hy = 0.5f*P0[10], hz = 0.5f*P0[11];
    float tx = hx + D[0]*hx + D[1]*hy + D[2]*hz;
    float ty = hy + D[3]*hx + D[4]*hy + D[5]*hz;
    float tz = hz + D[6]*hx + D[7]*hy + D[8]*hz;
    float tn = sqrtf(tx*tx + ty*ty + tz*tz) + 1e-9f;
    float push = D[9];
    tx += push*tx/tn; ty += push*ty/tn; tz += push*tz/tn;
    float r2 = reach*reach;
    float acc = 0.f, acck = 0.f;
    for (int i = threadIdx.x; i < nshell + ncshell; i += blockDim.x) {
        const float* x = (i < nshell) ? shell + 3*(long)i : cshell + 3*(long)(i - nshell);
        float dx = x[0]-tx, dy = x[1]-ty, dz = x[2]-tz;
        if (dx*dx + dy*dy + dz*dz > r2) continue;
        // q = R^T (x - t)
        float qx = P[0]*dx + P[3]*dy + P[6]*dz;
        float qy = P[1]*dx + P[4]*dy + P[7]*dz;
        float qz = P[2]*dx + P[5]*dy + P[8]*dz;
        int ix = __float2int_rn(qx/h) + origin, iy = __float2int_rn(qy/h) + origin, iz = __float2int_rn(qz/h) + origin;
        if (ix < 0 || iy < 0 || iz < 0 || ix >= n || iy >= n || iz >= n) continue;
        long c = ((long)ix*n + iy)*n + iz;
        if (i < nshell) acc += E[c]; else acck += K[c];
    }
    // block reduction
    __shared__ float sa[256], sk[256];
    sa[threadIdx.x] = acc; sk[threadIdx.x] = acck; __syncthreads();
    for (int w = blockDim.x/2; w > 0; w >>= 1) {
        if (threadIdx.x < w) { sa[threadIdx.x] += sa[threadIdx.x+w]; sk[threadIdx.x] += sk[threadIdx.x+w]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float cell = h*h*h;
        overlap[(long)p*gridDim.y + s] = sa[0]*cell;
        clash[(long)p*gridDim.y + s] = sk[0]*cell;
    }
}
''', 'score')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def body_grids(shape, rd, h):
    X = np.array([a['center'] for a in shape['atoms']], float)
    r = np.array([a['radius'] for a in shape['atoms']], float)
    reach = float(np.max(np.linalg.norm(X, axis=1)+r))+rd
    o = int(np.ceil(reach/h))+2; n = 2*o+1
    Y = cp.asarray(X[None], cp.float32)
    def vox(radii):
        grid = cp.zeros((1, n, n, n), cp.float32)
        m = int(np.ceil(max(radii.max(), h)/h))+1
        VOXELIZE(((len(X)*(2*m+1)**3+255)//256, 1), (256,), (Y, cp.asarray(radii, cp.float32), np.int32(len(X)),
                 np.int32(m), np.int32(n), np.int32(o), np.float32(h), grid))
        return grid[0] > 0
    E, K = vox(r+rd), vox(r)
    inner = vox(np.maximum(r-rd, 0.))
    kin = vox(np.maximum(r-1.5, 0.))
    coords = lambda mask: ((cp.argwhere(mask)-o)*h).astype(cp.float32)
    shell, cshell = coords(E & ~inner), coords(K & ~kin)
    return E.astype(cp.uint8), K.astype(cp.uint8), n, o, reach, shell, cshell


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scan', required=True)
    p.add_argument('--shape', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--rd', type=float, default=1.4)
    p.add_argument('--spacing', type=float, default=0.5)
    p.add_argument('--basin-rotations', type=int, default=20000)
    p.add_argument('--basin-translation', type=float, default=3.0)
    p.add_argument('--prefilter', type=int, default=3000000, help='best basins by coarse grid overlap to rescore')
    p.add_argument('--pushes', type=float, nargs='+', default=[0., .5, 1., 1.5, 2.])
    p.add_argument('--tilts', type=float, nargs='+', default=[1.5, 3.],
                   help='pivot rotations (degrees) about +-x,+-y,+-z, combined with every push')
    p.add_argument('--clash-max', type=float, default=1.0)
    p.add_argument('--batch', type=int, default=4096)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--part', type=int, default=0)
    p.add_argument('--parts', type=int, default=1)
    a = p.parse_args()
    cp.cuda.Device(a.device).use()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if (out/f'manifest-{a.part}.json').exists():
        raise SystemExit('refusing to overwrite part '+str(a.part))
    started = time.time()
    shards = sorted(glob.glob(str(Path(a.scan)/'shard-*.npy')))
    rows = np.concatenate([np.load(f).astype(float) for f in shards])
    basins = basin_maxima(rows, a.basin_rotations, a.basin_translation)
    basins = basins[np.argsort(-basins[:, 9], kind='stable')][:a.prefilter]
    E, K, n, o, reach, shell, cshell = body_grids(json.loads(Path(a.shape).read_text()), a.rd, a.spacing)
    R = Rotation.from_quat(basins[:, [3, 4, 5, 2]]).as_matrix().reshape(-1, 9)
    poses = np.hstack([R, basins[:, 6:9]]).astype(np.float32)
    rots = [np.eye(3)]+[Rotation.from_rotvec(np.radians(sgn*deg)*np.eye(3)[ax]).as_matrix()
                        for deg in a.tilts for ax in range(3) for sgn in (1, -1)]
    pert_list = [np.r_[dR.ravel(), push] for dR in rots for push in a.pushes]
    perts = cp.asarray(np.array(pert_list), cp.float32); npert = len(pert_list)
    lo, hi = a.part*len(poses)//a.parts, (a.part+1)*len(poses)//a.parts
    poses, basins = poses[lo:hi], basins[lo:hi]
    ov = np.empty((len(poses), npert), np.float32); cl = np.empty_like(ov)
    for s in range(0, len(poses), a.batch):
        P = cp.asarray(poses[s:s+a.batch]); m = len(P)
        o_ = cp.zeros((m, npert), cp.float32); c_ = cp.zeros_like(o_)
        SCORE((m, npert), (256,), (P, np.int32(m), np.int32(npert), perts, shell, np.int32(len(shell)),
              cshell, np.int32(len(cshell)), E, K, np.int32(n), np.int32(o), np.float32(a.spacing),
              np.float32(reach+0.5), o_, c_))
        ov[s:s+m], cl[s:s+m] = o_.get(), c_.get()
    ok = cl <= a.clash_max
    masked = np.where(ok, ov, -1.); arg = masked.argmax(1); best = masked.max(1)
    pert = np.array(pert_list)[arg]
    moved = basins.copy()
    for i in np.where(best > 0)[0]:   # apply the chosen perturbation exactly as the kernel did
        dR, push = pert[i, :9].reshape(3, 3), pert[i, 9]
        Rm = Rotation.from_quat(basins[i, [3, 4, 5, 2]]).as_matrix(); t = basins[i, 6:9]
        R2 = dR@Rm; t2 = t/2+dR@(t/2); t2 = t2+push*t2/np.linalg.norm(t2)
        x, y, z, w = Rotation.from_matrix(R2).as_quat(); moved[i, 2:6] = [w, x, y, z]; moved[i, 6:9] = t2
    moved[:, 9] = best; moved[:, 10] = cl[np.arange(len(cl)), arg]
    keep = best > 0
    np.save(out/f'shard-rescored-{a.part}.npy', moved[keep].astype(np.float32))
    manifest = dict(complete=True, native_information=False, arguments=vars(a), script_sha256=sha(__file__),
                    scan_shards={Path(f).name: sha(f) for f in shards}, peaks=len(rows), rescored=len(poses),
                    clash_free=int(keep.sum()), shell_voxels=len(shell), core_shell_voxels=len(cshell),
                    wall_seconds=time.time()-started,
                    columns='as scan; grid_overlap_A3 = fine-grid shell overlap at best clash-free push; translation includes push')
    manifest.update(part=a.part, parts=a.parts, perturbations=npert)
    (out/f'manifest-{a.part}.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k: manifest[k] for k in ('peaks', 'rescored', 'clash_free', 'shell_voxels', 'core_shell_voxels', 'wall_seconds')}))


if __name__ == '__main__':
    main()
