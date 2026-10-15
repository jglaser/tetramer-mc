"""Nucleation free-energy profile along the native growth path: each step adds one molecule at
its crystal site with site constant K_n (pair-binding with the current cluster as environment);
dG(n) = -sum_{k<n} ln(K_k rho) at 1 mM (kT; path free energy without symmetry or degeneracy
factors). Prints ln(K rho) per step and dG(n) per eps (as eps / eps_c, eps_c = 0.00325)."""
import json

import numpy as np

D = '/vast/xvg/tetramer-mc-runs/morph-growth-20261012/b22'
LN_RHO = np.log(1000e-6 * 6.02214076e23 / 1e27)
EPS_C = 0.00325
jobs = json.load(open(f'{D}/site_path_jobs.json'))
res = json.load(open(f'{D}/site_path_w6.json'))
eps = sorted({c['epsilon'] for c in res[0]['K']})
print(f"{'step':40s}" + ''.join(f'{e / EPS_C:>8.2f}' for e in eps) + '   (ln K rho; columns eps/eps_c)')
lk = []
for j, r in zip(jobs, res):
    k = {c['epsilon']: (c['lnK'] + LN_RHO, c['lnK_se'], c['ess']) for c in r['K']}
    lk.append([k[e][0] for e in eps])
    ess = min(k[e][2] for e in eps)
    print(f"{j['label']:40s}" + ''.join(f'{k[e][0]:8.1f}' for e in eps) + f'   ess>={ess:.0f}')
lk = np.array(lk)
dg = -np.cumsum(lk, axis=0)
print(f"\n{'cluster size n':40s}" + ''.join(f'{e / EPS_C:>8.2f}' for e in eps) + '   (dG(n), kT)')
for n in range(len(jobs)):
    print(f"{n + 2:<40d}" + ''.join(f'{dg[n, i]:8.1f}' for i in range(len(eps))))
for i, e in enumerate(eps):
    top = int(np.argmax(dg[:, i]))
    print(f'eps/eps_c {e / EPS_C:.2f}: barrier max dG = {dg[top, i]:.1f} kT at n = {top + 2}; '
          f'dG(n={len(jobs) + 1}) = {dg[-1, i]:.1f}')
