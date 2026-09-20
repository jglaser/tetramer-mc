# Geometry of the newly found distant-contact population

The explicit-far SMC run `site0-far5-r1.5-z0.035-n512-r1` found a strongly
overlapping, physically distinct contact arrangement. Independent geometry
checks confirm its selected endpoint poses are hard-valid and have exclusion
overlap C around 800–1,100 Å³. They do not confirm its reported logQ=27.273:
all 512 final particles descend from one initial family, so that estimate
still requires independent populations and an independent normalizer.

The physical parameters remain rd=1.5 Å, z=0.035 Å⁻³, with the original 18 Å
capture sphere. Recomputed original registration errors are q=25.936–28.238;
the saved internal q is exactly this value divided by five. Every endpoint
center is within the capture sphere, at distance 9.685–13.555 Å. No q
partition or original source data were changed during this annotation audit.

## Independent physical check

The six endpoints were selected before new integration: the three highest
saved overlap counts plus minimum, maximum and median q. Each was checked
using the frozen archived/current geometry probe, 1,048,576 common uniform
points, and 32 independent current Poisson envelope clouds at λ=64z=2.24.
Both hard-overlap implementations accept all six. Archived and current
exclusion predicates agree on all 6,291,456 common points.

| Endpoint | Selection | Common-box C ± SE / Å³ | Current envelope C ± SE / Å³ |
|---|---|---:|---:|
|452|highest saved count|1102.52 ±6.81|1108.12 ±3.93|
|078|second highest saved count|1002.41 ±6.42|1006.84 ±3.75|
|428|third highest saved count|1015.41 ±6.63|1011.65 ±3.76|
|237|minimum q|795.22 ±6.02|791.71 ±3.32|
|196|maximum q|1075.75 ±6.61|1063.57 ±3.85|
|305|median q|1095.71 ±6.79|1092.59 ±3.90|

The two independent C estimates differ by at most 1.60 combined standard
errors. These computations took 6.36 CPU seconds. The saved final count
intensity was only 0.0134896 Å⁻³: its largest individual count, 27, suggested
C≈2002 Å³ at endpoint452. The accurate fresh estimate is 1108 Å³. Ranking
those original noisy counts therefore exaggerated the apparent deepest
overlap. Their population mean C≈1055 Å³ is consistent with the fresh values,
but descendants are not independent equilibrium samples.

## Registry and internal symmetry

All 24 permutations of the four member centers were fitted by proper Kabsch
rotations. The resulting candidate body symmetries were then checked against
member orientations and a radius-preserving full-atom nearest-neighbor
bijection, using a 1e−7 tolerance. Only identity is a full shape symmetry.

There is one additional exact *center-only* symmetry, permutation (3,2,1,0),
with center error below 1.1e−14 Å. It fails the physical shape test: member
orientation mismatch is 180°, and the maximum radius-matched atom distance
is 11.60 Å. It cannot be used to declare the endpoints equivalent to native
registry. Minimizing q over validated shape symmetries leaves the original
range unchanged.

The archived capture certificate lists 28 nearby crystallographic placements.
Only the identity placement has its center inside the capture sphere; the
nearest alternative is 21.194 Å away. Comparing all 512 endpoint poses to all
28 listed placements gives minimum registration errors ranging from 18.556
to 20.710. The new contact is therefore not identified with any of those
listed crystal neighbors. This uses the archived certificate and is not a
fresh exhaustive reconstruction of the crystal lattice.

## Scope and reproduction

This establishes strong distant contacts under the same physical geometry,
and excludes the tested internal-relabeling explanation. It does not
establish their equilibrium population, their full basin volume, or the
correctness of the single SMC normalizer. A frozen guide fitted to this
population can support a new independent importance calculation, provided
the full proposal density is evaluated and all unconditional draws are
retained.

The [result bundle](../runs/explicit-far-endpoint-audit/results.json),
[symmetry annotations](../runs/explicit-far-endpoint-audit/symmetry.json),
[population audit](../runs/explicit-far-endpoint-audit/population.json),
and [source hashes](../runs/explicit-far-endpoint-audit/provenance.json)
are saved separately from the input run. Reproduce with a fresh output path:

```bash
/home/xvg/protein-nucleation/.venv/bin/python tools/audit_explicit_far_endpoints.py \
  --out runs/explicit-far-endpoint-reproduction
```

The tool archives source inputs and the previously frozen geometry executable.
The [separate archived count-law audit](archived-smc-count-audit.md) validates
the exact old Poisson sampling function at three additional fixed poses.
