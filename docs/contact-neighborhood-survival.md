# Contact compatibility as native neighbors are added

A geometric screen finds that the newly discovered distant pair arrangement
does not fit unchanged into the prescribed two-neighbor crystal environment.
This makes collective registry a concrete next test. It does not determine
the equilibrium preference, because adding a neighbor also removes most
inspected native and near-native configurations.

`tools/check_contact_neighborhood_survival.py` uses the existing site0
one-, two-, and three-neighbor environments. Their fixed poses form a nested
sequence; the native references and 18 Å capture domain are identical.
The added fixed neighbors are mutually hard-valid. No moving pose is relaxed,
reweighted, or retried.

| Snapshot cohort | Inspected | Valid with one neighbor | With two | With three |
|---|---:|---:|---:|---:|
| Newly discovered distant contact | 512 | 512 | 0 | 0 |
| Archived native region | 512 | 512 | 5 | 3 |
| Fresh shoulder region, 1 < q < 2 | 512 | 512 | 2 | 1 |

The distant cohort contains every endpoint of the discovery SMC population.
Those endpoints share one ancestry family and are not 512 independent
equilibrium observations. The native cohort takes 64 evenly spaced endpoints
from each of eight archived populations. The shoulder cohort takes 128 evenly
spaced hard-valid proposal observations from each of four independent
importance-sampling populations. The latter are **not importance weighted**.
Consequently none of the table's fractions is an equilibrium survival
probability, nor are the fractions comparable as physical basin populations.

Every check uses the unchanged 4,004-sphere tetramer and actual atomic radii.
A KD tree enumerates all potentially overlapping atom pairs, followed by
direct center-distance/radius tests. Every rejected pose has an explicit
clashing atom pair and negative surface gap recorded for each offending
neighbor. All 1,536 inspected poses pass the original one-neighbor predicate
and capture constraint. This is a floating-point geometric check, not a
formal interval certificate over a continuous pose region.

Results and input hashes are in
[`runs/contact-neighborhood-survival-20260920/report.md`](../runs/contact-neighborhood-survival-20260920/report.md)
and its companion `results.json`; the shape, environments, and executed script
are archived under `provenance/`.

```bash
/home/xvg/protein-nucleation/.venv/bin/python \
  tools/check_contact_neighborhood_survival.py \
  --out runs/contact-neighborhood-survival-new
```

## Physical question this motivates

Zero surviving distant snapshots does not prove zero weight for their entire
continuous basin, exclude a nearby rearrangement, or bound other competing
basins. Conversely, a surviving native pose proves only geometric
accessibility. The next physical comparison must integrate

\[
Q_b(S)=\int_{\mathcal D} H_S(x)1_b(x)
 \exp\!\left[z\left|E(x)\cap\bigcup_{j\in S}E_j\right|\right]dx
\]

with the full exclusion union, fixed capture domain and exhaustive region
partition. It must retain newly accessible competing arrangements and recover
the very narrow allowed native region. Reusing a one-neighbor proposal
without checking this coverage could create another false conclusion.

The testable hypothesis is that additional crystal neighbors suppress the
strong isolated-pair competitors more than native registry, after accounting
for the loss of native translational and rotational freedom. One-to-two
neighbor differences measure conditional selectivity; attributing an effect
to nonadditivity requires the full empty/first/second/both-neighbor contrast.
The formation cost and fluctuations of the prescribed neighbors remain
outside this conditional calculation. Complete the independent one-neighbor
mass controls before interpreting that extension as evidence about assembly.
