# Continuous exclusion of the frozen distant-contact region

The **entire frozen R3 pose region clashes with the prescribed second native
neighbor**, by a cell-cover certificate with zero unresolved cells. The same
construction also certifies the larger R5 and R8 balls of this unchanged
Gaussian chart. This strengthens the earlier observation that all 512
discovery endpoints clash: the conclusion now concerns continuous regions,
not just those snapshots.

The certified set is actually larger than the requested physical region:
every pose decoded from the full latent ball is covered, including poses
outside the capture sphere, below the original q threshold, or already
invalid against the first neighbor. Its physical subset is therefore excluded
as well. The second neighbor remains fixed at its prescribed native pose;
neighbor rearrangements and other competing basins are outside this test.

| Frozen-chart radius | Visited cells | Collision leaves | Outside-ball leaves | Unresolved | Smallest certified penetration / Å |
|---|---:|---:|---:|---:|---:|
| R3 |33|17|0|0|0.006652|
| R5 |495|231|17|0|0.001563|
| R8 |5651|2479|347|0|0.000196|

All three traversals and their numerical audits completed in 7.07 seconds.
The collision threshold included an additional 1e−6 Å slack. The input
shape SHA-256, frozen chart, second-neighbor pose, Cholesky factor and source
hashes are retained in
[the provenance record](../runs/frozen-deep-region-neighbor2-certificate-v2/provenance.json).
No scientific input was modified and no equilibrium sampling was run.

## Why each cell can be excluded

The implementation follows the existing Rust chart:

```text
x(u) = mean + L u
t(u) = t_fixed + R_fixed (anchor_t + x_translation)
R(u) = R_fixed Cayley(x_rotation / ell) R_anchor
```

Here `ell = 55.02283113084892 Å`; the translation and rotation blocks of L
have Frobenius norms 0.6875664310 and 0.7891556776, respectively. The lower
triangular factor uses the same Cholesky loop as the Rust implementation.

The quaternion of the Cayley rotation is
`q(c) = (1,c) / sqrt(1 + |c|²)`. Its pullback metric is

```text
Dq(c)^T Dq(c) = [(1+|c|²) I - c c^T] / (1+|c|²)^2  <= I.
```

A straight segment in c-space therefore has SO(3) angular length at most
`2 |delta c|`. An atom at body-frame radius r moves at most
`2 r |delta c|` under the corresponding rotation. These are global bounds;
they do not use a small-angle approximation.

For a box centered at u0, with half-widths w, intersected with a ball of
radius R, set `h = min(|w|, R + |u0|)`. Every relevant displacement obeys
`|u-u0| <= h`. Each linear-map variation has two valid bounds:

```text
|A delta u| <= h |A|_F
|A delta u| <= | abs(A) w |.
```

The implementation takes their minimum separately for translation and
Cayley parameters, then adds the translational and rotational atom-motion
bounds. If one actual atom pair at the cell center has penetration greater
than that bound plus the stated slack, that same pair overlaps throughout
the cell's intersection with the ball. One pair suffices; different cells
may use different pairs.

Uncertified cells are split along the largest estimated geometric-uncertainty
coordinate. Both children are retained. A cell is discarded as outside only
when its minimum distance to the origin is strictly greater than R, with
additional numerical slack. Recorded child boxes were checked to partition
every split parent, so the completed traversal covers the closed ball.

## Numerical audit and limits

Every recorded witness was recomputed by direct atom-pair distance,
independently of the nearest-neighbor search. The binary partition structure
was verified in a separate pass. Another 1,088 / 8,192 / 8,192 deterministic
cell/ball boundary-point checks for R3 / R5 / R8 all respected the displacement
bounds and retained the certified collisions. A 5,616-case deterministic
Cayley check also satisfied the analytic factor-two bound.

These boundary checks audit the implementation; coverage comes from the
analytic bounds and full partition. The computation uses ordinary floating
point, small outward inflation of the bounds, and positive penetration slack.
It is **an analytic enclosure with a numerical audit, not a formally rounded
interval-arithmetic proof**.

The result applies only to this frozen chart and the specified second
neighbor. It does not exclude other distant contacts, account for the cost
of assembling the neighbors, or establish an equilibrium or nucleation
mechanism.

## Artifacts and reproduction

The [standalone tool](../tools/certify_frozen_region_clashes.py) records every
split, outside cell and atom-pair witness in `R*-tree.jsonl`, together with
explicit unresolved-cell files, even when those files are empty. The
[result bundle](../runs/frozen-deep-region-neighbor2-certificate-v2/results.json)
includes budgets, timing and audit outcomes. The initial diagnostic run is
retained separately; v2 adds meaningful boundary checks even when all box
corners lie outside the ball.

```bash
PYTHONDONTWRITEBYTECODE=1 /home/xvg/protein-nucleation/.venv/bin/python -B \
  tools/certify_frozen_region_clashes.py \
  --out runs/frozen-region-clash-reproduction \
  --radii 3 5 8 --max-nodes 100000 --max-seconds 300
```

Choices that can be revisited are the box partition, geometric splitting
score, cached witness pairs, Frobenius/rowwise bounds and traversal budget.
The KD trees group fixed atoms by radius and only discover candidate pairs;
the exclusion criterion always uses an explicit core-sphere distance.
