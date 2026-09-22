# Frozen bias for finite-system assembly

The optional `assembly_bias` configuration supplies dimensionless values for the
instantaneous largest exclusion-contact component, in order **B(1), …, B(N)**.
For example, a three-body fixture can use

```json
{"assembly_bias": {"values": [0.0, -1.0, -2.0]}}
```

Negative values favor those sizes. The array must contain exactly N finite
entries, and their differences must be representable. Omit the field for the
original ensemble. The table is frozen for an entire production trajectory;
a pilot-trained table must be frozen before starting that trajectory. The
implementation does not learn, interpolate or adapt the table.

For each pair of rigid bodies, an edge exists if any two atom spheres expanded
by the depletant radius intersect strictly. Thus the minimum atomic surface gap
is less than twice the depletant radius; exact tangency is excluded. A separate
sphere-union BVH accelerates this exact atomic predicate. Periodic boundaries
use minimum-image displacements under the runner's existing box-size condition.
The observable depends only on the current configuration: no native template,
residue label, previous contact or hysteresis is involved.

## Balance and elementary updates

Let n(X) be that largest component size and let the physical measure be

\[
\pi(dX) \propto 1_{\rm hard,wall}(X)\exp[-zV_{\rm excluded}(X)]\,dX.
\]

The sampled measure is \(\pi_B(dX)\propto e^{-B(n(X))}\pi(dX)\).
After an elementary physical kernel P proposes its retained configuration Y,
apply an additional acceptance

\[
a_B(X,Y)=\min(1,e^{B(n(X))-B(n(Y))}).
\]

If P satisfies physical detailed balance, its off-diagonal corrected flow is

\[
\pi(dX)P(X,dY)\min(e^{-B(n(X))},e^{-B(n(Y))}),
\]

which is symmetric. Add all rejections to the diagonal. This proves invariance
for each corrected elementary kernel. Their ordered sweep preserves the same
measure, although the entire ordered sweep need not itself be reversible.
Applying one correction after a whole sweep would not follow from this argument.

The runner applies the correction after each local move, frozen capture or
posterior-transport move, spherical GCA, and center shift. It retains all poses
on rejection; the display-frame wall center changes only after final acceptance.
A center shift preserves all pair distances mathematically. Its score is still
recomputed on stored floating-point coordinates so cached labels cannot become
history-dependent. Rejections restore both the physical poses and their frame.
The proof assumes the pre-existing physical kernels and exact arithmetic;
floating-point predicates retain the implementation's usual numerical limits.

Additional independent RNG streams leave the existing physical streams intact.
An absent table preserves old behavior and output fields. A constant table
performs no bias RNG draws and yields the same physical proposals, decisions,
poses and display center as an unbiased run, with extra bias diagnostics only.

Auxiliary transport, reversible-jump models, contact memory, conditional closure,
atlas transport and atlas masks are deliberately rejected when the bias is
present. The current extension supports immutable proposals; it does not silently
assume that partially restored auxiliary state preserves the expanded ensemble.

## Output and physical reweighting

The exact table appears in the configuration, manifest, checkpoint and summary.
Checkpoints also store and validate the instantaneous score and counters against
the recovered configuration. Continuation requires an identical input hash,
table and valid cached score.

Each JSONL frame records `assembly_bias` with `largest_component_size`, `bias`
and `log_reweight`. GSD frames contain corresponding
`log/tetramer_mc/assembly_largest_component`, `assembly_bias` and
`assembly_log_reweight` chunks. The log reweighting factor is B; physical
expectations use

\[
\langle f\rangle_\pi =
\frac{\langle f\exp(B)\rangle_{\pi_B}}{\langle\exp(B)\rangle_{\pi_B}}.
\]

Use log-space arithmetic and retain rejected states in averages. A frozen bias
does not by itself demonstrate adequate overlap, independent sampling or assembly
stability. Report importance-weight concentration and contact autocorrelation.

Move records distinguish `physical_accepted` from final `accepted`, and report
`assembly_bias_decision` only when the physical kernel accepted. Existing
`log_acceptance` remains the physical gate; the additional gate has its own
`log_acceptance`. Counts under `assembly_bias` record physical accepts and bias
accepts/rejections separately for each move family. Ordinary accepted and
transformed-body counters count only finally retained updates.

The older physical-only mobile observer explicitly refuses biased configurations.
A bias-aware observer must reconstruct both acceptance factors and reweight its
physical population estimates. The supplied runner tests do this for exact sphere
fixtures; this feature has not launched any protein assembly production runs.

## Validation

Focused Rust checks cover strict geometric thresholds, periodic images,
sphere-union contacts against a brute-force predicate, finite-state detailed
balance and composition, physical reweighting, absent/constant-table path
identity, elementary move replay including GCA rollback, frame bookkeeping, GSD
metadata, exact checkpoint continuation, tampered checkpoint rejection, and
unsupported-mode rejection. A two-sphere run checks contact occupancy and
physical reweighting against the analytic spherical-wall separation integral.

```bash
CARGO_TARGET_DIR=target/assembly-bias-review cargo test --offline --locked \
  --release --lib assembly_bias --jobs 4
CARGO_TARGET_DIR=target/assembly-bias-review cargo test --offline --locked \
  --release --test assembly_bias_runner --jobs 4
```
