# Frozen source-chart coverage diagnostic

`contact-source-coverage` separates a distant selected anchor from a contact
arrangement that the frozen proposal does not represent. It reads immutable
spherical trajectory frames and never evolves a configuration, draws a
Poisson bath, fits parameters, or uses a native-contact label. Activity is
recorded as provenance but does not affect these source-density diagnostics.
The contact graph is the exact strict overlap graph of the atomic sphere union
with every radius enlarged by the configured depletant radius.

```bash
cargo build --release --locked --offline -j2 --bin contact-source-coverage
./target/release/contact-source-coverage \
  --config runs/singleton-source-control-20261001/input/config.json \
  --trajectory runs/singleton-source-control-20261001/input/trajectory.jsonl \
  --model runs/singleton-source-control-20261001/input/model.json \
  --sweeps 23000,40000,60900 \
  --max-contexts-per-size 8 \
  --out runs/singleton-source-control-20261001/source-coverage
```

The output directory must be new. Omit expensive fused-catalogue construction
with `--no-fusion`; use `--max-contexts-per-size 0` for the all-body pair
coverage census alone. No production binary or running job is replaced by
building this separate binary.

## Fixed allocations and output

Every selected frame produces one `singletons.jsonl` record for every body.
Each record includes one reproducible uniformly drawn spectator anchor and
**every** actual touching spectator, along with the parent exclusion-contact
component. Thus isolated particles, internal cluster members, poorly fitted
contacts, and chart seams remain in the denominator. The two directions of an
edge are separate observations, not independent samples.

For each pair, the output records the complete learned label density
`w_b q_b`, its normalized source posterior, its standardized Gaussian norm,
and any encoding failure. It distinguishes the minimum norm across charts,
the norm of the highest-density label, and posterior mean/RMS norms. These
need not identify the same component. `q_b` uses translation volume times
normalized rotational Haar measure, including the production Jacobian and
reciprocal branch convention. The common defensive uniform branch is not part
of this learned-density diagnostic. All branch weights and covariance widths,
eigenvalues and numerical ranks are in `plan.json`.

`contexts.jsonl` contains up to the requested number of uniformly sampled
connected subsets per size (one, two, and three) and frame. Distinct subsets
are sampled without replacement, without native filtering, irrespective of
production channel rates. Eligibility and sample sizes are recorded. For each
subset:

1. Draw a primary spectator from the configured cluster-phase law (contact
   weighting plus the configured uniform floor, or uniform if unspecified).
2. Build the **production** pool from that primary and nearest spectators.
3. Construct a diagnostic **complete contact** pool containing all external
   touching neighbors. Empty pools are recorded explicitly, not substituted.
4. Evaluate both pools with the member-chart mixture alone and, unless
   disabled, with the production fused two-contact catalogue and its exact
   pruning limits.

A singleton context uses the cluster anchor law as a **control**. It does not
pretend that the original uniformly anchored single-body update uses that law.
All singleton global-anchor diagnostics remain in `singletons.jsonl`.

The source norms for fused labels use the exact production chart, member
transforms and reciprocal convention. Catalogues retain original limits and
report candidate fits, hard checks, retained fused labels, and construction
CPU time. Comparing full-contact and production pools can reveal a shortlist
problem; comparing member-only and fused catalogues can reveal how fusion
changes source coverage. Different pool sizes also change normalized weights,
so a log-density change alone is not evidence of better coverage.

## Interpretation and correctness boundary

- A poor uniformly selected anchor with at least one well-covered actual
  contact indicates an anchor-selection problem.
- Poor coverage even for actual contacts is evidence of a deficient source
  representation, but does not determine whether extra basins, broader tails,
  a different local chart, or different component weights are preferable.
- A high minimum norm has no universal physical threshold. Norms of 3, 6, and
  10 are reported as descriptive bins. Source posteriors are proposal
  allocation, **not equilibrium basin probabilities**.
- Complete-contact pools depend on the moving subset. They are an observer,
  **not a validated replacement proposal**. Using them in a move requires the
  appropriate reverse construction and selection correction.
- All densities are evaluated without conditioning on hard validity of a
  future trial. These are coverage and cost diagnostics, not acceptance,
  mixing-time, free-energy or equilibrium measurements.

The input trajectory must remain unchanged during reading. Requested sweeps
must exist exactly once. Its complete SHA256 and exact selected frame bytes
are recorded. Configuration, shape, model, selected frames, executable hash,
and the embedded source bundle are archived. Frame poses are already in the
sphere-center frame; `coordinate_wall_center` is not subtracted a second time.
The reader validates pose format and boundary/count consistency; it does not
repeat the run's global hard-state validation. All encountered pair/catalogue
errors are recorded and the remaining allocation is drained before a nonzero
exit. Cayley seams are explicit per-label failures with zero learned weight,
not resampled or discarded observations.

Tests (`cargo test --locked --offline -j2 --bin contact-source-coverage`)
cover represented sphere contacts versus distant anchors and absent basins,
rigid coordinate invariance, covariance eigenvalues, independent member
mixture density reconstruction, empty contact pools, and explicit chart seams.
The production involution and cluster-kernel tests remain the balance tests;
this tool adds read-only source encoders and does not alter any move law.
