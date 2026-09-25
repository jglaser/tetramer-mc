# Pair-overlap recruitment under the current spherical GCA

The 2026-09-25 frozen-state diagnostic finds little two-body recruitment benefit
from replacing pointwise Poisson links with integrated pair-overlap links under
the **existing isotropic, centered half-turns**. This is a diagnosis of the move
geometry in three saved configurations, not a conclusion about equilibrium
assembly or a measured cluster-sampling speedup. Production GCA is unchanged.

## Rules being compared

Let A=E_i and B=E_j be old depletant-exclusion regions, with C=T E_i and D=T E_j
under one common proper involution T. Use V=|AB| and Vcross=|AD|=|CB|. The
pairwise reference potential is beta u_ij=-z V, giving the generalized GCA bond

    p_pair = 1 - exp[-z max(V - Vcross, 0)].

Hard-core crossings remain mandatory. This is the energy-difference rule of
[Liu and Luijten](https://csml.northwestern.edu/resources/Reprints/prl9.pdf),
applied to the integrated two-body depletion potential.

The current collapsed explicit-bath implementation uses, for two bodies,

    L = |AB minus (C union D)|,
    R = |AD minus (B union C)|,
    p_PPP_two = 1 - exp(-z L).

Involution symmetry implies V-Vcross=L-R after integration. Thus the pair rule
cancels spatial overlap gains against losses before recruitment, and
p_pair <= p_PPP_two. The gain is evaluated stably as

    exp[-z max(L-R,0)] * (1-exp[-z min(L,R)]).

This avoids losing tiny nonzero differences when both bond probabilities round
to one. No-link probabilities and their logarithms are also retained.

For the full bath, the relevant unpruned Poisson hyperedge region is
AB minus the union of **all** transformed exclusion regions. Its volume Lfull
is at most L. The diagnostic records p_full=1-exp(-z Lfull), but these pair
marginals belong to a correlated hypergraph. An integrated pair reference can
recruit more strongly than this full-bath marginal. Marginal comparisons are
not cluster-graph couplings or exchange-rate predictions.

A physical replacement would require the residual many-body correction:

    M(X) = volume(union E_i) - sum_i volume(E_i) + sum_{i<j} volume(E_i intersect E_j),
    alpha(X,Y) = min(1, exp[-z(M(Y)-M(X))]).

This applies to a cluster proposal reversible for the pairwise reference.
The diagnostic does not implement that proposal or correction. Its noisy
volume estimates must not be inserted into a physical acceptance exponential.

## Frozen allocation and complete geometric census

The input checkpoints were already frozen for the body-envelope benchmark.
Each received 32 independent axes from the production isotropic Gaussian-axis
construction, with a separate recorded diagnostic seed. Poses never evolve.
Axes are centered on the sphere, using the stored sphere-centered coordinates.
No favorable axes, native labels, or observed volumes select the probes.

Conservative bounding spheres and transformed atom-union AABBs identify all
old pair-overlap candidates. Existing hard-core components identify pairs
already forced to move together. If both crossed bounds are disjoint, the
two-body recruitment reduction is exactly zero. Remaining pair-axis cases are
labelled `cross_possible`; that label does not certify an actual overlap.

| Frozen snapshot | Radius A / activity A^-3 | Old bounding pair candidates | Hard-connected pair-axis cases | Cross-disjoint cases | Cross-possible cases |
|---|---|---:|---:|---:|---:|
| Native-contact N=12, sweep 2,000 | 1.5 / 0.035 | 38 | 27 | 1,178 | 11 |
| Aggregate N=48, sweep 29,300 | 1.4 / 0.025 | 81 | 403 | 2,157 | 32 |
| Seeded N=264, sweep 5,300 | 1.4 / 0.0275 | 454 | 1,581 | 12,884 | 63 |

These are three workloads, not a controlled system-size sweep. The N=264
snapshot contains eight seed and 256 additional free tetramers. All use the
same repaired rigid shape, hash c7442034… .

Probe selection is uniform without replacement over all cases within each of
two global strata, and frozen before volume evaluation. The cap was 32 per
stratum for N=12 and N=48, and 256 for N=264. Consequently **all 106
cross-possible cases were probed**, plus 32/32/256 disjoint controls. Unmeasured
disjoint cases have exactly zero two-body reduction; their many-body shielding
was not exhaustively measured. Hard-connected cases receive no soft-link
freedom score because another mandatory path already connects them.

Every probe has four independent populations of 4,096 unconditional draws
from a conservative envelope of A intersect (B union D), with a 2,047-cell
budget. Misses and certified empty envelopes retain their full denominator.
All six masks (old, crossed, lost, reverse-exclusive, full-bath lost, and
old triple coverage) are recorded per population. The total allocation is
426 probes / 1,704 populations / 6,979,584 unconditional draws. The three
release diagnostic jobs used about 8.62 process CPU seconds in total.

## Observed compensation

Only five probed cases had both old and crossed overlap hits: none in N=12,
one in N=48, and four in N=264. Zero-hit volumes retain nonzero uncertainty
unless geometry certifies an empty envelope.

The N=48 example (axis 23, pair 12-16) had estimated old/crossed overlap
1450.96 / 85.96 A^3. At z=0.025, its pair penalty remains 34.12 kBT. The
no-link probability rises by about 8.58x, but only from 1.76e-16 to 1.51e-15.
This is a small absolute change despite the relative multiplier.

The strongest observed change was N=264, axis 14, pair 173-198:

- Estimated L=343.55 and R=17.67 A^3; the residual pair penalty is 8.96 kBT.
- Recruitment reduction is 4.94e-5, with simultaneous cloud-integration
  interval [5.55e-6, 3.71e-4].
- This is one marginal pair probability, not a native-contact exchange rate.

For a concise finite-case summary, sum all two-body marginal reductions for
each sampled axis, then average over its 32 axes. The per-snapshot simultaneous
95% cloud-error bounds are:

| Snapshot | Estimated sum of reductions per axis | Upper bound |
|---|---:|---:|
| N=12 | 0 observed | 1.24e-4 |
| N=48 | 4.18e-17 | 2.06e-6 |
| N=264 | 1.54e-6 | 5.08e-4 |

The bounds use all cross-possible cases and exact-zero reductions elsewhere,
so they do not require extrapolating a sampled subset of those possible cases.
They remain conditional on these finite axes and configurations, and do not
cover unobserved equilibrium states or the distribution of future axes. A
sum of marginal changes must not be translated into a cluster-exchange
frequency: the current links have many-body correlations.

The N=264 probes also observed 50 old triple-coverage hits across three probes
and 20 shadow-shielded lost hits in one probe. These are different spatial
sets. They show why a pairwise approximation cannot silently replace the
many-body target. Full-bath losses elsewhere may still have undetected
corrections; unmeasured disjoint controls remain explicit.

The immediate obstacle in these workloads is that the centered half-turn
usually destroys an existing pair contact without retaining comparable crossed
overlap. Improving recruitment alone therefore has little demonstrated value.
A next proposal should generate more compensating arrangements, and exchange
between different partners also needs cancellation across a whole neighborhood;
ordinary pair-GCA recruitment only cancels changes within the same pair.

## Statistical and implementation checks

Seven Rust tests passed: analytic two-sphere overlaps, identity/empty envelopes,
triple/shielding masks, traversal-budget consistency, deterministic continuation
of population seeds, and invalid inputs. The analytic equal-volume control uses
core radius1, exclusion radius1.5, centers (1.1,+/-1.1,0), and a z half-turn.
Both pair distances are2.2 with no core crossings; V=Vcross=L=R=1.373923 A^3.
At diagnostic z=1, pair recruitment is exactly0 while Poisson recruitment is
0.746888. This validates the cancellation mechanism in a case designed to have
it; it does not establish accessibility for protein poses.

Analysis uses inverted binomial KL-Chernoff bounds, with simultaneous bounds
allocated over six masks and all measured probes in each report. Nonlinear
probabilities are diagnostic plug-in values with propagated intervals. The
integrated identity V-Vcross=L-R passed all simultaneous interval checks.
Analyzer self-tests include extreme no-link probabilities, zero/all-hit counts,
valid overlapping triple/shielding masks, empty censuses and missing-probe
rejection. An independent finite-binomial coverage grid also passed.

## Files and commands

- Runner: `src/bin/gca-overlap-diagnostic.rs`.
- Geometry estimator: `src/gca_overlap_diagnostic.rs`.
- Analyzer: `tools/analyze_gca_overlap_diagnostic.py`.
- Tests: `tests/gca_overlap_diagnostic.rs`.
- Local frozen inputs: `runs/body-envelope-cache-benchmark-20260924/*-config.json`.
- Reports: `runs/gca-overlap-compensation-20260925/{native12,aggregate48,seeded264}.json`.
- Final analyses: corresponding `*-analysis-v2.json` files. Earlier analyses are
  retained; v2 evaluates rare no-link differences without cancellation.

Census/selection files are saved before probing, and every completed probe is
flushed to a separate JSONL file. Input, executable, source and shape hashes are
recorded. No background assembly or contact-weight process was modified.

```bash
cargo build --locked --release --bin gca-overlap-diagnostic
cargo test --locked --offline --test gca_overlap_diagnostic
python3 tools/analyze_gca_overlap_diagnostic.py --self-test

target/release/gca-overlap-diagnostic \
  --config runs/body-envelope-cache-benchmark-20260924/aggregate48-config.json \
  --out runs/gca-overlap-repeat.json \
  --axes 32 --max-probes-per-stratum 32 \
  --draws-per-population 4096 --populations 4 --max-cells 2047
python3 tools/analyze_gca_overlap_diagnostic.py \
  --input runs/gca-overlap-repeat.json --out runs/gca-overlap-repeat-analysis.json
```

Use fresh output names. A different configuration can be supplied, including
portable examples, but then the result is a new conditional diagnostic rather
than a reproduction of the measured snapshots above.
