# Contact-conditioned spherical half-turns

Status (2026-09-25): implemented as an optional `gca_axis` setting in the Rust
runner, with the existing physical GCA inside an auxiliary-exchange wrapper.
The option is absent by default; existing configurations and running campaigns
retain their previous move law. Validation and protein benchmark results are
recorded below. An exchange-rate gain is distinct from improved equilibrium
mixing or native assembly.

## Purpose and geometry

The previous [overlap diagnostic](gca-overlap-compensation.md) tested isotropic
axes in frozen configurations. Same-pair compensation was small. A different
opportunity is to select a half-turn that exchanges *different* contact partners
of a uniformly chosen, retained particle label i.

All coordinates below are relative to the spherical vessel center. A common
proper half-turn is

    T_u r = 2 u (u . r) - r,        u ~ -u.

It acts on positions and orientations. It preserves chirality, each body's
radius from the vessel center, the atomic wall, and relative geometry when
both bodies are transformed. If the tag center is r != 0 and a prospective
endpoint s satisfies |s|=|r|, the axis is

    u = (r+s) / |r+s|.

The antipodal endpoint has a family of perpendicular axes and zero probability
under a continuous endpoint density. At r=0, the tag center cannot translate;
its orientation can still change. One half-turn has two continuous parameters,
not six-dimensional docking freedom. Center shifts change the accessible
radial geometry between attempts.

Let E_j be each body's depletant-exclusion region and U_T = union_j T_u E_j.
The physical GCA recruits old owners using Poisson points in multiply covered
old regions outside U_T. For an old contact i-j, its relevant region includes

    (E_i intersect E_j) minus U_T.

A proposed new contact of T_u E_i with E_k implies a crossed contact of E_i
with T_u E_k. That transformed neighbor can cover part of the old i-j overlap,
reducing recruitment of the old partner. This is possible many-body shielding;
contact existence alone does not guarantee useful shielding or registry.

## A concrete, deterministic contact score

Use the existing geometric contact definition: two exclusion sphere unions
intersect (atomic radii inflated by the configured depletant radius). Build
an immutable inflated sphere tree once for this diagnostic/selector.

For the retained tag i, compute its current neighbor set C_i(X). For a trial
axis u, transform only the tag as a *score calculation*, leaving all other
bodies fixed, and compute C_i^trial(X,u). One minimal predicate is

    P_i(X,u) =
        the trial tag has no hard-core collision with any other body
        and C_i(X) minus C_i^trial(X,u) is nonempty
        and C_i^trial(X,u) minus C_i(X) is nonempty.

The tag automatically remains within the spherical wall. The physical move
still uses the full cluster kernel: this tag-only trial is neither an endpoint
constraint on that kernel nor an additional physical energy. The predicate
uses geometry, not native labels. Restricting the score to tag-only hard-valid
poses is conservative for discovery; collective rearrangements may be useful
even when this predicate fails.

A positive, bounded score is

    h_i(X,u) = epsilon + (1-epsilon) P_i(X,u),    0 < epsilon <= 1.

This permits uniformly choosing *any* label, including unbound tags, without
an empty conditional distribution. A smooth deterministic score in
[epsilon,1] is also permitted. Contact-overlap size, multiple-neighbor gain,
and contact retention can be score refinements; they must remain deterministic
or have their randomness represented in a separate auxiliary proof. A noisy
volume plugged into the formula is not covered by this derivation.

## Exact conditional selection without an angular normalizer

Let r0(du) be the uniform measure on unoriented axes, and define

    q_X(du) = r0(du) h_i(X,u) / Z_i(X),
    Z_i(X) = integral h_i(X,u) r0(du).

Draw q_X exactly by independent rejection sampling from r0 with acceptance
h_i(X,u). No geometric integration over the set of qualifying axes is needed.

For each attempt:

1. Choose tag i uniformly from the fixed labels; retain it throughout.
2. Draw u from q_X.
3. With fresh independent physical GCA randomness, propose Y using the current
   fixed-axis kernel K_u(X,dY).
4. Independently draw an auxiliary axis v from q_Y, for the same tag i. This
   auxiliary axis does not generate another physical move.
5. Accept Y with

       alpha = min(1, h_i(Y,u) h_i(X,v) / [h_i(X,u) h_i(Y,v)]).

Use the *actual whole-cluster endpoint* Y for all reverse scores. Keep every
rejection, tag nonflip, and failed exchange in the Markov trajectory. Existing
auxiliary-model and frozen assembly-bias gates are still needed for those
extended or biased targets. No extra physical depletion-energy gate is added.

### Balance argument

The premise is fixed-axis physical detailed balance:

    pi(dX) K_u(X,dY) = pi(dY) K_u(Y,dX).

This is the existing all-mobile spherical GCA, including hard components,
Poisson hyperedges and independent fair component flips. State-independent
axis selection in its existing caller inherits that already-established
fixed-axis reversibility; state-dependent selection needs the wrapper derived
here.

Pair extended flows under (X,Y,u,v) <-> (Y,X,u,v):

    F = pi(dX) q_X(du) K_u(X,dY) q_Y(dv),
    R = pi(dY) q_Y(du) K_u(Y,dX) q_X(dv).

The physical factors cancel by the premise. Both flows contain
r0(du) r0(dv) / [Z_i(X) Z_i(Y)]. Their ratio is precisely the acceptance
ratio above. Taking min(F,R), adding rejection on X, and integrating u,v
therefore gives a pi-reversible Markov kernel. The tag-selection factor 1/N
also cancels. The proposal and selector need not individually preserve pi.

This is an auxiliary-exchange construction. It avoids the otherwise missing
Z_i(X)/Z_i(Y) factor in a naive conditioned-axis move. It is not generally
rejection-free. An axis satisfying the forward predicate can have a poor
reverse score; both cross-scores are useful recorded diagnostics.

### Bounded runtime with failures retained

Each of steps 2 and 4 may use the same deterministic cap M on rejection
candidates. Failure at either stage means an overall self-loop. The successful
subdensity at state X is

    s(X) q_X,    s(X) = 1 - (1-Z_i(X))^M.

The extra factors s(X)s(Y) cancel in the paired flows as well. Thus the same
acceptance rule stays exact, without computing either success probability.
The cap may also be a fixed deterministic function of the state, provided the
same rule is used by the forward and reverse routines. Unrecorded retries
until a whole attempt succeeds are not allowed. An approximate MCMC draw for
v is not an exact replacement for the conditional rejection draw.

The epsilon floor bounds uncapped mean candidates by 1/epsilon, but also limits
how strongly rare qualifying directions can dominate. With qualifying-axis
fraction p, their conditional probability is p/[p+epsilon(1-p)]. A small p
still calls for a geometric candidate generator, not just a smaller floor.

## Geometric acceleration: normalized spherical bands

A separate tractable axis guide can put the prospective tag center near other
particles before checking detailed shape contacts. Let R_i=|r_i| and consider
endpoints R_i v, |v|=1, within a fixed distance band [dmin,dmax] of r_j. For
R_i R_j > 0, the band has

    a_ij = (R_i^2 + R_j^2 - dmax^2)/(2 R_i R_j),
    b_ij = (R_i^2 + R_j^2 - dmin^2)/(2 R_i R_j),
    Omega_ij = 2 pi max(0, min(1,b_ij)-max(-1,a_ij)).

For an empty clipped interval use zero area. If R_j=0, the band is either the
whole endpoint sphere or empty according to dmin <= R_i <= dmax. If R_i=0,
fall back to uniform axes. Choose j proportional to Omega_ij and draw v
uniformly in that band. The complete endpoint density with respect to solid
angle is

    H_X(v) = multiplicity_X(v) / sum_{j != i} Omega_ij.

All overlapping bands contribute to multiplicity, not just the selected j.
Recover u from the endpoint. The solid-angle Jacobian is

    dOmega_v = 4 |u . rhat_i| dOmega_u.

Relative to uniform projective-axis probability measure, a defensive guide is

    g_X(u) = eta + (1-eta) 8 pi |u . rhat_i| H_X(T_u rhat_i),
    0 < eta <= 1.

Empty band collections use the pure-uniform guide. Normalization follows
from the endpoint construction and Jacobian. Use fixed label eligibility and
fixed distance bands; all R_i, R_j and band areas are preserved by any partial
centered half-turn. The absolute projection |u.rhat_i| is also unchanged. Thus
at a physical GCA endpoint define the common factor

    b = 8 pi |u . rhat_i| / sum Omega_ij.

Then g_Y(u)/g_X(u) = [eta+(1-eta)b m_Y]/[eta+(1-eta)b m_X].
Only with eta=0 does this reduce to m_Y/m_X. With a defensive floor the
invariant factor b must still be included numerically.

There are two choices:

- Use g directly as the normalized selector and gate the physical GCA with
  min(1,g_Y(u)/g_X(u)). No second auxiliary axis is needed.
- Use g_X as the base for rejection by the detailed score h_i(X,u). The
  exchange-auxiliary construction remains valid but its ratio must also include

      g_Y(u) g_X(v) / [g_X(u) g_Y(v)].

Do not silently treat a state-dependent candidate generator as isotropic.
Discarding current contact partners from the bands changes their normalizer;
its invariance claim then no longer applies. Center-distance bands can guide
proximity but do not certify useful protein patch orientation.

## Validation and a useful first benchmark

`python3 tools/validate_conditional_axis_exchange.py` enumerates finite states,
nonuniform physical probabilities, and several reversible marked kernels.
It passed 12 exact-fraction checks: three physical kernels and nine corrected
selector variants, including binary/positive scores, capped null outcomes, and
state-dependent bases. Both naive-conditioning controls violate stationarity
and detailed balance as expected. This validates the algebra;
it does not validate floating-point geometric predicates, spherical-band
sampling, or a protein implementation of this selector.

The implementation passes 11 focused tests covering the band Jacobian and
normalization, overlapping-band multiplicity, contact/core predicates,
degenerate geometry, partial-transform invariance, capped rollback,
deterministic replay, and four-density ratio reconstruction. Three runner tests
cover configuration validation, unchanged disabled behavior, bias composition,
full retained-state replay and exact checkpoint continuation. Existing spherical,
assembly-bias and conditional-closure runner tests also pass.

An independent physical reference uses two core spheres of radius 0.3 A,
depletant radius 0.6 A, activity 1.5 A^-3, and centers fixed 1.5 A from the center of a 4 A vessel. The separation density is proportional to
`d exp[z V_overlap(d)]` for 0.6 <= d <= 3. Both isotropic and band bases were
checked from near/far starts, each with 2,000 burn-in and 20,000 retained moves
(88,000 total attempts). Exact contact probability is 0.55030; estimates are
0.55385 +/- 0.00636 for isotropic and 0.54320 +/- 0.01464 for bands, using block
standard errors. These validate the target in a tractable reference; they are
not protein-mixing or speedup measurements.

Actual completed partner exchanges per CPU and contact-fingerprint ESS matter;
axis success or acceptance alone does not establish improved mixing.
The selector conditions the proposal toward an exchange. It cannot promise
that the stochastic cluster outcome exchanges contacts. Forcing the tagged
component to flip or repeating GCA draws until an exchange occurs requires
a separate reversible-kernel derivation and tests.

## Runner configuration

The optional spherical-only setting is

```json
"gca_axis": {
  "score_floor": 0.02,
  "max_candidates": 64,
  "uniform_axis_weight": 0.25
}
```

All three values have these defaults when the object is empty. A missing/null
object keeps the previous isotropic GCA sequence, including its RNG order.
`uniform_axis_weight: 1` uses the isotropic base for the conditional searches;
`score_floor: 1` removes the contact-score preference while retaining the
chosen base. Both searches use the same cap, and every exhausted search is a
recorded self-loop. The implemented band distance range is
`[0, 2*(body_bound + depletant_radius)]`, with every other label eligible.
The inflated atom sphere tree is constructed once using the effective bath
radius, including startup radius overrides.

The selector uses a separate named guide RNG stream. The physical GCA consumes
its original named stream. `counts.gca_axis` records both searches, physical
proposals, selector decisions and actual tagged exchanges after all subsequent
gates. With move recording enabled (the default), each GCA row additionally records both axes,
forward/reverse scores and base densities, proposed/selector-accepted contact
labels, and `retained_tag_contacts/lost/gained` plus `completed_tag_exchange`
for the final move after all gates.
Search geometry counters count candidate evaluations; cross-score evaluations
are additional work included in elapsed times. No native labels enter these
contact definitions.

For example, from the repository directory:

```bash
cargo build --release --locked --bin tetramer-mc
target/release/tetramer-mc run \
  --config examples/spherical-conditioned-axis.json \
  --model examples/frozen-coverage-reciprocal-mixture.json \
  --out runs/conditional-axis-assembly \
  --sweeps 10000 --sample-every 100
```

This example adds the selector to the existing native-informed seeded
reciprocal-transport control. `--free-tetramers`, concentration and bath
overrides work as in the other assembly examples; the selector itself is
geometry-only. The whole example is not a template-free assembly test.

A separate diagnostic resets to the same frozen configuration on every attempt:

```bash
cargo build --release --locked --bin conditional-axis-benchmark
target/release/conditional-axis-benchmark \
  --config runs/body-envelope-cache-benchmark-20260924/aggregate48-config.json \
  --out runs/conditional-axis-benchmark/aggregate48.json \
  --attempts 128 --populations 4 --seed 2026092801
python3 tools/analyze_conditional_axis_benchmark.py \
  --input runs/conditional-axis-benchmark/aggregate48.json \
  --out runs/conditional-axis-benchmark/aggregate48-analysis.json
```

The three arms are unchanged isotropic GCA, contact-conditioned isotropic axes,
and contact-conditioned band axes. The attempt/population allocation applies
to each arm. Independent labels and RNG streams are declared in a manifest
before sampling. Attempt rows stream to JSONL; errors are recorded and remaining
attempts drained, with an unsuccessful completion flag if any error occurred.
Sampler CPU includes searches, rejected physical proposals and in-memory
result-to-JSON conversion. Streamed JSONL encoding/I/O is excluded. Exhaustive
all-pair diagnostic classification is timed separately. These frozen-state
measurements do not estimate equilibrium weights or contact-fingerprint ESS.

## Frozen protein benchmark: geometry improves, exchanges remain absent

The fixed allocation in `runs/conditional-axis-20260925/plan.json` completed
without errors: four independent populations per arm, 128 attempts per
population for N=12 and N=48, and 16 for N=264. A separate two-attempt cost
calibration is excluded. Every attempt starts at its archived snapshot; these
are independent conditional one-step tests, not trajectory populations.

| Snapshot and bath (rd A, z A^-3) | Attempts per arm | Isotropic CPU s | Conditional uniform CPU s | Conditional bands CPU s | Tagged exchanges in each arm |
|---|---:|---:|---:|---:|---:|
| Native-contact N=12 (1.5, 0.035) | 512 | 20.68 | 16.11 | 17.46 | 0 |
| Aggregate N=48 (1.4, 0.025) | 512 | 49.26 | 47.57 | 48.66 | 0 |
| Seeded N=264 (1.4, 0.0275) | 64 | 76.23 | 75.92 | 74.96 | 0 |

Total: 3,264 attempts and 426.84 sampler CPU seconds. Different archived baths
make these separate workloads, not a controlled size-scaling study. Shorter
CPU time in some guided arms partly reflects capped searches avoiding a
physical GCA call; it does not demonstrate better physical sampling.

The guide does find prospective exchanges:

| Snapshot | Qualifying selected axes, conditional uniform | Qualifying selected axes, conditional bands | Actual proposed tagged exchanges before selector gate |
|---|---:|---:|---:|
| N=12 | 54 | 257 | 0 |
| N=48 | 300 | 358 | 0 |
| N=264 | 40 | 42 | 0 |

In all **1,051** qualifying proposals, every original tagged partner shared
the tag's flip status. The tag flipped in 516 of these. Consequently their
old contact geometry was preserved even though the hypothetical tag-only
move would have exchanged partners. Sharing flip status need not mean two
labels belonged to the same component; separate fair coins can agree. Of
these proposals, 174 gained additional partners without losing old partners.
No actual proposed tagged exchange reached the final selector gate. Thus
the absence of exchanges here is not explained by rejection at that gate.
The mean largest component in N=264 was about 16% of the system, so this
failure does not require systemwide percolation.

All four populations in each arm gave zero exchanges. The per-arm Wilson 95%
interval upper bounds on probability per attempt are 0.00745 for N=12/N=48
and 0.05662 for N=264. These are individual finite-test intervals, not
simultaneous guarantees or bounds on equilibrium mixing. No nonzero
exchange-per-CPU rate or speedup was established.

The directly indicated refinement is a score favoring small residual old
overlap volume outside the complete shadow union: that region generates the
Poisson recruitment links. A Boolean new-contact predicate leaves its size
uncontrolled. This refinement has not been implemented or validated here.

Machine-readable analyses:

- `runs/conditional-axis-20260925/native12-analysis-v2.json`
- `runs/conditional-axis-20260925/aggregate48-analysis.json`
- `runs/conditional-axis-20260925/seeded264-analysis.json`
- `runs/conditional-axis-20260925/mechanism-summary.json`

The optional runner example also completed a four-sweep protein smoke test,
recorded under `runs/conditional-axis-20260925/runner-smoke`. These results
establish an implemented reversible selector and diagnose a remaining sampling
limitation. They do not determine native assembly's equilibrium stability.
