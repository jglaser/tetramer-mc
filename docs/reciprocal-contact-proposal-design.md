# Exact reciprocal contact proposals

**Implemented and reference-tested for frozen spherical capture/posterior
proposals.** The [completed validation record](../runs/reciprocal-completed-validation-20260921/validation.json)
binds 54 passing Rust tests, 53 distinct Python checks, the executable/source
bundle, and all eight audited mobile runs. The earlier mobile atlas control
used a first-order reciprocal Gaussian; its archived inputs and conclusions
are unchanged. A [new matched control](../runs/mobile-reciprocal-atlas-benchmark-20260921/protocol.json)
wraps all 150 unchanged base components, producing 300 exact virtual branches.
Both controls remain native-informed. Reciprocal representation alone does
not establish template-free discovery, adaptation, or equilibrium.

The implemented format is a `reciprocal-pose-mixture-v1` envelope containing
`base_model` and one Boolean `reciprocal_components` flag per base component.
Older Gaussian readers reject the envelope because it has no top-level
Gaussian arrays. Active reciprocal models reject periodic boundaries and
unsupported adaptive or single-Gaussian region consumers. The whole-domain
`basin-normalizer` now supports unchanged reciprocal models at covariance
scale 1, with complete anchor-marginalized importance density. Bare legacy models and envelopes
whose flags are all false retain their previous draw and trace behavior.

Capture records keep base-component indices plus an optional inversion flag.
Posterior traces use virtual indices, ordered by base component with ordinary
then inverse branch; active reciprocal traces also record base indices and
inversion flags. Density evaluation and source responsibilities always include
every virtual branch. These index conventions matter when replaying a move.

## Physical measure and reciprocal representation

For a relative pose `T=(t,R)` of identical rigid particles, define

```
I(T) = (-R^T t, R^T),          I(I(T)) = T.
```

Both rotations are proper. This operation changes the ordering of a relative
pair description; it is not a reflection, a change of chirality, or permission
to swap the physical moving body with its anchor during an update.

Use the measure `dµ(T)=d³t dHaar(R)`, with normalized SO(3) Haar measure.
Rotational inversion preserves Haar measure. At fixed rotation, translation
has derivative `-R^T`, with absolute determinant one. Rotation does not depend
on translation, so the translation–rotation cross derivative does not change
the absolute determinant of the full block-triangular map. Thus I preserves µ.
No determinant of a Cartesian quaternion parameterization should be inserted.

For a normalized base mixture `G(T)=sum_k w_k g_k(T)`, an exact reciprocal
mixture is

```
F(T) = Gsym(T) = 0.5 G(T) + 0.5 G(I(T)).
```

It is normalized and satisfies `F(I(T))=F(T)`. The second term is generally
not a Gaussian in the original translation/Cayley coordinates. Transforming a
Gaussian mean and covariance cannot implement this identity exactly.

Store each base Gaussian once and represent its two virtual branches by
labels `(k,s)`, where `s` is zero or one:

```
g_(k,s)(T) = g_k(I^s(T)),       alpha_(k,s) = w_k/2.
```

The smallest useful generalization can also leave selected legacy components
unwrapped, with one branch of weight `w_k`. A new contact family of total
weight 0.2 can then contribute exact forward/reciprocal branches of weight
0.1 each, while preserving the original 150 components at total weight 0.8.
That full atlas is a normalized mixture but need not itself be reciprocal;
only the selected contact family is. Symmetrizing every family gives the F
above. The representation and allocation must be explicit in the frozen model.

## Independent capture branch

To draw from a symmetrized family, draw its base Gaussian pose and apply I
with probability one half. For a selected fixed anchor j, transform the
resulting relative pose back into the laboratory frame in the usual way.
Retain the base-component index and inversion label in the move record.

The existing capture proposal must continue to use

```
Q_j(x) = eta U_lab(x) + (1-eta) F(T_j(x)),
log correction = log Q_j(old) - log Q_j(new).
```

The outer uniform cube/Haar draw is unchanged. Inverting a laboratory uniform
draw would change its support and distribution; it is not the construction
above. Evaluate the full learned mixture and the full uniform-plus-learned
density even when a particular component or inversion branch generated the
candidate. The selected component's density is not the Hastings denominator.

## Correlated posterior branch and balance

Let `D_k(z)` be the existing base chart decoder, with standard Gaussian latent
coordinates and physical volume Jacobian `J_k(z)`. The virtual chart is

```
D_(k,s)(z) = I^s(D_k(z)),
E_(k,s)(T) = E_k(I^s(T)).
```

Because I preserves physical measure, its chart volume remains `J_k(z)`.
For old relative pose T, select a joint source label `(a,s)` with probability

```
p_source(a,s | T) = alpha_(a,s) g_a(I^s(T)) / F(T).
```

Select destination `(b,t)` independently with probability `alpha_(b,t)` and
draw independent standard Gaussian noise ξ. With correlation c and
`h=sqrt(1-c²)`, apply

```
z  = E_a(I^s(T))
z' = c z + h ξ
ξ' = h z - c ξ
T' = I^t(D_b(z')).
```

The inverse retains `(b,t)` as source, `(a,s)` as destination, and ξ' as
noise. The joint latent/noise map is the existing orthogonal involution. Its
extended physical Jacobian is `J_b(z')/J_a(z)`; the two reciprocal wrappers
add zero log absolute Jacobian.

The base chart/noise correction is

```
log[phi(ξ')/phi(ξ)] + log[J_b(z')/J_a(z)]
    = log g_a(I^s(T)) - log g_b(I^t(T')).
```

The reverse/forward source-and-destination label ratio cancels these selected
component terms and the alpha factors. The complete Gaussian-branch result is

```
log correction = log F(T) - log F(T').
```

The source inversion label is generally **not** a fair coin: its probability
depends on the directional densities at the old pose. A fair destination
inversion coin is valid for a symmetrized family. Implementing only the fair
source coin while retaining the full-F correction would use the wrong law.
Use the existing log-category sampler over all supported virtual labels.

At c=0 the Gaussian-branch candidate is an independent draw from F. The
separately selected posterior uniform branch retains its existing support and
zero correction; it does not supply a uniform floor inside this Gaussian
correction. The independent capture kernel still uses Q, not F alone.

The physical acceptance remains the existing conditional Poisson gate plus
this proposal correction. All spectators remain in the exclusion union and
hard test, and the original atomic wall remains active. A reciprocal proposal
does not make the physical environment pairwise or remove its depletion factor.

## Implemented architecture

The existing Gaussian map can remain the numerical core. Prefer a thin,
explicit reciprocal wrapper over pretending an inverted component is another
Gaussian with transformed covariance.

1. In `src/math.rs`, add a clearly named whole-relative-pose inversion helper.
   Existing `Pose::inverse(x)` transforms a point and is not this operation.
2. In `src/proposal.rs`, extend the frozen representation with a branch table
   linking virtual labels to stored base Gaussians and inversion flags.
   `relative_log_density_unchecked` sums all branch densities in log space;
   `log_density` retains the outer uniform law; `propose` inverts only learned
   relative draws before the laboratory transform. Factor a base-component
   density/draw interface so capture and posterior code use the same semantics.
3. In `src/docking.rs`, update `DockingProposal::new` and `propose` to construct
   joint label probabilities and retain both inversion flags. Compose I before
   and after `FixedBasinInvolution::apply` in `src/basin_involution.rs`. Its
   `encode`, `decode_and_log_volume`, latent update, and Gaussian auxiliary
   correction need not change. Wrap its inverse trace with swapped parity
   labels. Record wrapped physical poses as well as the underlying base step.
4. Keep `src/simulation.rs::posterior_for_body` responsible for global anchor
   indices, and leave its physical environment/gate path unchanged. Initially
   support only immutable open/spherical models and the capture/posterior
   kernels; reject unsupported periodic or adaptive consumers explicitly.
5. Extend Python `Density.evaluate`/`draw_component` in
   `tools/prepare_smc_normalizer_atlas.py`, `ChartAudit` replay in
   `tools/analyze_involution_docking_campaign.py`, and the mobile/docking
   density auditors before using the representation in an audited campaign.
   A finite latent-region definition around a virtual chart also requires an
   explicit decoder/encoder wrapper; existing Gaussian-only region files must
   not silently acquire this meaning.

Treat base component count and virtual branch count as different quantities.
Do not make `component_parameters()` claim that a nonlinear branch has lossless
Gaussian parameters. Parameter replacement, shifted models, component subsets,
and contact-memory helpers must either preserve the representation explicitly
or reject it in the initial implementation.

The JSON representation must be versioned and fail closed in unsupported
consumers. `RawModel` currently ignores unknown JSON fields, including a new
ad hoc symmetry flag; older binaries could silently read the base Gaussians
only. A new `schema` string alone would not fix that parser behavior. Use a
compatibility marker that existing consumers actually reject, or an explicit
new loading path, and bind feature-capable binaries and Python observers in
the campaign archive. Omitting the new feature should retain old draws and
serialized trace behavior.

## Pitfalls requiring explicit checks

- **Identity shortcut:** `DockingProposal::propose` currently treats
  `c==1 && source==target` as identity. With wrapped labels this is identity
  only when both the base index and inversion flag match. Equal base indices
  with different flags produce a genuine reciprocal-pose move. Erasing that
  move would invalidate the intended trace and benchmark behavior.
- Invert in the anchor-relative frame. Keep the physical anchor fixed and the
  moving-body index unchanged; a parity label is not an anchor-selection rule.
- Evaluate every label density after the appropriate inversion, including
  nonlinear translation–rotation coupling. Keep the physical Haar conversion
  once, inside each base chart density; do not add a second inversion Jacobian.
- Null seams and numerical failures remain recorded self-loops or explicit
  errors under the declared policy. Do not retry, truncate responsibilities,
  drop duplicate labels, or renormalize over hard/wall-valid candidates.
- A lab cube or unique-image periodic null policy is not automatically
  reciprocal. Open/spherical support is the appropriate first implementation.
- Freezing discovered geometry for a new campaign is different from adapting
  a proposal during a chain. Birth/death, compression, online learning, and
  template-free discovery need their own state, inverse/support, or frozen
  epoch argument; this representation supplies none of those automatically.

## Reference-test requirements

| Control | Required discriminating evidence |
|---|---|
| Reciprocal geometry and measure | Generic nonzero translations/proper rotations; I² recovers the pose up to quaternion sign; independent six-dimensional physical differential has unit absolute determinant; chirality remains proper. |
| Density normalization and symmetry | Unequal weighted, overlapping, full-covariance Gaussians with nonzero means; independently evaluate `0.5(G(T)+G(I(T)))`, verify normalization and `F(T)=F(I(T))`; include a case where linearized reciprocal-Gaussian density measurably disagrees. |
| Independent draws | Frequencies of base and inversion labels, reciprocal pose moments, full Q correction with nonzero cube defense, nonzero lab anchor and capture center, invalid zeros, and unchanged uniform draws. |
| Source responsibilities | Empirical joint `(k,s)` probabilities match the full posterior; include a strongly asymmetric old pose that rejects an erroneous fair source-parity coin. |
| Correlated inverse trace | Every source/destination parity combination, same and different base charts, c in `{−1,−0.6,0,0.9,1}`; recover pose, labels, noise, and cancel forward/reverse log corrections. Explicitly test c=1 same-base/opposite-parity as nonidentity. |
| Expanded balance | Component/noise/Haar/label calculation agrees independently with `log F(old)-log F(new)`; Gaussian-branch target F has unit acceptance and preserves independently drawn F observables. At c=0, destination latents are independent standard normals. |
| Physical target | Extend `tests/involution_depletion.rs` with independent analytic sphere/lens references and unequal branch weights; retain an active non-anchor spectator. Negative controls omitting parity responsibility, nonlinear inversion, Haar, or the spectator must fail. |
| Assembly integration | Extend `tests/frozen_posterior_assembly.rs`: all moving/anchor indices, actual per-slot schedule, full gate replay, null/rejection records, wrapped trace replay, and exact restart. No-option and explicitly disabled controls preserve old physical draws. |
| Serialization and provenance | Round-trip base/virtual mapping; unsupported old/periodic/adaptive readers reject the representation; archived Rust and Python evaluators agree; raw model hash and embedded feature-capable source bundle are bound. |

Passing these controls would validate a frozen reciprocal proposal
representation. Sampling efficiency, discovery, reversible compression, and
assembly remain separate empirical and algorithmic questions.

The implemented tests include numerical six-dimensional volume preservation,
exact density normalization/symmetry, source responsibilities, all inversion
label combinations, correlations from −1 through 1, independent mixture
reference samples, analytic sphere and many-body depletion references, and
actual all-mobile runner replay/restart. The source/destination wrapper tests
explicitly cover a common base index with opposite inversion labels at c=1:
that update is a real pose inversion, not an identity shortcut. The initial
test compile failure used a non-cloneable RNG in test code; the corrected test
uses two independently constructed same-seed RNGs. No physical run used that
failed build.
