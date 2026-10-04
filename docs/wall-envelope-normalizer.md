# Optional wall envelope for the independent normalizer

`basin-normalizer --wall-uniform-envelope` now replaces the vessel proposal's
uniform cube branch with a normalized geometric envelope of the atomic-wall
domain. The flag requires `--wall-radius`. The default cube proposal, physical
wall, hard cores, depletant model, integration measure and assembly kernels are
unchanged.

This addresses the [completed sphere reference's weak defensive coverage](native-class-sphere-normalization-20261004.md).
It is implemented and has passed density/geometry/CLI checks. Its sampling
efficiency and stochastic normalization have **not** yet been established by
the new fixed reference. The old reference remains failed.

## Proposal and measure

Choose the first largest-radius atom, with body-frame center \(a\) and radius
\(r_a\), before drawing poses. For spherical wall center \(c\) and radius \(R_w\),
draw independent normalized Haar rotation \(R\) and uniform vector \(y\) in the
ball of radius \(b=R_w-r_a\). Set

\[
t=c-Ra+y,\qquad
q_E(t,R)=\frac{\mathbf1\{|t+Ra-c|\le b\}}{4\pi b^3/3}.
\]

The conditional translation is a shift with unit Jacobian. Its normalized
volume is known. Every fully wall-valid pose must place this atom inside the
wall, including shapes with offset origins, unequal atom radii or concavities.
The envelope does not condition on other atoms, fixed neighbors, or depletion.
Their invalid poses remain zeros in the full attempted-draw denominator.

If \(\epsilon\) is the existing vessel uniform probability and \(G_j\) is the
complete learned law at fixed anchor \(j\), the vessel density is

\[
q_v=\epsilon q_E+(1-\epsilon)\frac1m\sum_jG_j.
\]

Both contributions are evaluated directly; the code never subtracts the cube
term from a nearly equal total. Reciprocal branches remain in \(G_j\). An atlas
draw outside the envelope still has its learned density and may have zero
physical weight. With the class guide, the complete proposal is still
\(q=\tfrac12q_v+\tfrac12q_{\rm class}/J\), evaluated on every draw. The estimator
is the unchanged arithmetic two-cloud weight divided by this complete density.

The implementation retains existing vessel branch/anchor coins and learned
candidates. On the uniform branch it replaces the preliminary cube candidate
by one envelope draw from a separately named RNG stream, before any physical
evaluation. A failed preliminary proposal, unrepresentable draw, or loss of
support during the capture/world coordinate roundtrip is fatal. There are no
retries or validity-conditioned redraws.

## Interface, records and validation

Use the new flag alongside the existing command's wall and guide arguments:

```sh
basin-normalizer ... --wall-radius 223.3261767 --wall-uniform-envelope
```

The ellipsis represents the existing required configuration/model/output
arguments; this is an option example, not a production launch command.

The manifest uses **schema 8**, records `pre_envelope_schema`, and archives the
atom index/center/radius, wall, envelope radius and log volume. This prevents old
auditors from silently applying the cube density. The Python implementation
reconstructs the atom choice and volume from the archived shape and wall, checks
both saved-world and reconstructed-candidate support, and independently scores
all anchors. Stable hypot norms prevent underflow from falsely admitting tiny
exterior points and avoid overflow on large finite exterior points.

Nine Rust tests passed in an isolated target. They cover exact sphere density,
offset/asymmetric shapes, wall-support inclusion, deterministic atom selection,
unit translation Jacobian, Haar/radial moments, complete branch mixing,
pure-uniform exterior zeros and exact disabled-option density parity.
[Receipt](../results/wall-envelope-validation-20261004/attempt02/validation.json).

The CLI check retained **80 new toy attempts**: 32 hard class-guide, 32 depletion
class-guide and 16 pure-uniform wall draws. All journals, hard-invalid zeros,
cloud weights and complete mixture densities were checked. Independent class
reconstruction's maximum log-density difference was 1.82×10⁻¹². The pure-uniform
hard sphere has the expected constant inverse density on valid poses.
[Receipt](../results/wall-envelope-cli-validation-20261004/attempt01/validation.json).
This is a wiring check, with no stochastic mass-precision or efficiency gate.

After strengthening the Python boundary and extreme-scale checks, **35 Python
tests** and re-audits of the same 64 class-guide poses passed, with zero new draws.
[Receipt](../results/wall-envelope-saved-audit-20261004/attempt02/validation.json).
Earlier audit-source versions were retained before editing. The production and
active observer executables were not rebuilt.

Validation currently covers **no latent guide** and the **native-class guide**.
The generic Rust composition also accepts Gaussian/hard-free guides, but the
independent schema-8 auditor deliberately refuses those combinations. They are
not admitted to a physical campaign until their audit path is extended and
validated. Existing guide formats remain available with the default cube law.

Real-arithmetic normalization does not prove floating-point geometry, the random
generator, inverse-map arithmetic or Poisson thinning. Those remain explicit
implementation obligations, with failure stopping the allocation.

## Prospective reference allocation

For the same sphere target, the envelope volume is 82.44796 rather than the
cube's 1,000. At total defensive probability 0.2, its density floor is 12.13
times higher. This is a proposal-density bound, **not a measured speedup**.

A deterministic calculation and independent mathematical review give the
following fixed design for a new reference, with fresh independent seeds:

| Activity | Independent populations | Attempts per population |
| --- | ---: | ---: |
| 0 | 4 | 12,288 |
| 0.4 | 4 | 20,480 |

Total: **131,072 attempts**, two independent clouds per valid pose, unchanged
radius/regions and lambda/activity ratio 16. This design has not been launched;
it requires its own frozen execution plan. The completed cube allocation will
not be extended or rerun to replace its failure.

Here is the explicit statistical assurance, rather than a budget inferred from
whether a previous realization happened to pass. For each of the five radial
regions, let \(V_A\) be its hard volume and \(a_A=V_E/(0.2V_A)\). Conservatively,
\(e^{zV}\le M=1.49\), \(Q_A(2z)/Q_A(z)^2\le K/V_A\) with \(K=1.041\), and the
two-cloud conditional variance inflation is at most \(\eta=0.013\). Thus hard
relative variance is bounded by \(a_A-1\), exact-weight relative variance by
\(Ka_A-1\), and noisy-minus-exact mean relative variance by \(\eta Ka_A/N\).

Take \(\delta=1-e^{-0.1}\), reserve \(\rho=0.022\) for the Poisson residual, and
bound the exact contribution at \(\delta-\rho\). Bernstein's inequality applies
to bounded exact/hard contributions, while Chebyshev applies to the **unbounded**
Poisson residual. A union bound across all five regions and triangle inequality
for each noisy estimate yield **0.0449913** as an upper bound on missing any of
the pooled 0.1-log point requirements. Independence between overlapping regions
is not assumed. At zero activity, hard/exact/noisy estimators are identical.

This bound assumes the exact normalized proposal and independent unbiased cloud
laws. It is not a guarantee of passing the separate empirical four-SE checks,
and it has no implication for protein convergence or assembly stability.
The conservative constants, region contributions, derivation assumptions and
source hash are in
[allocation.json](../results/wall-envelope-allocation-design-20261004/allocation.json),
computed by the accompanying `bounds.py` without sampling.

The original protein weight campaign remains frozen at its own conditions and
proposal. Its remaining sensitivity checks, full-vessel mass coverage and
finite-system assembly evidence remain separate requirements.
