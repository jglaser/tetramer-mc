# Retained-index comparison on the completed regional pilot

The existing regional draws can evaluate an indexed estimator without new
poses, clouds, geometry or native classifications. The draw law stays unchanged.
The new tool requires the entire controller, all eight independent audits and
the saved classification pass to be complete before reading any population.
It writes a separate result; the frozen full-mixture weights and convergence
gates remain unchanged.

## Estimator and support

Let the retained primitive label be the uniform branch, an unconditioned
Gaussian component, or a conditioned Gaussian component and axis. Its prior
probability is \(\eta_\ell\). Write its original density as \(q_\ell\), its
actual sampling density as \(h_\ell\), and

\[
Q_0(u)=\sum_\ell\eta_\ell q_\ell(u),\qquad
Q_1(u)=\sum_\ell\eta_\ell h_\ell(u),\qquad
\psi_\ell(u)=\eta_\ell q_\ell(u)/Q_0(u).
\]

On physical support in the original R4 region, successful conditioning gives
\(h_\ell=q_\ell/Z_\ell\). The conditional Normal interval mass \(Z_\ell\)
depends only on the five retained raw coordinates. Uniform, unconditioned and
fallback branches instead have \(h_\ell=q_\ell\); define their effective
\(Z_\ell=1\), even when the rejected conditioning mass was zero.

For the physical integrand \(F\), the old-responsibility auxiliary target is
\(F(u)\psi_\ell(u)\). Dividing by the actual joint sampling density gives

\[
W_{\rm index}=F(u)Z_\ell/Q_0(u)
=W_{\rm full}\exp[\log Q_1(u)-\log Q_0(u)]Z_\ell.
\]

Summing over labels recovers \(\int F\), provided every label has actual
support wherever its auxiliary target is positive. This condition holds for
the hard-free regional construction. **It does not hold for the unchanged
full vessel**, where a conditioned component can vanish outside the source
region although its original Gaussian remains positive. The tool rejects the
full-vessel schema rather than extending the proof by assumption.

The physical cloud estimator is conditionally unbiased given pose and label;
the implementation uses separate indexed RNG roles for pose generation and
each cloud replica. The same multiplicative correction therefore applies to
the saved two-cloud mean, both individual cloud weights, and the hard-volume
weight. Every invalid or exterior attempt stays a zero in its original
denominator. The beta=0 control is exactly unchanged.

The correction uses the **saved selected draw mass**, not a new evaluation
after a coordinate roundtrip. This preserves the actual implemented draw's
normalization. Existing independent audits cover the geometric and numerical
obligations; the retrospective performs only arithmetic and hash checks.

## What the comparison measures

Full and indexed weights use the same pose, label and clouds. Their errors are
correlated. The report includes paired linear population-mean differences and
sample standard errors, along with separate four-population mass estimates,
importance ESS, largest contributions, native/competing ratios and two-cloud
noise diagnostics. An unobserved region remains unresolved.

In exact arithmetic,
\(E[W_{\rm index}\mid u,\text{clouds}]=W_{\rm full}\): the full estimator
Rao–Blackwellizes the retained index. Thus index weighting cannot improve its
true variance at a fixed draw count. Observed finite-sample ESS or second
moments can nevertheless reverse that ordering. Ratios and free energies
computed from the means are diagnostics with finite-sample bias, rather than
additional unbiased estimators or independent physical evidence.

The production rows retain total sampler CPU, but no per-stage density or
cloud timing. Both retrospective ESS/CPU summaries use that same actual
full-scorer CPU. Source and draw labels determine exact geometry-request
counts: the indexed implementation could remove the all-axis density
traversals while retaining any selected-axis traversal and cheap \(Q_0\).
Those counts do not quantify time saved. An index-only CPU speedup remains
unmeasured; any earlier proposal-only timing is a different workload.

## Dormant command

After all completion gates pass:

```bash
python tools/report_hard_free_line_physical_index.py \
  --execution /vast/xvg/tetramer-mc-runs/hard-free-line-physical-pilot-20261001 \
  --out results/hard-free-line-physical-index-retrospective-20261001
```

The output directory must be fresh. The tool never starts physical jobs,
replays classifiers, or changes the completed campaign.
