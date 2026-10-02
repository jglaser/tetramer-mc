# Joint dimer map: combined equilibrium reference

The combined learned two-edge map and general world-space Poisson gate completed
an analytic two-sphere equilibrium control. Every attempted row passed independent
Python reconstruction of the complete proposal-density ratio, Metropolis decision,
state continuity and retained rejection. The largest log-density error was
1.60e-14. This supports this implementation on the reference problem; it does not
demonstrate improved protein sampling.

The physical system contains two spheres, each with core radius 0.2 and exclusion
radius 1. The root center lies inside a ball of radius 2; root-child distance lies
in [0.4, 2.5]. The child has no separate wall. The tree's anchor is a virtual
coordinate frame, not a third physical sphere. Root volume, relative radial
measure and orientation Haar measures factorize, giving the exact distance law

\[
p(d)\propto d^2\exp[1.5\,V_{\mathrm{overlap}}(d)].
\]

The factor d² belongs in this reference integral, not in the pose-space MH
correction. Independent quadrature gives mean distance 1.223173497 and contact
probability P(d<1)=0.484047109. The root's mean squared distance is 2.4. All
quaternion squared components have Haar expectation 0.25.

## Frozen allocation and results

Eight independent streams used two contact and two dispersed initializations in
each of two arms: analytic depletion or the conditional Poisson gate with
lambda=24 and z=1.5. Each chain has 2,048 burn-in and 16,384 production attempts:
147,456 total. A frozen equal-weight four-chart Gaussian atlas centered on the
quaternion basis, covariance I6 and correlation 0.7 supplies both relative-pose
maps. Each joint MH decision is followed by independent exact Haar refreshes at
fixed world centers. Angular moments are measured **before and after** those
refreshes, allowing checks of the monitored moments before refresh. Higher-order
or joint angular bias could still escape these second-moment diagnostics.

| Production diagnostic | Analytic depletion | Poisson gate |
|---|---:|---:|
| Contact probability | 0.47980 | 0.47093 |
| Four-stream SE | 0.00687 | 0.00545 |
| Population 95% interval half-width (t, 3 df) | 0.02187 | 0.01734 |
| Mean distance | 1.22489 | 1.24318 |
| Mean root squared distance | 2.39997 | 2.39876 |
| Completed roundtrips across d=1 | 3,362 | 2,835 |
| Contact ESS, summed 1,024-step batch estimates | 7,966 | 5,067 |
| Production CPU seconds | 2.219 | 8.316 |
| Contact ESS / CPU second | 3,590 | 609 |
| Raw Poisson points | 0 | 98,416,722 |

These CPU figures include the attempted-state logging between samples. The
analytic arm is a correctness/cost reference with an unavailable exact protein
energy, not a competing protein algorithm. Within the Poisson arm, individual
streams complete 691--744 roundtrips and have contact ESS estimates 853--1,867.
These threshold roundtrips are not full binding/unbinding events: d>=1 includes
poses whose exclusion spheres still overlap. The intervals use independent
stream means; the batch estimates retain all
rejections. Both contact probabilities agree with the exact reference.

Four of 38 monitored arm/moment checks exceed three **between-stream** SE:
Poisson mean distance, two pre-refresh child quaternion moments in the analytic
arm, and one post-refresh root quaternion moment in that arm. Their deviations
are 1.05--2.79 of the separately reported 1,024-step batch SE. The post-refresh
example is particularly instructive: its deviation from 0.25 is only 0.000983,
versus the known independent-Haar SE 0.000977, despite being 4.35 estimated
between-stream SE. Four streams can give a noisy uncertainty estimate. These
flags remain in the report; none was removed by changing a cutoff or extending
the allocation. This is not an all-diagnostics-pass convergence certificate.

## Reproducibility and scope

The [example](../examples/dimer_tree_equilibrium.rs) rejects existing output
directories, separates proposal/cloud/MH/Haar random streams, logs every attempt
and writes a failure row before terminating. Three example contract tests and
four independent Python analysis tests pass. A separate
[composed accepted-flow test](../tests/dimer_tree_flow.rs) uses an actual nonunit
tree-proposal correction and inverse trace with 12,000 clouds in each direction;
the analytic flow identity passes, including the zero-activity limit. That
complements the earlier seven map and six general-gate tests.

The frozen campaign is
[`results/dimer-tree-equilibrium-20261002`](../results/dimer-tree-equilibrium-20261002).
Its [completed review](../results/dimer-tree-equilibrium-20261002/completed-review.json)
authenticates 84 frozen source/dependency files, the executable, all eight raw
ledgers and the [independent analysis](../results/dimer-tree-equilibrium-20261002/analysis.json).
The allocation was frozen before sampling:

- Plan SHA256: `4efb0d00314725759aff46e573257bde0466afc8cbe933f7765e4437db1a1684`.
- Config SHA256: `d136a3b0d733b25ecbe8fc50b9a0808708005e6282ccd496f82ead58a6fae4bb`.
- Executable SHA256: `ab5aaae640615694d8da6e8be15f6f0786b034afbaf095ab76de02530e262ca0`.
- Analysis SHA256: `8e1f47d9c8353d8a40497d270c1f23171d5c31f072f1c63561b6a926bd336704`.

For a new development reproduction, with a fresh output directory:

```sh
CARGO_TARGET_DIR=target-validation-line-guide CARGO_BUILD_JOBS=4 \
  cargo build --offline --locked --release --example dimer_tree_equilibrium
target-validation-line-guide/release/examples/dimer_tree_equilibrium \
  --config examples/dimer-tree-equilibrium.json --out /tmp/dimer-tree-reference
/home/xvg/protein-nucleation/.venv/bin/python -B \
  tools/analyze_dimer_tree_equilibrium.py \
  --config examples/dimer-tree-equilibrium.json \
  --directory /tmp/dimer-tree-reference --out /tmp/dimer-tree-reference-review.json
```

The existing production executable, assembly schedules, protein contact-weight
campaigns and physical parameters were unchanged. Before a protein campaign,
the world-gate cost should be compared with the two singleton-body auxiliary
paths described in the [proposal derivation](dimer-tree-proposal.md). A saved
endpoint cost comparison can guide that decision without generating clouds.
