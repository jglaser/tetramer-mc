# Fresh physical validation of hard-free line conditioning

The [feasibility-only guide](hard-free-line-guide.md) preserves the original
Gaussian mixture and conditions one translation coordinate on the complete
hard-free line. Saved-pose scores justify fresh validation, but do not measure
physical sampling efficiency. The assembly kernels and main production binary
remain unchanged.

The production normalizer now recognizes
`defensive-hard-free-line-guide-v1` as a separate proposal schema. Its estimator
is the unchanged unconditional average of `I_valid J W/q`, with two independent
clouds at every valid pose. The complete mixture density includes all Gaussian
components and all declared axes. Invalid and exterior attempts contribute
zero; their denominator entries remain. A compact trace saves every axis's
whole hard-free interval union without serializing 92 component records per
axis. The independent Python audit reconstructs those unions from atomic
geometry, as well as the density, coordinate Jacobian, selected inverse CDF,
physical predicates and cloud-weight arithmetic.

An attempted-draw journal is written before geometry or cloud work. Completed
rows are flushed individually. A handled error records the failed draw and
cloud context and retains prior rows. These records support failure auditing;
they are **not a restart or continuation implementation**. New campaigns refuse
to retry or overwrite a started population.

## Fixed prospective allocation

The new physical controller remains gated on exact-binary sphere references,
the independent production audit, and the completed fresh proposal-only pilot.
No physical protein draws are authorized by a mere successful source import.

| Arm | Conditional probability | Populations | Draws per population |
|---|---:|---:|---:|
| Original Gaussian92 law through the new interface | 0 | 4 | 16,384 |
| Hard-free xyz mixture | 1 | 4 | 16,384 |

Both retain 50% uniform R4 support, the original 92 means/covariances/weights,
radius 1.5 Å, activity 0.035 Å⁻³, the repaired tetramer, scaffold, capture domain,
region and full native classifier. The new guide has no selected contact labels
or widths. The physical allocation is 131,072 fresh attempted draws, two clouds
per valid pose, auxiliary intensity/activity ratio 128. At most two physical
jobs or four subsequent independent audit jobs run concurrently, under the
existing eight-physical-job/32-worker global caps. Native and exclusion-contact
classification is a separate single pass over valid new poses.

The old Gaussian92 populations informed the choice of this guide. They remain
retrospective evidence rather than serving as the only control for an independent
efficiency comparison. Their completed physical draws, audits and classifications
are reused without replay or pooling. Both new arms use previously undeclared
seeds and fixed sizes; the controller cannot fit proposals or extend allocations
after observing results.

The analysis retains native, contact-without-native-entry and unbound classes,
the old-R5/native-complement partition, and all original radial, angular and
orthant strata. Importance ESS per sampler CPU is reported separately from
audit and classification costs. Numeric nonuniform branch membership is
preserved but displayed as `nonuniform_guide`, so the conditioned branch is not
mistaken for an unmodified Gaussian draw.

The original relative-error, ESS, largest-contribution, agreement and
population-interval criteria remain. A favorable small-pilot result still leaves
population-size, cloud-intensity, unseen-mass and full-vessel checks outstanding.
Neither a passing proposal reference nor a failed weight diagnostic establishes
finite-system assembly or instability. This integration atlas remains
historically native-informed; it is not a geometry-only assembly proposal.

Implementation: [controller](../tools/run_hard_free_line_physical_pilot.py),
[independent physical audit](../tools/hard_free_line_physical_reference.py), and
[single-pass contact analysis](../tools/analyze_hard_free_line_physical.py).

## Started campaign

The prerequisite checks passed and the immutable controller passed preflight.
The background campaign is
`/vast/xvg/tetramer-mc-runs/hard-free-line-physical-pilot-20261001`;
its protocol SHA-256 is
`1b788b08df239319ac69126627e891893b214c35204e337c95f4a89ede23c050`.
Check `status.json` and `controller.log`; do not rerun a started campaign.
The exact executable SHA-256 is
`752c5d7aa36249cb923ca0959b0872b4f96204101090ac1cfca4606732554c7b`.
The [prerequisite receipt](../results/hard-free-line-physical-prerequisites-20261001/prerequisites.json)
binds reference tests, independent audits, the fresh protein proposal comparison,
and the five controller contract checks. No physical comparison result is
claimed until the entire declared allocation and subsequent checks complete.
