# Independent native-neighborhood comparison: design and geometry screen

The minimal empty/B/AB screen was subsequently executed with the fixed
budgets below; ABC remains unrun. Its [result report](native-factorial-screen.md)
records the unresolved AB weight and completed controls. The prepared-only
manifest remains unchanged as the historical design record.

The existing complete-native cover applies without alteration to the site-0
M2 and M3 environments. Their native reference, four rigid-member centers,
capture sphere, and coordinate frame match the M1 experiment exactly. The
fixed-neighbor lists are nested: M1 is A, M2 is A+B, and M3 is A+B+C.

Configuration-only variants are prepared under
[native-cooperativity-plan-20260920](/home/xvg/tetramer-mc/runs/native-cooperativity-plan-20260920).
Only fixed poses were transplanted from the archived environments. Their
historical bath fields are **not** the current bath: the prepared variants
retain radius 1.5 Å and activity 0.035 Å⁻³, with lambda/z=64 and two clouds.
The original 2 Å / 15° native metric, shape, and 18 Å capture sphere remain
identical. `manifest.json` records source hashes, configurations and proposed
budgets. These configurations were prepared before any new physical
production; the later execution has its own manifest and no Gaussian fit.

## Existing-pose geometry screen

The prior 512-pose native SMC snapshot had only five survivors after adding B
and three after adding B+C, but those particles represented its concentrated
M1 distribution. A new hard-only check uses 2,048 uniformly selected poses
from the 20,041 hard-native configurations in the independent uniform-cover
campaign. It does not generate new poses or change any original weight.

| Existing M1-hard-native subset | Added B survives | Added B+C survives |
|---|---:|---:|
| Uniform subset, n=2048 | 725 (35.40%) | 313 (15.28%) |
| Twenty leading M1-weight poses | 1 | 0 |

The geometric standard errors of the first row are approximately 1.06 and
0.80 percentage points (ignoring the small finite-population correction).
This is a conditional hard-support diagnostic, not an M2/M3 equilibrium
population or normalizer. The leading subset is deliberately selected and
cannot be interpreted as a probability sample. It shows why reusing M1
Boltzmann weights for a larger neighborhood would be misleading.

The independent atom-radius-group KD-tree check took 15.12 s. Source hashes,
per-pose collision flags and negative-gap witnesses are retained in
[native-neighbor-survival-screen-20260920](/home/xvg/tetramer-mc/runs/native-neighbor-survival-screen-20260920).
The reproducible tool is `tools/native_neighbor_survival_screen.py`.
Its reusable input guards now require the native campaign's shape hash,
original M1 neighbor, member/reference metric and capture region to match.
Bath fields are deliberately ignored for this hard-only screen. The uniform
subset seed is 98670920; the extracted-pose file SHA-256 is recorded. The
original executed source was archived before adding these guards, and an
`input-guard-audit.json` confirms the existing run satisfies them without
rerunning or replacing its geometry results.

## What the factorial comparison measures

Let Ω be the same complete native region in every calculation, including the
same capture restriction. Define, for a fixed-neighbor subset S,

\[
 Q_S(z)=\int_\Omega H_S(x)\exp[zC_S(x)]\,dx\,dR,
 \quad C_S(x)=|E(x)\cap\bigcup_{j\in S}E_j|,
\]

using normalized Haar measure. The empty-neighbor mass Q_empty is purely
geometric and independent of activity. The contrast

\[
 \log\mathcal C(z)=\log Q_{AB}(z)+\log Q_{empty}
 -\log Q_A(z)-\log Q_B(z)
\]

is conditional free-energy cooperativity on Ω. It includes shared-pose
correlations and hard support, and may be nonzero even for additive pair
energies. Its free-energy convention is `ΔΔF=-log C` in kBT units. It is not
an assembly free energy for creating A and B from bulk, nor a global native
probability. Fixed-neighbor background free energies are not included in
these conditional insertion normalizers.

Every production campaign also estimates its z=0 geometric volume from the
same hard/native/capture indicators. Therefore no separate hard-only runs
for A, B or AB are necessary. Reporting C(0) alongside C(z), and the
individual depletion enhancements `Q_S(z)/Q_S(0)`, separates hard-support
effects from activity-dependent changes. It still does not isolate the
many-body part of depletion.

## A distinct control for true depletion nonadditivity

At any fixed pose, the exact overlap obeys

\[
 C_{AB}=C_A+C_B-T_{AB},\qquad
 T_{AB}=|E(x)\cap E_A\cap E_B|\ge0.
\]

The proper pair-additive counterfactual uses the **same** hard support:

\[
 Q_{AB}^{pair}=\int_\Omega H_AH_B\exp[z(C_A+C_B)]\,dx\,dR.
\]

Then `Q_AB≤Q_AB^pair`, and `−log(Q_AB/Q_AB^pair)≥0` is the many-body depletion
penalty for this conditional region. The factorial contrast alone cannot
supply it. This extra estimator is optional later work, not part of the
minimal configuration-only comparison.

Two independent single-neighbor positive weights at the same pose give an
unbiased product for `exp[z(C_A+C_B)]`. They must use independent cloud
streams; multiplying weights from a shared correlated cloud is generally
wrong. An alternative exact Poisson construction uses one envelope cloud
and multiplicity `m=1_E(x)(1_EA+1_EB)`: the point product
`∏[1+z*m/λ]` has expectation `exp[z(C_A+C_B)]`. The union version uses
`m=1_E(x)1_(EA union EB)` instead. With the same points these give a monotone
coupling and the correct sign; using `(1+z/λ)^m` at a doubly covered point
would introduce an unwanted extra term. Certified deterministic volumes
would need matching multiplicity bookkeeping before this optimization is
implemented.

## Cheapest initial budget and controls

Reuse the completed, independent M1 reference for Q_A; its observed relative
SE remains about 30%. The minimum new factorial work is Q_empty, Q_B and
Q_AB. Q_ABC is an optional nested-neighborhood check motivated by M3, but
does not constitute a full three-neighbor factorial contrast. That would
also require C, AC and BC and is not proposed at this stage.

| Target | Proposed new fixed budget | Expected informative geometry |
|---|---:|---|
| Empty | 4 × 262,144 | About 18,000 q-valid poses; purely geometric |
| A | Reuse completed 8 × 4,194,304 | 20,041 hard-native poses already recorded |
| B | 4 × 1,048,576 | Hard fraction not yet independently measured |
| AB | 4 × 1,048,576 | About 890 hard-native poses from the geometric screen |
| ABC, optional | 4 × 1,048,576 | About 383 hard-native poses from the geometric screen |

The full initial plan is 13,631,488 new draws; omitting ABC saves 4,194,304.
Use at most eight workers. The M1 timing of 17.8 microseconds per draw gives
about 224 CPU seconds for the three nonempty proposed targets, before
allowing for different union-query cost and survival rates. This is a rough
cost scale, not a timing guarantee: B's hard fraction and multi-neighbor
cloud envelopes are still unknown. Larger neighborhoods reject more poses
but cost more per surviving cloud. The proposed valid-pose counts do not
predict weighted ESS, which may remain very small.

Required controls before any launch:

1. Check exact equality of shape, region metric, native reference, capture,
   bath and cover across variants; only the specified fixed-neighbor list
   may change. Confirm fixed-fixed and reference-pose hard validity.
2. Retain every unconditional draw and independent replicate, including zero
   estimates. Use each variant's full union in the existing positive overlap
   estimator. Archive the frozen binary and exact configuration hashes.
3. Preserve the exact q label; do not redefine the native region around
   surviving high-weight poses. Report hard volumes, Q(z), core/shell mass,
   ESS, largest weights and independent-replicate variability for each target.
4. If variants share pose seeds, treat their estimates as correlated when
   propagating the factorial uncertainty. Shared proposal streams preserve
   each target's unbiased mass estimate; ratios and logarithms are not
   themselves unbiased. Independent cloud streams are required for the
   optional product-weight counterfactual.
5. Treat this fixed initial budget as a precision screen. Neither an unstable
   factorial ratio nor a few very large surviving weights justify an
   equilibrium or nucleation conclusion. Any extension gets a fresh fixed
   budget; existing outcomes remain in the record.

The native cover and its analytic tests already handle empty, single and
multiple fixed-neighbor lists. The original frozen native executable can
run these configuration variants directly. No sampler change is needed for
the minimum whole-native comparison.
