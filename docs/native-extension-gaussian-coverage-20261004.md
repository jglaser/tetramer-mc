# Fixed-width Gaussian coverage around adjusted native contacts

**Adjusting the Gaussian centers raises full-environment native coverage from
2.95% to 15.00%: 393 versus 1,997 successful draws out of 13,312 per arm.**
That is **5.08× proposal coverage** with the same covariance and allocation.
This is stronger evidence than finding isolated feasible points: the improvement
survives finite-width perturbations. It is not an MC acceptance, mixing-time or
assembly-speedup measurement.

The comparison remains **native-informed**, using 52 predefined extension slots
around four fixed measured quartets from the historical **1.4 Å depletant radius,
0.04 Å⁻³ activity, 500 μM, N=264 spherical environment**. It is separate from
physical assembly evidence at the decision conditions of 1.5 Å / 0.035 Å⁻³ /
106.8 μM. The bath is not sampled in this geometric coverage calculation.

![Fixed-width Gaussian native coverage](../results/native-extension-gaussian-coverage-report-20261004/render01/gaussian-coverage.png)

## Controlled proposal comparison

The [incoming-only search](native-extension-incoming-search-20261004.md) found
nearby adjusted centers while holding every existing particle fixed. Here the
original and saved adjusted centers each receive **four independent populations
of 64 draws per slot**. All 52 slots remain, including the two unsuccessful
adjustments and the initially open point. The same six-dimensional standard
normal variate is used for both arms of each pair. Draws are untruncated;
invalid cores, walls and native entry all remain in the denominator.

Every center uses the same covariance, borrowed from base component 150 of the
frozen atlas: translational standard deviation **0.25 Å per coordinate**, and a
**0.25° nominal one-axis angular width**. The angular length is
55.02283113084892 Å. This borrows one geometric kernel; it does not sample the
178-component atlas or assign native classes to mixture components. Neither
widths nor allocations were selected by observed coverage.

The coordinate construction is `v = Lz`, `t = t_center + v[:3]`, and
`R = Cayley(v[3:]/ell) R_center`. Its density is normalized with respect to
translation times normalized Haar measure on SO(3). Stored Jacobians and
densities were independently reconstructed; coverage itself is the ordinary
success fraction under this kernel, without claiming a uniform pose volume.
“Native” means that the hard-clear candidate retains **all prescribed intended
neighbor contacts** for its slot, not arbitrary native contacts elsewhere.

| Certificate | Original centers | Adjusted centers | Paired improvement |
| --- | ---: | ---: | ---: |
| Quartet-only native | 545 / 13,312 = 4.09% | 3,201 / 13,312 = 24.05% | +19.95 percentage points |
| Full-environment native | 393 / 13,312 = 2.95% | 1,997 / 13,312 = 15.00% | +12.05 percentage points |

The full-environment paired improvement has a four-population Student-t 95%
half-width of **0.83 percentage points**. Separate population intervals are
2.95% ±0.67% and 15.00% ±1.01%. These are finite-replicate diagnostics.
Marginal conservative Hoeffding intervals are also retained: the paired gain
is **[9.70, 14.40] percentage points** under that bound. Intervals are not
simultaneous over every slot and metric, and do not represent equilibrium
uncertainty.

## Heterogeneity and retained controls

| Quartet | Slots | Draws per arm | Original full-native hits | Adjusted full-native hits |
| --- | ---: | ---: | ---: | ---: |
| Q1: 16, 77, 217, 237 | 12 | 3,072 | 43 (1.40%) | 262 (8.53%) |
| Q2: 42, 184, 194, 206 | 16 | 4,096 | 53 (1.29%) | 571 (13.94%) |
| Q3: 62, 67, 80, 134 | 12 | 3,072 | 53 (1.73%) | 457 (14.88%) |
| Q4: 69, 89, 91, 229 | 12 | 3,072 | 244 (7.94%) | 707 (23.01%) |

Every quartet improves in all four paired populations. The aggregate gives each
of the 52 slots equal weight, preserving the 12/16/12/12 allocation rather than
averaging the four quartet means equally.

All **17 initially spectator-blocked slots have zero full-environment hits in
both arms**. They remain in the aggregate. The 35 initially spectator-clear
slots improve from 4.39% to 22.29%. Zero hits do not establish zero probability:
the declared conservative 95% bound for the 17-slot stratum still extends to
2.06% in either arm.

Both previously unresolved Q2 slots, 16 and 41, remain at **0/256** in each arm,
even when testing quartet-only native clearance. The initially open control,
Q4 slot 66, remains unchanged and gives **84/256 in both arms**, with identical
paired outcomes. None of these controls was discarded or retried.

Adjusted centers somewhat increase native-entry failures under perturbation
(4.08% versus 0.38% of all draws). Consequently full-environment hard clearance
is 15.65%, while the stricter intended-native score is 15.00%. Reporting only
hard-valid poses would overstate native coverage.

## Cost of constructing and using the guide

| Recorded CPU | Original arm | Adjusted arm |
| --- | ---: | ---: |
| Chart construction, decode and complete geometry | 107.681 s | 105.992 s |
| Prior incoming-center optimization | 0 s | 2.382 s |
| Arm subtotal including that one-time cost | 107.681 s | 108.374 s |

The coverage run also uses **43.627 CPU seconds of shared setup, hashing, random
number generation, journaling and other work**. That overhead is not arbitrarily
divided between arms. Total coverage time is **257.300 CPU / 353.458 wall
seconds**; adding the previous center search gives 259.682 CPU seconds for the
combined workflow.

On the recorded arm subtotals, full-native hits per CPU second rise from 3.650
to 18.427 including the optimization cost, a **5.05× geometry-coverage efficiency
increase excluding shared overhead**. This is not an end-to-end sampler speedup.
The 2.382-second construction cost can be amortized only when the same centers
are reused; at this allocation it contributes about 0.179 ms per adjusted draw.
No current benchmark establishes whether frequently changing neighborhoods
would retain that advantage.

## Implication and next comparison

A small adjustment can move a useful portion of a fixed Gaussian kernel into
hard-clear native contact configurations. This supports incorporating such
centers into a proposal benchmark. It does not establish a physical statistical
weight, reversible acceptance rate, independent contact samples per CPU or
blind discovery of these centers.

The next sampling test should evaluate the normalized forward and reverse
proposal densities in a reversible kernel and include physical depletion
acceptance and contact decorrelation. A separate geometry-only center-discovery
control remains necessary. Spectator obstruction and the two unresolved local
searches remain distinct limitations; neither is evidence against the model’s
equilibrium assembly.

## Validation and artifacts

Twenty producer tests passed. The independent review replayed all **13,312
stored latent vectors**, **26,624 candidate records** and **93,184 journal
events**, checked every unconditional denominator and scalar native/core
certificate, and independently reconstructed rotations with SciPy rotation
vectors and densities from the covariance determinant. Maximum rotation-element
and log-density discrepancies were 3.33×10⁻¹⁶ and 7.11×10⁻¹⁵, respectively.
Per-candidate CPU counters reconcile with the reported timing totals.

The independent audit completed and drained in 6.253 CPU seconds; it performed
no geometry queries or new scientific draws. Seven focused tests validate its
scalar reconstruction and denominator handling. Floating-point geometry
execution remains an implementation obligation, not an exact-real proof.

- [Independent analysis](../results/native-extension-gaussian-coverage-independent-review-20261004/analysis-note.md)
  and [complete scalar results](../results/native-extension-gaussian-coverage-independent-review-20261004/analysis.json).
- Scientific root: `results/native-extension-gaussian-coverage-20261004/`;
  terminal receipt `f3326bb0ec8e9fd784b463e8c92d829e6b7a163eb17aa979a48e50352cdde663`.
- Independent receipt: `e9ec14a9fff117125b4c41feb7adf1d3aba59dd543bd10787a177d8990f67174`.
- [Figure and plotted values](../results/native-extension-gaussian-coverage-report-20261004/render01/)
  are generated solely from authenticated completed summaries. Rendering
  completed and drained in 0.926 wall seconds under 30 CPU seconds / 60 wall
  seconds / 2 GiB / one-thread limits, without geometry queries or draws.
