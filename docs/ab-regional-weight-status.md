# Native and competing weights in the same fixed AB neighborhood

The current regional calculations use the same physical target: one mobile
rigid tetramer, two fixed AB tetramers, capture radius 18 Å, depletant radius
1.5 Å and activity 0.035 Å⁻³. The original registration metric remains
2 Å member displacement and 15° proper rotation. Translation is integrated
in Å³ and proper SO(3) Haar measure is normalized to one.

`tools/summarize_ab_region_weights.py` checks the actual per-population
configuration, bath, q window, sample count, seed, metric and shape content
for two proposal controls in each region. All configuration values agree
apart from the locations of archived shape files; those files have identical
contents. It verifies terminal production statuses and preserves the source
analysis hashes. It does not repeat their raw-row audits or generate draws.

The four unchanged windows partition all captured poses:

| Region | Original q | Current log Q estimates | Remaining qualification |
| --- | --- | --- | --- |
| Native | 0≤q≤1 | 35.77224, 35.83594 | Complete geometric repeat gives 25% tail error; one pose carries 21% of that tail. |
| Shoulder | 1<q<2 | 25.09446, 25.12349 | Independent larger width repeats give 1.6–1.7% observed error; component differences remain. |
| Intermediate | 2≤q<5 | 16.69976, 16.77534 | Larger frozen-model repeats agree; positive remainder is less precise. |
| Far | 5≤q<37 | 17.68457, 17.67871 | Repeat rises from pilot; matching totals hide differing regional contributions. |

The [capture argument](far-capture-reference.md) proves that q<37 covers
every captured pose. Each regional estimate retains its own entire
unconditional sample count, zero contributions and normalized proposal
density. Two proposal estimates of one region are controls; they are not
added. Disjoint physical regions can be added in linear weight, never by
averaging their log weights.

The current ledger is
`runs/ab-region-weight-status-shoulder-repeat-20260921`. It replaces only
the full-shoulder controls in the previous repeat ledger, after verifying
the same physical configuration, shape, original windows and full sample
counts. The independent native-tail repeat is a separate diagnostic, not an
additional native contribution. All 16 choices of one proposal control per
region give

\[
\log\widehat Q_{\rm native}
-\log(\widehat Q_{\rm shoulder}+\widehat Q_{\rm intermediate}
      +\widehat Q_{\rm far})=10.648\text{–}10.741.
\]

This is the numerical spread of ratios of existing point estimates,
**not a confidence interval**. The corresponding observed other/native
weight ratios are 2.16×10⁻⁵ to 2.38×10⁻⁵. The unresolved shoulder and native
tails remain qualifications on this comparison. These numbers do not yet
establish a converged global occupancy.

The separately preserved pilot and repeat ledgers gave the earlier envelope
10.535–10.828. Replacing only their far estimates by the larger far repeat
did not change that envelope at the displayed precision, because the
shoulder dominates observed competing weight. The first calibrated shoulder
controls gave 10.694–10.765; the independent doubled repeat changes that
to the current interval above. Neither insensitivity to a small region nor
agreement of two proposal totals is a convergence test; regional concentration
and width/prefix controls remain necessary.

The free-energy difference is per mobile tetramer conditioned on this
specific fixed neighborhood. Both the tetramer's internal native geometry
and the AB scaffold are supplied. It is not a fluid chemical potential,
the cost of forming AB, or evidence of template-free assembly.

## What this implies for a mixing benchmark

For the exact stationary distribution on this fixed captured system, let
N be the native window and O its full complement. A reversible Markov
kernel satisfies the integrated flux identity

\[
Q_N\,\overline P(N\to O)=Q_O\,\overline P(O\to N),
\qquad
\overline P(N\to O)\le Q_O/Q_N.
\]

The bars denote averages over the equilibrium conditional distribution
within each set. The identity follows by integrating detailed balance
over N×O. It is a statement about attempted transitions of a specified
kernel, not a bound on physical time or on an arbitrary unequilibrated
starting pose.

If the current weight ratios survive the remaining checks, even perfect
entry from O could yield only about one native exit per 42,000–46,000
attempted kernel applications under that equilibrium conditional average.
Thus equal forward and reverse transition probabilities would be the wrong
test of correct sampling. The equilibrium *fluxes* must agree. A useful
mixing test must retain this distinction, test initialization sensitivity,
and measure independent contact samples per CPU where both environments
have appreciable probability. Accepted-move counts alone are insufficient.

Native-tail precision is not equally important for every inference. For a
fixed subset C of the native region, positivity gives the exact inequalities

\[
Q_N\ge Q_C,\qquad
\overline P(N\to O)\le Q_O/Q_N\le Q_O/Q_C.
\]

The existing fresh selected and wide guides give **log Q=35.60255 and
35.66055** for the original-chart `r<=4` core, retaining native q, both AB
neighbors and all original draw denominators. The selected `r<=2` subset
alone has log Q=34.75918 with 4.50% row and 2.98% population error.
These masks are archived in
`runs/native-ab-refined-validation-20260920/original-chart-tail-comparison.json`.
Independently, uniform sampling of the native `4<r<=5` shell gives
log Q=33.30200 with 8.87% row error. Even that finite shell exceeds the
largest current competing **point estimate**, log Q=25.12431, by 8.178 nats.

Missing native mass would strengthen this preference and lower the exit
ceiling. The unresolved native tail therefore need not block useful
one-sided sampling decisions. Estimated subset weights are not certified
lower bounds, however, and missing competing mass can change the conclusion.
This argument does not establish the exact normalized occupancy, arbitrary-start
transition probabilities or assembly without the supplied AB scaffold.

## The smallest remaining checks

The [shoulder analysis](shoulder-proposal-guides.md) identifies a specific
unresolved event: its direct 1<q<1.1 cover has log Q 26.0943 but a single
pose supplies 73% of that estimate. That pose already has appreciable
density under the frozen guides. It does not by itself establish a new
missing mode. An independently sampled finite geometric neighborhood,
compared with exactly the same mask on the historical rows, can calibrate
this event without discarding the rest of the shoulder.

That [local calibration is now complete](inner-shoulder-local-reference.md):
the direct reference assigned about 60 times the fresh weight to the
radius-0.25 neighborhood. Independent uniform proposals reproduce the
small-ball result. Most historical guided inner-shoulder weight lies outside
that single radius-0.5 ball, so the local result was never substituted for
a whole-shoulder estimate.

The [three-neighborhood control](shoulder-guide-peak-references.md) now
adds independent references at both guided maxima. Its priority-disjoint
finite union has log Q=24.42474 with 2.92% row and 3.31% population error,
agreeing in scale with both guided estimates of that same union. Historical
guided data place 26–30% of inner-shoulder weight outside all three balls;
their outside estimates are log Q=23.33294 and 23.79311, with 8.58% and
22.97% row errors. The fresh finite-union means and historical remainder
remain separate.

The [complete shoulder atlas](shoulder-contact-atlas.md) fits one Gaussian to each independently
sampled radius-.5 neighborhood, retains the legacy mixture and complete
geometric support, and evaluates all branches in the importance denominator.
In its first campaign, the width arms agree on the full shoulder, finite union, and directly sampled
complement. The complement still supplies 25–27% of inner weight, with
7.45% and 4.80% row errors. The broad arm's observed importance-weight ESS
per sampling CPU is 10.8 times the earlier mixture guide; learning/calibration
cost is excluded, and the full hybrid allocation also changed (model
probability 0.75 to 0.99), so this is a comparison of whole proposal designs.
That first narrow arm's early-to-final prefix change is
3.15 estimated correlated difference errors. This is substantial integration
progress, without a claim of full convergence or trajectory mixing.

The new independent shoulder repeats double each arm to 1,048,576 draws
without refitting or changing the proposal. Their full estimates are those
in the table above; the inner-complement row errors fall to 4.50% and 3.25%.
The broad repeat rises 5.43% from its original estimate. Its importance-weight
ESS per sampling CPU is 0.655, versus 0.860 previously; the narrow repeat is
0.323 versus 0.469 previously. Larger populations found additional weight,
so efficiency did not simply retain its initial value. Original, repeat,
width and fixed-prefix comparisons remain separate in the shoulder report.

The [native uniform references](native-ab-uniform-tail-reference.md)
reproduce the wider guide in the 8–12 shell. Their 5–8 confirmation agrees
in scale but still has appreciable Poisson noise and weight concentration.
The [complete native-cover repeat](native-tail-complete-cover.md) directly
measures the region beyond 12 with 4,194,304 fresh draws: log Q=26.10900,
24.64% row error, 11,301 positive poses and weight ESS 16.48. The largest
contribution supplies 21.37% of the tail. Its tail/full point ratios are
about 6.0–6.4×10⁻⁵; neither these ratios nor finite geometric support bound
the physical tail. The repeat now agrees with the targeted 8–12 shell,
but the 5–8 estimate is still 47.5% below its targeted reference and only
six repeat draws hit the core. It cannot replace the previous full-native
or targeted finite-shell estimates. The tail's last nested prefix rises
despite the increased sample count, so this is improved coverage without
an established convergence claim.

The [conditional shoulder sampler](shoulder-docking-benchmark.md) is the
next mixing control. It compares identical local slots with either a third
local move, independent atlas redraw, or correlated reversible transport.
Contact-region probabilities have appreciable weights under this conditional
target, allowing a meaningful exchange test without waiting for rare native
exits in the full target.

The [complete-far atlas](far-contact-atlas.md) and
[intermediate repeat](contact-atlas-repeat.md) address other disjoint
regions. Progress in those regions does not remove these remaining
native/shoulder qualifications.
