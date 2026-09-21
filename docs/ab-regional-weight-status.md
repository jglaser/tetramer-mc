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
| Native | 0≤q≤1 | 35.77224, 35.83594 | Native outer-chart tail remains incompletely referenced. |
| Shoulder | 1<q<2 | 25.00691, 25.23622 | Three inner neighborhoods calibrated; 26–30% of historical guided inner weight lies outside their union. |
| Intermediate | 2≤q<5 | 16.69976, 16.77534 | Larger frozen-model repeats agree; positive remainder is less precise. |
| Far | 5≤q<37 | 17.68457, 17.67871 | Repeat rises from pilot; matching totals hide differing regional contributions. |

The [capture argument](far-capture-reference.md) proves that q<37 covers
every captured pose. Each regional estimate retains its own entire
unconditional sample count, zero contributions and normalized proposal
density. Two proposal estimates of one region are controls; they are not
added. Disjoint physical regions can be added in linear weight, never by
averaging their log weights.

The initial ledger is
`runs/ab-region-weight-status-pilot-20260921`. The separately retained repeat
ledger is `runs/ab-region-weight-status-repeat-20260921`. With the pilot far estimates,
all 16 choices of one proposal control per region give

\[
\log\widehat Q_{\rm native}
-\log(\widehat Q_{\rm shoulder}+\widehat Q_{\rm intermediate}
      +\widehat Q_{\rm far})=10.535\text{–}10.828.
\]

This is the numerical spread of ratios of existing point estimates,
**not a confidence interval**. The corresponding observed other/native
weight ratios are 1.98×10⁻⁵ to 2.66×10⁻⁵. The unresolved shoulder and native
tails remain qualifications on this comparison. These numbers do not yet
establish a converged global occupancy.

Replacing only the far estimates by their larger-repeat results leaves
the displayed ratio range unchanged at this precision, because the
shoulder dominates the observed competing weight. This insensitivity is
not a convergence test for either region. The larger far repeat specifically
shows why small observed errors or matching whole-region totals can miss
regional concentration.

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
entry from O could yield only about one native exit per 38,000–50,000
attempted kernel applications under that equilibrium conditional average.
Thus equal forward and reverse transition probabilities would be the wrong
test of correct sampling. The equilibrium *fluxes* must agree. A useful
mixing test must retain this distinction, test initialization sensitivity,
and measure independent contact samples per CPU where both environments
have appreciable probability. Accepted-move counts alone are insufficient.

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
small-ball result. Most guided inner-shoulder weight remains outside
radius 0.5, so no new full-shoulder estimate has been substituted in this
ledger.

The [three-neighborhood control](shoulder-guide-peak-references.md) now
adds independent references at both guided maxima. Its priority-disjoint
finite union has log Q=24.42474 with 2.92% row and 3.31% population error,
agreeing in scale with both guided estimates of that same union. Historical
guided data place 26–30% of inner-shoulder weight outside all three balls;
their outside estimates are log Q=23.33294 and 23.79311, with 8.58% and
22.97% row errors. The fresh finite-union means and historical remainder
are kept separate. A new complete-support shoulder measurement remains
necessary; the whole-window values in the ledger are unchanged.

The [native uniform references](native-ab-uniform-tail-reference.md)
reproduce the wider guide in the 8–12 shell. Their 5–8 confirmation agrees
in scale but still has appreciable Poisson noise and weight concentration.
No independent reference yet controls the old-chart region beyond 12.
The full native target is compact; a complete geometric cover with that
reporting mask is a possible check. Neither small observed tail fractions
nor finite geometric support bound its physical statistical weight.

The [complete-far atlas](far-contact-atlas.md) and
[intermediate repeat](contact-atlas-repeat.md) address other disjoint
regions. Progress in those regions does not remove these remaining
native/shoulder qualifications.
