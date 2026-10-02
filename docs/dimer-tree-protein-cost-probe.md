# Saved protein dimer: auxiliary cost and proposal coverage

A frozen 32-proposal geometry screen supports testing the two-singleton
auxiliary path before paying for world-space Poisson clouds. It also exposes a
separate coverage limitation in the historical geometry-only atlas. No physical
acceptances, clouds, native classifications or state updates were performed.

The source is the archived sweep-7400, 264-tetramer state at radius 1.4 Å,
activity 0.0275 Å^-3 and 500 μM, not the decision conditions of 1.5 Å / 0.035 Å^-3
near 106.8 μM. Labels 27 and 132 form the purposively selected moving dimer;
228 is the fixed proposal anchor and remains a physical spectator. Every trial
resets to the same source state. Sixteen traces per atlas use correlation 0.7.
The geometry-only atlas here is the historical **96-component** model, not the
newer 512-component blind model. The native-informed atlas has 178 stored
components and 328 reciprocal branches.

Both auxiliary path orders and the direct world-space cover were constructed
for every finite endpoint, including hard-invalid ones. The path intermediate
is an exact copy of endpoint poses; it receives no physical rejection. Cover
budgets are 255 created cells and depth 8, with unresolved cells retained. With
lambda=1.76 Å^-3, expected raw points equal 1.7875 times cover volume. A fair
order coin averages the two paths' costs; choosing the cheaper path would need
an additional selection argument.

| All 16 attempted endpoints per atlas | Historical geometry-only | Native-informed |
|---|---:|---:|
| Hard-valid final endpoints | 2 | 3 |
| Median world-cover expected raw points | 6,892,389 | 5,458,564 |
| Median fair-order expected raw points | 314,036 | 280,000 (rounded) |
| Ratio of mean world / fair-order point costs | 21.74 | 17.07 |
| Median learned joint log proposal correction | −1,385.62 | −40.09 |

All 32 maps produced finite endpoints and all 160 cover constructions completed.
Total probe CPU was 2.56 seconds. The valid-only summaries remain separately
available; no successful endpoints were selected to replenish the fixed sample.
These point counts are cost predictions for these endpoints and budgets, not
runtime or effective-sample speedups. They do not measure changed solvent volume
or predict the auxiliary acceptance probability.

The geometry-only atlas assigns source-edge log densities −1771.57 (external)
and −769.51 (internal). Its median proposal-factor contributions are −954.15 and
−398.72 respectively; medians do not add to the median joint correction. The
current old internal separation is 45.44 Å. Thus poor reverse proposal coverage
is present even before evaluating the bath. These numbers are proposal-density
factors, not physical free-energy differences, and do not refute this model or
geometry-only learning generally.

## Controlled response

The first improvement to test is the exact two-singleton auxiliary path with
one final MH decision. The second is a separate, normalized defensive
**independent** redraw in tree coordinates:

\[
F(h)=\alpha U_L(h)+(1-\alpha)G(h),\qquad
\log R=\sum_{i=0}^1[\log F(h_i)-\log F(h'_i)].
\]

U_L is uniform translation in a fixed relative cube times normalized Haar
orientation; G is the unchanged learned pair atlas. Both density contributions
must enter every endpoint evaluation, regardless of the selected branch.
The child is decoded relative to the new root. This provides an explicit
reference for the source-tail problem while preserving support. The cube and
its normalization must be fixed and cover the intended old-state domain.

It would be incorrect simply to replace G by F in the existing correlated-map
acceptance formula: its generation law must also change. A separately selected
uniform MC kernel likewise does not put a uniform floor inside the learned
kernel's density ratio. The original involutive map remains a separate
building block and control.

## Artifacts and limits

The [standalone example](../examples/dimer_tree_cover_probe.rs) requires frozen
source/config/model bindings and refuses an existing output directory. The
[saved-data analysis](../results/dimer-tree-cover-probe-20261002/analysis.json)
passes 2,906 binding, ledger and arithmetic checks, including every attempted
row, copied intermediate and summed path cost. It authenticates the geometry
implementation but does not independently recompute protein overlaps or Gaussian
densities. Earlier module references cover those implementation components.

- Protocol SHA256: `4b08c1ed3bd647f129962f0c403928a7cbee6442c971f0294e7683936de5b681`.
- Attempt ledger SHA256: `73692830060ba8ed53b5d5da04ddf5f24ece82cb915bf2ceffc01c856fd02b15`.
- Analysis SHA256: `3b0347ed2bdedbcd34368b27d5b129bbe1d317456ece088dfe7fbbebfc5d2c6d`.

Both proposed improvements require equilibrium controls before a physical
protein benchmark. This passive conditional scaffold supplies no finite-system
assembly or instability evidence.
