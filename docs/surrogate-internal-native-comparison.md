# Internal native registration in the surrogate benchmarks

The completed native audit already supplies a nontrivial control for the
current conditional pair `[27,132]`. Local moves starting from the original
growth configuration reached native motif 6 in all four streams. Their
production occupancies were 30.20%, 70.87%, 32.89%, and 55.59%. The corresponding
four proposal-prepared starts had zero native occupancy. These results establish
accessibility with strong initialization dependence, not equilibrium weights.

The source-start native-presence apparent ESS values were only 3.88, 11.09,
4.04, and 3.29 over 4,096 production blocks. There were respectively 2, 9, 4,
and 1 observed returns to nonempty internal native labels. These are threshold
events, not independent basin visits or physical kinetic rates. The earlier
`m4` proposal also reached internal native registration in five of eight matched
chains, including two proposal-prepared starts. All sixteen local/m4 controls
had zero external native occupancy.

These are copied completed summary statistics, not a new trajectory pass:

`/vast/xvg/tetramer-mc-runs/conditional-native-registry-audit-20261004/analysis/summary.json`

SHA256 `07f25356cff250001bbc64fe3eef5b800581883fca9da6ec45a8a366136b8d68`.

## Declared internal-native comparison

All 24 flexible producers and their complete contact observer have finished.
A separate internal-native observer has been launched to classify the internal
pair at every initial and retained endpoint of all 24 rigid and 24 flexible
chains. The maximum is 221,232 full pair calls,
before exact-pose caching. Reuse the sixteen existing native controls; do not
reclassify them. Keep all 4,096 production blocks, rejected-state residence,
block512 as the transition baseline, separate starts/streams, and full sampler
CPU as the sampling denominator. Observer/setup cost is reported separately.

Use the same frozen repaired shape, native definition, complete fourteen-motif
classifier, and atom-identity witness. Keep every matching motif. Add two
explicit, separately counted implementation fixtures: source pair `[9,24]`
with native motif6, and the original source pair `[27,132]` with no native
match. Neither fixture contributes to the scientific comparison.

The internal-only observer does not infer registry from contact patches or
exclude candidates because a cached exclusion edge is absent. It also makes
no external or cycle-consistency claim. The old controls' internal descriptor
can be projected only when the previously observed mobile graph was resolved;
its graph-specific fields are removed, rather than fabricated for new chains.

The declared ceiling is one worker, 3,600 CPU seconds, 7,200 wall seconds and
16 GiB. A stopped or incomplete audit retains its partial outputs and does not
replace chains. Complete input hashes and prior arithmetic audits authorize
reading retained poses without repeating proposal or depletant arithmetic.

The observer is implemented in `tools/analyze_surrogate_internal_native.py`.
All nine focused synthetic tests passed, including retained-state cadence,
metadata identity failures, baseline/CPU metrics, multiple motif labels,
exact-pose reuse, query caps, fixtures and durable failure prefixes. The
complete source closure was unchanged across validation. Receipt:
`results/surrogate-internal-native-validation-20261004/attempt01/validation.json`,
SHA256 `75778ed7c2d79c4459fca8747efdd9652f5a25c30cd703e19e8f11e35535ef6a`.
No protein geometry or trajectory was evaluated during this validation.

The diagnostic asks whether extra internal flexibility improves registry
sampling per CPU and agreement between initializations, beyond coarse patch
motion. Higher native occupancy alone is not an efficiency measure. A short
record with no return to an appreciably occupied alternative is insufficient.

Interpretation is conditional on the recorded uncertainty and initialization
differences. More patch motion without improved registry sampling would motivate
changing the proposal geometry. Improved internal registry with unresolved
external docking would motivate a separate mobile-neighborhood/docking test;
this internal observer cannot itself establish absence of external bonds in the
new chains. Persistent disagreement between starts leaves the conditional
sampling question open. None of these branches refutes assembly.

This remains the two-mobile conditional target at `rd=1.4 Å`,
`z=0.0275 Å^-3`, with 262 fixed spectators inherited from the 500 μM growth
run. It does not decide assembly at `rd=1.5 Å`, `z=0.035 Å^-3` and
approximately 106.8 μM, nor does it establish equilibrium even for the
conditional pair.

## Contact analysis and figures

The whole-contact analysis completed its declared 24 new and 40 cached chains,
classifying 110,616 new endpoints without old geometry queries, replacement
chains or additional physical draws. The completed report is
`/vast/xvg/tetramer-mc-runs/flexible-surrogate-dimer-analysis-20261004/analysis/analysis.json`,
SHA256 `58307808150d4f3fc35879dbe28e6f617998c40b580eccdb97b2d1dd616b8f16`.

The authenticated figures retain all eight start/stream contexts separately:

- [Contact organization](../results/flexible-surrogate-contact-figure-20261004/contact-organization.png)
  ([SVG](../results/flexible-surrogate-contact-figure-20261004/contact-organization.svg)).
- [Contact efficiency](../results/flexible-surrogate-contact-figure-20261004/contact-efficiency.png)
  ([SVG](../results/flexible-surrogate-contact-figure-20261004/contact-efficiency.svg)).
- [Initialization differences](../results/flexible-surrogate-contact-figure-20261004/initialization-differences.png)
  ([SVG](../results/flexible-surrogate-contact-figure-20261004/initialization-differences.svg)).

The [plotted scalar table](../results/flexible-surrogate-contact-figure-20261004/plotted-values.json)
has SHA256 `326f7f6457dee722a61d97255ca0ba18f3dedffddf71e435a92f593a23d17276`;
its [plot receipt](../results/flexible-surrogate-contact-figure-20261004/plot-receipt.json)
has SHA256 `1b868b7ec002bebc69e08774ebe6088b76f8cf754ebf45686d228e7b6f688f00`.
The plotter copied completed observer scalars; it read no scientific journals,
performed no geometry queries, fitted no new statistics and pooled no starts
or streams. Undefined ESS remains undefined.

Guided m8 does not establish a general efficiency improvement. It used
18–46% less full sampler CPU than flat8 in these matched contexts, but its
whole-fingerprint apparent ESS/CPU was lower in six of eight. Its two higher
values occurred in proposal-prepared streams 0 and 2, which were also its only
two improvements over local updates. The saved values below use full sampler
CPU, including setup and warmup, and are finite-record descriptors rather than
equilibrium ESS:

| Start / stream | Local ESS/s | Flexible m1 ESS/s | Guided m8 ESS/s | Flat8 ESS/s |
| --- | ---: | ---: | ---: | ---: |
| Source / 0 | 0.304 | 0.280 | 0.105 | 0.209 |
| Source / 1 | 0.245 | 0.099 | 0.076 | 0.176 |
| Source / 2 | 0.163 | 0.096 | 0.096 | 0.192 |
| Source / 3 | 0.321 | 0.078 | 0.146 | 0.171 |
| Proposal-prepared / 0 | 0.369 | 0.274 | 0.759 | 0.084 |
| Proposal-prepared / 1 | 0.314 | 0.718 | 0.294 | 0.410 |
| Proposal-prepared / 2 | 0.234 | 0.409 | 0.479 | 0.250 |
| Proposal-prepared / 3 | 0.177 | 0.118 | 0.076 | 0.356 |

Guided m8's internal patch-return rate per CPU exceeded flat8 in all four
proposal-prepared streams, by factors 1.54, 1.45, 1.68 and 1.34. For source
starts it was higher in streams 0 and 2 and lower in streams 1 and 3. Internal
patch-change rates also varied: the guided/flat ratios were
`[1.10, 0.93, 1.11, 0.66]` for source starts and
`[0.82, 1.54, 1.28, 1.60]` for proposal-prepared starts. Faster repeated patch
changes or returns did not consistently yield higher whole-fingerprint ESS.

Occupancy and initialization differences limit these comparisons. All source
starts in the table retained internal contact throughout their 4,096 production
frames. In prepared stream 0, guided m8 had 4,096 internally occupied frames
and 33 distinct internal patch sets, versus flat8's 3,130 frames and 504 sets.
In prepared stream 2 those counts were 4,096/94 versus 3,734/330. In prepared
stream 3 the occupied-frame counts were 1,179, 3,160, 2,131 and 4,096 for local,
m1, guided m8 and flat8, respectively. Every matched source/prepared internal
patch-set occupancy comparison for these four arms had saved total variation
1.0: their observed supports were disjoint. Agreement between initializations
and equilibrium coverage therefore remain unestablished.

All 24 flexible chains had zero external-contact occupied frames, external
patch changes and external patch returns; their external-presence ESS is
undefined. They demonstrate no external-contact exploration benefit here.
The cached rigid controls illustrate why occupied-frame counts must accompany
presence ESS: rigid m8 source stream 0 had only one occupied frame and zero
external returns, yet apparent external ESS 4,096, or 31.12/s. Rigid m8 source
stream 1 had seven occupied frames and apparent ESS 586.87. Such high apparent
ESS from rare presence is not evidence of effective contact exploration.

These figures contain contact labels, not native classifications. The separate
internal-native observer was launched at
`/vast/xvg/tetramer-mc-runs/surrogate-internal-native-20261004`, with
execution-plan SHA256
`8749c0aa965f128e22ea3c9d78a9ce25057090d4d96963e5d9e44219cddf1466`
and protocol SHA256
`e7d51de9b01a4356b4524de401ba099e397016c34cdc87099406b5bf8416ea7c`.
No partial native-observer outcome is included here. The completed contact
results retain the conditional physical scope above and do not decide
finite-system assembly or native registry.

The completed-only native presentation tool is now implemented in
`tools/plot_surrogate_internal_native.py`. It shows all 64 chains separately:
native occupancy and occupied-frame counts, motif-presence ESS per full CPU,
nonempty returns, all fourteen motif marginals and 32 same-stream initialization
contrasts. Undefined ESS remains null; positive ESS uses a logarithmic axis.
It authenticates the drained observer and its frozen input identities before
reading scalar outcomes, without reopening trajectory journals.

Six synthetic checks passed, including incomplete-result rejection, input and
source binding, sample/CPU consistency, rare and constant occupancy, and all
three PNG/SVG renderings. Receipt:
`results/surrogate-internal-native-plot-validation-20261004/attempt01/validation.json`,
SHA256 `7e3efde58f3670b8ba6fb42fa2f91bf26d627ec3d6f4ac4f46983aedcf973028`.
No scientific plot has been generated from the running observer. After its
complete successful terminal record, use a fresh output directory:

```sh
/home/xvg/protein-nucleation/.venv/bin/python -B tools/plot_surrogate_internal_native.py \
  --root /vast/xvg/tetramer-mc-runs/surrogate-internal-native-20261004 \
  --output /absolute/fresh/native-figure-directory
```
