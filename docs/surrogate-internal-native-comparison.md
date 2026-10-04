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

## Fixed next comparison

After all 24 flexible producers and their complete contact observer finish,
classify the internal pair at every initial and retained endpoint of all
24 rigid and 24 flexible chains. The maximum is 221,232 full pair calls,
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

This remains the two-mobile conditional target at `rd=1.4 Å`,
`z=0.0275 Å^-3`, with 262 fixed spectators inherited from the 500 μM growth
run. It does not decide assembly at `rd=1.5 Å`, `z=0.035 Å^-3` and
approximately 106.8 μM, nor does it establish equilibrium even for the
conditional pair.

## Contact analysis and figures

The completed-inventory contact dispatcher is prepared as source under
`results/flexible-surrogate-analysis-preparation-20261004/`. Its twelve
synthetic tests passed; its actual completion gate remains closed while the
producers run. It classifies 110,616 new endpoints and reuses forty contact
controls, without old geometry queries.

`tools/plot_flexible_surrogate_benchmark.py` consumes only completed,
authenticated contact summaries. Its three figures separate cached and new
chains and show contact organization, efficiency, and initialization
differences. Undefined ESS remains undefined; occupied-frame counts accompany
presence ESS to expose rare-event artifacts. Six synthetic tests, including
PNG/SVG rendering, passed. Run after the observer completes:

```sh
/home/xvg/protein-nucleation/.venv/bin/python -B tools/plot_flexible_surrogate_benchmark.py \
  --root /vast/xvg/tetramer-mc-runs/flexible-surrogate-dimer-analysis-20261004 \
  --output /absolute/fresh/figure-directory
```

These contact figures do not contain native classifications. The separately
bounded internal-native observer supplies that additional measurement.
