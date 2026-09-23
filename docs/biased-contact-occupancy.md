# Physical occupancy from a frozen assembly bias

`tools/analyze_biased_contact_occupancy.py` is a passive observer for the optional
[frozen assembly bias](frozen-assembly-bias.md). It reads completed all-mobile
Rust trajectories; it does not run MC, tune a bias, choose a burn-in, or infer
native stability. The ordinary contact-efficiency observer continues to refuse
biased trajectories by default.

For the runner's sampled measure

\[
\pi_B(dX)\propto e^{-B(n(X))}\pi(dX),
\]

the physical occupancy estimate of a frozen region R is

\[
\widehat p_R=\frac{\sum_t e^{B(n(X_t))}1_R(X_t)}{\sum_t e^{B(n(X_t))}}.
\]

The sign is **+B**. A common additive constant in B cancels. The implementation
subtracts the largest observed log weight before exponentiation and reports
numerical underflow explicitly. This self-normalized ratio is consistent under
appropriate ergodic/overlap assumptions; it is **not a finite-sample unbiased
estimator**. Every retained endpoint, including repeated rejected states, enters
the sum. The declared burn endpoint establishes the first interval and is not an
extra sample. A complete regular saved axis is required, including endpoints
outside the selected window. Events between saved endpoints remain unresolved.

The observer reconstructs the instantaneous body contact graph using the existing
atom-level `ContactObserver`: an edge requires a strict atomic surface gap below
twice the depletant radius. Exact tangency is excluded, with the same minimum-image
box restriction as the runner. Graph traversal independently checks every recorded
largest-component size. It verifies the exact finite frozen table, +B state fields,
manifest elementary-kernel protocol, terminal checkpoint/summary state, physical
and bias acceptance dispositions, all-frame cadence and counts, effective config,
archived shape/model/config/source hashes, and dependency hashes before writing.
The report binds both analysis modules and the shared autocorrelation implementation.
Collective `completed` counts finished physical kernels, including subsequent
bias rejections. Its value matches `physical_accepted`; retained collective
acceptance is tracked separately in the bias counters. Transformed-body counts
include only retained moves (and an accepted GCA may flip zero bodies).
These checks audit saved output; they do not independently prove physical gate
correctness or reconstruct unrecorded elementary proposals.

Native labels remain a separate instantaneous observable. Supply a hash-bound
`native_definition` to use `native_consistent_connected`; the existing native pair
registry and graph-cycle consistency checks run anew on each pose. Largest cluster
size does not imply native registry. No ancestral contact, hysteresis, or previous
native label enters either the region or bias score. All unmatched configurations
belong to `remaining`; no assembled-only conditioning or contact filtering is used.

A plan uses schema `frozen-biased-contact-occupancy-plan-v1` and the same target,
patch-map, window, preparation and region fields as the ordinary frozen contact
plan. For example (replace hash placeholders with SHA-256 bindings):

```json
{
  "schema": "frozen-biased-contact-occupancy-plan-v1",
  "physical_identity_sha256": "<physical_identity(config, shape_sha256) digest>",
  "patch_map": {"path": "patch.json", "sha256": "<file sha256>"},
  "native_definition": {"path": "native-definition.json", "sha256": "<file sha256>"},
  "environments": [
    {"id": "native", "native_consistent_connected": true},
    {"id": "competing-contact", "native_consistent_connected": false,
     "required_tokens": [[0, 1, "surface", "surface"]]}
  ],
  "burn_sweep": 1000,
  "end_sweep": 10000,
  "preparation_id": "dispersed",
  "proposal_arm": "frozen-capture"
}
```

This example's competing region covers the specified body/patch contact; other
competing configurations and unbound states are retained in `remaining`. Define
the desired exhaustive scientific partition before looking at production data.
Declared regions must be disjoint on observed configurations, and the observer
fails on overlap. It cannot prove global disjointness or adequate coverage from
a finite trajectory. Patch IDs must exist in the frozen full-atom patch map.

```bash
python3 tools/analyze_biased_contact_occupancy.py \
  --run /path/to/completed-run --plan /path/to/frozen-plan.json \
  --out /path/to/new-occupancy-report

python3 tools/analyze_biased_contact_occupancy.py \
  --compare /path/to/stream1/analysis.json /path/to/stream2/analysis.json \
  --out /path/to/new-independent-comparison
```

The report separates `physical_occupancies` from raw `biased_chain_diagnostics`.
It reports first/second-half physical ratios as drift diagnostics, never a
stationarity verdict. `weight_concentration.effective_count` is
\((\sum w)^2/\sum w^2\): it ignores time ordering. Constant weights give the saved
sample count even for a completely trapped chain. Contact fingerprints, region
indicators and the ratio residual \(w_t(I_t-\widehat p)\) have separate apparent
serial-correlation diagnostics. Constant observed traces yield null autocorrelation
ESS. An unobserved native region has a reported sample ratio of zero and unresolved
support, not demonstrated zero equilibrium probability. One stream yields no
confidence interval.

Independent comparison requires matching physical target, region/native observer,
measurement cadence/window, frozen bias table and implementations. It refuses
reused seeds and resumed segments from the same seed, keeps proposal arms and
preparations separate, and requires a single frozen proposal within an arm.
Unique seeds check bookkeeping but do not establish independent preparations.
Ordinary physical-only comparisons cannot ingest these reports. Different bias
tables are deliberately not pooled; their normalizers differ.

For each group, comparison reports the equal-stream mean of individual physical
ratios and its between-stream standard error, plus the ratio of stream numerator
and denominator means. The latter includes the full numerator/denominator
covariance in an asymptotic delta-method standard error:

\[
\operatorname{SE}_{\rm delta}=\frac{
 s(U_j-\widehat p V_j)}{\sqrt{m}\,\overline V},\quad
 U_j=\overline{wI}_j,\quad V_j=\overline w_j.
\]

All stream moments share one log-weight gauge and equal saved-window lengths.
Chains are never concatenated for an ESS calculation. With fewer than two streams,
standard errors are null. Zero observed variance is explicitly marked; common
trapping, few preparations, failed overlap and unseen competing states can make
these uncertainties misleading. They are diagnostics, not a convergence test.
Conditional scaffold probabilities and their normalizers are not accepted as
independent evidence for the all-mobile occupancy ratio. Establishing finite-system
native assembly additionally requires independent starts, demonstrated exchanges
and coverage of competing/unbound support, and appropriate free-energy evidence.

Focused checks use synthetic retained histories and a finite fully enumerated
ensemble with native/competing/unbound physical masses `(0.6, 0.3, 0.1)` and biased
masses `(0.4, 0.4, 0.2)`. They test the sign, additive gauge, neutral limit, native
registry independence, retained repeats, missing/altered output rejection, unequal
weights, provenance rechecking, and independent-stream covariance arithmetic.
They launch no physical simulation:

```bash
python3 -m unittest discover -s tools -p 'test_analyze_biased_contact_occupancy.py' -v
```
