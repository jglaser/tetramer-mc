# Capped protein dimer proposals: completed passive comparison

Conditioning whole joint draws on a feasible internally contacting dimer
produced candidates in every frozen source/anchor context. It did **not** solve
external docking: at cap 32, only 5 of 66 feasible FFT-guided destinations have
any external exclusion contact, versus 31 of 176 for the blind contact-memory
atlas and 55 of 208 for the native-informed atlas. These are proposal endpoints,
not accepted moves or equilibrium observations.

The frozen experiment and its independent audit completed without errors or
replacement draws. The audit passed **1,692,219 checks**. This validates the
recorded proposal/geometry calculation within the stated numerical checks; it
does not resolve the [retained stationarity failure](dimer-one-step-stationarity.md)
or any physical contact-weight convergence gate.

## Fixed design and actual density law

The source is the same eight contexts from the sweep-7400 growth snapshot used
by the [earlier destination screen](dimer-destination-probe.md): 264 tetramers,
**1.4 Å depletants, activity 0.0275 Å^-3 and 500 μM**. These settings are distinct
from the original physical decision conditions, 1.5 Å, 0.035 Å^-3 and approximately
106.8 μM. Every outer attempt resets to this source. The repaired shape, atomic
wall, panel, labels, source files and three atlases are unchanged.

There are 32 outer attempts per atlas/context/cap, with caps **1, 8 and 32**:
**2,304 outer attempts**, from **768 independently seeded source/attempt slots**.
Each cap receives exactly the same random-stream seed within a slot. Consequently
the caps share raw prefixes; they are paired controls, not three independent
populations. Maximum allocation was 31,488 executed raw trials, including repeated
prefixes, and 24,576 unique raw-prefix positions. Actual execution used **22,227
raw trials**, of which **16,013** are unique positions and **6,214** are verified
prefix duplicates. There were no allocation extensions.

The guide on each relative tree edge is

\[
F(h)=\tfrac12 U_{160}(h)+\tfrac12G(h).
\]

U is uniform in the relative translation cube [-160,160]^3 Å³ and normalized
Haar orientation. G includes every frozen Gaussian and reciprocal branch.
Both edges are independently redrawn. The child is decoded relative to the new
root. The full correction is the sum of both old log F values minus both new
log F values, regardless of generated branch. Hard and contact failures retry
the **whole joint draw**; numerical errors would terminate the screen with their
records. The first endpoint satisfying full hard/wall validity and internal
exclusion contact ends the retry sequence.

This executable includes the [prepared-factor scoring correction](docking-factor-closure.md),
core implementation commit `e8ce9c0`. The map and learned-density scorer use the
same prepared factors. The new audit independently reconstructs that map-factor
law; it does not reuse the obsolete scoring-factor/source binder from the prior
screen. The old data and audit remain unchanged. Exact executable and source
bundle hashes are recorded below.

All eight sources are hard-valid and internally contacting. Every source, anchor
and outer attempt is retained, without native-label or outcome filtering. The
source frame itself is not claimed to have been generated without native
information. The atlas arms are the 64-slot all-slot blind contact memory,
512-slot blind FFT atlas (1,024 stored Gaussians), and native-informed coverage
atlas (178 stored Gaussians).

## Candidate counts and measured costs

Each row below contains **256 outer attempts**. Every candidate is hard-valid
and retains the internal root–child exclusion contact. “External” means at least
one contact between either selected body and any fixed spectator. “Both” means
both intended tree contacts: root–child and root–chosen-anchor. Neither label
measures native registry or overlap volume.

| Atlas | Cap | Raw trials | Candidates | External | Both | Candidates/raw | Candidates/proposal CPU s | External/proposal CPU s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Blind memory | 1 | 256 | 11 | 0 | 0 | 4.30% | 55.21 | 0.00 |
| Blind memory | 8 | 1,773 | 65 | 7 | 6 | 3.67% | 133.62 | 14.39 |
| Blind memory | 32 | 4,890 | 176 | 31 | 21 | 3.60% | 161.70 | 28.48 |
| Blind FFT | 1 | 256 | 5 | 0 | 0 | 1.95% | 7.27 | 0.00 |
| Blind FFT | 8 | 1,956 | 19 | 1 | 0 | 0.97% | 4.54 | 0.24 |
| Blind FFT | 32 | 7,093 | 66 | 5 | 1 | 0.93% | 4.46 | 0.34 |
| Native-informed | 1 | 256 | 11 | 3 | 3 | 4.30% | 43.41 | 11.84 |
| Native-informed | 8 | 1,717 | 87 | 25 | 19 | 5.07% | 101.34 | 29.12 |
| Native-informed | 32 | 4,030 | 208 | 55 | 39 | 5.16% | 114.11 | 30.17 |

These CPU values are descriptive point measurements. The caps always ran in
order 1, 8, 32 within each slot, so cache/order effects and amortization of source
checks can affect comparisons. There were no randomized timing replicates.
The denominator is time inside the proposal call; it excludes the additional
external-contact diagnostic, model/context setup and audit. It includes raw
retries and source feasibility checks. It is not total assembly cost or contact
ESS per CPU, and the table does not establish an algorithmic speedup.

Increasing the cap raises candidate probability per outer call as expected.
For a fixed context, it leaves the success-conditional destination law unchanged
and does not improve successes per independent raw trial in expectation. Contexts
have different success probabilities and exposure, so pooled percentages should
not be treated as a single homogeneous probability or as equilibrium weights.

At cap 32, the median proposal-only log reverse/forward factor is **−18.26** for
blind memory, **+6.83** for blind FFT and **−12.75** for native-informed. These are
neither physical free-energy differences nor acceptance probabilities. A
favorable proposal factor can accompany severe loss of physical depletion
weight. No Poisson bath clouds, physical acceptance decisions, state updates or
native classifier were used in this experiment.

## Source and anchor coverage

Cap-32 results below retain all eight contexts; each entry is
**candidates / external / both**, out of 32 outer attempts. “Whole” describes the
source pair being an entire component; “embedded” describes a pair within a
larger source component. Labels and both anchors are fixed by the earlier panel.

| Source pair | Anchor | Blind memory | Blind FFT | Native-informed |
|---|---:|---:|---:|---:|
| Whole 13–215 | 183 | 24 / 9 / 6 | 5 / 1 / 0 | 24 / 9 / 9 |
| Whole 13–215 | 20 | 24 / 6 / 4 | 12 / 0 / 0 | 24 / 4 / 2 |
| Whole 18–234 | 104 | 22 / 2 / 2 | 8 / 0 / 0 | 28 / 8 / 6 |
| Whole 18–234 | 230 | 23 / 3 / 2 | 6 / 0 / 0 | 30 / 10 / 6 |
| Embedded 0–1 | 2 | 18 / 3 / 1 | 6 / 0 / 0 | 22 / 2 / 0 |
| Embedded 0–1 | 109 | 22 / 2 / 2 | 5 / 1 / 1 | 30 / 6 / 4 |
| Embedded 8–255 | 148 | 26 / 2 / 1 | 14 / 2 / 0 | 29 / 11 / 7 |
| Embedded 8–255 | 233 | 17 / 4 / 3 | 10 / 1 / 0 | 21 / 5 / 5 |

This is full coverage of the predeclared small panel, not coverage of all fluid,
aggregate or crystal surface environments. Retaining an exclusion contact does
not retain its registry, volume, strength or other simultaneous contacts.

The fixed-label proposal correction also cannot be copied into production as a
complete acceptance rule. Production needs the reverse law of pair and anchor
selection, the same retained context in both directions, any required rate or
frozen-bias correction, and the exact many-body bath gate. Sources outside the
conditioning set must self-loop in this channel. The detailed argument and
numerical-error policy are in [capped-dimer-conditioning.md](capped-dimer-conditioning.md).

## Exploratory decomposition of the saved raw failures

A separate posthoc [counting script](../tools/decompose_capped_dimer_probe.py)
reads the already audited raw predicates. It generated **no new proposals** and
changed neither the frozen analysis nor its allocation. It uses all cap-32
prefixes exactly once; shorter caps are omitted solely because their records
were already verified as duplicates. These raw sequences stop at first success,
and different contexts have different numbers of trials.

The following columns form a disjoint ordered partition: first test internal
core validity plus exclusion contact, then root versus fixed spectators/wall,
then child versus fixed spectators/wall. This order is an explanatory accounting
choice, not the order in which the original runner evaluated predicates.

| Atlas | Unique raw trials | Internal pair fails | Root fails after pair passes | Child fails after pair and root pass | Feasible |
|---|---:|---:|---:|---:|---:|
| Blind memory | 4,890 | 4,277 | 333 | 104 | 176 |
| Blind FFT | 7,093 | 6,778 | 232 | 17 | 66 |
| Native-informed | 4,030 | 3,230 | 506 | 86 | 208 |

The internal failures separate into core overlap versus no exclusion contact:
910 / 3,367 for blind memory, 3,406 / 3,372 for FFT and 1,173 / 2,057 for the
native-informed atlas. These two causes are disjoint here because the exclusion
shape encloses the core. Additional root/child failures can overlap them; the
complete artifact also reports those nonexclusive counts.

Among the FFT draws whose **child edge used the learned branch**, **3,284 of
3,580 (91.7%)** have an internal core collision; all 3,580 have an exclusion
contact. This is compatible with the [center-clearance audit](dimer-mean-clearance.md):
hard-valid Gaussian centers do not imply that their finite-width draws are
hard-valid. It points toward conditioning individual edges on their independent
geometric constraints before the final joint check. The
[factorized conditioning derivation](factorized-dimer-conditioning.md) explains
when the unknown normalizers cancel. It remains a future implementation and
cannot change destination quality conditional on the same final domain by
itself. No measured staged speedup or physical acceptance result is claimed.

## Audit, artifacts and limits

The passive executable consumed **26.729 CPU seconds**. Its frozen independent
Python audit consumed **259.264 CPU seconds**, using one worker sequentially.
All 16,013 unique raw endpoints were reconstructed for generation, full mixture
densities, coordinate/Jacobian corrections and independent all-atom geometry;
6,214 repeated-prefix records were checked for exact equality. The maximum
logged learned-density discrepancy was **3.46e−11**, and the maximum joint
proposal-factor discrepancy was **1.75e−11**. The predeclared log comparison
bound was 2e−7 absolute plus 2e−10 times the reference magnitude.

**Five threshold-ambiguity witnesses** within the independent geometry check's
1e−8 Å reporting margin remain in the result. Their strict predicates agree;
none was dropped or given a relaxed geometric tolerance. This is a numerical
comparison, not certified exact arithmetic. The independent source/attempt seeds
and paired-prefix checks establish reproducibility of this frozen run; they do
not establish equilibrium, mixing or absence of unseen important regions.

The new runner's one stream/allocation contract test and seven independent audit
tests pass. The underlying capped module's eight tests were completed separately
and reused. New audit tests cover stopping/exhaustion, source-ineligible identity,
corrupted prefixes/RNG fingerprints, actual map-factor scoring, reciprocal
latent decoding, geometry/density corruption, source binding and unconditional
outer denominators.

Artifacts are under
[`results/capped-dimer-probe-20261002`](../results/capped-dimer-probe-20261002):

- [Frozen protocol](../results/capped-dimer-probe-20261002/protocol.json),
  [prelaunch closure](../results/capped-dimer-probe-20261002/prelaunch.json) and
  [root prelaunch review](../results/capped-dimer-probe-20261002/root-prelaunch-review.json).
- [Independent analysis](../results/capped-dimer-probe-20261002/analysis.json),
  [complete raw ledger](../results/capped-dimer-probe-20261002/execution/attempts.jsonl) and
  [completion receipt](../results/capped-dimer-probe-20261002/completed-review.json).
- [Exploratory failure decomposition](../results/capped-dimer-probe-20261002/exploratory-failure-decomposition.json),
  kept separate from the frozen analysis.

Bindings:

| Artifact | SHA256 |
|---|---|
| Config | `95c4a062fea549b4a3c8c1cf538293fa895dc2f3c71a800985074f50c1ab68eb` |
| Protocol | `f1aa96c00af53fdeaed40cbe66e97915756f47cb749bd4fb7b8a357456989a7e` |
| Compiled source bundle | `178d6a44a5b538c6a737eec5b1dca0d30f4d7d8ab7b878782cb36b7f1b2106bc` |
| Executable | `2ef337f40e7816670e6a35765297cf25beeb7ae8f8cc54f6dc812d125a7d9c57` |
| Raw ledger | `f441c310fd05a27f45ca662f456c0b079fe884cfeac30a2bfe291da276e30525` |
| Independent analysis | `691f7ee684eab32a764d35f8355bb2ba4674d7da228141e86303589f1e8a1334` |
| Completion receipt | `17a885f9ea4543d9d32e2038a333c950936c291bdf51e47e8525d23ce5970ee9` |
| Exploratory decomposition | `8794908c5ff0a9cdeba95bc9d77bc459cd337ebbb1dead4d0056a77aa6c8186f` |

The completion receipt reauthenticates all 504 frozen closure files, nine
historical inputs, unchanged output bindings and both successful terminal exits.
No running production executable was replaced. This calculation advances the
proposal-feasibility diagnosis while leaving finite-system native assembly and
its thermodynamic stability unresolved.
