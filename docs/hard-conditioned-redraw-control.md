# Fixed-state independent redraw control

This is a prospective algorithmic comparison, separate from the original
1.5 Å / 0.035 Å⁻³ equilibrium-contact campaign. It uses the same archived
sweep-60,900 state as the singleton controls: 264 mobile tetramers,
1.4 Å depletants, activity 0.0275 Å⁻³, and 500 μM.

The retained contact-pool control improved two-neighbor coverage and
contact-preserving acceptance, but produced only one accepted contact exchange
and no native pair-entry changes. Its completed populations are retained as a
correlated-transport reference; they will not be rerun.

## Allocation, fixed before sampling

| Learned proposal | Charts | Populations | Reset phases/population | Seed base |
|---|---|---:|---:|---:|
| Independent, cap 1 | Unfused / fused | 4 per arm | 128 | 2026100300 |
| Independent, cap 8 | Unfused / fused | 4 per arm | 128 | 2026100400 |
| Local, 0.2 Å / 1° | Single local arm | 4 | 128 | 2026100500 |

Within each cap, streams use `seed_base + 2*population + arm_index`, with
unfused index 0 and fused index 1. Local streams use `seed_base + population`.
Each phase starts independently from the same archived state. There are 2,560
phases in total and an expected 6,758.4 channel
events; the realized event count follows the fixed-duration clock. No allocation
is extended or replaced based on its results. At most two benchmark children
run concurrently, after the contact-weight pilot releases its physical slots.

The four independent arms use singleton rate 1, phase duration 0.01, transport probability 1,
two anchors sampled with the retained contact-without-replacement law, 10%
uniform anchor defense, and the existing 10% uniform pose branch. Dimer and
trimer channels are disabled. The old correlation value remains in the config
for compatibility but is unused by the independent learned branch. Physical
shape, atlas, bath, wall, native observer and uniform branch are unchanged.

The local arm uses the same singleton clock and source configuration's 0.2 Å
translation / 1° small-angle parameters, with transport probability zero. It
was added before any new benchmark draws, after the preceding pool pilot showed
that many accepted moves were small contact-preserving motions. It tests whether
the added proposal cost earns useful motion relative to inexpensive local MC.
No local/GCA/shift trajectory schedule is run inside these isolated reset phases.

The hard-conditioned kernel must first pass independent decode/density checks,
finite-cap balance, restart/replay, disabled-mode compatibility and a physical
sphere reference. Only then are the exact executable, source bundle, scripts,
configurations and complete allocation frozen and launched.

## Decision metrics

Retain every event, every raw candidate, all hard failures and exhausted caps,
and every physical rejection. Report independent populations separately.

- Exact raw-trial counts, candidate hard validity, cap exhaustion and CPU cost.
- Accepted changes in exclusion-contact partners, with attachments, detachments
  and completed exchanges distinguished from motion within the same contacts.
- Accepted surface-patch changes using the existing native-blind 32-patch map,
  including changes that retain partners, with displacement and rotation sizes.
- Instantaneous native pair-entry gains/losses, observed after acceptance only.
- Full guide-density and ordered-pool corrections separately from the realized
  Poisson factor. Auxiliary log factors are not free energies.
- Catalogue cost and accepted contact changes per CPU, compared with cap 1 and
  the completed correlated controls.

The hypothesis is **amortization**: reuse one catalogue and source-density
evaluation across several cheap independent candidates. Repeating hard tests
does not itself improve valid candidates per raw trial. A higher acceptance
count without changed contact environments does not satisfy the sampling goal.
This reset experiment cannot estimate equilibrium occupancy, trajectory ESS,
native assembly stability or a nucleation rate. A favorable result earns a
separate trajectory-mixing test; an unfavorable result leaves the physical
assembly question unresolved.

See [the exact construction](hard-conditioned-independent-redraw.md) for the
stopping rule and balance argument. Retrying correlated moves, retaining one
Gaussian label across retries, or retrying after depletion rejection would
implement different laws and is excluded.
