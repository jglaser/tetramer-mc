# Missing chemistry at lysozyme crystal contacts: findings and hand-off for a learned model

Status 2026-10-15, branch `morphometric-solvation`. This note records what a physics-only
minimal model of tetragonal hen egg-white lysozyme (HEWL) crystallization gets right, where it
fails, and the chemistry that the failure points to. It ends with two machine-learning
problem statements that another agent can pick up without re-deriving the context. Run
directories are on `/vast/xvg/tetramer-mc-runs/`; their `README.md` files are the detailed
logs. Analysis scripts are copied to `tools/chemistry_gap/` (they use absolute `/vast` paths).

## 1. The minimal model

- Bodies: rigid 1LYZ heavy-atom monomers (`examples/monomer-shape.json`, 1001 atoms; atom names
  in `oligomer-capture-20261010/dft-ref/atom_types_monomer.json`).
- Solvent: hard-sphere morphometry, White Bear II coefficients at eta 0.383 and solvent radius
  1.4 A, on exact union measures (`src/morphometry.rs`, `src/solvation.rs`).
- Attraction: atom square well, width 6 A, depth eps (`src/morph_mc.rs`).
- Chemistry: like-charge contact penalty, w = 4 kT per pair of like-charged groups within 5 A
  (pH 4.5 charge groups, `morph-growth-20261012/pair-binding/charge_groups_monomer_ph4.5.json`).
- Calibration: eps_c(6 A) = 0.00325 from the crystal coexistence estimate at 1 mM; at eps_c the
  second virial coefficient is about -7.6e-4 mL mol/g^2, inside the crystallization window
  (-2 to -8e-4), and no pair basin has ln(K rho) > 0 at 1 mM except m6 near threshold
  (`morph-growth-20261012/b22/`).
- Samplers (all exact Metropolis-Hastings, tests in `src/morph_mc.rs`): local moves, docking
  from a fixed contact list, a frozen learned map, the current contact library (contacts
  between the other bodies), and a Poisson point-process memory with an optional prior. The
  correctness of context-dependent proposals is proved in Lean
  (`formal/ReversibleSampling/FrozenContext.lean`, `PoissonMemory.lean`).

## 2. What the model reproduces

- Growth on a native seed near eps_c with native-only contacts; the free pool stays mostly
  monomeric (6 A runs; library + memory docking is the fastest, 19-21 lattice molecules).
- The threshold for growth agrees with the coexistence estimate within about 10% (growth starts
  between 0.9 and 1.0 eps_c).
- The second virial coefficient in the experimental crystallization window.
- Hard-sphere DFT (White Bear II functional, GPU solver `tools/fmt_dft.py`; results in
  `dft-contacts-eta0383-20261008/full-v2/`) is pairwise additive at every cluster tested:

| site | DFT (kT) | DFT pair sum | morphometry |
|---|---|---|---|
| trimer (m1 + m3) | -11.35 | -11.6 | -9.31 |
| 5-mer (m1 + m2) | -14.13 | -14.8 | -10.22 |
| 9-mer (m0 + m3 + m5 + m6) | -16.10 | -16.1 | -10.02 |
| crystal site (11 neighbours) | -50.2 | -51.1 | -34.75 |

  Morphometry is also pairwise there (a valuation; triple overlaps are absent at 1.4 A). Its
  error is pairwise and uneven: it underbinds m6 by 6.3 kT, m2 by 2.6, m1 by 2.0, m4 by 0.6,
  and m0, m3, m5 by less than 1 kT (fixed-code DFT, `full-v2/monomer-v3/`).

## 3. Where it fails: no spontaneous nucleation

- Seedless runs (with library, memory, blind prior, replica exchange in eps) form m6 dimers
  but never a third native contact.
- Site constants along native growth paths (`pair-binding` with an `environment`; greedy path
  `b22/greedy_path.py`): adding a third molecule to an m6 dimer costs 6-8 kT at 1.0-1.14 eps_c,
  and later steps 5-9 kT unless an m6 contact closes. Path free-energy maximum: 87 kT (1.0
  eps_c), 64 (1.14), 19 (1.4).
- Decomposition (trimer site): ln(K rho) ~ well depth (23.5 kT) + ln(translation volume) (-0.8)
  + ln(rotation volume) (-11.8) + ln rho (-14.3). Immobilizing a molecule costs 27-35 kT;
  m6 alone pays it (~40 kT), the weak contacts m1 + m3 together do not (~23 kT).
- Correcting morphometry to DFT pair values and recalibrating to the same crystal stability
  does not help the trimer (-0.7 kT), because the extra attraction goes mostly into m6.

## 4. The chemistry at the contacts

Contacts m0-m6 are the seven distinct native relative poses (`observe_growth.NAT`). Analysis of
1LYZ (`interface_chemistry.py`) and of high-resolution and neutron tetragonal structures
(`hydration_contacts.py`; downloaded to `morph-growth-20261012/pdb-hewl/`: 1IEE 0.94 A,
9RZL 0.90 A, 7DER 1.03 A, 6K8G X-ray + neutron, 1IO5 neutron):

| contact | DFT (kT) | direct polar pairs | Cl- bridge | bridging waters |
|---|---|---|---|---|
| m6 | -16.5 | 16-20 | no | 4-8 |
| m1 | -11.3 | 5-9 | one Cl- per contact (all 4 Cl-containing structures), between Asn113 ND2 (A) and Tyr23 OH (B) | ~7, plus second-shell links |
| m4 | -5.2 | 11-17 | no | 2-12 |
| m2 | -3.5 | 1-2 | no | 2-8 |
| m0, m3, m5 | ~0 | 0 (no atoms within 4 A) | no | 0-1 |

m1, the contact that carries the trimer step, is polar, hydrogen bonded, hydrated and
chloride bridged. The model has no ions and no hydrogen-bond or ordered-water chemistry.

## 5. Physics-based anion terms tried, and why they fail

- Chloride-bridge free energy, -ln(1 + c/K_d) at 0.5 M NaCl with Cl- protein K_d of 30-60 mM
  (range 2.5-150 mM; Collins 2004): 1.5-5.3 kT per m1 contact, about 1-3 kT at the trimer step
  after recalibration. Helpful but not sufficient alone.
- Generic donor-pair rule (`anion_sites.py`): two donors from different molecules 4.8-7.2 A
  apart with room for an anion. Not selective: sites at 34 of 37 non-native contacts (mean
  3.2), and it misses the real m1 site.
- Screened electrostatics (`anion_electrostatics.py`; Debye-Hueckel, 0.5 M): ranks 1 of the 4-5
  crystallographic Cl- sites of a single molecule at the top (99.5th percentile), the others at
  the 5th-62nd percentile; the best bridging point at m1 is 9.4 A from the observed chloride;
  78% of non-native contacts offer a higher-potential bridging point than m1.

Conclusion: in this model neither donor geometry nor electrostatics identifies the real
anion sites. What selects them (hydrogen-bond geometry, water network, desolvation) needs
either explicit-solvent free energies or a learned predictor.

## 6. Machine-learning alternative

### 6a. Anion (and bridging-water) site prediction  (recommended first)

- Task: for a point near a protein surface or between two proteins, predict the probability
  that a halide (Cl-, Br-, I-) or nitrate occupies it, or equivalently a site binding free
  energy -ln K_site.
- Input: the local atomic environment within ~8 A of the point (both molecules when at an
  interface): element, residue, atom name, hydrogen-bond donor/acceptor type, partial charge,
  and geometry (distances, angles to donors). Equivariant point-cloud or graph models fit.
- Labels: ordered anions in high-resolution crystal structures (X-ray <= 1.5 A; neutron
  structures add H/D positions). Negatives: first-shell points without an anion at the same
  resolution. Start from HEWL in all crystal forms (tetragonal, monoclinic, triclinic,
  orthorhombic; nitrate sites in the monoclinic form) and extend to other proteins.
  `tools/chemistry_gap/rcsb_search.py` shows the RCSB search-API pattern (tetragonal HEWL:
  121 entries <= 1.2 A, 6 neutron entries).
- Splits: by protein and by crystal form, so that a site never appears in both training and
  test sets through symmetry or re-refinement of the same crystal.
- Validation targets: (i) recover the m1 bridging chloride (Asn113 ND2 / Tyr23 OH) in the
  model frame (observed positions, model frame: about (8.1, 10.6, -2.7) A,
  `anion_electrostatics.json`); (ii) rank native contacts' bridging sites above non-native
  ones (contact poses in `pair-binding/contacts_monomer.json`, `contacts_blind_top32.json`);
  (iii) recover the single-molecule chloride sites of section 5.
- Use in the model: a contact term -sum_sites ln(1 + c / K_d(site)) over predicted bridging
  sites, with K_d calibrated on measured anion-binding data. It must be a deterministic
  function of the body poses (see 6c).

### 6b. Contact free-energy residual  (only if 6a plus simple terms are insufficient)

- Task: predict Delta G_true(pose) - Delta G_model(pose) for a pair of rigid bodies at a
  relative pose.
- Labels: explicit-solvent potentials of mean force at native and non-native contacts
  (expensive but direct); hard-sphere DFT pair values exist for about 60 contacts and give the
  solvent-physics part only. Weak global labels: B22, solubility, crystal stability.
- Main risk: non-native interfaces. The blind-run offenders (`blind-agg #0`, `#3`; hard-sphere
  DFT -23.9 and -25.2 kT) are exactly where a learned term would be extrapolating; the training
  set must contain many non-native poses with reliable labels.

### 6c. Requirements for any learned term in the sampler

- A deterministic function of the configuration (rigid-body poses); no stochastic inference.
  Then any proposal used by the samplers stays exact (Metropolis-Hastings with the energy
  difference; frozen-context moves per `FrozenContext.lean`).
- Incremental evaluation per moved body, at roughly millisecond cost (the solvation and well
  terms already set that scale).
- Keep the physics backbone (DFT-calibrated pair solvation, like-charge term, square well)
  and learn only the chemistry residual, so that B22 and the DFT additivity results stay
  interpretable.
- Re-check after adding it: the B22 window, the crystal coexistence eps (solubility), the
  native vs non-native ranking including the blind offenders, the trimer-step free energy
  (needs about +6-8 kT), and seeded growth.

## 7. Open questions

- How much of the m1 deficit is hydrogen bonding versus anion bridging versus water network?
  An explicit-solvent free-energy difference for m1 with and without the chloride would
  separate them.
- Are the weak water-separated neighbours (m0, m3, m5) stabilized by structured water in
  reality, and does that matter for nucleation or only for the bulk crystal?
- Protein flexibility (side-chain rearrangement at m1) is outside the rigid-body model.
