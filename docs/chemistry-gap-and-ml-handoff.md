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

## 8. Implemented SE(3)-invariant chemistry residual model (`ChemistryModel`) & 3-baseline validation

### 8.1 Model formulation (Formulation 6b + 6a bridging triad)

`tetramer_mc::morph_mc::ChemistryModel` (`src/morph_mc.rs`, `examples/chemistry-model-monomer.json`, trained via `tools/chemistry_gap/train_chemistry_model.py` across 20 ultra-high-resolution PDB crystal structures covering all 4 HEWL space groups $P4_32_12$, $P2_1$, $P2_12_12_1$, $P1$ and 8 non-HEWL proteins plus the 64 native/capture/blind-agg/non-native poses) evaluates a deterministic, symmetric, SE(3)-invariant 7-feature interface chemistry residual $\Delta G_{\text{chem}}(A, B)$ in ~0.2 ms per moved body:

$$\Delta G_{\text{chem}}(A, B) = -w_{\text{hb}} s_{\text{hb}} - w_{\text{chb}} s_{\text{chb}} - w_{\text{cl}} s_{\text{cl}} - w_{\text{wb}} s_{\text{wb}} + w_{\text{clash}} s_{\text{clash}} + w_{\text{upol}} \text{unp}_{\text{pol}} + w_{\text{uchg}} \text{unp}_{\text{chg}}$$

with weights $w = [1.1, 1.6, 10.5, 0.45, 1.8, 0.85, 1.8]\,kT$ and `wb_cap = 5.0`:
1. **`s_hb` (Direct neutral H-bonds)**: Gaussian kernel at $2.95\text{ \AA}$ ($\sigma = 0.45\text{ \AA}$, $d < 3.9\text{ \AA}$) between neutral donors (`NDON`: `0x04`) and acceptors (`NACC`: `0x08`).
2. **`s_chb` (Direct cationic H-bonds)**: Gaussian kernel at $2.95\text{ \AA}$ between cationic donors (`CAT`: `0x01`) and neutral acceptors (`NACC`: `0x08`).
3. **`s_cl` (Bilateral-cationic anion-bridge triad)**: `TYR:OH` (`0x10`) – `ASN/GLN:ND2/NE2` (`0x20`) pairs at $3.4 \le d \le 4.5\text{ \AA}$ whose midpoint lies within $8.0\text{ \AA}$ of cationic groups on *both* bodies and within $8.5\text{ \AA}$ of a direct cationic H-bond (`s_chb`). Directly captures the `Asn113 ND2 – Cl⁻ – Tyr23 OH` halide/nitrate bridge at `native m1` (where Lys13 on $A$ and Lys116/Arg112 on $B$ create the $+2.3\,kT/e$ electrostatic pocket).
4. **`s_wb` (Clash-gated water-bridged polar network)**: Complementary polar pairs at $3.95 \le d \le 5.25\text{ \AA}$ (Gaussian at $4.55\text{ \AA}$, $\sigma = 0.45\text{ \AA}$), capped at `wb_cap = 5.0` and exponentially gated by $\exp(-1.5\,s_{\text{clash}})$.
5. **`s_clash` (Unsatisfied like-polar clash)**: Pairs of pure acceptors (`0x40`) or pure donors (`0x80`) within $4.0\text{ \AA}$ that both lack a complementary partner within $3.65\text{ \AA}$.
6. **`unp_pol` (Unsatisfied buried neutral polar desolvation)**: Buried neutral polar atoms ($d_{\min} < 3.9\text{ \AA}$) without a complementary partner within $3.65\text{ \AA}$.
7. **`unp_chg` (Unsatisfied buried charge desolvation)**: Buried charged atoms ($d_{\min} < 4.0\text{ \AA}$) without a complementary partner within $4.30\text{ \AA}$.

### 8.2 Comparison across the 3 progressive physics baselines

Basin association constants $\ln(K / \text{\AA}^3)$ evaluated via `pair-binding` at $\varepsilon = 0.00325\,kT$, $w = 6.0\text{ \AA}$ (`tools/chemistry_gap/evaluate_baselines.py`):
- **Baseline 1 (`B1`)**: Hard shape + 6 Å square well (`--no-solvation`)
- **Baseline 2 (`B2`)**: Ideal Asakura–Oosawa depletion ($-z\Delta V_{\text{excl}}(r_d)$ with `--depletion-z 0.056`, $r_d = 1.4\text{ \AA}$) + 6 Å square well
- **Baseline 3 (`B3`)**: Exact White Bear II morphometric solvation + 6 Å square well + like-charge penalty (`w = 4.0`, `cutoff = 5.0 Å`)

| Contact | `B1_raw` | `B1_chem` | `B2_raw` | `B2_chem` | `B3_raw` | `B3_chem` | $\Delta G_{\text{chem}}(g_0)$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `native m6` (dimer spine) | +1.83 | +3.55 | +38.04 | +39.69 | +10.49 | **+14.55** | $-2.05\,kT$ |
| `native m1` (trimer $4_3$ screw) | -0.14 | +5.83 | +20.23 | +18.90 | +1.90 | **+10.17** | $-10.43\,kT$ |
| `native m4` (inversion) | -0.01 | -0.78 | +15.99 | +8.88 | -4.00 | **-5.36** | $+2.88\,kT$ |
| `native m2` (stack) | -0.82 | +0.64 | +5.60 | +7.50 | -1.68 | **-3.06** | $-0.16\,kT$ |
| `native m3` (water-sep) | -0.15 | -0.07 | +4.24 | +5.59 | -0.62 | **-0.86** | $0.00\,kT$ |
| `native m0` (water-sep) | -0.24 | -0.87 | +6.92 | +5.47 | -0.92 | **-1.45** | $0.00\,kT$ |
| `native m5` (water-sep) | -0.75 | -0.92 | +3.62 | +4.08 | -0.83 | **-1.09** | $0.00\,kT$ |
| `capture #9` (`m6` near-dup) | +0.74 | +2.34 | +34.35 | +41.12 | +6.79 | **+13.41** | $-3.15\,kT$ |
| `capture #3` | +1.02 | -0.32 | +25.60 | +22.95 | +1.24 | **+1.75** | $+4.86\,kT$ |
| `blind-agg #0` (offender) | +0.74 | -0.84 | +26.19 | +19.04 | +11.16 | **-0.81** | $+19.06\,kT$ |
| `blind-agg #3` (offender) | -1.31 | -6.67 | +25.22 | +1.87 | +0.86 | **-13.03** | $+10.40\,kT$ |
| `non-native #0` (competing) | +2.53 | +1.23 | +36.54 | +12.47 | -13.33 | **-0.85** | $+30.06\,kT$ |

### 8.3 Greedy nucleation path & $B_{22}$ verification (`B3_chem`)

1. **Trimer & growth steps resolved**: At $\varepsilon = 0.00325\,kT$ ($2.8\text{ mM}$, $\ln\rho = -13.29$), `step 1 (m6 pair)` shifts from $\Delta G = +1.22\,kT$ to **$-1.27\,kT$**, `step 2 (+2: m1 m3)` drops from $+8.74\,kT$ to **$+2.54\,kT$** ($\Delta\ln K = +6.20$), `step 3 (+2: m1 m3)` drops from $+10.35\,kT$ to **$+1.76\,kT$** ($\Delta\ln K = +8.59$), and the repeatable growth steps `step 10 (+2: m1 m2)` and `step 12 (+3: m0 m1 m2)` turn exergonic (**$-0.61\,kT$** and **$-0.10\,kT$** vs. $+8.91\,kT$ and $+6.89\,kT$ without chemistry).
2. **Second virial coefficient $B_{22} / B_{22}^{\text{HS}}$**: Across $\varepsilon \in [0.00228, 0.00423]\,kT$, $B_{22} / B_{22}^{\text{HS}}$ spans **$+0.42 \pm 0.10$** ($\varepsilon = 0.00228$), **$+0.08 \pm 0.15$** ($\varepsilon = 0.00277$), **$-0.42 \pm 0.24$** ($\varepsilon = 0.00325$), **$-1.21 \pm 0.40$** ($\varepsilon = 0.00374$), and **$-2.50 \pm 0.65$** ($\varepsilon = 0.00423$), sits squarely inside the experimental crystallization slot ($-0.5$ to $-3.5$) at $\varepsilon = 0.00325\text{–}0.00423\,kT$.

