# Contact refinement and separate confirmation

The completed [131,072-attempt pilot](contact-bank-reference-results.md) remains an unresolved earlier experiment. Its near agreement in total native weight hid opposite changes in the old R5 intersection and the remaining native R4 mass. Its rows are now training evidence. The new allocation described here is a separate, frozen confirmation design; this document does not report its outcome.

The [saved-data diagnostic](../runs/contact-refinement-score-20260922/report.md) evaluated twelve Gaussian designs using eight whole-pilot-population holdouts. Each held-out population was excluded from both new component anchors and covariance neighbors. The three training strata were full native intersected with old R5, remaining full native R4, and contact without native entry. Old R5 uses its original chart radius ≤5, strict original metric q>1 and original D170 capture; the physical weights receive no second chart Jacobian.

Each training population supplies one largest-original-weight anchor per stratum. Its 64 or 128 nearest distinct same-stratum poses in original u determine a full covariance. Fitting weights are proportional to the original importance weight divided by that row's source unconditional N, capped only during fitting at normalized weight 1/16. The grid compared anchor versus capped-neighborhood weighted-mean centers and geometric-floor covariance multipliers 1, 0.25 and 0.0625. Physical scoring weights were never clipped.

The declared selection minimizes the largest pooled held-out physical second-moment ratio across the three named strata. It selected **weighted-mean centers, 128 neighbors and floor factor 1**:

| Training stratum | M2 ratio to old bank | M2 contribution ESS | Largest M2 term | Improving folds |
|---|---:|---:|---:|---:|
| Old R5 intersection native | 0.4638 | 16.23 | 22.98% | 8/8 |
| Remaining native R4 | 0.1341 | 9.49 | 19.98% | 8/8 |
| Contact without native entry | 0.5065 | 41.77 | 9.17% | 7/8 |

Dropping one scored population, without refitting its already constructed fold proposal, gives ratio ranges 0.334–0.562, 0.126–0.157 and 0.444–0.599 respectively. One no-entry fold worsens to 1.818 times the old-bank M2. The quarter-floor counterpart is a close alternative: its three ratios are 0.2818, 0.1152 and 0.5232. The sparse moment ESS does not establish a meaningful superiority between these nearby designs.

For a row drawn from its actual original proposal q_s, each saved cloud weight is J W_k/q_s. The scoring term `(J W_1/q_s)(J W_2/q_s) q_s/q_target` estimates the physical second moment because the clouds are conditionally independent. All original attempted-draw denominators remain intact, including exterior and hard-invalid zeros. Pooled scores average different fold-specific proposal moments; they are not the second moment of the final all-data guide. Physical-only ESS proxies at 524,288 attempts are approximately 3,030, 1,948 and 7,486 for the three strata, and double at 1,048,576 attempts. They omit new Poisson-cloud noise and inherit the uncertain observed tails; they are not predicted measured ESS or convergence evidence.

The [preparation](../runs/refined-contact-bank-preparation-20260922/plan.json) fits 24 new components from all eight training populations and retains the complete old 56-component bank. With U the normalized uniform density on the six-dimensional radius-4 ball, and G the globally normalized, **untruncated** Gaussian mixtures, the main density is

`q_bank(u) = 0.5 U(u) + 0.25 G_old56(u) + 0.25 G_new24(u)`.

The new Gaussian group allocates one third of its mass to each training stratum, equally among its eight population components. This gives **q_bank ≥ 0.5 q_old_bank globally**. The wide arm multiplies every full covariance by four and has the corresponding guarantee **q_wide ≥ 0.5 q_old_wide**; this is not a bound against the old bank width. The alpha02 control instead uses `0.2 U + 0.4 G_old56 + 0.4 G_new24`, giving `q_alpha02 ≥ 0.4 q_old_bank` and `q_alpha02 ≥ 0.4 q_bank`. The 50% uniform floor and half-old-density bound therefore describe the main bank and corresponding wide proposals, not every control. Gaussian attempts outside R4 remain zero-weight attempts; no truncation, retries or renormalization by acceptance is allowed.

The [confirmation controller](../tools/run_contact_confirmation.py) fixes **2,228,224 new attempts** across twenty independent populations:

| Arm | Populations | Attempts per population | Uniform probability | Covariance factor | λ/z |
|---|---:|---:|---:|---:|---:|
| bank | 4 | 131,072 | 0.5 | 1 | 128 |
| wide | 4 | 131,072 | 0.5 | 4 | 128 |
| small | 4 | 32,768 | 0.5 | 1 | 128 |
| alpha02 | 4 | 131,072 | 0.2 | 1 | 128 |
| intensity256 | 4 | 131,072 | 0.5 | 1 | 256 |

The physical target remains the repaired shape, two fixed neighbors, complete R4, activity 0.035 Å⁻³, depletant radius 1.5 Å and full native classifier. Every population finishes its fixed allocation before the new campaign's archived audits. No earlier physical campaign, classifier output or audit is replayed. At most eight physical workers and four classification workers run concurrently.

The [confirmation analyzer](../tools/analyze_contact_confirmation.py) records full native, contact without native entry, native R5 intersection and remaining native R4 as decision regions. It computes the supplementary partition during the same first native/contact classification pass, saves both masks, and verifies that the two subsets sum to full native for Qz and Q0 at row and population levels. Records retain original u, proposal log density, original metric q, cloud pairs, old-chart radius, draw identities and source N for later read-only diagnostics. The original radial bins [0,2,3,4], angular projection-squared bins [0,4,9,16] and all 64 original orthants remain visible; no bin is merged away.

Bank is compared directly with each of wide, small, alpha02 and intensity256. Each of the four decision-region Qz and Q0 masses, and the native-versus-no-entry ΔF itself, must agree within both **0.2 log units/kBT and three combined observed population standard errors**. Total Q0 agreement is also required. Original bins with at least 1% of their parent region's observed Qz in either arm face the same 0.2/three-SE check. Compensating native/no-entry shifts cannot pass merely because each individual shift is small.

For bank, wide, alpha02 and intensity256, every decision region also requires population RSE ≤10%, importance ESS ≥200 and largest draw ≤2%; the approximate paired-population Student-t3 95% ΔF interval requires half-width ≤0.5 kBT. The small arm's standalone quality and interval precision are diagnostic because it has one quarter of the attempts. **Every bank-versus-small mass, direct ΔF and significant-bin agreement check remains mandatory.** Unobserved mass remains unresolved. Contact without entry is a reporting class, not evidence for a distinct competing basin.

An independent geometric density-bridge SMC control is under implementation and requires its own frozen target, tests and validation. It is separate from this importance proposal and is not an already achieved coverage result. Neither a favorable training score nor passing this importance confirmation would establish full-vessel coverage, stationary contact mixing, finite-system assembly, native stability or native instability.

The commands below document the workflow; writing this document does not execute them. Existing completed output directories are immutable, so these construction/run commands apply only to a fresh output tree. The controller's `validate` and `preflight` actions are read-only checks. The archived executable and its exact archived source tree are reused rather than rebuilt.

```bash
cd /home/xvg/tetramer-mc

# Saved-data scoring only; the named scoring output already exists.
/home/xvg/protein-nucleation/.venv/bin/python tools/score_contact_refinements.py \
  --out runs/contact-refinement-score-20260922

# Fit and freeze the selected guide without physical draws.
/home/xvg/protein-nucleation/.venv/bin/python tools/prepare_refined_contact_bank.py \
  --out runs/refined-contact-bank-preparation-20260922

# Freeze the separate confirmation and its reviewed executable/source pair.
/home/xvg/protein-nucleation/.venv/bin/python tools/run_contact_confirmation.py freeze \
  --out runs/contact-confirmation-campaign-20260922 \
  --package runs/refined-contact-bank-preparation-20260922 \
  --binary runs/contact-bank-pilot-20260922/common/latent-region-normalizer \
  --source-bundle runs/contact-bank-pilot-20260922/common/source-bundle.json \
  --source-root runs/contact-bank-pilot-20260922/common/source

/home/xvg/protein-nucleation/.venv/bin/python \
  runs/contact-confirmation-campaign-20260922/common/controller.py validate \
  --out runs/contact-confirmation-campaign-20260922

/home/xvg/protein-nucleation/.venv/bin/python \
  runs/contact-confirmation-campaign-20260922/common/controller.py preflight \
  --out runs/contact-confirmation-campaign-20260922

# New physical confirmation only, after tests and the separate freeze.
/home/xvg/protein-nucleation/.venv/bin/python \
  runs/contact-confirmation-campaign-20260922/common/controller.py run \
  --out runs/contact-confirmation-campaign-20260922

# One first classification pass, only after all new populations/audits complete.
/home/xvg/protein-nucleation/.venv/bin/python \
  runs/contact-confirmation-campaign-20260922/common/analyze_contact_confirmation.py \
  --campaign runs/contact-confirmation-campaign-20260922 \
  --out runs/contact-confirmation-comparison-20260922 --workers 4
```

The original pilot and its failed convergence gates remain documented in [contact-bank-reference-results.md](contact-bank-reference-results.md). The new preparation, campaign status and eventual comparison are separate records; neither overwrites that failure.
