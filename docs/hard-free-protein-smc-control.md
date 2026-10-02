# Protein SMC bridge control

The independent contact integrals suggest that the historical broad-mutation
SMC estimate lost an important part of the native region during annealing.
The [saved-cloud diagnostic](smc-bridge-bottleneck.md) predicts a less severe
intermediate bottleneck when the frozen hard-free guide initializes the path.
A fresh SMC calculation is needed to test that prediction. Retrospectively
reweighting the IID draws cannot supply that independent evidence.

[`run_hard_free_protein_smc.py`](../tools/run_hard_free_protein_smc.py) freezes
and executes four fresh populations. It requires completed, hash-bound larger
IID and probability/cloud-intensity controls, including their failed scientific
diagnostics. Successful execution is the prerequisite; convergence is not
silently promoted to a pass. The eleven completed sphere/Haar references and
their exact binary are reused without rerunning them.

The new control retains the historical broad-mutation allocation:

| Quantity | Fixed value |
|---|---:|
| Independent populations | 4 |
| Unconditional initialization draws per population | 262,144 |
| SMC particles | 2,048 |
| Annealing increments | 128 |
| Local attempts per particle per increment | 4 |
| Translation steps | 0.2, 2 Å |
| Rotation steps | 1.5°, 15° |
| Independent incremental clouds | 2 |
| Auxiliary intensity ratio | 128 |

The 92-component guide uses defensive probability 0.5 and all three hard-free
translation axes. The physical endpoint remains the repaired shape, radius
1.5 Å, activity 0.035 Å⁻³, original fixed scaffold, R4 and physical Jacobian.
The change is the normalized initializer and corresponding intermediate
density, with the [validated acceptance correction](hard-free-smc-initial-guide.md).
The test has no native-label filtering or adaptive refitting.

The primary physical comparison will use each population's normalizer times
its terminal indicator fraction, for total native, old-R5 native intersection
and the remaining native region. Both the historical broad SMC control and
the independently completed larger IID populations are references; their
populations are not pooled. Intermediate class occupancies and surviving
initial families can test the proposed bottleneck mechanism. A terminal
competing-contact count of zero cannot supply a competing mass or a finite
native/competing free-energy contrast. The class-balanced IID evidence remains
necessary for that quantity.

The runner starts at most two physical workers, then four independent full
Python audits, under the existing eight-physical/32-worker global limits.
Every initial attempt, invalid zero and mutation journal entry is retained.
Any failure stops new launches and drains started children; a claimed campaign
cannot be retried or overwritten. It finishes at `audited_awaiting_classification`:
that state certifies execution/audit completion only. Contact classification
and statistical comparison are subsequent, once-only operations.

The frozen config retains its existing absolute shape path, which is checked
before dispatch. Executed Python source, binary, guide and other inputs are
copied and hash-bound. Preparation-time repository files are recorded without
requiring unrelated future repository development to stop while the copied
workers run. Seven new synthetic tests cover exact allocation, incomplete and
tampered prerequisites, preserved failed convergence diagnostics, host capacity,
failure accounting and refusal to restart.

Full guide auditing may cost tens to over a hundred CPU-hours. A fixed-scaffold
precomputation prototype saved approximately 23% on a single timed pass through
32 archived poses, with identical complete densities and interval predicates.
That exploratory timing is insufficient to justify another production audit
engine. The prototype is archived at
`results/smc-audit-precomputation-prototype-20261002`; the validated audit remains
unchanged. No new physical draws, clouds or classifiers were used by that timing.

Completion of this control cannot decide finite-system assembly or instability.
It can resolve, or expose continued uncertainty in, the disagreement between
two methods for the same conditional contact integral. Full-vessel coverage
and finite-system stability remain separate requirements.

## Frozen execution

The controller was frozen and launched after the full 786,432-draw IID campaign
completed. Directory:
`/vast/xvg/tetramer-mc-runs/hard-free-protein-smc-control-20261002`.
Plan SHA-256:
`a683f6615a86c8ac90cc6692321e770c82fbbda0f68441c205d552ebfed6afdb`.
Controller PID: 667281. Read `status.json` for current phase; do not rerun a
claimed execution. All six preexisting user simulations were preserved.
The exact command and observed prelaunch capacity are retained in `launch.json`.
The [seven-test receipt](../results/hard-free-protein-smc-controller-validation-20261002/validation.json)
binds the controller source used for preparation.

The [contact-analysis preparation](hard-free-protein-smc-analysis.md) is now
frozen separately and queued behind successful completion of all four audits.
It preserves the predeclared stage profile, complete native classifier and
whole-population estimator. The queued analysis starts no physical jobs and
does not change this execution or its guide.
