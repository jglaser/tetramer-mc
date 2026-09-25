# Fixed-length oligomer guide: independent correctness argument

Fix the labelled subset S, its internal rigid geometry D, and spectator environment E for the complete proposal. Use a reference measure on the six collective coordinates. Let pi be the physical conditional density and H the strictly positive, finite, possibly unnormalized guide on physical support. Neither H nor its normalizer is an approximation to the physical target in the correctness argument: H only defines a proposal.

1. Inner guide step. From an arbitrary raw proposal q, retain a self-loop on rejection using
   alpha_H(x,y) = min(1, H(y) q(y,x) / (H(x) q(x,y))).
   The completed kernel K obeys H(dx) K(x,dy) = H(dy) K(y,dx).
   An augmented involutive raw proposal may establish the same symmetry without a common transition density; its full guide acceptance must include the existing auxiliary and chart correction.

2. Fixed length. For the SAME K repeated m times, reverse every intermediate path:
   H(x_0) product_{r=0}^{m-1} K(x_r,dx_{r+1})
     = H(x_m) product_{r=0}^{m-1} K(x_{r+1},dx_r).
   Integration over the intermediate variables proves K^m is H-reversible. Rejected self-loops are part of that path, including a rejection at the final step. A state-independent random m is also valid by mixing reversible kernels. A stopping rule depending on acceptance, position, score, or CPU time is not justified. Repeating different individually H-reversible kernels in a fixed order need not be reversible; use the same random-scan kernel each step or a palindromic schedule.

3. Outer physical acceptance. Write w(x)=pi(x)/H(x). The endpoint correction is
   alpha_outer(x,y) = min(1, w(y)/w(x))
                    = min(1, pi(y) H(x)/(pi(x) H(y))).
   Its accepted stationary flow is
   pi(dx) K^m(x,dy) alpha_outer(x,y)
     = H(dx) K^m(x,dy) min(w(x),w(y)).
   Both factors on the right are symmetric under x <-> y. Rejection completion is Markov and preserves pi by the existing accepted-flow theorem. The unknown normalization of H cancels, as does the normalization of pi. If pi can be zero, work on physical support, or use zero-flow conventions as in the existing MH theorem. H must cover that support.

4. Many-body depletion. The exact exponential physical ratio is not replaced by a noisy plug-in estimate. The implemented conditional gained/lost Poisson gate receives log H(old)-log H(new) as its base/proposal correction. Its independent count-reversal identity is the bridge to the same accepted-flow symmetry. This requires correct gained/lost geometry/count laws and the existing map's guide balance; these are implementation obligations, not consequences of this finite-state test.

5. One-step limitation. For m=1 the nested scheme is delayed acceptance. At an off-diagonal raw transition let a=H(y)q(y,x)/(H(x)q(x,y)) and b=pi(y)H(x)/(pi(x)H(y)). Then
   min(1,a) min(1,b) <= min(1,ab).
   Thus the m=1 construction cannot increase physical off-diagonal transition probabilities over applying direct physical MH to the same raw proposal. It can save expensive physical evaluations. A sampling gain requires useful multiple inner steps, a changed raw proposal, or a better cost tradeoff; it cannot follow from guide rejection alone.

## Independent exhaustive executable reference

tests/oligomer_guide_balance.rs is a dependency-free Rust integration test. It enumerates all entries of a four-state matrix with nonuniform H and pi, and an asymmetric raw q. Six tests cover:

- Normalization, guide detailed balance, physical detailed balance, and stationary pi at m=0,1,4,16.
- Equality between the H endpoint correction and full MH using the exactly computed K^m transition matrix.
- Arbitrary positive guide/physical rescaling invariance.
- Deliberately omitting the outer correction, which preserves H rather than pi.
- The one-step delayed-acceptance inequality, strict on the fixture.
- Rejection self-loops and fixed/state-independent path length; accepted-jump and state-dependent stopping counterexamples.

Command:
```bash
cargo test --offline --locked --test oligomer_guide_balance -- --nocapture
```

All six pass. Maximum corrected detailed-balance error is 1.322e-18; maximum corrected invariance error is 4.229e-17. Omitting the outer correction produces physical invariance errors 0.478242, 0.618574, and 0.649652 for m=1,4,16. Dropping rejected states while retaining the H correction produces error 0.065428; selecting one or four inner steps from the current state produces error 0.000755. These are deterministic finite-state checks, not formal verification of the continuous geometry or Rust production sampler.

No new Lean theorem is claimed. Existing accepted-flow completion/invariance applies once the displayed symmetric-flow premise is established. The previous iterate-invariance theorem alone is insufficient: endpoint H-ratio correction needs REVERSIBILITY of K^m, not only invariance.

## Physical validation allocation

[`oligomer_guide_stationarity.rs`](../tests/oligomer_guide_stationarity.rs) runs the actual production cluster phase and conditional Poisson gate for three hard spheres with depletant radius 0.14, activity 5, auxiliary intensity 5, and wall radius 3.5. Triple exclusion intersections are geometrically impossible in this limit; the equilibrium reference was independently computed from analytic pair lenses. The completed reference is reused from `runs/cluster-phase-validation-20260925/stationarity.log` (SHA-256 `ef6c1d564a84a7dbd03577ca1e40d4883d38c51aafebf97644ff5a7a58cd3697`).

The fixed allocation is four independent streams per inner length 1, 4, and 16: two dispersed and two aggregated starts, each with 2,000 discarded burn-in sweeps and 8,000 retained production sweeps. Each sweep contains the same three single-body attempts and fixed-duration collective phase. Rejections remain in the trajectory. Tests compare contact count, largest component size, mean pair distance squared, and orientation against the archived reference, across initializations, and across guide lengths, using 500-sweep blocks and independent-stream variation. They require accepted nonidentity guide endpoints, as well as contact attachment/detachment/exchange coverage, so unchanged ordinary kernels cannot alone constitute success.

The physical test **passed** for all twelve chains. Reference, initialization, and guide-length comparisons met the predeclared test thresholds, with actual accepted nonidentity guide endpoints in each arm. The full guide-validation command passed 18 tests overall, including six exhaustive finite-state tests and the physical test. No completed reference calculation was rerun.

Pooled means and approximate diagnostic 95% interval half-widths are:

| Observable | Independent reference | 1 inner step | 4 inner steps | 16 inner steps |
|---|---:|---:|---:|---:|
| Exclusion-contact count | 0.50787 ± 0.00486 | 0.50759 ± 0.01556 | 0.49872 ± 0.01783 | 0.50666 ± 0.01861 |
| Largest component size | 1.50284 ± 0.00467 | 1.50281 ± 0.01482 | 1.49359 ± 0.01679 | 1.50075 ± 0.01808 |
| Mean squared pair separation | 9.60459 ± 0.01717 | 9.62899 ± 0.05721 | 9.61581 ± 0.08080 | 9.62540 ± 0.06524 |
| Mean quaternion scalar squared | 0.25021 ± 0.00098 | 0.24656 ± 0.00909 | 0.25065 ± 0.00515 | 0.24805 ± 0.00474 |

Each guide interval uses Student's t multiplier with three degrees of freedom times the larger of the between-stream standard error and pooled block standard error. This is a conservative diagnostic choice for four independently seeded streams, not a rigorous convergence guarantee. The independent importance reference uses the normal 1.96 multiplier. The physical test's separately predeclared assertions used five combined standard errors plus 0.002, not these descriptive 95% intervals.

| Guide attempts / inner length | 1 | 4 | 16 |
|---|---:|---:|---:|
| Outer guide attempts | 23,848 | 23,029 | 23,486 |
| Inner attempts | 23,061 | 88,996 | 363,616 |
| Nonidentity endpoints proposed | 605 | 1,966 | 5,073 |
| Nonidentity endpoints physically accepted | 154 | 441 | 1,158 |
| Collective attachments / detachments | 550 / 570 | 572 / 623 | 726 / 724 |
| Collective contact exchanges | 26 | 35 | 49 |

Whole-system trimers have no external anchor and produce null guide events without inner attempts. Attachment, detachment and exchange counts include ordinary local collective moves; they must not be attributed exclusively to the guide. Accepted-guide counts exclude unchanged endpoints. These checks establish nontrivial execution and reference agreement in this analytic limit. More accepted guide endpoints alone do not establish a sampling speedup.

The parsed reference, all twelve chains, all three pooled estimates, counts, interval construction, and source hashes are archived in `runs/oligomer-guide-validation-20260925/physical-reference-summary.json`. The original output is `runs/oligomer-guide-validation-20260925/guide-tests.log`. This validation does not establish protein assembly, protein mixing efficiency, or convergence in the protein system.
