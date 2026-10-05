# Descriptive covariance of the saved source cage

The four completed source-start local-control trajectories show much narrower rotational motion than the smallest isotropic source guide, together with correlated translation and rotation. This is a description of their sampled cage, not an equilibrium width or a converged contact-basin distribution.

The reduction retains every elementary state, including rejected-state residence: 40,960 production observations and 5,120 warmup observations, summarized separately. Production contained only 130 accepted moves (29, 22, 36 and 43 by stream). Repeated observations therefore do not represent 40,960 independent samples. There were no Cayley seams or discarded states.

The fixed chart uses anchor 16 and the certified original body 77 pose. Its coordinates are translation relative to that source and `ell*q_vector/q_scalar`, where the relative rotation is `R_relative * R_source_relative^T`. The report also gives dimensionless Cayley coordinates and scaling by 0.1 Å for translation and `ell*tan(0.5°/2)` for angular coordinates. The covariance denominator is N, representing the empirical residence distribution. No proposal density was fitted or evaluated.

Streams 0–1 were designated training and streams 2–3 held out before reduction. The training coordinate standard deviations relative to the smallest isotropic guide are:

| Coordinate | Translation x | Translation y | Translation z | Cayley x | Cayley y | Cayley z |
|---|---:|---:|---:|---:|---:|---:|
| Standard deviation / guide scale |0.486 |0.442 |0.365 |0.192 |0.122 |0.293 |

Translation standard deviations are approximately 0.049, 0.044 and 0.037 Å. Angular standard deviations correspond to approximately 0.096°, 0.061° and 0.146° per axis in the small-angle limit. The training scaled covariance eigenvalues range from 0.00601 to 0.40750, with condition number 67.8. Its strongest coordinate correlation is −0.639 between translation x and y.

Across all production states, the 95th-percentile displacement from the original source is 0.242 Å and 0.513°; maxima are 0.263 Å and 0.644°. Training and held-out means and covariances differ. Between-stream mean differences account for 32.8% of the pooled scaled covariance trace, so a Gaussian built from these data still needs an independent coverage test. A full-covariance versus diagonal-covariance comparison with the same center and regularization can separate correlated shape from simple narrowing. The listed regularization formulas are unevaluated candidates, not results of an optimization.

The complete report includes all per-stream and pooled means, full 6×6 covariances, within/between decomposition, quantiles, physical units, scaled units, residence counts and authenticated input paths:

- [Completed report](../results/source-cage-covariance-20261005-v3/result/report.json), SHA256 `6e8d68a3ae0fe5aef0622172d7555521b7acc8907017bb42da28dcda50e0ef61`.
- [Execution status](../results/source-cage-covariance-20261005-v3/execution/status.json), SHA256 `15249204eaaebf55cf2af8ce17d002804663fe1967aed91d79e03214b315be36`; both jobs passed and drained. The reduction consumed 2.09 CPU seconds under the frozen 20 CPU / 45 wall-second limit, one thread and 2 GiB.
- [Seven deterministic tests](../results/source-cage-covariance-20261005-v3/tests.json), SHA256 `8085f8622c5cf8053be557e78ecfeef93bee40131821416acd40ce7496b90fcc`.
- [Frozen protocol](../results/source-cage-covariance-20261005-v3/protocol.json), SHA256 `7bde674bdabf90fb91b9ad1b44d47fb26eb9936138d1b1f411b0d33d42fb8fc1`.
- [Reducer](../tools/analyze_source_cage_covariance.py), SHA256 `0d669136cdc8346d8a1201db4956dc831435b2b48c926e6107467998e5bd36f9`; [tests](../tools/test_source_cage_covariance.py), SHA256 `a1640f82950c2b6d488341c3602196ae36474dd7eb39afb37444e97489d069ed`.

The unlaunched first metadata version is preserved. Version 2 passed six tests but stopped at metadata admission because it required a null initial-pose field; the actual controls explicitly stored the exact certified source pose. Its failure occurred before reading coordinates or calculating statistics. Version 3 admits either that exact pose or the equivalent null/default representation and adds a regression test. No scientific data, stream selection, chart, reporting threshold or allocation changed.

No atom geometry, Poisson clouds or new poses were evaluated. This remains a conditional one-mobile-body result in the fixed historical 500 μM neighborhood at radius 1.5 Å and activity 0.035 Å⁻³, not evidence for finite-system assembly at 106.8 μM.
