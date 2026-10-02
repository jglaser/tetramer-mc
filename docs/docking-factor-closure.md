# Docking density and the prepared map factors

Map-based docking now draws and scores with the same prepared Gaussian factors.
Previously, `DockingProposal` exported the input model's factor as a covariance
and refactored it twice: the map subtracted an accumulated sum, while separate
component scorers repeatedly subtracted individual products. Singleton full
mixture diagnostics also used the original input-model factor. These binary64
operations can produce different coefficients for an ill-conditioned covariance.

[`src/docking.rs`](../src/docking.rs) now evaluates every learned map branch with
`FixedBasinInvolution::log_density`, after the same reciprocal pose inversion
used by generation. Full mixtures sum these branch densities and their original
normalized weights. The separate component scorer objects are removed. This
covers static and posterior involutions, member-chart mixtures, target-only
independent draws, and the dimer proposals that use those interfaces. Reciprocal
inversion still has unit absolute Jacobian in translation volume times normalized
Haar measure. Uniform branches, periodic image restrictions, numerical nulls,
and public trace formats retain their existing roles.

`DockingMethod::Mixture` remains a separate control: it delegates both generation
and scoring to `FrozenRelativePoseProposal`, whose original prepared factors
already agree internally. The change introduces no random draws. Corrected
posterior probabilities can change subsequent fixed-seed trajectories; this is
an explicit numerical behavior change, not a promise of historical bitwise replay.
The direct mixture control retains its exact draw, score, format and RNG replay.

The fix shares the coefficients actually used by the map; it does not eliminate
floating-point pose round trips, finite RNG precision, Cayley seams, or all
numerical error. It neither refits an atlas nor establishes physical convergence.
The retained one-step stationarity flag and other physical validation gates
remain in force. The diagonal identity-covariance control did not have the
factor discrepancy demonstrated by the anisotropic fixture.

Validation used `target-validation-line-guide`, at most four build jobs and one
test thread: **28 docking/member/dimer tests and six periodic transport tests
passed**. The four new tests in
[`tests/docking_factor_closure.rs`](../tests/docking_factor_closure.rs) include a
fixed anisotropic covariance whose old duplicate scorer differs by approximately
3.3e-8 in log density, complete reciprocal/member mixtures, target-only generation,
singleton full densities, expanded corrections, exact seams, and direct-mixture
RNG replay. Existing runner restart, posterior, reciprocal and periodic image
controls also passed. No physical campaign was launched and the production
executable was not replaced.

The [protein destination screen](dimer-destination-probe.md), its frozen binaries,
source bundles, original authentication failure and recovered audit are unchanged.
Their [factor report](../results/dimer-destination-probe-20261002/factor-quantification.json)
describes the historical implementation. Its Python constructor binder is
deliberately tied to that old source and must not be silently reused for a new
executable. Future numerical audits must bind the new source and reconstruct
the map's prepared-factor density path; historical receipts remain historical.
