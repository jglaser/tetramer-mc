# Can the Gaussian mixture keep changing during physical sampling?

Yes. The mixture can become an auxiliary part of a fixed joint Markov chain, including reversible births and deaths of components. The essential requirement is a joint target whose **colloid marginal is the intended physical distribution**. Reversible-jump MCMC supplies the dimension-changing transition rule; it does not supply that joint target or make arbitrary online fitting valid.

There are three distinct possibilities:

| Construction | Physical marginal | What can keep changing? | Main limitation |
|---|---|---|---|
| Independent model target `π(X) ρ(θ)` | Exactly `π` | Component count, weights, centers, covariances | A fixed model target cannot retain unlimited uncorrected training updates from the same production history. |
| Normalized conditional target `π(X) ρ(θ\|X)` | Exactly `π` | A model statistically correlated with the current configuration | The conditional must be normalized, and particle moves need its reverse correction or a joint transport. |
| Independent continuing pilot teaches production | Preserves production stationarity | Arbitrary continuing pilot-based fitting | Information must flow from the pilot to production, without production feedback. This is not joint RJ learning from the same chain. |

The constructions and algebra below are our proposed application, not claims that the cited papers already implemented a protein sampler. **The normalized conditional construction now has a fixed-K, Gaussian-mean-only implementation in the spherical runner; see [the implementation and validation](spherical-ensemble.md). RJ births/deaths and accumulating adaptation remain unimplemented.**

## 1. Physical target and the normalization trap

All rigid tetramers are mobile. Let `X` denote their periodic centers and proper orientations, `ν` the common position/Haar reference measure, and `U(X)` the union of depletant-exclusion regions. With ideal depletant activity `z`,

\[
\pi(dX)=Z^{-1}\mathbf1_{\rm hard}(X)e^{-z|U(X)|}\nu(dX).
\]

The proposal model is \(\theta=(K,w_{1:K},\mu_{1:K},C_{1:K})\), together with any discrete chart labels. A sufficient augmented target is

\[
\widetilde\pi(dX,d\theta)=\pi(dX)\rho(d\theta\mid X),\qquad
\sum_{K=1}^{K_{\max}}\int\rho_K(\theta\mid X)\,d\theta=1
\quad\text{for every allowed }X.
\]

Exact stationarity means starting from this joint law. Having only `X ∼ π` with an arbitrary, incompatible auxiliary model is insufficient for that claim. From arbitrary starts, convergence still requires suitable irreducibility and recurrence; the joint-target identity alone is not a mixing proof. A directly generative conditional, as below, can initialize the auxiliary variables exactly given any current `X`.

The ordinary Bayesian-looking alternative

\[
\widehat\pi(X,\theta)\propto\pi(X)p_0(\theta)e^{-\beta L(\theta;D(X))}
\]

has colloid marginal proportional to \(\pi(X)M(X)\), where

\[
M(X)=\sum_K\int p_0(\theta)e^{-\beta L(\theta;D(X))}\,d\theta.
\]

That evidence is generally geometry dependent. It may favor configurations that are easier to fit, which is an extra, unintended many-body interaction. To use the normalized conditional posterior, divide by `M(X)`. This denominator cancels in model updates at fixed `X`; it **does not cancel in physical moves**. An unbiased estimate of `M` is not automatically an unbiased estimate of `1/M`, and inserting a noisy ratio is not a solution without a separate extended-target construction.

## 2. Safe first implementation: a fixed training posterior

Freeze a finite training collection `D0`, but allow the model itself to fluctuate. For example, define one proper auxiliary target

\[
\rho_{D_0}(\theta)\propto
p(K)\,p(w,\mu,c\mid K)
\exp\left[-\beta L(\theta;D_0)
-\alpha\sum_b\frac{w_b}{\sqrt{\det C_b}}\right],
\]

where `c` denotes log-Cholesky covariance coordinates and `1 ≤ K ≤ Kmax`. Choose proper priors and verify that the displayed target is integrable. A simple sufficient choice is bounded log-Cholesky coordinates, which bound the covariance eigenvalues away from zero and infinity and bound a finite-data Gaussian likelihood, together with proper priors on means and weights. A component-count prior can favor smaller mixtures. The displayed peak penalty can suppress narrow components but is not itself a component-count penalty. Any covariance bounds, transformed-coordinate prior densities, and constants depending on `K` must be included consistently.

The full target is `π(X) ρ_D0(θ)`. At fixed model, the physical move is our usual corrected learned proposal. At fixed configuration, any MH/RJ model update targeting `ρ_D0` is valid. Model-update **proposals** can use the current physical configuration to suggest a promising center, but their forward and reverse probabilities must be evaluated. At stationarity, this product target still makes `X` and `θ` independent: current-state guidance cannot silently change the model's target into an accumulating online posterior.

There is only one irrelevant overall normalization constant for the displayed joint model target. Do not replace it by separately normalized conditional posteriors for each `K` and then omit their `K`-dependent evidences.

Bayesian mixtures with a variable component count are a standard RJ application; Richardson and Green used proper hierarchical models and trans-dimensional sampling for Gaussian mixtures. Their inference problem concerns fixed observed data, not a physical system whose own configuration supplies a changing training set. [Richardson–Green (1997)](https://doi.org/10.1111/1467-9868.00095)

## 3. A concrete birth/death rule

Use labelled components; do not silently sort, merge, or delete them. A six-dimensional full-covariance Gaussian has six mean coordinates and 21 covariance coordinates. With `K−1` free simplex weight coordinates, the continuous dimension is `28K−1`.

At fixed `X`, choose a birth with probability \(b_K(X,\theta)\), draw \(a\in(0,1)\) from a known density \(g_K(a\mid X,\theta)\), draw the new mean/covariance coordinates \(\eta_*\) from \(h_K(\eta_*\mid X,\theta)\), and select insertion slot `j` with probability \(s_{\rm ins}(j)\). Set

\[
w'_j=a,\qquad w'_{\ell\ne j}=(1-a)w_\ell,
\]

and retain the existing components in order. This map is invertible: death removes slot `j` and divides the remaining weights by `1−a`. Its absolute Jacobian, relative to free simplex and mean/log-Cholesky coordinates, is

\[
|J_{\rm birth}|=(1-a)^{K-1}.
\]

If the reverse death probability is \(d_{K+1}(X,\theta')s_{\rm del}(j\mid X,\theta')\), the birth ratio is

\[
R_b=
\frac{\rho_{K+1}(\theta'\mid X)\,
d_{K+1}(X,\theta')s_{\rm del}(j\mid X,\theta')}
{\rho_K(\theta\mid X)\,
b_K(X,\theta)s_{\rm ins}(j)g_K(a\mid X,\theta)h_K(\eta_*\mid X,\theta)}
(1-a)^{K-1}.
\]

Accept with `min(1,R_b)`; the matching death has reciprocal ratio. Uniform insertion and uniform deletion both contribute `1/(K+1)` and cancel, but nonuniform choices do not. At component-count boundaries use the actual birth/death scheduling probabilities. If labels are instead quotiented out, the combinatorial factors must be rederived; adding a factorial by intuition is unsafe. Exact zero-weight pruning is not a reversible continuous-coordinate move: use an explicit death or a model with defined discrete inclusion indicators.

Within fixed `K`, means, weight logits, and log-Cholesky coordinates can use random walks or data-guided proposals, with their complete MH ratios. A deterministic EM or covariance shrinkage step is generally many-to-one and cannot simply be called a reversible update. It can define the center of a stochastic proposal with an evaluable density.

This is the dimension-matching and Jacobian principle of RJMCMC specialized to a minimal Gaussian birth map. Split/merge proposals may improve acceptance, but require their own invertible maps and reverse selection probabilities. [Green (1995)](https://doi.org/10.1093/biomet/82.4.711), [Hastie–Green review](https://arxiv.org/abs/1001.2055)

## 4. Genuinely state-correlated learning without an evidence integral

We need not choose a Bayesian posterior as the auxiliary conditional. We can **define a normalized conditional generatively**.

For each `K`, let `v` collect unconstrained mean, log-Cholesky, and `K−1` weight-logit coordinates. Let `F_K(X)` be a deterministic finite fitting procedure applied to the current configuration's contact data. Fix its tie handling, iteration count or convergence rule, chart conventions, and any regularization. Choose a nonsingular matrix `L_K(X)` and a normalized explicit count law `p(K|X)`. Define

\[
K\sim p(\cdot\mid X),\qquad
\eta\sim\mathcal N(0,I),\qquad
v=F_K(X)+L_K(X)\eta.
\]

Its density with respect to `dv` is known exactly:

\[
\rho_K(v\mid X)=p(K\mid X)
\frac{\varphi\!\left(L_K(X)^{-1}[v-F_K(X)]\right)}{|\det L_K(X)|}.
\]

Thus it integrates to one regardless of how good the fit is. This explicitly allows statistically different model distributions in different fluid or crystal environments. It is not a claim that the fitted model is an equilibrium density estimate.

The density coordinates matter. If an RJ implementation uses free simplex weights instead of weight logits, multiply this density by `1/∏_b w_b`; this is the inverse softmax Jacobian. Densities specified in log-Cholesky coordinates need no covariance-space Jacobian until the reference measure is changed to the entries of `C`. The preceding birth formula can then use these correctly converted conditional densities.

For a physical move at fixed model,

\[
R_X=
\frac{\pi(Y)q_\theta(X\mid Y)}{\pi(X)q_\theta(Y\mid X)}
\frac{\rho(\theta\mid Y)}{\rho(\theta\mid X)}.
\]

The second factor can be costly or restrictive if the auxiliary conditional is sharply centered on a different fit at `Y`. Refitting must be reproducible at both endpoints; reusing an old fit only on the forward side changes the rule.

### Jointly transport the model to retain its latent residual

For fixed `K`, carry the same `η` to the proposed configuration:

\[
v'=F_K(Y)+L_K(Y)L_K(X)^{-1}[v-F_K(X)].
\]

The model-map Jacobian is `|det L_K(Y)|/|det L_K(X)|`. It cancels the Gaussian conditional-density ratio. The joint move therefore has ratio

\[
R_{X,v}=
\frac{\pi(Y)\,q_{\theta'}(X\mid Y)}
     {\pi(X)\,q_\theta(Y\mid X)}
\frac{p(K\mid Y)}{p(K\mid X)}.
\]

For a fixed count prior `p(K)`, the last factor is one. Crucially, the reverse proposal uses **the transported model `θ′`**, not the old model. Implementing the state as `(X,K,η)` is particularly clean: the target is simply `π(X)p(K|X)φ(η)` and the density cancellation is a change of coordinates, not an assumption about the fitting procedure's derivatives. Invertibility is required in `η`; the fitter need not be invertible as a function of `X`.

The model distribution can change indefinitely at equilibrium. This is ordinary stationary MCMC on an enlarged state space, not an adaptive kernel whose history is omitted from the state. It does not monotonically accumulate discoveries. A growing training archive would itself need a normalized conditional law and reversible updates, or a separate adaptation theorem. Arbitrary appending to the archive is not supplied by this construction.

### Compatibility with the existing ideal-depletant gate

At fixed endpoint proposals, the existing body-frame auxiliary Poisson cloud gives the physical factor `(1+z/λ)^(G−L)`, with the documented conditional cloud law. For a fixed-model move multiply the existing forward/reverse proposal factor by `ρ(θ|Y)/ρ(θ|X)` before applying the auxiliary MH acceptance. For the jointly transported model use the new reverse density `q_θ′` and the count-law ratio above. The model-density/Jacobian cancellation is independent of the many-body exclusion-union calculation. This statement uses the extended Poisson target; it is not a substitution of an unbiased energy estimate into ordinary Metropolis.

Random particle/anchor selection must also be reversed correctly. Uniform choices from fixed index sets cancel. A neighborhood-dependent selection list generally does not. The new particle endpoint must not be chosen using a fresh Poisson cloud while pretending that cloud was independent of proposal selection.

## 5. A cheaper exact local special case

For a single-particle move, let `Ξ=X_{−i}` be the fixed spectators. A freshly constructed auxiliary model may depend on `Ξ` alone. Its law is identical at the old and new endpoints and cancels automatically. One can fit the spectators, add stochastic exploration, use the resulting proposal for particle `i`, and discard that temporary model. Even a complicated finite randomized fitter is valid if its same auxiliary randomness and unchanged spectator input define the reverse construction.

This is a potentially useful first implementation of continuously environment-informed proposals. It cannot use the moving particle's old pose as training information without adding the corresponding reverse correction. Nor can a persistent fit cache depend on unrecorded history; safe caching is a deterministic function of the current spectator data. RJ may be used inside an auxiliary construction, but is not required for this cancellation.

## 6. Continuing accumulation using an independent learner

There is also a less intrusive route: an independent exploration process maintains the growing training set and model. The production chain uses an externally scheduled supplied model, with the exact fixed-model physical acceptance. Conditioned on the entire externally generated model schedule, every production transition preserves `π`; therefore production started at `π` remains there. Model changes need not diminish for this stationarity argument. The learner must not consume the production chain's states, acceptance decisions, or feedback-dependent scheduling. Use independent randomness and predetermined production-iteration/model-checkpoint assignments. Taking whichever model is newest in wall time is not automatically an independent schedule when production step duration depends on its physical configuration. Arbitrary initialization still requires convergence, and a poorly chosen model sequence can mix badly.

Roberts and Rosenthal explicitly describe this one-way continuing pilot construction in their discussion. Their separate diminishing-adaptation plus containment results concern history-dependent adaptive chains; those hypotheses must not be replaced by the assertion that every frozen kernel is individually correct. [Roberts–Rosenthal (2007), Theorem 2 and discussion](https://www2.stat.duke.edu/homeweb/scs/Courses/Stat376/Papers/AdaptiveMC/RobRosCouplingErgod2007JAP.pdf)

Adaptive incremental Gaussian-mixture algorithms also exist. Their correctness arguments concern controlled adaptation, with nontrivial restrictions on the proposal family; they are not a consequence of attaching an RJ name to component births. [Maire–Friel–Mira–Raftery, Adaptive Incremental Mixture MCMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7224357/)

## 7. What has actually been checked here?

Run the standard-library-only calculation:

```bash
python3 research/validate_rjmcmc_auxiliary.py
```

The [machine-readable results](../research/rjmcmc-validation.json) establish the following limited checks:

| Finite-state construction | Stationary physical marginal |
|---|---|
| Intended target | `(0.2, 0.3, 0.5)` |
| Normalized state-correlated auxiliary, complete correction | `(0.2, 0.3, 0.5)` within `1e−14` |
| Independent auxiliary target, X-guided corrected model proposals | `(0.2, 0.3, 0.5)` within `1e−14` |
| Unnormalized fit factor | `(0.272727, 0.175325, 0.551948)` |
| Normalized conditional but omitted particle-step auxiliary ratio | `(0.298835, 0.407328, 0.293837)` |

These are transition-matrix calculations, not empirical trajectory histograms. The finite models carry different `K` labels but do not themselves test a continuous trans-dimensional Gaussian implementation. Separately, 35 numerical Jacobian/inverse checks cover all birth insertion positions for `K=1,…,7`; the maximum Jacobian error is `5.5e−11`. Another 200 nonlinear continuous Gaussian auxiliary-transport checks verify forward/reverse log flux to `3.6e−15` and the full map Jacobian to `3.2e−10`.

No protein performance claim follows from these algebraic checks. An actionable sequence is: implement fixed-data model RJ as a simple control, implement the spectator-conditioned or normalized current-state auxiliary route for genuine environment dependence, then compare joint transport against holding the auxiliary model fixed. Measure useful environment transitions and repeated association/dissociation per total CPU time, including fitting and reverse-density evaluation. The existing static mixture is the reference control.
