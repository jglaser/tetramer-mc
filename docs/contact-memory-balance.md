# A reversible memory of complementary contact poses

The memory contains a fixed number `M` of relative pair poses. Each pose describes
one copy of the rigid sphere-union shape relative to an identical partner fixed
at the origin. Slots are separate pair systems, not mutually interacting bodies.
No production history is appended, and no native contact template is needed.

For exclusion union `E_A` at pose `A=(t,R)`, the memory target is

\[
\rho(A_{1:M})=Z_{\rm pair}^{-M}\prod_{m=1}^M
 \mathbf1_{|t_m|<R_{\rm mem}}\mathbf1_{\rm hard}(A_m)
 \exp[-z_{\rm mem}|E_0\cup E_{A_m}|].
\]

Translations use ordinary volume measure; orientations use normalized proper
Haar measure. A bounded translation ball and compact rotation group make this
law normalizable whenever its hard-free support has positive volume. The
unknown pair partition function is independent of the production configuration.
The ball constrains the pair separation; it is **not** an impermeable solvent
wall or a condition that the moving body's atoms fit inside that ball.

The joint target, with production `X`, active chart labels `c` and Gaussian mean
residuals `eta`, is

\[
 \Pi=\pi(X)\rho(A_{1:M})p(K)
      \prod_{k=1}^K p(c_k)\varphi_6(\eta_k).
\]

Integrating the auxiliary variables leaves the original physical target `pi`.
The memory bath can differ from production's bath: it changes the distribution
of proposal information, not the production interaction. This is a fixed joint
Markov chain, not an unrestricted adaptive fitting procedure.

## Memory transitions

Select one of the `M` slots uniformly. A state-independent mixture selects either
a local Gaussian translation and proper Cayley rotation, or an independent
uniform-ball/Haar redraw. The radius for the latter is `R_mem * U^(1/3)`, not a
uniformly drawn radial coordinate. The local increment law is inverse-symmetric;
the redraw density is constant at both allowed endpoints. Thus both proposal
ratios are one. Hard-overlapping or outside-ball endpoints are rejected without
retrying. The existing exact conditional Poisson gate against the one fixed
partner supplies the remaining depletion acceptance.

Here is the explicit auxiliary balance calculation. Let `E` be the moving body's
exclusion union in its own body frame, `T_A` its rigid pose map, and

\[
 O_A=\{u\in E:T_Au\in E_0\},\qquad
 v_g=|O_B\setminus O_A|,\quad v_l=|O_A\setminus O_B|.
\]

Thus gain/loss refers to **overlap coverage**, not additional excluded volume.
Both maps preserve volume. At allowed endpoints the pair-density ratio is

\[
 \frac{\rho(B)}{\rho(A)}=\exp[z_{\rm mem}(v_g-v_l)].
\]

The implemented gate samples the two independent conditional Poisson counts

\[
 G\sim\operatorname{Pois}(\lambda v_g),\qquad
 L\sim\operatorname{Pois}((\lambda+z_{\rm mem})v_l),\qquad
 W=(1+z_{\rm mem}/\lambda)^{G-L}.
\]

Only membership checks and a conservative sampling envelope are needed;
the volumes in these equations are not evaluated numerically. The reverse
auxiliary realization uses the same body-frame points and exchanges the gain
and loss roles. Its count law relative to the forward law satisfies

\[
 \frac{P_{B\to A}(L,G)}{P_{A\to B}(G,L)}
 =e^{-z_{\rm mem}(v_g-v_l)}
   (1+z_{\rm mem}/\lambda)^{G-L}.
\]

Uniform point-location factors cancel on the paired regions as well. Multiplying
by the pair-density ratio cancels the volume exponential, leaving precisely
`W`. The Metropolis rule `min(1,W)` and its reverse `min(1,1/W)` therefore give
detailed balance on this extended space; integrating out the points preserves
the pair-memory target. At zero activity the implementation directly returns
`W=1`. This is not a noisy plug-in energy or the Metropolis rule applied to an
unbiased weight estimate.

Other memory slots and production bodies never enter this pair's spectator
union. Accepted updates replace one memory pose. For spheres of core radius `a`,
an independent reference law is

\[
 p(r)\propto r^2\exp[z_{\rm mem}V_{\cap}(a+r_d,r)],
 \qquad 2a\le r<R_{\rm mem}.
\]

Only initialization uses an outermost radial contact followed by an explicit
outward gap. Initialization is reproducible and geometry-only, but is not an
equilibrium sample. Subsequent transitions use the entire hard-free ball,
including disconnected or interlocking pockets inaccessible along that initial
entry ray. Practical visits to such pockets are not guaranteed.

## From memory to a normalized proposal

The dynamic dictionary consists of the original frozen charts followed by one
Gaussian chart at each current memory pose. Original weights are multiplied by
`1-memory_mass`; each memory slot has weight `memory_mass/M`. These weights and
the label order do not depend on memory coordinates. Their sum is one at every
memory state, so the existing RJ birth prior stays fixed.

Each new chart has mean zero in its own translation/Cayley coordinates, with
positive independent Cartesian translation and per-axis small-angle widths.
Its covariance is full rank. The existing normalized Haar Jacobian is retained.
This initial implementation does not infer a contact Hessian or use rolling
cross covariance. A Gaussian may propose overlap or a distant pose; its full
unconditioned density is evaluated and the physical rejection rule handles it.

Production physical moves hold memory poses, labels and `eta` fixed. The current
history-free mean fitter may still change as `X` changes: the reverse proposal
must therefore be reconstructed at the proposed `X`. The memory-density factor
cancels because memory is unchanged. For a production move, use its own
many-body spectator exclusion union in `O_X`; the same auxiliary calculation
gives acceptance

\[
 \min\!\left[1,
   \frac{q_{\theta(Y)}(X\mid Y)}{q_{\theta(X)}(Y\mid X)}
   (1+z_{\rm prod}/\lambda)^{G-L}\right].
\]

The endpoint proposal is selected before the cloud is sampled. Uniform
particle/anchor selection probabilities cancel; nonuniform selections would
need their own reverse factors. A memory update changes `A` with `X`, labels
and `eta` held fixed; its acceptance contains only the pair-memory target ratio.
The implied Gaussian parameters change deterministically, which adds no extra
factor in the stored latent coordinates. No acceptance factor should be added
for whether the new memory produces a more useful production proposal.

At fixed memory, the existing labelled RJ insertion/deletion and unit latent
Jacobian remain valid. The original dictionary must be used when rebuilding;
repeatedly appending to an already augmented dictionary would change dimensions
and the label prior. Checkpoints need the complete memory poses, labels, `eta`,
configuration, and RNG continuation information.

## Interpretation and limitations

This is bounded equilibrium memory. It can retain a complementary arrangement
while production explores elsewhere, then suggest related arrangements again.
It can also lose that arrangement. It does not monotonically accumulate a
training archive or estimate basin weights from accepted-move counts.

The pair memory omits cooperative interactions with a third or later neighbor.
Production's many-body acceptance corrects proposed configurations, but cannot
make an uninformative memory efficient. In particular, pair contact discovery,
registered incorporation, and equilibrium crystal stability are separate tests.
The memory ball size, bath activity, proposal widths, number of slots and update
budget are explicit tunable choices; defaults are a pilot, not an optimization.
Setting `attempts_per_sweep` to zero freezes the prepared bank and provides a
control for the benefit of its extra initial charts. This still preserves the
physical production target with a fixed bank. It does not equilibrate memory
under `rho`; the complete auxiliary chain cannot mix over bank configurations.
Calling this native-free refers to interparticle proposal information. The
rigid input shape can itself encode a native tetramer, and any retained frozen
base atlas must be separately labelled as native-informed or geometry-only.

## Running and controls

Add `"contact_memory": {}` to a spherical learned-proposal configuration to
enable defaults. The bank works with a fixed dictionary, conditional mean
transport, or transport plus RJ. It currently requires spherical boundaries.
`resolved_contact_memory` in the effective config records inherited defaults.
The principal settings are:

| Setting | Default | Purpose |
|---|---:|---|
| `slots` | 16 | Fixed number of independent pair systems |
| `attempts_per_sweep` | 2 | Random-slot updates; zero gives the frozen-bank control |
| `depletant_radius`, `depletant_activity` | Production values | Auxiliary pair bath, independently configurable |
| `radius` | Twice the exclusion bound plus preparation gap and numerical guard | Bounds pair-origin separation |
| `global_probability` | 0.1 | Uniform-ball/Haar redraw versus symmetric local update |
| `local_translation_std_A` | 0.3 | Pair-memory translation scale |
| `local_small_angle_std_degrees` | 3 | Pair-memory rotation scale |
| `proposal_mass` | 0.5 | Total dynamic-dictionary mass allocated to memory slots |
| `proposal_translation_std_A` | 1 | Production Gaussian width around a memory pose |
| `proposal_small_angle_std_degrees` | 3 | Production Gaussian angular width |
| `initial_gap` | 0.25 | Geometry-only radial preparation gap in Å |

For matched free-tetramer controls using an existing geometry-only atlas and
the previous exact starts (the launcher requires NumPy):

```bash
/home/xvg/protein-nucleation/.venv/bin/python tools/run_free_tetramer_campaign.py \
  --out runs/my-contact-memory \
  --reuse-starts runs/free-tetramer-rj-1000 \
  --model runs/free-tetramer-atlases/geometry.json --model-label geometry-only \
  --reversible-jump --memory-json examples/contact-memory-defaults.json \
  --compare-memory --sweeps 1000 --sample-every 10 --workers 8
```

For a second campaign with the same initial bank held fixed, use another output
directory and replace `--compare-memory` with `--freeze-memory`. The analysis
tools distinguish memory-slot, base-atlas and uniform proposals; bank-native
discovery is separate from production registration. All native classifications
are post-hoc and never supply the geometry-only sampler with target poses.

## Validation and initial result

All 38 Rust tests pass; the [validation transcript](contact-memory-validation.txt)
includes both independent analytic tests and runner continuation checks:

- 24,000 independent exact two-sphere starts test the auxiliary pair law,
  including `r^2` translation measure, Haar orientation, local/global updates,
  and nonzero depletion.
- 6,000 independent joint equilibrium starts test the composed physical,
  memory, RJ, GCA and center-shift kernels. The memory activity is 0.8 and the
  production activity 0.4, with depletant radius 0.7 in this sphere test. Physical,
  auxiliary and cross moments remain consistent with their known target.
- Runner tests reconstruct full forward/reverse mixture densities and exactly
  resume physical states, memory, labels, residuals and display coordinates.
- Separate Python and JavaScript tests verify the moving-wall display
  convention and legacy-log reconstruction.

The [matched pilot report](contact-memory-pilot.md) compares memory-off,
frozen-bank and evolving-bank arms across four dispersed starts, each with 12
tetramers for 1,000 sweeps at physical radius 1.5 Å and activity 0.035 Å⁻³.
The frozen and evolving banks start identically. No arm produced a registered
native contact, and no inspected bank pose matched a native arrangement.

The evolving bank accepted 3,564 of 8,000 pair updates. Of 10,547 production
proposals drawn from its charts, 4,906 passed hard geometry, but none was
accepted. Their median log reverse/forward proposal ratio was about -25.4;
the sampled physical depletion factors did not compensate. The frozen-bank
control likewise accepted no memory-chart proposals. This identifies a
forward/reverse weight problem even after hard geometry passes, not simply
failure to construct collision-free poses. Total production CPU increased
from 135 s without memory to 208 s with it. These results establish a correct
auxiliary construction, not an efficiency improvement or a model limitation.
