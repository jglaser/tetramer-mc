# A complete reference for the distant AB region

The original distant region is **q ≥ 5**, with both fixed tetramers A and B,
depletant radius 1.5 Å, activity 0.035 Å⁻³, and an 18 Å capture sphere. The
proposal does not redefine that region. The physical integral uses center
Lebesgue measure and normalized proper SO(3) Haar measure:

\[
Q_{\rm far}=\int H(x;A,B)I_{\rm capture}I_{q\ge5}
 \exp\{z|E(x)\cap[E(A)\cup E(B)]|\}\,d^3t\,dH(R).
\]

This preparation uses the existing frozen `native-region-normalizer`
executable, SHA256
`3a6a2dbba0ec5234c36cf66d7c40877336f22358a6a1ea91ea8366b344ee027b`.
Its archived proposal, geometry, normalizer, and depletion source files
match the current repository bytes. No Rust kernel is changed.

## A finite q bound covers the entire capture sphere

The reference and capture centers are both zero. For each rigid-member
center m, a captured pose obeys

\[
|t+Rm-m|\le18+2|m|,
\qquad\theta(R)\le\pi.
\]

The maximum member radius is 27.21763845355 Å. The original scales are
2 Å and 15°, so

\[
q\le\max[(18+2r_{\max})/2,180/15]
=36.21763845355<37.
\]

Consequently **5 ≤ q < 37 is exactly the full captured q ≥ 5 region**.
The strict upper cutoff has a large geometric margin. The proof tool
bounds norms of the binary64 input coordinates using upward-rounded decimal
products and sums, an outward square-root step, and an outward conversion
to binary64. It adds explicit slack
`4096*epsilon*(1+R+2*r_max)/delta` before checking q < 37. This is an
analytic support proof with a floating-point guard; the physical collision
kernel is not formal interval arithmetic.

A deterministic half-turn about an axis perpendicular to an extreme
member, with center displacement −18 m/|m|, saturates the displacement bound.
The recorded witness checks rotation, original q, and the capture edge.
It is a kinematic witness; its hard validity is not asserted.

## A flat proposal using the existing executable

At q-max 37, the complete product cover is a radius-74 Å center ball and
full Haar orientation. This is normalized but only 1.439% of its centers
lie in the capture sphere. The control instead mixes 99% uniform
cube/Haar draws with 1% complete-cover draws. On the capture sphere,

\[
g(x)=\frac{0.99}{36^3}+\frac{0.01}{(4\pi/3)74^3}
=2.12250271716\times10^{-5}\ \mathrm{Å}^{-3}.
\]

Both density terms enter every importance denominator. The guide's
`uniform_probability=1` gives its Gaussian exactly zero sampling and
density weight. A simple, untrained Gaussian file satisfies the existing
schema and has no physical effect. The Python density audit explicitly
handles epsilon=1, including half-turn chart seams and cube boundaries.

The mixture sends 51.851% of centers into capture before hard and q checks.
A dedicated capture-ball/Haar proposal could remove that remaining geometry
waste, but would require a kernel change. No performance gain for protein
contact sampling follows from this capture fraction alone.

The estimator remains the unconditional fixed-N mean of
`H * capture * I(5<=q<37) * mean(W_clouds) / g`. Invalid draws are retained
as zeros. The positive Poisson cloud identity uses the full union of fixed
exclusion regions, so it includes many-body depletion. The two clouds are
averaged before taking logarithms.

## Analytic controls before protein sampling

Two four-population controls each use 16,384 unconditional draws, fixed
seeds, lambda/activity ratio 64, and two clouds. They share the exact
5 ≤ q < 37 mask and the 99% cube/1% cover mixture. The toy has one fixed
sphere of core radius 0.5 Å, depletant radius 0.5 Å, capture radius 3 Å,
one centered metric member, displacement scale 0.25 Å, and angular scale
15°. Thus q=max(4r,theta/15°); the far condition is r ≥ 1.25 Å **or**
theta ≥ 75°. This exercises radial and angular membership together.

Hard validity requires r ≥ 1 Å. The exclusion overlap is
`C(r)=pi*(4+r)*(2-r)^2/12` for 1 ≤ r < 2, and zero for r ≥ 2. Writing
`h=(75°-sin(75°))/pi` in radians, the exact one-dimensional reference is

\[
Q(z)=(1-h)\int_1^{1.25}4\pi r^2e^{zC(r)}dr
+\int_{1.25}^{2}4\pi r^2e^{zC(r)}dr+V_3-V_2.
\]

The controls use z=0 and z=0.4 Å⁻³. Their physical parameters deliberately
differ from the protein; they test the estimator against tractable sphere
geometry. At zero activity the answer is also the closed expression
`V3-V1-h*(V1.25-V1)`.

The frozen acceptance rule requires complete input/hash/density/Poisson
audits, analytic sphere geometry and overlap-envelope checks, and agreement
of both means with their exact references within six observed row standard
errors. This tolerance is an implementation regression screen, not a
claim of precise protein convergence. Independent population errors are
reported separately. No failed or unusual population is replaced.

## Prepared artifacts and commands

Preparation: `runs/ab-far-capture-control-20260920`. `protocol.json` and
`freeze.json` precede every physical control. `capture-proof.json` retains
the bound and witness; each sphere campaign retains all rows and a complete
Python density audit. `AB-flat-prepared/manifest.json` holds the four
protein commands at 65,536 draws each, with seeds `105201010+1009*i`.
Preparation and the sphere-control action do not launch the protein jobs.
The separate `run-protein` action requires completed matching control audits
and preserves the four predeclared budgets and seeds.

```bash
python tools/prepare_far_capture_control.py prepare --out runs/ab-far-capture-control-20260920
python tools/prepare_far_capture_control.py run-controls --out runs/ab-far-capture-control-20260920 --workers 8
python tools/prepare_far_capture_control.py audit --out runs/ab-far-capture-control-20260920
python tools/prepare_far_capture_control.py run-protein --out runs/ab-far-capture-control-20260920 --workers 4
python tools/prepare_far_capture_control.py audit-protein --out runs/ab-far-capture-control-20260920
```

An optional later partition is [5,9) and [9,37). The former's complete
cover is a radius-18 Å ball with angular cap 104.224829° and volume
6607.480849 Å³, so it has no capture rejection. The latter still requires
complete broad support. Each cover must be recomputed from its own q-max:
scaling the q-max-37 cover by 9/37 also scales its angle and does not
reproduce the complete q ≤ 9 cover. This split is documented, not launched.

## Completed controls and AB reference

All eight sphere jobs completed, followed by the independent full-density
audits and analytic geometry checks on all 131,072 unconditional rows.
The maximum log-density discrepancy was 1.78×10⁻¹⁵; no analytic overlap
volume lay outside its recorded envelope. At epsilon=1, every guide draw
used its uniform branch. Each hard, capture, and q rejection category was
observed and retained.

| Sphere activity (Å⁻³) | Estimated Q ± row SE (Å³) | Exact Q (Å³) | Difference / row SE | Population relative SE |
| --- | ---: | ---: | ---: | ---: |
| 0 | 108.5268 ± 0.4259 | 108.4726 | +0.127 | 0.117% |
| 0.4 | 112.3038 ± 0.4473 | 112.5642 | −0.582 | 0.384% |

The control report is `analysis.json`. Its `protein_launched=false` field
records the state when those controls passed. Following that validation,
the four predeclared AB jobs were authorized and completed without changing
their budgets or seeds. Their result is recorded separately in
`AB-analysis.json`; all populations and every original row remain intact.

| Full AB original q ≥ 5 | Result |
| --- | ---: |
| Unconditional draws / valid poses | 262,144 / 5,096 |
| Estimated log Q | 12.814664 |
| Observed row / population relative SE | 59.14% / 56.95% |
| Importance weight ESS | 2.86 |
| Largest single contribution | 57.71% |
| Hard far volume | 915.886 Å³, 1.387% row relative SE |
| Physical sampler CPU | 45.35 seconds |

The remaining draws comprise 3,118 q rejections, 122,941 capture
rejections, and 130,989 hard rejections. These categories follow the
kernel's q-then-capture-then-hard test order. Its full density, original q,
input hashes, and Poisson factors were reconstructed on every row by the
archived Python auditor. Eight top-weight poses also passed independent
atom-union hard checks; their smallest gap was 0.0178525 Å.

The largest contribution lies at q=32.2662, with atomic gaps 0.0637061 Å
and 0.2100341 Å to A and B. The distant region therefore includes actual
valid contact samples far beyond q=12, despite relative rotation itself
being bounded by 180°. The member-displacement term makes the larger
complete q bound necessary.

This is a complete-support baseline with poor contact-weight precision.
The reasonably concentrated hard-volume estimate validates broad geometric
sampling; the physical weight is dominated by a few contact poses. The
reported error cannot bound unseen narrow contacts or establish agreement
with finite-region contact references. Those references remain separate;
there is no pooling, refitting, or replacement of the original proposal
density. This calculation does not establish equilibrium assembly or a
model limitation.
