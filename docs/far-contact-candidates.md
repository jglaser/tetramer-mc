# Historical AB far-contact candidates

`tools/prepare_far_contact_candidates.py` freezes eight largest original
`q>=5` importance contributions per population in the three historical widths.
The resulting 96 poses are in
`runs/ab-far-contact-candidates-20260920/analysis.json`. It retains the exact
original rows, full unconditional budgets, source hashes, full proposal density,
and independent atom-gap checks. No physical samples or new Poisson points are
generated. These weights select geometry; they are not equilibrium mixture
allocations.

The two largest width-four observations are indices **88** (population
`scale-4-r03`, draw 15662) and **72** (`scale-4-r01`, draw 2401). Both came from
component 118, the existing AB-compatible competitor fitted to 30 endpoints
from one old SMC ancestry family. Their rigid-member RMS separation is
**0.320476 Å**. Their radii in the old fitted chart are 3.84213 and 3.69746:
both lie outside its frozen radius-three reference. The second is at radius
0.320477 Å in a geometric chart centered on the first, so a radius-0.5 local
reference contains both historical peaks. Together they supply 93.48% of the
observed width-four far estimate; this is evidence of unresolved importance
weights, not evidence that two poses carry that equilibrium probability.

| Historical width | Unconditional draws | Far log Q | Row RSE | Population RSE |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 32,768 | 15.295009 | 28.28% | 9.11% |
| 2 | 65,536 | 16.556066 | 22.97% | 24.46% |
| 4 | 65,536 | 18.937010 | 66.39% | 52.69% |

These figures use `q>=5`; the width-four all-other result of 26.257574 is
dominated by shoulder poses and is a different observable. Historical
`q>=2` and `q>=5` estimates happen to be close because those runs found little
intermediate mass. The old R3 uniform reference, log Q 14.437005 with row RSE
14.72% and population RSE 17.27%, estimates only that local intersection.

The panel also retains distinct geometries: width-one component 97 at draw
6554 of `scale-1-r01` is approximately 29 Å member RMS from the new peak and
has only A in depletion contact; width-four uniform draw 10442 of `scale-4-r00`
is 7.65 Å away and contacts both neighbors; width-four component-six draw 7224
of `scale-4-r01` is 26.13 Å away and also contacts both. The full 96-by-96
member-RMS matrix is stored to support subsequent geometric coverage choices.
Large distances in the old anisotropic Cayley chart alone do not imply equally
large physical distances.

The old A-only deep contact that clashes with B is excluded. Neither these
selected rows nor a local reference certify coverage of the entire far region.
The frozen 119-component model SHA-256 is
`5226ea364a07b30be07b48e4d76f4f8b0daa9b5965bd5a93d09edc16fb4b95a0`;
the old R3 region hash is
`e4b8cf45567ccc8a8106e4853fc5cdfab3b56f21dcc9abba86145acb3d223a98`.
Four focused tests cover far-only selection, tie ordering, nonfinite weights,
and frame-invariant member distances.
