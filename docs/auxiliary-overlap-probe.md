# Protein overlap-threshold probe

The [auxiliary guide](auxiliary-overlap-threshold.md) preserves substantially
more internal overlap in feasible native-informed proposals, while exposing
poor overlap coverage in the older blind memory atlas. This is a passive
proposal result. Physical acceptance, contact ESS and assembly are not measured
by these counts.

All **1,536 guided outer attempts** completed, retaining 1,212 nulls and 324
candidates. Independent reconstruction passed **1,595,400 checks**, including
every raw cloud coordinate, root-retention predicate, stage draw and stopping
decision, guidance count in four frames, complete mixture density and separate
auxiliary correction. The prior 768 factorized baseline attempts were reused
without new baseline proposals. Common raw proposal prefixes and nested
threshold draws were checked across arms.

Each row below contains 256 outer attempts over the same eight fixed contexts.
Overlap retention is the candidate/source **guidance-count ratio**, conditional
on obtaining a candidate; it is not an exact overlap-volume ratio or an
unconditional efficiency measure. The same freshly drawn cloud also measures
the cached baseline candidate.

| Atlas | Baseline candidates | m=1 candidates | m=4 candidates | Median retention: baseline / m=1 / m=4 |
|---|---:|---:|---:|---|
| Blind memory, all 64 slots | 174 | 16 | 0 | 0.028 / 0.083 / unavailable |
| Blind FFT, 512 slots | 159 | 52 | 10 | 0.207 / 0.287 / 0.621 |
| Native-informed, 178 components | 178 | 142 | 104 | 0.029 / 0.833 / 0.987 |

The stronger threshold cannot manufacture high-overlap poses absent from the
base proposal. Under m=4, the memory atlas exhausted all 256 attempts. In the
native-informed arm it found 104 candidates, including 35 with external
exclusion contacts. Neither an external contact nor a retained internal overlap
count establishes native registry. Native classification did not filter any
proposal or source slot.

The proposal correction still matters. For native-informed m=4 candidates,
the median complete proposal log correction is −29.32, despite near-unity
median overlap-count retention. Only the full many-body physical gate can
determine how favorable overlap changes compensate this term. Candidate
throughput therefore supplies no reason to promote this kernel into assembly
production yet.

## Fixed experiment and cost

The source is the archived sweep-7400, N=264 growth configuration at **1.4 Å,
0.0275 Å⁻³ and 500 μM**. These diagnostic conditions remain separate from the
original **1.5 Å, 0.035 Å⁻³, approximately 106.8 μM** decision conditions.
The same selected root/child labels and spectator anchor define each fixed
context. Root and internal caps are 32, joint cap is one, root sampled first;
each edge retains the 50% uniform defensive component.

For each of 768 independent context/atlas/slot combinations, 16,384 raw uniform
points in the inflated root's atomic AABB are thinned by root membership only.
All 12,582,912 raw triples are archived. The retained cloud has a fixed-size
binomial-thinning law, **not a Poisson bath**; its arbitrary state-independent
law is allowed by the auxiliary argument. One cloud is shared between m=1 and
m=4, with independent proposal and threshold RNG streams. No cloud is selected
or redrawn based on the old overlap count. Points move with the root in world
coordinates while remaining fixed in its body frame.

The sampler used 12.619 process CPU seconds and 45,931 raw edge draws. Charging
the full cloud-construction cost separately to each arm gives 13.608 seconds of
standalone proposal work across both guided arms; the actual execution shared
cloud construction. Independent auditing took 231.019 CPU seconds. Baseline
timings are archived historical measurements, not simultaneous timing
replicates. All nulls belong in subsequent physical acceptance denominators.

## Evidence and next physical comparison

The [completed review](../results/auxiliary-overlap-probe-20261003/completed-review.json)
authenticates 513 frozen files and 17 external inputs. The
[protocol](../results/auxiliary-overlap-probe-20261003/protocol.json),
[independent analysis](../results/auxiliary-overlap-probe-20261003/analysis.json)
and [execution ledger](../results/auxiliary-overlap-probe-20261003/execution/attempts.jsonl)
retain every attempt. SHA256 values:

- Analysis: `0f8a299d5efd945466fecdc54530b886c41006a94f693d42794e4d8f690b06c3`.
- Completed review: `d8555ed4a062784f1842cc1712b8673ac006592a336a255951e5bc3dd2cc563e`.

The separately frozen [physical replay](auxiliary-overlap-physical.md) is now
complete: all 1,536 guided outers, zero accepts. It adds the auxiliary correction
exactly once, reuses the cached baseline physical results, and counts every
null and rejection. Improved overlap retention does not yet give useful
physical acceptance; no assembly-kernel promotion follows from this pilot.
