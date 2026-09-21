# Native entry regions for independent pose integrals

The original registration score compares a pose with one supplied placement.
Consequently, its `q > 1` region can contain other native crystallographic
contacts. `tools/native_contact_regions.py` adds a separate instantaneous
classifier without changing that original partition or any physical weight.

For each of the two fixed anchors, the classifier retains every matching
directed tetramer motif. Entry requires at most 2 Å maximum member-center
error and 15 degrees body-orientation error, plus at least one prescribed
external monomer bond. That bond must pass the archived entry criteria:
3 Å center error, 20 degrees orientation error, an atomic gap within 2 Å,
and at least one shared native residue pair. Reference residue patches use
the original 1 Å cutoff and the hash-bound repaired atomic coordinates.

These are the earlier trajectory observer's **entry** criteria. Independent
poses have no history, so exit tolerances, contact persistence and hysteresis
are excluded. The caller separately checks the hard shape and physical wall.
No match means no entry under these criteria, not proof of nonnative geometry.

The output includes the union of native entries, matched anchors and motif
identities. Matches to both anchors are reported separately from ideal motif
cycle closure around the three bodies. The former describes simultaneous
geometric contacts; it does not measure energetic cooperativity. The latter
is local registry consistency, not certification of a global crystal lattice.
Overlapping motif labels must not be summed as disjoint statistical regions.

The [frozen definition](../runs/mobile-native-region-definition-20260921/definition.json)
binds the complete classifier, motif catalogue, coordinate records, fixed
scaffold and numerical criteria before production. Nine tests compare entry
results independently with the archived observer, including every exact
motif, reversed ordering, common proper isometries, multiple matches,
residue-patch dependence and the preserved native placement outside D170.
The classification is a native-informed diagnostic, not a geometry-only
discovery mechanism or an additional attraction.
