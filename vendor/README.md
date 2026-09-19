# Pinned third-party source

`hoomd-rs/` is an unmodified source snapshot from
https://github.com/glotzerlab/hoomd-rs at commit
`04d5c8fe66b9b9155a01916fd9158854431ca946` (2026-09-08).
Its upstream BSD-3-Clause license is preserved in `hoomd-rs/LICENSE`.

The current sampler links `hoomd-gsd` and its transitive dependencies. The
remaining source is retained to keep upstream workspace paths intact and allow
future adapters without a mutable remote checkout. `hoomd-bevy` is present in
this snapshot but is **not a dependency of this application** and was not built
or tested for this port. The lightweight offline viewer has no Bevy dependency.

The root `Cargo.lock` pins registry dependencies. `cargo build --locked --offline`
works when these dependencies are already in Cargo's cache; an initial build on
a fresh machine needs registry access.
