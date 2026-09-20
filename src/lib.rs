//! Frozen learned rigid-body Monte Carlo with an implicit ideal depletant bath.
//!
//! Proposal density, hard geometry, and physical acceptance are separate layers.
//! No crystallographic registry or fit objective appears in physical acceptance.
pub mod depletion;
pub mod geometry;
pub mod math;
pub mod proposal;
pub mod rj;
pub mod simulation;
pub mod spherical;
pub mod trajectory;

pub mod auxiliary;
pub mod contact_memory;
