//! Frozen learned rigid-body Monte Carlo with an implicit ideal depletant bath.
//!
//! Proposal density, hard geometry, and physical acceptance are separate layers.
//! No crystallographic registry or fit objective appears in physical acceptance.
pub mod depletion;
pub mod docking;
pub mod geometry;
pub mod latent_region;
pub mod math;
pub mod native_region;
pub mod normalizer;
pub mod overlap_weight;
pub mod proposal;
pub mod rj;
pub mod simulation;
pub mod spherical;
pub mod trajectory;

pub mod atlas_mask;
pub mod atlas_transport;
pub mod auxiliary;
pub mod basin_involution;
pub mod conditional;
pub mod contact_memory;
