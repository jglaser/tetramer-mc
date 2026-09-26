//! Frozen learned rigid-body Monte Carlo with an implicit ideal depletant bath.
//!
//! Proposal density, hard geometry, and physical acceptance are separate layers.
//! No crystallographic registry or fit objective appears in physical acceptance.
pub mod assembly_bias;
pub mod depletion;
pub mod docking;
pub mod geometry;
pub mod gca_overlap_diagnostic;
pub mod initialization;
pub mod latent_region;
pub mod math;
pub mod native_entry;
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
pub mod conditional_axis;
pub mod contact_memory;

pub mod cluster_phase;
pub mod oligomer_proposal;
pub mod rigid_subset;
