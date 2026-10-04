//! Frozen learned rigid-body Monte Carlo with an implicit ideal depletant bath.
//!
//! Proposal density, hard geometry, and physical acceptance are separate layers.
//! No crystallographic registry or fit objective appears in physical acceptance.
pub mod assembly_bias;
pub mod auxiliary_overlap_threshold;
pub mod capped_dimer;
pub mod cayley_axis_geometry;
pub mod circle_geometry;
pub mod contact_distances;
pub mod defensive_dimer_proposal;
pub mod depletion;
pub mod depletion_surrogate;
pub mod dimer_tree_proposal;
pub mod docking;
pub mod factorized_dimer;
pub mod flexible_subset;
pub mod geometry;
pub mod gca_overlap_diagnostic;
pub mod initialization;
pub mod latent_region;
pub mod line_geometry;
pub mod math;
pub mod native_entry;
pub mod native_region;
pub mod normalizer;
pub mod overlap_weight;
pub mod proposal;
pub mod rj;
pub mod simulation;
pub mod singleton_path;
pub mod spherical;
pub mod trajectory;

pub mod atlas_mask;
pub mod atlas_transport;
pub mod auxiliary;
pub mod basin_involution;
pub mod conditional;
pub mod conditional_axis;
pub mod contact_discovery;
pub mod contact_memory;

pub mod cluster_phase;
pub mod oligomer_proposal;
pub mod rigid_subset;
pub mod rigid_surrogate_chain;

pub mod bounded_singleton_path;
pub mod evolving_dimer;
pub mod two_neighbor_singleton;
