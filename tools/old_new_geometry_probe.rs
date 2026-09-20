//! Diagnostic crate main: compile the archived HardUnion directly beside the current tree.
mod geometry;
use anyhow::{Result, ensure};
use hoomd_microstate::property::OrientedPoint;
use hoomd_vector::{Cartesian, Quaternion, Rotate, Rotation, Versor};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde::Deserialize;
use serde_json::json;
use std::{fs, path::PathBuf};
use tetramer_mc::{
    depletion::GateOptions,
    geometry::{Environment, Placed, Shape, SphereTree},
    math::Pose,
    overlap_weight::{self, OverlapEnvelope},
};
type V = Cartesian<3>;
type P = OrientedPoint<V, Versor>;

#[derive(Deserialize)]
struct Case {
    id: String,
    pose: Pose,
    fixed: Vec<Pose>,
    box_lengths: [f64; 3],
    bounds_lo: [f64; 3],
    bounds_hi: [f64; 3],
    seed: u64,
}
#[derive(Deserialize)]
struct Input {
    shape: PathBuf,
    rd: f64,
    activity: f64,
    common_points: usize,
    cloud_repeats: usize,
    lambda_ratio: f64,
    gate_options: GateOptions,
    cases: Vec<Case>,
}
fn old_pose(p: Pose) -> Result<P> {
    Ok(P { position: V::from(p.position), orientation: Quaternion::from(p.orientation).to_versor()? })
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(args.len() == 3, "input.json output.json");
    let input: Input = serde_json::from_str(&fs::read_to_string(&args[1])?)?;
    let raw = fs::read(&input.shape)?;
    let old_shape: geometry::Shape = serde_json::from_slice(&raw)?;
    let old_tree = geometry::HardUnion::new(&old_shape.atoms);
    let new_tree = SphereTree::new(serde_json::from_slice::<Shape>(&raw)?)?;
    let mut results = Vec::new();
    for case in input.cases {
        let old_moving = old_pose(case.pose)?;
        let old_fixed: Vec<_> = case.fixed.iter().copied().map(old_pose).collect::<Result<_>>()?;
        let new_moving = Placed::new(case.pose);
        let env = Environment { tree: &new_tree, fixed: case.fixed.iter().copied().map(Placed::new).collect(),
            labels: case.fixed.iter().enumerate().map(|(i, _)| (i,[0;3])).collect(), rd: input.rd };
        let old_hard = !old_fixed.iter().any(|f| old_tree.overlaps_handed(&old_moving,1,f,1));
        let new_hard = env.hard_valid(case.pose);
        let volume: f64 = (0..3).map(|k| case.bounds_hi[k]-case.bounds_lo[k]).product();
        ensure!(volume > 0., "empty intersection AABB");
        let mut rng = StdRng::seed_from_u64(case.seed);
        let (mut old_count,mut new_count,mut mismatches,mut member_mismatches) = (0usize,0usize,0usize,0usize);
        for _ in 0..input.common_points {
            let p: [f64;3] = std::array::from_fn(|k| case.bounds_lo[k]+rng.random::<f64>()*(case.bounds_hi[k]-case.bounds_lo[k]));
            let own_old = old_tree.excludes_sphere_center(old_moving.orientation.inverted().rotate(&(V::from(p)-old_moving.position)),input.rd);
            let own_new = new_tree.contains(new_moving.unapply(p),input.rd);
            member_mismatches += usize::from(own_old != own_new);
            let fixed_old = old_fixed.iter().any(|f| {
                let mut delta = V::from(p)-f.position;
                for k in 0..3 { delta[k] -= case.box_lengths[k]*(delta[k]/case.box_lengths[k]).round(); }
                old_tree.excludes_sphere_center(f.orientation.inverted().rotate(&delta),input.rd)
            });
            let fixed_new = env.contains(p);
            member_mismatches += usize::from(fixed_old != fixed_new);
            let a = own_old && fixed_old;
            let b = own_new && fixed_new;
            old_count += usize::from(a); new_count += usize::from(b); mismatches += usize::from(a != b);
        }
        let estimate = volume*new_count as f64/input.common_points as f64;
        let fraction = new_count as f64/input.common_points as f64;
        let se = volume*(fraction*(1.-fraction)/input.common_points as f64).sqrt();
        let envelope = OverlapEnvelope::build(&env,case.pose,input.gate_options)?;
        let lambda = input.lambda_ratio*input.activity;
        let mut cloud_counts = Vec::new();
        for _ in 0..input.cloud_repeats {
            let cloud = overlap_weight::sample_with_envelope(&mut rng,&env,case.pose,lambda,input.activity,&envelope)?;
            cloud_counts.push(cloud.overlap_points);
        }
        let sum: u64 = cloud_counts.iter().sum();
        let exposure = lambda*input.cloud_repeats as f64;
        let cloud_estimate = envelope.lower_volume+sum as f64/exposure;
        let cloud_se = (sum as f64).sqrt()/exposure;
        let agreement_z = (cloud_estimate-estimate)/(se*se+cloud_se*cloud_se).sqrt();
        let result = json!({"id":case.id,"old_hard_valid":old_hard,"new_hard_valid":new_hard,
            "common_points":input.common_points,"integration_volume_A3":volume,
            "old_overlap_count":old_count,"new_overlap_count":new_count,
            "overlap_predicate_mismatches":mismatches,"individual_union_predicate_mismatches":member_mismatches,
            "common_C_A3":estimate,"common_C_standard_error_A3":se,"common_zC_standard_error_kBT":input.activity*se,
            "current_envelope_lower_A3":envelope.lower_volume,"current_envelope_upper_A3":envelope.upper_volume(),
            "current_cloud_C_A3":cloud_estimate,"current_cloud_C_standard_error_A3":cloud_se,
            "current_cloud_counts":cloud_counts,"current_cloud_lambda":lambda,
            "current_cloud_vs_common_standard_errors":agreement_z});
        println!("{}",result);
        results.push(result);
    }
    fs::write(&args[2],serde_json::to_string_pretty(&json!({"results":results,
        "scope":"Archived HardUnion source and current SphereTree on identical independent world-coordinate points; current Poisson overlap-envelope count checked against their common C estimate."}))?)?;
    Ok(())
}
