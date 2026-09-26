//! Exact hard check and fixed-allocation union-overlap score for supplied
//! anchored pair poses. Diagnostic scores only; no physical acceptance.
use anyhow::Result;
use clap::Parser;
use serde_json::json;
use std::{fs, path::PathBuf};
use tetramer_mc::{
    contact_discovery::{DiscoveryConfig, estimate_pair_overlap},
    geometry::{Placed, Shape, SphereTree},
    math::Pose,
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    shape: PathBuf,
    /// JSON list of poses of the moving body relative to the anchored body.
    #[arg(long)]
    poses: PathBuf,
    #[arg(long, default_value_t = 1.4)]
    rd: f64,
    #[arg(long, default_value_t = 8192)]
    points: usize,
    #[arg(long, default_value_t = 2026092602)]
    seed: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let tree = SphereTree::new(serde_json::from_slice::<Shape>(&fs::read(&args.shape)?)?)?;
    let poses: Vec<Pose> = serde_json::from_slice(&fs::read(&args.poses)?)?;
    let opts = DiscoveryConfig::default().gate_options();
    let anchor = Placed::new(Pose {
        position: [0.; 3],
        orientation: [1., 0., 0., 0.],
    });
    let rows = poses
        .iter()
        .map(|&pose| {
            let hard_valid = !tree.overlaps(&Placed::new(pose), &anchor);
            let score = if hard_valid {
                Some(estimate_pair_overlap(&tree, pose, args.rd, args.points, args.seed, opts)?)
            } else {
                None
            };
            Ok(json!({"hard_valid": hard_valid, "score": score}))
        })
        .collect::<Result<Vec<_>>>()?;
    println!("{}", serde_json::to_string(&rows)?);
    Ok(())
}
