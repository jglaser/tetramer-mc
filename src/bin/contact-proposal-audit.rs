//! Fixed-allocation proposal geometry audit, with optional many-neighbor score.
//! No rejection conditioning, native labels, or physical MC updates occur here.
use anyhow::{Result, ensure};
use clap::Parser;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde::Serialize;
use serde_json::json;
use std::{
    fs,
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};
use tetramer_mc::{
    contact_discovery::{DiscoveryConfig, estimate_environment_overlap},
    geometry::{Environment, Placed, Shape, SphereTree},
    math::*,
    proposal::FrozenRelativePoseProposal,
    simulation::hash_bytes,
};

#[derive(Parser, Serialize)]
struct Args {
    #[arg(long)]
    shape: PathBuf,
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    out: PathBuf,
    /// All fixed poses in anchor coordinates. The first must be identity.
    #[arg(long)]
    fixed_poses: Option<PathBuf>,
    #[arg(long, default_value_t = 512)]
    draws_per_component: usize,
    #[arg(long, default_value_t = 1.4)]
    rd: f64,
    /// Zero gives geometry-only audit. Positive values score every valid draw.
    #[arg(long, default_value_t = 0)]
    score_points: usize,
    #[arg(long, default_value_t = 20260928001)]
    seed: u64,
    #[arg(long, default_value_t = false)]
    full_density: bool,
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(!args.out.exists(), "Fresh output required");
    ensure!(
        args.draws_per_component > 0 && args.rd.is_finite() && args.rd >= 0.,
        "Invalid audit allocation"
    );
    let start = Instant::now();
    let shape_bytes = fs::read(&args.shape)?;
    let model_bytes = fs::read(&args.model)?;
    let shape_hash = hash_bytes(&shape_bytes);
    let tree = SphereTree::new(serde_json::from_slice::<Shape>(&shape_bytes)?)?;
    let model = FrozenRelativePoseProposal::from_json_str_open(
        std::str::from_utf8(&model_bytes)?,
        [1e6; 3],
        0.1,
        &shape_hash,
    )?;
    let id = Pose {
        position: [0.; 3],
        orientation: [1., 0., 0., 0.],
    };
    let poses: Vec<Pose> = if let Some(p) = &args.fixed_poses {
        serde_json::from_slice(&fs::read(p)?)?
    } else {
        vec![id]
    };
    ensure!(
        !poses.is_empty() && poses[0] == id,
        "First fixed pose must be exact identity"
    );
    for p in &poses {
        p.validate()?;
    }
    let fixed: Vec<_> = poses.iter().map(|&p| Placed::new(p)).collect();
    for i in 0..fixed.len() {
        for j in 0..i {
            ensure!(
                !tree.overlaps(&fixed[i], &fixed[j]),
                "Fixed scaffold has a hard clash"
            );
        }
    }
    let env = Environment {
        tree: &tree,
        fixed,
        labels: vec![],
        rd: args.rd,
    };
    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(&args.out)?;
    fs::write(args.out.join("shape.json"), &shape_bytes)?;
    fs::write(args.out.join("model.json"), &model_bytes)?;
    fs::write(
        args.out.join("fixed-poses.json"),
        serde_json::to_vec_pretty(&poses)?,
    )?;
    let source = include_bytes!(concat!(env!("OUT_DIR"), "/source-bundle.json"));
    fs::write(args.out.join("source-bundle.json"), source)?;
    fs::write(
        args.out.join("plan.json"),
        serde_json::to_vec_pretty(&json!({"arguments":args,"shape_sha256":shape_hash,
        "model_sha256":hash_bytes(&model_bytes),"components":model.component_count(),
        "source_bundle_sha256":hash_bytes(source),"scope":"Unconditional draws per base component, parity sampled equally where available; all nulls/hard failures retained. Geometry diagnostics, no MC or native input."}))?,
    )?;
    let mut writer = BufWriter::new(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(args.out.join("draws.jsonl"))?,
    );
    let weights = model.component_weights();
    let reciprocal = model.reciprocal_components();
    let mut rows = Vec::new();
    for k in 0..model.component_count() {
        let key = hash_bytes(format!("contact-proposal-audit-v1:{}:{}", args.seed, k).as_bytes());
        let mut rng = StdRng::seed_from_u64(u64::from_str_radix(&key[..16], 16)?);
        let mut pair_valid = 0;
        let mut all_valid = 0;
        let mut nulls = 0;
        for draw in 0..args.draws_per_component {
            let inverse = reciprocal[k] && rng.random::<bool>();
            let pose = model.draw_relative_branch(&mut rng, k, inverse)?;
            let pair = pose.is_some_and(|p| !tree.overlaps(&Placed::new(p), &env.fixed[0]));
            let valid = pose.is_some_and(|p| env.hard_valid(p));
            pair_valid += usize::from(pair);
            all_valid += usize::from(valid);
            nulls += usize::from(pose.is_none());
            let score = if valid && args.score_points > 0 {
                Some(estimate_environment_overlap(
                    &env,
                    pose.unwrap(),
                    args.score_points,
                    rng.random(),
                    DiscoveryConfig::default().gate_options(),
                )?)
            } else {
                None
            };
            let log_q = if args.full_density {
                pose.map(|p| model.relative_log_density(p.position, rotation(p.orientation)))
                    .transpose()?
            } else {
                None
            };
            serde_json::to_writer(
                &mut writer,
                &json!({"component":k,"draw":draw,"inverted":inverse,"pose":pose,
                "pair_hard_valid":pair,"all_neighbors_hard_valid":valid,"score":score,"log_learned_density":log_q}),
            )?;
            writer.write_all(b"\n")?;
        }
        rows.push(
            json!({"component":k,"weight":weights[k],"draws":args.draws_per_component,"nulls":nulls,
            "pair_valid":pair_valid,"all_valid":all_valid}),
        );
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    let weighted = |name: &str| {
        rows.iter()
            .map(|r| {
                r["weight"].as_f64().unwrap() * r[name].as_u64().unwrap() as f64
                    / args.draws_per_component as f64
            })
            .sum::<f64>()
    };
    let summary = json!({"complete":true,"components":rows,"weighted_pair_hard_valid":weighted("pair_valid"),
        "weighted_all_neighbors_hard_valid":weighted("all_valid"),"wall_seconds":start.elapsed().as_secs_f64(),
        "draws_sha256":hash_bytes(&fs::read(args.out.join("draws.jsonl"))?),"physical_updates":0});
    fs::write(
        args.out.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    println!(
        "{}",
        json!({"complete":true,"out":args.out,"weighted_pair_hard_valid":weighted("pair_valid"),"weighted_all_neighbors_hard_valid":weighted("all_valid")})
    );
    Ok(())
}
