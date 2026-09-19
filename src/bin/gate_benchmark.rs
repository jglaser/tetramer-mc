//! Fixed-endpoint benchmark: isolate acceptance cost from trajectory differences.
use anyhow::{Result, ensure};
use clap::Parser;
use rand::{SeedableRng, rngs::StdRng};
use serde::Deserialize;
use serde_json::json;
use std::{fs, path::PathBuf};
use tetramer_mc::{
    depletion::{self, GateOptions},
    geometry::{Environment, Shape, SphereTree},
    math::Pose,
    simulation::{cpu_seconds, save},
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 3)]
    repetitions: usize,
}
#[derive(Deserialize)]
struct Cases {
    shape: PathBuf,
    box_lengths: [f64; 3],
    depletant_radius: f64,
    activity: f64,
    lambda_ratio: f64,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    label: String,
    poses: Vec<Pose>,
    moving_index: usize,
    new_pose: Pose,
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.repetitions > 0, "positive repetitions required");
    let data: Cases = serde_json::from_slice(&fs::read(&args.input)?)?;
    let tree = SphereTree::new(serde_json::from_slice::<Shape>(&fs::read(&data.shape)?)?)?;
    let mut records = Vec::new();
    for (number, case) in data.cases.iter().enumerate() {
        let old = case.poses[case.moving_index];
        let start = cpu_seconds();
        let env = Environment::new(
            &tree,
            &case.poses,
            case.moving_index,
            old,
            case.new_pose,
            data.box_lengths,
            data.depletant_radius,
        )?;
        ensure!(
            env.hard_valid(old) && env.hard_valid(case.new_pose),
            "hard-invalid benchmark {}",
            case.label
        );
        let setup_cpu = cpu_seconds() - start;
        let mut samples = Vec::new();
        let start = cpu_seconds();
        let mut rng = StdRng::seed_from_u64(313001 + number as u64);
        for _ in 0..args.repetitions {
            samples.push(depletion::sample(
                &mut rng,
                &env,
                old,
                case.new_pose,
                data.lambda_ratio * data.activity,
                data.activity,
                GateOptions::default(),
            )?);
        }
        records.push(json!({"label":case.label,"setup_cpu_seconds":setup_cpu,"gate_cpu_seconds":cpu_seconds()-start,"repetitions":args.repetitions,"samples":samples}));
    }
    save(
        &args.out,
        &json!({"implementation":"Rust body-frame sphere BVH","cases":records,"scope":"Identical fixed endpoints; independent RNG, conservative plans may differ"}),
    )?;
    Ok(())
}
