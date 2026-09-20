//! Thin fixed-pose driver for the existing absolute overlap-weight API.
//! No pose proposals, importance denominators, or physical kernel changes.
use anyhow::{Result, ensure};
use rand::{SeedableRng, rngs::StdRng};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::Path};
use tetramer_mc::{
    depletion::GateOptions,
    geometry::{Environment, Placed, Shape, SphereTree},
    math::Pose,
    native_region::NativeMetric,
    overlap_weight::{self, OverlapEnvelope},
    simulation::cpu_seconds,
};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(
        args.len() == 3,
        "usage: fixed_pose_clouds INPUT_JSON OUTPUT_JSONL"
    );
    ensure!(!Path::new(&args[2]).exists(), "output already exists");
    let input: Value = serde_json::from_slice(&fs::read(&args[1])?)?;
    let cfg: Value = serde_json::from_slice(&fs::read(input["config"].as_str().unwrap())?)?;
    let shape: Shape = serde_json::from_slice(&fs::read(cfg["shape"].as_str().unwrap())?)?;
    let tree = SphereTree::new(shape)?;
    let fixed: Vec<Pose> = serde_json::from_value(cfg["fixed_poses"].clone())?;
    let options: GateOptions = serde_json::from_value(cfg["endpoint_gate"].clone())?;
    options.validate()?;
    let z = cfg["reservoir_density"].as_f64().unwrap();
    let ratio = input["lambda_ratio"].as_f64().unwrap();
    let lambda = z * ratio;
    let metric: NativeMetric = serde_json::from_value(cfg["metadata"].clone())?;
    let capture: [f64; 3] = serde_json::from_value(cfg["capture_center"].clone())?;
    let radius = cfg["capture_radius"].as_f64().unwrap();
    let env = Environment {
        tree: &tree,
        fixed: fixed.iter().copied().map(Placed::new).collect(),
        labels: fixed.iter().enumerate().map(|(j, _)| (j, [0; 3])).collect(),
        rd: cfg["depletant_radius"].as_f64().unwrap(),
    };
    ensure!(
        z == 0.035 && ratio == 64. && env.rd == 1.5 && fixed.len() == 2,
        "fixed physical protocol changed"
    );
    for i in 0..env.fixed.len() {
        for j in 0..i {
            ensure!(
                !tree.overlaps(&env.fixed[i], &env.fixed[j]),
                "fixed neighbors clash"
            );
        }
    }
    let mut output = fs::File::create_new(&args[2])?;
    for case in input["cases"].as_array().unwrap() {
        let pose: Pose = serde_json::from_value(case["pose"].clone())?;
        pose.validate()?;
        let q = metric.q(pose);
        let distance = pose
            .position
            .iter()
            .zip(capture)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        ensure!(
            q >= 2. && q < 5. && distance <= radius && env.hard_valid(pose),
            "invalid fixed pose"
        );
        let count = case["clouds"].as_u64().unwrap();
        let seed = case["seed"].as_u64().unwrap();
        ensure!(count == 256, "fixed cloud budget changed");
        let started = cpu_seconds();
        let envelope = OverlapEnvelope::build(&env, pose, options)?;
        let envelope_cpu = cpu_seconds() - started;
        ensure!(
            (envelope.lower_volume - case["original_lower_volume"].as_f64().unwrap()).abs() < 1e-8,
            "fixed-pose lower envelope differs from original"
        );
        ensure!(
            (envelope.upper_volume() - case["original_upper_volume"].as_f64().unwrap()).abs()
                < 1e-8,
            "fixed-pose upper envelope differs from original"
        );
        for cloud in 0..count {
            let mut hash = Sha256::new();
            hash.update(b"fixed-pose-overlap-weight-control-v1");
            hash.update(seed.to_le_bytes());
            hash.update(cloud.to_le_bytes());
            let mut rng = StdRng::from_seed(hash.finalize().into());
            let tick = cpu_seconds();
            let weight =
                overlap_weight::sample_with_envelope(&mut rng, &env, pose, lambda, z, &envelope)?;
            let row = json!({"case":case["id"], "seed":seed, "cloud":cloud, "pose":pose,
                "original_q":q,"activity":z,"lambda":lambda,"lambda_ratio":ratio,
                "weight":weight,"sampling_CPU_seconds":cpu_seconds()-tick,
                "envelope_CPU_seconds":if cloud == 0 {envelope_cpu} else {0.}});
            writeln!(output, "{}", serde_json::to_string(&row)?)?;
        }
    }
    output.sync_all()?;
    Ok(())
}
