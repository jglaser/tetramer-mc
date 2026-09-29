//! Build native-blind contact data independently of every production trajectory.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use serde::Serialize;
use serde_json::json;
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};
use tetramer_mc::{
    contact_discovery::{DiscoveryConfig, DiscoveryResult, discover_slot_from},
    geometry::{Shape, SphereTree},
    math::Pose,
    simulation::hash_bytes,
};

#[derive(Parser, Debug, Serialize)]
#[command(
    about = "Native-blind offline pair-overlap search, independent validation, and exact pair refinement"
)]
struct Args {
    #[arg(long)]
    shape: PathBuf,
    /// Fresh output directory; existing output paths are never overwritten.
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 8)]
    starts: usize,
    #[arg(long, default_value_t = 128)]
    search_steps: usize,
    #[arg(long, default_value_t = 2048)]
    search_points: usize,
    #[arg(long, default_value_t = 8192)]
    validation_points: usize,
    #[arg(long, default_value_t = 256)]
    refine_steps: usize,
    #[arg(long, default_value_t = 64)]
    burn: usize,
    #[arg(long, default_value_t = 4)]
    save_every: usize,
    #[arg(long, default_value_t = 1.4)]
    rd: f64,
    #[arg(long, default_value_t = 0.0275)]
    activity: f64,
    #[arg(long, default_value_t = 2026092601)]
    seed: u64,
    #[arg(long, default_value_t = 0.25)]
    gap: f64,
    #[arg(long, default_value_t = 0.2)]
    refine_translation_std_a: f64,
    #[arg(long, default_value_t = 1.)]
    refine_angle_std_degrees: f64,
    /// Tune the common local step multiplier during discarded burn only.
    #[arg(long, default_value_t = false)]
    refine_adapt: bool,
    #[arg(long, default_value_t = 0.3)]
    refine_adapt_target: f64,
    #[arg(long, default_value_t = 16)]
    refine_adapt_window: usize,
    #[arg(long, default_value_t = 2.)]
    refine_adapt_gain: f64,
    #[arg(long, default_value_t = 1e-4)]
    refine_adapt_min_scale: f64,
    #[arg(long, default_value_t = 10.)]
    refine_adapt_max_scale: f64,
    /// Rotate about a frozen body material point from the optimized contact.
    #[arg(long, default_value_t = false)]
    refine_contact_pivot: bool,
    #[arg(long, default_value_t = 64.)]
    lambda_ratio: f64,
    #[arg(long, default_value_t = 2047)]
    envelope_max_cells: usize,
    /// Optional JSON list of shape-only start poses, one per slot (`--starts`
    /// must equal its length). Such slots use the local, non-projecting search.
    #[arg(long)]
    initial_poses: Option<PathBuf>,
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| {
            format!(
                "create {} without overwriting existing data",
                path.display()
            )
        })?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    write_new(path, &bytes)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let config = DiscoveryConfig {
        starts: args.starts,
        search_steps: args.search_steps,
        search_points: args.search_points,
        validation_points: args.validation_points,
        refine_steps: args.refine_steps,
        burn: args.burn,
        save_every: args.save_every,
        rd: args.rd,
        activity: args.activity,
        seed: args.seed,
        gap: args.gap,
        refine_translation_std_a: args.refine_translation_std_a,
        refine_angle_std_degrees: args.refine_angle_std_degrees,
        refine_adapt: args.refine_adapt,
        refine_adapt_target: args.refine_adapt_target,
        refine_adapt_window: args.refine_adapt_window,
        refine_adapt_gain: args.refine_adapt_gain,
        refine_adapt_min_scale: args.refine_adapt_min_scale,
        refine_adapt_max_scale: args.refine_adapt_max_scale,
        refine_contact_pivot: args.refine_contact_pivot,
        lambda_ratio: args.lambda_ratio,
        envelope_max_cells: args.envelope_max_cells,
    };
    config.validate()?;
    ensure!(
        !args.out.exists(),
        "output path already exists; use a fresh directory"
    );
    let started = Instant::now();
    let shape_bytes = fs::read(&args.shape)?;
    let shape_sha256 = hash_bytes(&shape_bytes);
    let tree = SphereTree::new(serde_json::from_slice::<Shape>(&shape_bytes)?)?;
    config.memory_config(&tree)?;
    let (initial_poses, initial_poses_sha256) = match &args.initial_poses {
        Some(path) => {
            let bytes = fs::read(path)?;
            let poses: Vec<Pose> = serde_json::from_slice(&bytes)?;
            ensure!(
                poses.len() == config.starts,
                "--starts must equal the number of supplied initial poses"
            );
            (Some(poses), Some(hash_bytes(&bytes)))
        }
        None => (None, None),
    };
    let source_bundle = include_bytes!(concat!(env!("OUT_DIR"), "/source-bundle.json"));
    if let Some(parent) = args.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    // create_dir, rather than create_dir_all on the leaf, closes the overwrite race.
    fs::create_dir(&args.out)?;
    write_new(&args.out.join("shape.json"), &shape_bytes)?;
    write_new(&args.out.join("source-bundle.json"), source_bundle)?;
    write_json(
        &args.out.join("provenance.json"),
        &json!({
            "schema":"native-blind-depletion-contact-discovery-v1", "arguments":args,
            "shape_path":fs::canonicalize(&args.shape)?, "shape_sha256":shape_sha256,
            "source_bundle_sha256":hash_bytes(source_bundle),
            "executable_sha256":hash_bytes(&fs::read(std::env::current_exe()?)?),
            "native_information":false,
            "native_information_scope":"Only the supplied rigid body's internal geometry is used; no inter-body native motif, label, covariance, atlas, or production pose is read.",
            "search":"Offline greedy optimization on outermost radial contacts. Equal-probability orientation-only, ray-only, and paired perturbations; independent 1/4/12 degree scales. Fixed pseudorandom score uniforms mapped through each pose's union-overlap envelope. Not an equilibrium trajectory.",
            "validation":"Two fresh fixed-size clouds for each initial and optimized pose, never used for selection. Binomial standard errors and Wilson intervals are pointwise diagnostic uncertainty, not selection-adjusted confidence statements.",
            "refinement":"Exact MemoryState depletion gate, one slot, no global refresh, finite anchored pair ball. Optional material pivot c is frozen from the optimized witness; p=t+Rc has unit translation/Haar Jacobian and centered displacement/Cayley increments are inverse symmetric. Optional scalar adaptation occurs only after discarded warmup windows, with frozen widths during all retained samples. Fixed burn/save stride include rejections; finite runs do not establish stationarity or basin masses.",
            "selection":"Every initialized slot is preserved. No validation winner, native score, or trajectory from production enters construction.",
        "rng":"StdRng with SHA256-derived master-seed/slot/stream keys; pinned rand version in source bundle Cargo.lock",
            "completion":"manifest.json is written only after all outputs are complete and hashed",
            "initial_poses_sha256":initial_poses_sha256,
            "initial_poses_scope":"When supplied, start poses must come from a shape-only procedure (no native motif, label or production pose); each slot then uses local non-projecting greedy search and nearest-pair witnesses."
        }),
    )?;
    let mut slots = Vec::with_capacity(config.starts);
    for slot in 0..config.starts {
        let slot_started = Instant::now();
        let supplied = initial_poses.as_ref().map(|poses| poses[slot]);
        let result = discover_slot_from(&tree, &config, slot, supplied)?;
        eprintln!(
            "{}",
            json!({"slot":slot,"slots":config.starts,
            "wall_seconds":slot_started.elapsed().as_secs_f64(),
            "search_initial_volume":result.search_initial_score.volume,
            "search_optimized_volume":result.search_optimized_score.volume,
            "refinement_accepted":result.refinement_counts.accepted,
            "refinement_production_accepted":result.refinement_production_counts.accepted,
            "refinement_final_scale":result.refinement_protocol.final_scale,
            "refinement_distinct_retained_poses":result.refinement_distinct_retained_poses,
            "refinement_samples":result.refinement_samples.len()})
        );
        slots.push(result);
    }
    let retained_samples: usize = slots.iter().map(|s| s.refinement_samples.len()).sum();
    let result = DiscoveryResult {
        schema: "native-blind-depletion-contact-discovery-v1",
        config: config.clone(),
        gate_options: config.gate_options(),
        slots,
    };
    let mut discovery = serde_json::to_value(&result)?;
    discovery["shape_sha256"] = json!(shape_sha256);
    write_json(&args.out.join("discovery.json"), &discovery)?;
    let mut outputs_sha256 = BTreeMap::new();
    for name in [
        "shape.json",
        "source-bundle.json",
        "provenance.json",
        "discovery.json",
    ] {
        outputs_sha256.insert(name, hash_bytes(&fs::read(args.out.join(name))?));
    }
    let manifest = json!({"schema":"native-blind-depletion-contact-discovery-v1",
        "complete":true,"native_information":false,"shape_sha256":shape_sha256,
        "outputs_sha256":outputs_sha256,"config":config,"slots":result.slots.len(),
        "retained_samples":retained_samples,"wall_seconds":started.elapsed().as_secs_f64()});
    write_json(&args.out.join("manifest.json"), &manifest)?;
    println!(
        "{}",
        json!({"out":args.out,"complete":true,"slots":result.slots.len(),
        "retained_samples":retained_samples,"wall_seconds":started.elapsed().as_secs_f64(),
        "manifest_sha256":hash_bytes(&fs::read(args.out.join("manifest.json"))?)})
    );
    Ok(())
}
