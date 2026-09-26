//! Fixed-allocation conditional diagnostic of the production cluster phase.
//! Each phase starts from the same frozen physical state. This is not a
//! continuous assembly trajectory, equilibrium estimator, or ESS benchmark.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use rand::{SeedableRng, rngs::StdRng};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};
use tetramer_mc::{
    cluster_phase::{ClusterPhase, ClusterPhaseCounts, ContactGraph, TransportCharts},
    docking::{DockingMethod, DockingProposal},
    geometry::{Shape, SphereTree},
    math::Pose,
    proposal::FrozenRelativePoseProposal,
    simulation::{Boundary, Config, cpu_seconds, hash_bytes, hash_file, save},
    spherical::Container,
};

#[derive(Debug, Parser, Serialize)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value_t = 128)]
    phases: u64,
    #[arg(long)]
    seed: u64,
    #[arg(long)]
    out: PathBuf,
}
#[derive(Default, Serialize)]
struct Counts {
    attempted: u64,
    hard_valid: u64,
    accepted: u64,
    proposal_nulls: u64,
    proposed_attachments: u64,
    proposed_detachments: u64,
    proposed_exchanges: u64,
    accepted_attachments: u64,
    accepted_detachments: u64,
    accepted_exchanges: u64,
    accepted_contact_changing: u64,
    pool_recorded: u64,
    pool_contains_direct_partner: u64,
    old_external_contacts: u64,
    conditional_acceptance_probability_sum: f64,
}
impl Counts {
    fn add(
        &mut self,
        row: &Value,
        gained: usize,
        lost: usize,
        pool_covered: Option<bool>,
        external: usize,
    ) {
        let hard = row["hard_valid"] == true;
        let accepted = row["accepted"] == true;
        self.attempted += 1;
        self.hard_valid += u64::from(hard);
        self.accepted += u64::from(accepted);
        self.proposal_nulls += u64::from(row["proposed_poses"].is_null());
        self.proposed_attachments += u64::from(gained > 0 && lost == 0);
        self.proposed_detachments += u64::from(gained == 0 && lost > 0);
        self.proposed_exchanges += u64::from(gained > 0 && lost > 0);
        self.accepted_attachments += u64::from(accepted && gained > 0 && lost == 0);
        self.accepted_detachments += u64::from(accepted && gained == 0 && lost > 0);
        self.accepted_exchanges += u64::from(accepted && gained > 0 && lost > 0);
        self.accepted_contact_changing += u64::from(accepted && gained + lost > 0);
        self.pool_recorded += u64::from(pool_covered.is_some());
        self.pool_contains_direct_partner += u64::from(pool_covered == Some(true));
        self.old_external_contacts += external as u64;
        self.conditional_acceptance_probability_sum +=
            row["log_acceptance"].as_f64().map_or(0., f64::exp);
    }
}
fn stream(seed: u64, phase: u64, label: &str) -> StdRng {
    // Same stream construction as simulation::stream. The seed and reset
    // protocol distinguish this explicitly conditional experiment.
    let mut h = Sha256::new();
    h.update(b"tetramer-mc-rng-v1");
    h.update(seed.to_le_bytes());
    h.update(phase.to_le_bytes());
    h.update(label.as_bytes());
    StdRng::from_seed(h.finalize().into())
}
fn line(writer: &mut impl Write, value: &Value) -> Result<()> {
    serde_json::to_writer(&mut *writer, value)?;
    writer.write_all(b"\n")?;
    Ok(())
}
fn resolve(config: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_owned()
    } else {
        config.parent().unwrap_or(Path::new(".")).join(path)
    }
}
fn execute(args: &Args) -> Result<()> {
    let all_start = Instant::now();
    let total_cpu_start = cpu_seconds();
    ensure!(args.phases > 0, "phase allocation must be positive");
    let config_raw = fs::read(&args.config)?;
    let mut config: Config = serde_json::from_slice(&config_raw)?;
    config.validate()?;
    let shape_path = resolve(&args.config, &config.shape);
    let shape_raw = fs::read(&shape_path)?;
    let shape_sha = hash_bytes(&shape_raw);
    let tree = SphereTree::new(serde_json::from_slice::<Shape>(&shape_raw)?)?;
    let radius = match config.boundary {
        Boundary::Spherical { radius } => radius,
        _ => anyhow::bail!("conditional cluster diagnostic requires spherical boundaries"),
    };
    ensure!(
        config.assembly_bias.is_none(),
        "diagnostic presently requires unbiased physical target"
    );
    ensure!(
        config.fixed_body_indices.is_empty(),
        "diagnostic requires all-mobile system"
    );
    let settings = config
        .cluster_phase
        .clone()
        .context("missing cluster phase config")?;
    ensure!(
        settings.enabled() && settings.transport_charts == TransportCharts::Members,
        "diagnostic needs enabled member charts"
    );
    let wall = Container::new(radius, &tree)?;
    let initial = config.initial_poses.clone();
    tetramer_mc::spherical::validate_state(&tree, &wall, &initial)?;
    let model_raw = fs::read(&args.model)?;
    let model = FrozenRelativePoseProposal::from_json_str_open(
        std::str::from_utf8(&model_raw)?,
        [2. * (radius + tree.bound); 3],
        config.learned_uniform_weight,
        &shape_sha,
    )?;
    let virtual_branches = model.virtual_branches();
    let proposal = DockingProposal::new(
        model,
        DockingMethod::PosteriorInvolution,
        settings.correlation,
        [0.; 3],
    )?;
    let engine = ClusterPhase::new(
        &tree,
        &wall,
        config.depletant_radius,
        config.reservoir_density,
        if config.reservoir_density > 0. {
            config.reservoir_density * config.poisson_lambda_ratio
        } else {
            1.
        },
        config.endpoint_gate,
        settings.clone(),
        Some(proposal),
    )?;
    let mut exclusion_shape = tree.shape.clone();
    for atom in &mut exclusion_shape.atoms {
        atom.radius += config.depletant_radius;
    }
    let exclusion = SphereTree::new(exclusion_shape)?;
    let initial_graph = ContactGraph::build(&exclusion, &initial);
    let source_bundle = include_str!(concat!(env!("OUT_DIR"), "/source-bundle.json"));
    let compiled_sources: Value = serde_json::from_str(source_bundle)?;
    let source_hashes: BTreeMap<_, _> = compiled_sources["files"]
        .as_object()
        .context("invalid embedded sources")?
        .iter()
        .map(|(p, v)| (p.clone(), v["sha256"].clone()))
        .collect();
    fs::create_dir(&args.out)
        .context("output must not already exist: preserving previous attempts")?;
    fs::write(args.out.join("source-config.json"), &config_raw)?;
    fs::write(args.out.join("source-bundle.json"), source_bundle)?;
    config.shape = fs::canonicalize(&shape_path)?;
    save(&args.out.join("effective-config.json"), &config)?;
    let initial_hash = hash_bytes(&serde_json::to_vec(&initial)?);
    let manifest = json!({"protocol":"frozen-state-reset-fixed-duration-cluster-phase-v1","args":args,
        "physical_scope":"conditional transition diagnostic; each phase resets to frozen state; no equilibrium or mixing-time inference",
        "initial_poses_sha256":initial_hash,"config_sha256":hash_bytes(&config_raw),"model_sha256":hash_bytes(&model_raw),
        "shape_sha256":shape_sha,"shape_path":fs::canonicalize(&shape_path)?,"model_path":fs::canonicalize(&args.model)?,
        "virtual_branches":virtual_branches,"source_bundle_sha256":hash_bytes(source_bundle.as_bytes()),"source_sha256":source_hashes,"executable_sha256":hash_file(&std::env::current_exe()?)?,
        "cluster_phase":settings,"body_count":initial.len(),"depletant_radius":config.depletant_radius,
        "depletant_activity":config.reservoir_density,"poisson_lambda_ratio":config.poisson_lambda_ratio,
        "defensive_uniform_probability":config.learned_uniform_weight,"initial_eligible_subsets":initial_graph.channels(&settings).len(),
        "stream_protocol":"production SHA256(seed, phase, stream-label); phase numbers 1..=allocation",
        "timing":"kernel CPU includes graph build, proposal, full many-body gate, acceptance, production trace construction; passive graph replay and file I/O separately timed"});
    save(&args.out.join("manifest.json"), &manifest)?;
    let mut events = BufWriter::new(File::create(args.out.join("events.jsonl"))?);
    let mut phases = BufWriter::new(File::create(args.out.join("phases.jsonl"))?);
    let setup_cpu = cpu_seconds() - total_cpu_start;
    let mut kernel_cpu = 0.;
    let mut diagnostic_cpu = 0.;
    let mut total = ClusterPhaseCounts::default();
    let mut groups: BTreeMap<String, Counts> = BTreeMap::new();
    for phase in 1..=args.phases {
        let mut state = initial.clone();
        let mut clock = stream(args.seed, phase, "cluster-phase-clock-v1");
        let mut proposal_rng = stream(args.seed, phase, "cluster-phase-proposal-v1");
        let mut gate_rng = stream(args.seed, phase, "cluster-phase-gate-v1");
        let mut accept_rng = stream(args.seed, phase, "cluster-phase-accept-v1");
        let mut bias_rng = stream(args.seed, phase, "cluster-phase-bias-v1");
        let started = cpu_seconds();
        let (counts, rows) = engine.run(
            &mut state,
            &mut clock,
            &mut proposal_rng,
            &mut gate_rng,
            &mut accept_rng,
            &mut bias_rng,
            None,
            &mut None,
            true,
        )?;
        let elapsed = cpu_seconds() - started;
        kernel_cpu += elapsed;
        total.add(&counts);
        let diag_started = cpu_seconds();
        let mut replay = initial.clone();
        let mut graph = initial_graph.clone();
        let mut attempted = 0;
        for mut row in rows {
            row["phase"] = json!(phase);
            row["population_seed"] = json!(args.seed);
            if row["kind"] == "cluster_event" {
                attempted += 1;
                let members: Vec<usize> = serde_json::from_value(row["members"].clone())?;
                let old: Vec<Pose> = serde_json::from_value(row["old_poses"].clone())?;
                let retained: Vec<Pose> = serde_json::from_value(row["retained_poses"].clone())?;
                ensure!(
                    members.len() == old.len() && members.len() == retained.len(),
                    "event pose count mismatch"
                );
                for (&i, &p) in members.iter().zip(&old) {
                    ensure!(replay[i] == p, "event old pose disagrees with replay");
                }
                let ctx = graph.subset_context(&members)?;
                ensure!(
                    serde_json::to_value(&ctx)? == row["subset_context"],
                    "pre-event context differs from independently replayed graph"
                );
                let edges = graph.boundary_edges(&members);
                let partners: BTreeSet<usize> = edges
                    .iter()
                    .flat_map(|e| e.iter().copied())
                    .filter(|i| !members.contains(i))
                    .collect();
                let pool: Option<Vec<usize>> = row["proposal"]["anchor_pool"]
                    .as_array()
                    .map(|v| v.iter().map(|x| x.as_u64().unwrap() as usize).collect());
                let covered = pool
                    .as_ref()
                    .map(|p| p.iter().any(|i| partners.contains(i)));
                let in_pool: Vec<usize> = pool.as_ref().map_or(vec![], |p| {
                    p.iter().filter(|i| partners.contains(i)).copied().collect()
                });
                let gained = row["proposed_gained_contacts"]
                    .as_array()
                    .context("missing gained contacts")?
                    .len();
                let lost = row["proposed_lost_contacts"]
                    .as_array()
                    .context("missing lost contacts")?
                    .len();
                let topology = if ctx.whole_component {
                    "whole"
                } else {
                    "embedded"
                };
                let seed_parent = ctx
                    .parent_component_members
                    .iter()
                    .any(|i| config.seed_labels.contains(i));
                let branch = row["proposal"]["branch"]
                    .as_str()
                    .context("missing branch")?
                    .to_owned();
                row["diagnostic"] = json!({"external_partners":partners,"pool_direct_partners":in_pool,"pool_contains_direct_partner":covered,
                    "topology":topology,"seed_parent":seed_parent,"parent_size":ctx.parent_component_members.len(),"subset_size":members.len()});
                for key in [
                    "all".into(),
                    format!("branch={branch}"),
                    format!("branch={branch}/topology={topology}"),
                    format!("branch={branch}/topology={topology}/size={}", members.len()),
                    format!("branch={branch}/topology={topology}/seed_parent={seed_parent}"),
                ] {
                    groups
                        .entry(key)
                        .or_default()
                        .add(&row, gained, lost, covered, edges.len());
                }
                for (&i, &p) in members.iter().zip(&retained) {
                    replay[i] = p;
                }
                if row["accepted"] == true {
                    graph = graph.updated(&exclusion, &replay, &members);
                } else {
                    ensure!(retained == old, "rejected event changed state");
                }
            }
            line(&mut events, &row)?;
        }
        ensure!(
            attempted == counts.events,
            "not every attempted event retained"
        );
        ensure!(replay == state, "phase endpoint differs from replay");
        events.flush()?;
        line(
            &mut phases,
            &json!({"phase":phase,"counts":counts,"kernel_cpu_seconds":elapsed,"endpoint_poses_sha256":hash_bytes(&serde_json::to_vec(&state)?),"all_attempts_replayed":true}),
        )?;
        phases.flush()?;
        diagnostic_cpu += cpu_seconds() - diag_started;
    }
    let summary = json!({"complete":true,"phases":args.phases,"population_seed":args.seed,"counts":total,"groups":groups,
        "setup_cpu_seconds":setup_cpu,"kernel_cpu_seconds":kernel_cpu,"diagnostic_io_cpu_seconds":diagnostic_cpu,
        "total_cpu_seconds":cpu_seconds()-total_cpu_start,"wall_seconds":all_start.elapsed().as_secs_f64(),
        "all_attempted_events_retained":true,"all_phase_endpoints_replayed":true,
        "events_sha256":hash_file(&args.out.join("events.jsonl"))?,"phases_sha256":hash_file(&args.out.join("phases.jsonl"))?});
    save(&args.out.join("summary.json"), &summary)?;
    println!(
        "{}",
        serde_json::to_string(
            &json!({"out":args.out,"complete":true,"events":total.events,"accepted":total.accepted,"kernel_cpu_seconds":kernel_cpu})
        )?
    );
    Ok(())
}
fn main() -> Result<()> {
    let args = Args::parse();
    if let Err(e) = execute(&args) {
        if args.out.is_dir() {
            let _ = save(
                &args.out.join("failure.json"),
                &json!({"complete":false,"error":format!("{e:#}"),"args":args}),
            );
        }
        return Err(e);
    }
    Ok(())
}
