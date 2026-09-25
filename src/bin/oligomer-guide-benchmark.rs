//! Complete fixed-time phase probes reset to a frozen full-system configuration.
//! This measures conditional transition performance, not equilibrium or ESS.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use rand::{SeedableRng, rngs::StdRng};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};
use tetramer_mc::{
    cluster_phase::{ClusterPhase, ClusterPhaseCounts, ContactGraph},
    docking::{DockingMethod, DockingProposal},
    geometry::{Shape, SphereTree},
    math::{Pose, norm, sub},
    proposal::FrozenRelativePoseProposal,
    simulation::{Boundary, Config, cpu_seconds, hash_bytes, hash_file, save},
    spherical::{self, Container},
};

const SOURCE_BUNDLE: &str = include_str!(concat!(env!("OUT_DIR"), "/source-bundle.json"));

#[derive(Parser, Serialize)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    preparation: String,
    /// Zero is the unchanged baseline; positive values enable the guide.
    #[arg(long)]
    steps: usize,
    #[arg(long, default_value_t = 4)]
    anchor_count: usize,
    #[arg(long, default_value_t = 1.0)]
    score_power: f64,
    #[arg(long)]
    population: usize,
    #[arg(long, default_value_t = 16)]
    phases: usize,
    #[arg(long, default_value_t = 202609251400)]
    seed: u64,
}

fn stream(args: &Args, phase: usize, kind: &str) -> StdRng {
    let mut h = Sha256::new();
    h.update(b"tetramer-mc-oligomer-guide-frozen-phase-v1");
    h.update(args.seed.to_le_bytes());
    h.update((args.preparation.len() as u64).to_le_bytes());
    h.update(args.preparation.as_bytes());
    h.update((args.population as u64).to_le_bytes());
    h.update((phase as u64).to_le_bytes());
    h.update(kind.as_bytes());
    // Intentionally omit arm/steps: coupled random streams across proposal arms.
    StdRng::from_seed(h.finalize().into())
}
fn resolve(config: &Path, path: &Path) -> Result<PathBuf> {
    Ok(if path.is_absolute() {
        path.to_path_buf()
    } else {
        config.parent().context("config parent")?.join(path)
    }
    .canonicalize()?)
}
fn edges(graph: &ContactGraph) -> BTreeSet<(usize, usize)> {
    let mut result = BTreeSet::new();
    for i in 0..graph.adjacency.len() {
        for j in i + 1..graph.adjacency.len() {
            if graph.adjacency[i][j] {
                result.insert((i, j));
            }
        }
    }
    result
}
fn changed(old: &[Pose], new: &[Pose]) -> bool {
    old.iter().zip(new).any(|(a, b)| {
        let qdot: f64 = a
            .orientation
            .iter()
            .zip(b.orientation)
            .map(|(u, v)| u * v)
            .sum();
        norm(sub(a.position, b.position)) > 1e-10 || qdot.abs() < 1. - 1e-12
    })
}
fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}
fn distribution(values: &[f64]) -> Value {
    if values.is_empty() {
        return json!({"n":0});
    }
    let mut x = values.to_vec();
    x.sort_by(f64::total_cmp);
    let q = |p: f64| x[((x.len() - 1) as f64 * p).round() as usize];
    json!({"n":x.len(),"mean":mean(&x),"min":x[0],"q05":q(0.05),"median":q(0.5),"q95":q(0.95),"max":x[x.len()-1]})
}

#[derive(Default, Serialize)]
struct EventMetrics {
    attempted: u64,
    proposal_nulls: u64,
    hard_valid: u64,
    hard_rejected: u64,
    internal_geometry_nulls: u64,
    physical_accepted: u64,
    accepted: u64,
    proposed_pose_changes: u64,
    accepted_pose_changes: u64,
    proposed_attachments: u64,
    proposed_detachments: u64,
    proposed_exchanges: u64,
    accepted_attachments: u64,
    accepted_detachments: u64,
    completed_exchanges: u64,
    proposed_gained_contacts: u64,
    proposed_lost_contacts: u64,
    accepted_gained_contacts: u64,
    accepted_lost_contacts: u64,
    conditional_acceptance_sum: f64,
    contact_change_acceptance_sum: f64,
    accepted_fingerprint_distance_sum: f64,
    guide_events: u64,
    guide_counts: BTreeMap<String, u64>,
    guide_changed_endpoints: u64,
    guide_cpu_seconds: f64,
    guide_geometry_cpu_seconds: f64,
    guide_density_cpu_seconds: f64,
    #[serde(skip)]
    corrections: Vec<f64>,
    #[serde(skip)]
    valid_corrections: Vec<f64>,
    #[serde(skip)]
    changing_corrections: Vec<f64>,
    #[serde(skip)]
    valid_depletion: Vec<f64>,
    #[serde(skip)]
    valid_log_acceptance: Vec<f64>,
}
impl EventMetrics {
    fn record(&mut self, row: &Value, pose_change: bool, gained: usize, lost: usize, jaccard: f64) {
        let null = row["proposed_poses"].is_null();
        let hard = row["hard_valid"] == true;
        let preserved = row["internal_contact_graph_preserved"] == true;
        let accept = row["accepted"] == true;
        let changing = gained + lost > 0;
        self.attempted += 1;
        if row["proposal"]["branch"] == "oligomer_guide" {
            self.guide_events += 1;
            if let Some(counts) = row["proposal"]["guide_counts"].as_object() {
                for (key, value) in counts {
                    if let Some(n) = value.as_u64() {
                        *self.guide_counts.entry(key.clone()).or_default() += n;
                    }
                }
                self.guide_changed_endpoints +=
                    u64::from(counts.get("endpoint_changed") == Some(&json!(true)));
            }
            self.guide_cpu_seconds += row["proposal"]["guide_cpu_seconds"].as_f64().unwrap_or(0.);
            self.guide_geometry_cpu_seconds += row["proposal"]["guide_geometry_cpu_seconds"]
                .as_f64()
                .unwrap_or(0.);
            self.guide_density_cpu_seconds += row["proposal"]["guide_density_cpu_seconds"]
                .as_f64()
                .unwrap_or(0.);
        }
        self.proposal_nulls += u64::from(null);
        self.hard_valid += u64::from(hard);
        self.hard_rejected += u64::from(!null && !hard);
        self.internal_geometry_nulls += u64::from(hard && !preserved);
        self.physical_accepted += u64::from(row["physical_accepted"] == true);
        self.accepted += u64::from(accept);
        self.proposed_pose_changes += u64::from(pose_change);
        self.accepted_pose_changes += u64::from(accept && pose_change);
        self.proposed_attachments += u64::from(gained > 0 && lost == 0);
        self.proposed_detachments += u64::from(gained == 0 && lost > 0);
        self.proposed_exchanges += u64::from(gained > 0 && lost > 0);
        self.proposed_gained_contacts += gained as u64;
        self.proposed_lost_contacts += lost as u64;
        if accept {
            self.accepted_attachments += u64::from(gained > 0 && lost == 0);
            self.accepted_detachments += u64::from(gained == 0 && lost > 0);
            self.completed_exchanges += u64::from(gained > 0 && lost > 0);
            self.accepted_gained_contacts += gained as u64;
            self.accepted_lost_contacts += lost as u64;
            self.accepted_fingerprint_distance_sum += jaccard;
        }
        if let Some(x) = row["proposal"]["log_reverse_forward"]
            .as_f64()
            .filter(|x| x.is_finite())
        {
            self.corrections.push(x);
            if hard && preserved {
                self.valid_corrections.push(x);
            }
            if changing {
                self.changing_corrections.push(x);
            }
        }
        if let Some(x) = row["gate"]["log_weight"].as_f64().filter(|x| x.is_finite()) {
            self.valid_depletion.push(x);
        }
        if let Some(x) = row["log_acceptance"].as_f64().filter(|x| x.is_finite()) {
            self.valid_log_acceptance.push(x);
            // Inner-guide self-loops are deliberately excluded from moved-endpoint acceptance.
            if pose_change {
                self.conditional_acceptance_sum += x.min(0.).exp();
            }
            if changing {
                self.contact_change_acceptance_sum += x.min(0.).exp();
            }
        }
    }
    fn report(&self) -> Value {
        let mut v = serde_json::to_value(self).unwrap();
        v["proposal_correction_all_nonnull"] = distribution(&self.corrections);
        v["proposal_correction_hard_valid"] = distribution(&self.valid_corrections);
        v["proposal_correction_contact_changing"] = distribution(&self.changing_corrections);
        v["sampled_depletion_hard_valid"] = distribution(&self.valid_depletion);
        v["log_acceptance_hard_valid"] = distribution(&self.valid_log_acceptance);
        v
    }
}
fn write_row(writer: &mut impl Write, value: &Value) -> Result<()> {
    serde_json::to_writer(&mut *writer, value)?;
    writer.write_all(b"\n")?;
    Ok(())
}

fn main() -> Result<()> {
    // Read-only provenance interface works before any sampling arguments or I/O.
    if std::env::args().len() == 2
        && std::env::args().nth(1).as_deref() == Some("--source-manifest")
    {
        println!("{SOURCE_BUNDLE}");
        return Ok(());
    }
    let args = Args::parse();
    ensure!(
        args.phases > 0 && args.phases <= 65536,
        "invalid fixed phase allocation"
    );
    ensure!(
        [0, 1, 4, 16].contains(&args.steps),
        "predeclared benchmark steps are 0,1,4,16"
    );
    ensure!(!args.out.exists(), "output exists; use a fresh directory");
    let overall_cpu = cpu_seconds();
    let overall_wall = Instant::now();
    let config_path = args.config.canonicalize()?;
    let raw_config = fs::read(&config_path)?;
    let mut value: Value = serde_json::from_slice(&raw_config)?;
    ensure!(
        value["cluster_phase"].is_object(),
        "input must explicitly supply common cluster phase settings"
    );
    value["cluster_phase"]["guide"] = if args.steps == 0 {
        Value::Null
    } else {
        json!({"steps":args.steps,"anchor_count":args.anchor_count,"score_power":args.score_power})
    };
    let config: Config = serde_json::from_value(value)?;
    config.validate()?;
    ensure!(
        config.assembly_bias.is_none(),
        "this physical probe refuses a biased target"
    );
    let radius = match config.boundary {
        Boundary::Spherical { radius } => radius,
        _ => anyhow::bail!("spherical benchmark only"),
    };
    let shape_path = resolve(&config_path, &config.shape)?;
    let shape_raw = fs::read(&shape_path)?;
    let shape_hash = hash_bytes(&shape_raw);
    let tree = SphereTree::new(serde_json::from_slice::<Shape>(&shape_raw)?)?;
    let wall = Container::new(radius, &tree)?;
    spherical::validate_state(&tree, &wall, &config.initial_poses)?;
    let settings = config
        .cluster_phase
        .clone()
        .context("cluster config missing")?;
    ensure!(
        settings.enabled() && settings.transport_probability > 0.,
        "benchmark requires enabled transport"
    );
    let model_path = args.model.canonicalize()?;
    let model_raw = fs::read(&model_path)?;
    let model = FrozenRelativePoseProposal::from_json_str_open(
        std::str::from_utf8(&model_raw)?,
        [2. * (radius + tree.bound); 3],
        config.learned_uniform_weight,
        &shape_hash,
    )?;
    let proposal = DockingProposal::new(
        model,
        DockingMethod::PosteriorInvolution,
        settings.correlation,
        [0.; 3],
    )?;
    let lambda = if config.reservoir_density > 0. {
        config.reservoir_density * config.poisson_lambda_ratio
    } else {
        1.
    };
    let engine = ClusterPhase::new(
        &tree,
        &wall,
        config.depletant_radius,
        config.reservoir_density,
        lambda,
        config.endpoint_gate,
        settings.clone(),
        Some(proposal),
    )?;
    let mut exclusion_shape = tree.shape.clone();
    for a in &mut exclusion_shape.atoms {
        a.radius += config.depletant_radius;
    }
    let exclusion = SphereTree::new(exclusion_shape)?;
    let original_graph = ContactGraph::build(&exclusion, &config.initial_poses);
    let original_edges = edges(&original_graph);
    let initial_channels = original_graph.channels(&settings);
    let initial_rate: f64 = initial_channels.iter().map(|c| c.rate).sum();
    let setup_cpu = cpu_seconds() - overall_cpu;
    fs::create_dir_all(&args.out)?;
    fs::write(args.out.join("input-config.json"), &raw_config)?;
    save(&args.out.join("effective-config.json"), &config)?;
    let manifest = json!({
        "schema":"oligomer-guide-frozen-phase-v1","args":args,
        "config":{"path":config_path,"sha256":hash_bytes(&raw_config)},
        "shape":{"path":shape_path,"sha256":shape_hash},
        "model":{"path":model_path,"sha256":hash_bytes(&model_raw)},
        "executable_sha256":hash_file(&std::env::current_exe()?)?,
        "source_bundle_sha256":hash_bytes(SOURCE_BUNDLE.as_bytes()),
        "physical_target":"hard cores and atomic spherical wall; exp(-z times exclusion-union volume), bath permeates wall",
        "scope":"Complete fixed-duration cluster phase from frozen full-system configuration, reset after each phase. Conditional transition diagnostic, not equilibrium trajectory, contact ESS, assembly stability or physical kinetics.",
        "allocation":"Fixed phase count; all events, rejections, inner self-loops, and final holding time retained; no timing or success-based stopping.",
        "rng":"SHA256 named streams paired across arms; independent population/phase/preparation labels; arm omitted intentionally",
        "other_kernel_schedule":"No ordinary single-body, GCA or center-shift updates between reset probes. This isolates the unchanged full-duration cluster phase; full simulation efficiency requires a subsequent benchmark.",
        "failure_protocol":"Error phases are recorded, reset and allocation drained; partial event history inside an errored call is unavailable and is marked incomplete, never silently counted as success.",
        "timing":"Sampler CPU includes production phase construction of contact graph and record collection; post-hoc graph/replay diagnostics and JSON I/O timed separately. Model/shape setup excluded but reported.",
        "pose_change_tolerance":{"translation_A":1e-10,"quaternion_abs_dot_deficit":1e-12},
        "initial_contact_edges":original_edges,"initial_eligible_channels":initial_channels.len(),"initial_rate":initial_rate,
        "cluster_phase":settings,"setup_cpu_seconds":setup_cpu
    });
    save(&args.out.join("manifest.json"), &manifest)?;
    let mut events = BufWriter::new(fs::File::create(args.out.join("events.jsonl"))?);
    let mut phases = BufWriter::new(fs::File::create(args.out.join("phases.jsonl"))?);
    let mut errors = BufWriter::new(fs::File::create(args.out.join("errors.jsonl"))?);
    let mut counts = ClusterPhaseCounts::default();
    let mut metrics = EventMetrics::default();
    let mut groups: BTreeMap<String, EventMetrics> = BTreeMap::new();
    let mut sampler_cpu = 0.;
    let mut diagnostic_cpu = 0.;
    let mut output_cpu = 0.;
    let mut completed = 0;
    let mut failed = 0;
    let mut phase_gained = 0;
    let mut phase_lost = 0;
    let mut fingerprint_distances = Vec::new();
    for phase in 0..args.phases {
        let mut state = config.initial_poses.clone();
        let before = cpu_seconds();
        let before_wall = Instant::now();
        let result = engine.run(
            &mut state,
            &mut stream(&args, phase, "clock"),
            &mut stream(&args, phase, "proposal"),
            &mut stream(&args, phase, "gate"),
            &mut stream(&args, phase, "accept"),
            &mut stream(&args, phase, "bias"),
            None,
            &mut None,
            true,
        );
        let phase_cpu = cpu_seconds() - before;
        sampler_cpu += phase_cpu;
        let phase_wall = before_wall.elapsed().as_secs_f64();
        let (stats, mut rows) = match result {
            Ok(x) => x,
            Err(e) => {
                failed += 1;
                let before = cpu_seconds();
                write_row(
                    &mut errors,
                    &json!({"phase":phase,"error":format!("{e:#}"),"sampler_cpu_seconds":phase_cpu,"sampler_wall_seconds":phase_wall,"partial_history_available":false}),
                )?;
                errors.flush()?;
                output_cpu += cpu_seconds() - before;
                continue;
            }
        };
        let before = cpu_seconds();
        let diagnostics = (|| -> Result<Value> {
            let mut replay = config.initial_poses.clone();
            let mut graph = original_graph.clone();
            let mut observed = 0;
            for row in &mut rows {
                row["benchmark_phase"] = json!(phase);
                row["benchmark_population"] = json!(args.population);
                row["benchmark_preparation"] = json!(args.preparation);
                row["benchmark_steps"] = json!(args.steps);
                if row["kind"] != "cluster_event" {
                    continue;
                }
                observed += 1;
                let members: Vec<usize> = serde_json::from_value(row["members"].clone())?;
                let old: Vec<Pose> = serde_json::from_value(row["old_poses"].clone())?;
                ensure!(
                    old == members.iter().map(|&i| replay[i]).collect::<Vec<_>>(),
                    "old pose replay mismatch"
                );
                let proposed: Option<Vec<Pose>> =
                    serde_json::from_value(row["proposed_poses"].clone())?;
                let pose_change = proposed.as_ref().is_some_and(|p| changed(&old, p));
                let context = graph.subset_context(&members)?;
                let seed_connected = context
                    .parent_component_members
                    .iter()
                    .any(|i| config.seed_labels.contains(i));
                let old_edges = graph.boundary_edges(&members);
                let gained: Vec<(usize, usize)> =
                    serde_json::from_value(row["proposed_gained_contacts"].clone())?;
                let lost: Vec<(usize, usize)> =
                    serde_json::from_value(row["proposed_lost_contacts"].clone())?;
                if row["hard_valid"] == true && row["internal_contact_graph_preserved"] == true {
                    let p = proposed.as_ref().context("hard-valid endpoint missing")?;
                    ensure!(p.len() == members.len(), "proposed member count mismatch");
                    let mut candidate = replay.clone();
                    for (&i, &pose) in members.iter().zip(p) {
                        candidate[i] = pose;
                    }
                    let proposed_graph = graph.updated(&exclusion, &candidate, &members);
                    let new_edges = proposed_graph.boundary_edges(&members);
                    let actual_gained: Vec<_> = new_edges
                        .difference(&old_edges)
                        .map(|e| (e[0], e[1]))
                        .collect();
                    let actual_lost: Vec<_> = old_edges
                        .difference(&new_edges)
                        .map(|e| (e[0], e[1]))
                        .collect();
                    ensure!(
                        actual_gained == gained && actual_lost == lost,
                        "proposed external contact changes mismatch"
                    );
                }
                let jaccard = if old_edges.len() + gained.len() > 0 {
                    (gained.len() + lost.len()) as f64 / (old_edges.len() + gained.len()) as f64
                } else {
                    0.
                };
                let key = format!(
                    "{}|{}|{}|{}",
                    members.len(),
                    if context.whole_component {
                        "whole"
                    } else {
                        "embedded"
                    },
                    if seed_connected {
                        "seed_connected"
                    } else {
                        "solution"
                    },
                    row["proposal"]["branch"].as_str().unwrap_or("unknown")
                );
                metrics.record(row, pose_change, gained.len(), lost.len(), jaccard);
                groups.entry(key.clone()).or_default().record(
                    row,
                    pose_change,
                    gained.len(),
                    lost.len(),
                    jaccard,
                );
                row["benchmark_diagnostic"] = json!({"group":key,"parent_seed_connected":seed_connected,
                    "proposed_pose_changed":pose_change,"accepted_pose_changed":row["accepted"]==true&&pose_change,
                    "proposed_cut_contact_jaccard_distance":jaccard,
                    "retained_cut_contact_jaccard_distance":if row["accepted"]==true{jaccard}else{0.}});
                let retained: Vec<Pose> = serde_json::from_value(row["retained_poses"].clone())?;
                ensure!(
                    retained.len() == members.len(),
                    "retained member count mismatch"
                );
                for (&i, &p) in members.iter().zip(&retained) {
                    replay[i] = p;
                }
                if row["accepted"] == true {
                    graph = graph.updated(&exclusion, &replay, &members);
                }
            }
            ensure!(observed == stats.events, "event count mismatch");
            ensure!(replay == state, "phase endpoint replay mismatch");
            let endpoint = edges(&ContactGraph::build(&exclusion, &state));
            ensure!(edges(&graph) == endpoint, "incremental graph mismatch");
            let gained: Vec<_> = endpoint.difference(&original_edges).copied().collect();
            let lost: Vec<_> = original_edges.difference(&endpoint).copied().collect();
            let union = endpoint.union(&original_edges).count();
            let distance = if union > 0 {
                (gained.len() + lost.len()) as f64 / union as f64
            } else {
                0.
            };
            phase_gained += gained.len();
            phase_lost += lost.len();
            fingerprint_distances.push(distance);
            Ok(
                json!({"phase":phase,"counts":stats,"sampler_cpu_seconds":phase_cpu,"sampler_wall_seconds":phase_wall,
                "pose_changed":changed(&config.initial_poses,&state),"gained_contacts":gained,"lost_contacts":lost,
                "contact_jaccard_distance":distance,"final_poses_sha256":hash_bytes(&serde_json::to_vec(&state)?),
                "replay_exact":true}),
            )
        })();
        diagnostic_cpu += cpu_seconds() - before;
        let before = cpu_seconds();
        // Preserve returned attempts even if independent diagnostics identify a failure.
        for row in &rows {
            write_row(&mut events, row)?;
        }
        match diagnostics {
            Ok(mut p) => {
                completed += 1;
                counts.add(&stats);
                p["diagnostics_valid"] = json!(true);
                write_row(&mut phases, &p)?;
            }
            Err(e) => {
                failed += 1;
                write_row(
                    &mut errors,
                    &json!({"phase":phase,"stage":"diagnostics","error":format!("{e:#}"),"partial_history_available":true}),
                )?;
            }
        }
        events.flush()?;
        phases.flush()?;
        errors.flush()?;
        output_cpu += cpu_seconds() - before;
    }
    let summary = json!({
        "schema":"oligomer-guide-frozen-phase-v1","complete":failed==0&&completed==args.phases,
        "preparation":args.preparation,"population":args.population,"steps":args.steps,
        "requested_phases":args.phases,"completed_phases":completed,"failed_phases":failed,
        "counts":counts,"metrics":metrics.report(),
        "groups":groups.iter().map(|(k,v)|(k.clone(),v.report())).collect::<BTreeMap<_,_>>(),
        "phase_fingerprint_distance":distribution(&fingerprint_distances),"phase_endpoint_gained_contacts":phase_gained,"phase_endpoint_lost_contacts":phase_lost,
        "sampler_cpu_seconds":sampler_cpu,"diagnostics_cpu_seconds":diagnostic_cpu,"output_cpu_seconds":output_cpu,
        "guide_cpu_seconds":metrics.guide_cpu_seconds,"guide_geometry_cpu_seconds":metrics.guide_geometry_cpu_seconds,
        "guide_density_cpu_seconds":metrics.guide_density_cpu_seconds,
        "guide_cpu_fraction":if sampler_cpu>0.{Some(metrics.guide_cpu_seconds/sampler_cpu)}else{None},
        "setup_cpu_seconds":setup_cpu,"total_cpu_seconds":cpu_seconds()-overall_cpu,"wall_seconds":overall_wall.elapsed().as_secs_f64(),
        "accepted_contact_changes_per_sampler_cpu_second":if sampler_cpu>0.{Some((metrics.accepted_attachments+metrics.accepted_detachments+metrics.completed_exchanges)as f64/sampler_cpu)}else{None},
        "completed_exchanges_per_sampler_cpu_second":if sampler_cpu>0.{Some(metrics.completed_exchanges as f64/sampler_cpu)}else{None},
        "scope":"Frozen-state conditional transition probe. No contact ESS, equilibrium, assembly-stability or general mixing-speedup claim. All denominators include rejections and nulls; accepted self-loops are distinguished from actual pose changes."
    });
    save(&args.out.join("summary.json"), &summary)?;
    println!("{}", serde_json::to_string(&summary)?);
    ensure!(
        failed == 0,
        "benchmark contains failed phases; see errors.jsonl"
    );
    Ok(())
}
