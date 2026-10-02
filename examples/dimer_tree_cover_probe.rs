//! Fixed-allocation, passive saved-state cover costs. No Poisson sampling,
//! acceptance decision, state update, classifier, or physical trajectory.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use rand::{SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    path::PathBuf,
};
use tetramer_mc::{
    depletion::{Envelope, GateOptions},
    dimer_tree_proposal::{DimerTreeProposal, tree_coordinates},
    docking::{DockingMethod, DockingProposal},
    flexible_subset::FlexibleSubset,
    geometry::{Placed, Shape, SphereTree},
    math::{Pose, norm, sub},
    proposal::FrozenRelativePoseProposal,
    rigid_subset::RigidSubset,
    simulation::{cpu_seconds, hash_bytes, hash_file, save},
    spherical::{Container, validate_state},
};

const BUNDLE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/source-bundle.json"));
const EXAMPLE: &[u8] = include_bytes!("dimer_tree_cover_probe.rs");

#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    /// Written and reviewed after building, before the one claimed invocation.
    #[arg(long)]
    binding: PathBuf,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BoundFile {
    path: PathBuf,
    sha256: String,
}
impl BoundFile {
    fn read(&self) -> Result<Vec<u8>> {
        let bytes = fs::read(&self.path).with_context(|| self.path.display().to_string())?;
        ensure!(
            hash_bytes(&bytes) == self.sha256,
            "changed input {}",
            self.path.display()
        );
        Ok(bytes)
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Atlas {
    name: String,
    model: BoundFile,
    seed: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    schema: String,
    protocol: BoundFile,
    source_config: BoundFile,
    source_frame: BoundFile,
    source_freeze_manifest: BoundFile,
    shape: BoundFile,
    atlases: Vec<Atlas>,
    compiled_source_sha256: BTreeMap<String, String>,
    root: usize,
    child: usize,
    anchor: usize,
    attempts_per_atlas: usize,
    correlation: f64,
    depletant_radius: f64,
    activity: f64,
    lambda: f64,
    wall_radius: f64,
    envelope: GateOptions,
    output: PathBuf,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Binding {
    schema: String,
    config_sha256: String,
    protocol_sha256: String,
    example_source_sha256: String,
    compiled_source_bundle_sha256: String,
    executable_sha256: String,
}

#[derive(Serialize)]
struct Cover {
    volume: f64,
    root_volume: f64,
    created_cells: usize,
    retained_cells: usize,
    spectator_count: usize,
    construction_cpu_seconds: f64,
    envelope_cpu_seconds: f64,
    expected_raw_points: f64,
}
fn cover(
    env: Envelope,
    root_volume: f64,
    spectators: usize,
    construction_cpu_seconds: f64,
    envelope_cpu_seconds: f64,
    plan: &Plan,
) -> Result<Cover> {
    let expected = (plan.lambda + plan.activity) * env.volume;
    ensure!(expected.is_finite(), "nonfinite expected raw point cost");
    Ok(Cover {
        volume: env.volume,
        root_volume,
        created_cells: env.created,
        retained_cells: env.cells.len(),
        spectator_count: spectators,
        construction_cpu_seconds,
        envelope_cpu_seconds,
        expected_raw_points: expected,
    })
}
fn singleton(
    tree: &SphereTree,
    state: &[Pose],
    member: usize,
    proposed: Pose,
    plan: &Plan,
) -> Result<Cover> {
    let start = cpu_seconds();
    let gate = RigidSubset::new(
        tree,
        state,
        &[member],
        member,
        proposed,
        plan.depletant_radius,
    )?;
    let construction = cpu_seconds() - start;
    let clock = cpu_seconds();
    let env = gate.envelope(plan.envelope)?;
    cover(
        env,
        gate.root_bounds().volume(),
        gate.spectator_count(),
        construction,
        cpu_seconds() - clock,
        plan,
    )
}
fn reported<T: Serialize>(result: &Result<T>) -> Value {
    match result {
        Ok(value) => json!({"status":"ok", "value":value}),
        Err(error) => json!({"status":"error", "error":format!("{error:#}")}),
    }
}
fn singleton_path(
    tree: &SphereTree,
    state: &[Pose],
    proposed: [Pose; 2],
    root_first: bool,
    plan: &Plan,
) -> Value {
    let order = if root_first { [0, 1] } else { [1, 0] };
    let labels = [plan.root, plan.child];
    let mut intermediate = state.to_vec();
    // Direct copies preserve endpoint poses exactly. No coordinate re-decoding,
    // hard check, acceptance, or retry is inserted at the auxiliary intermediate.
    intermediate[labels[order[0]]] = proposed[order[0]];
    let a = singleton(tree, state, labels[order[0]], proposed[order[0]], plan);
    let b = singleton(
        tree,
        &intermediate,
        labels[order[1]],
        proposed[order[1]],
        plan,
    );
    let sum = match (&a, &b) {
        (Ok(a), Ok(b)) => json!({
            "volume":a.volume+b.volume,
            "created_cells":a.created_cells+b.created_cells,
            "retained_cells":a.retained_cells+b.retained_cells,
            "construction_cpu_seconds":a.construction_cpu_seconds+b.construction_cpu_seconds,
            "envelope_cpu_seconds":a.envelope_cpu_seconds+b.envelope_cpu_seconds,
            "expected_raw_points":a.expected_raw_points+b.expected_raw_points
        }),
        _ => Value::Null,
    };
    json!({"order":order.map(|i| labels[i]),
        "intermediate_selected":[intermediate[plan.root],intermediate[plan.child]],
        "intermediate_hard_filter":false,"legs":[reported(&a),reported(&b)],"sum":sum})
}
fn line(writer: &mut impl Write, row: &Value) -> Result<()> {
    serde_json::to_writer(&mut *writer, row)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

fn execute(plan: &Plan, writer: &mut impl Write) -> Result<Value> {
    ensure!(
        plan.schema == "dimer-tree-passive-cover-v1",
        "unknown plan schema"
    );
    ensure!(
        plan.root == 27 && plan.child == 132 && plan.anchor == 228,
        "changed labels"
    );
    ensure!(
        plan.attempts_per_atlas == 16 && plan.atlases.len() == 2,
        "changed allocation"
    );
    ensure!(
        plan.correlation == 0.7
            && plan.depletant_radius == 1.4
            && plan.activity == 0.0275
            && plan.lambda == 1.76,
        "changed physical/map settings"
    );
    ensure!(
        plan.envelope.max_cells == 255
            && plan.envelope.max_depth == 8
            && plan.envelope.min_width == 0.,
        "changed cover settings"
    );
    plan.envelope.validate()?;
    let embedded: Value = serde_json::from_slice(BUNDLE)?;
    let embedded_hashes: BTreeMap<String, String> = embedded["files"]
        .as_object()
        .context("invalid embedded source bundle")?
        .iter()
        .map(|(name, v)| {
            Ok((
                name.clone(),
                v["sha256"].as_str().context("missing source hash")?.into(),
            ))
        })
        .collect::<Result<_>>()?;
    ensure!(
        embedded_hashes == plan.compiled_source_sha256,
        "compiled source differs from frozen plan"
    );
    let config: Value = serde_json::from_slice(&plan.source_config.read()?)?;
    let frame: Value = serde_json::from_slice(&plan.source_frame.read()?)?;
    let archived: Value = serde_json::from_slice(&plan.source_freeze_manifest.read()?)?;
    ensure!(
        archived["frame_sha256"] == plan.source_frame.sha256
            && archived["shape_sha256"] == plan.shape.sha256,
        "archive binding differs"
    );
    ensure!(
        config["initial_poses"] == frame["poses"] && frame["sweep"] == 7400,
        "saved poses differ"
    );
    ensure!(
        config["depletant_radius"] == plan.depletant_radius
            && config["reservoir_density"] == plan.activity
            && config["poisson_lambda_ratio"] == 64.
            && config["boundary"]["kind"] == "spherical"
            && config["boundary"]["radius"] == plan.wall_radius,
        "saved physical settings differ"
    );
    let state: Vec<Pose> = serde_json::from_value(config["initial_poses"].clone())?;
    ensure!(state.len() == 264, "saved body count differs");
    let tree = SphereTree::new(serde_json::from_slice::<Shape>(&plan.shape.read()?)?)?;
    let wall = Container::new(plan.wall_radius, &tree)?;
    // These archived initial_poses are sphere-centered, as in the old reset
    // benchmark. coordinate_wall_center is trajectory metadata, not a second
    // shift applied to these already centered pose values.
    validate_state(&tree, &wall, &state)?;
    let old_edges = tree_coordinates(state[plan.anchor], state[plan.root], state[plan.child])?;
    let mut models = Vec::new();
    for atlas in &plan.atlases {
        let bytes = atlas.model.read()?;
        let model = FrozenRelativePoseProposal::from_json_str_open(
            std::str::from_utf8(&bytes)?,
            [2. * (plan.wall_radius + tree.bound); 3],
            0.1,
            &plan.shape.sha256,
        )?;
        models.push(DockingProposal::new(
            model,
            DockingMethod::PosteriorInvolution,
            plan.correlation,
            [0.; 3],
        )?);
    }
    let mut attempts = 0;
    let mut finite = 0;
    let mut full_hard_valid = 0;
    let mut cover_errors = 0;
    for (atlas, docking) in plan.atlases.iter().zip(&models) {
        let proposal = DimerTreeProposal::new(docking)?;
        let mut rng = StdRng::seed_from_u64(atlas.seed);
        for attempt in 0..plan.attempts_per_atlas {
            attempts += 1;
            let start = cpu_seconds();
            let result = proposal.propose(
                &mut rng,
                state[plan.anchor],
                state[plan.root],
                state[plan.child],
            );
            let proposal_cpu = cpu_seconds() - start;
            let outcome = match result {
                Ok(outcome) => outcome,
                Err(error) => {
                    line(
                        writer,
                        &json!({"atlas":atlas.name,"attempt":attempt,
                        "status":"proposal_error","error":format!("{error:#}"),
                        "proposal_cpu_seconds":proposal_cpu}),
                    )?;
                    continue;
                }
            };
            let Some(candidate) = &outcome.candidate else {
                line(
                    writer,
                    &json!({"atlas":atlas.name,"attempt":attempt,"status":"proposal_null",
                    "proposal_cpu_seconds":proposal_cpu,"outcome":outcome}),
                )?;
                continue;
            };
            finite += 1;
            let proposed = [candidate.root, candidate.child];
            let clock = cpu_seconds();
            let placed = proposed.map(Placed::new);
            let core_valid = !tree.overlaps(&placed[0], &placed[1])
                && state
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != plan.root && *i != plan.child)
                    .all(|(_, p)| placed.iter().all(|q| !tree.overlaps(q, &Placed::new(*p))));
            let wall_valid = proposed.iter().all(|p| wall.contains(*p));
            let endpoint_valid = core_valid && wall_valid;
            full_hard_valid += usize::from(endpoint_valid);
            let hard_cpu = cpu_seconds() - clock;
            let new_edges = tree_coordinates(state[plan.anchor], proposed[0], proposed[1]);
            let relative = new_edges.map(|new| {
                json!({"old":old_edges[1],"new":new[1],
                "old_distance":norm(old_edges[1].position),"new_distance":norm(new[1].position),
                "displacement":sub(new[1].position,old_edges[1].position),
                "displacement_norm":norm(sub(new[1].position,old_edges[1].position))})
            });
            let world = (|| -> Result<Cover> {
                let start = cpu_seconds();
                let gate = FlexibleSubset::new(
                    &tree,
                    &state,
                    &[plan.root, plan.child],
                    &proposed,
                    plan.depletant_radius,
                )?;
                let construction = cpu_seconds() - start;
                let clock = cpu_seconds();
                let env = gate.envelope(plan.envelope)?;
                cover(
                    env,
                    gate.root_bounds().volume(),
                    gate.spectator_count(),
                    construction,
                    cpu_seconds() - clock,
                    plan,
                )
            })();
            // Every finite endpoint receives all three cover diagnostics, even
            // when the endpoint is invalid or a different cover reports error.
            let root_first = singleton_path(&tree, &state, proposed, true, plan);
            let child_first = singleton_path(&tree, &state, proposed, false, plan);
            cover_errors += usize::from(world.is_err())
                + usize::from(root_first["sum"].is_null())
                + usize::from(child_first["sum"].is_null());
            line(
                writer,
                &json!({"atlas":atlas.name,"attempt":attempt,"status":"finite_endpoint",
                "proposal_cpu_seconds":proposal_cpu,"outcome":outcome,
                "core_valid":core_valid,"wall_valid":wall_valid,
                "full_endpoint_hard_valid":endpoint_valid,"hard_check_cpu_seconds":hard_cpu,
                "internal_relative_change":reported(&relative),"world":reported(&world),
                "root_first":root_first,"child_first":child_first}),
            )?;
        }
    }
    ensure!(attempts == 32, "attempt ledger incomplete");
    Ok(
        json!({"attempts":attempts,"finite_endpoints":finite,"full_hard_valid":full_hard_valid,
        "cover_errors":cover_errors,"all_covers_ok":cover_errors==0,
        "poisson_clouds":0,"acceptance_decisions":0,"state_updates":0,
        "scope":"purposive saved-state passive cover cost; no probability, equilibrium or assembly inference"}),
    )
}

fn main() -> Result<()> {
    let args = Args::parse();
    let config_bytes = fs::read(&args.config)?;
    let plan: Plan = serde_json::from_slice(&config_bytes)?;
    let binding_bytes = fs::read(&args.binding)?;
    let binding: Binding = serde_json::from_slice(&binding_bytes)?;
    ensure!(
        binding.schema == "dimer-tree-cover-prelaunch-binding-v1",
        "unknown binding schema"
    );
    ensure!(
        hash_bytes(&config_bytes) == binding.config_sha256,
        "changed plan"
    );
    ensure!(
        hash_bytes(EXAMPLE) == binding.example_source_sha256,
        "different example source"
    );
    ensure!(
        hash_bytes(BUNDLE) == binding.compiled_source_bundle_sha256,
        "different source bundle"
    );
    ensure!(
        hash_file(&std::env::current_exe()?)? == binding.executable_sha256,
        "different executable"
    );
    let protocol = plan.protocol.read()?;
    ensure!(
        hash_bytes(&protocol) == binding.protocol_sha256,
        "different protocol"
    );
    // create_dir fails if this run has ever been claimed. No overwrite/resume.
    fs::create_dir(&plan.output).context("output already exists or parent is missing")?;
    fs::write(plan.output.join("config.json"), config_bytes)?;
    fs::write(plan.output.join("prelaunch-binding.json"), binding_bytes)?;
    fs::write(plan.output.join("protocol.json"), protocol)?;
    fs::write(plan.output.join("source-bundle.json"), BUNDLE)?;
    fs::write(plan.output.join("example.rs"), EXAMPLE)?;
    let process = fs::read_to_string("/proc/self/stat")?;
    let birth = process
        .rsplit_once(") ")
        .context("invalid process stat")?
        .1
        .split_whitespace()
        .nth(19)
        .context("missing process birth")?
        .to_string();
    let status = fs::read_to_string("/proc/self/status")?;
    let threads = status
        .lines()
        .find_map(|s| s.strip_prefix("Threads:"))
        .context("missing thread count")?
        .trim()
        .parse::<usize>()?;
    ensure!(threads == 1, "probe must have one scientific thread");
    save(
        &plan.output.join("started.json"),
        &json!({"pid":std::process::id(),
        "birth_ticks":birth,"threads":threads,"executable_sha256":binding.executable_sha256}),
    )?;
    let rows = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(plan.output.join("attempts.jsonl"))?;
    let mut writer = BufWriter::new(rows);
    let start = cpu_seconds();
    let result = execute(&plan, &mut writer);
    writer.flush()?;
    save(
        &plan.output.join("terminal.json"),
        &json!({"complete":result.is_ok(),
        "result":reported(&result),"cpu_seconds":cpu_seconds()-start,
        "attempts_sha256":hash_file(&plan.output.join("attempts.jsonl"))?}),
    )?;
    result.map(|_| ())
}
