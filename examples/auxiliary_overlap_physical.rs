//! Prepared-only reset-state physical replay of an immutable passive ledger.
//! No proposal draws, adaptation, intermediate hard filtering or trajectory.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use rand::{RngExt, SeedableRng, distr::Open01, rngs::StdRng};
use rand_distr::{Distribution, Poisson};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    path::PathBuf,
};
use tetramer_mc::{
    capped_dimer::FixedDimerContext,
    depletion::{GateOptions, GateResult},
    geometry::{Shape, SphereTree},
    math::Pose,
    rigid_subset::RigidSubset,
    simulation::{cpu_seconds, hash_bytes, hash_file, save},
    singleton_path::{SingletonOrder, SingletonPath, SingletonPathResult},
    spherical::{Container, validate_state},
};

const BUNDLE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/source-bundle.json"));
const SOURCE: &[u8] = include_bytes!("auxiliary_overlap_physical.rs");
#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
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
        let bytes = fs::read(&self.path)?;
        ensure!(
            hash_bytes(&bytes) == self.sha256,
            "changed input {}",
            self.path.display()
        );
        Ok(bytes)
    }
    fn json(&self) -> Result<Value> {
        Ok(serde_json::from_slice(&self.read()?)?)
    }
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Limits {
    raw_per_leg: u64,
    raw_per_outer: u64,
    raw_campaign: u64,
    retained_per_leg: u64,
    retained_per_outer: u64,
    retained_campaign: u64,
    cpu_seconds: f64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    schema: String,
    protocol: BoundFile,
    reference_config: BoundFile,
    passive: BTreeMap<String, BoundFile>,
    candidate_ledger: BoundFile,
    baseline: BTreeMap<String, BoundFile>,
    baseline_cache: Option<BoundFile>,
    total_outer: usize,
    total_candidates: usize,
    master_seed: u64,
    depletant_radius: f64,
    activity: f64,
    lambda: f64,
    envelope: GateOptions,
    limits: Limits,
    compiled_source_sha256: BTreeMap<String, String>,
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
#[derive(Deserialize, Serialize)]
struct Case {
    name: String,
    root: usize,
    child: usize,
    anchor: usize,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CachedRow {
    index: usize,
    passive_row_sha256: String,
    atlas: String,
    atlas_index: usize,
    case: Case,
    case_index: usize,
    attempt: usize,
    method: String,
    old: [Pose; 2],
    anchor_pose: Pose,
    proposal_status: String,
    candidate: Value,
    proposal_cpu_seconds: f64,
    standalone_proposal_cpu_seconds: f64,
    cloud_construction_cpu_seconds: f64,
    guidance_setup_cpu_seconds: f64,
    m: usize,
    guidance: Value,
    complete_log_correction: Value,
    contact_diagnostic_cpu_seconds: f64,
    raw_edge_draws: usize,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
struct LegProgress {
    gate: GateResult,
    /// Planned Poisson count is in gate.raw_points, even when its cap is exceeded.
    processed_points: u64,
    complete: bool,
}
#[derive(Debug, Serialize)]
struct PathFailure {
    reason: String,
    order: SingletonOrder,
    ordered_members: [usize; 2],
    intermediate_selected: [Pose; 2],
    completed_legs: Vec<GateResult>,
    failed_leg: usize,
    failed_progress: LegProgress,
}
struct Budget {
    limits: Limits,
    started: f64,
    raw: u64,
    retained: u64,
}
impl Budget {
    fn check_cpu(&self) -> Result<()> {
        ensure!(
            cpu_seconds() - self.started <= self.limits.cpu_seconds,
            "fatal campaign CPU budget exceeded"
        );
        Ok(())
    }
}

/// Same draw order and arithmetic as RigidSubset::sample_envelope. Only fatal
/// resource guards are added. A partial cloud never supplies an MH factor.
fn bounded_leg(
    gate: &RigidSubset<'_>,
    rng: &mut StdRng,
    lambda: f64,
    z: f64,
    opts: GateOptions,
    raw_cap: u64,
    retained_cap: u64,
    budget: &Budget,
    progress: &mut LegProgress,
) -> Result<GateResult> {
    budget.check_cpu()?;
    if z == 0. {
        progress.complete = true;
        return Ok(GateResult::default());
    }
    let envelope = gate.envelope(opts)?;
    progress.gate = GateResult {
        envelope_volume: envelope.volume,
        retained_cells: envelope.cells.len(),
        created_cells: envelope.created,
        ..Default::default()
    };
    budget.check_cpu()?;
    if envelope.volume == 0. {
        progress.complete = true;
        return Ok(progress.gate);
    }
    let mean = (lambda + z) * envelope.volume;
    ensure!(mean.is_finite() && mean < 9e15, "unsupported Poisson mean");
    let number = Poisson::<f64>::new(mean)?.sample(rng) as u64;
    progress.gate.raw_points = number;
    ensure!(number <= raw_cap, "fatal planned raw-point budget exceeded");
    for index in 0..number {
        if index % 1024 == 0 {
            budget.check_cpu()?;
        }
        let target = rng.random::<f64>() * envelope.volume;
        let k = envelope
            .cumulative
            .partition_point(|&v| v <= target)
            .min(envelope.cells.len() - 1);
        let cell = envelope.cells[k];
        let point =
            std::array::from_fn(|j| cell.lo[j] + rng.random::<f64>() * (cell.hi[j] - cell.lo[j]));
        let (old, new) = gate.overlap_indicators(point);
        if old && !new {
            progress.gate.lost += 1;
        }
        if new && !old && rng.random::<f64>() < lambda / (lambda + z) {
            progress.gate.gained += 1;
        }
        progress.processed_points = index + 1;
        progress.gate.retained_points = progress.gate.gained + progress.gate.lost;
        ensure!(
            progress.gate.retained_points <= retained_cap,
            "fatal retained-point budget exceeded"
        );
    }
    let ratio = z / lambda;
    let coefficient = if ratio.is_finite() {
        ratio.ln_1p()
    } else {
        (lambda + z).ln() - lambda.ln()
    };
    progress.gate.log_weight =
        coefficient * (progress.gate.gained as f64 - progress.gate.lost as f64);
    progress.complete = true;
    Ok(progress.gate)
}

fn aggregate(a: GateResult, b: GateResult) -> Result<GateResult> {
    let result = GateResult {
        gained: a.gained.checked_add(b.gained).context("gained overflow")?,
        lost: a.lost.checked_add(b.lost).context("lost overflow")?,
        raw_points: a
            .raw_points
            .checked_add(b.raw_points)
            .context("raw overflow")?,
        retained_points: a
            .retained_points
            .checked_add(b.retained_points)
            .context("retained overflow")?,
        retained_cells: a
            .retained_cells
            .checked_add(b.retained_cells)
            .context("cell overflow")?,
        created_cells: a
            .created_cells
            .checked_add(b.created_cells)
            .context("created overflow")?,
        envelope_volume: a.envelope_volume + b.envelope_volume,
        log_weight: a.log_weight + b.log_weight,
    };
    ensure!(
        result.envelope_volume.is_finite() && result.log_weight.is_finite(),
        "nonfinite path aggregate"
    );
    Ok(result)
}

/// Production call uses None: exactly one fair order coin. Explicit order is
/// only used by tiny reference tests and is unavailable through the config.
fn bounded_path(
    tree: &SphereTree,
    state: &[Pose],
    members: [usize; 2],
    proposed: [Pose; 2],
    path: &SingletonPath<'_>,
    rng: &mut StdRng,
    rd: f64,
    lambda: f64,
    z: f64,
    opts: GateOptions,
    budget: &mut Budget,
    explicit_order: Option<SingletonOrder>,
) -> std::result::Result<SingletonPathResult, PathFailure> {
    let order = explicit_order.unwrap_or_else(|| {
        if rng.random::<bool>() {
            SingletonOrder::SecondThenFirst
        } else {
            SingletonOrder::FirstThenSecond
        }
    });
    let ordered_members = path.ordered_members(order);
    let intermediate_selected = path.intermediate_selected(order);
    let indices = if order == SingletonOrder::FirstThenSecond {
        [0, 1]
    } else {
        [1, 0]
    };
    let mut intermediate = state.to_vec();
    intermediate[members[indices[0]]] = proposed[indices[0]];
    let mut completed = Vec::new();
    let mut outer_raw = 0u64;
    let mut outer_retained = 0u64;
    for (leg, &i) in indices.iter().enumerate() {
        let mut progress = LegProgress::default();
        let result = (|| -> Result<GateResult> {
            let source = if leg == 0 { state } else { &intermediate };
            let gate = RigidSubset::new(tree, source, &[members[i]], members[i], proposed[i], rd)?;
            let raw_cap = budget
                .limits
                .raw_per_leg
                .min(budget.limits.raw_per_outer - outer_raw)
                .min(budget.limits.raw_campaign - budget.raw);
            let retained_cap = budget
                .limits
                .retained_per_leg
                .min(budget.limits.retained_per_outer - outer_retained)
                .min(budget.limits.retained_campaign - budget.retained);
            bounded_leg(
                &gate,
                rng,
                lambda,
                z,
                opts,
                raw_cap,
                retained_cap,
                budget,
                &mut progress,
            )
        })();
        match result {
            Ok(gate) => {
                outer_raw += gate.raw_points;
                outer_retained += gate.retained_points;
                budget.raw += gate.raw_points;
                budget.retained += gate.retained_points;
                completed.push(gate);
            }
            Err(error) => {
                return Err(PathFailure {
                    reason: format!("{error:#}"),
                    order,
                    ordered_members,
                    intermediate_selected,
                    completed_legs: completed,
                    failed_leg: leg,
                    failed_progress: progress,
                });
            }
        }
    }
    let legs = [completed[0], completed[1]];
    let aggregate = aggregate(legs[0], legs[1]).map_err(|error| PathFailure {
        reason: format!("{error:#}"),
        order,
        ordered_members,
        intermediate_selected,
        completed_legs: completed,
        failed_leg: 2,
        failed_progress: LegProgress::default(),
    })?;
    Ok(SingletonPathResult {
        order,
        ordered_members,
        intermediate_selected,
        legs,
        aggregate,
    })
}

fn seed(master: u64, ledger_sha: &str, row: &CachedRow, role: &str) -> u64 {
    let text = format!(
        "auxiliary-overlap-physical-v1/{master}/{ledger_sha}/{}/{}/{}/{}/{role}",
        row.atlas_index, row.case_index, row.attempt, row.method
    );
    u64::from_str_radix(&hash_bytes(text.as_bytes())[..16], 16).unwrap()
}
fn log_value(value: &Value) -> Result<f64> {
    if value.as_str() == Some("-inf") {
        return Ok(f64::NEG_INFINITY);
    }
    let x = value.as_f64().context("missing/nonfinite log density")?;
    ensure!(x.is_finite(), "invalid log density");
    Ok(x)
}
fn log_json(value: f64) -> Value {
    if value == f64::NEG_INFINITY {
        json!("-inf")
    } else {
        json!(value)
    }
}

/// Revalidate the passive checked accessor using its retained count trace.
/// The original typed outcome is Serialize-only, so this replay consumes its
/// exact frozen JSON value, not a newly generated or reconditioned proposal.
fn checked_correction(cached: &CachedRow) -> Result<(f64, f64, f64)> {
    ensure!(!cached.candidate.is_null(), "no candidate correction");
    ensure!(
        (cached.method == "m1" && cached.m == 1)
            || (cached.method == "m4" && cached.m == 4)
            || (cached.method == "unguided" && cached.m == 0),
        "method/m differs"
    );
    let d = &cached.candidate["diagnostics"];
    let full = log_value(&d["log_reverse_forward"])?;
    let old_f = log_value(&d["full_old_log_density"])?;
    let new_f = log_value(&d["full_new_log_density"])?;
    ensure!(
        new_f.is_finite()
            && full == old_f - new_f
            && d["selection_log_reverse_forward"] == 0.
            && d["log_tree_coordinate_jacobian"] == 0.,
        "cached full-F correction differs"
    );
    let g = &cached.guidance;
    if cached.m == 0 {
        ensure!(g.is_null(), "unguided proposal has an auxiliary trace");
        ensure!(
            full == log_value(&cached.complete_log_correction)?,
            "unguided correction differs"
        );
        return Ok((full, 0., full));
    }
    let point_count = g["point_count"].as_u64().context("point count")?;
    let old = g["old_count"].as_u64().context("old count")?;
    let new = g["new_count"].as_u64().context("new count")?;
    let threshold = g["threshold"].as_u64().context("threshold")?;
    let draws = g["integer_draws"].as_array().context("threshold trace")?;
    ensure!(
        point_count <= (1u64 << 53) - 1
            && old <= point_count
            && new <= point_count
            && threshold <= old
            && threshold <= new
            && g["m"] == cached.m
            && draws.len() == cached.m,
        "invalid guidance support"
    );
    let values = draws
        .iter()
        .map(|x| x.as_u64().context("integer trace value"))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        values.iter().all(|&x| x <= old) && values.iter().max() == Some(&threshold),
        "invalid threshold trace"
    );
    let auxiliary = log_value(&g["aux_log_correction"])?;
    let expected = (cached.m as f64) * ((old as f64).ln_1p() - (new as f64).ln_1p());
    ensure!(
        auxiliary.is_finite() && auxiliary == expected,
        "cached auxiliary correction differs"
    );
    let complete = full + auxiliary;
    ensure!(
        (complete.is_finite() || complete == f64::NEG_INFINITY)
            && complete == log_value(&cached.complete_log_correction)?,
        "complete correction omitted or duplicated auxiliary term"
    );
    Ok((full, auxiliary, complete))
}

fn validate_baseline(plan: &Plan) -> Result<()> {
    let mut data = BTreeMap::new();
    for (key, value) in &plan.baseline {
        data.insert(key.clone(), value.read()?);
    }
    let review: Value = serde_json::from_slice(
        data.get("completed-review.json")
            .context("missing baseline review")?,
    )?;
    ensure!(
        review["complete"] == true && review["passed"] == true,
        "baseline physical audit not reviewed"
    );
    let analysis = data
        .get("analysis.json")
        .context("missing baseline analysis")?;
    ensure!(
        hash_bytes(analysis) == "9b7fc8916c8f3e32a15dc65c2b9f025488e39b7b4de44179b9bf069a82999c8b",
        "baseline physical analysis differs"
    );
    for (name, bytes) in &data {
        if name != "completed-review.json" {
            ensure!(
                review["output_hashes"][name] == hash_bytes(bytes),
                "unreviewed baseline file: {name}"
            );
        }
    }
    let ledger = data
        .get("execution/attempts.jsonl")
        .context("missing baseline decisions")?;
    let mut selected = Vec::new();
    let mut count = 0;
    for line in ledger.split_inclusive(|&x| x == b'\n') {
        let row: Value = serde_json::from_slice(line)?;
        if row["cached"]["method"] == "factorized" {
            selected.extend_from_slice(line);
            count += 1;
        }
    }
    ensure!(
        count == 768
            && selected
                == plan
                    .baseline_cache
                    .as_ref()
                    .context("missing baseline cache")?
                    .read()?,
        "baseline cache not exact 768-row subset"
    );
    Ok(())
}

fn line(out: &mut impl Write, value: &Value) -> Result<()> {
    serde_json::to_writer(&mut *out, value)?;
    out.write_all(b"\n")?;
    out.flush()?;
    Ok(())
}

fn execute(plan: &Plan, out: &mut impl Write) -> Result<Value> {
    let campaign_started = cpu_seconds();
    let width_mode = plan.schema == "fft-width-physical-reset-v1";
    ensure!(
        plan.total_outer == 1536
            && ((plan.schema == "auxiliary-overlap-physical-reset-v1"
                && plan.master_seed == 6100300301)
                || (width_mode && plan.master_seed == 6100300501)),
        "unknown plan/allocation"
    );
    ensure!(
        plan.depletant_radius == 1.4 && plan.activity == 0.0275 && plan.lambda == 1.76,
        "wrong growth diagnostic parameters"
    );
    ensure!(
        plan.envelope.max_cells == 255
            && plan.envelope.max_depth == 8
            && plan.envelope.min_width == 0.,
        "changed reference envelope"
    );
    plan.envelope.validate()?;
    let limits = plan.limits;
    ensure!(
        limits.raw_per_leg == 20_000_000
            && limits.raw_per_outer == 40_000_000
            && limits.raw_campaign == 2_000_000_000
            && limits.retained_per_leg == 20_000_000
            && limits.retained_per_outer == 40_000_000
            && limits.retained_campaign == 2_000_000_000
            && limits.cpu_seconds == 1200.,
        "invalid fatal resource caps"
    );
    let embedded: Value = serde_json::from_slice(BUNDLE)?;
    let hashes: BTreeMap<String, String> = embedded["files"]
        .as_object()
        .context("source bundle")?
        .iter()
        .map(|(name, entry)| {
            Ok((
                name.clone(),
                entry["sha256"].as_str().context("source SHA")?.into(),
            ))
        })
        .collect::<Result<_>>()?;
    ensure!(
        hashes == plan.compiled_source_sha256,
        "compiled source mismatch"
    );
    if width_mode {
        ensure!(
            plan.baseline.is_empty() && plan.baseline_cache.is_none(),
            "width replay must not redraw baseline"
        );
    } else {
        validate_baseline(plan)?;
    }
    let mut passive = BTreeMap::new();
    for (name, file) in &plan.passive {
        passive.insert(name.clone(), file.read()?);
    }
    let audit: Value = serde_json::from_slice(
        passive
            .get("analysis.json")
            .context("missing independent audit")?,
    )?;
    ensure!(
        audit["complete"] == true
            && audit["passed"] == true
            && audit["failures"].as_array().is_some_and(Vec::is_empty),
        "passive audit did not pass"
    );
    for (name, expected) in audit["input_hashes"]
        .as_object()
        .context("missing audit input hashes")?
    {
        let bytes = passive
            .get(name)
            .with_context(|| format!("missing audited passive file: {name}"))?;
        ensure!(
            expected == &json!(hash_bytes(bytes)),
            "changed audited passive file: {name}"
        );
    }
    let passive_config: Value = serde_json::from_slice(
        passive
            .get("config.json")
            .context("missing passive config")?,
    )?;
    ensure!(
        passive_config["reference_config"]["sha256"] == plan.reference_config.sha256,
        "physical source/reference differs from spectator-conditioned passive proposal"
    );
    ensure!(
        passive_config["schema"]
            == if width_mode {
                "fft-width-screen-v1"
            } else {
                "auxiliary-overlap-screen-v1"
            },
        "wrong passive experiment"
    );
    let passive_binding: Value = serde_json::from_slice(
        passive
            .get("binding.json")
            .context("missing passive binding")?,
    )?;
    for (name, key) in [
        ("config.json", "config_sha256"),
        ("protocol.json", "protocol_sha256"),
        ("source-bundle.json", "compiled_source_bundle_sha256"),
        ("example.rs", "example_source_sha256"),
    ] {
        ensure!(
            passive_binding[key]
                == hash_bytes(passive.get(name).context("missing bound passive file")?),
            "passive binding mismatch: {name}"
        );
    }
    let passive_ledger = passive
        .get("attempts.jsonl")
        .context("missing passive ledger")?;
    ensure!(
        audit["input_hashes"]["attempts.jsonl"] == hash_bytes(passive_ledger),
        "audit/ledger binding mismatch"
    );
    let passive_source: Value = serde_json::from_slice(
        passive
            .get("source-bundle.json")
            .context("missing passive source")?,
    )?;
    for name in [
        "src/docking.rs",
        "src/proposal.rs",
        "src/basin_involution.rs",
        "src/math.rs",
        "src/defensive_dimer_proposal.rs",
        "src/dimer_tree_proposal.rs",
        "src/capped_dimer.rs",
        "src/factorized_dimer.rs",
        "src/auxiliary_overlap_threshold.rs",
        "src/geometry.rs",
        "src/spherical.rs",
    ] {
        ensure!(
            passive_source["files"][name]["sha256"] == hashes[name],
            "cached proposal source differs: {name}"
        );
    }
    let reference = plan.reference_config.json()?;
    let bound = |name: &str| -> Result<Value> {
        serde_json::from_value::<BoundFile>(reference[name].clone())?.json()
    };
    let source = bound("source_config")?;
    let frame = bound("source_frame")?;
    let shape = bound("shape")?;
    ensure!(
        source["initial_poses"] == frame["poses"]
            && reference["depletant_radius"] == plan.depletant_radius
            && reference["activity"] == plan.activity,
        "source settings differ"
    );
    let state: Vec<Pose> = serde_json::from_value(source["initial_poses"].clone())?;
    ensure!(state.len() == 264, "wrong source state");
    if !width_mode {
        ensure!(
            plan.baseline
                .get("execution/source-state.json")
                .context("missing baseline state")?
                .json()?
                == serde_json::to_value(&state)?,
            "baseline and guided reset states differ"
        );
    }
    save(&plan.output.join("source-state.json"), &state)?;
    let tree = SphereTree::new(serde_json::from_value::<Shape>(shape.clone())?)?;
    let mut inflated: Shape = serde_json::from_value(shape)?;
    for atom in &mut inflated.atoms {
        atom.radius += plan.depletant_radius;
    }
    let exclusion = SphereTree::new(inflated)?;
    let wall_radius = reference["wall_radius"].as_f64().context("wall radius")?;
    let wall = Container::new(wall_radius, &tree)?;
    validate_state(&tree, &wall, &state)?;
    let cached = plan.candidate_ledger.read()?;
    let cached_rows = std::str::from_utf8(&cached)?
        .lines()
        .map(serde_json::from_str::<CachedRow>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let original_rows = std::str::from_utf8(passive_ledger)?
        .lines()
        .collect::<Vec<_>>();
    ensure!(
        cached_rows.len() == plan.total_outer && original_rows.len() == plan.total_outer,
        "lost outer rows"
    );
    let mut budget = Budget {
        limits,
        started: campaign_started,
        raw: 0,
        retained: 0,
    };
    let mut candidates = 0usize;
    let mut accepted = 0usize;
    let mut reused_cpu = 0.;
    let mut gate_cpu = 0.;
    let mut replay_cpu = 0.;
    for (index, cached) in cached_rows.iter().enumerate() {
        let started = cpu_seconds();
        let mut row = json!({"index":index,"cached":cached,"status":"in_progress","accepted":false,
            "source_state_sha256":hash_file(&plan.output.join("source-state.json"))?,
            "source_selected":cached.old,"proposed_selected":null,"retained_selected":cached.old,
            "retained_state":state,"gate":null,"gate_failure":null,"log_ratio":null,
            "full_f_correction":null,"auxiliary_correction":null,"q_correction":null});
        let result = (|| -> Result<()> {
            budget.check_cpu()?;
            ensure!(
                cached.index == index
                    && cached.passive_row_sha256 == hash_bytes(original_rows[index].as_bytes()),
                "changed cached row index/binding"
            );
            let original: Value = serde_json::from_str(original_rows[index])?;
            ensure!(
                if width_mode {
                    (cached.method == "unguided" && cached.m == 0)
                        || (cached.method == "m4" && cached.m == 4)
                } else {
                    (cached.method == "m1" && cached.m == 1)
                        || (cached.method == "m4" && cached.m == 4)
                },
                "unallocated arm"
            );
            ensure!(
                original["status"] == "completed"
                    && original["outcome"]["candidate"] == cached.candidate
                    && original["outcome"]["status"] == cached.proposal_status
                    && original["method"] == cached.method
                    && original["atlas"] == cached.atlas
                    && original["case"] == serde_json::to_value(&cached.case)?
                    && original["attempt"] == cached.attempt
                    && original["proposal_cpu_seconds"] == cached.proposal_cpu_seconds
                    && original["standalone_proposal_cpu_seconds"]
                        == cached.standalone_proposal_cpu_seconds
                    && original["cloud_construction_cpu_seconds"]
                        == cached.cloud_construction_cpu_seconds
                    && original["guidance_setup_cpu_seconds"] == cached.guidance_setup_cpu_seconds
                    && original["m"] == cached.m
                    && original["outcome"]["guidance"] == cached.guidance
                    && original["complete_log_correction"] == cached.complete_log_correction,
                "cached proposal metadata differs"
            );
            let members = [cached.case.root, cached.case.child];
            ensure!(
                members.iter().all(|&i| i < state.len()) && cached.case.anchor < state.len(),
                "invalid labels"
            );
            ensure!(
                cached.old == members.map(|i| state[i])
                    && cached.anchor_pose == state[cached.case.anchor],
                "source was not reset"
            );
            ensure!(
                cached.proposal_cpu_seconds.is_finite() && cached.proposal_cpu_seconds >= 0.,
                "invalid saved CPU"
            );
            ensure!(
                cached.standalone_proposal_cpu_seconds.is_finite()
                    && cached.standalone_proposal_cpu_seconds
                        == if cached.m == 0 {
                            cached.proposal_cpu_seconds
                        } else {
                            cached.cloud_construction_cpu_seconds
                                + cached.guidance_setup_cpu_seconds
                                + cached.proposal_cpu_seconds
                        },
                "invalid standalone saved CPU"
            );
            reused_cpu += cached.standalone_proposal_cpu_seconds;
            let gate_seed = seed(
                plan.master_seed,
                &plan.candidate_ledger.sha256,
                cached,
                "gate",
            );
            let mh_seed = seed(
                plan.master_seed,
                &plan.candidate_ledger.sha256,
                cached,
                "mh",
            );
            row["gate_seed"] = json!(gate_seed);
            row["mh_seed"] = json!(mh_seed);
            let mut mh_rng = StdRng::seed_from_u64(mh_seed);
            let uniform: f64 = mh_rng.sample(Open01);
            let log_uniform = uniform.ln();
            row["log_uniform"] = json!(log_uniform);
            row["mh_rng_after_fingerprint"] =
                json!(std::array::from_fn::<_, 4, _>(|_| mh_rng.random::<u64>()));
            if cached.candidate.is_null() {
                ensure!(
                    cached.proposal_status == "cap_exhausted"
                        || cached.proposal_status == "source_outside_domain"
                        || cached.proposal_status == "source_outside_contact",
                    "unclassified null proposal"
                );
                ensure!(
                    cached.complete_log_correction.is_null()
                        && cached.guidance["aux_log_correction"].is_null(),
                    "null has a correction"
                );
                row["status"] = json!("proposal_null");
                return Ok(());
            }
            ensure!(
                cached.proposal_status == "candidate",
                "candidate/status mismatch"
            );
            candidates += 1;
            let proposed: [Pose; 2] = [
                serde_json::from_value(cached.candidate["root"].clone())?,
                serde_json::from_value(cached.candidate["child"].clone())?,
            ];
            row["proposed_selected"] = json!(proposed);
            let (full_f, auxiliary, correction) = checked_correction(cached)?;
            row["full_f_correction"] = log_json(full_f);
            row["auxiliary_correction"] = log_json(auxiliary);
            row["q_correction"] = log_json(correction);
            let clock = cpu_seconds();
            let context = FixedDimerContext::new(
                &tree,
                &exclusion,
                &state,
                members,
                cached.case.anchor,
                Some(wall_radius),
                [0.; 3],
            )?;
            let source_valid = context.evaluate(cached.old)?;
            let final_valid = context.evaluate(proposed)?;
            row["source_feasibility"] = json!(source_valid);
            row["endpoint_feasibility"] = json!(final_valid);
            ensure!(
                source_valid.feasible() && final_valid.feasible(),
                "audited cached geometry changed; fatal, not rejection"
            );
            let path =
                SingletonPath::new(&tree, &state, &members, &proposed, plan.depletant_radius)?;
            ensure!(
                path.hard_valid(Some(&wall), [0.; 3]),
                "path endpoint check disagrees"
            );
            let mut gate_rng = StdRng::seed_from_u64(gate_seed);
            let sampled = bounded_path(
                &tree,
                &state,
                members,
                proposed,
                &path,
                &mut gate_rng,
                plan.depletant_radius,
                plan.lambda,
                plan.activity,
                plan.envelope,
                &mut budget,
                None,
            );
            let elapsed = cpu_seconds() - clock;
            gate_cpu += elapsed;
            row["gate_cpu_seconds"] = json!(elapsed);
            row["gate_rng_after_fingerprint"] =
                json!(std::array::from_fn::<_, 4, _>(|_| gate_rng.random::<u64>()));
            let sampled = match sampled {
                Ok(sampled) => sampled,
                Err(failure) => {
                    row["gate_failure"] = json!(failure);
                    anyhow::bail!("fatal gate/budget failure; partial cloud retained");
                }
            };
            row["gate"] = json!(sampled);
            let log_ratio = correction + sampled.aggregate.log_weight;
            ensure!(
                log_ratio.is_finite() || log_ratio == f64::NEG_INFINITY,
                "invalid composed log ratio"
            );
            let accept = log_uniform < log_ratio.min(0.);
            row["log_ratio"] = if log_ratio == f64::NEG_INFINITY {
                json!("-inf")
            } else {
                json!(log_ratio)
            };
            row["accepted"] = json!(accept);
            row["status"] = json!("physical_decision");
            if accept {
                accepted += 1;
                let mut retained = state.clone();
                retained[members[0]] = proposed[0];
                retained[members[1]] = proposed[1];
                row["retained_selected"] = json!(proposed);
                row["retained_state"] = json!(retained);
            }
            Ok(())
        })();
        let elapsed = cpu_seconds() - started;
        replay_cpu += elapsed;
        row["physical_replay_cpu_seconds"] = json!(elapsed);
        row["campaign_completed_raw_points"] = json!(budget.raw);
        row["campaign_completed_retained_points"] = json!(budget.retained);
        if let Err(error) = result {
            row["status"] = json!("fatal");
            row["fatal_error"] = json!(format!("{error:#}"));
            line(out, &row)?;
            return Err(error);
        }
        line(out, &row)?;
    }
    ensure!(candidates == plan.total_candidates, "lost cached candidate");
    Ok(
        json!({"outer_attempts":plan.total_outer,"candidates":candidates,"accepted":accepted,
        "proposal_nulls":plan.total_outer-candidates,"raw_points":budget.raw,"retained_points":budget.retained,
        "gate_cpu_seconds":gate_cpu,"saved_proposal_cpu_seconds":reused_cpu,
        "gate_plus_saved_proposal_cpu_seconds":gate_cpu+reused_cpu,"new_proposal_draws":0,
        "physical_replay_cpu_seconds":replay_cpu,"replay_plus_saved_proposal_cpu_seconds":replay_cpu+reused_cpu,
        "sequential_state_updates":0,"reset_state_decisions":plan.total_outer}),
    )
}

fn main() -> Result<()> {
    let clock = cpu_seconds();
    let args = Args::parse();
    let config = fs::read(&args.config)?;
    let binding_bytes = fs::read(&args.binding)?;
    let plan: Plan = serde_json::from_slice(&config)?;
    let binding: Binding = serde_json::from_slice(&binding_bytes)?;
    ensure!(
        binding.schema == "auxiliary-overlap-physical-binding-v1"
            && hash_bytes(&config) == binding.config_sha256
            && hash_bytes(SOURCE) == binding.example_source_sha256
            && hash_bytes(BUNDLE) == binding.compiled_source_bundle_sha256
            && hash_file(&std::env::current_exe()?)? == binding.executable_sha256,
        "physical executable/config binding differs"
    );
    let protocol = plan.protocol.read()?;
    ensure!(
        hash_bytes(&protocol) == binding.protocol_sha256,
        "protocol binding differs"
    );
    fs::create_dir(&plan.output).context("output exists or parent missing")?;
    for (name, bytes) in [
        ("config.json", config.as_slice()),
        ("binding.json", binding_bytes.as_slice()),
        ("protocol.json", protocol.as_slice()),
        ("source-bundle.json", BUNDLE),
        ("example.rs", SOURCE),
    ] {
        fs::write(plan.output.join(name), bytes)?;
    }
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(plan.output.join("attempts.jsonl"))?;
    let mut out = BufWriter::new(file);
    let result = execute(&plan, &mut out);
    out.flush()?;
    let whole_process_cpu = cpu_seconds() - clock;
    let summary = match &result {
        Ok(value) => json!({"complete":true,"result":value,
            "whole_process_plus_saved_proposal_cpu_seconds":whole_process_cpu+value["saved_proposal_cpu_seconds"].as_f64().unwrap()}),
        Err(error) => json!({"complete":false,"error":format!("{error:#}")}),
    };
    save(
        &plan.output.join("terminal.json"),
        &json!({"summary":summary,"cpu_seconds":whole_process_cpu,
         "attempts_sha256":hash_file(&plan.output.join("attempts.jsonl"))?}),
    )?;
    result.map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tetramer_mc::geometry::Atom;
    fn pose(x: f64) -> Pose {
        Pose {
            position: [x, 0., 0.],
            orientation: [1., 0., 0., 0.],
        }
    }
    fn tree() -> SphereTree {
        SphereTree::new(Shape {
            name: "bounded path sphere".into(),
            volume: 1.,
            atoms: vec![Atom {
                center: [0.; 3],
                radius: 0.2,
            }],
        })
        .unwrap()
    }
    fn budget() -> Budget {
        Budget {
            limits: Limits {
                raw_per_leg: 1_000_000,
                raw_per_outer: 2_000_000,
                raw_campaign: 10_000_000,
                retained_per_leg: 1_000_000,
                retained_per_outer: 2_000_000,
                retained_campaign: 10_000_000,
                cpu_seconds: 60.,
            },
            started: cpu_seconds(),
            raw: 0,
            retained: 0,
        }
    }
    fn opts() -> GateOptions {
        GateOptions {
            max_cells: 31,
            max_depth: 4,
            min_width: 0.,
        }
    }

    #[test]
    fn bounded_path_matches_unchanged_path_and_rng_for_orders_zero_and_empty() -> Result<()> {
        let tree = tree();
        let state = [pose(0.), pose(1.), pose(2.)];
        for (proposed, z) in [
            ([pose(-0.3), pose(1.2)], 1.5),
            ([pose(-0.3), pose(1.2)], 0.),
            ([state[0], state[1]], 1.5),
        ] {
            let path = SingletonPath::new(&tree, &state, &[0, 1], &proposed, 0.8)?;
            for order in [
                None,
                Some(SingletonOrder::FirstThenSecond),
                Some(SingletonOrder::SecondThenFirst),
            ] {
                for seed in [11, 29, 67] {
                    let mut a = StdRng::seed_from_u64(seed);
                    let mut b = StdRng::seed_from_u64(seed);
                    let expected = if let Some(order) = order {
                        path.sample_with_order(&mut a, 2., z, opts(), order)?
                    } else {
                        path.sample(&mut a, 2., z, opts())?
                    };
                    let actual = bounded_path(
                        &tree,
                        &state,
                        [0, 1],
                        proposed,
                        &path,
                        &mut b,
                        0.8,
                        2.,
                        z,
                        opts(),
                        &mut budget(),
                        order,
                    )
                    .map_err(|e| anyhow::anyhow!("{e:?}"))?;
                    assert_eq!(
                        serde_json::to_value(expected)?,
                        serde_json::to_value(actual)?
                    );
                    assert_eq!(a.random::<u64>(), b.random::<u64>());
                }
            }
        }
        Ok(())
    }

    #[test]
    fn fatal_caps_keep_planned_and_partial_counts_without_mh() -> Result<()> {
        let tree = tree();
        let state = [pose(0.), pose(1.), pose(2.)];
        let proposed = [pose(-0.3), pose(1.2)];
        let path = SingletonPath::new(&tree, &state, &[0, 1], &proposed, 0.8)?;
        let mut raw_budget = budget();
        raw_budget.limits.raw_per_leg = 0;
        let raw = bounded_path(
            &tree,
            &state,
            [0, 1],
            proposed,
            &path,
            &mut StdRng::seed_from_u64(11),
            0.8,
            200.,
            1.5,
            opts(),
            &mut raw_budget,
            None,
        )
        .unwrap_err();
        assert!(raw.reason.contains("planned raw-point"));
        assert!(raw.failed_progress.gate.raw_points > 0);
        assert_eq!(raw.failed_progress.processed_points, 0);
        assert!(!raw.failed_progress.complete);
        assert!(raw.completed_legs.is_empty());
        let mut retained_budget = budget();
        retained_budget.limits.retained_per_leg = 0;
        let retained = bounded_path(
            &tree,
            &state,
            [0, 1],
            proposed,
            &path,
            &mut StdRng::seed_from_u64(11),
            0.8,
            200.,
            1.5,
            opts(),
            &mut retained_budget,
            None,
        )
        .unwrap_err();
        assert!(retained.reason.contains("retained-point"));
        assert!(retained.failed_progress.processed_points > 0);
        assert_eq!(retained.failed_progress.gate.retained_points, 1);
        assert!(!retained.failed_progress.complete);
        assert!(
            retained.failed_progress.processed_points <= retained.failed_progress.gate.raw_points
        );
        assert!(serde_json::to_string(&retained)?.contains("failed_progress"));
        Ok(())
    }

    #[test]
    fn cached_rng_domains_cover_both_methods_without_collision() -> Result<()> {
        let mut seeds = std::collections::BTreeSet::new();
        for atlas in 0..3 {
            for case in 0..8 {
                for attempt in 0..32 {
                    for method in ["m1", "m4"] {
                        let row: CachedRow = serde_json::from_value(
                            json!({"index":0,"passive_row_sha256":"frozen","atlas":"a","atlas_index":atlas,
                "case":{"name":"c","root":0,"child":1,"anchor":2},"case_index":case,"attempt":attempt,"method":method,
                "old":[pose(0.),pose(1.)],"anchor_pose":pose(2.),"proposal_status":"cap_exhausted","candidate":null,
                "proposal_cpu_seconds":0.,"standalone_proposal_cpu_seconds":0.,"cloud_construction_cpu_seconds":0.,"guidance_setup_cpu_seconds":0.,
                "m":1,"guidance":null,"complete_log_correction":null,"contact_diagnostic_cpu_seconds":0.,"raw_edge_draws":64}),
                        )?;
                        for role in ["gate", "mh"] {
                            assert!(seeds.insert(seed(6100300301, "ledger", &row, role)));
                        }
                    }
                }
            }
        }
        assert_eq!(seeds.len(), 3072);
        Ok(())
    }

    #[test]
    fn explicit_zero_reverse_density_rejects_without_replacing_the_cloud() -> Result<()> {
        assert_eq!(log_value(&json!("-inf"))?, f64::NEG_INFINITY);
        for value in [Value::Null, json!("NaN"), json!("+inf"), json!("Infinity")] {
            assert!(log_value(&value).is_err());
        }
        let composed = log_value(&json!("-inf"))? + 4.25;
        assert_eq!(composed, f64::NEG_INFINITY);
        assert!(!(0.5_f64.ln() < composed.min(0.)));
        Ok(())
    }
    fn corrected_row() -> CachedRow {
        let auxiliary = 11_f64.ln() - 5_f64.ln();
        serde_json::from_value(json!({"index":0,"passive_row_sha256":"frozen","atlas":"a","atlas_index":0,
            "case":{"name":"c","root":0,"child":1,"anchor":2},"case_index":0,"attempt":0,"method":"m1",
            "old":[pose(0.),pose(1.)],"anchor_pose":pose(2.),"proposal_status":"candidate",
            "candidate":{"diagnostics":{"log_reverse_forward":-3.,"full_old_log_density":-9.,"full_new_log_density":-6.,
                "selection_log_reverse_forward":0.,"log_tree_coordinate_jacobian":0.}},
            "proposal_cpu_seconds":0.,"standalone_proposal_cpu_seconds":0.,"cloud_construction_cpu_seconds":0.,"guidance_setup_cpu_seconds":0.,
            "m":1,"guidance":{"point_count":20,"m":1,"old_count":10,"new_count":4,"threshold":1,"integer_draws":[1],"aux_log_correction":auxiliary},
            "complete_log_correction":-3.+auxiliary,"contact_diagnostic_cpu_seconds":0.,"raw_edge_draws":2})).unwrap()
    }

    #[test]
    fn auxiliary_is_added_exactly_once_and_invalid_support_is_fatal() -> Result<()> {
        let cached = corrected_row();
        let (full, aux, complete) = checked_correction(&cached)?;
        assert_eq!(full, -3.);
        assert!(aux > 0.);
        assert_eq!(complete, full + aux);
        for change in ["omitted", "doubled", "count", "threshold", "null", "aux"] {
            let mut value = corrected_row();
            match change {
                "omitted" => value.complete_log_correction = json!(full),
                "doubled" => value.complete_log_correction = json!(full + 2. * aux),
                "count" => value.guidance["new_count"] = json!(21),
                "threshold" => value.guidance["threshold"] = json!(5),
                "null" => value.complete_log_correction = Value::Null,
                _ => value.guidance["aux_log_correction"] = json!(aux + 0.1),
            }
            assert!(checked_correction(&value).is_err(), "{change}");
        }
        Ok(())
    }

    #[test]
    fn unguided_width_control_uses_full_density_without_auxiliary() -> Result<()> {
        let mut row = corrected_row();
        row.method = "unguided".into();
        row.m = 0;
        row.guidance = Value::Null;
        row.complete_log_correction = json!(-3.);
        assert_eq!(checked_correction(&row)?, (-3., 0., -3.));
        row.complete_log_correction = json!(-2.);
        assert!(checked_correction(&row).is_err());
        row.complete_log_correction = json!(-3.);
        row.guidance = json!({"m":0});
        assert!(checked_correction(&row).is_err());
        row.guidance = Value::Null;
        row.candidate["diagnostics"]["full_old_log_density"] = json!("-inf");
        row.candidate["diagnostics"]["log_reverse_forward"] = json!("-inf");
        row.complete_log_correction = json!("-inf");
        assert_eq!(
            checked_correction(&row)?,
            (f64::NEG_INFINITY, 0., f64::NEG_INFINITY)
        );
        Ok(())
    }

    #[test]
    fn zero_reverse_density_keeps_auxiliary_trace_and_rejects() -> Result<()> {
        let mut row = corrected_row();
        row.candidate["diagnostics"]["full_old_log_density"] = json!("-inf");
        row.candidate["diagnostics"]["log_reverse_forward"] = json!("-inf");
        row.complete_log_correction = json!("-inf");
        let (full, aux, complete) = checked_correction(&row)?;
        assert_eq!(full, f64::NEG_INFINITY);
        assert!(aux.is_finite());
        assert_eq!(complete, f64::NEG_INFINITY);
        Ok(())
    }
}
