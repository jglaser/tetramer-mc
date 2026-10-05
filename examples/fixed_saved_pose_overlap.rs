//! Two independent absolute-overlap clouds at explicitly supplied saved poses.
//! This is a fixed-panel diagnostic, not a pose sampler or a normalizer.
#[path = "support/context_source_geometry.rs"]
mod bridge;

use anyhow::{Context, Result, ensure};
use bridge::patch_geometry::{Limits, Work};
use bridge::{Asset, Inputs, Loaded, log_add, region_data, sha};
use clap::Parser;
use rand::{SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};
use tetramer_mc::{
    depletion::GateOptions,
    geometry::Environment,
    math::Pose,
    overlap_weight::{self, CloudLimits, CloudProgress, OverlapEnvelope, OverlapWeight},
    simulation::cpu_seconds,
};

const SCHEMA: &str = "fixed-saved-pose-overlap-v1";
const PANEL_SCHEMA: &str = "context-fixed-pose-overlap-panel-v1";

#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    out: PathBuf,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CloudBudget {
    raw_per_cloud: u64,
    raw_per_pose: u64,
    raw_total: u64,
    processed_per_cloud: u64,
    processed_per_pose: u64,
    processed_total: u64,
    callback_interval: u64,
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Tolerance {
    absolute: f64,
    relative: f64,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Config {
    schema: String,
    inputs: Inputs,
    panel: Asset,
    seed: u64,
    lambda: f64,
    activity: f64,
    envelope: GateOptions,
    envelope_tolerance: Tolerance,
    limits: Limits,
    cloud_limits: CloudBudget,
}

impl Config {
    fn validate(&self) -> Result<()> {
        ensure!(self.schema == SCHEMA, "Unknown saved-pose scorer schema");
        ensure!(
            self.lambda == 2.24 && self.activity == 0.035,
            "Changed fixed physical/cloud intensities"
        );
        self.envelope.validate()?;
        ensure!(
            self.envelope.max_cells == 255
                && self.envelope.max_depth == 8
                && self.envelope.min_width == 0.,
            "Changed saved envelope construction"
        );
        ensure!(
            self.envelope_tolerance.absolute.is_finite()
                && (0. ..=1e-9).contains(&self.envelope_tolerance.absolute)
                && self.envelope_tolerance.relative.is_finite()
                && (0. ..=1e-12).contains(&self.envelope_tolerance.relative),
            "Invalid or overly loose saved-envelope tolerance"
        );
        let c = &self.cloud_limits;
        ensure!(
            c.raw_per_cloud > 0
                && c.raw_per_pose > 0
                && c.raw_total > 0
                && c.processed_per_cloud > 0
                && c.processed_per_pose > 0
                && c.processed_total > 0
                && c.callback_interval > 0,
            "Invalid cloud budget"
        );
        Ok(())
    }
}

#[derive(Deserialize, Serialize)]
struct Panel {
    schema: String,
    entries: Vec<PanelEntry>,
    /// Frozen selector proofs/strata are authenticated, but never drive geometry.
    #[serde(flatten)]
    provenance: BTreeMap<String, Value>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PanelEntry {
    id: String,
    source_rows: Asset,
    ordinal: usize,
    pose: Pose,
    metadata: Value,
}

struct SavedRow {
    value: Value,
    line_sha256: String,
}

struct Journal {
    file: BufWriter<File>,
    next: u64,
}

impl Journal {
    fn new(path: &Path) -> Result<Self> {
        Ok(Self {
            file: BufWriter::new(OpenOptions::new().write(true).create_new(true).open(path)?),
            next: 0,
        })
    }

    fn emit(&mut self, mut row: Value) -> Result<()> {
        row["event_index"] = json!(self.next);
        serde_json::to_writer(&mut self.file, &row)?;
        self.file.write_all(b"\n")?;
        self.file.flush()?;
        self.next += 1;
        Ok(())
    }
}

fn save(path: &Path, value: &Value) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.flush()?;
    Ok(())
}

fn asset_bytes(asset: &Asset, hashes: &mut BTreeMap<String, String>) -> Result<Vec<u8>> {
    ensure!(
        asset.path.is_absolute(),
        "Saved assets require absolute paths"
    );
    let path = asset.path.canonicalize()?;
    let bytes = fs::read(&path)?;
    let digest = sha(&bytes);
    ensure!(
        digest == asset.sha256,
        "Saved asset hash changed: {}",
        path.display()
    );
    let name = path.display().to_string();
    if let Some(previous) = hashes.insert(name, digest.clone()) {
        ensure!(previous == digest, "Conflicting saved-asset binding");
    }
    Ok(bytes)
}

fn load_panel(
    asset: &Asset,
    hashes: &mut BTreeMap<String, String>,
) -> Result<(Panel, BTreeMap<PathBuf, Vec<SavedRow>>)> {
    let panel: Panel = serde_json::from_slice(&asset_bytes(asset, hashes)?)?;
    ensure!(
        panel.schema == PANEL_SCHEMA && !panel.entries.is_empty(),
        "Invalid fixed panel"
    );
    let mut ids = BTreeSet::new();
    let mut selected = BTreeSet::new();
    let mut rows = BTreeMap::new();
    for entry in &panel.entries {
        entry.pose.validate()?;
        ensure!(
            !entry.id.is_empty() && ids.insert(entry.id.clone()),
            "Repeated/empty pose ID"
        );
        let path = entry.source_rows.path.canonicalize()?;
        ensure!(
            selected.insert((path.clone(), entry.ordinal)),
            "Repeated saved draw in panel"
        );
        if !rows.contains_key(&path) {
            let bytes = asset_bytes(&entry.source_rows, hashes)?;
            let text = std::str::from_utf8(&bytes)?;
            ensure!(text.ends_with('\n'), "Incomplete saved JSONL file");
            let saved = text
                .lines()
                .map(|line| {
                    Ok(SavedRow {
                        value: serde_json::from_str(line)?,
                        line_sha256: sha(line.as_bytes()),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            rows.insert(path.clone(), saved);
        } else {
            ensure!(
                hashes.get(&path.display().to_string()) == Some(&entry.source_rows.sha256),
                "Inconsistent repeated saved-file hash"
            );
        }
        let saved = rows[&path]
            .get(entry.ordinal)
            .context("Saved ordinal out of range")?;
        validate_saved_entry(entry, &saved.value)?;
    }
    Ok((panel, rows))
}

fn validate_saved_entry(entry: &PanelEntry, saved: &Value) -> Result<()> {
    ensure!(
        saved["kind"] == "candidate"
            && saved["complete"] == true
            && saved["input"]["ordinal"].as_u64() == Some(entry.ordinal as u64)
            && saved["actual"]["physical_valid"] == true
            && saved["actual"]["wall_valid"] == true
            && saved["actual"]["core_valid"] == true
            && saved["physical_zero"] == false
            && saved["clouds"] == json!([])
            && saved["log_physical_contribution"].is_null()
            && saved["physical_weight_status"] == "not_estimated",
        "Panel entry is not a completed hard-valid geometry-only saved draw"
    );
    let pose: Pose = serde_json::from_value(saved["input"]["proposed_pose"].clone())?;
    ensure!(
        pose == entry.pose,
        "Supplied pose differs from authenticated saved draw"
    );
    ensure!(
        saved["region"].is_string()
            && saved["patches"].is_object()
            && saved["envelope"].is_object(),
        "Saved geometry is incomplete"
    );
    if let Some(original) = entry.metadata.get("original_row") {
        ensure!(original == saved, "Panel's original-row snapshot differs");
    }
    if let Some(log_q) = entry.metadata.get("log_q_balanced") {
        ensure!(
            log_q.as_f64().is_some_and(f64::is_finite),
            "Invalid echoed proposal metadata"
        );
    }
    Ok(())
}

fn cloud_seed(config_sha256: &str, seed: u64, pose_index: usize, cloud: usize) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"fixed-saved-pose-overlap-v1\0");
    digest.update(config_sha256.as_bytes());
    digest.update(seed.to_le_bytes());
    digest.update((pose_index as u64).to_le_bytes());
    digest.update(format!("cloud{cloud}").as_bytes());
    digest.finalize().into()
}

fn seed_hex(seed: [u8; 32]) -> String {
    seed.iter().map(|b| format!("{b:02x}")).collect()
}

#[derive(Default)]
struct Counts {
    poses_begun: usize,
    poses_completed: usize,
    clouds_begun: u64,
    clouds_completed: u64,
    raw: u64,
    processed: u64,
}

fn consume_cloud_counts(
    counts: &mut Counts,
    pose: &mut (u64, u64),
    p: &CloudProgress,
) -> Result<()> {
    let planned = p.planned_points.unwrap_or(0);
    let raw = counts
        .raw
        .checked_add(planned)
        .context("Total raw count overflow")?;
    let processed = counts
        .processed
        .checked_add(p.processed_points)
        .context("Total processed count overflow")?;
    let pose_raw = pose
        .0
        .checked_add(planned)
        .context("Pose raw count overflow")?;
    let pose_processed = pose
        .1
        .checked_add(p.processed_points)
        .context("Pose processed count overflow")?;
    counts.raw = raw;
    counts.processed = processed;
    *pose = (pose_raw, pose_processed);
    Ok(())
}

fn remaining_caps(b: &CloudBudget, counts: &Counts, pose: (u64, u64)) -> Result<CloudLimits> {
    Ok(CloudLimits {
        raw_points: b
            .raw_per_cloud
            .min(
                b.raw_per_pose
                    .checked_sub(pose.0)
                    .context("Pose raw cap exceeded")?,
            )
            .min(
                b.raw_total
                    .checked_sub(counts.raw)
                    .context("Total raw cap exceeded")?,
            ),
        processed_points: b
            .processed_per_cloud
            .min(
                b.processed_per_pose
                    .checked_sub(pose.1)
                    .context("Pose processed cap exceeded")?,
            )
            .min(
                b.processed_total
                    .checked_sub(counts.processed)
                    .context("Total processed cap exceeded")?,
            ),
        callback_interval: b.callback_interval,
    })
}

fn envelope_json(envelope: &OverlapEnvelope, labels: &[usize]) -> Value {
    json!({"lower_volume":envelope.lower_volume,"upper_volume":envelope.upper_volume(),
        "uncertain_volume":envelope.uncertain_volume,"retained_cells":envelope.cells.len(),
        "created_cells":envelope.created,"certified_cells":envelope.certified_cells,"fixed_labels":labels})
}

fn compare_envelope(actual: &Value, saved: &Value, tolerance: Tolerance) -> Result<()> {
    for name in ["lower_volume", "upper_volume", "uncertain_volume"] {
        let a = actual[name]
            .as_f64()
            .context("Missing reconstructed envelope volume")?;
        let b = saved[name]
            .as_f64()
            .context("Missing saved envelope volume")?;
        ensure!(
            a.is_finite()
                && b.is_finite()
                && a >= 0.
                && b >= 0.
                && (a - b).abs() <= tolerance.absolute + tolerance.relative * a.abs().max(b.abs()),
            "Saved envelope volume differs: {name}"
        );
    }
    for name in [
        "retained_cells",
        "created_cells",
        "certified_cells",
        "fixed_labels",
    ] {
        ensure!(
            actual[name] == saved[name],
            "Saved envelope structure differs: {name}"
        );
    }
    Ok(())
}

#[derive(Debug, Serialize)]
struct Score {
    overlap_volume: f64,
    z_overlap: f64,
    variance_estimate: f64,
    variance_upper: f64,
}

fn two_cloud_score(
    lower: f64,
    uncertain: f64,
    lambda: f64,
    z: f64,
    counts: [u64; 2],
) -> Result<Score> {
    ensure!(
        lower.is_finite()
            && lower >= 0.
            && uncertain.is_finite()
            && uncertain >= 0.
            && lambda.is_finite()
            && lambda > 0.
            && z.is_finite()
            && z >= 0.,
        "Invalid score inputs"
    );
    let sum = counts[0]
        .checked_add(counts[1])
        .context("Combined overlap count overflow")?;
    ensure!(
        uncertain > 0. || sum == 0,
        "Positive overlap count in a zero-volume envelope"
    );
    let overlap_volume = lower + sum as f64 / (2. * lambda);
    let z_overlap = z * overlap_volume;
    let variance_estimate = z * z * sum as f64 / (4. * lambda * lambda);
    let variance_upper = z * z * uncertain / (2. * lambda);
    ensure!(
        [overlap_volume, z_overlap, variance_estimate, variance_upper]
            .iter()
            .all(|x| x.is_finite()),
        "Unrepresentable score arithmetic"
    );
    Ok(Score {
        overlap_volume,
        z_overlap,
        variance_estimate,
        variance_upper,
    })
}

#[derive(Serialize)]
struct CompletedCloud {
    progress: CloudProgress,
    weight: OverlapWeight,
}

#[allow(clippy::too_many_arguments)]
fn sample_cloud(
    pose_index: usize,
    cloud: usize,
    seed: [u8; 32],
    environment: &Environment,
    pose: Pose,
    envelope: &OverlapEnvelope,
    lambda: f64,
    activity: f64,
    budget: &CloudBudget,
    counts: &mut Counts,
    pose_counts: &mut (u64, u64),
    events: &mut Journal,
    work: &Work,
) -> Result<CompletedCloud> {
    let limits = remaining_caps(budget, counts, *pose_counts)?;
    let mut progress = CloudProgress::default();
    events.emit(
        json!({"kind":"cloud_begun","pose_index":pose_index,"cloud":cloud,
        "seed_sha256":seed_hex(seed),"remaining_limits":limits}),
    )?;
    counts.clouds_begun += 1;
    let result = overlap_weight::sample_with_envelope_bounded(
        &mut StdRng::from_seed(seed),
        environment,
        pose,
        lambda,
        activity,
        envelope,
        limits,
        &mut progress,
        |event, progress| {
            events.emit(
                json!({"kind":"cloud_progress","pose_index":pose_index,"cloud":cloud,
                "event":event,"progress":progress}),
            )?;
            work.check()
        },
    );
    // Both success and failure consume the entire planned count exactly once;
    // failed prefixes are neither zero weights nor replacement proposals.
    consume_cloud_counts(counts, pose_counts, &progress)?;
    match result {
        Ok(weight) => {
            ensure!(
                progress.complete
                    && progress.log_weight == Some(weight.log_weight)
                    && progress.processed_points == weight.raw_points
                    && progress.overlap_points == weight.overlap_points,
                "Cloud terminal accounting differs"
            );
            counts.clouds_completed += 1;
            events.emit(
                json!({"kind":"cloud_complete","pose_index":pose_index,"cloud":cloud,
                "progress":progress,"weight":weight}),
            )?;
            Ok(CompletedCloud { progress, weight })
        }
        Err(error) => {
            events.emit(
                json!({"kind":"cloud_failed","pose_index":pose_index,"cloud":cloud,
                "progress":progress,"reason":format!("{error:#}")}),
            )?;
            Err(error.context("Fatal fixed-pose cloud; no retry or usable pose score"))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn score_pose(
    pose_index: usize,
    entry: &PanelEntry,
    saved: &SavedRow,
    loaded: &Loaded,
    cfg: &Config,
    config_sha256: &str,
    counts: &mut Counts,
    events: &mut Journal,
    work: &mut Work,
) -> Result<Value> {
    work.begin()?;
    validate_saved_entry(entry, &saved.value)?;
    let source_identity = json!({"source_rows":entry.source_rows,"ordinal":entry.ordinal,
        "line_sha256":saved.line_sha256,"input":saved.value["input"]});
    events.emit(
        json!({"kind":"saved_row_checked","pose_index":pose_index,"id":entry.id,
        "saved_row_identity":source_identity}),
    )?;
    let (wall, core) = loaded.geometry.validity(entry.pose, work)?;
    events.emit(json!({"kind":"geometry_complete","pose_index":pose_index,
        "wall_valid":wall,"core_valid":core}))?;
    ensure!(
        wall && core == Some(true),
        "Saved hard-valid pose no longer passes full geometry"
    );
    let (tokens, neighbors, indices) = loaded.geometry.patches(entry.pose, work)?;
    let (region, patches) = region_data(&tokens, &neighbors, &cfg.inputs.regions)?;
    events.emit(json!({"kind":"patches_complete","pose_index":pose_index,"region":region,"patches":patches}))?;
    ensure!(
        saved.value["region"] == region && saved.value["patches"] == patches,
        "Saved contact/region classification differs"
    );
    let labels: Vec<_> = indices
        .iter()
        .map(|&i| loaded.geometry.context.bodies[i].label)
        .collect();
    let environment = Environment {
        tree: &loaded.geometry.tree,
        fixed: indices.iter().map(|&i| loaded.geometry.fixed[i]).collect(),
        labels: labels.iter().map(|&i| (i, [0; 3])).collect(),
        rd: loaded.geometry.rd,
    };
    work.check()?;
    let envelope = OverlapEnvelope::build(&environment, entry.pose, cfg.envelope)?;
    work.check()?;
    let envelope_data = envelope_json(&envelope, &labels);
    compare_envelope(
        &envelope_data,
        &saved.value["envelope"],
        cfg.envelope_tolerance,
    )?;
    events.emit(
        json!({"kind":"envelope_complete","pose_index":pose_index,"envelope":envelope_data}),
    )?;
    let mut pose_counts = (0, 0);
    let mut clouds = Vec::new();
    for cloud in 0..2 {
        clouds.push(sample_cloud(
            pose_index,
            cloud,
            cloud_seed(config_sha256, cfg.seed, pose_index, cloud),
            &environment,
            entry.pose,
            &envelope,
            cfg.lambda,
            cfg.activity,
            &cfg.cloud_limits,
            counts,
            &mut pose_counts,
            events,
            work,
        )?);
    }
    let score = two_cloud_score(
        envelope.lower_volume,
        envelope.uncertain_volume,
        cfg.lambda,
        cfg.activity,
        [
            clouds[0].weight.overlap_points,
            clouds[1].weight.overlap_points,
        ],
    )?;
    let log_mean_positive_weight =
        log_add(clouds[0].weight.log_weight, clouds[1].weight.log_weight)? - 2f64.ln();
    Ok(
        json!({"kind":"pose_score","complete":true,"pose_index":pose_index,"id":entry.id,
        "pose":entry.pose,"saved_row_identity":source_identity,"metadata":entry.metadata,
        "log_q_balanced":entry.metadata.get("log_q_balanced"),
        "actual":{"wall_valid":wall,"core_valid":core,"physical_valid":true},
        "region":region,"patches":patches,"envelope":envelope_data,"clouds":clouds,"score":score,
        "log_mean_positive_weight":log_mean_positive_weight,
        "physical_weight_status":"fixed_pose_two_cloud_diagnostic_not_region_normalizer",
        "work":{"raw_points":pose_counts.0,"processed_points":pose_counts.1,
            "patch_node_visits":work.nodes,"patch_leaf_tests":work.leaves}}),
    )
}

fn run(
    args: &Args,
    cfg: &Config,
    digest: &str,
    counts: &mut Counts,
    events: &mut Journal,
    rows: &mut BufWriter<File>,
    work: &mut Work,
) -> Result<Value> {
    cfg.validate()?;
    events.emit(json!({"kind":"setup_begun"}))?;
    let loaded = bridge::load(&cfg.inputs)?;
    ensure!(
        loaded.lambda == cfg.lambda && loaded.z == cfg.activity && loaded.geometry.rd == 1.5,
        "Saved physical context differs from fixed scoring conditions"
    );
    let mut input_hashes = loaded.input_hashes.clone();
    let (panel, saved_rows) = load_panel(&cfg.panel, &mut input_hashes)?;
    work.check()?;
    save(
        &args.out.join("protocol.json"),
        &json!({"schema":SCHEMA,"config":cfg,"config_sha256":digest,
        "input_sha256":input_hashes,"panel_entries":panel.entries.len(),
        "source_bundle_sha256":sha(include_str!(concat!(env!("OUT_DIR"),"/source-bundle.json")).as_bytes()),
        "coordinate_frame":"Unchanged saved poses and fixed full atomic-wall context; no translation or rotation preprocessing.",
        "rng":"StdRng; SHA256(fixed-saved-pose-overlap-v1 NUL,configSHA ASCII,seed u64LE,pose_index u64LE,cloud0 or cloud1 ASCII).",
        "score_law":"K0+K1 ~ Poisson(2 lambda (O-L)); z_overlap=z L+z(K0+K1)/(2lambda). variance_estimate=z^2(K0+K1)/(4lambda^2), variance_upper=z^2 U/(2lambda), with U=uncertain_volume.",
        "positive_weight_law":"Each cloud log_weight=z L+K log(1+z/lambda); positive weight is unbiased for exp(zO). Its logarithm and log_mean_positive_weight are not unbiased estimates of zO.",
        "zero_counts":"A zero count yields zero estimated variance but not certainty; variance_upper remains positive when uncertain volume is nonzero. Exact Poisson intervals belong in the independent audit.",
        "failure":"All expectation identities refer to the uncapped Poisson law. A budget or geometry failure invalidates completion of the fixed panel; no conditional-success unbiasedness claim, retries, skipped poses or clipped counts.",
        "scope":"Fixed saved-pose diagnostic only. No pose proposal, basin fit, region normalizer, equilibrium occupancy or assembly conclusion."}),
    )?;
    for (pose_index, entry) in panel.entries.iter().enumerate() {
        counts.poses_begun += 1;
        events.emit(
            json!({"kind":"pose_begun","pose_index":pose_index,"id":entry.id,"pose":entry.pose,
            "source_rows":entry.source_rows,"ordinal":entry.ordinal}),
        )?;
        let saved = &saved_rows[&entry.source_rows.path.canonicalize()?][entry.ordinal];
        match score_pose(
            pose_index, entry, saved, &loaded, cfg, digest, counts, events, work,
        ) {
            Ok(output) => {
                serde_json::to_writer(&mut *rows, &output)?;
                rows.write_all(b"\n")?;
                rows.flush()?;
                counts.poses_completed += 1;
                events
                    .emit(json!({"kind":"pose_complete","pose_index":pose_index,"id":entry.id}))?;
            }
            Err(error) => {
                events.emit(
                    json!({"kind":"pose_failed","pose_index":pose_index,"id":entry.id,
                    "reason":format!("{error:#}")}),
                )?;
                return Err(error);
            }
        }
    }
    ensure!(
        counts.poses_completed == panel.entries.len()
            && counts.poses_begun == panel.entries.len()
            && counts.clouds_begun == 2 * panel.entries.len() as u64
            && counts.clouds_completed == counts.clouds_begun,
        "Incomplete fixed panel"
    );
    for (path, expected) in &input_hashes {
        ensure!(
            sha(&fs::read(path)?) == *expected,
            "Bound input changed during scoring: {path}"
        );
    }
    Ok(
        json!({"schema":SCHEMA,"complete":true,"passed":true,"panel_entries":panel.entries.len(),
        "new_poses_generated":0,"normalizer_estimated":false,"retries":0,"replacements":0,
        "physical_weight_status":"fixed_pose_two_cloud_diagnostic_not_region_normalizer"}),
    )
}

fn main() -> Result<()> {
    let args = Args::parse();
    fs::create_dir(&args.out).context("Output must be fresh")?;
    let mut events = Journal::new(&args.out.join("events.jsonl"))?;
    let mut rows = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(args.out.join("rows.jsonl"))?,
    );
    let mut counts = Counts::default();
    let started = cpu_seconds();
    let wall = std::time::Instant::now();
    let result = (|| {
        let bytes = fs::read(&args.config)?;
        let cfg: Config = serde_json::from_slice(&bytes)?;
        let mut work = Work::new(cfg.limits.clone())?;
        let mut result = run(
            &args,
            &cfg,
            &sha(&bytes),
            &mut counts,
            &mut events,
            &mut rows,
            &mut work,
        )?;
        result["patch_node_visits"] = json!(work.total_nodes);
        result["patch_leaf_tests"] = json!(work.total_leaves);
        Ok::<_, anyhow::Error>(result)
    })();
    let mut report = match &result {
        Ok(value) => value.clone(),
        Err(error) => {
            let _ = events.emit(json!({"kind":"fatal","reason":format!("{error:#}"),
                "poses_begun":counts.poses_begun,"poses_completed":counts.poses_completed,
                "raw_points":counts.raw,"processed_points":counts.processed}));
            json!({"schema":SCHEMA,"complete":false,"passed":false,"error":format!("{error:#}"),
                "physical_weight_status":"incomplete_unusable","prefix_preserved":true,
                "normalizer_estimated":false,"new_poses_generated":0,"retries":0,"replacements":0})
        }
    };
    report["poses_begun"] = json!(counts.poses_begun);
    report["poses_completed"] = json!(counts.poses_completed);
    report["clouds_begun"] = json!(counts.clouds_begun);
    report["clouds_completed"] = json!(counts.clouds_completed);
    report["raw_points"] = json!(counts.raw);
    report["processed_points"] = json!(counts.processed);
    report["journal_events"] = json!(events.next);
    report["cpu_seconds"] = json!(cpu_seconds() - started);
    report["wall_seconds"] = json!(wall.elapsed().as_secs_f64());
    save(&args.out.join("summary.json"), &report)?;
    result.map(|_| ())
}

#[cfg(test)]
mod panel_tests {
    use super::*;
    use tetramer_mc::geometry::{Atom, Placed, Shape, SphereTree};

    fn identity_pose() -> Pose {
        Pose {
            position: [0.; 3],
            orientation: [1., 0., 0., 0.],
        }
    }

    fn near(a: f64, b: f64) {
        assert!(
            (a - b).abs() < 1e-12 * (1. + a.abs() + b.abs()),
            "{a} != {b}"
        );
    }

    fn budget() -> CloudBudget {
        CloudBudget {
            raw_per_cloud: 1000,
            raw_per_pose: 1500,
            raw_total: 2000,
            processed_per_cloud: 900,
            processed_per_pose: 1300,
            processed_total: 1800,
            callback_interval: 4,
        }
    }

    fn work() -> Result<Work> {
        Work::new(Limits {
            cpu_seconds: 30.,
            wall_seconds: 60.,
            patch_node_visits_per_candidate: 10000,
            patch_leaf_tests_per_candidate: 10000,
            patch_node_visits_total: 10000,
            patch_leaf_tests_total: 10000,
        })
    }

    fn temp_dir(name: &str) -> Result<PathBuf> {
        let path = std::env::temp_dir().join(format!(
            "fixed-pose-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        fs::create_dir(&path)?;
        Ok(path)
    }

    #[test]
    fn two_cloud_score_and_positive_weights_are_distinct() -> Result<()> {
        let (lower, uncertain, lambda, z) = (3., 20., 2.24, 0.035);
        let counts = [4, 10];
        let score = two_cloud_score(lower, uncertain, lambda, z, counts)?;
        near(score.overlap_volume, lower + 14. / (2. * lambda));
        near(score.z_overlap, z * score.overlap_volume);
        near(
            score.variance_estimate,
            z * z * 14. / (4. * lambda * lambda),
        );
        near(score.variance_upper, z * z * uncertain / (2. * lambda));
        let logs = counts.map(|k| z * lower + k as f64 * (z / lambda).ln_1p());
        let log_mean = log_add(logs[0], logs[1])? - 2f64.ln();
        near(log_mean, ((logs[0].exp() + logs[1].exp()) / 2.).ln());
        assert!((score.z_overlap - log_mean).abs() > 1e-5);
        assert!((score.z_overlap - (logs[0] + logs[1]) / 2.).abs() > 1e-5);
        Ok(())
    }

    #[test]
    fn poisson_reference_confirms_linear_score_and_positive_weight_expectations() -> Result<()> {
        let (lower, overlap, lambda, z): (f64, f64, f64, f64) = (0.7, 2.1, 1.3, 0.4);
        let residual = overlap - lower;
        let mean_sum = 2. * lambda * residual;
        let mut probability = (-mean_sum).exp();
        let (mut total, mut score_mean, mut score_second, mut exponentiated_score_mean) =
            (0., 0., 0., 0.);
        for k in 0..120 {
            let s = two_cloud_score(lower, 5., lambda, z, [k, 0])?;
            total += probability;
            score_mean += probability * s.z_overlap;
            score_second += probability * s.z_overlap * s.z_overlap;
            exponentiated_score_mean += probability * s.z_overlap.exp();
            probability *= mean_sum / (k + 1) as f64;
        }
        near(total, 1.);
        near(score_mean, z * overlap);
        near(
            score_second - score_mean * score_mean,
            z * z * residual / (2. * lambda),
        );
        assert!(exponentiated_score_mean > (z * overlap).exp() + 0.01);
        let mean_cloud = lambda * residual;
        let mut probability = (-mean_cloud).exp();
        let mut positive_mean = 0.;
        for k in 0..120 {
            positive_mean += probability * (z * lower + k as f64 * (z / lambda).ln_1p()).exp();
            probability *= mean_cloud / (k + 1) as f64;
        }
        near(positive_mean, (z * overlap).exp());
        Ok(())
    }

    #[test]
    fn zero_counts_keep_uncertainty_bound_and_zero_volume_is_deterministic() -> Result<()> {
        let zero_hits = two_cloud_score(2., 10., 2.24, 0.035, [0, 0])?;
        near(zero_hits.overlap_volume, 2.);
        near(zero_hits.z_overlap, 0.07);
        assert_eq!(zero_hits.variance_estimate, 0.);
        assert!(zero_hits.variance_upper > 0.);
        let empty = two_cloud_score(2., 0., 2.24, 0.035, [0, 0])?;
        assert_eq!(empty.variance_upper, 0.);
        assert!(two_cloud_score(2., 0., 2.24, 0.035, [1, 0]).is_err());
        assert!(two_cloud_score(0., 1., 0., 0.035, [0, 0]).is_err());
        assert!(two_cloud_score(0., 1., 2.24, 0.035, [u64::MAX, 1]).is_err());
        Ok(())
    }

    #[test]
    fn source_identity_and_envelope_checks_reject_changed_saved_geometry() -> Result<()> {
        let envelope = json!({"lower_volume":2.,"upper_volume":12.,"uncertain_volume":10.,
            "retained_cells":3,"created_cells":7,"certified_cells":1,"fixed_labels":[16,217]});
        let tolerance = Tolerance {
            absolute: 1e-9,
            relative: 1e-12,
        };
        compare_envelope(&envelope, &envelope, tolerance)?;
        for key in [
            "lower_volume",
            "upper_volume",
            "uncertain_volume",
            "fixed_labels",
        ] {
            let mut changed = envelope.clone();
            changed[key] = if key == "fixed_labels" {
                json!([16, 56])
            } else {
                json!(100.)
            };
            assert!(compare_envelope(&changed, &envelope, tolerance).is_err());
        }
        let pose = identity_pose();
        let saved = json!({"kind":"candidate","complete":true,"input":{"ordinal":3,"proposed_pose":pose},
            "actual":{"wall_valid":true,"core_valid":true,"physical_valid":true},"physical_zero":false,
            "clouds":[],"log_physical_contribution":null,"physical_weight_status":"not_estimated",
            "region":"A_patch_complete","patches":{},"envelope":envelope});
        let mut entry = PanelEntry {
            id: "toy".into(),
            source_rows: Asset {
                path: "/synthetic-unused".into(),
                sha256: "0".repeat(64),
            },
            ordinal: 3,
            pose,
            metadata: json!({"original_row":saved,"log_q_balanced":-5.}),
        };
        validate_saved_entry(&entry, &saved)?;
        entry.pose.position[0] = 0.1;
        assert!(validate_saved_entry(&entry, &saved).is_err());
        entry.pose = pose;
        entry.metadata["original_row"]["region"] = json!("B");
        assert!(validate_saved_entry(&entry, &saved).is_err());
        Ok(())
    }

    #[test]
    fn whole_count_caps_and_failure_prefix_are_charged_without_score() -> Result<()> {
        let tree = SphereTree::new(Shape {
            name: "synthetic sphere".into(),
            volume: 0.,
            atoms: vec![Atom {
                center: [0.; 3],
                radius: 1.,
            }],
        })?;
        let pose = identity_pose();
        let environment = Environment {
            tree: &tree,
            fixed: vec![Placed::new(pose)],
            labels: vec![(0, [0; 3])],
            rd: 0.,
        };
        let envelope = OverlapEnvelope::build(
            &environment,
            pose,
            GateOptions {
                max_cells: 1,
                max_depth: 0,
                min_width: 0.,
            },
        )?;
        assert!(envelope.uncertain_volume > 0.);
        let mut limits = budget();
        limits.raw_per_cloud = 1;
        let mut counts = Counts::default();
        let mut pose_counts = (0, 0);
        let directory = temp_dir("failure")?;
        {
            let mut journal = Journal::new(&directory.join("events.jsonl"))?;
            let result = sample_cloud(
                0,
                0,
                [7; 32],
                &environment,
                pose,
                &envelope,
                8.,
                0.035,
                &limits,
                &mut counts,
                &mut pose_counts,
                &mut journal,
                &work()?,
            );
            assert!(result.is_err());
        }
        assert_eq!(counts.clouds_begun, 1);
        assert_eq!(counts.clouds_completed, 0);
        assert!(counts.raw > 1);
        assert_eq!(counts.processed, 0);
        assert_eq!(pose_counts, (counts.raw, 0));
        let records = fs::read_to_string(directory.join("events.jsonl"))?
            .lines()
            .map(serde_json::from_str::<Value>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        assert_eq!(records.first().unwrap()["kind"], "cloud_begun");
        let count = records
            .iter()
            .find(|r| r["event"] == "count_drawn")
            .unwrap();
        assert_eq!(count["progress"]["planned_points"], counts.raw);
        let last = records.last().unwrap();
        assert_eq!(last["kind"], "cloud_failed");
        assert_eq!(last["progress"]["complete"], false);
        assert!(last["progress"]["log_weight"].is_null());
        assert!(
            records
                .iter()
                .all(|r| r["kind"] != "cloud_complete" && r["kind"] != "pose_score")
        );
        fs::remove_dir_all(directory)?;
        Ok(())
    }

    #[test]
    fn remaining_budgets_and_role_seeds_preserve_exact_accounting() -> Result<()> {
        let budget = budget();
        let mut counts = Counts::default();
        let mut pose_counts = (0, 0);
        let progress = CloudProgress {
            begun: true,
            planned_points: Some(850),
            processed_points: 850,
            overlap_points: 12,
            complete: true,
            log_weight: Some(0.5),
        };
        consume_cloud_counts(&mut counts, &mut pose_counts, &progress)?;
        let caps = remaining_caps(&budget, &counts, pose_counts)?;
        assert_eq!(caps.raw_points, 650);
        assert_eq!(caps.processed_points, 450);
        counts.raw = 2001;
        assert!(remaining_caps(&budget, &counts, pose_counts).is_err());
        let cloud0 = cloud_seed("frozen-config", 53, 0, 0);
        assert_eq!(cloud0, cloud_seed("frozen-config", 53, 0, 0));
        assert_ne!(cloud0, cloud_seed("frozen-config", 53, 0, 1));
        assert_ne!(cloud0, cloud_seed("frozen-config", 53, 1, 0));
        assert_ne!(cloud0, cloud_seed("changed-panel-config", 53, 0, 0));
        Ok(())
    }

    #[test]
    fn deterministic_empty_clouds_complete_without_poisson_count() -> Result<()> {
        let tree = SphereTree::new(Shape {
            name: "empty environment".into(),
            volume: 0.,
            atoms: vec![Atom {
                center: [0.; 3],
                radius: 1.,
            }],
        })?;
        let pose = identity_pose();
        let environment = Environment {
            tree: &tree,
            fixed: vec![],
            labels: vec![],
            rd: 0.,
        };
        let envelope = OverlapEnvelope::build(
            &environment,
            pose,
            GateOptions {
                max_cells: 1,
                max_depth: 0,
                min_width: 0.,
            },
        )?;
        let directory = temp_dir("empty")?;
        let mut counts = Counts::default();
        let mut pose_counts = (0, 0);
        {
            let mut journal = Journal::new(&directory.join("events.jsonl"))?;
            for cloud in 0..2 {
                let value = sample_cloud(
                    0,
                    cloud,
                    cloud_seed("empty", 5, 0, cloud),
                    &environment,
                    pose,
                    &envelope,
                    2.24,
                    0.035,
                    &budget(),
                    &mut counts,
                    &mut pose_counts,
                    &mut journal,
                    &work()?,
                )?;
                assert!(value.progress.complete);
                assert_eq!(value.progress.planned_points, None);
                assert_eq!(value.weight.raw_points, 0);
                assert_eq!(value.weight.log_weight, 0.);
            }
        }
        assert_eq!(counts.clouds_completed, 2);
        assert_eq!(counts.raw, 0);
        assert_eq!(counts.processed, 0);
        fs::remove_dir_all(directory)?;
        Ok(())
    }

    #[test]
    fn panel_provenance_is_preserved_and_saved_file_hash_is_mandatory() -> Result<()> {
        let directory = temp_dir("bindings")?;
        let pose = identity_pose();
        let row = json!({"kind":"candidate","complete":true,"input":{"ordinal":0,"proposed_pose":pose},
            "actual":{"wall_valid":true,"core_valid":true,"physical_valid":true},"physical_zero":false,
            "clouds":[],"log_physical_contribution":null,"physical_weight_status":"not_estimated",
            "region":"A_patch_complete","patches":{},"envelope":{}});
        let row_path = directory.join("source.jsonl");
        let row_text = serde_json::to_string(&row)? + "\n";
        fs::write(&row_path, &row_text)?;
        let panel_value = json!({"schema":PANEL_SCHEMA,"entries":[{"id":"toy", "source_rows":{
            "path":row_path,"sha256":sha(row_text.as_bytes())},"ordinal":0,"pose":pose,
            "metadata":{"original_row":row,"log_q_balanced":-3.}}],
            "selection":{"salt":"frozen"},"proof":{"scope":"synthetic"},"prospective_scoring":{"clouds":2}});
        let panel_path = directory.join("panel.json");
        let panel_bytes = serde_json::to_vec(&panel_value)?;
        fs::write(&panel_path, &panel_bytes)?;
        let asset = Asset {
            path: panel_path,
            sha256: sha(&panel_bytes),
        };
        let mut hashes = BTreeMap::new();
        let (panel, rows) = load_panel(&asset, &mut hashes)?;
        assert_eq!(panel.entries.len(), 1);
        assert_eq!(panel.provenance["selection"]["salt"], "frozen");
        assert_eq!(rows[&row_path][0].value, row);
        assert_eq!(hashes.len(), 2);
        fs::write(&row_path, row_text + " ")?;
        assert!(load_panel(&asset, &mut BTreeMap::new()).is_err());
        fs::remove_dir_all(directory)?;
        Ok(())
    }
}
