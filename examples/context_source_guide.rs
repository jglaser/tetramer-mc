//! Conditional importance draws, not an adaptive Markov chain or assembly run.
//! q = .5 U + .25 G_context + .25 g_source in d^3t times normalized Haar.
//! Every draw is retained, including hard zeros. Never redraw until valid.
#[path = "support/context_source_geometry.rs"]
mod bridge;
use anyhow::{Context, Result, ensure};
use bridge::patch_geometry::{Limits, Work};
use bridge::{Inputs, Loaded, log_add, reconstructed_map_lower, region_data, relative, sha};
use clap::Parser;
use rand::{RngExt, SeedableRng, distr::Open01, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{BufWriter, Write},
    path::PathBuf,
};
use tetramer_mc::{
    basin_involution::{BasinPair, FixedBasinInvolution},
    depletion::GateOptions,
    geometry::Environment,
    math::*,
    overlap_weight::{self, CloudLimits, CloudProgress, OverlapEnvelope},
    proposal::GaussianComponentParameters,
    simulation::cpu_seconds,
};
const SCHEMA: &str = "context-source-guide-v1";
#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    out: PathBuf,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
enum Mode {
    Geometry,
    Clouds,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SourceChart {
    angular_length: f64,
    translation_sigma: f64,
    rotation_scale_deg: f64,
    covariance: [[f64; 6]; 6],
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CloudBudget {
    raw_per_cloud: u64,
    raw_per_candidate: u64,
    raw_total: u64,
    processed_per_cloud: u64,
    processed_per_candidate: u64,
    processed_total: u64,
    callback_interval: u64,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Config {
    schema: String,
    inputs: Inputs,
    identity: Value,
    population: u64,
    seed: u64,
    draws: usize,
    mode: Mode,
    mixture: [f64; 3],
    source_chart: SourceChart,
    envelope: GateOptions,
    limits: Limits,
    cloud_limits: CloudBudget,
}
impl Config {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == SCHEMA && self.draws > 0,
            "Invalid guide allocation"
        );
        ensure!(
            self.mixture == [0.5, 0.25, 0.25],
            "Changed proposal mixture"
        );
        self.envelope.validate()?;
        ensure!(
            self.envelope.max_cells == 255
                && self.envelope.max_depth == 8
                && self.envelope.min_width == 0.,
            "Changed envelope allocation"
        );
        let c = &self.cloud_limits;
        ensure!(
            c.callback_interval > 0
                && c.raw_per_cloud > 0
                && c.raw_per_candidate > 0
                && c.raw_total > 0
                && c.processed_per_cloud > 0
                && c.processed_per_candidate > 0
                && c.processed_total > 0,
            "Invalid cloud caps"
        );
        Ok(())
    }
}
struct Journal {
    file: BufWriter<File>,
    next: u64,
}
impl Journal {
    fn new(path: &std::path::Path) -> Result<Self> {
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
fn save(path: &std::path::Path, value: &Value) -> Result<()> {
    let mut f = OpenOptions::new().write(true).create_new(true).open(path)?;
    serde_json::to_writer_pretty(&mut f, value)?;
    f.write_all(b"\n")?;
    f.flush()?;
    Ok(())
}
fn seed(cfg: &Config, digest: &str, ordinal: usize, role: &str) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"context-source-guide-independent-v1\0");
    h.update(digest.as_bytes());
    h.update(cfg.seed.to_le_bytes());
    h.update(cfg.population.to_le_bytes());
    h.update((ordinal as u64).to_le_bytes());
    h.update(role.as_bytes());
    h.finalize().into()
}
fn seed_hex(seed: [u8; 32]) -> String {
    seed.iter().map(|b| format!("{b:02x}")).collect()
}
fn world(relative: Pose, anchor: Pose) -> Pose {
    let r = rotation(anchor.orientation);
    Pose {
        position: add(anchor.position, matvec(r, relative.position)),
        orientation: quaternion(matmul(r, rotation(relative.orientation))),
    }
}
fn source_parameters(source: Pose, spec: &SourceChart) -> Result<GaussianComponentParameters> {
    ensure!(
        spec.angular_length.is_finite()
            && spec.angular_length > 0.
            && spec.translation_sigma.is_finite()
            && spec.translation_sigma > 0.
            && spec.rotation_scale_deg.is_finite()
            && spec.rotation_scale_deg > 0.
            && spec.rotation_scale_deg < 180.,
        "Invalid chart scales"
    );
    let angular = spec.angular_length * (spec.rotation_scale_deg.to_radians() * 0.5).tan();
    for i in 0..6 {
        for j in 0..6 {
            let want = if i != j {
                0.
            } else if i < 3 {
                spec.translation_sigma.powi(2)
            } else {
                angular.powi(2)
            };
            let got = spec.covariance[i][j];
            ensure!(
                got.is_finite() && (got - want).abs() <= 2e-14 * want.abs().max(1e-20),
                "Source covariance differs from fixed width at {i},{j}"
            );
        }
    }
    source.validate()?;
    Ok(GaussianComponentParameters {
        anchor_position: source.position,
        anchor_rotation: rotation(source.orientation),
        mean: [0.; 6],
        covariance: spec.covariance,
        weight: 1.,
    })
}
fn one_chart(p: GaussianComponentParameters, ell: f64) -> Result<FixedBasinInvolution> {
    FixedBasinInvolution::new(
        vec![p],
        ell,
        0.,
        vec![BasinPair {
            first: 0,
            second: 0,
            weight: 1.,
        }],
    )
}
/// Same Gumbel-max categorical rule as production, with its own recorded role seed.
fn label_draw(rng: &mut StdRng, logs: &[f64]) -> Result<usize> {
    let mut best = f64::NEG_INFINITY;
    let mut selected = None;
    for (i, &w) in logs.iter().enumerate() {
        ensure!(w.is_finite(), "Nonfinite context prior");
        let u: f64 = Open01.sample(rng);
        let score = w - (-u.ln()).ln();
        if score > best {
            best = score;
            selected = Some(i);
        }
    }
    selected.context("Empty context prior")
}
fn branch(u: f64) -> Result<&'static str> {
    ensure!((0. ..1.).contains(&u), "Invalid component uniform");
    Ok(if u < 0.5 {
        "uniform"
    } else if u < 0.75 {
        "context"
    } else {
        "source"
    })
}
fn draw(
    cfg: &Config,
    digest: &str,
    ordinal: usize,
    loaded: &Loaded,
    source: &FixedBasinInvolution,
) -> Result<Value> {
    let mut roles = BTreeMap::new();
    for r in [
        "component",
        "label",
        "latent",
        "uniform",
        "cloud0",
        "cloud1",
    ] {
        roles.insert(r, seed_hex(seed(cfg, digest, ordinal, r)));
    }
    let u = StdRng::from_seed(seed(cfg, digest, ordinal, "component")).random::<f64>();
    let chosen = branch(u)?;
    let anchor = loaded.geometry.source.anchor_pose;
    let (mut label, mut latent, mut cube, mut normals) = (None, None, None, None);
    let pose = if chosen == "uniform" {
        let mut rng = StdRng::from_seed(seed(cfg, digest, ordinal, "uniform"));
        let v: [f64; 3] = std::array::from_fn(|_| rng.random::<f64>());
        let q: [f64; 4] = std::array::from_fn(|_| StandardNormal.sample(&mut rng));
        let length = q.iter().fold(0_f64, |a, x| a.hypot(*x));
        ensure!(length > 0. && length.is_finite(), "Invalid Haar normal");
        cube = Some(v);
        normals = Some(q);
        Pose {
            position: v.map(|x| (2. * x - 1.) * loaded.half),
            orientation: q.map(|x| x / length),
        }
    } else {
        let mut rng = StdRng::from_seed(seed(cfg, digest, ordinal, "latent"));
        let z: [f64; 6] = std::array::from_fn(|_| StandardNormal.sample(&mut rng));
        latent = Some(z);
        let p = if chosen == "source" {
            source.decode(0, z)?
        } else {
            let i = label_draw(
                &mut StdRng::from_seed(seed(cfg, digest, ordinal, "label")),
                &loaded.log_prior,
            )?;
            label = Some(i);
            let p = loaded.atlas.decode(i, z)?;
            if loaded.flags[i] {
                invert_relative_pose(p)
            } else {
                p
            }
        };
        world(p, anchor)
    };
    pose.validate()?;
    Ok(
        json!({"ordinal":ordinal,"identity":cfg.identity,"population":cfg.population,"branch":chosen,"component_uniform":u,
  "selected_virtual_label":label,"latent":latent,"cube_uniforms":cube,"quaternion_normals":normals,"proposed_pose":pose,"role_seeds":roles}),
    )
}
fn learned(loaded: &Loaded, r: Pose) -> Result<f64> {
    let mut g = f64::NEG_INFINITY;
    for i in 0..loaded.flags.len() {
        let p = if loaded.flags[i] {
            invert_relative_pose(r)
        } else {
            r
        };
        g = log_add(
            g,
            loaded.log_prior[i] + loaded.atlas.checked_log_density(i, p)?,
        )?;
    }
    Ok(g)
}
fn full_log_q(g: f64, s: f64, u: f64) -> Result<f64> {
    log_add(
        log_add(0.25f64.ln() + g, 0.25f64.ln() + s)?,
        0.5f64.ln() + u,
    )
}
fn density(loaded: &Loaded, source: &FixedBasinInvolution, pose: Pose) -> Result<Value> {
    let r = relative(pose, loaded.geometry.source.anchor_pose);
    let g = learned(loaded, r)?;
    let s = source.checked_log_density(0, r)?;
    let inside = pose.position.iter().all(|x| x.abs() <= loaded.half);
    let u = if inside {
        -3. * (2. * loaded.half).ln()
    } else {
        f64::NEG_INFINITY
    };
    let q = full_log_q(g, s, u)?;
    ensure!(q.is_finite(), "Invalid generated full Q");
    Ok(
        json!({"log_g":g.is_finite().then_some(g),"log_g_status":log_status(g),"log_source":s.is_finite().then_some(s),"log_source_status":log_status(s),
  "log_u":u.is_finite().then_some(u),"uniform_contains":inside,"log_q":q}),
    )
}
fn log_status(v: f64) -> &'static str {
    if v.is_finite() {
        "finite"
    } else {
        "negative_infinity"
    }
}
#[derive(Default)]
struct Counts {
    begun: usize,
    generated: usize,
    completed: usize,
    valid: usize,
    clouds_begun: u64,
    clouds_completed: u64,
    raw: u64,
    processed: u64,
    uncertain: f64,
}
#[derive(Default)]
struct Mass {
    count: usize,
    hard: Option<f64>,
    physical: Option<f64>,
}
fn add_log(total: &mut Option<f64>, value: f64) -> Result<()> {
    ensure!(value.is_finite(), "Nonfinite importance contribution");
    *total = Some(match *total {
        Some(old) => log_add(old, value)?,
        None => value,
    });
    Ok(())
}
fn masses_json(masses: &BTreeMap<String, Mass>, denominator: usize, mode: Mode) -> Value {
    json!(masses.iter().map(|(name,m)|json!({"region":name,"count":m.count,
 "unconditional_fraction":m.count as f64/denominator as f64,"log_hard_only_mass":m.hard.map(|x|x-(denominator as f64).ln()),
 "log_physical_mass":if mode==Mode::Clouds{m.physical.map(|x|x-(denominator as f64).ln())}else{None}})).collect::<Vec<_>>())
}
fn consume_cloud_counts(
    c: &mut Counts,
    candidate: &mut (u64, u64),
    p: &CloudProgress,
) -> Result<()> {
    let raw = p.planned_points.unwrap_or(0);
    c.raw = c.raw.checked_add(raw).context("Raw count overflow")?;
    c.processed = c
        .processed
        .checked_add(p.processed_points)
        .context("Processed count overflow")?;
    candidate.0 = candidate
        .0
        .checked_add(raw)
        .context("Candidate raw overflow")?;
    candidate.1 = candidate
        .1
        .checked_add(p.processed_points)
        .context("Candidate processed overflow")?;
    Ok(())
}
fn remaining_caps(b: &CloudBudget, c: &Counts, candidate: (u64, u64)) -> Result<CloudLimits> {
    Ok(CloudLimits {
        raw_points: b
            .raw_per_cloud
            .min(
                b.raw_per_candidate
                    .checked_sub(candidate.0)
                    .context("Candidate raw cap exceeded")?,
            )
            .min(
                b.raw_total
                    .checked_sub(c.raw)
                    .context("Raw campaign cap exceeded")?,
            ),
        processed_points: b
            .processed_per_cloud
            .min(
                b.processed_per_candidate
                    .checked_sub(candidate.1)
                    .context("Candidate processed cap exceeded")?,
            )
            .min(
                b.processed_total
                    .checked_sub(c.processed)
                    .context("Processed campaign cap exceeded")?,
            ),
        callback_interval: b.callback_interval,
    })
}
fn run(
    args: &Args,
    cfg: &Config,
    digest: &str,
    events: &mut Journal,
    rows: &mut BufWriter<File>,
    c: &mut Counts,
    work: &mut Work,
) -> Result<Value> {
    cfg.validate()?;
    events.emit(json!({"kind":"setup_begun"}))?;
    let mut loaded = bridge::load(&cfg.inputs)?;
    work.check()?;
    ensure!(
        cfg.source_chart.angular_length == loaded.angular_length,
        "Source/context angular lengths differ"
    );
    let parameters = source_parameters(
        relative(
            loaded.geometry.source.pose,
            loaded.geometry.source.anchor_pose,
        ),
        &cfg.source_chart,
    )?;
    let source = one_chart(parameters.clone(), cfg.source_chart.angular_length)?;
    loaded.manifest["source_chart"] = json!({"parameters":parameters,"angular_length":cfg.source_chart.angular_length,"reconstructed_map_lower":reconstructed_map_lower(&parameters)?,"covariance_origin":"explicit fixed isotropic scales; no ridge, fitting, or optimizer"});
    save(&args.out.join("chart-manifest.json"), &loaded.manifest)?;
    save(
        &args.out.join("protocol.json"),
        &json!({"schema":SCHEMA,"config":cfg,"config_sha256":digest,"input_sha256":loaded.input_hashes,
  "source_bundle_sha256":sha(include_str!(concat!(env!("OUT_DIR"),"/source-bundle.json")).as_bytes()),
  "source_pose":loaded.geometry.source.pose,"fixed_body_count":263,"density_measure":"translation volume times normalized rotational Haar",
  "rng":"StdRng; SHA256(domain,configSHA,seed,population,ordinal,role); separate component,label,latent,uniform,cloud0,cloud1; mode/config change creates fresh streams",
  "scope":"Source-informed conditional fixed-context importance control, no blind assembly claim; geometry mode estimates no depletion weights."}),
    )?;
    events.emit(json!({"kind":"source_begun"}))?;
    work.begin()?;
    ensure!(
        loaded
            .geometry
            .validity(loaded.geometry.source.pose, work)?
            == (true, Some(true)),
        "Original source hard-invalid"
    );
    let (tokens, neighbors, _) = loaded.geometry.patches(loaded.geometry.source.pose, work)?;
    let (name, data) = region_data(&tokens, &neighbors, &cfg.inputs.regions)?;
    ensure!(
        name == "A_patch_complete"
            && data["secondary_tokens"] == json!(cfg.inputs.regions.source_secondary_tokens),
        "Source reference differs"
    );
    events.emit(json!({"kind":"source_complete","patches":data}))?;
    let names = [
        "hard_invalid",
        "A_patch_0_0.25",
        "A_patch_0.25_0.5",
        "A_patch_0.5_0.75",
        "A_patch_0.75_1",
        "A_patch_complete",
        "B",
        "other_contact",
        "unbound",
    ];
    let mut masses: BTreeMap<String, Mass> = names
        .into_iter()
        .map(|s| (s.into(), Mass::default()))
        .collect();
    for ordinal in 0..cfg.draws {
        events.emit(json!({"kind":"candidate_begun","ordinal":ordinal}))?;
        c.begun += 1;
        work.begin()?;
        let input = draw(cfg, digest, ordinal, &loaded, &source)?;
        c.generated += 1;
        events.emit(json!({"kind":"candidate_generated","ordinal":ordinal,"input":input}))?;
        let pose: Pose = serde_json::from_value(input["proposed_pose"].clone())?;
        let den = density(&loaded, &source, pose)?;
        work.check()?;
        events.emit(json!({"kind":"density_complete","ordinal":ordinal,"density":den}))?;
        let (wall, core) = loaded.geometry.validity(pose, work)?;
        let valid = wall && core == Some(true);
        events.emit(json!({"kind":"geometry_complete","ordinal":ordinal,"wall_valid":wall,"core_valid":core}))?;
        let (mut patch_data, mut env_data, mut region) =
            (Value::Null, Value::Null, "hard_invalid".to_owned());
        let mut clouds = Vec::new();
        let mut log_contribution = None;
        let log_q = den["log_q"].as_f64().context("Missing Q")?;
        if valid {
            c.valid += 1;
            let (tokens, neighbors, indices) = loaded.geometry.patches(pose, work)?;
            let (name, data) = region_data(&tokens, &neighbors, &cfg.inputs.regions)?;
            region = name;
            patch_data = data;
            events.emit(json!({"kind":"patches_complete","ordinal":ordinal,"patches":patch_data,"region":region}))?;
            let labels: Vec<_> = indices
                .iter()
                .map(|&i| loaded.geometry.context.bodies[i].label)
                .collect();
            let env = Environment {
                tree: &loaded.geometry.tree,
                fixed: indices.iter().map(|&i| loaded.geometry.fixed[i]).collect(),
                labels: labels.iter().map(|&i| (i, [0; 3])).collect(),
                rd: loaded.geometry.rd,
            };
            work.check()?;
            let envelope = OverlapEnvelope::build(&env, pose, cfg.envelope)?;
            work.check()?;
            c.uncertain += envelope.uncertain_volume;
            ensure!(c.uncertain.is_finite(), "Envelope sum overflow");
            env_data = json!({"lower_volume":envelope.lower_volume,"upper_volume":envelope.upper_volume(),"uncertain_volume":envelope.uncertain_volume,"retained_cells":envelope.cells.len(),"created_cells":envelope.created,"certified_cells":envelope.certified_cells,"fixed_labels":labels});
            events
                .emit(json!({"kind":"envelope_complete","ordinal":ordinal,"envelope":env_data}))?;
            if cfg.mode == Mode::Clouds {
                let mut candidate_counts = (0, 0);
                for cloud in 0..2 {
                    let limits = remaining_caps(&cfg.cloud_limits, c, candidate_counts)?;
                    let mut progress = CloudProgress::default();
                    c.clouds_begun += 1;
                    let result = overlap_weight::sample_with_envelope_bounded(
                        &mut StdRng::from_seed(seed(
                            cfg,
                            digest,
                            ordinal,
                            &format!("cloud{cloud}"),
                        )),
                        &env,
                        pose,
                        loaded.lambda,
                        loaded.z,
                        &envelope,
                        limits,
                        &mut progress,
                        |event, progress| {
                            events.emit(json!({"kind":"cloud_progress","ordinal":ordinal,"cloud":cloud,"event":event,"progress":progress}))?;
                            work.check()
                        },
                    );
                    consume_cloud_counts(c, &mut candidate_counts, &progress)?;
                    match result {
                        Ok(weight) => {
                            ensure!(progress.complete, "Cloud failed terminal contract");
                            c.clouds_completed += 1;
                            events.emit(json!({"kind":"cloud_complete","ordinal":ordinal,"cloud":cloud,"weight":weight,"progress":progress}))?;
                            clouds.push(weight);
                        }
                        Err(e) => {
                            events.emit(json!({"kind":"cloud_failed","ordinal":ordinal,"cloud":cloud,"progress":progress,"reason":format!("{e:#}")}))?;
                            return Err(e.context(
                                "Fatal absolute cloud; no retry or usable candidate weight",
                            ));
                        }
                    }
                }
                log_contribution =
                    Some(log_add(clouds[0].log_weight, clouds[1].log_weight)? - 2f64.ln() - log_q);
            }
        }
        let m = masses.get_mut(&region).context("Unknown region")?;
        m.count += 1;
        if valid {
            add_log(&mut m.hard, -log_q)?;
            if let Some(x) = log_contribution {
                add_log(&mut m.physical, x)?;
            }
        }
        let output = json!({"kind":"candidate","complete":true,"input":input,"density":den,"actual":{"wall_valid":wall,"core_valid":core,"physical_valid":valid},
   "patches":patch_data,"region":if valid{Some(&region)}else{None},"envelope":env_data,"physical_zero":!valid,
   "clouds":clouds,"log_hard_only_contribution":valid.then_some(-log_q),"log_physical_contribution":log_contribution,
   "physical_weight_status":if cfg.mode==Mode::Geometry{"not_estimated"}else if valid{"two_cloud_arithmetic_mean"}else{"hard_zero"},
   "work":{"patch_node_visits":work.nodes,"patch_leaf_tests":work.leaves}});
        serde_json::to_writer(&mut *rows, &output)?;
        rows.write_all(b"\n")?;
        rows.flush()?;
        c.completed += 1;
        events.emit(json!({"kind":"candidate_complete","ordinal":ordinal}))?;
    }
    ensure!(
        c.completed == cfg.draws && c.begun == cfg.draws,
        "Incomplete importance allocation"
    );
    ensure!(
        c.clouds_completed
            == if cfg.mode == Mode::Clouds {
                2 * c.valid as u64
            } else {
                0
            },
        "Wrong cloud count"
    );
    Ok(
        json!({"schema":SCHEMA,"complete":true,"passed":true,"identity":cfg.identity,"mode":cfg.mode,"denominator":cfg.draws,"regions":masses_json(&masses,cfg.draws,cfg.mode),
  "expected_complete_allocation_raw_points":2.*loaded.lambda*c.uncertain,"expected_raw_scope":"Uncapped expected count for two fresh absolute clouds at every valid pose; not a realized count bound.",
  "physical_weight_status":if cfg.mode==Mode::Clouds{"estimated_not_convergence_certified"}else{"not_estimated"},"unconditional_denominator":true,
  "scope":"Conditional source-informed importance control; zero pocket hits do not imply zero physical mass or assembly failure."}),
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
    let mut c = Counts::default();
    let started = cpu_seconds();
    let wall = std::time::Instant::now();
    let result = (|| {
        let bytes = fs::read(&args.config)?;
        let cfg: Config = serde_json::from_slice(&bytes)?;
        let mut work = Work::new(cfg.limits.clone())?;
        let result = run(
            &args,
            &cfg,
            &sha(&bytes),
            &mut events,
            &mut rows,
            &mut c,
            &mut work,
        );
        if let Ok(mut report) = result {
            report["patch_node_visits"] = json!(work.total_nodes);
            report["patch_leaf_tests"] = json!(work.total_leaves);
            Ok(report)
        } else {
            result
        }
    })();
    let mut report = match &result {
        Ok(report) => report.clone(),
        Err(e) => {
            let _=events.emit(json!({"kind":"fatal","reason":format!("{e:#}"),"attempted_records":c.begun,"completed_records":c.completed,"raw_points":c.raw,"processed_points":c.processed}));
            json!({"schema":SCHEMA,"complete":false,"passed":false,"error":format!("{e:#}"),"physical_weight_status":"incomplete_unusable","prefix_preserved":true})
        }
    };
    report["attempted_records"] = json!(c.begun);
    report["completed_records"] = json!(c.completed);
    report["new_poses_generated"] = json!(c.generated);
    report["physical_valid_records"] = json!(c.valid);
    report["clouds_begun"] = json!(c.clouds_begun);
    report["clouds_completed"] = json!(c.clouds_completed);
    report["raw_points"] = json!(c.raw);
    report["processed_points"] = json!(c.processed);
    report["cpu_seconds"] = json!(cpu_seconds() - started);
    report["wall_seconds"] = json!(wall.elapsed().as_secs_f64());
    report["journal_events"] = json!(events.next);
    save(&args.out.join("summary.json"), &report)?;
    result.map(|_| ())
}

#[cfg(test)]
mod source_guide_tests {
    use super::*;
    use bridge::patch_geometry::PatchBvh;
    use bridge::{Asset, FixedContext, Geometry, Regions, SourceState};
    use tetramer_mc::{
        docking::{DockingMethod, DockingProposal},
        geometry::{Atom, Shape, SphereTree},
        proposal::FrozenRelativePoseProposal,
        spherical::Container,
    };
    fn pose(p: [f64; 3], c: [f64; 3]) -> Pose {
        Pose {
            position: p,
            orientation: quaternion(cayley(c)),
        }
    }
    fn spec() -> SourceChart {
        let ell = 1.3;
        let t = 0.2;
        let theta: f64 = 1.;
        let a = ell * (theta.to_radians() / 2.).tan();
        SourceChart {
            angular_length: ell,
            translation_sigma: t,
            rotation_scale_deg: theta,
            covariance: std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    if i != j {
                        0.
                    } else if i < 3 {
                        t * t
                    } else {
                        a * a
                    }
                })
            }),
        }
    }
    fn config() -> Config {
        let asset = || Asset {
            path: "/synthetic-unused".into(),
            sha256: "0".repeat(64),
        };
        Config {
            schema: SCHEMA.into(),
            inputs: Inputs {
                invocation_config: asset(),
                model: asset(),
                prior: asset(),
                patch_map: asset(),
                expected_virtual_branches: 2048,
                regions: Regions {
                    a_neighbors: vec![16, 217],
                    b_neighbors: vec![16, 56],
                    secondary_label: 217,
                    source_secondary_tokens: (0..16)
                        .map(|i| (77, 217, format!("p{i}"), "q".into()))
                        .collect(),
                    inclusion_boundaries: vec![0., 0.25, 0.5, 0.75, 1.],
                },
            },
            identity: json!({"synthetic":true}),
            population: 2,
            seed: 20261005,
            draws: 64,
            mode: Mode::Geometry,
            mixture: [0.5, 0.25, 0.25],
            source_chart: spec(),
            envelope: GateOptions {
                max_cells: 255,
                max_depth: 8,
                min_width: 0.,
            },
            limits: Limits {
                cpu_seconds: 30.,
                wall_seconds: 60.,
                patch_node_visits_per_candidate: 10000,
                patch_leaf_tests_per_candidate: 10000,
                patch_node_visits_total: 100000,
                patch_leaf_tests_total: 100000,
            },
            cloud_limits: CloudBudget {
                raw_per_cloud: 10,
                raw_per_candidate: 15,
                raw_total: 20,
                processed_per_cloud: 9,
                processed_per_candidate: 13,
                processed_total: 18,
                callback_interval: 2,
            },
        }
    }
    fn fixture() -> Result<(Loaded, FixedBasinInvolution, Config)> {
        let cfg = config();
        let center = pose([0.7, -0.2, 0.4], [0.2, -0.1, 0.3]);
        let anchor = pose([-0.5, 0.1, -0.3], [0.1, 0.3, -0.2]);
        let source_world = world(center, anchor);
        let parameters = source_parameters(center, &cfg.source_chart)?;
        let mut other = parameters.clone();
        other.anchor_position = [-0.2, 0.5, 0.3];
        other.anchor_rotation = cayley([-0.3, 0.1, 0.2]);
        let atlas = FixedBasinInvolution::new(
            vec![parameters.clone(), parameters.clone(), other],
            1.3,
            0.,
            vec![BasinPair {
                first: 0,
                second: 0,
                weight: 1.,
            }],
        )?;
        let shape = Shape {
            name: "toy".into(),
            volume: 0.,
            atoms: vec![Atom {
                center: [0.; 3],
                radius: 0.1,
            }],
        };
        let tree = SphereTree::new(shape.clone())?;
        let wall = Container::new(2., &tree)?;
        let patch = PatchBvh::new(&shape, &["p".into()])?;
        let geo = Geometry {
            tree,
            contact: SphereTree::new(shape)?,
            wall,
            patch,
            worlds: vec![],
            fixed: vec![],
            context: FixedContext {
                schema: "toy".into(),
                anchor_label: 16,
                excluded_moving_labels: vec![77],
                bodies: vec![],
            },
            source: SourceState {
                schema: "toy".into(),
                moving_label: 77,
                pose: source_world,
                anchor_label: 16,
                anchor_pose: anchor,
                boundary: "spherical".into(),
                spherical_wall_radius: 2.,
                coordinate_frame: "toy".into(),
                endpoint_sha256: "0".repeat(64),
            },
            rd: 0.,
        };
        Ok((
            Loaded {
                geometry: geo,
                atlas,
                flags: vec![false, true, false],
                log_prior: vec![0.275f64.ln(), 0.275f64.ln(), 0.45f64.ln()],
                half: 0.05,
                angular_length: 1.3,
                manifest: json!({}),
                input_hashes: BTreeMap::new(),
                z: 0.035,
                lambda: 2.24,
            },
            one_chart(parameters, 1.3)?,
            cfg,
        ))
    }
    fn near(a: f64, b: f64) {
        assert!(
            (a - b).abs() < 1e-9 * (1. + a.abs() + b.abs()),
            "{a} != {b}"
        );
    }
    fn same_pose(a: Pose, b: Pose) {
        for (x, y) in a.position.into_iter().zip(b.position) {
            near(x, y)
        }
        for (x, y) in rotation(a.orientation)
            .iter()
            .flatten()
            .zip(rotation(b.orientation).iter().flatten())
        {
            near(*x, *y)
        }
    }
    #[test]
    fn source_center_widths_and_haar_normalization() -> Result<()> {
        let cfg = config();
        let center = pose([1., -2., 0.5], [0.2, 0.1, -0.3]);
        let p = source_parameters(center, &cfg.source_chart)?;
        let map = one_chart(p.clone(), 1.3)?;
        same_pose(map.decode(0, [0.; 6])?, center);
        let log_det = 0.5 * (0..6).map(|i| p.covariance[i][i].ln()).sum::<f64>();
        let expected = -3. * (2. * std::f64::consts::PI).ln() - log_det
            + 3. * 1.3f64.ln()
            + 2. * std::f64::consts::PI.ln();
        near(map.checked_log_density(0, center)?, expected);
        let z = [0., 0., 0., 1., 0., 0.];
        let moved = map.decode(0, z)?;
        let q = quaternion(matmul(
            rotation(moved.orientation),
            transpose(rotation(center.orientation)),
        ));
        near(2. * q[1].atan2(q[0]), 1f64.to_radians());
        let mut invalid = spec();
        invalid.covariance[3][3] *= 1.001;
        assert!(source_parameters(center, &invalid).is_err());
        Ok(())
    }
    #[test]
    fn full_q_is_normalized_component_sum_with_haar_and_support() -> Result<()> {
        near(
            full_log_q(2f64.ln(), 3f64.ln(), 5f64.ln())?.exp(),
            0.5 * 5. + 0.25 * 2. + 0.25 * 3.,
        );
        near(
            full_log_q(2f64.ln(), 3f64.ln(), f64::NEG_INFINITY)?.exp(),
            1.25,
        );
        near(
            full_log_q(f64::NEG_INFINITY, f64::NEG_INFINITY, 5f64.ln())?.exp(),
            2.5,
        );
        assert_eq!(
            full_log_q(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY)?,
            f64::NEG_INFINITY
        );
        assert!(full_log_q(f64::NAN, 0., 0.).is_err());
        assert_eq!(branch(0.5)?, "context");
        assert_eq!(branch(0.75)?, "source");
        assert!(branch(1.).is_err());
        Ok(())
    }
    #[test]
    fn reduced_pair_table_preserves_actual_production_map_and_reciprocal_density() -> Result<()> {
        let spec = spec();
        let zero = [0.; 6];
        let raw = json!({"schema":"reciprocal-pose-mixture-v1","reciprocal_components":[true,false],"base_model":{"angular_length":1.3,"shape_sha256":"0".repeat(64),"coordinate_convention":"anchor-body-relative","anchors":[{"position":[0.7,-0.2,0.4],"rotation":cayley([0.2,-0.1,0.3])},{"position":[-0.2,0.5,0.3],"rotation":cayley([-0.3,0.1,0.2])}],"means":[zero,zero],"covariances":[spec.covariance,spec.covariance],"weights":[0.55,0.45]}});
        let model = FrozenRelativePoseProposal::from_json_str_open(
            &raw.to_string(),
            [4.; 3],
            0.5,
            &"0".repeat(64),
        )?;
        let branches = model.virtual_branches();
        let base = model.component_parameters();
        let expanded = branches
            .iter()
            .map(|b| {
                let mut p = base[b.component_index].clone();
                p.weight = b.weight;
                p
            })
            .collect();
        let sparse = FixedBasinInvolution::new(
            expanded,
            1.3,
            0.,
            vec![BasinPair {
                first: 0,
                second: 0,
                weight: 1.,
            }],
        )?;
        let production =
            DockingProposal::new(model, DockingMethod::PosteriorInvolution, 0., [0.; 3])?;
        let (dense, flags, _) = production.member_chart_parts();
        assert_eq!(flags, vec![false, true, false]);
        for i in 0..3 {
            for z in [[0.; 6], [0.1, -0.3, 0.5, 2., -1., 0.2]] {
                let a = dense.decode(i, z)?;
                let b = sparse.decode(i, z)?;
                assert_eq!(a, b);
                assert_eq!(
                    dense.checked_log_density(i, a)?.to_bits(),
                    sparse.checked_log_density(i, b)?.to_bits()
                );
                if flags[i] {
                    same_pose(invert_relative_pose(invert_relative_pose(a)), a)
                }
            }
        }
        Ok(())
    }
    #[test]
    fn generated_traces_reconstruct_without_conditioning_or_rng_role_reuse() -> Result<()> {
        let (loaded, source, cfg) = fixture()?;
        let mut kinds = BTreeSet::new();
        let mut outside = 0;
        for ordinal in 0..64 {
            let row = draw(&cfg, "held-synthetic", ordinal, &loaded, &source)?;
            assert_eq!(
                row,
                draw(&cfg, "held-synthetic", ordinal, &loaded, &source)?
            );
            let pose: Pose = serde_json::from_value(row["proposed_pose"].clone())?;
            let kind = row["branch"].as_str().unwrap();
            kinds.insert(kind.to_owned());
            if kind == "uniform" {
                let v: [f64; 3] = serde_json::from_value(row["cube_uniforms"].clone())?;
                for i in 0..3 {
                    near(pose.position[i], (2. * v[i] - 1.) * loaded.half);
                }
            } else {
                let z: [f64; 6] = serde_json::from_value(row["latent"].clone())?;
                let expected = if kind == "source" {
                    source.decode(0, z)?
                } else {
                    let i = row["selected_virtual_label"].as_u64().unwrap() as usize;
                    let x = loaded.atlas.decode(i, z)?;
                    if loaded.flags[i] {
                        invert_relative_pose(x)
                    } else {
                        x
                    }
                };
                same_pose(pose, world(expected, loaded.geometry.source.anchor_pose));
                if pose.position.iter().any(|x| x.abs() > loaded.half) {
                    outside += 1;
                }
            }
            let den = density(&loaded, &source, pose)?;
            assert!(den["log_q"].as_f64().unwrap().is_finite());
            assert_ne!(
                seed(&cfg, "held-synthetic", ordinal, "cloud0"),
                seed(&cfg, "held-synthetic", ordinal, "cloud1")
            );
            assert_ne!(
                seed(&cfg, "held-synthetic", ordinal, "latent"),
                seed(&cfg, "held-synthetic", ordinal, "label")
            );
        }
        assert_eq!(kinds.len(), 3);
        assert!(outside > 0);
        assert_ne!(
            seed(&cfg, "held-synthetic", 0, "latent"),
            seed(&cfg, "another-config-mode", 0, "latent")
        );
        Ok(())
    }
    #[test]
    fn physical_density_is_invariant_under_common_frame_change() -> Result<()> {
        let (loaded, source, _) = fixture()?;
        let pose = world(
            source.decode(0, [0.3, -0.2, 0.1, 0.5, -0.7, 0.2])?,
            loaded.geometry.source.anchor_pose,
        );
        let frame = pose_with_frame();
        let before = relative(pose, loaded.geometry.source.anchor_pose);
        let after = relative(
            world(pose, frame),
            world(loaded.geometry.source.anchor_pose, frame),
        );
        same_pose(before, after);
        near(learned(&loaded, before)?, learned(&loaded, after)?);
        near(
            source.checked_log_density(0, before)?,
            source.checked_log_density(0, after)?,
        );
        Ok(())
    }
    fn pose_with_frame() -> Pose {
        pose([3., -1., 2.], [-0.3, 0.4, 0.2])
    }
    #[test]
    fn hard_zero_and_geometry_mode_preserve_unconditional_denominator() -> Result<()> {
        let mut m: BTreeMap<String, Mass> = [
            (
                "hard_invalid".into(),
                Mass {
                    count: 6,
                    ..Mass::default()
                },
            ),
            (
                "A_patch_complete".into(),
                Mass {
                    count: 2,
                    ..Mass::default()
                },
            ),
        ]
        .into_iter()
        .collect();
        add_log(&mut m.get_mut("A_patch_complete").unwrap().hard, 4f64.ln())?;
        add_log(&mut m.get_mut("A_patch_complete").unwrap().hard, 12f64.ln())?;
        let rows = masses_json(&m, 8, Mode::Geometry);
        let a = rows
            .as_array()
            .unwrap()
            .iter()
            .find(|x| x["region"] == "A_patch_complete")
            .unwrap();
        near(a["unconditional_fraction"].as_f64().unwrap(), 0.25);
        near(a["log_hard_only_mass"].as_f64().unwrap().exp(), 2.);
        assert!(a["log_physical_mass"].is_null());
        let zero = rows
            .as_array()
            .unwrap()
            .iter()
            .find(|x| x["region"] == "hard_invalid")
            .unwrap();
        assert!(zero["log_hard_only_mass"].is_null());
        Ok(())
    }
    #[test]
    fn cloud_caps_charge_failed_planned_points_once_and_never_publish_partial_weight() -> Result<()>
    {
        let cfg = config();
        let mut counts = Counts::default();
        let mut candidate = (0, 0);
        let caps = remaining_caps(&cfg.cloud_limits, &counts, candidate)?;
        assert_eq!(caps.raw_points, 10);
        assert_eq!(caps.processed_points, 9);
        let progress = CloudProgress {
            begun: true,
            planned_points: Some(11),
            processed_points: 0,
            overlap_points: 0,
            complete: false,
            log_weight: None,
        };
        consume_cloud_counts(&mut counts, &mut candidate, &progress)?;
        assert_eq!(counts.raw, 11);
        assert_eq!(candidate, (11, 0));
        assert!(!progress.complete && progress.log_weight.is_none());
        assert_eq!(
            remaining_caps(&cfg.cloud_limits, &counts, candidate)?.raw_points,
            4
        );
        counts.raw = 21;
        assert!(remaining_caps(&cfg.cloud_limits, &counts, candidate).is_err());
        Ok(())
    }
    #[test]
    fn source_tokens_with_extra_neighbors_remain_in_complement() -> Result<()> {
        let cfg = config();
        let tokens = cfg
            .inputs
            .regions
            .source_secondary_tokens
            .iter()
            .cloned()
            .collect();
        let (a, p) = region_data(&tokens, &[16, 217], &cfg.inputs.regions)?;
        assert_eq!(a, "A_patch_complete");
        assert_eq!(p["source_fraction"], 1.);
        let (b, p) = region_data(&tokens, &[16, 56, 217], &cfg.inputs.regions)?;
        assert_eq!(b, "other_contact");
        assert_eq!(p["source_fraction"], 1.);
        let smaller = cfg.inputs.regions.source_secondary_tokens[..4]
            .iter()
            .cloned()
            .collect();
        assert_eq!(
            region_data(&smaller, &[16, 217], &cfg.inputs.regions)?.0,
            "A_patch_0.25_0.5"
        );
        Ok(())
    }
    #[test]
    fn config_and_failure_prefix_reject_silent_repairs() -> Result<()> {
        let mut cfg = config();
        cfg.validate()?;
        cfg.mixture = [0.5, 0.2, 0.3];
        assert!(cfg.validate().is_err());
        let dir = std::env::temp_dir().join(format!(
            "source-guide-prefix-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        fs::create_dir(&dir)?;
        {
            let mut journal = Journal::new(&dir.join("events.jsonl"))?;
            journal.emit(json!({"kind":"candidate_begun","ordinal":0}))?;
            journal.emit(json!({"kind":"fatal","reason":"synthetic cap"}))?;
        }
        let rows: Vec<Value> = fs::read_to_string(dir.join("events.jsonl"))?
            .lines()
            .map(serde_json::from_str)
            .collect::<std::result::Result<_, _>>()?;
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["event_index"], 0);
        assert_eq!(rows[1]["event_index"], 1);
        assert!(rows.iter().all(|r| r["kind"] != "candidate_complete"));
        fs::remove_dir_all(dir)?;
        Ok(())
    }
}
