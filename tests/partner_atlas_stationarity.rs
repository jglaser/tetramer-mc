//! Fixed stationarity reference from all 8192 previously audited IID sources.
//! No source sampler is called. Each arm starts afresh at each cached source;
//! neither earlier kernel acceptance nor current success selects the source set.
//! Compile normally; numerical execution is ignored until a frozen one-off run.
#![recursion_limit = "256"]
use anyhow::{Context, Result, ensure};
use rand::{SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    f64::consts::PI,
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};
use tetramer_mc::{
    bounded_singleton_path::{Budget, Limits},
    depletion::GateOptions,
    docking::{DockingMethod, DockingProposal},
    flexible_surrogate_chain::{FlexibleSurrogateConfig, FlexibleSurrogateKernel},
    geometry::{Atom, Shape, SphereTree},
    math::{Pose, cayley},
    proposal::FrozenRelativePoseProposal,
    simulation::cpu_seconds,
    spherical::Container,
};

const STREAMS: usize = 4;
const SOURCES: usize = 2048;
const CORE: f64 = 0.1;
const RD: f64 = 0.9;
const R: f64 = CORE + RD;
const WALL: f64 = 1.6;
const CENTER_RADIUS: f64 = WALL - CORE;
const Z: f64 = 0.5;
const LAMBDA: f64 = 2.;
const CPU_CAP: f64 = 600.;
const SIGMA: f64 = 6.;
const MIN_NEGATIVE_DRIFT: f64 = 0.0005;
// (horizon, strength, direct physical gate, name)
const ARMS: [(usize, f64, bool, &str); 4] = [
    (1, 1., true, "direct"),
    (1, 1., false, "m1_guided"),
    (8, 1., false, "m8_guided"),
    (8, 0., false, "flat8"),
];
// The following 38 observable definitions and direct sphere formulas are
// copied unchanged from the archived reference source SHA256
// 7f57221b5d5ac25695bc99185fa8f6c38f49ceec0aa02ce11d4ba69d95990875.
const FIXED: Pose = Pose {
    position: [0.; 3],
    orientation: [1., 0., 0., 0.],
};
const OBS: usize = 38;
const NAMES: [&str; OBS] = [
    "x0",
    "y0",
    "z0",
    "x1",
    "y1",
    "z1",
    "radius0_squared",
    "radius1_squared",
    "radius_squared_label_difference",
    "internal_separation",
    "analytic_internal_lens",
    "analytic_fixed_lens0",
    "analytic_fixed_lens1",
    "world_quadrature_spectator_coverage",
    "world_quadrature_triple_coverage",
    "attraction_volume_diagnostic",
    "internal_contact",
    "R0_00",
    "R0_10",
    "R0_20",
    "R0_01",
    "R0_11",
    "R0_21",
    "R0_02",
    "R0_12",
    "R0_22",
    "R1_00",
    "R1_10",
    "R1_20",
    "R1_01",
    "R1_11",
    "R1_21",
    "R1_02",
    "R1_12",
    "R1_22",
    "q0_scalar_squared",
    "q1_scalar_squared",
    "relative_quaternion_scalar_squared",
];

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.into_iter().zip(b).map(|(x, y)| x * y).sum()
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| a[i] - b[i])
}
fn norm2(a: [f64; 3]) -> f64 {
    dot(a, a)
}
fn inside(p: [f64; 3], center: [f64; 3]) -> bool {
    norm2(sub(p, center)) <= R * R
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn rotate(q: [f64; 4], x: [f64; 3]) -> [f64; 3] {
    let v = [q[1], q[2], q[3]];
    let t = cross(v, x).map(|v| 2. * v);
    let u = cross(v, t);
    std::array::from_fn(|i| x[i] + q[0] * t[i] + u[i])
}
fn hard_valid(pair: [Pose; 2]) -> bool {
    pair.iter().all(|p| {
        norm2(p.position) <= CENTER_RADIUS * CENTER_RADIUS
            && norm2(p.position) >= (2. * CORE).powi(2)
    }) && norm2(sub(pair[0].position, pair[1].position)) >= (2. * CORE).powi(2)
}
fn normalized(pair: [Pose; 2]) -> bool {
    pair.iter().all(|p| {
        p.position
            .iter()
            .chain(p.orientation.iter())
            .all(|x| x.is_finite())
            && (p.orientation.iter().map(|x| x * x).sum::<f64>() - 1.).abs() < 2e-12
    })
}
fn lens(distance: f64) -> f64 {
    if distance >= 2. * R {
        0.
    } else {
        PI * (4. * R + distance) * (2. * R - distance).powi(2) / 12.
    }
}
fn grid(side: usize) -> (Vec<[f64; 3]>, f64) {
    let dx = 2. * R / side as f64;
    let mut points = Vec::new();
    for x in 0..side {
        for y in 0..side {
            for z in 0..side {
                let p = [x, y, z].map(|i| -R + (i as f64 + 0.5) * dx);
                if inside(p, [0.; 3]) {
                    points.push(p);
                }
            }
        }
    }
    (points, dx.powi(3)) // Raw box volume / side^3, never / retained count.
}
fn observables(pair: [Pose; 2], world: &[[f64; 3]], weight: f64) -> [f64; OBS] {
    let mut o = [0.; OBS];
    o[0..3].copy_from_slice(&pair[0].position);
    o[3..6].copy_from_slice(&pair[1].position);
    o[6] = norm2(pair[0].position);
    o[7] = norm2(pair[1].position);
    o[8] = o[6] - o[7];
    o[9] = norm2(sub(pair[0].position, pair[1].position)).sqrt();
    o[10] = lens(o[9]);
    o[11] = lens(o[6].sqrt());
    o[12] = lens(o[7].sqrt());
    for &point in world {
        let covered = pair.map(|p| inside(point, p.position));
        o[13] += weight * f64::from(covered[0] || covered[1]);
        o[14] += weight * f64::from(covered[0] && covered[1]);
    }
    // Exact two-body lens terms; ONLY the triple term is world-grid quadrature.
    o[15] = o[10] + o[11] + o[12] - o[14];
    o[16] = f64::from(o[9] < 2. * R);
    for (slot, p) in pair.iter().enumerate() {
        for column in 0..3 {
            let mut basis = [0.; 3];
            basis[column] = 1.;
            let start = 17 + 9 * slot + 3 * column;
            o[start..start + 3].copy_from_slice(&rotate(p.orientation, basis));
        }
    }
    o[35] = pair[0].orientation[0].powi(2);
    o[36] = pair[1].orientation[0].powi(2);
    o[37] = pair[0]
        .orientation
        .into_iter()
        .zip(pair[1].orientation)
        .map(|(a, b)| a * b)
        .sum::<f64>()
        .powi(2);
    o
}

struct Journal(BufWriter<File>);
impl Journal {
    fn new(path: &Path) -> Result<Self> {
        Ok(Self(BufWriter::new(
            OpenOptions::new().write(true).create_new(true).open(path)?,
        )))
    }
    fn write(&mut self, value: &Value) -> Result<()> {
        serde_json::to_writer(&mut self.0, value)?;
        self.0.write_all(b"\n")?;
        self.0.flush()?;
        Ok(())
    }
    fn sync(&mut self) -> Result<()> {
        self.0.flush()?;
        self.0.get_ref().sync_all()?;
        Ok(())
    }
}
fn publish(path: &Path, value: &Value) -> Result<()> {
    let mut f = OpenOptions::new().write(true).create_new(true).open(path)?;
    serde_json::to_writer_pretty(&mut f, value)?;
    f.write_all(b"\n")?;
    f.sync_all()?;
    Ok(())
}
fn digest(path: &Path) -> Result<String> {
    let mut f = File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

fn number(v: &Value, key: &str) -> Result<f64> {
    let x = v[key]
        .as_f64()
        .with_context(|| format!("missing numeric {key}"))?;
    ensure!(x.is_finite(), "nonfinite {key}");
    Ok(x)
}
fn sphere(radius: f64) -> Result<SphereTree> {
    SphereTree::new(Shape {
        name: "independent flexible sphere reference".into(),
        volume: 4. * PI * radius.powi(3) / 3.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius,
        }],
    })
}

const EXTRA: usize = 21;
fn extra_names() -> Vec<String> {
    let mut names = Vec::new();
    for suffix in ["", "_squared"] {
        for i in 0..3 {
            for j in 0..3 {
                names.push(format!("relative_R_{i}{j}{suffix}"));
            }
        }
    }
    names.extend(["R0_00_squared", "R1_00_squared", "x0_times_R0_00"].map(str::to_owned));
    names
}
fn orientation_observables(pair: [Pose; 2]) -> [f64; EXTRA] {
    let columns: [[[f64; 3]; 3]; 2] = pair.map(|p| {
        std::array::from_fn(|j| {
            let mut e = [0.; 3];
            e[j] = 1.;
            rotate(p.orientation, e)
        })
    });
    let mut out = [0.; EXTRA];
    for i in 0..3 {
        for j in 0..3 {
            let value = dot(columns[0][i], columns[1][j]);
            out[3 * i + j] = value;
            out[9 + 3 * i + j] = value * value;
        }
    }
    out[18] = columns[0][0][0].powi(2);
    out[19] = columns[1][0][0].powi(2);
    out[20] = pair[0].position[0] * columns[0][0][0];
    out
}

#[derive(Clone, Copy, Default, Serialize)]
struct Moments {
    n: usize,
    sum: f64,
    sum2: f64,
}
impl Moments {
    fn add(&mut self, x: f64) {
        self.n += 1;
        self.sum += x;
        self.sum2 += x * x;
    }
    fn mean(self) -> f64 {
        self.sum / self.n as f64
    }
    fn se(self) -> f64 {
        ((self.sum2 - self.sum * self.sum / self.n as f64).max(0.) / (self.n * (self.n - 1)) as f64)
            .sqrt()
    }
    fn report(self) -> Value {
        json!({"n":self.n,"sum":self.sum,"sum2":self.sum2,
            "mean":if self.n>0 {Some(self.mean())} else {None},
            "se":if self.n>1 {Some(self.se())} else {None}})
    }
    fn check(self, name: &str, truth: f64) -> Value {
        let threshold = if self.n > 1 {
            SIGMA * self.se() + 1e-11
        } else {
            0.
        };
        json!({"name":name,"passed":self.n>1 && (self.mean()-truth).abs()<=threshold,
            "truth":truth,"threshold":threshold,"moments":self.report()})
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CachedSource {
    kind: String,
    stream: usize,
    source_index: usize,
    source_attempt: u64,
    old: [Pose; 2],
}
struct Inputs {
    rows: Vec<CachedSource>,
    sources: PathBuf,
    receipt: PathBuf,
    source_hash: String,
    receipt_hash: String,
    original_protocol: Value,
    source_receipt: Value,
}
fn prospective_allocation() -> Value {
    json!({"arms":ARMS.map(|(horizon,_,_,name)|json!({"id":name,"candidates_per_source":horizon})),
        "kernel_calls":32768,"fixed_candidates":147456,"streams":STREAMS,"sources_per_stream":SOURCES,
        "new_source_draws":0,"retries":0,"replacements":0,"extensions":0})
}
fn env_path(name: &str) -> Result<PathBuf> {
    let path = PathBuf::from(std::env::var_os(name).with_context(|| format!("missing {name}"))?);
    ensure!(path.is_absolute(), "{name} must be absolute");
    Ok(path)
}
fn checked_json(path: &Path, expected: &str) -> Result<Value> {
    ensure!(path.metadata()?.len() <= 1024 * 1024, "oversized metadata");
    ensure!(
        digest(path)? == expected,
        "changed metadata {}",
        path.display()
    );
    Ok(serde_json::from_reader(File::open(path)?)?)
}
fn inputs() -> Result<Inputs> {
    let sources = env_path("PARTNER_ATLAS_SOURCE_CACHE")?;
    let receipt = env_path("PARTNER_ATLAS_SOURCE_RECEIPT")?;
    let source_hash = std::env::var("PARTNER_ATLAS_SOURCE_CACHE_SHA256")?;
    let receipt_hash = std::env::var("PARTNER_ATLAS_SOURCE_RECEIPT_SHA256")?;
    let source_receipt = checked_json(&receipt, &receipt_hash)?;
    ensure!(
        source_receipt["schema"] == "partner-atlas-iid-source-cache-v1"
            && source_receipt["complete"] == true
            && source_receipt["passed"] == true
            && source_receipt["streams"] == STREAMS
            && source_receipt["sources_per_stream"] == SOURCES
            && source_receipt["total_sources"] == STREAMS * SOURCES
            && source_receipt["sources"] == json!({"path":sources,"sha256":source_hash})
            && source_receipt["prospective_reference"] == prospective_allocation(),
        "cache receipt mismatch"
    );
    let original = &source_receipt["original_protocol"];
    let original_path = Path::new(
        original["path"]
            .as_str()
            .context("missing source protocol path")?,
    );
    let original_protocol = checked_json(
        original_path,
        original["sha256"]
            .as_str()
            .context("missing source protocol hash")?,
    )?;
    ensure!(
        original_protocol["schema"] == "flexible-surrogate-independent-reference-protocol-v1"
            && original_protocol["streams"] == STREAMS
            && original_protocol["sources_per_stream"] == SOURCES
            && original_protocol["source_sampler_exact"] == true,
        "wrong original source law"
    );
    for (name, value) in [
        ("core_radius", CORE),
        ("rd", RD),
        ("inflated_radius", R),
        ("wall_radius", WALL),
        ("center_radius", CENTER_RADIUS),
        ("activity", Z),
        ("lambda", LAMBDA),
        ("translation_std", 0.25),
        ("rotation_std_degrees", 30.),
        ("point_volume", (2. * R / 8.).powi(3)),
    ] {
        ensure!(
            number(&original_protocol, name)? == value,
            "changed source target/settings {name}"
        );
    }
    ensure!(
        original_protocol["fixed_spectator"] == json!(FIXED)
            && original_protocol["members"] == json!([0, 1])
            && original_protocol["observable_names"] == json!(NAMES.to_vec())
            && original_protocol["body_midpoint_grid_side"] == 8
            && original_protocol["world_observable_grid_side"] == 9,
        "changed source geometry or observable panel"
    );
    let (body_points, _) = grid(8);
    let body_hash = format!("{:x}", Sha256::digest(serde_json::to_vec(&body_points)?));
    ensure!(
        body_hash == "c68700a110eba1324d1d50fb5498a1be724b1a4f5693bef880cb2fb9ae8fd482"
            && original_protocol["body_points_sha256"] == body_hash
            && original_protocol["body_raw_count"] == 512
            && original_protocol["body_retained_count"] == body_points.len()
            && original_protocol["gate_options"]
                == json!({"max_cells":31,"max_depth":8,"min_width":0.}),
        "changed original cloud/bath reference"
    );
    ensure!(
        sources.metadata()?.len() <= 32 * 1024 * 1024 && digest(&sources)? == source_hash,
        "changed/oversized source cache"
    );
    let mut rows = Vec::with_capacity(STREAMS * SOURCES);
    let mut previous_attempt = [0; STREAMS];
    for line in BufReader::new(File::open(&sources)?).lines() {
        let line = line?;
        ensure!(
            line.len() <= 4096 && rows.len() < STREAMS * SOURCES,
            "cache row cap"
        );
        let row: CachedSource = serde_json::from_str(&line)?;
        let n = rows.len();
        ensure!(
            row.kind == "cached_iid_source"
                && row.stream == n / SOURCES
                && row.source_index == n % SOURCES
                && row.source_attempt > previous_attempt[row.stream],
            "cache order/identity/attempt mismatch"
        );
        ensure!(
            normalized(row.old) && hard_valid(row.old),
            "invalid cached physical source"
        );
        previous_attempt[row.stream] = row.source_attempt;
        rows.push(row);
    }
    ensure!(
        rows.len() == STREAMS * SOURCES,
        "incomplete cache inventory"
    );
    Ok(Inputs {
        rows,
        sources,
        receipt,
        source_hash,
        receipt_hash,
        original_protocol,
        source_receipt,
    })
}

fn seed(stream: usize, arm: &str, role: &str) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"partner-atlas-cached-iid-physical-reference-v1");
    h.update((stream as u64).to_le_bytes());
    h.update(arm.as_bytes());
    h.update(b"/");
    h.update(role.as_bytes());
    h.finalize().into()
}
struct KernelRng {
    proposal: StdRng,
    inner: StdRng,
    bath: StdRng,
    outer: StdRng,
}
impl KernelRng {
    fn new(stream: usize, arm: &str) -> Self {
        Self {
            proposal: StdRng::from_seed(seed(stream, arm, "proposal")),
            inner: StdRng::from_seed(seed(stream, arm, "inner")),
            bath: StdRng::from_seed(seed(stream, arm, "bath")),
            outer: StdRng::from_seed(seed(stream, arm, "outer")),
        }
    }
}
fn atlas_model(shape_sha: &str) -> Value {
    // Analytic toy choices fixed before source extraction or new physical work.
    let mut lower = [[0.; 6]; 6];
    for (i, scale) in [0.4, 0.35, 0.45, 0.6, 0.5, 0.65].into_iter().enumerate() {
        lower[i][i] = scale;
    }
    lower[3][0] = 0.12;
    lower[5][1] = -0.1;
    let covariance: [[f64; 6]; 6] = std::array::from_fn(|i| {
        std::array::from_fn(|j| (0..6).map(|k| lower[i][k] * lower[j][k]).sum())
    });
    let base = json!({"coordinate_convention":"anchor-body-relative","shape_sha256":shape_sha,
        "angular_length":1.,"weights":[0.35,0.65],
        "anchors":[{"position":[0.8,0.1,-0.05],"rotation":cayley([0.1,-0.05,0.15])},
            {"position":[-0.3,0.8,0.15],"rotation":cayley([-0.1,0.15,0.05])}],
        "means":vec![[0.;6];2],"covariances":[covariance,covariance]});
    json!({"schema":"reciprocal-pose-mixture-v1","base_model":base,"reciprocal_components":[true,false]})
}
fn protocol(input: &Inputs, model: &Value) -> Value {
    let (points, _) = grid(8);
    let roles: Vec<_> = (0..STREAMS)
        .map(|stream| {
            let arms = ARMS.map(|(_, _, _, arm)| {
                let seeds = ["proposal", "inner", "bath", "outer"]
                    .map(|role| json!({"role":role,"seed":seed(stream,arm,role)}));
                json!({"arm":arm,"seeds":seeds})
            });
            json!({"stream":stream,"arms":arms})
        })
        .collect();
    json!({"schema":"partner-atlas-cached-reference-protocol-v1","streams":STREAMS,"sources_per_stream":SOURCES,
        "arms":ARMS,"prospective_reference":prospective_allocation(),"total_outer_calls":32768,"total_candidate_attempts":147456,"new_source_draws":0,
        "sources":{"path":input.sources,"sha256":input.source_hash},
        "source_receipt":{"path":input.receipt,"sha256":input.receipt_hash},"source_authority":input.source_receipt,
        "original_protocol":input.original_protocol,"core_radius":CORE,"rd":RD,"inflated_radius":R,
        "wall_radius":WALL,"center_radius":CENTER_RADIUS,"activity":Z,"lambda":LAMBDA,"fixed_spectator":FIXED,
        "members":[0,1],"translation_std":0.25,"rotation_std_degrees":30.,"mode_probabilities":{"partner_atlas":0.25,"local":0.75},
        "selected_slot_probabilities":[0.5,0.5],"atlas_model":model,"atlas_correlation":0.6,
        "atlas_uniform_probability":0.1,"atlas_cube":[3.,3.,3.],"atlas_center":[0.,0.,0.],
        "body_midpoint_grid_side":8,"body_raw_count":512,"body_retained_count":points.len(),
        "body_points_sha256":format!("{:x}",Sha256::digest(serde_json::to_vec(&points).unwrap())),
        "point_volume":(2.*R/8.).powi(3),"world_observable_grid_side":9,
        "observable_names":NAMES.to_vec(),"extra_observable_names":extra_names(),"role_seeds":roles,
        "bath_raw_cap":10000000,"bath_retained_cap":10000000,"bath_per_leg_cap":100000,"bath_per_outer_cap":100000,
        "cpu_limit_seconds":CPU_CAP,"gate_options":{"max_cells":31,"max_depth":8,"min_width":0.},
        "moment_se_multiplier":SIGMA,"negative_control_arm":"direct","negative_controls":["omitted","wrong_sign"],
        "negative_diagnostic":"accepted * (atan(logG_new)-atan(logG_old))/pi for learned direct proposals; zero otherwise; ALL8192 denominator",
        "negative_minimum_drift":MIN_NEGATIVE_DRIFT,"negative_gate":"mean > max(minimum_drift,6SE); failure leaves sensitivity unresolved, no retuning",
        "inference":"paired IID-source changes including every rejection; arms correlated; no trajectory, efficiency, or simultaneous-confidence claim",
        "source_reuse":"all cached sources in fixed order; independent of every old/new kernel outcome",
        "reference_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("partner_atlas_stationarity.rs"))),
        "compiled_library_bundle_sha256":format!("{:x}",Sha256::digest(include_bytes!(concat!(env!("OUT_DIR"),"/source-bundle.json"))))})
}

#[derive(Default, Serialize)]
struct Work {
    attempts: usize,
    candidate_attempts: usize,
    accepted: usize,
    physical_decisions: usize,
    identities: usize,
    selected_slots: [usize; 2],
    local: usize,
    atlas: usize,
    learned: usize,
    nonzero_atlas_corrections: usize,
    learned_physical_decisions: usize,
    learned_physical_acceptances: usize,
    uniform: usize,
    null_proposals: usize,
    zero_reverse_support: usize,
    hard_rejections: usize,
    mh_rejections: usize,
    changed_lens_candidates: usize,
    changed_lens_accepted: usize,
    path_orders: [usize; 2],
    raw_bath_points: u64,
    retained_bath_points: u64,
    bath_gained: u64,
    bath_lost: u64,
}
fn work_record(work: &mut Work, record: &Value, horizon: usize) -> Result<()> {
    work.attempts += 1;
    let steps = record["steps"].as_array().context("missing steps")?;
    ensure!(steps.len() == horizon, "incomplete fixed horizon");
    work.candidate_attempts += steps.len();
    let mut learned_change = false;
    for step in steps {
        let slot = step["selected_slot"]
            .as_u64()
            .context("missing selected slot")? as usize;
        ensure!(
            slot < 2 && step["selected_label"] == slot,
            "slot/label mismatch"
        );
        work.selected_slots[slot] += 1;
        match step["mode"].as_str().context("missing mode")? {
            "local" => work.local += 1,
            "partner_atlas" => {
                work.atlas += 1;
                match step["proposal_trace"]["branch"].as_str() {
                    Some("uniform") => work.uniform += 1,
                    Some("involution") => work.learned += 1,
                    _ => anyhow::bail!("unknown atlas branch"),
                }
            }
            _ => anyhow::bail!("unknown mode"),
        }
        if step["proposal_trace"]["branch"] == "involution"
            && step["proposal_trace"]["log_reverse_forward"]
                .as_f64()
                .is_some_and(|q| q.abs() > 1e-9)
        {
            work.nonzero_atlas_corrections += 1;
            learned_change |= step["accepted"] == true || step["status"] == "direct_candidate";
        }
        match step["status"].as_str().context("missing inner status")? {
            "hard_rejected" => work.hard_rejections += 1,
            "null_proposal" => work.null_proposals += 1,
            "zero_reverse_support" => work.zero_reverse_support += 1,
            "completed" => {
                work.mh_rejections += usize::from(step["accepted"] == false);
            }
            "direct_candidate" => {}
            _ => anyhow::bail!("unfinished inner attempt"),
        }
    }
    let accepted = record["accepted"]
        .as_bool()
        .context("missing final decision")?;
    work.accepted += usize::from(accepted);
    if record["status"] == "identity_self_loop" {
        ensure!(
            !accepted && record["physical_decisions"] == 0,
            "bad identity decision"
        );
        work.identities += 1;
    } else {
        ensure!(
            record["status"] == "completed" && record["physical_decisions"] == 1,
            "unfinished physical decision"
        );
        work.physical_decisions += 1;
        work.learned_physical_decisions += usize::from(learned_change);
        work.learned_physical_acceptances += usize::from(learned_change && accepted);
        let bath = &record["bath"];
        work.bath_gained += bath["aggregate"]["gained"]
            .as_u64()
            .context("missing gained count")?;
        work.bath_lost += bath["aggregate"]["lost"]
            .as_u64()
            .context("missing lost count")?;
        let order = match bath["order"].as_str() {
            Some("first_then_second") => 0,
            Some("second_then_first") => 1,
            _ => anyhow::bail!("unknown path order"),
        };
        work.path_orders[order] += 1;
        work.raw_bath_points += bath["aggregate"]["raw_points"]
            .as_u64()
            .context("missing raw count")?;
        work.retained_bath_points += bath["aggregate"]["retained_points"]
            .as_u64()
            .context("missing retained count")?;
    }
    Ok(())
}

fn experiment(
    out: &Path,
    input: &Inputs,
    model: &Value,
    journal: &mut Journal,
    summary: &mut Value,
) -> Result<()> {
    let core = sphere(CORE)?;
    let exclusion = sphere(R)?;
    let wall = Container::new(WALL, &core)?;
    let shape_sha = format!("{:x}", Sha256::digest(serde_json::to_vec(&core.shape)?));
    let atlas = DockingProposal::new(
        FrozenRelativePoseProposal::from_json_str_open(
            &model.to_string(),
            [3.; 3],
            0.1,
            &shape_sha,
        )?,
        DockingMethod::PosteriorInvolution,
        0.6,
        [0.; 3],
    )?;
    let (points, volume) = grid(8);
    let (world, weight) = grid(9);
    let mut budget = Budget::new(Limits {
        raw_per_leg: 100000,
        raw_per_outer: 100000,
        raw_campaign: 10000000,
        retained_per_leg: 100000,
        retained_per_outer: 100000,
        retained_campaign: 10000000,
        cpu_seconds: CPU_CAP,
    })?;
    let mut work: [[Work; 4]; STREAMS] =
        std::array::from_fn(|_| std::array::from_fn(|_| Work::default()));
    let mut changes = [[[Moments::default(); OBS]; 4]; STREAMS];
    let mut extras = [[[Moments::default(); EXTRA]; 4]; STREAMS];
    let mut total = [[Moments::default(); OBS]; 4];
    let mut total_extra = [[Moments::default(); EXTRA]; 4];
    let mut source_stats = [Moments::default(); OBS];
    let mut source_extra = [Moments::default(); EXTRA];
    let mut drift = [[Moments::default(); 3]; STREAMS];
    let mut total_drift = [Moments::default(); 3];
    let mut completed_sources = 0;
    let run = (|| -> Result<()> {
        for stream in 0..STREAMS {
            let mut randoms = ARMS.map(|(_, _, _, arm)| KernelRng::new(stream, arm));
            for source in &input.rows[stream * SOURCES..(stream + 1) * SOURCES] {
                let old = source.old;
                let before = observables(old, &world, weight);
                let before_extra = orientation_observables(old);
                for k in 0..OBS {
                    source_stats[k].add(before[k]);
                }
                for k in 0..EXTRA {
                    source_extra[k].add(before_extra[k]);
                }
                for (arm, &(horizon, strength, direct, name)) in ARMS.iter().enumerate() {
                    budget.check_cpu()?;
                    journal.write(&json!({"kind":"kernel_begin","stream":stream,"source_index":source.source_index,
                        "source_attempt":source.source_attempt,"arm":name,"old":old}))?;
                    let kernel = FlexibleSurrogateKernel {
                        core: &core,
                        exclusion: &exclusion,
                        wall: Some(&wall),
                        wall_center: [0.; 3],
                        members: [0, 1],
                        rd: RD,
                        activity: Z,
                        lambda: LAMBDA,
                        envelope: GateOptions {
                            max_cells: 31,
                            max_depth: 8,
                            min_width: 0.,
                        },
                        body_points: &points,
                        point_volume: volume,
                        config: FlexibleSurrogateConfig {
                            inner_steps: horizon,
                            translation_std: 0.25,
                            rotation_std_degrees: 30.,
                            guidance_strength: strength,
                        },
                    };
                    let mut state = [old[0], old[1], FIXED];
                    let mut record = Value::Null;
                    let rng = &mut randoms[arm];
                    let call = if direct {
                        kernel.step_partner_atlas_direct(
                            &atlas,
                            &mut state,
                            &mut rng.proposal,
                            &mut rng.inner,
                            &mut rng.bath,
                            &mut rng.outer,
                            &mut budget,
                            &mut record,
                        )
                    } else {
                        kernel.step_partner_atlas(
                            &atlas,
                            &mut state,
                            &mut rng.proposal,
                            &mut rng.inner,
                            &mut rng.bath,
                            &mut rng.outer,
                            &mut budget,
                            &mut record,
                        )
                    };
                    let retained = [state[0], state[1]];
                    let mut negative = Value::Null;
                    let mut drift_values = [0.; 3];
                    let mut audit_error = None;
                    let mut saved_observables = None;
                    let inspected = (|| -> Result<()> {
                        if call.is_err() {
                            return Ok(());
                        }
                        ensure!(
                            state[2] == FIXED && normalized(retained) && hard_valid(retained),
                            "invalid retained physical state"
                        );
                        let candidate: [Pose; 2] =
                            serde_json::from_value(record["proposed"].clone())?;
                        let accepted = record["accepted"]
                            .as_bool()
                            .context("missing final decision")?;
                        ensure!(
                            retained == if accepted { candidate } else { old },
                            "wrong retained endpoint"
                        );
                        if record["physical_decisions"] == 1 {
                            let q = if direct {
                                number(&record["steps"][0], "proposal_log_reverse_forward")?
                            } else {
                                number(&record["old_score"], "log_surrogate")?
                                    - number(&record["proposed_score"], "log_surrogate")?
                            };
                            let ratio = number(&record["bath"]["aggregate"], "log_weight")? + q;
                            ensure!(
                                number(&record, "complete_log_correction")? == q
                                    && number(&record, "log_acceptance_ratio")? == ratio
                                    && accepted == (number(&record, "log_u")? < ratio.min(0.)),
                                "wrong final correction/decision"
                            );
                        }
                        if direct {
                            negative = json!({});
                            let mut density_change = 0.;
                            let trace = &record["steps"][0];
                            if trace["proposal_trace"]["branch"] == "involution"
                                && record["physical_decisions"] == 1
                            {
                                let old_g = number(
                                    &trace["proposal_trace"],
                                    "full_old_member_log_density",
                                )?;
                                let new_g = number(
                                    &trace["proposal_trace"],
                                    "full_new_member_log_density",
                                )?;
                                density_change = (new_g.atan() - old_g.atan()) / PI;
                            }
                            drift_values[0] = if accepted { density_change } else { 0. };
                            for (index, label) in ["omitted", "wrong_sign"].iter().enumerate() {
                                let take = if record["physical_decisions"] == 1 {
                                    let logw = number(&record["bath"]["aggregate"], "log_weight")?;
                                    let q = number(trace, "proposal_log_reverse_forward")?;
                                    number(&record, "log_u")?
                                        < (if index == 0 { logw } else { logw - q }).min(0.)
                                } else {
                                    false
                                };
                                let kept = if take { candidate } else { old };
                                negative[*label] = json!({"accepted":take,"retained":kept});
                                drift_values[index + 1] = if take { density_change } else { 0. };
                            }
                        }
                        // These are the same reductions previously made after
                        // journaling. Save their already computed vectors so an
                        // independent reader need not reclassify any geometry.
                        let after = observables(retained, &world, weight);
                        let after_extra = orientation_observables(retained);
                        let candidate_lens =
                            lens(norm2(sub(candidate[0].position, candidate[1].position)).sqrt());
                        saved_observables = Some((after, after_extra, candidate_lens));
                        Ok(())
                    })();
                    if let Err(error) = &inspected {
                        audit_error = Some(format!("{error:#}"));
                    }
                    journal.write(&json!({"kind":"kernel_outcome","stream":stream,"source_index":source.source_index,
                        "source_attempt":source.source_attempt,"arm":name,"record":record,"retained":retained,
                        "negative_controls":negative,"bounded_density_drift":if direct {Some(drift_values)} else {None},
                        "source_observables":before.to_vec(),"source_extra_observables":before_extra.to_vec(),
                        "retained_observables":saved_observables.map(|(values,_,_)|values.to_vec()),
                        "retained_extra_observables":saved_observables.map(|(_,values,_)|values.to_vec()),
                        "candidate_internal_lens":saved_observables.map(|(_,_,value)|value),
                        "kernel_error":call.as_ref().err().map(|e|format!("{e:#}")),"audit_error":audit_error}))?;
                    call?;
                    inspected?;
                    work_record(&mut work[stream][arm], &record, horizon)?;
                    let (after, after_extra, candidate_lens) =
                        saved_observables.context("missing completed observable vectors")?;
                    work[stream][arm].changed_lens_candidates +=
                        usize::from((candidate_lens - before[10]).abs() > 1e-10);
                    work[stream][arm].changed_lens_accepted +=
                        usize::from((after[10] - before[10]).abs() > 1e-10);
                    for k in 0..OBS {
                        let delta = after[k] - before[k];
                        changes[stream][arm][k].add(delta);
                        total[arm][k].add(delta);
                    }
                    for k in 0..EXTRA {
                        let delta = after_extra[k] - before_extra[k];
                        extras[stream][arm][k].add(delta);
                        total_extra[arm][k].add(delta);
                    }
                    if direct {
                        for k in 0..3 {
                            drift[stream][k].add(drift_values[k]);
                            total_drift[k].add(drift_values[k]);
                        }
                    }
                }
                completed_sources += 1;
            }
            journal.sync()?;
        }
        ensure!(
            digest(&input.sources)? == input.source_hash
                && digest(&input.receipt)? == input.receipt_hash,
            "source cache changed during experiment"
        );
        Ok(())
    })();
    let mut checks = Vec::new();
    if run.is_ok() {
        for (arm, &(horizon, _, _, name)) in ARMS.iter().enumerate() {
            let attempts: usize = work.iter().map(|p| p[arm].attempts).sum();
            let candidates: usize = work.iter().map(|p| p[arm].candidate_attempts).sum();
            let accepted: usize = work.iter().map(|p| p[arm].accepted).sum();
            let learned: usize = work.iter().map(|p| p[arm].learned).sum();
            let uniform: usize = work.iter().map(|p| p[arm].uniform).sum();
            let changed: usize = work.iter().map(|p| p[arm].changed_lens_accepted).sum();
            let nonzero_q: usize = work.iter().map(|p| p[arm].nonzero_atlas_corrections).sum();
            let hard_rejections: usize = work.iter().map(|p| p[arm].hard_rejections).sum();
            let physical_decisions: usize = work.iter().map(|p| p[arm].physical_decisions).sum();
            let selected_slots: [usize; 2] =
                std::array::from_fn(|i| work.iter().map(|p| p[arm].selected_slots[i]).sum());
            let path_orders: [usize; 2] =
                std::array::from_fn(|i| work.iter().map(|p| p[arm].path_orders[i]).sum());
            let learned_physical: usize =
                work.iter().map(|p| p[arm].learned_physical_decisions).sum();
            let learned_accepted: usize = work
                .iter()
                .map(|p| p[arm].learned_physical_acceptances)
                .sum();
            let raw_points: u64 = work.iter().map(|p| p[arm].raw_bath_points).sum();
            let gained: u64 = work.iter().map(|p| p[arm].bath_gained).sum();
            let lost: u64 = work.iter().map(|p| p[arm].bath_lost).sum();

            checks.push(json!({"name":format!("{name}/inventory_and_activity"),"passed":attempts==8192
                && candidates==8192*horizon && accepted>200 && learned>100 && uniform>10 && changed>100
                && nonzero_q>100 && hard_rejections>0 && physical_decisions>500
                && selected_slots.iter().all(|n|*n>100) && path_orders.iter().all(|n|*n>100)
                && learned_physical>10 && learned_accepted>0 && raw_points>100 && gained>0 && lost>0,
                "attempts":attempts,"candidate_attempts":candidates,"accepted":accepted,"learned":learned,"uniform":uniform,
                "changed_lens_accepted":changed,"nonzero_atlas_corrections":nonzero_q,
                "hard_rejections":hard_rejections,"physical_decisions":physical_decisions,
                "selected_slots":selected_slots,"path_orders":path_orders,
                "learned_physical_decisions":learned_physical,"learned_physical_acceptances":learned_accepted,
                "raw_bath_points":raw_points,"bath_gained":gained,"bath_lost":lost}));
            for k in 0..OBS {
                checks.push(total[arm][k].check(&format!("{name}/{}", NAMES[k]), 0.));
            }
            for (k, label) in extra_names().iter().enumerate() {
                checks.push(total_extra[arm][k].check(&format!("{name}/{label}"), 0.));
            }
        }
        for k in (0..6).chain([8]).chain(17..35) {
            checks.push(source_stats[k].check(&format!("source/{}", NAMES[k]), 0.));
        }
        for k in 35..38 {
            checks.push(source_stats[k].check(&format!("source/{}", NAMES[k]), 0.25));
        }
        for (k, label) in extra_names().iter().enumerate() {
            checks.push(source_extra[k].check(
                &format!("source/{label}"),
                if (9..20).contains(&k) { 1. / 3. } else { 0. },
            ));
        }
        checks.push(total_drift[0].check("direct/bounded_learned_density_drift", 0.));
        for (i, name) in ["omitted", "wrong_sign"].iter().enumerate() {
            let value = total_drift[i + 1];
            let threshold = MIN_NEGATIVE_DRIFT.max(SIGMA * value.se());
            checks.push(json!({"name":format!("negative/{name}/bounded_density_drift"),"passed":value.n==8192 && value.mean()>threshold,
                "threshold":threshold,"moments":value.report(),"failure_meaning":"unresolved negative-control sensitivity; no retuning"}));
        }
    }
    let passed = run.is_ok() && !checks.is_empty() && checks.iter().all(|c| c["passed"] == true);
    let panels: Vec<_> = (0..STREAMS)
        .map(|stream| {
            json!({"stream":stream,"work":work[stream],
        "paired_changes":changes[stream].map(|m|m.map(Moments::report).to_vec()),
        "paired_extra_changes":extras[stream].map(|m|m.map(Moments::report).to_vec()),
        "bounded_density_drift":drift[stream].map(Moments::report)})
        })
        .collect();
    *summary = json!({"schema":"partner-atlas-cached-reference-summary-v1","complete":run.is_ok(),"passed":passed,
        "completed_sources":completed_sources,"streams":panels,"arms":ARMS,
        "paired_changes":total.map(|m|m.map(Moments::report).to_vec()),
        "paired_extra_changes":total_extra.map(|m|m.map(Moments::report).to_vec()),
        "source_moments":source_stats.map(Moments::report).to_vec(),"source_extra_moments":source_extra.map(Moments::report).to_vec(),
        "bounded_density_drift":total_drift.map(Moments::report),"checks":checks,"statistical_checks_passed":passed,
        "error":run.as_ref().err().map(|e|format!("{e:#}")),"raw_bath_points":budget.raw,"retained_bath_points":budget.retained,
        "cpu_seconds":cpu_seconds()-budget.started,"new_source_draws":0,"retries":0,"replacement_draws":0});
    publish(&out.join("summary.json"), summary)?; // Preserve ALL results even when a statistical check fails.
    run?;
    ensure!(
        passed,
        "completed reference has failed diagnostic checks; inspect preserved summary, no retuning"
    );
    Ok(())
}

#[test]
#[ignore = "fixed cached-source physical reference; requires frozen inputs/plan and fresh output"]
fn partner_atlas_preserves_cached_physical_sources() -> Result<()> {
    let input = inputs()?;
    let out = env_path("PARTNER_ATLAS_REFERENCE_OUT")?;
    fs::create_dir(&out)?;
    let core = sphere(CORE)?;
    let shape_sha = format!("{:x}", Sha256::digest(serde_json::to_vec(&core.shape)?));
    let model = atlas_model(&shape_sha);
    publish(&out.join("protocol.json"), &protocol(&input, &model))?;
    let mut journal = Journal::new(&out.join("kernel-attempts.jsonl"))?;
    let mut summary = Value::Null;
    let result = experiment(&out, &input, &model, &mut journal, &mut summary);
    journal.sync()?;
    let files = ["protocol.json", "kernel-attempts.jsonl", "summary.json"];
    let mut hashes = serde_json::Map::new();
    for name in files {
        if out.join(name).exists() {
            hashes.insert(name.into(), json!(digest(&out.join(name))?));
        }
    }
    let receipt = json!({"schema":"partner-atlas-cached-reference-receipt-v1","complete":true,"passed":result.is_ok(),
        "numerical_allocation_complete":summary["complete"]==true,"statistical_checks_passed":summary["statistical_checks_passed"]==true,
        "error":result.as_ref().err().map(|e|format!("{e:#}")),"output_sha256":hashes,
        "source_cache_sha256":input.source_hash,"source_receipt_sha256":input.receipt_hash,
        "new_source_draws":0,"retries":0,"replacement_draws":0});
    publish(&out.join("receipt.json"), &receipt)?;
    eprintln!("{receipt}");
    result
}
