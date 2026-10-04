//! Independent physical stationarity reference, explicitly opt-in and fixed allocation.
//!
//! Uniform independent centers and Haar orientations are rejected by direct sphere
//! core/wall predicates. Independent PPPs of intensity z on the two inflated-ball
//! cubes are restricted to disjoint ownership E_i \\ union_{j<i} E_j, then to the
//! complement of the fixed spectator. Their joint void probability is exactly
//! exp(-z |(E_0 union E_1) \\ S|). Since 2v is constant, this is the physical target
//! exp(z [2v - |(E_0 union E_1) \\ S|]), including variable INTERNAL union volume.
//! Bath points outside the protein wall are deliberately retained. No library
//! geometry, surrogate or physical gate is used by this source sampler.
//!
//! Four streams x 2048 IID sources are each shared by three actual kernel arms.
//! Paired changes include every rejection. World-grid coverage observables are
//! bounded diagnostics, not exact volumes; source sampling itself is exact.
//! Six-SE moment checks are not a simultaneous finite-sample confidence theorem.
//! Negative controls reuse the m8 guided endpoint, bath and final uniform only.
//! No burn-in, source replacement, adaptive allocation or sensitivity tuning.
//!
//! Compile only before authorization. Numerical invocation requires a fresh
//! absolute FLEXIBLE_SURROGATE_REFERENCE_OUT and --ignored --exact
//! flexible_kernel_preserves_independent_physical_sources --test-threads=1.
#![recursion_limit = "256"]
use anyhow::{Context, Result, ensure};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Poisson, StandardNormal};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    f64::consts::PI,
    fs::{self, File, OpenOptions},
    io::{BufWriter, Read, Write},
    path::Path,
};
use tetramer_mc::{
    bounded_singleton_path::{Budget, Limits},
    depletion::GateOptions,
    flexible_surrogate_chain::{FlexibleSurrogateConfig, FlexibleSurrogateKernel},
    geometry::{Atom, Shape, SphereTree},
    math::Pose,
    simulation::cpu_seconds,
    spherical::Container,
};

const STREAMS: usize = 4;
const SOURCES: usize = 2048;
const ATTEMPT_CAP: u64 = 1_000_000;
const POINT_CAP: u64 = 5_000_000;
const CORE: f64 = 0.1;
const RD: f64 = 0.9;
const R: f64 = CORE + RD;
const WALL: f64 = 1.6;
const CENTER_RADIUS: f64 = WALL - CORE;
const Z: f64 = 0.5;
const LAMBDA: f64 = 2.;
const CPU_CAP: f64 = 600.;
const SIGMA: f64 = 6.;
const MIN_NEGATIVE_SIGNAL: f64 = 0.005;
const ARMS: [(usize, f64, &str); 3] = [
    (1, 1., "m1_guided"),
    (8, 1., "m8_guided"),
    (8, 0., "m8_zero"),
];
const FIXED: Pose = Pose {
    position: [0.; 3],
    orientation: [1., 0., 0., 0.],
};
const OBS: usize = 38;
const ATTRACTION: usize = 15;
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
fn seed(stream: usize, role: &str) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"flexible-surrogate-independent-source-v1");
    h.update((stream as u64).to_le_bytes());
    h.update(role.as_bytes());
    h.finalize().into()
}
fn rng(stream: usize, role: &str) -> StdRng {
    StdRng::from_seed(seed(stream, role))
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
#[derive(Clone, Default, Serialize)]
struct SourceCounts {
    attempts: u64,
    hard_valid: u64,
    raw_points: u64,
    void_accepted: u64,
}
struct SourceRng {
    positions: StdRng,
    haar: StdRng,
    poisson: StdRng,
}
impl SourceRng {
    fn new(stream: usize) -> Self {
        Self {
            positions: rng(stream, "source/positions"),
            haar: rng(stream, "source/Haar"),
            poisson: rng(stream, "source/PPP"),
        }
    }
}
fn source(
    stream: usize,
    index: usize,
    random: &mut SourceRng,
    counts: &mut SourceCounts,
    journal: &mut Journal,
    budget: &Budget,
) -> Result<[Pose; 2]> {
    let poisson = Poisson::new(Z * (2. * R).powi(3))?;
    loop {
        budget.check_cpu()?;
        ensure!(counts.attempts < ATTEMPT_CAP, "fatal source attempt cap");
        counts.attempts += 1;
        journal.write(&json!({"kind":"source_begin","stream":stream,"source_index":index,"attempt":counts.attempts}))?;
        let positions: [[f64; 3]; 2] = std::array::from_fn(|_| {
            std::array::from_fn(|_| random.positions.random_range(-CENTER_RADIUS..CENTER_RADIUS))
        });
        let normals: [[f64; 4]; 2] = std::array::from_fn(|_| {
            std::array::from_fn(|_| StandardNormal.sample(&mut random.haar))
        });
        let lengths = normals.map(|q| q.iter().map(|x| x * x).sum::<f64>().sqrt());
        let mut row = json!({"kind":"source_outcome","stream":stream,"source_index":index,
            "attempt":counts.attempts,"positions":positions,"haar_normals":normals,
            "clouds":[],"void_accepted":false,"status":"completed"});
        if !lengths.iter().all(|v| v.is_finite() && *v > 0.) {
            row["status"] = json!("fatal");
            row["error"] = json!("invalid source Haar normal length");
            journal.write(&row)?;
            anyhow::bail!("invalid source Haar normal length");
        }
        let pair = std::array::from_fn(|i| Pose {
            position: positions[i],
            orientation: normals[i].map(|x| x / lengths[i]),
        });
        let hard = hard_valid(pair);
        row["poses"] = json!(pair);
        row["hard_valid"] = json!(hard);
        if !hard {
            journal.write(&row)?;
            continue;
        }
        counts.hard_valid += 1;
        let mut vacant = true;
        for owner in 0..2 {
            let n = poisson.sample(&mut random.poisson) as u64;
            row["clouds"]
                .as_array_mut()
                .unwrap()
                .push(json!({"owner":owner,"planned_points":n,"points":[]}));
            if n > POINT_CAP - counts.raw_points {
                row["status"] = json!("fatal");
                row["error"] = json!("source planned-point cap");
                journal.write(&row)?;
                anyhow::bail!("fatal source planned-point cap");
            }
            counts.raw_points += n;
            for _ in 0..n {
                let p = std::array::from_fn(|j| {
                    pair[owner].position[j] + random.poisson.random_range(-R..R)
                });
                row["clouds"][owner]["points"]
                    .as_array_mut()
                    .unwrap()
                    .push(json!(p));
                // Thin separate PPPs to DISJOINT pieces; a triple-covered point
                // is shielded, and no protein-wall predicate enters this bath.
                let owned =
                    inside(p, pair[owner].position) && (owner == 0 || !inside(p, pair[0].position));
                if owned && !inside(p, FIXED.position) {
                    vacant = false;
                }
            }
        }
        row["void_accepted"] = json!(vacant);
        journal.write(&row)?;
        if vacant {
            counts.void_accepted += 1;
            return Ok(pair);
        }
    }
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
        if self.n < 2 {
            f64::NAN
        } else {
            ((self.sum2 - self.sum * self.sum / self.n as f64).max(0.)
                / (self.n * (self.n - 1)) as f64)
                .sqrt()
        }
    }
    fn report(self) -> Value {
        json!({"n":self.n,"sum":self.sum,"sum2":self.sum2,"mean":self.mean(),"se":self.se()})
    }
    fn check(self, truth: f64, name: &str) -> Result<()> {
        ensure!(
            self.n > 1 && (self.mean() - truth).abs() <= SIGMA * self.se() + 1e-11,
            "{name}: mean={} truth={truth} SE={} n={}",
            self.mean(),
            self.se(),
            self.n
        );
        Ok(())
    }
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
            proposal: rng(stream, &format!("{arm}/proposal")),
            inner: rng(stream, &format!("{arm}/inner")),
            bath: rng(stream, &format!("{arm}/bath")),
            outer: rng(stream, &format!("{arm}/outer")),
        }
    }
}
#[derive(Clone, Default, Serialize)]
struct Work {
    attempts: usize,
    accepted: usize,
    physical_decisions: usize,
    identities: usize,
    hard_rejections: usize,
    inner_rejections: usize,
    selected_slots: [usize; 2],
    changed_separation_candidates: usize,
    changed_lens_candidates: usize,
    changed_lens_accepted: usize,
    path_orders: [usize; 2],
    raw_bath_points: u64,
    retained_bath_points: u64,
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
fn protocol() -> Value {
    let (body_points, _) = grid(8);
    let roles:Vec<_>=(0..STREAMS).map(|stream| {
        let mut names=vec!["source/positions".to_owned(),"source/Haar".to_owned(),"source/PPP".to_owned()];
        for (_,_,arm) in ARMS { for role in ["proposal","inner","bath","outer"] { names.push(format!("{arm}/{role}")); } }
        json!({"stream":stream,"seeds":names.iter().map(|role|(role.clone(),json!(seed(stream,role)))).collect::<serde_json::Map<_,_>>()})
    }).collect();
    json!({"schema":"flexible-surrogate-independent-reference-protocol-v1",
        "streams":STREAMS,"sources_per_stream":SOURCES,"arms":ARMS,"core_radius":CORE,
        "rd":RD,"inflated_radius":R,"wall_radius":WALL,"center_radius":CENTER_RADIUS,
        "activity":Z,"lambda":LAMBDA,"translation_std":0.25,"rotation_std_degrees":30.,
        "fixed_spectator":FIXED,"members":[0,1],"body_midpoint_grid_side":8,"body_raw_count":512,
        "body_retained_count":body_points.len(),
        "body_points_sha256":format!("{:x}",Sha256::digest(serde_json::to_vec(&body_points).expect("finite frozen cloud"))),
        "world_observable_grid_side":9,"point_volume":(2.*R/8.).powi(3),
        "source_attempt_cap_per_stream":ATTEMPT_CAP,"source_point_cap_per_stream":POINT_CAP,
        "bath_raw_cap":10000000,"bath_retained_cap":10000000,"bath_per_leg_cap":100000,"bath_per_outer_cap":100000,
        "cpu_limit_seconds":CPU_CAP,"gate_options":{"max_cells":31,"max_depth":8,"min_width":0.},
        "moment_se_multiplier":SIGMA,"negative_control_minimum_volume_change":MIN_NEGATIVE_SIGNAL,
        "negative_controls":["omitted","wrong_sign"],"negative_control_arm":"m8_guided",
        "observable_names":NAMES.to_vec(),"role_seeds":roles,
        "source_law":"uniform independent centers/Haar; direct hard wall; disjoint-owned PPP void on moving union minus spectator; bath permeates wall",
        "source_sampler_exact":true,"triple_volume_observable":"fixed world midpoint quadrature, not exact geometry",
        "inference":"paired IID-source moments per arm; arms correlated, no equilibrium trajectory or efficiency claim",
        "journal_flush":"every row; fsync at stream boundaries and final/failure",
        "kernel_bath_replay":"all full kernel records and role seeds; internal raw bath coordinates recoverable from pinned RNG/code, not duplicated in journal",
        "reference_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("flexible_surrogate_stationarity.rs"))),
        "compiled_library_bundle_sha256":format!("{:x}",Sha256::digest(include_bytes!(concat!(env!("OUT_DIR"),"/source-bundle.json"))))})
}

fn experiment(
    out: &Path,
    sources: &mut Journal,
    kernels: &mut Journal,
    summary: &mut Value,
) -> Result<()> {
    let core = sphere(CORE)?;
    let exclusion = sphere(R)?;
    let wall = Container::new(WALL, &core)?;
    let (points, volume) = grid(8);
    let (world, world_weight) = grid(9);
    let mut budget = Budget {
        limits: Limits {
            raw_per_leg: 100000,
            raw_per_outer: 100000,
            raw_campaign: 10000000,
            retained_per_leg: 100000,
            retained_per_outer: 100000,
            retained_campaign: 10000000,
            cpu_seconds: CPU_CAP,
        },
        started: cpu_seconds(),
        raw: 0,
        retained: 0,
    };
    let mut all_changes = [[Moments::default(); OBS]; 3];
    let mut source_moments = [Moments::default(); OBS];
    let mut negatives = [Moments::default(); 2];
    let mut triples = 0;
    let mut no_internal_contact = 0;
    let mut source_range = [f64::INFINITY, 0.0_f64];
    let mut counts = vec![SourceCounts::default(); STREAMS];
    let mut panels = vec![];
    // Publish current counters on every returned error, before the final receipt.
    let run = (|| -> Result<()> {
        for stream in 0..STREAMS {
            let mut random = SourceRng::new(stream);
            let mut randoms = ARMS.map(|(_, _, arm)| KernelRng::new(stream, arm));
            let mut changes = [[Moments::default(); OBS]; 3];
            let mut source_stats = [Moments::default(); OBS];
            let mut negative_stats = [Moments::default(); 2];
            let mut work: [Work; 3] = std::array::from_fn(|_| Work::default());
            for index in 0..SOURCES {
                let old = source(
                    stream,
                    index,
                    &mut random,
                    &mut counts[stream],
                    sources,
                    &budget,
                )?;
                ensure!(
                    normalized(old) && hard_valid(old),
                    "invalid independent source"
                );
                let before = observables(old, &world, world_weight);
                triples += usize::from(before[14] > 0.);
                no_internal_contact += usize::from(before[16] == 0.);
                source_range[0] = source_range[0].min(before[9]);
                source_range[1] = source_range[1].max(before[9]);
                for k in 0..OBS {
                    source_moments[k].add(before[k]);
                    source_stats[k].add(before[k]);
                }
                for (arm, &(steps, strength, name)) in ARMS.iter().enumerate() {
                    kernels.write(
                        &json!({"kind":"kernel_begin","stream":stream,"source_index":index,
                        "arm":name,"source_attempt":counts[stream].attempts,"old":old}),
                    )?;
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
                            inner_steps: steps,
                            translation_std: 0.25,
                            rotation_std_degrees: 30.,
                            guidance_strength: strength,
                        },
                    };
                    let mut state = [old[0], old[1], FIXED];
                    let mut record = Value::Null;
                    let r = &mut randoms[arm];
                    let result = kernel.step(
                        &mut state,
                        &mut r.proposal,
                        &mut r.inner,
                        &mut r.bath,
                        &mut r.outer,
                        &mut budget,
                        &mut record,
                    );
                    let retained = [state[0], state[1]];
                    let mut negative = Value::Null;
                    let negative_result = (|| -> Result<()> {
                        if result.is_ok() && name == "m8_guided" {
                            let candidate: [Pose; 2] =
                                serde_json::from_value(record["proposed"].clone())?;
                            negative = json!({});
                            for (j, control) in ["omitted", "wrong_sign"].iter().enumerate() {
                                let accepted = if record["status"] == "identity_self_loop" {
                                    false
                                } else {
                                    let bath = number(&record["bath"]["aggregate"], "log_weight")?;
                                    let correction = if j == 0 {
                                        0.
                                    } else {
                                        -number(&record, "complete_log_correction")?
                                    };
                                    number(&record, "log_u")? < (bath + correction).min(0.)
                                };
                                negative[*control] = json!({"accepted":accepted,"retained":if accepted {candidate} else {old}});
                            }
                        }
                        Ok(())
                    })();
                    // Full fatal records are retained before propagating any error.
                    kernels.write(
                        &json!({"kind":"kernel_outcome","stream":stream,"source_index":index,
                        "arm":name,"source_attempt":counts[stream].attempts,"record":record,
                        "retained":retained,"negative_controls":negative,
                        "negative_control_error":negative_result.as_ref().err().map(|e|format!("{e:#}"))}),
                    )?;
                    result?;
                    negative_result?;
                    ensure!(
                        state[2] == FIXED && normalized(retained) && hard_valid(retained),
                        "retained endpoint invalid"
                    );
                    let inner = record["steps"]
                        .as_array()
                        .context("missing inner history")?;
                    ensure!(inner.len() == steps, "fixed horizon truncated");
                    work[arm].attempts += 1;
                    for step in inner {
                        let slot = step["selected_slot"]
                            .as_u64()
                            .context("missing selected slot")?
                            as usize;
                        ensure!(
                            slot < 2 && step["selected_label"] == slot,
                            "scan label mismatch"
                        );
                        work[arm].selected_slots[slot] += 1;
                        let current: [Pose; 2] = serde_json::from_value(step["old"].clone())?;
                        let proposed: [Pose; 2] = serde_json::from_value(step["proposed"].clone())?;
                        let kept: [Pose; 2] = serde_json::from_value(step["retained"].clone())?;
                        ensure!(
                            current[1 - slot] == proposed[1 - slot]
                                && normalized(proposed)
                                && normalized(kept)
                                && hard_valid(kept),
                            "single-member trace invalid"
                        );
                        if step["status"] == "hard_rejected" {
                            ensure!(
                                !hard_valid(proposed) && kept == current,
                                "wrong saved hard rejection"
                            );
                            work[arm].hard_rejections += 1;
                        } else {
                            ensure!(hard_valid(proposed), "accepted hard-invalid proposal");
                            work[arm].inner_rejections += usize::from(step["accepted"] == false);
                        }
                    }
                    let candidate: [Pose; 2] = serde_json::from_value(record["proposed"].clone())?;
                    let candidate_obs = observables(candidate, &world, world_weight);
                    work[arm].changed_separation_candidates +=
                        usize::from((candidate_obs[9] - before[9]).abs() > 1e-10);
                    work[arm].changed_lens_candidates +=
                        usize::from((candidate_obs[10] - before[10]).abs() > 1e-10);
                    let correction = number(&record, "complete_log_correction")?;
                    let old_score = number(&record["old_score"], "log_surrogate")?;
                    let new_score = number(&record["proposed_score"], "log_surrogate")?;
                    ensure!(
                        (correction - old_score + new_score).abs() < 1e-12,
                        "wrong outer correction"
                    );
                    if strength == 0. {
                        ensure!(
                            (old_score, new_score, correction) == (0., 0., 0.),
                            "nonzero zero-guidance score"
                        );
                    }
                    let accepted = record["accepted"].as_bool().context("missing decision")?;
                    work[arm].accepted += usize::from(accepted);
                    work[arm].changed_lens_accepted +=
                        usize::from(accepted && (candidate_obs[10] - before[10]).abs() > 1e-10);
                    if record["status"] == "identity_self_loop" {
                        ensure!(
                            candidate == old
                                && retained == old
                                && record["physical_decisions"] == 0
                                && !accepted,
                            "bad identity"
                        );
                        work[arm].identities += 1;
                    } else {
                        ensure!(
                            record["status"] == "completed" && record["physical_decisions"] == 1,
                            "unexpected final status"
                        );
                        let bath = &record["bath"];
                        let logw = number(&bath["aggregate"], "log_weight")?;
                        ensure!(
                            (number(&record, "log_acceptance_ratio")? - logw - correction).abs()
                                < 1e-12,
                            "wrong final factor"
                        );
                        ensure!(
                            accepted == (number(&record, "log_u")? < (logw + correction).min(0.)),
                            "wrong final decision"
                        );
                        ensure!(
                            retained == if accepted { candidate } else { old },
                            "wrong retained state"
                        );
                        work[arm].physical_decisions += 1;
                        let order = match bath["order"].as_str() {
                            Some("first_then_second") => 0,
                            Some("second_then_first") => 1,
                            _ => anyhow::bail!("invalid bath order"),
                        };
                        work[arm].path_orders[order] += 1;
                        work[arm].raw_bath_points += bath["aggregate"]["raw_points"]
                            .as_u64()
                            .context("missing raw count")?;
                        work[arm].retained_bath_points += bath["aggregate"]["retained_points"]
                            .as_u64()
                            .context("missing retained count")?;
                    }
                    if name == "m8_guided" {
                        for (j, control) in ["omitted", "wrong_sign"].iter().enumerate() {
                            let pair: [Pose; 2] =
                                serde_json::from_value(negative[*control]["retained"].clone())?;
                            let delta = observables(pair, &world, world_weight)[ATTRACTION]
                                - before[ATTRACTION];
                            negatives[j].add(delta);
                            negative_stats[j].add(delta);
                        }
                    }
                    let after = observables(retained, &world, world_weight);
                    for k in 0..OBS {
                        changes[arm][k].add(after[k] - before[k]);
                        all_changes[arm][k].add(after[k] - before[k]);
                    }
                }
            }
            ensure!(
                counts[stream].void_accepted as usize == SOURCES,
                "wrong source inventory"
            );
            panels.push(
                json!({"stream":stream,"source_counts":counts[stream],"work":work,
                "source_moments":source_stats.map(Moments::report).to_vec(),
                "paired_changes":changes.map(|v|v.map(Moments::report).to_vec()),
                "negative_controls":negative_stats.map(Moments::report)}),
            );
            sources.sync()?;
            kernels.sync()?;
        }
        Ok(())
    })();
    *summary = json!({"schema":"flexible-surrogate-independent-reference-summary-v1",
        "complete":run.is_ok(),"statistical_checks":"recorded separately in receipt.json",
        "source_counts":counts,"stream_panel":panels,
        "observable_names":NAMES.to_vec(),"source_moments":source_moments.map(Moments::report).to_vec(),
        "paired_changes":all_changes.map(|v|v.map(Moments::report).to_vec()),
        "negative_controls":negatives.map(Moments::report),"source_triple_covered":triples,
        "source_no_internal_contact":no_internal_contact,"source_separation_range":source_range,
        "raw_bath_points":budget.raw,"retained_bath_points":budget.retained,
        "cpu_seconds":cpu_seconds()-budget.started});
    if let Err(error) = &run {
        summary["error"] = json!(format!("{error:#}"));
    }
    publish(&out.join("summary.json"), summary)?; // Before ANY inferential check.
    run?;
    for (arm, &(_, strength, name)) in ARMS.iter().enumerate() {
        let field_sum = |field: &str| -> u64 {
            panels
                .iter()
                .map(|p| p["work"][arm][field].as_u64().unwrap())
                .sum()
        };
        ensure!(
            field_sum("attempts") == (STREAMS * SOURCES) as u64,
            "outer inventory mismatch"
        );
        ensure!(
            field_sum("accepted") > 200 && field_sum("physical_decisions") > 500,
            "inactive arm {name}"
        );
        ensure!(
            field_sum("hard_rejections") > 0,
            "no hard-boundary exercise {name}"
        );
        ensure!(
            field_sum("changed_lens_accepted") > 100,
            "no flexible internal-volume exercise {name}"
        );
        if strength > 0. {
            ensure!(
                field_sum("inner_rejections") > 0,
                "no surrogate rejection {name}"
            );
        }
        for slot in 0..2 {
            ensure!(
                panels
                    .iter()
                    .map(|p| p["work"][arm]["selected_slots"][slot].as_u64().unwrap())
                    .sum::<u64>()
                    > 100,
                "missing member scan"
            );
        }
        for k in 0..OBS {
            ensure!(
                all_changes[arm][k].n == STREAMS * SOURCES,
                "moment denominator"
            );
            all_changes[arm][k].check(0., &format!("{name}/{}", NAMES[k]))?;
        }
    }
    for k in (0..6).chain([8]).chain(17..35) {
        source_moments[k].check(0., NAMES[k])?;
    }
    for k in 35..38 {
        source_moments[k].check(0.25, NAMES[k])?;
    }
    ensure!(
        triples > 100 && no_internal_contact > 100 && source_range[1] - source_range[0] > 1.,
        "source did not exercise variable union and triple shielding"
    );
    for (j, label) in ["omitted", "wrong_sign"].iter().enumerate() {
        ensure!(
            negatives[j].n == STREAMS * SOURCES,
            "negative-control denominator"
        );
        ensure!(
            negatives[j].mean() > MIN_NEGATIVE_SIGNAL.max(SIGMA * negatives[j].se()),
            "negative control {label} lacks predeclared sensitivity: mean={} SE={}",
            negatives[j].mean(),
            negatives[j].se()
        );
    }
    Ok(())
}

#[test]
#[ignore = "fixed numerical reference: requires frozen source/allocation and explicit fresh output"]
fn flexible_kernel_preserves_independent_physical_sources() -> Result<()> {
    let output = std::env::var_os("FLEXIBLE_SURROGATE_REFERENCE_OUT")
        .context("explicit fresh FLEXIBLE_SURROGATE_REFERENCE_OUT required")?;
    let out = Path::new(&output);
    ensure!(out.is_absolute(), "output must be absolute");
    fs::create_dir(out)?;
    publish(&out.join("protocol.json"), &protocol())?;
    let mut sources = Journal::new(&out.join("source-attempts.jsonl"))?;
    let mut kernels = Journal::new(&out.join("kernel-attempts.jsonl"))?;
    let mut summary = Value::Null;
    let result = experiment(out, &mut sources, &mut kernels, &mut summary);
    sources.sync()?;
    kernels.sync()?;
    let files = [
        "protocol.json",
        "source-attempts.jsonl",
        "kernel-attempts.jsonl",
        "summary.json",
    ];
    let mut hashes = serde_json::Map::new();
    for file in files {
        if out.join(file).exists() {
            hashes.insert(file.into(), json!(digest(&out.join(file))?));
        }
    }
    let receipt = json!({"schema":"flexible-surrogate-independent-reference-receipt-v1",
        "complete":true,"passed":result.is_ok(),
        "error":result.as_ref().err().map(|e|format!("{e:#}")),"source_counts":summary["source_counts"],
        "output_sha256":hashes,"retries":0,"replacement_draws":0,
        "numerical_allocation_complete":summary["complete"]==true,
        "scientific_checks_passed":result.is_ok()});
    publish(&out.join("receipt.json"), &receipt)?;
    eprintln!("{}", receipt);
    result
}
