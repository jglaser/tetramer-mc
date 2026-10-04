//! Fixed-allocation physical stationarity check of the actual 3D rigid kernel.
//!
//! Each source is an independent rejection draw, never a prior endpoint. A
//! uniform handle position and Haar orientation first pass direct sphere hard
//! and wall predicates. A fresh PPP in a fixed canonical rigid-union box then
//! has no point in B \\ S with probability exp(-z |B \\ S|). Since |B| is fixed,
//! this is exactly the conditional physical weight exp(z |B intersect S|).
//! Source membership, transforms and observables do not use SphereTree, the
//! surrogate or the physical-gate implementation. Triple coverage is allowed.
//!
//! Allocation is four named streams x 1024 independent sources, each shared by
//! three correlated arms: m=1 and m=8 with guidance, and m=8 without guidance.
//! All rejections remain in paired changes. Six-SE thresholds are a conservative
//! joint moment diagnostic, not an exact finite-sample confidence guarantee.
//! No burn-in, adaptive allocation, retries after failure, or angular refresh.
use anyhow::{Result, ensure};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Poisson, StandardNormal};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::f64::consts::PI;
use tetramer_mc::{
    bounded_singleton_path::{Budget, Limits},
    depletion::GateOptions,
    geometry::{Atom, Shape, SphereTree},
    math::Pose,
    rigid_surrogate_chain::{RigidSurrogateConfig, RigidSurrogateKernel},
    simulation::cpu_seconds,
    spherical::Container,
};

const STREAMS: usize = 4;
const SOURCES_PER_STREAM: usize = 1024;
const SOURCE_ATTEMPT_CAP: u64 = 1_000_000;
const SOURCE_POINT_CAP: u64 = 5_000_000;
const CORE: f64 = 0.1;
const RD: f64 = 0.9;
const EXCLUSION: f64 = CORE + RD;
const LENGTH: f64 = 0.6;
const WALL: f64 = 3.;
const CENTER_RADIUS: f64 = WALL - CORE;
const ACTIVITY: f64 = 0.5;
const LAMBDA: f64 = 2.;
const RELATIVE_Q: [f64; 4] = [0.5, 0.5, 0.5, 0.5];
const FIXED: Pose = Pose {
    position: [0.; 3],
    orientation: [1., 0., 0., 0.],
};
const ARMS: [(usize, f64, &str); 3] = [(1, 1., "m1"), (8, 1., "m8"), (8, 0., "m8-zero")];
const OBS: usize = 20;
const NAMES: [&str; OBS] = [
    "handle_x",
    "handle_y",
    "handle_z",
    "handle_radius_squared",
    "axis_x",
    "axis_y",
    "axis_z",
    "r00",
    "r10",
    "r20",
    "r01",
    "r11",
    "r21",
    "r02",
    "r12",
    "r22",
    "quaternion_scalar_squared",
    "any_contact",
    "both_contacts",
    "spectator_union_quadrature",
];

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.into_iter().zip(b).map(|(x, y)| x * y).sum()
}
fn subtract(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|k| a[k] - b[k])
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
// Quaternion-vector formula independent of the library matrix conversion.
fn rotate(q: [f64; 4], x: [f64; 3]) -> [f64; 3] {
    let v = [q[1], q[2], q[3]];
    let t = cross(v, x).map(|a| 2. * a);
    let u = cross(v, t);
    std::array::from_fn(|k| x[k] + q[0] * t[k] + u[k])
}
fn multiply(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    let av = [a[1], a[2], a[3]];
    let bv = [b[1], b[2], b[3]];
    let c = cross(av, bv);
    [
        a[0] * b[0] - dot(av, bv),
        a[0] * b[1] + b[0] * a[1] + c[0],
        a[0] * b[2] + b[0] * a[2] + c[1],
        a[0] * b[3] + b[0] * a[3] + c[2],
    ]
}
fn world(handle: Pose, point: [f64; 3]) -> [f64; 3] {
    let v = rotate(handle.orientation, point);
    std::array::from_fn(|k| handle.position[k] + v[k])
}
fn direct_valid(pair: [Pose; 2]) -> bool {
    pair.iter().all(|p| {
        let r2 = dot(p.position, p.position);
        r2 <= CENTER_RADIUS.powi(2) && r2 >= (2. * CORE).powi(2)
    }) && dot(
        subtract(pair[0].position, pair[1].position),
        subtract(pair[0].position, pair[1].position),
    ) >= (2. * CORE).powi(2)
}
fn assert_fiber(pair: [Pose; 2]) {
    let delta = subtract(pair[1].position, pair[0].position);
    let expected = rotate(pair[0].orientation, [LENGTH, 0., 0.]);
    for k in 0..3 {
        assert!(
            (delta[k] - expected[k]).abs() < 2e-11,
            "rigid translation changed"
        );
    }
    for p in pair {
        assert!((p.orientation.iter().map(|x| x * x).sum::<f64>() - 1.).abs() < 2e-12);
    }
    for k in 0..3 {
        let mut basis = [0.; 3];
        basis[k] = 1.;
        let a = rotate(pair[1].orientation, basis);
        let b = rotate(pair[0].orientation, rotate(RELATIVE_Q, basis));
        for j in 0..3 {
            assert!((a[j] - b[j]).abs() < 2e-11, "relative rotation changed");
        }
    }
}
fn role_rng(stream: usize, role: &str) -> StdRng {
    let mut hash = Sha256::new();
    hash.update(b"rigid-surrogate-independent-source-v1");
    hash.update((stream as u64).to_le_bytes());
    hash.update(role.as_bytes());
    StdRng::from_seed(hash.finalize().into())
}

#[derive(Default)]
struct SourceCounts {
    attempts: u64,
    hard_valid: u64,
    raw_points: u64,
    void_accepted: u64,
}
fn independent_source(rng: &mut StdRng, counts: &mut SourceCounts) -> Result<[Pose; 2]> {
    let box_volume = (LENGTH + 2. * EXCLUSION) * (2. * EXCLUSION).powi(2);
    let poisson = Poisson::new(ACTIVITY * box_volume)?;
    loop {
        ensure!(
            counts.attempts < SOURCE_ATTEMPT_CAP,
            "fatal fixed source-attempt cap"
        );
        counts.attempts += 1;
        let position = std::array::from_fn(|_| rng.random_range(-CENTER_RADIUS..CENTER_RADIUS));
        let q: [f64; 4] = std::array::from_fn(|_| StandardNormal.sample(rng));
        let qn = q.iter().map(|x| x * x).sum::<f64>().sqrt();
        ensure!(qn.is_finite() && qn > 0., "fatal source quaternion draw");
        let first = Pose {
            position,
            orientation: q.map(|x| x / qn),
        };
        let second = Pose {
            position: world(first, [LENGTH, 0., 0.]),
            orientation: multiply(first.orientation, RELATIVE_Q),
        };
        let pair = [first, second];
        if !direct_valid(pair) {
            continue;
        }
        counts.hard_valid += 1;
        let n = poisson.sample(rng) as u64;
        ensure!(
            n <= SOURCE_POINT_CAP - counts.raw_points,
            "fatal fixed source-point cap"
        );
        counts.raw_points += n;
        let mut vacant = true;
        for _ in 0..n {
            let point = [
                rng.random_range(-EXCLUSION..LENGTH + EXCLUSION),
                rng.random_range(-EXCLUSION..EXCLUSION),
                rng.random_range(-EXCLUSION..EXCLUSION),
            ];
            let offset = subtract(point, [LENGTH, 0., 0.]);
            let inside_union =
                dot(point, point) <= EXCLUSION.powi(2) || dot(offset, offset) <= EXCLUSION.powi(2);
            let p = world(first, point);
            // Boolean B \\ S, rather than a sum over the two mobile balls.
            if inside_union && dot(p, p) > EXCLUSION.powi(2) {
                vacant = false;
            }
        }
        if vacant {
            counts.void_accepted += 1;
            return Ok(pair);
        }
    }
}

fn grid(side: usize) -> (Vec<[f64; 3]>, f64) {
    let step = 2. * EXCLUSION / side as f64;
    let mut points = vec![];
    for i in 0..side {
        for j in 0..side {
            for k in 0..side {
                let p = [i, j, k].map(|v| -EXCLUSION + (v as f64 + 0.5) * step);
                if dot(p, p) <= EXCLUSION.powi(2) {
                    points.push(p);
                }
            }
        }
    }
    (points, step.powi(3))
}
fn observables(pair: [Pose; 2], world_grid: &[[f64; 3]], weight: f64) -> [f64; OBS] {
    let mut out = [0.; OBS];
    out[..3].copy_from_slice(&pair[0].position);
    out[3] = dot(pair[0].position, pair[0].position);
    let axis = subtract(pair[1].position, pair[0].position).map(|x| x / LENGTH);
    out[4..7].copy_from_slice(&axis);
    for k in 0..3 {
        let mut basis = [0.; 3];
        basis[k] = 1.;
        out[7 + 3 * k..10 + 3 * k].copy_from_slice(&rotate(pair[0].orientation, basis));
    }
    out[16] = pair[0].orientation[0].powi(2);
    let contact = pair.map(|p| dot(p.position, p.position) < (2. * EXCLUSION).powi(2));
    out[17] = if contact[0] || contact[1] { 1. } else { 0. };
    out[18] = if contact[0] && contact[1] { 1. } else { 0. };
    // A fixed WORLD cloud inside the spectator supplies an independent bounded
    // overlap observable. Quadrature accuracy is irrelevant to stationarity.
    out[19] = weight
        * world_grid
            .iter()
            .filter(|&&point| {
                pair.iter().any(|p| {
                    let delta = subtract(point, p.position);
                    dot(delta, delta) <= EXCLUSION.powi(2)
                })
            })
            .count() as f64;
    out
}

#[derive(Clone, Copy, Default)]
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
    fn assert_mean(self, truth: f64, name: &str) {
        assert!(
            (self.mean() - truth).abs() <= 6. * self.se() + 1e-11,
            "{name}: mean={}, truth={truth}, SE={}, n={}",
            self.mean(),
            self.se(),
            self.n
        );
    }
}
struct KernelStreams {
    proposal: StdRng,
    inner: StdRng,
    bath: StdRng,
    outer: StdRng,
}
impl KernelStreams {
    fn new(stream: usize, arm: &str) -> Self {
        Self {
            proposal: role_rng(stream, &format!("{arm}/proposal")),
            inner: role_rng(stream, &format!("{arm}/inner")),
            bath: role_rng(stream, &format!("{arm}/bath")),
            outer: role_rng(stream, &format!("{arm}/outer")),
        }
    }
}
fn sphere(radius: f64) -> Result<SphereTree> {
    SphereTree::new(Shape {
        name: "independent-source rigid sphere".into(),
        volume: 4. * PI * radius.powi(3) / 3.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius,
        }],
    })
}
fn number(record: &Value, key: &str) -> f64 {
    record[key]
        .as_f64()
        .unwrap_or_else(|| panic!("missing finite {key}: {record}"))
}

#[test]
fn actual_three_dimensional_kernel_preserves_independent_physical_sources() -> Result<()> {
    let core = sphere(CORE)?;
    let exclusion = sphere(EXCLUSION)?;
    let wall = Container::new(WALL, &core)?;
    let (points, point_volume) = grid(8);
    let (world_grid, world_weight) = grid(5);
    let mut budget = Budget {
        limits: Limits {
            raw_per_leg: 100_000,
            raw_per_outer: 100_000,
            raw_campaign: 10_000_000,
            retained_per_leg: 100_000,
            retained_per_outer: 100_000,
            retained_campaign: 10_000_000,
            cpu_seconds: 180.,
        },
        started: cpu_seconds(),
        raw: 0,
        retained: 0,
    };
    let mut changes = [[Moments::default(); OBS]; ARMS.len()];
    let mut source_moments = [Moments::default(); OBS];
    let mut source_triple_covered = 0usize;
    let mut wrong_sign = Moments::default();
    let mut accepted = [0usize; ARMS.len()];
    let mut bath_calls = [0usize; ARMS.len()];
    let mut hard_rejections = [0usize; ARMS.len()];
    let mut inner_rejections = [0usize; ARMS.len()];
    let mut panel = vec![];
    for stream in 0..STREAMS {
        let mut source_rng = role_rng(stream, "source");
        let mut counts = SourceCounts::default();
        let mut streams = ARMS.map(|(_, _, name)| KernelStreams::new(stream, name));
        let mut stream_changes = [[Moments::default(); OBS]; ARMS.len()];
        for _ in 0..SOURCES_PER_STREAM {
            budget.check_cpu()?;
            let old = independent_source(&mut source_rng, &mut counts)?;
            assert_fiber(old);
            source_triple_covered += usize::from(world_grid.iter().any(|&point| {
                old.iter().all(|p| {
                    let delta = subtract(point, p.position);
                    dot(delta, delta) <= EXCLUSION.powi(2)
                })
            }));
            let before = observables(old, &world_grid, world_weight);
            for (stat, x) in source_moments.iter_mut().zip(before) {
                stat.add(x);
            }
            for (arm, &(steps, strength, name)) in ARMS.iter().enumerate() {
                let kernel = RigidSurrogateKernel {
                    core: &core,
                    exclusion: &exclusion,
                    wall: Some(&wall),
                    wall_center: [0.; 3],
                    members: [0, 1],
                    handle: 0,
                    rd: RD,
                    activity: ACTIVITY,
                    lambda: LAMBDA,
                    envelope: GateOptions {
                        max_cells: 31,
                        max_depth: 8,
                        min_width: 0.,
                    },
                    body_points: &points,
                    point_volume,
                    config: RigidSurrogateConfig {
                        inner_steps: steps,
                        translation_std: 0.45,
                        rotation_std_degrees: 40.,
                        guidance_strength: strength,
                    },
                };
                let mut state = [old[0], old[1], FIXED];
                let mut record = Value::Null;
                let rngs = &mut streams[arm];
                kernel.step(
                    &mut state,
                    &mut rngs.proposal,
                    &mut rngs.inner,
                    &mut rngs.bath,
                    &mut rngs.outer,
                    &mut budget,
                    &mut record,
                )?;
                assert_eq!(state[2], FIXED, "spectator moved");
                let after_pair = [state[0], state[1]];
                assert_fiber(after_pair);
                assert!(direct_valid(after_pair));
                let inner = record["steps"].as_array().unwrap();
                assert_eq!(inner.len(), steps, "fixed horizon was truncated");
                for step in inner {
                    let proposed: [Pose; 2] = serde_json::from_value(step["proposed"].clone())?;
                    let retained: [Pose; 2] = serde_json::from_value(step["retained"].clone())?;
                    assert_fiber(proposed);
                    assert_fiber(retained);
                    assert!(direct_valid(retained));
                    if step["status"] == "hard_rejected" {
                        assert!(!direct_valid(proposed));
                        hard_rejections[arm] += 1;
                    } else {
                        assert!(direct_valid(proposed));
                        inner_rejections[arm] += usize::from(step["accepted"] == false);
                    }
                }
                let candidate: [Pose; 2] = serde_json::from_value(record["proposed"].clone())?;
                let correction = number(&record, "complete_log_correction");
                let source_score = number(&record["old_score"], "log_surrogate");
                let endpoint_score = number(&record["proposed_score"], "log_surrogate");
                assert!((correction - (source_score - endpoint_score)).abs() < 1e-12);
                if strength == 0. {
                    assert_eq!((source_score, endpoint_score, correction), (0., 0., 0.));
                }
                let did_accept = record["accepted"].as_bool().unwrap();
                accepted[arm] += usize::from(did_accept);
                if record["status"] == "identity_self_loop" {
                    assert_eq!(candidate, old);
                    assert_eq!(record["physical_decisions"], 0);
                    assert_eq!(after_pair, old);
                    if name == "m8" {
                        wrong_sign.add(0.);
                    }
                } else {
                    assert_eq!(record["status"], "completed");
                    assert_eq!(record["physical_decisions"], 1);
                    bath_calls[arm] += 1;
                    let bath = number(&record["bath"], "log_weight");
                    let log_u = number(&record, "log_u");
                    assert!(
                        (number(&record, "log_acceptance_ratio") - bath - correction).abs() < 1e-12
                    );
                    assert_eq!(did_accept, log_u < (bath + correction).min(0.));
                    assert_eq!(after_pair, if did_accept { candidate } else { old });
                    if name == "m8" {
                        // Deliberate negative control reuses the SAME candidate,
                        // bath and final uniform; it never enters a real chain.
                        let wrong_pair = if log_u < (bath - correction).min(0.) {
                            candidate
                        } else {
                            old
                        };
                        wrong_sign.add(
                            observables(wrong_pair, &world_grid, world_weight)[19] - before[19],
                        );
                    }
                }
                let after = observables(after_pair, &world_grid, world_weight);
                for k in 0..OBS {
                    changes[arm][k].add(after[k] - before[k]);
                    stream_changes[arm][k].add(after[k] - before[k]);
                }
            }
        }
        assert_eq!(counts.void_accepted as usize, SOURCES_PER_STREAM);
        panel.push(json!({"stream":stream,"sources":counts.void_accepted,
            "source_attempts":counts.attempts,"source_hard_valid":counts.hard_valid,
            "source_poisson_points":counts.raw_points,
            "paired_change_means":stream_changes.map(|arm| arm.map(Moments::mean))}));
    }
    eprintln!(
        "{}",
        json!({"allocation":STREAMS*SOURCES_PER_STREAM,"arms":ARMS,
        "observable_names":NAMES,"stream_panel":panel,"accepted":accepted,"bath_calls":bath_calls,
        "hard_rejections":hard_rejections,"inner_rejections":inner_rejections,
        "raw_bath_points":budget.raw,"retained_bath_points":budget.retained,
        "source_triple_covered":source_triple_covered,
        "wrong_sign_overlap_change":{"mean":wrong_sign.mean(),"se":wrong_sign.se()}})
    );
    for (arm, &(_, strength, name)) in ARMS.iter().enumerate() {
        assert!(
            accepted[arm] > 200 && bath_calls[arm] > 500,
            "inactive fixture {name}"
        );
        assert!(hard_rejections[arm] > 0, "no hard-boundary exercise {name}");
        if strength > 0. {
            assert!(inner_rejections[arm] > 0, "no surrogate rejection {name}");
        }
        for k in 0..OBS {
            assert_eq!(changes[arm][k].n, STREAMS * SOURCES_PER_STREAM);
            changes[arm][k].assert_mean(0., &format!("{name}/{}", NAMES[k]));
        }
    }
    // Rotational symmetry of the centered sphere fixture supplies separate
    // source checks with exact means, including all nine rotation entries.
    for k in (0..3).chain(4..16) {
        source_moments[k].assert_mean(0., NAMES[k]);
    }
    source_moments[16].assert_mean(0.25, NAMES[16]);
    assert!(
        source_triple_covered > 100,
        "fixture did not exercise genuine triple coverage"
    );
    assert!(
        source_moments[17].sum < (STREAMS * SOURCES_PER_STREAM - 100) as f64,
        "fixture did not exercise detached sources"
    );
    assert_eq!(wrong_sign.n, STREAMS * SOURCES_PER_STREAM);
    assert!(
        wrong_sign.mean() > 4. * wrong_sign.se(),
        "fixed negative control lacks wrong-sign sensitivity: mean={}, SE={}",
        wrong_sign.mean(),
        wrong_sign.se()
    );
    Ok(())
}
