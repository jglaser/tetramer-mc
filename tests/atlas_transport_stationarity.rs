//! Independent controls for an imported, full-covariance atlas with reversible
//! parameter transport. The exact target in stored coordinates is pi(X)phi(eta):
//! no fit likelihood, parameter Jacobian, or component-count factor belongs in
//! physical acceptance. The reverse proposal must nevertheless be rebuilt at Y.
use anyhow::Result;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde_json::{Value, json};
use std::f64::consts::PI;
use tetramer_mc::{
    atlas_transport::{AtlasTransportConfig, AtlasTransportEngine, AtlasTransportState},
    depletion::{self, GateOptions},
    geometry::{Atom, Environment, Placed, Shape, SphereTree},
    math::*,
    proposal::{FrozenRelativePoseProposal, GaussianComponentParameters},
    spherical::{self, Container, HalfTurn},
};

const SHA: &str = "0000000000000000000000000000000000000000000000000000000000000000";
const CORE: f64 = 1.;
const WALL: f64 = 4.;
const RD: f64 = 0.7;
const ACTIVITY: f64 = 0.4;
const REPLICATES: usize = 6000;
const ROUNDS: usize = 3;

fn config(gains: [f64; 3], noise: f64) -> Result<AtlasTransportConfig> {
    Ok(serde_json::from_value(json!({
        "mean_gain":gains[0], "covariance_gain":gains[1], "weight_gain":gains[2],
        "mean_noise":noise, "covariance_noise":noise, "weight_noise":noise,
        "assignment_cutoff":100., "residual_clip":4., "shrinkage":0.5,
        "initialization":"random"
    }))?)
}

fn broad_model() -> Result<FrozenRelativePoseProposal> {
    let components = (0..2)
        .map(|k| {
            let mut lower = [[0.; 6]; 6];
            for (d, row) in lower.iter_mut().enumerate() {
                row[d] = 1.2 + 0.1 * d as f64 + 0.15 * k as f64;
            }
            lower[3][0] = 0.25;
            lower[4][1] = -0.2;
            let covariance = std::array::from_fn(|i| {
                std::array::from_fn(|j| (0..6).map(|d| lower[i][d] * lower[j][d]).sum())
            });
            GaussianComponentParameters {
                anchor_position: if k == 0 { [2.4, 0., 0.] } else { [0., 2.8, 0.] },
                anchor_rotation: IDENTITY,
                mean: [0.; 6],
                covariance,
                weight: if k == 0 { 0.55 } else { 0.45 },
            }
        })
        .collect();
    FrozenRelativePoseProposal::from_components_open(components, 1., [8.; 3], 0.25, SHA, SHA)
}

fn normal<const N: usize>(rng: &mut StdRng) -> [f64; N] {
    std::array::from_fn(|_| StandardNormal.sample(rng))
}

fn direction(rng: &mut StdRng) -> Vec3 {
    let v: Vec3 = normal(rng);
    scale(v, 1. / norm(v))
}

fn overlap(radius: f64, d: f64) -> f64 {
    if d >= 2. * radius {
        0.
    } else {
        PI * (4. * radius + d) * (2. * radius - d).powi(2) / 12.
    }
}

fn independent_pose(rng: &mut StdRng) -> Pose {
    let position = scale(direction(rng), (WALL - CORE) * rng.random::<f64>().cbrt());
    let q: [f64; 4] = normal(rng);
    let length = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    Pose {
        position,
        orientation: q.map(|v| v / length),
    }
}

fn exact_physical_start(rng: &mut StdRng) -> Vec<Pose> {
    let maximum = overlap(CORE + RD, 2. * CORE);
    loop {
        let state = vec![independent_pose(rng), independent_pose(rng)];
        let d = norm(sub(state[0].position, state[1].position));
        if d >= 2. * CORE
            && rng.random::<f64>().max(f64::MIN_POSITIVE).ln()
                < ACTIVITY * (overlap(CORE + RD, d) - maximum)
        {
            return state;
        }
    }
}

fn integrate(f: impl Fn(f64) -> f64, lo: f64, hi: f64) -> f64 {
    let n = 8192;
    let h = (hi - lo) / n as f64;
    let mut sum = f(lo) + f(hi);
    for i in 1..n {
        sum += if i % 2 == 0 { 2. } else { 4. } * f(lo + i as f64 * h);
    }
    sum * h / 3.
}

fn reference() -> [f64; 15] {
    let c = WALL - CORE;
    let w = |d: f64| d * d * overlap(c, d) * (ACTIVITY * overlap(CORE + RD, d)).exp();
    let z = integrate(w, 2. * CORE, 2. * c);
    let midpoint_integral = |d: f64| {
        let h = d / 2.;
        let height = c - h;
        let a = c * c - h * h;
        2. * PI
            * (0.5 * a * a * height - a * h * height.powi(2) + (2. / 3.) * h * h * height.powi(3)
                - 0.1 * height.powi(5))
    };
    [
        integrate(|d| d * w(d), 2. * CORE, 2. * c) / z,
        integrate(|d| d * d * w(d), 2. * CORE, 2. * c) / z,
        integrate(w, 2. * CORE, 2. * (CORE + RD)) / z,
        integrate(
            |d| d * d * midpoint_integral(d) * (ACTIVITY * overlap(CORE + RD, d)).exp(),
            2. * CORE,
            2. * c,
        ) / z,
        0.,
        1. / 3.,
        0.,
        1. / 3.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
    ]
}

const NAMES: [&str; 15] = [
    "distance",
    "distance_squared",
    "AO_contact",
    "midpoint_squared",
    "absolute_R00",
    "absolute_R00_squared",
    "relative_R00",
    "relative_R00_squared",
    "eta_first",
    "eta_first_squared_residual",
    "eta_last",
    "eta_last_squared_residual",
    "eta_first_distance",
    "eta_squared_distance",
    "eta_last_relative_rotation",
];

fn observables(state: &[Pose], auxiliary: &AtlasTransportState) -> [f64; 15] {
    let d = norm(sub(state[0].position, state[1].position));
    let midpoint = scale(add(state[0].position, state[1].position), 0.5);
    let a = rotation(state[0].orientation);
    let relative = matmul(transpose(a), rotation(state[1].orientation))[0][0];
    let first = auxiliary.eta[0];
    let last = *auxiliary.eta.last().unwrap();
    [
        d,
        d * d,
        f64::from(d < 2. * (CORE + RD)),
        dot(midpoint, midpoint),
        a[0][0],
        a[0][0].powi(2),
        relative,
        relative * relative,
        first,
        first * first - 1.,
        last,
        last * last - 1.,
        first * d,
        (first * first - 1.) * d,
        last * relative,
    ]
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
    fn score(self, target: f64) -> f64 {
        let se = ((self.sum2 - self.sum * self.sum / self.n as f64).max(0.)
            / (self.n - 1) as f64
            / self.n as f64)
            .sqrt();
        (self.mean() - target) / se.max(1e-15)
    }
}

#[derive(Default, Debug)]
struct Exercised {
    learned_valid: usize,
    learned_accepted: usize,
    local_valid: usize,
    local_accepted: usize,
    gate_points: u64,
    gca_points: u64,
    partial_gca: usize,
    shifts: usize,
    max_reverse_change: f64,
    max_fit_change: f64,
}

fn compose(
    engine: &AtlasTransportEngine,
    tree: &SphereTree,
    wall: &Container,
    state: &mut Vec<Pose>,
    auxiliary: &mut AtlasTransportState,
    rng: &mut StdRng,
    exercised: &mut Exercised,
) -> Result<()> {
    for round in 0..ROUNDS {
        for learned in [true, false] {
            let moving = rng.random_range(0..2);
            let anchor = 1 - moving;
            let old = state[moving];
            let fit = engine.fit(state)?;
            let proposal = if learned {
                Some(engine.model(&fit, auxiliary)?.propose(rng, state, moving)?)
            } else {
                None
            };
            let candidate = match &proposal {
                Some(p) => p.candidate,
                None => Some(Pose {
                    position: add(old.position, scale(normal(rng), 0.65)),
                    orientation: quaternion(matmul(
                        cayley(scale(normal(rng), 0.25)),
                        rotation(old.orientation),
                    )),
                }),
            };
            let Some(new) = candidate else {
                continue;
            };
            let env = Environment {
                tree,
                fixed: vec![Placed::new(state[anchor])],
                labels: vec![(anchor, [0; 3])],
                rd: RD,
            };
            if !wall.contains(new) || !env.hard_valid(new) {
                continue;
            }
            if learned {
                exercised.learned_valid += 1;
            } else {
                exercised.local_valid += 1;
            }
            let mut next = state.clone();
            next[moving] = new;
            let next_fit = engine.fit(&next)?;
            exercised.max_fit_change = fit
                .coordinates
                .iter()
                .zip(&next_fit.coordinates)
                .map(|(a, b)| (a - b).abs())
                .fold(exercised.max_fit_change, f64::max);
            let correction = if let Some(proposal) = proposal {
                let reverse = engine
                    .model(&next_fit, auxiliary)?
                    .log_density(&old, &state[anchor])?;
                let ratio = reverse - proposal.new_log_density.unwrap();
                exercised.max_reverse_change = exercised
                    .max_reverse_change
                    .max((ratio - proposal.log_reverse_forward.unwrap()).abs());
                ratio
            } else {
                0.
            };
            let gate = depletion::sample(
                rng,
                &env,
                old,
                new,
                4. * ACTIVITY,
                ACTIVITY,
                GateOptions {
                    max_cells: 1,
                    max_depth: 0,
                    min_width: 0.,
                },
            )?;
            exercised.gate_points += gate.raw_points;
            if rng.random::<f64>().max(f64::MIN_POSITIVE).ln()
                < (correction + gate.log_weight).min(0.)
            {
                *state = next;
                if learned {
                    exercised.learned_accepted += 1;
                } else {
                    exercised.local_accepted += 1;
                }
            }
        }
        // These physical kernels do not depend on eta, so neither needs an
        // auxiliary fit-density acceptance factor in stored latent coordinates.
        let stats = spherical::update(
            tree,
            wall,
            state,
            HalfTurn::new(direction(rng))?,
            RD,
            ACTIVITY,
            rng,
        )?;
        exercised.gca_points += stats.poisson_probes;
        exercised.partial_gca += usize::from(stats.partial_flip);
        let u = ((rng.random::<u64>() >> 12) as f64 + 0.5) / 4503599627370496.;
        wall.center_shift(state, direction(rng), u)?;
        exercised.shifts += 1;
        // Keep the last eta during the final physical sweep, so the endpoint
        // cross-moments test its independence from the physical configuration.
        if round + 1 < ROUNDS {
            engine.refresh(auxiliary, rng)?;
        }
    }
    Ok(())
}

#[test]
fn transported_full_atlas_local_poisson_gca_shift_preserves_exact_joint_law() -> Result<()> {
    let tree = SphereTree::new(Shape {
        name: "AO sphere".into(),
        volume: 4. * PI / 3.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius: CORE,
        }],
    })?;
    let wall = Container::new(WALL, &tree)?;
    let engine = AtlasTransportEngine::new(broad_model()?, config([1., 0.6, 0.8], 0.2)?)?;
    let exact = reference();
    let mut initial = [Moments::default(); 15];
    let mut after = [Moments::default(); 15];
    let mut change = [Moments::default(); 15];
    let mut exercised = Exercised::default();
    let mut starts = StdRng::seed_from_u64(20261002854);
    for replicate in 0..REPLICATES {
        let mut state = exact_physical_start(&mut starts);
        let mut auxiliary = engine.initialize(&engine.fit(&state)?, &mut starts)?;
        let before = observables(&state, &auxiliary);
        let mut rng = StdRng::seed_from_u64(822019 + replicate as u64);
        compose(
            &engine,
            &tree,
            &wall,
            &mut state,
            &mut auxiliary,
            &mut rng,
            &mut exercised,
        )?;
        let values = observables(&state, &auxiliary);
        for j in 0..15 {
            initial[j].add(before[j]);
            after[j].add(values[j]);
            change[j].add(values[j] - before[j]);
        }
    }
    for j in 0..15 {
        eprintln!(
            "{}: exact={:.7} initial={:.7} after={:.7} paired_z={:.3}",
            NAMES[j],
            exact[j],
            initial[j].mean(),
            after[j].mean(),
            change[j].score(0.)
        );
        assert!(
            initial[j].score(exact[j]).abs() < 5.5,
            "invalid exact start: {}",
            NAMES[j]
        );
        assert!(
            after[j].score(exact[j]).abs() < 5.5,
            "changed joint marginal: {}",
            NAMES[j]
        );
        assert!(
            change[j].score(0.).abs() < 5.5,
            "paired stationarity: {}",
            NAMES[j]
        );
    }
    eprintln!("Atlas-transport exact-start kernels exercised: {exercised:?}");
    assert!(exercised.learned_valid > 500 && exercised.learned_accepted > 100);
    assert!(exercised.local_valid > 3000 && exercised.local_accepted > 1000);
    assert!(
        exercised.gate_points > 10000
            && exercised.gca_points > 1000
            && exercised.partial_gca > 1000
    );
    assert_eq!(exercised.shifts, REPLICATES * ROUNDS);
    assert!(exercised.max_reverse_change > 0.1 && exercised.max_fit_change > 0.1);
    Ok(())
}

#[test]
fn discrete_pose_control_detects_stale_reverse_for_each_parameter_family() -> Result<()> {
    // A finite quadrature target lets us integrate all transitions exactly.
    // Rows are normalized evaluations of the REAL pose density, not a surrogate
    // probability invented to make the correction pass.
    let anchor = Pose {
        position: [0.; 3],
        orientation: [1., 0., 0., 0.],
    };
    let moving = [
        Pose {
            position: [2.1, 0.2, 0.],
            orientation: quaternion(cayley([0.; 3])),
        },
        Pose {
            position: [0., 2.5, 0.2],
            orientation: quaternion(cayley([0.3, 0.1, 0.])),
        },
        Pose {
            position: [-2.2, 0.5, 0.],
            orientation: quaternion(cayley([-0.4, 0., 0.2])),
        },
        Pose {
            position: [0., 0., 2.2],
            orientation: quaternion(cayley([0.; 3])),
        },
    ];
    let pi = [0.1, 0.2, 0.3, 0.4];
    for (label, gains) in [
        ("means", [1., 0., 0.]),
        ("covariances", [0., 1., 0.]),
        ("weights", [0., 0., 1.]),
    ] {
        let engine = AtlasTransportEngine::new(broad_model()?, config(gains, 0.)?)?;
        let state = AtlasTransportState { eta: vec![0.; 55] };
        let mut q = [[0.; 4]; 4];
        for i in 0..4 {
            let model = engine.model(&engine.fit(&[anchor, moving[i]])?, &state)?;
            for j in 0..4 {
                q[i][j] = model.log_density(&moving[j], &anchor)?.exp();
            }
            let sum = q[i].iter().sum::<f64>();
            for value in &mut q[i] {
                *value /= sum;
            }
        }
        let mut good = [0.; 4];
        let mut bad = [0.; 4];
        let mut max_flux_error = 0_f64;
        for x in 0..4 {
            let mut good_out = 0.;
            let mut bad_out = 0.;
            for y in 0..4 {
                if x == y {
                    continue;
                }
                let accepted = q[x][y] * (pi[y] * q[y][x] / (pi[x] * q[x][y])).min(1.);
                let stale = q[x][y] * (pi[y] * q[x][x] / (pi[x] * q[x][y])).min(1.);
                let reverse = q[y][x] * (pi[x] * q[x][y] / (pi[y] * q[y][x])).min(1.);
                max_flux_error = max_flux_error.max((pi[x] * accepted - pi[y] * reverse).abs());
                good[y] += pi[x] * accepted;
                bad[y] += pi[x] * stale;
                good_out += accepted;
                bad_out += stale;
            }
            good[x] += pi[x] * (1. - good_out);
            bad[x] += pi[x] * (1. - bad_out);
        }
        let correct_error = good
            .iter()
            .zip(pi)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f64::max);
        let stale_error = bad
            .iter()
            .zip(pi)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f64::max);
        eprintln!(
            "{label}: corrected residual {correct_error:.3e}, flux error {max_flux_error:.3e}, stale reverse marginal drift {stale_error:.6}"
        );
        assert!(correct_error < 1e-13 && max_flux_error < 1e-13);
        assert!(
            stale_error > 1e-4,
            "{label} negative control must have power"
        );
    }
    Ok(())
}

#[test]
fn zero_update_preserves_actual_narrow_imported_atlas_densities_and_draws() -> Result<()> {
    let raw: Value = serde_json::from_str(include_str!("fixtures/proposal_reference.json"))?;
    let base = FrozenRelativePoseProposal::from_json_str_open(
        include_str!("fixtures/proposal_model.json"),
        [600.; 3],
        0.1,
        raw["shape_sha256"].as_str().unwrap(),
    )?;
    let params = base.component_parameters();
    assert!(
        params
            .iter()
            .any(|p| (0..3).all(|i| p.covariance[i][i].sqrt() < 0.1)),
        "control must include the previously learned narrow translational basins"
    );
    let engine = AtlasTransportEngine::new(base.clone(), config([0.; 3], 0.)?)?;
    let auxiliary = AtlasTransportState {
        eta: vec![0.; 28 * params.len() - 1],
    };
    for case in raw["cases"].as_array().unwrap() {
        let old: Pose = serde_json::from_value(case["old"].clone())?;
        let anchor: Pose = serde_json::from_value(case["anchor"].clone())?;
        let next: Pose = serde_json::from_value(case["candidate"].clone())?;
        let poses = [old, anchor];
        let model = engine.model(&engine.fit(&poses)?, &auxiliary)?;
        for pose in [old, next] {
            let a = base.log_density(&pose, &anchor)?;
            let b = model.log_density(&pose, &anchor)?;
            assert!(
                (a - b).abs() < 1e-10,
                "imported narrow model density changed: {a} vs {b}"
            );
        }
        let mut a = StdRng::seed_from_u64(43210);
        let mut b = StdRng::seed_from_u64(43210);
        for _ in 0..16 {
            let p = base.propose(&mut a, &poses, 0)?;
            let q = model.propose(&mut b, &poses, 0)?;
            assert_eq!(
                serde_json::to_value(p)?,
                serde_json::to_value(q)?,
                "zero update should be an exact frozen-proposal control"
            );
        }
    }
    Ok(())
}
