//! Independent physical-marginal oracle for the conditional variable-K closure.
//! Each replicate starts from the exact hard-sphere + AO equilibrium law in a
//! spherical wall. No burn-in, fitted physical reference, or time-series error
//! estimate is used. The count/Gaussian law is refreshed every sweep; K and eta
//! are retained during every physical proposal, including LOCAL and GCA.
use anyhow::Result;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use std::f64::consts::PI;
use tetramer_mc::{
    conditional::{
        ConditionalConfig, ConditionalEngine, ConditionalInitialization, ConditionalState,
    },
    depletion::{self, GateOptions},
    geometry::{Atom, Environment, Placed, Shape, SphereTree},
    math::*,
    spherical::{self, Container, HalfTurn},
};

const CORE: f64 = 1.;
const WALL: f64 = 4.;
const RD: f64 = 0.7;
const ACTIVITY: f64 = 0.4;
const REPLICATES: usize = 6000;
const ROUNDS: usize = 3;
const SHA: &str = "0000000000000000000000000000000000000000000000000000000000000000";

fn settings() -> ConditionalConfig {
    ConditionalConfig {
        k_max: 3,
        basin_activity: 1.5,
        score_temperature: 1.,
        penalty_strength: 0.3,
        covariance_exponent: 6.,
        pair_cutoff_a: Some(4.8),
        translation_scale_a: 2.,
        angular_length_a: 1.,
        fit_iterations: 3,
        covariance_floor: 0.5,
        covariance_ceiling: 4.,
        residual_scale: 0.2,
        uniform_weight: 0.25,
        initialization: ConditionalInitialization::Random,
        ..ConditionalConfig::default()
    }
}

fn overlap(radius: f64, d: f64) -> f64 {
    if d >= 2. * radius {
        0.
    } else {
        PI * (4. * radius + d) * (2. * radius - d).powi(2) / 12.
    }
}

fn normal<const N: usize>(rng: &mut StdRng) -> [f64; N] {
    std::array::from_fn(|_| StandardNormal.sample(rng))
}

fn direction(rng: &mut StdRng) -> Vec3 {
    let v: Vec3 = normal(rng);
    scale(v, 1. / norm(v))
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

fn reference() -> [f64; 14] {
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
    ]
}

const NAMES: [&str; 14] = [
    "distance",
    "distance_squared",
    "AO_contact",
    "midpoint_squared",
    "absolute_R00",
    "absolute_R00_squared",
    "relative_R00",
    "relative_R00_squared",
    "K_conditional_residual",
    "K_distance_residual",
    "K_rotation_residual",
    "eta0",
    "eta0_squared_residual",
    "eta0_squared_distance_residual",
];

fn observables(
    engine: &ConditionalEngine,
    state: &[Pose],
    auxiliary: &ConditionalState,
) -> Result<[f64; 14]> {
    let fit = engine.fit(state)?;
    let expected_k: f64 = fit
        .log_probabilities
        .iter()
        .enumerate()
        .map(|(k, p)| k as f64 * p.exp())
        .sum();
    let k_residual = auxiliary.k as f64 - expected_k;
    let d = norm(sub(state[0].position, state[1].position));
    let midpoint = scale(add(state[0].position, state[1].position), 0.5);
    let a = rotation(state[0].orientation);
    let relative = matmul(transpose(a), rotation(state[1].orientation))[0][0];
    let eta0 = auxiliary.eta.first().copied().unwrap_or(0.);
    let eta2_residual = eta0 * eta0 - f64::from(auxiliary.k > 0);
    Ok([
        d,
        d * d,
        f64::from(d < 2. * (CORE + RD)),
        dot(midpoint, midpoint),
        a[0][0],
        a[0][0].powi(2),
        relative,
        relative * relative,
        k_residual,
        k_residual * d,
        k_residual * relative,
        eta0,
        eta2_residual,
        eta2_residual * d,
    ])
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
        let error = ((self.sum2 - self.sum * self.sum / self.n as f64).max(0.)
            / (self.n - 1) as f64
            / self.n as f64)
            .sqrt();
        (self.mean() - target) / error.max(1e-15)
    }
}

#[derive(Debug, Default)]
struct Exercised {
    count_histogram: [usize; 4],
    count_changes: usize,
    global_valid: usize,
    global_accepted: usize,
    local_valid: usize,
    local_accepted: usize,
    gate_points: u64,
    gca_points: u64,
    gca_partial: usize,
    gca_accepted: usize,
    shifts: usize,
    max_count_correction: f64,
    max_local_count_correction: f64,
    max_reverse_change: f64,
}

#[allow(clippy::too_many_arguments)]
fn compose(
    engine: &ConditionalEngine,
    tree: &SphereTree,
    wall: &Container,
    state: &mut Vec<Pose>,
    auxiliary: &mut ConditionalState,
    rng: &mut StdRng,
    exercised: &mut Exercised,
) -> Result<()> {
    for _ in 0..ROUNDS {
        let fit = engine.fit(state)?;
        let old_k = auxiliary.k;
        engine.refresh(&fit, auxiliary, rng)?;
        exercised.count_histogram[auxiliary.k] += 1;
        exercised.count_changes += usize::from(old_k != auxiliary.k);
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
                exercised.global_valid += 1;
            } else {
                exercised.local_valid += 1;
            }
            let mut next = state.clone();
            next[moving] = new;
            let next_fit = engine.fit(&next)?;
            let count_correction =
                next_fit.log_probabilities[auxiliary.k] - fit.log_probabilities[auxiliary.k];
            exercised.max_count_correction =
                exercised.max_count_correction.max(count_correction.abs());
            if !learned {
                exercised.max_local_count_correction = exercised
                    .max_local_count_correction
                    .max(count_correction.abs());
            }
            let mut correction = count_correction;
            if let Some(proposal) = proposal {
                let reverse = engine
                    .model(&next_fit, auxiliary)?
                    .log_density(&old, &state[anchor])?;
                let ratio = reverse - proposal.new_log_density.unwrap();
                exercised.max_reverse_change = exercised
                    .max_reverse_change
                    .max((ratio - proposal.log_reverse_forward.unwrap()).abs());
                correction += ratio;
            }
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
                    exercised.global_accepted += 1;
                } else {
                    exercised.local_accepted += 1;
                }
            }
        }
        // The collapsed GCA kernel already balances pi(X). Its MH wrapper
        // supplies only p_K(D_Y)/p_K(D_X), with K and eta held fixed.
        let fit = engine.fit(state)?;
        let mut next = state.clone();
        let stats = spherical::update(
            tree,
            wall,
            &mut next,
            HalfTurn::new(direction(rng))?,
            RD,
            ACTIVITY,
            rng,
        )?;
        exercised.gca_points += stats.poisson_probes;
        exercised.gca_partial += usize::from(stats.partial_flip);
        let next_fit = engine.fit(&next)?;
        let correction =
            next_fit.log_probabilities[auxiliary.k] - fit.log_probabilities[auxiliary.k];
        if rng.random::<f64>().max(f64::MIN_POSITIVE).ln() < correction.min(0.) {
            *state = next;
            exercised.gca_accepted += 1;
        }
        let fit = engine.fit(state)?;
        let mut next = state.clone();
        let axis = direction(rng);
        let u = ((rng.random::<u64>() >> 12) as f64 + 0.5) / 4503599627370496.;
        wall.center_shift(&mut next, axis, u)?;
        let next_fit = engine.fit(&next)?;
        let correction =
            next_fit.log_probabilities[auxiliary.k] - fit.log_probabilities[auxiliary.k];
        if rng.random::<f64>().max(f64::MIN_POSITIVE).ln() < correction.min(0.) {
            *state = next;
        }
        exercised.shifts += 1;
    }
    Ok(())
}

#[test]
fn conditional_count_learned_local_poisson_gca_shift_preserves_exact_ao_marginal() -> Result<()> {
    let tree = SphereTree::new(Shape {
        name: "AO sphere".into(),
        volume: 4. * PI / 3.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius: CORE,
        }],
    })?;
    let wall = Container::new(WALL, &tree)?;
    let engine = ConditionalEngine::new(CORE, WALL, SHA, settings())?;
    let exact = reference();
    let mut initial = [Moments::default(); 14];
    let mut after = [Moments::default(); 14];
    let mut change = [Moments::default(); 14];
    let mut exercised = Exercised::default();
    let mut starts = StdRng::seed_from_u64(202609251439);
    for replicate in 0..REPLICATES {
        let mut state = exact_physical_start(&mut starts);
        let mut auxiliary = engine.initialize(&engine.fit(&state)?, &mut starts)?;
        let before = observables(&engine, &state, &auxiliary)?;
        let mut rng = StdRng::seed_from_u64(985613 + replicate as u64);
        compose(
            &engine,
            &tree,
            &wall,
            &mut state,
            &mut auxiliary,
            &mut rng,
            &mut exercised,
        )?;
        let values = observables(&engine, &state, &auxiliary)?;
        for j in 0..14 {
            initial[j].add(before[j]);
            after[j].add(values[j]);
            change[j].add(values[j] - before[j]);
        }
    }
    for j in 0..14 {
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
            "invalid exact initializer: {}",
            NAMES[j]
        );
        assert!(
            after[j].score(exact[j]).abs() < 5.5,
            "changed joint marginal: {}",
            NAMES[j]
        );
        assert!(
            change[j].score(0.).abs() < 5.5,
            "paired stationarity failure: {}",
            NAMES[j]
        );
    }
    eprintln!("Conditional exact-start kernel exercised: {exercised:?}");
    assert!(
        exercised
            .count_histogram
            .iter()
            .filter(|&&n| n > 100)
            .count()
            >= 2
    );
    assert!(exercised.count_changes > 1000);
    assert!(exercised.global_valid > 500 && exercised.global_accepted > 100);
    assert!(exercised.local_valid > 3000 && exercised.local_accepted > 1000);
    assert!(
        exercised.gate_points > 10000
            && exercised.gca_points > 1000
            && exercised.gca_partial > 1000
    );
    assert!(exercised.gca_accepted > 1000);
    assert!(REPLICATES * ROUNDS - exercised.gca_accepted > 100);
    assert_eq!(exercised.shifts, REPLICATES * ROUNDS);
    assert!(exercised.max_count_correction > 0.1 && exercised.max_reverse_change > 0.1);
    assert!(exercised.max_local_count_correction > 0.1);
    Ok(())
}

#[test]
fn exact_finite_state_control_detects_omitted_conditional_count_correction() -> Result<()> {
    // Use the real fitted count law on two fixed physical configurations, but
    // integrate every possible count and physical transition exactly. This
    // negative control has no statistical-power or random-seed assumption.
    let engine = ConditionalEngine::new(CORE, WALL, SHA, settings())?;
    let pose = |x| Pose {
        position: [x, 0., 0.],
        orientation: [1., 0., 0., 0.],
    };
    let fits = [
        engine.fit(&[pose(-1.05), pose(1.05)])?,
        engine.fit(&[pose(-2.6), pose(2.6)])?,
    ];
    let p: Vec<Vec<f64>> = fits
        .iter()
        .map(|f| f.log_probabilities.iter().map(|v| v.exp()).collect())
        .collect();
    let nk = p[0].len();
    for row in &p {
        assert!((row.iter().sum::<f64>() - 1.).abs() < 1e-12);
    }
    let count_difference: f64 = p[0].iter().zip(&p[1]).map(|(a, b)| (a - b).abs()).sum();
    assert!(
        count_difference > 0.01,
        "negative control requires state-dependent count probabilities"
    );
    let pi = [0.35, 0.65];
    let target: Vec<f64> = (0..2 * nk)
        .map(|i| pi[i / nk] * p[i / nk][i % nk])
        .collect();
    let transition = |correct: bool| {
        let mut matrix = vec![vec![0.; 2 * nk]; 2 * nk];
        for (i, row) in matrix.iter_mut().enumerate() {
            let x = i / nk;
            let y = 1 - x;
            for k in 0..nk {
                let q = |site| {
                    if site == 0 {
                        0.85 / (k + 1) as f64
                    } else {
                        0.2 + 0.1 * k as f64
                    }
                };
                let mut ratio = pi[y] * q(y) / (pi[x] * q(x));
                if correct {
                    ratio *= p[y][k] / p[x][k];
                }
                let leave = q(x) * ratio.min(1.);
                row[y * nk + k] += p[x][k] * leave;
                row[x * nk + k] += p[x][k] * (1. - leave);
            }
            assert!((row.iter().sum::<f64>() - 1.).abs() < 1e-12);
        }
        matrix
    };
    let apply = |matrix: Vec<Vec<f64>>| -> Vec<f64> {
        (0..2 * nk)
            .map(|j| (0..2 * nk).map(|i| target[i] * matrix[i][j]).sum())
            .collect()
    };
    let good = apply(transition(true));
    let bad = apply(transition(false));
    let good_residual = good
        .iter()
        .zip(&target)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    let bad_physical_drift = (bad[..nk].iter().sum::<f64>() - pi[0]).abs();
    eprintln!(
        "Exact finite-state conditional control: count_L1={count_difference:.8}, corrected_residual={good_residual:.3e}, omitted_count_physical_drift={bad_physical_drift:.8}"
    );
    assert!(good_residual < 1e-13);
    assert!(
        bad_physical_drift > 1e-4,
        "omitting count correction must bias the physical marginal"
    );
    Ok(())
}
