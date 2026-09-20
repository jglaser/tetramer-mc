//! Physical-marginal stationarity from independent EXACT equilibrium starts.
//!
//! This is not a long-chain/burn-in test. Two hard spheres in an origin-centered
//! spherical wall have an analytic AO weight exp(z * pair exclusion overlap).
//! Each independent replicate starts from that law and the independent RJ prior,
//! then uses the actual proposal, conditional model, Poisson gate, RJ, GCA and
//! center-shift implementations. The statistical checks supplement balance and
//! runner-replay tests; they do not establish ergodicity or mixing speedups.
use anyhow::Result;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde_json::json;
use std::f64::consts::PI;
use tetramer_mc::{
    auxiliary::{self, AuxiliaryConfig},
    depletion::{self, GateOptions},
    geometry::{Atom, Environment, Placed, Shape, SphereTree},
    math::*,
    proposal::FrozenRelativePoseProposal,
    rj::{RjConfig, RjState},
    spherical::{self, Container, HalfTurn},
};

const CORE: f64 = 1.;
const WALL: f64 = 4.;
const RD: f64 = 0.7;
const ACTIVITY: f64 = 0.4;
const REPLICATES: usize = 6000;
const ROUNDS: usize = 3;
const SHA: &str = "0000000000000000000000000000000000000000000000000000000000000000";

fn overlap(radius: f64, distance: f64) -> f64 {
    if distance >= 2. * radius {
        0.
    } else {
        PI * (4. * radius + distance) * (2. * radius - distance).powi(2) / 12.
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
    // Radius C U^(1/3) gives a uniform center in B_C; sphere-core walls mean
    // C=R-a. Independent normalized 4D normal gives normalized Haar on SO(3).
    let position = scale(direction(rng), (WALL - CORE) * rng.random::<f64>().cbrt());
    let q: [f64; 4] = normal(rng);
    let length = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    Pose {
        position,
        orientation: q.map(|v| v / length),
    }
}

fn exact_physical_start(rng: &mut StdRng) -> Vec<Pose> {
    // Rejection from two independent uniform center balls, using the maximum
    // allowed AO overlap at hard contact. Orientations remain independent Haar.
    let maximum = overlap(CORE + RD, 2. * CORE);
    loop {
        let state = vec![independent_pose(rng), independent_pose(rng)];
        let d = norm(sub(state[0].position, state[1].position));
        if d < 2. * CORE {
            continue;
        }
        let log_ratio = ACTIVITY * (overlap(CORE + RD, d) - maximum);
        if rng.random::<f64>().max(f64::MIN_POSITIVE).ln() < log_ratio {
            return state;
        }
    }
}

fn count_prior(config: &RjConfig) -> Vec<f64> {
    let mut p = vec![0.; config.max_components + 1];
    let mut term = 1.;
    for (k, probability) in p.iter_mut().enumerate() {
        if k > 0 {
            term *= config.poisson_mean / k as f64;
        }
        if k >= config.min_components {
            *probability = term;
        }
    }
    let sum: f64 = p.iter().sum();
    p.iter_mut().for_each(|v| *v /= sum);
    p
}

fn categorical(weights: &[f64], rng: &mut StdRng) -> usize {
    let u = rng.random::<f64>();
    let mut sum = 0.;
    for (i, p) in weights.iter().enumerate() {
        sum += p;
        if u < sum {
            return i;
        }
    }
    weights.len() - 1
}

fn exact_joint_start(
    config: &RjConfig,
    prior: &[f64],
    labels: &[f64],
    rng: &mut StdRng,
) -> (Vec<Pose>, RjState) {
    let state = exact_physical_start(rng);
    let k = categorical(prior, rng);
    let rj = RjState {
        labels: (0..k).map(|_| categorical(labels, rng)).collect(),
        eta: (0..k).map(|_| normal(rng)).collect(),
    };
    rj.validate(config, labels.len()).unwrap();
    (state, rj)
}

fn dictionary() -> Result<FrozenRelativePoseProposal> {
    // Deliberately anisotropic full-covariance proposal components although the
    // physical particles are spheres. Haar and geometry moments then exercise
    // both rotational/translation proposal corrections, including cross terms.
    let covariance: Vec<Vec<f64>> = (0..6)
        .map(|i| {
            (0..6)
                .map(|j| {
                    if i == j {
                        if i < 3 { 4. } else { 0.6 }
                    } else if (i == 0 && j == 4) || (i == 4 && j == 0) {
                        0.45
                    } else {
                        0.
                    }
                })
                .collect()
        })
        .collect();
    let model = json!({
        "angular_length":1., "weights":[0.4,0.3,0.2,0.1],
        "anchors":[
            {"position":[0.,0.,0.],"rotation":IDENTITY},
            {"position":[0.,0.,0.],"rotation":rotation([0.,1.,0.,0.])},
            {"position":[0.,0.,0.],"rotation":rotation([0.,0.,1.,0.])},
            {"position":[0.,0.,0.],"rotation":rotation([0.,0.,0.,1.])}],
        "means":[[2.4,0.,0.,0.,0.,0.],[-2.4,0.,0.,0.,0.,0.],[0.,2.4,0.,0.,0.,0.],[0.,0.,2.4,0.,0.,0.]],
        "covariances":[covariance.clone(),covariance.clone(),covariance.clone(),covariance],
        "shape_sha256":SHA, "coordinate_convention":"anchor-body-relative"});
    FrozenRelativePoseProposal::from_json_str_open(
        &model.to_string(),
        [2. * (WALL + CORE); 3],
        0.2,
        SHA,
    )
}

#[derive(Default, Debug)]
struct Exercised {
    physical_attempts: usize,
    hard_valid: usize,
    accepted: usize,
    jumps: usize,
    births: usize,
    deaths: usize,
    boundary_nulls: usize,
    gate_points: u64,
    gca_points: u64,
    gca_partial: usize,
    shifts: usize,
    stale_log_correction_max: f64,
}

#[allow(clippy::too_many_arguments)]
fn compose(
    state: &mut Vec<Pose>,
    rj_state: &mut RjState,
    base: &FrozenRelativePoseProposal,
    tree: &SphereTree,
    wall: &Container,
    rj_config: &RjConfig,
    auxiliary: &AuxiliaryConfig,
    rng: &mut StdRng,
    stale_reverse: bool,
    exercised: &mut Exercised,
) -> Result<()> {
    let labels = base.component_weights();
    for _ in 0..ROUNDS {
        rj_state.refresh_eta(rng);
        for _ in 0..rj_config.attempts_per_sweep {
            let jump = rj_state.update(rj_config, &labels, rng)?;
            exercised.jumps += 1;
            exercised.boundary_nulls += usize::from(jump.boundary_null);
            if jump.accepted {
                if jump.birth {
                    exercised.births += 1;
                } else {
                    exercised.deaths += 1;
                }
            }
        }
        let selected = base.selected_components(&rj_state.labels)?;
        let forward = auxiliary::model(&selected, state, &rj_state.eta, auxiliary)?;
        let moving = rng.random_range(0..2);
        exercised.physical_attempts += 1;
        let proposal = forward.propose(rng, state, moving)?;
        if let Some(new) = proposal.candidate {
            let old = state[moving];
            let spectator = 1 - moving;
            let env = Environment {
                tree,
                fixed: vec![Placed::new(state[spectator])],
                labels: vec![(spectator, [0; 3])],
                rd: RD,
            };
            if wall.contains(new) && env.hard_valid(new) {
                exercised.hard_valid += 1;
                let mut next = state.clone();
                next[moving] = new;
                let reverse = auxiliary::model(&selected, &next, &rj_state.eta, auxiliary)?;
                let corrected = reverse.log_density(&old, &state[spectator])?
                    - proposal.new_log_density.unwrap();
                let stale = proposal.log_reverse_forward.unwrap();
                exercised.stale_log_correction_max = exercised
                    .stale_log_correction_max
                    .max((corrected - stale).abs());
                let correction = if stale_reverse { stale } else { corrected };
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
                    state[moving] = new;
                    exercised.accepted += 1;
                }
            }
        }
        let transform = HalfTurn::new(direction(rng))?;
        let stats = spherical::update(tree, wall, state, transform, RD, ACTIVITY, rng)?;
        exercised.gca_points += stats.poisson_probes;
        exercised.gca_partial += usize::from(stats.partial_flip);
        let axis = direction(rng);
        let u = ((rng.random::<u64>() >> 12) as f64 + 0.5) / 4503599627370496.;
        wall.center_shift(state, axis, u)?;
        exercised.shifts += 1;
    }
    Ok(())
}

fn integrate(function: impl Fn(f64) -> f64, lo: f64, hi: f64) -> f64 {
    let intervals = 8192;
    let h = (hi - lo) / intervals as f64;
    let mut sum = function(lo) + function(hi);
    for i in 1..intervals {
        sum += if i % 2 == 0 { 2. } else { 4. } * function(lo + i as f64 * h);
    }
    sum * h / 3.
}

fn midpoint_integral(center_radius: f64, d: f64) -> f64 {
    // Integral |m|^2 over B_C(-d/2) intersect B_C(d/2), by circular slices.
    let h = d / 2.;
    let height = center_radius - h;
    let a = center_radius.powi(2) - h * h;
    2. * PI
        * (0.5 * a * a * height - a * h * height.powi(2) + (2. / 3.) * h * h * height.powi(3)
            - 0.1 * height.powi(5))
}

#[derive(Clone)]
struct Reference {
    distance: f64,
    distance2: f64,
    contact: f64,
    midpoint2: f64,
    count: f64,
}

fn reference(prior: &[f64]) -> Reference {
    let center_radius = WALL - CORE;
    let weight = |d| d * d * overlap(center_radius, d) * (ACTIVITY * overlap(CORE + RD, d)).exp();
    let z = integrate(weight, 2. * CORE, 2. * center_radius);
    Reference {
        distance: integrate(|d| d * weight(d), 2. * CORE, 2. * center_radius) / z,
        distance2: integrate(|d| d * d * weight(d), 2. * CORE, 2. * center_radius) / z,
        contact: integrate(weight, 2. * CORE, 2. * (CORE + RD)) / z,
        midpoint2: integrate(
            |d| {
                d * d
                    * midpoint_integral(center_radius, d)
                    * (ACTIVITY * overlap(CORE + RD, d)).exp()
            },
            2. * CORE,
            2. * center_radius,
        ) / z,
        count: prior.iter().enumerate().map(|(k, p)| k as f64 * p).sum(),
    }
}

const NAMES: [&str; 15] = [
    "distance",
    "distance_squared",
    "depletion_contact",
    "midpoint_squared",
    "absolute_R00",
    "absolute_R00_squared",
    "relative_R00",
    "relative_R00_squared",
    "K",
    "K_distance_cross",
    "K_contact_cross",
    "K_rotation_cross",
    "first_label0",
    "first_eta0",
    "first_eta0_squared",
];

fn observables(state: &[Pose], auxiliary: &RjState, reference: &Reference) -> [f64; 15] {
    let d = norm(sub(state[0].position, state[1].position));
    let contact = f64::from(d < 2. * (CORE + RD));
    let midpoint = scale(add(state[0].position, state[1].position), 0.5);
    let a = rotation(state[0].orientation);
    let b = rotation(state[1].orientation);
    let relative = matmul(transpose(a), b)[0][0];
    let k = auxiliary.labels.len() as f64;
    [
        d,
        d * d,
        contact,
        dot(midpoint, midpoint),
        a[0][0],
        a[0][0].powi(2),
        relative,
        relative.powi(2),
        k,
        (k - reference.count) * (d - reference.distance),
        (k - reference.count) * (contact - reference.contact),
        (k - reference.count) * relative,
        f64::from(auxiliary.labels[0] == 0),
        auxiliary.eta[0][0],
        auxiliary.eta[0][0].powi(2),
    ]
}

#[derive(Clone, Copy, Default)]
struct Moments {
    n: usize,
    sum: f64,
    sum2: f64,
}
impl Moments {
    fn add(&mut self, value: f64) {
        self.n += 1;
        self.sum += value;
        self.sum2 += value * value;
    }
    fn mean(self) -> f64 {
        self.sum / self.n as f64
    }
    fn error(self) -> f64 {
        ((self.sum2 - self.sum * self.sum / self.n as f64).max(0.)
            / (self.n - 1) as f64
            / self.n as f64)
            .sqrt()
    }
    fn score(self, target: f64) -> f64 {
        (self.mean() - target) / self.error().max(1e-15)
    }
}

#[test]
fn actual_rj_learned_poisson_gca_shift_preserves_exact_two_sphere_marginal() -> Result<()> {
    let base = dictionary()?;
    let tree = SphereTree::new(Shape {
        name: "AO sphere".into(),
        volume: 4. * PI / 3.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius: CORE,
        }],
    })?;
    let wall = Container::new(WALL, &tree)?;
    let settings = RjConfig {
        min_components: 1,
        max_components: 6,
        initial_components: 2,
        poisson_mean: 2.5,
        attempts_per_sweep: 2,
    };
    // Deliberately stronger history-free guidance than the production default
    // stresses reconstruction of the reverse model. No parameters are fit.
    let guidance = AuxiliaryConfig {
        gain: 1.2,
        noise: 0.35,
        cutoff: 8.,
        clip: 4.,
        shrinkage: 1.,
    };
    let prior = count_prior(&settings);
    let exact = reference(&prior);
    let expected = [
        exact.distance,
        exact.distance2,
        exact.contact,
        exact.midpoint2,
        0.,
        1. / 3.,
        0.,
        1. / 3.,
        exact.count,
        0.,
        0.,
        0.,
        0.4,
        0.,
        1.,
    ];
    let mut initial = [Moments::default(); 15];
    let mut after = [Moments::default(); 15];
    let mut change = [Moments::default(); 15];
    let mut stale_change = [Moments::default(); 15];
    let mut counters = Exercised::default();
    let mut stale_counters = Exercised::default();
    let mut starts = StdRng::seed_from_u64(202609230921);
    for replicate in 0..REPLICATES {
        let (start, rj_start) =
            exact_joint_start(&settings, &prior, &base.component_weights(), &mut starts);
        let before = observables(&start, &rj_start, &exact);
        let mut state = start.clone();
        let mut rj = rj_start.clone();
        let mut rng = StdRng::seed_from_u64(8301551 + replicate as u64);
        let mut stale_rng = StdRng::seed_from_u64(8301551 + replicate as u64);
        compose(
            &mut state,
            &mut rj,
            &base,
            &tree,
            &wall,
            &settings,
            &guidance,
            &mut rng,
            false,
            &mut counters,
        )?;
        let values = observables(&state, &rj, &exact);
        let mut bad_state = start;
        let mut bad_rj = rj_start;
        compose(
            &mut bad_state,
            &mut bad_rj,
            &base,
            &tree,
            &wall,
            &settings,
            &guidance,
            &mut stale_rng,
            true,
            &mut stale_counters,
        )?;
        let bad = observables(&bad_state, &bad_rj, &exact);
        for j in 0..15 {
            initial[j].add(before[j]);
            after[j].add(values[j]);
            change[j].add(values[j] - before[j]);
            stale_change[j].add(bad[j] - before[j]);
        }
    }
    for j in 0..15 {
        eprintln!(
            "{}: exact={:.7} initial={:.7} after={:.7} paired_z={:.3} stale_paired_z={:.3}",
            NAMES[j],
            expected[j],
            initial[j].mean(),
            after[j].mean(),
            change[j].score(0.),
            stale_change[j].score(0.)
        );
        assert!(
            initial[j].score(expected[j]).abs() < 5.5,
            "iid initializer disagrees with analytic {}",
            NAMES[j]
        );
        assert!(
            after[j].score(expected[j]).abs() < 5.5,
            "physical/RJ marginal changed: {}",
            NAMES[j]
        );
        assert!(
            change[j].score(0.).abs() < 5.5,
            "paired stationarity failure: {}",
            NAMES[j]
        );
    }
    eprintln!("actual implementation exercised: {counters:?}");
    eprintln!(
        "Correct composition maximum |paired z|={:.3} across 15 predeclared observables from {REPLICATES} independent exact starts and {ROUNDS} rounds; acceptance bound 5.5",
        change.iter().map(|m| m.score(0.).abs()).fold(0., f64::max)
    );
    eprintln!(
        "Stale-reverse mean-drift control is underpowered here (not required to fail); maximum |paired z|={:.3}. The separate exact Poisson-sum flux control detects this defect without a sampling-power assumption.",
        stale_change
            .iter()
            .map(|m| m.score(0.).abs())
            .fold(0., f64::max)
    );
    assert!(counters.accepted > 300 && counters.hard_valid > 1000);
    assert!(counters.births > 1000 && counters.deaths > 1000 && counters.boundary_nulls > 1000);
    assert!(
        counters.gate_points > 10000 && counters.gca_points > 1000 && counters.gca_partial > 1000
    );
    assert_eq!(counters.shifts, REPLICATES * ROUNDS);
    assert!(
        counters.stale_log_correction_max > 1.,
        "fixture must exercise a materially different reverse model"
    );
    Ok(())
}

fn poisson_probabilities(mean: f64) -> Vec<f64> {
    // All means in this control are below10. The omitted mass beyond99 is far
    // smaller than floating-point roundoff, checked explicitly below.
    assert!(mean >= 0. && mean < 10.);
    let mut p = vec![(-mean).exp()];
    for k in 1..100 {
        p.push(p[k - 1] * mean / k as f64);
    }
    assert!((p.iter().sum::<f64>() - 1.).abs() < 2e-14);
    p
}

fn exact_poisson_acceptance(correction: f64, gained_volume: f64, lost_volume: f64) -> f64 {
    let lambda = 4. * ACTIVITY;
    let coefficient = (ACTIVITY / lambda).ln_1p();
    let gained = poisson_probabilities(lambda * gained_volume);
    let lost = poisson_probabilities((lambda + ACTIVITY) * lost_volume);
    gained
        .iter()
        .enumerate()
        .map(|(g, pg)| {
            lost.iter()
                .enumerate()
                .map(|(l, pl)| {
                    pg * pl
                        * (correction + coefficient * (g as f64 - l as f64))
                            .min(0.)
                            .exp()
                })
                .sum::<f64>()
        })
        .sum()
}

#[test]
fn exact_poisson_flux_control_detects_stale_reverse_reconstruction() -> Result<()> {
    let base = dictionary()?.selected_components(&[0, 0, 1, 2])?;
    let guidance = AuxiliaryConfig {
        gain: 1.2,
        noise: 0.35,
        cutoff: 8.,
        clip: 4.,
        shrinkage: 1.,
    };
    let eta: Vec<[f64; 6]> = (0..4)
        .map(|k| std::array::from_fn(|d| ((k * 6 + d) as f64 * 0.7).sin()))
        .collect();
    let anchor = Pose {
        position: [0.; 3],
        orientation: [1., 0., 0., 0.],
    };
    let mut worst_correct: f64 = 0.;
    let mut worst_stale: f64 = 0.;
    for (old_d, new_d) in [(2.15, 2.8), (2.4, 2.9), (2.05, 2.6)] {
        let old = Pose {
            position: [-old_d, 0., 0.],
            orientation: [(PI / 6.).cos(), 0., 0., (PI / 6.).sin()],
        };
        let new = Pose {
            position: [new_d, 0., 0.],
            ..old
        };
        let x = vec![old, anchor];
        let y = vec![new, anchor];
        let mx = auxiliary::model(&base, &x, &eta, &guidance)?;
        let my = auxiliary::model(&base, &y, &eta, &guidance)?;
        let xy = mx.log_density(&new, &anchor)?;
        let yx = my.log_density(&old, &anchor)?;
        let old_overlap = overlap(CORE + RD, old_d);
        let new_overlap = overlap(CORE + RD, new_d);
        // In the moving body's frame the old and new spectator exclusion balls
        // lie on opposite sides, separated by old_d+new_d>2(CORE+RD). Thus their
        // overlap caps inside the moving exclusion ball are disjoint. The exact
        // gained/lost volumes are individually the analytic pair overlaps.
        assert!(old_d + new_d > 2. * (CORE + RD));
        let flux = |forward_correction: f64, reverse_correction: f64| {
            let f = (ACTIVITY * old_overlap + xy).exp()
                * exact_poisson_acceptance(forward_correction, new_overlap, old_overlap);
            let r = (ACTIVITY * new_overlap + yx).exp()
                * exact_poisson_acceptance(reverse_correction, old_overlap, new_overlap);
            (f - r).abs() / f.max(r)
        };
        let correct = flux(yx - xy, xy - yx);
        // The defective implementation recomputes a model at the start of each
        // direction but evaluates both endpoint densities using that same model.
        let stale = flux(
            mx.log_density(&old, &anchor)? - xy,
            my.log_density(&new, &anchor)? - yx,
        );
        worst_correct = worst_correct.max(correct);
        worst_stale = worst_stale.max(stale);
    }
    eprintln!(
        "Independent Poisson-sum relative flux defect: correct={worst_correct:.3e}, stale={worst_stale:.6}"
    );
    assert!(
        worst_correct < 2e-12,
        "correct reconstructed model must satisfy exact noise-averaged flux"
    );
    assert!(
        worst_stale > 0.05,
        "negative control must detect the stale reverse error"
    );
    Ok(())
}
