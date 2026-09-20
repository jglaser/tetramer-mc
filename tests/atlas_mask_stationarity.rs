//! Independent physical and finite-state checks of the retained-mask target
//! pi(X) phi(eta) rho(A). Mask probabilities are independent of X; forward and
//! reverse physical proposals retain the same A and eta, but rebuild the fit.
use anyhow::Result;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde_json::json;
use std::f64::consts::PI;
use tetramer_mc::{
    atlas_mask::{AtlasMaskConfig, AtlasMaskEngine, AtlasMaskState},
    atlas_transport::{AtlasTransportConfig, AtlasTransportEngine, AtlasTransportState},
    depletion::{self, GateOptions},
    geometry::{Atom, Environment, Placed, Shape, SphereTree},
    math::*,
    proposal::{FrozenRelativePoseProposal, GaussianComponentParameters},
    spherical::{self, Container, HalfTurn},
};

const SHA: &str = "0000000000000000000000000000000000000000000000000000000000000000";
const WALL: f64 = 4.;
const CORE: f64 = 1.;
const RD: f64 = 0.7;
const Z: f64 = 0.4;
const REPLICATES: usize = 5000;
const ROUNDS: usize = 3;

fn base_model() -> Result<FrozenRelativePoseProposal> {
    let components = (0..2)
        .map(|j| {
            let mut lower = [[0.; 6]; 6];
            for (i, row) in lower.iter_mut().enumerate() {
                row[i] = 0.85 + 0.12 * i as f64 + 0.2 * j as f64;
            }
            lower[3][0] = 0.25;
            lower[4][1] = -0.2;
            GaussianComponentParameters {
                anchor_position: if j == 0 { [2.5, 0., 0.] } else { [0., 2.5, 0.] },
                anchor_rotation: IDENTITY,
                mean: [0.; 6],
                covariance: std::array::from_fn(|a| {
                    std::array::from_fn(|b| (0..6).map(|i| lower[a][i] * lower[b][i]).sum())
                }),
                weight: if j == 0 { 0.7 } else { 0.3 },
            }
        })
        .collect();
    FrozenRelativePoseProposal::from_components_open(components, 1., [8.; 3], 0.2, SHA, SHA)
}

fn engines() -> Result<(AtlasTransportEngine, AtlasMaskEngine)> {
    let model = base_model()?;
    let weights = model
        .component_parameters()
        .iter()
        .map(|p| p.weight)
        .collect::<Vec<_>>();
    let transport: AtlasTransportConfig = serde_json::from_value(json!({
        "mean_gain":0.8,"covariance_gain":0.6,"weight_gain":0.8,
        "mean_noise":0.2,"covariance_noise":0.2,"weight_noise":0.2,
        "assignment_cutoff":100.,"residual_clip":4.,"shrinkage":0.5,
        "initialization":"random"
    }))?;
    let mask: AtlasMaskConfig = serde_json::from_value(json!({
        "activity":1.4,"label_law":"atlas_weight","initial_full":false
    }))?;
    Ok((
        AtlasTransportEngine::new(model, transport)?,
        AtlasMaskEngine::new(&weights, mask)?,
    ))
}

fn masks() -> [AtlasMaskState; 4] {
    [vec![], vec![0], vec![1], vec![0, 1]].map(|labels| AtlasMaskState { labels })
}

fn normal<const N: usize>(rng: &mut StdRng) -> [f64; N] {
    std::array::from_fn(|_| StandardNormal.sample(rng))
}

fn direction(rng: &mut StdRng) -> Vec3 {
    let a: Vec3 = normal(rng);
    scale(a, 1. / norm(a))
}

fn lens(radius: f64, separation: f64) -> f64 {
    if separation >= 2. * radius {
        0.
    } else {
        PI * (4. * radius + separation) * (2. * radius - separation).powi(2) / 12.
    }
}

fn uniform_pose(rng: &mut StdRng) -> Pose {
    let position = scale(direction(rng), (WALL - CORE) * rng.random::<f64>().cbrt());
    let q: [f64; 4] = normal(rng);
    let length = q.iter().map(|x| x * x).sum::<f64>().sqrt();
    Pose {
        position,
        orientation: q.map(|x| x / length),
    }
}

fn exact_start(rng: &mut StdRng) -> Vec<Pose> {
    let bound = lens(CORE + RD, 2. * CORE);
    loop {
        let poses = vec![uniform_pose(rng), uniform_pose(rng)];
        let distance = norm(sub(poses[0].position, poses[1].position));
        if distance >= 2. * CORE
            && rng.random::<f64>().max(f64::MIN_POSITIVE).ln()
                < Z * (lens(CORE + RD, distance) - bound)
        {
            return poses;
        }
    }
}

fn integrate(f: impl Fn(f64) -> f64, low: f64, high: f64) -> f64 {
    let n = 8192;
    let h = (high - low) / n as f64;
    let mut sum = f(low) + f(high);
    for i in 1..n {
        sum += if i % 2 == 0 { 2. } else { 4. } * f(low + h * i as f64);
    }
    sum * h / 3.
}

const NAMES: [&str; 14] = [
    "distance",
    "AO_contact",
    "absolute_R00",
    "absolute_R00_squared",
    "relative_R00",
    "relative_R00_squared",
    "eta",
    "eta_variance_residual",
    "mask_count_residual",
    "mask_count_distance",
    "mask_label_residual",
    "mask_label_rotation",
    "mask_label_eta",
    "eta_distance",
];

fn expected() -> [f64; 14] {
    // Integrating over center of mass gives the intersection volume of two
    // radius (WALL-CORE) balls; no simulation histogram supplies this reference.
    let weight = |d: f64| d * d * lens(WALL - CORE, d) * (Z * lens(CORE + RD, d)).exp();
    let denominator = integrate(weight, 2. * CORE, 2. * (WALL - CORE));
    [
        integrate(|d| d * weight(d), 2. * CORE, 2. * (WALL - CORE)) / denominator,
        integrate(weight, 2. * CORE, 2. * (CORE + RD)) / denominator,
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
        0.,
    ]
}

fn observable(
    poses: &[Pose],
    auxiliary: &AtlasTransportState,
    mask: &AtlasMaskState,
    expected_count: f64,
    label_probability: f64,
) -> [f64; 14] {
    let distance = norm(sub(poses[0].position, poses[1].position));
    let r = rotation(poses[0].orientation);
    let relative = matmul(transpose(r), rotation(poses[1].orientation))[0][0];
    let eta = auxiliary.eta[0];
    let count = mask.labels.len() as f64 - expected_count;
    let label = f64::from(mask.labels.contains(&0)) - label_probability;
    [
        distance,
        f64::from(distance < 2. * (CORE + RD)),
        r[0][0],
        r[0][0].powi(2),
        relative,
        relative * relative,
        eta,
        eta * eta - 1.,
        count,
        count * distance,
        label,
        label * relative,
        label * eta,
        eta * distance,
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
    fn z(self, reference: f64) -> f64 {
        let se = ((self.sum2 - self.sum * self.sum / self.n as f64).max(0.)
            / (self.n - 1) as f64
            / self.n as f64)
            .sqrt();
        (self.mean() - reference) / se.max(1e-15)
    }
}

#[derive(Default, Debug)]
struct Work {
    masks: [usize; 3],
    learned_valid: usize,
    learned_accepted: usize,
    nonempty_accepted: usize,
    local_accepted: usize,
    gate_points: u64,
    gca_points: u64,
    partial_gca: usize,
    shifts: usize,
    reverse_change: f64,
}

#[test]
fn masked_atlas_with_gibbs_local_gca_shift_preserves_exact_ao_joint_law() -> Result<()> {
    let (engine, mask_engine) = engines()?;
    let tree = SphereTree::new(Shape {
        name: "masked atlas AO sphere".into(),
        volume: 4. * PI / 3.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius: CORE,
        }],
    })?;
    let wall = Container::new(WALL, &tree)?;
    let mut mean_count = 0.;
    let mut label_probability = 0.;
    for mask in masks() {
        let probability = mask_engine.log_probability(&mask)?.exp();
        mean_count += probability * mask.labels.len() as f64;
        label_probability += probability * f64::from(mask.labels.contains(&0));
    }
    let reference = expected();
    let mut before_moments = [Moments::default(); 14];
    let mut after_moments = [Moments::default(); 14];
    let mut differences = [Moments::default(); 14];
    let mut work = Work::default();
    let mut starts = StdRng::seed_from_u64(611034022);
    for replicate in 0..REPLICATES {
        let mut poses = exact_start(&mut starts);
        let mut auxiliary = engine.initialize(&engine.fit(&poses)?, &mut starts)?;
        let mut mask = mask_engine.initialize(&mut starts)?;
        let initial = observable(&poses, &auxiliary, &mask, mean_count, label_probability);
        let mut rng = StdRng::seed_from_u64(940240 + replicate as u64);
        for round in 0..ROUNDS {
            work.masks[mask.labels.len()] += 1;
            for learned in [true, false] {
                let moving = rng.random_range(0..2);
                let anchor = 1 - moving;
                let old = poses[moving];
                let forward = if learned {
                    let model = engine
                        .model(&engine.fit(&poses)?, &auxiliary)?
                        .weighted_subset(&mask.labels)?;
                    Some(model.propose(&mut rng, &poses, moving)?)
                } else {
                    None
                };
                let proposed = match &forward {
                    Some(proposal) => proposal.candidate,
                    None => Some(Pose {
                        position: add(old.position, scale(normal(&mut rng), 0.7)),
                        orientation: quaternion(matmul(
                            cayley(scale(normal(&mut rng), 0.3)),
                            rotation(old.orientation),
                        )),
                    }),
                };
                let Some(candidate) = proposed else {
                    continue;
                };
                let env = Environment {
                    tree: &tree,
                    fixed: vec![Placed::new(poses[anchor])],
                    labels: vec![(anchor, [0; 3])],
                    rd: RD,
                };
                if !wall.contains(candidate) || !env.hard_valid(candidate) {
                    continue;
                }
                let mut next = poses.clone();
                next[moving] = candidate;
                let correction = if let Some(proposal) = forward {
                    work.learned_valid += 1;
                    let reverse = engine
                        .model(&engine.fit(&next)?, &auxiliary)?
                        .weighted_subset(&mask.labels)?
                        .log_density(&old, &poses[anchor])?;
                    let ratio = reverse - proposal.new_log_density.unwrap();
                    work.reverse_change = work
                        .reverse_change
                        .max((ratio - proposal.log_reverse_forward.unwrap()).abs());
                    ratio
                } else {
                    0.
                };
                let gate = depletion::sample(
                    &mut rng,
                    &env,
                    old,
                    candidate,
                    4. * Z,
                    Z,
                    GateOptions {
                        max_cells: 1,
                        max_depth: 0,
                        min_width: 0.,
                    },
                )?;
                work.gate_points += gate.raw_points;
                if rng.random::<f64>().max(f64::MIN_POSITIVE).ln()
                    < (correction + gate.log_weight).min(0.)
                {
                    poses = next;
                    if learned {
                        work.learned_accepted += 1;
                        work.nonempty_accepted += usize::from(!mask.labels.is_empty());
                    } else {
                        work.local_accepted += 1;
                    }
                }
            }
            // Both kernels act only on pi(X); rho(A)phi(eta) remains untouched.
            let half_turn = HalfTurn::new(direction(&mut rng))?;
            let stats = spherical::update(&tree, &wall, &mut poses, half_turn, RD, Z, &mut rng)?;
            work.gca_points += stats.poisson_probes;
            work.partial_gca += usize::from(stats.partial_flip);
            let u = ((rng.random::<u64>() >> 12) as f64 + 0.5) / 4503599627370496.;
            wall.center_shift(&mut poses, direction(&mut rng), u)?;
            work.shifts += 1;
            if round + 1 < ROUNDS {
                engine.refresh(&mut auxiliary, &mut rng)?;
                mask_engine.refresh(&mut mask, &mut rng)?;
            }
        }
        // No terminal auxiliary redraw: cross moments retain sensitivity to
        // correlations wrongly induced by the final physical sweep.
        let final_values = observable(&poses, &auxiliary, &mask, mean_count, label_probability);
        for j in 0..14 {
            before_moments[j].add(initial[j]);
            after_moments[j].add(final_values[j]);
            differences[j].add(final_values[j] - initial[j]);
        }
    }
    for j in 0..14 {
        eprintln!(
            "{} expected={:.6} before={:.6} after={:.6} paired_z={:.3}",
            NAMES[j],
            reference[j],
            before_moments[j].mean(),
            after_moments[j].mean(),
            differences[j].z(0.)
        );
        assert!(
            before_moments[j].z(reference[j]).abs() < 5.5,
            "exact initial {}",
            NAMES[j]
        );
        assert!(
            after_moments[j].z(reference[j]).abs() < 5.5,
            "final marginal {}",
            NAMES[j]
        );
        assert!(
            differences[j].z(0.).abs() < 5.5,
            "paired stationarity {}",
            NAMES[j]
        );
    }
    eprintln!("Masked atlas AO kernels exercised: {work:?}");
    assert!(work.masks.iter().all(|&n| n > 1000));
    assert!(work.learned_valid > 500 && work.learned_accepted > 150);
    assert!(work.nonempty_accepted > 100 && work.local_accepted > 1000);
    assert!(work.gate_points > 10000 && work.gca_points > 1000 && work.partial_gca > 1000);
    assert_eq!(work.shifts, REPLICATES * ROUNDS);
    assert!(work.reverse_change > 0.1);
    Ok(())
}

#[test]
fn exact_discrete_pose_flux_detects_wrong_reverse_mask() -> Result<()> {
    let (engine, mask_engine) = engines()?;
    let anchor = Pose {
        position: [0.; 3],
        orientation: [1., 0., 0., 0.],
    };
    let candidates = [
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
    let physical = [0.1, 0.2, 0.3, 0.4];
    let auxiliary = AtlasTransportState { eta: vec![0.; 55] };
    let mask_list = masks();
    let rho: [f64; 4] =
        std::array::from_fn(|a| mask_engine.log_probability(&mask_list[a]).unwrap().exp());
    let mut q = [[[0.; 4]; 4]; 4];
    for a in 0..4 {
        for x in 0..4 {
            let model = engine
                .model(&engine.fit(&[anchor, candidates[x]])?, &auxiliary)?
                .weighted_subset(&mask_list[a].labels)?;
            for y in 0..4 {
                q[a][x][y] = model.log_density(&candidates[y], &anchor)?.exp();
            }
            let sum = q[a][x].iter().sum::<f64>();
            for value in &mut q[a][x] {
                *value /= sum;
            }
        }
    }
    let target: [[f64; 4]; 4] = std::array::from_fn(|a| physical.map(|p| rho[a] * p));
    let update = |input: [[f64; 4]; 4], variant: usize| {
        let mut output = [[0.; 4]; 4];
        for a in 0..4 {
            for x in 0..4 {
                let mut out = 0.;
                for y in 0..4 {
                    if x == y {
                        continue;
                    }
                    let reverse = |b: usize| {
                        q[a][x][y] * (physical[y] * q[b][y][x] / (physical[x] * q[a][x][y])).min(1.)
                    };
                    let probability = match variant {
                        0 => reverse(a),
                        // Wrong: all-component reverse for a retained subset.
                        1 => reverse(3),
                        // Wrong: redraw reverse mask, but do not store the new
                        // mask on acceptance. Proposing AND retaining it would
                        // instead be a legitimate joint auxiliary move.
                        2 => (0..4).map(|b| rho[b] * reverse(b)).sum(),
                        _ => unreachable!(),
                    };
                    output[a][y] += input[a][x] * probability;
                    out += probability;
                }
                output[a][x] += input[a][x] * (1. - out);
            }
        }
        output
    };
    let error = |a: [[f64; 4]; 4], b: [[f64; 4]; 4]| {
        a.iter()
            .flatten()
            .zip(b.iter().flatten())
            .map(|(x, y)| (x - y).abs())
            .fold(0., f64::max)
    };
    let correct = update(target, 0);
    assert!(error(correct, target) < 1e-13);
    for a in 0..4 {
        for x in 0..4 {
            for y in 0..4 {
                let forward = target[a][x]
                    * q[a][x][y]
                    * (physical[y] * q[a][y][x] / (physical[x] * q[a][x][y])).min(1.);
                let reverse = target[a][y]
                    * q[a][y][x]
                    * (physical[x] * q[a][x][y] / (physical[y] * q[a][y][x])).min(1.);
                assert!((forward - reverse).abs() < 1e-13);
            }
        }
    }
    // Composition with the exact mask Gibbs kernel retains the same joint law.
    let gibbs: [[f64; 4]; 4] = std::array::from_fn(|a| {
        std::array::from_fn(|x| rho[a] * (0..4).map(|b| correct[b][x]).sum::<f64>())
    });
    assert!(error(gibbs, target) < 1e-13);
    let stale_error = error(update(target, 1), target);
    let fresh_reverse = update(target, 2);
    let fresh_joint_error = error(fresh_reverse, target);
    let twice = update(fresh_reverse, 2);
    let physical_error = (0..4)
        .map(|x| ((0..4).map(|a| twice[a][x]).sum::<f64>() - physical[x]).abs())
        .fold(0., f64::max);
    eprintln!(
        "Mask negative controls: stale joint={stale_error:.6}, fresh reverse retained-old joint={fresh_joint_error:.6}, two-step physical={physical_error:.6}"
    );
    assert!(stale_error > 1e-4 && fresh_joint_error > 1e-4);
    assert!(physical_error > 1e-5);
    Ok(())
}
