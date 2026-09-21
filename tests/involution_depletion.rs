//! Independent exact-start validation of the deterministic basin involution
//! combined with the conditional many-body Poisson gate. The physical reference
//! is computed separately from the move, never from Gaussian mixture weights.
use anyhow::Result;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Poisson, StandardNormal};
use std::f64::consts::PI;
use tetramer_mc::{
    basin_involution::{BasinPair, FixedBasinInvolution},
    depletion::{self, GateOptions},
    docking::{DockingMethod, DockingProposal},
    geometry::{Atom, Environment, Placed, Shape, SphereTree},
    math::*,
    proposal::{FrozenRelativePoseProposal, GaussianComponentParameters},
};

const CORE: f64 = 1.;
const RD: f64 = 0.7;
const CAPTURE: f64 = 4.;
const Z: f64 = 0.4;
const N: usize = 6000;

fn parameters() -> Vec<GaussianComponentParameters> {
    (0..2)
        .map(|j| {
            let scales = if j == 0 {
                [0.6, 1.2, 0.9, 0.6, 0.8, 0.7]
            } else {
                [1.5, 0.8, 1.3, 1.1, 0.9, 1.2]
            };
            let mut lower = [[0.; 6]; 6];
            for i in 0..6 {
                lower[i][i] = scales[i];
            }
            lower[1][0] = 0.2;
            lower[3][0] = 0.25;
            lower[5][2] = -0.2;
            GaussianComponentParameters {
                anchor_position: if j == 0 { [2.4, 0., 0.] } else { [0., 2.5, 0.] },
                anchor_rotation: if j == 0 {
                    cayley([0.2, -0.3, 0.1])
                } else {
                    cayley([-0.6, 0.4, 0.3])
                },
                mean: [0.; 6],
                covariance: std::array::from_fn(|a| {
                    std::array::from_fn(|b| (0..6).map(|k| lower[a][k] * lower[b][k]).sum())
                }),
                weight: 0.5,
            }
        })
        .collect()
}

fn map(correlation: f64) -> Result<FixedBasinInvolution> {
    // Equivalent to independently uniform source and target chart labels;
    // stored as unordered pairs so direction symmetry is structural.
    FixedBasinInvolution::new(
        parameters(),
        1.,
        correlation,
        vec![
            BasinPair {
                first: 0,
                second: 0,
                weight: 1.,
            },
            BasinPair {
                first: 0,
                second: 1,
                weight: 2.,
            },
            BasinPair {
                first: 1,
                second: 1,
                weight: 1.,
            },
        ],
    )
}

fn posterior_update(
    proposal: &DockingProposal,
    env: &Environment,
    pose: &mut Pose,
    z: f64,
    rng: &mut StdRng,
    work: &mut Work,
) -> Result<()> {
    let fixed: Vec<_> = env
        .fixed
        .iter()
        .map(|p| Pose {
            position: p.position,
            orientation: quaternion(p.rotation),
        })
        .collect();
    let (candidate, info) = proposal.propose(rng, *pose, &fixed)?;
    let Some(candidate) = candidate else {
        return Ok(());
    };
    if norm(candidate.position) > CAPTURE || !env.hard_valid(candidate) {
        return Ok(());
    }
    work.valid += 1;
    let gate = depletion::sample(
        rng,
        env,
        *pose,
        candidate,
        4. * z,
        z,
        GateOptions {
            max_cells: 1,
            max_depth: 0,
            min_width: 0.,
        },
    )?;
    work.points += gate.raw_points;
    work.gained += gate.gained;
    work.lost += gate.lost;
    let correction = info["log_reverse_forward"].as_f64().unwrap();
    if rng.random::<f64>().max(f64::MIN_POSITIVE).ln() < (correction + gate.log_weight).min(0.) {
        work.accepted += 1;
        work.nonself += usize::from(candidate != *pose);
        *pose = candidate;
    }
    Ok(())
}

#[test]
fn posterior_transport_with_poisson_gate_preserves_independent_physical_references() -> Result<()> {
    check_posterior_physical_reference(false)
}

#[test]
fn reciprocal_posterior_with_poisson_gate_preserves_independent_physical_references() -> Result<()>
{
    check_posterior_physical_reference(true)
}

fn check_posterior_physical_reference(reciprocal: bool) -> Result<()> {
    let shape = tree()?;
    let sha = "0000000000000000000000000000000000000000000000000000000000000000";
    let mut model = FrozenRelativePoseProposal::from_components_open(
        parameters(),
        1.,
        [2. * CAPTURE; 3],
        0.2,
        sha,
        sha,
    )?;
    if reciprocal {
        let p = parameters();
        let envelope = serde_json::json!({
            "schema":"reciprocal-pose-mixture-v1", "reciprocal_components":[true,true],
            "base_model": {
                "coordinate_convention":"anchor-body-relative", "shape_sha256":sha,
                "angular_length":1., "weights":[0.23,0.77],
                "anchors":p.iter().map(|p| serde_json::json!({"position":p.anchor_position,"rotation":p.anchor_rotation})).collect::<Vec<_>>(),
                "means":p.iter().map(|p| p.mean).collect::<Vec<_>>(),
                "covariances":p.iter().map(|p| p.covariance).collect::<Vec<_>>()
            }
        });
        model = FrozenRelativePoseProposal::from_json_str_open(
            &envelope.to_string(),
            [2. * CAPTURE; 3],
            0.2,
            sha,
        )?;
    }
    for (case, centers, z) in [
        ("AO", vec![[0.; 3]], Z),
        ("many-body", vec![[-1.5, 0., 0.], [1.5, 0., 0.]], 0.12),
    ] {
        let env = Environment {
            tree: &shape,
            fixed: centers.iter().copied().map(fixed).collect(),
            labels: (0..centers.len()).map(|i| (i, [0; 3])).collect(),
            rd: RD,
        };
        for (index, c) in [0., 0.9].into_iter().enumerate() {
            let proposal = DockingProposal::new(
                model.clone(),
                DockingMethod::PosteriorInvolution,
                c,
                [0.; 3],
            )?;
            let mut starts = StdRng::seed_from_u64(106502);
            let mut change = [Moments::default(); 8];
            let mut after = [Moments::default(); 8];
            let mut work = Work::default();
            let values = |pose: Pose| {
                let mut a = observables(pose);
                if centers.len() > 1 {
                    let distances: Vec<_> = centers
                        .iter()
                        .map(|&p| norm(sub(pose.position, p)))
                        .collect();
                    a[2] = f64::from(distances.iter().all(|&d| d < 2. * (CORE + RD)));
                    a[6] = distances.iter().copied().fold(f64::INFINITY, f64::min);
                }
                a
            };
            for replicate in 0..N {
                let mut pose = if centers.len() == 1 {
                    exact_ao_start(&mut starts)
                } else {
                    exact_union_start(&mut starts, &centers, z)
                };
                let before = values(pose);
                let mut rng =
                    StdRng::seed_from_u64(934503 + index as u64 * 100000 + replicate as u64);
                for _ in 0..3 {
                    posterior_update(&proposal, &env, &mut pose, z, &mut rng, &mut work)?;
                }
                for (j, (a, b)) in values(pose).into_iter().zip(before).enumerate() {
                    change[j].add(a - b);
                    after[j].add(a);
                }
            }
            let maximum = change.iter().map(|m| m.z(0.).abs()).fold(0., f64::max);
            assert!(maximum < 5.5, "posterior {case} c={c}: paired z={maximum}");
            if centers.len() == 1 {
                for (m, e) in after.into_iter().zip(reference()) {
                    assert!(m.z(e).abs() < 5.5);
                }
            }
            assert!(
                work.accepted > 300 && work.nonself > 200 && work.gained > 100 && work.lost > 100
            );
            eprintln!(
                "Posterior reciprocal={reciprocal} {case} c={c}: max paired z={maximum:.3}; {work:?}"
            );
        }
    }
    Ok(())
}

fn normal<const D: usize>(rng: &mut StdRng) -> [f64; D] {
    std::array::from_fn(|_| StandardNormal.sample(rng))
}

fn direction(rng: &mut StdRng) -> Vec3 {
    let x = normal(rng);
    scale(x, 1. / norm(x))
}

fn uniform_ball(rng: &mut StdRng, radius: f64) -> Vec3 {
    scale(direction(rng), radius * rng.random::<f64>().cbrt())
}

fn orientation(rng: &mut StdRng) -> [f64; 4] {
    let q: [f64; 4] = normal(rng);
    let length = q.iter().map(|x| x * x).sum::<f64>().sqrt();
    q.map(|x| x / length)
}

fn lens(radius: f64, distance: f64) -> f64 {
    if distance >= 2. * radius {
        0.
    } else {
        PI * (4. * radius + distance) * (2. * radius - distance).powi(2) / 12.
    }
}

fn exact_ao_start(rng: &mut StdRng) -> Pose {
    let maximum = lens(CORE + RD, 2. * CORE);
    loop {
        // Uniform volume on the whole hard-valid radial shell.
        let radius = ((2. * CORE).powi(3)
            + rng.random::<f64>() * (CAPTURE.powi(3) - (2. * CORE).powi(3)))
        .cbrt();
        if rng.random::<f64>().max(f64::MIN_POSITIVE).ln() < Z * (lens(CORE + RD, radius) - maximum)
        {
            return Pose {
                position: scale(direction(rng), radius),
                orientation: orientation(rng),
            };
        }
    }
}

fn integrate(f: impl Fn(f64) -> f64, lo: f64, hi: f64) -> f64 {
    let count = 8192;
    let width = (hi - lo) / count as f64;
    let mut sum = f(lo) + f(hi);
    for i in 1..count {
        sum += if i % 2 == 0 { 2. } else { 4. } * f(lo + width * i as f64);
    }
    sum * width / 3.
}

fn reference() -> [f64; 8] {
    let weight = |r: f64| r * r * (Z * lens(CORE + RD, r)).exp();
    let normalizer = integrate(weight, 2. * CORE, CAPTURE);
    [
        integrate(|r| r * weight(r), 2. * CORE, CAPTURE) / normalizer,
        integrate(|r| r * r * weight(r), 2. * CORE, CAPTURE) / normalizer,
        integrate(weight, 2. * CORE, 2. * (CORE + RD)) / normalizer,
        0.,
        0.,
        1. / 3.,
        0.,
        0.,
    ]
}

fn observables(pose: Pose) -> [f64; 8] {
    let radius = norm(pose.position);
    let r00 = rotation(pose.orientation)[0][0];
    let contact = f64::from(radius < 2. * (CORE + RD));
    [
        radius,
        radius * radius,
        contact,
        pose.position[0] / radius,
        r00,
        r00 * r00,
        contact * r00,
        pose.position[0] * r00,
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
    fn z(self, reference: f64) -> f64 {
        let se = ((self.sum2 - self.sum * self.sum / self.n as f64).max(0.)
            / (self.n - 1) as f64
            / self.n as f64)
            .sqrt();
        (self.mean() - reference) / se.max(1e-15)
    }
}

fn tree() -> Result<SphereTree> {
    SphereTree::new(Shape {
        name: "AO involution control sphere".into(),
        volume: 4. * PI / 3.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius: CORE,
        }],
    })
}

fn fixed(position: Vec3) -> Placed {
    Placed::new(Pose {
        position,
        orientation: [1., 0., 0., 0.],
    })
}

#[derive(Default, Debug)]
struct Work {
    valid: usize,
    accepted: usize,
    nonself: usize,
    points: u64,
    gained: u64,
    lost: u64,
}

fn update(
    map: &FixedBasinInvolution,
    env: &Environment,
    pose: &mut Pose,
    z: f64,
    keep_correction: bool,
    rng: &mut StdRng,
    work: &mut Work,
) -> Result<()> {
    let trace = map.draw_trace(rng);
    let step = map.apply(*pose, &trace)?;
    if norm(step.pose.position) >= CAPTURE || !env.hard_valid(step.pose) {
        return Ok(());
    }
    work.valid += 1;
    let gate = depletion::sample(
        rng,
        env,
        *pose,
        step.pose,
        4. * z,
        z,
        GateOptions {
            max_cells: 1,
            max_depth: 0,
            min_width: 0.,
        },
    )?;
    work.points += gate.raw_points;
    work.gained += gate.gained;
    work.lost += gate.lost;
    let correction = if keep_correction {
        step.log_correction
    } else {
        0.
    };
    if rng.random::<f64>().max(f64::MIN_POSITIVE).ln() < (gate.log_weight + correction).min(0.) {
        *pose = step.pose;
        work.accepted += 1;
        work.nonself += usize::from(trace.source != trace.target);
    }
    Ok(())
}

#[test]
fn involution_and_poisson_gate_preserve_exact_conditional_ao_equilibrium() -> Result<()> {
    let shape = tree()?;
    let env = Environment {
        tree: &shape,
        fixed: vec![fixed([0.; 3])],
        labels: vec![(0, [0; 3])],
        rd: RD,
    };
    let expected = reference();
    for (index, c) in [0., 0.5, 0.9, 1.].into_iter().enumerate() {
        let map = map(c)?;
        let mut initial = [Moments::default(); 8];
        let mut after = [Moments::default(); 8];
        let mut change = [Moments::default(); 8];
        let mut work = Work::default();
        let mut starts = StdRng::seed_from_u64(615763);
        for replicate in 0..N {
            let mut pose = exact_ao_start(&mut starts);
            let before = observables(pose);
            let mut rng = StdRng::seed_from_u64(857104 + 100000 * index as u64 + replicate as u64);
            for _ in 0..3 {
                update(&map, &env, &mut pose, Z, true, &mut rng, &mut work)?;
            }
            let values = observables(pose);
            for j in 0..8 {
                initial[j].add(before[j]);
                after[j].add(values[j]);
                change[j].add(values[j] - before[j]);
            }
        }
        let mut maximum = 0_f64;
        for j in 0..8 {
            maximum = maximum.max(change[j].z(0.).abs());
            assert!(
                initial[j].z(expected[j]).abs() < 5.5,
                "initial AO reference {j}"
            );
            assert!(
                after[j].z(expected[j]).abs() < 5.5,
                "c={c}, AO marginal {j}: {}",
                after[j].z(expected[j])
            );
            assert!(
                change[j].z(0.).abs() < 5.5,
                "c={c}, paired AO {j}: {}",
                change[j].z(0.)
            );
        }
        eprintln!("AO c={c}: maximum paired z={maximum:.3}, {work:?}");
        assert!(work.accepted > 350 && work.nonself > 150);
        assert!(work.points > 10000 && work.gained > 100 && work.lost > 100);
    }
    Ok(())
}

#[test]
fn poisson_gate_does_not_correct_an_omitted_basin_map_factor() -> Result<()> {
    let shape = tree()?;
    let env = Environment {
        tree: &shape,
        fixed: vec![fixed([0.; 3])],
        labels: vec![(0, [0; 3])],
        rd: RD,
    };
    let map = map(0.)?;
    let expected = reference();
    let mut starts = StdRng::seed_from_u64(936052);
    let mut bad = [Moments::default(); 8];
    let mut work = Work::default();
    for replicate in 0..N {
        let mut pose = exact_ao_start(&mut starts);
        let mut rng = StdRng::seed_from_u64(530111 + replicate as u64);
        for _ in 0..3 {
            update(&map, &env, &mut pose, Z, false, &mut rng, &mut work)?;
        }
        for (stat, value) in bad.iter_mut().zip(observables(pose)) {
            stat.add(value);
        }
    }
    let power = bad
        .iter()
        .zip(expected)
        .map(|(m, e)| m.z(e).abs())
        .fold(0., f64::max);
    eprintln!(
        "Omitted map correction with exact Poisson gate: max drift={power:.2} SE, contact expected={:.5} got={:.5}; {work:?}",
        expected[2],
        bad[2].mean()
    );
    assert!(power > 10.);
    Ok(())
}

#[test]
fn zero_correlation_redraws_destination_gaussian_latents_before_any_gate() -> Result<()> {
    let map = map(0.)?;
    let mut rng = StdRng::seed_from_u64(777521);
    let mut moments = [[Moments::default(); 2]; 6];
    for k in 0..12000 {
        let old = Pose {
            position: [2.1 + (k % 7) as f64 / 10., 0.3, -0.2],
            orientation: quaternion(cayley([0.4, -0.7, 0.3])),
        };
        let trace = map.draw_trace(&mut rng);
        let step = map.apply(old, &trace)?;
        for d in 0..6 {
            assert!((step.target_latent[d] - trace.noise[d]).abs() < 1e-14);
            assert!((step.inverse_trace.noise[d] - step.source_latent[d]).abs() < 1e-14);
            moments[d][0].add(step.target_latent[d]);
            moments[d][1].add(step.target_latent[d].powi(2));
        }
    }
    for m in moments {
        assert!(m[0].z(0.).abs() < 5.5 && m[1].z(1.).abs() < 5.5);
    }
    Ok(())
}

/// Independent exact rejection sampler for one mobile sphere and any fixed
/// sphere union. A cloud of intensity z in the mobile exclusion sphere is
/// accepted iff every point is inside the fixed union. Its success probability
/// is exp[-z(V_mobile-C_union)], proportional to the required exp(z*C_union).
/// This deliberately avoids the move's gained/lost estimator and unknown union
/// volumes. A modest activity keeps rejection setup inexpensive.
fn exact_union_start(rng: &mut StdRng, centers: &[Vec3], z: f64) -> Pose {
    let volume = 4. * PI * (CORE + RD).powi(3) / 3.;
    let poisson = Poisson::<f64>::new(z * volume).unwrap();
    loop {
        let position = uniform_ball(rng, CAPTURE);
        if centers.iter().any(|&c| norm(sub(position, c)) < 2. * CORE) {
            continue;
        }
        let count = poisson.sample(rng) as usize;
        let accepted = (0..count).all(|_| {
            let point = add(position, uniform_ball(rng, CORE + RD));
            centers.iter().any(|&c| norm(sub(point, c)) < CORE + RD)
        });
        if accepted {
            return Pose {
                position,
                orientation: orientation(rng),
            };
        }
    }
}

#[test]
fn overlapping_neighbor_exclusions_preserve_exact_many_body_conditional_law() -> Result<()> {
    let shape = tree()?;
    let centers = [[-1.5, 0., 0.], [1.5, 0., 0.]];
    let env = Environment {
        tree: &shape,
        fixed: centers.map(fixed).to_vec(),
        labels: vec![(0, [0; 3]), (1, [0; 3])],
        rd: RD,
    };
    // At this hard-valid mobile center, a ball of radius0.2 around the origin
    // belongs to all THREE exclusion spheres. Pairwise lens sums overcount it.
    let witness = Pose {
        position: [0., 1.5, 0.],
        orientation: [1., 0., 0., 0.],
    };
    assert!(env.hard_valid(witness));
    assert!(centers.iter().all(|&c| norm(c) + 0.19 < CORE + RD));
    assert!(norm(witness.position) + 0.19 < CORE + RD);
    let z = 0.12;
    let map = map(0.5)?;
    let mut starts = StdRng::seed_from_u64(304629);
    let mut work = Work::default();
    let mut change = [Moments::default(); 8];
    let values = |pose: Pose| {
        let mut a = observables(pose);
        let distances = centers.map(|c| norm(sub(pose.position, c)));
        a[2] = f64::from(distances.iter().all(|&d| d < 2. * (CORE + RD)));
        a[6] = distances[0].min(distances[1]);
        a
    };
    for replicate in 0..5000 {
        let mut pose = exact_union_start(&mut starts, &centers, z);
        let before = values(pose);
        let mut rng = StdRng::seed_from_u64(948055 + replicate as u64);
        for _ in 0..3 {
            update(&map, &env, &mut pose, z, true, &mut rng, &mut work)?;
        }
        for (m, (a, b)) in change.iter_mut().zip(values(pose).into_iter().zip(before)) {
            m.add(a - b);
        }
    }
    let maximum = change.iter().map(|m| m.z(0.).abs()).fold(0., f64::max);
    eprintln!("Two-neighbor union target: max paired z={maximum:.3}; {work:?}");
    assert!(maximum < 5.5);
    assert!(work.accepted > 300 && work.nonself > 100 && work.points > 10000);
    assert!(work.gained > 100 && work.lost > 100);
    Ok(())
}
