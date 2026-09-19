//! Independent geometry and statistical references for the implicit bath gate.
//!
//! Brute-force atom predicates below do not call the BVH. Statistical checks
//! use fixed RNG seeds and conservative six-standard-error tolerances. They
//! check representative FP64 behavior, not formal interval-arithmetic bounds.
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::f64::consts::PI;
use tetramer_mc::{
    depletion::{self, Envelope, GateOptions},
    geometry::{Atom, Environment, Shape, SphereTree},
    math::Pose,
};

fn pose(position: [f64; 3]) -> Pose {
    Pose {
        position,
        orientation: [1., 0., 0., 0.],
    }
}

fn sphere(radius: f64) -> SphereTree {
    SphereTree::new(Shape {
        name: "reference sphere".into(),
        volume: 4. * PI * radius.powi(3) / 3.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius,
        }],
    })
    .unwrap()
}

fn distance_squared(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|k| (a[k] - b[k]).powi(2)).sum()
}

fn periodic_distance_squared(a: [f64; 3], b: [f64; 3], lengths: [f64; 3]) -> f64 {
    (0..3)
        .map(|k| {
            let difference = a[k] - b[k];
            let closest = difference - lengths[k] * (difference / lengths[k] + 0.5).floor();
            closest * closest
        })
        .sum()
}

fn brute_body(tree: &SphereTree, point: [f64; 3], rd: f64) -> bool {
    tree.shape
        .atoms
        .iter()
        .any(|atom| distance_squared(point, atom.center) <= (atom.radius + rd).powi(2))
}

fn brute_spectators(
    tree: &SphereTree,
    poses: &[Pose],
    moving: usize,
    point: [f64; 3],
    lengths: [f64; 3],
    rd: f64,
) -> bool {
    poses
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != moving)
        .any(|(_, placed)| {
            tree.shape.atoms.iter().any(|atom| {
                periodic_distance_squared(point, placed.apply(atom.center), lengths)
                    <= (atom.radius + rd).powi(2)
            })
        })
}

fn brute_hard_valid(
    tree: &SphereTree,
    poses: &[Pose],
    moving: usize,
    tested: Pose,
    lengths: [f64; 3],
) -> bool {
    !poses
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != moving)
        .any(|(_, fixed)| {
            tree.shape.atoms.iter().any(|a| {
                tree.shape.atoms.iter().any(|b| {
                    periodic_distance_squared(
                        tested.apply(a.center),
                        fixed.apply(b.center),
                        lengths,
                    ) < (a.radius + b.radius).powi(2)
                })
            })
        })
}

fn overlap_equal_spheres(radius: f64, separation: f64) -> f64 {
    if separation >= 2. * radius {
        0.
    } else {
        PI * (4. * radius + separation) * (2. * radius - separation).powi(2) / 12.
    }
}

#[derive(Default)]
struct Moments {
    count: usize,
    sum: f64,
    sum_squared: f64,
}
impl Moments {
    fn add(&mut self, x: f64) {
        self.count += 1;
        self.sum += x;
        self.sum_squared += x * x;
    }
    fn mean(&self) -> f64 {
        self.sum / self.count as f64
    }
    fn variance(&self) -> f64 {
        ((self.sum_squared - self.sum * self.sum / self.count as f64) / (self.count - 1) as f64)
            .max(0.)
    }
    fn standard_error(&self) -> f64 {
        (self.variance() / self.count as f64).sqrt()
    }
}

fn poisson_probabilities(mean: f64, terms: usize) -> Vec<f64> {
    let mut probabilities = vec![(-mean).exp()];
    for k in 1..terms {
        probabilities.push(probabilities[k - 1] * mean / k as f64);
    }
    probabilities
}

fn exact_averaged_acceptance(gain_mean: f64, loss_mean: f64, log_b: f64, log_h: f64) -> f64 {
    let gain = poisson_probabilities(gain_mean, 90);
    let loss = poisson_probabilities(loss_mean, 90);
    assert!((gain.iter().sum::<f64>() - 1.).abs() < 2e-14);
    assert!((loss.iter().sum::<f64>() - 1.).abs() < 2e-14);
    gain.iter()
        .enumerate()
        .map(|(g, pg)| {
            loss.iter()
                .enumerate()
                .map(|(l, pl)| pg * pl * (log_h + log_b * (g as f64 - l as f64)).min(0.).exp())
                .sum::<f64>()
        })
        .sum()
}

#[test]
fn sphere_pair_poisson_means_and_nonzero_hastings_flux() {
    let tree = sphere(1.);
    let rd = 0.6;
    let old = pose([16., 10., 10.]);
    let new = pose([12.2, 10., 10.]);
    let poses = [old, pose([10., 10., 10.])];
    let env = Environment::new(&tree, &poses, 0, old, new, [32.; 3], rd).unwrap();
    assert!(env.hard_valid(old) && env.hard_valid(new));
    let options = GateOptions {
        max_cells: 127,
        max_depth: 9,
        min_width: 0.,
    };
    let forward = Envelope::build(&env, old, new, options).unwrap();
    let reverse = Envelope::build(&env, new, old, options).unwrap();
    assert_eq!(forward.volume, reverse.volume);
    let overlap = overlap_equal_spheres(1. + rd, 2.2);
    let (lambda, z) = (0.8_f64, 0.4_f64);
    let log_b = (z / lambda).ln_1p();
    let log_h = 0.37_f64.ln();
    let analytic_f = exact_averaged_acceptance(lambda * overlap, 0., log_b, log_h);
    let analytic_r = exact_averaged_acceptance(0., (lambda + z) * overlap, log_b, -log_h);
    let flux_ratio = (z * overlap + log_h).exp();
    assert!((analytic_f - flux_ratio * analytic_r).abs() < 2e-14);
    // H=1 would conceal a wrong placement of the Hastings factor in the gate.
    for h in [0.09_f64, 0.37, 1.7, 4.2] {
        let a = exact_averaged_acceptance(lambda * overlap, 0., log_b, h.ln());
        let b = exact_averaged_acceptance(0., (lambda + z) * overlap, log_b, -h.ln());
        assert!((a - (z * overlap).exp() * h * b).abs() < 4e-14);
    }
    let mut rng = StdRng::seed_from_u64(202609201);
    let (mut gained, mut lost, mut acceptance_f, mut acceptance_r) = (
        Moments::default(),
        Moments::default(),
        Moments::default(),
        Moments::default(),
    );
    for _ in 0..6000 {
        let f =
            depletion::sample_with_envelope(&mut rng, &env, old, new, lambda, z, &forward).unwrap();
        let r =
            depletion::sample_with_envelope(&mut rng, &env, new, old, lambda, z, &reverse).unwrap();
        assert_eq!(f.lost, 0);
        assert_eq!(r.gained, 0);
        assert_eq!(f.retained_points, f.gained + f.lost);
        assert!((f.log_weight - log_b * f.gained as f64).abs() < 1e-13);
        gained.add(f.gained as f64);
        lost.add(r.lost as f64);
        acceptance_f.add((log_h + f.log_weight).min(0.).exp());
        acceptance_r.add((-log_h + r.log_weight).min(0.).exp());
    }
    assert!(
        (gained.mean() - lambda * overlap).abs()
            < 6. * (lambda * overlap / gained.count as f64).sqrt()
    );
    assert!(
        (lost.mean() - (lambda + z) * overlap).abs()
            < 6. * ((lambda + z) * overlap / lost.count as f64).sqrt()
    );
    for (counts, expected) in [(&gained, lambda * overlap), (&lost, (lambda + z) * overlap)] {
        let variance_se = ((expected + 2. * expected * expected) / counts.count as f64).sqrt();
        assert!((counts.variance() - expected).abs() < 6. * variance_se);
    }
    assert!((acceptance_f.mean() - analytic_f).abs() < 6. * acceptance_f.standard_error() + 1e-10);
    assert!((acceptance_r.mean() - analytic_r).abs() < 6. * acceptance_r.standard_error() + 1e-10);
    let flux_error = (acceptance_f.mean() - flux_ratio * acceptance_r.mean()).abs();
    let flux_se = (acceptance_f.standard_error().powi(2)
        + flux_ratio.powi(2) * acceptance_r.standard_error().powi(2))
    .sqrt();
    assert!(flux_error < 6. * flux_se + 1e-10);
}

#[test]
fn many_body_union_changes_match_independent_volume_sampling() {
    let tree = sphere(0.5);
    let rd = 1.;
    let lengths = [24.; 3];
    let old = pose([8.525, 8.303, 9.9]);
    let new = pose([10.3, 8.303, 8.6]);
    let poses = [
        old,
        pose([8., 8., 8.]),
        pose([9.05, 8., 8.]),
        pose([8.525, 8. + 1.05 * 3_f64.sqrt() / 2., 8.]),
    ];
    let env = Environment::new(&tree, &poses, 0, old, new, lengths, rd).unwrap();
    assert!(brute_hard_valid(&tree, &poses, 0, old, lengths));
    assert!(brute_hard_valid(&tree, &poses, 0, new, lengths));
    assert_eq!(env.labels.len(), 3);
    let radius = 1.5_f64;
    let bounding_volume = (2. * radius).powi(3);
    let samples = 120000;
    let mut rng = StdRng::seed_from_u64(202609202);
    let (mut gains, mut losses, mut covered_new, mut multiply_covered) =
        (0_usize, 0_usize, 0_usize, 0_usize);
    for _ in 0..samples {
        let p = std::array::from_fn(|_| rng.random_range(-radius..radius));
        if !brute_body(&tree, p, rd) {
            continue;
        }
        let a = brute_spectators(&tree, &poses, 0, old.apply(p), lengths, rd);
        let b = brute_spectators(&tree, &poses, 0, new.apply(p), lengths, rd);
        gains += usize::from(b && !a);
        losses += usize::from(a && !b);
        covered_new += usize::from(b);
        let multiplicity = poses[1..]
            .iter()
            .filter(|fixed| {
                periodic_distance_squared(new.apply(p), fixed.position, lengths) <= radius * radius
            })
            .count();
        multiply_covered += usize::from(multiplicity >= 2);
    }
    let volume = |count: usize| bounding_volume * count as f64 / samples as f64;
    let volume_variance = |count: usize| {
        let p = count as f64 / samples as f64;
        bounding_volume.powi(2) * p * (1. - p) / samples as f64
    };
    let pairwise_sum: f64 = poses[1..]
        .iter()
        .map(|fixed| {
            overlap_equal_spheres(
                radius,
                periodic_distance_squared(new.position, fixed.position, lengths).sqrt(),
            )
        })
        .sum();
    assert!(
        volume(multiply_covered) > 0.5,
        "The reference must contain genuine multiple exclusion coverage"
    );
    assert!(
        pairwise_sum - volume(covered_new) > 0.5,
        "A pairwise sum must measurably differ from the many-body union"
    );
    assert!(volume(gains) > 0.2 && volume(losses) > 0.2);
    let envelope = Envelope::build(
        &env,
        old,
        new,
        GateOptions {
            max_cells: 255,
            max_depth: 10,
            min_width: 0.,
        },
    )
    .unwrap();
    let (lambda, z) = (0.7_f64, 0.3_f64);
    let (mut gain_counts, mut loss_counts, mut products) =
        (Moments::default(), Moments::default(), Moments::default());
    for _ in 0..5000 {
        let result =
            depletion::sample_with_envelope(&mut rng, &env, old, new, lambda, z, &envelope)
                .unwrap();
        gain_counts.add(result.gained as f64);
        loss_counts.add(result.lost as f64);
        products.add(result.gained as f64 * result.lost as f64);
    }
    for (counts, estimate, variance, intensity) in [
        (&gain_counts, volume(gains), volume_variance(gains), lambda),
        (
            &loss_counts,
            volume(losses),
            volume_variance(losses),
            lambda + z,
        ),
    ] {
        let error = (counts.mean() - intensity * estimate).abs();
        let uncertainty = (counts.standard_error().powi(2) + intensity.powi(2) * variance).sqrt();
        assert!(
            error < 6. * uncertainty,
            "sample mean {} vs reference {}, uncertainty {}",
            counts.mean(),
            intensity * estimate,
            uncertainty
        );
    }
    // Independent thinning on disjoint gain/loss regions implies zero covariance.
    let covariance = products.mean() - gain_counts.mean() * loss_counts.mean();
    let covariance_se =
        (gain_counts.variance() * loss_counts.variance() / gain_counts.count as f64).sqrt();
    assert!(covariance.abs() < 6. * covariance_se);
}

#[test]
fn periodic_face_crossing_envelopes_are_symmetric_and_cover_brute_xor() {
    let tree = SphereTree::new(Shape {
        name: "asymmetric sphere union".into(),
        volume: 0.,
        atoms: vec![
            Atom {
                center: [-0.8, 0., 0.],
                radius: 0.65,
            },
            Atom {
                center: [0.5, 0.3, 0.],
                radius: 0.8,
            },
            Atom {
                center: [0., -0.4, 0.65],
                radius: 0.5,
            },
        ],
    })
    .unwrap();
    let lengths = [16.; 3];
    let rd = 0.35;
    let old = Pose {
        position: [0.25, 5., 5.],
        orientation: [(0.3_f64).cos(), 0., 0., (0.3_f64).sin()],
    };
    let new = Pose {
        position: [15.6, 5.4, 5.1],
        orientation: [(0.4_f64).cos(), (0.4_f64).sin(), 0., 0.],
    };
    let poses = [
        old,
        pose([14.1, 5.2, 5.4]),
        pose([2.3, 4.5, 5.]),
        pose([8., 8., 8.]),
    ];
    let mut reverse_poses = poses;
    reverse_poses[0] = new;
    let forward = Environment::new(&tree, &poses, 0, old, new, lengths, rd).unwrap();
    let reverse = Environment::new(&tree, &reverse_poses, 0, new, old, lengths, rd).unwrap();
    assert_eq!(forward.labels, reverse.labels);
    assert!(forward.labels.iter().any(|(_, image)| image[0] < 0));
    assert!(forward.labels.iter().any(|(_, image)| image[0] > 0));
    for endpoint in [old, new] {
        assert_eq!(
            forward.hard_valid(endpoint),
            brute_hard_valid(&tree, &poses, 0, endpoint, lengths)
        );
    }
    let options = [
        GateOptions {
            max_cells: 1,
            max_depth: 0,
            min_width: 0.,
        },
        GateOptions {
            max_cells: 15,
            max_depth: 4,
            min_width: 0.,
        },
        GateOptions {
            max_cells: 255,
            max_depth: 10,
            min_width: 0.,
        },
    ];
    let mut envelopes = Vec::new();
    for settings in options {
        let a = Envelope::build(&forward, old, new, settings).unwrap();
        let b = Envelope::build(&reverse, new, old, settings).unwrap();
        assert_eq!(a.volume, b.volume);
        assert_eq!(a.cells.len(), b.cells.len());
        for (left, right) in a.cells.iter().zip(&b.cells) {
            assert_eq!(left.lo, right.lo);
            assert_eq!(left.hi, right.hi);
        }
        assert!(a.created <= settings.max_cells);
        envelopes.push(a);
    }
    let mut rng = StdRng::seed_from_u64(202609203);
    let mut xor_points = 0;
    for _ in 0..12000 {
        let p = std::array::from_fn(|_| rng.random_range(-2.5..2.5));
        let in_body = brute_body(&tree, p, rd);
        assert_eq!(tree.contains(p, rd), in_body);
        let a = brute_spectators(&tree, &poses, 0, old.apply(p), lengths, rd);
        let b = brute_spectators(&tree, &poses, 0, new.apply(p), lengths, rd);
        // The neighborhood only promises complete coverage within the moving
        // exclusion domain. Points outside it may see omitted distant bodies.
        if !in_body {
            continue;
        }
        assert_eq!(forward.contains(old.apply(p)), a);
        assert_eq!(forward.contains(new.apply(p)), b);
        if a == b {
            continue;
        }
        xor_points += 1;
        for envelope in &envelopes {
            let covering = envelope
                .cells
                .iter()
                .filter(|cell| (0..3).all(|k| p[k] >= cell.lo[k] && p[k] <= cell.hi[k]))
                .count();
            assert_eq!(
                covering, 1,
                "Every independently detected XOR point must occupy exactly one retained cell"
            );
        }
    }
    assert!(xor_points > 100);
}

#[test]
fn zero_activity_is_a_deterministic_unit_physical_factor() {
    let tree = sphere(1.);
    let old = pose([5., 5., 5.]);
    let new = pose([5.3, 5., 5.]);
    let poses = [old, pose([7.5, 5., 5.])];
    let env = Environment::new(&tree, &poses, 0, old, new, [24.; 3], 0.6).unwrap();
    let mut rng = StdRng::seed_from_u64(202609204);
    let mut untouched = StdRng::seed_from_u64(202609204);
    let result =
        depletion::sample(&mut rng, &env, old, new, 1., 0., GateOptions::default()).unwrap();
    assert_eq!(result.gained, 0);
    assert_eq!(result.lost, 0);
    assert_eq!(result.raw_points, 0);
    assert_eq!(result.log_weight, 0.);
    assert_eq!(rng.random::<u64>(), untouched.random::<u64>());
    // Zero physical log weight leaves any proposal Hastings correction intact.
    let log_h = 0.23_f64.ln();
    assert!(((log_h + result.log_weight).min(0.).exp() - 0.23).abs() < 1e-15);
}

#[test]
fn invalid_inputs_do_not_escape_through_empty_or_zero_activity_paths() {
    let tree = sphere(1.);
    let old = pose([5., 5., 5.]);
    let new = pose([5.3, 5., 5.]);
    let poses = [old, pose([7.5, 5., 5.])];
    let mut invalid = new;
    invalid.position[0] = f64::NAN;
    assert!(Environment::new(&tree, &poses, 0, old, invalid, [24.; 3], 0.6).is_err());
    invalid = new;
    invalid.orientation = [2., 0., 0., 0.];
    assert!(Environment::new(&tree, &poses, 0, old, invalid, [24.; 3], 0.6).is_err());
    let noncanonical = [old, pose([24., 5., 5.])];
    assert!(Environment::new(&tree, &noncanonical, 0, old, new, [24.; 3], 0.6).is_err());
    let env = Environment::new(&tree, &poses, 0, old, new, [24.; 3], 0.6).unwrap();
    let mut rng = StdRng::seed_from_u64(202609205);
    assert!(
        depletion::sample(
            &mut rng,
            &env,
            old,
            new,
            f64::NAN,
            0.,
            GateOptions::default()
        )
        .is_err()
    );
    assert!(
        depletion::sample(
            &mut rng,
            &env,
            old,
            new,
            1.,
            0.,
            GateOptions {
                max_cells: 0,
                ..GateOptions::default()
            }
        )
        .is_err()
    );
    assert!(depletion::sample(&mut rng, &env, old, new, 1., -0.1, GateOptions::default()).is_err());
    let empty_env = Environment::new(&tree, &[old], 0, old, new, [24.; 3], 0.6).unwrap();
    assert!(Envelope::build(&empty_env, old, invalid, GateOptions::default()).is_err());
    // The accepted norm tolerance must not turn an otherwise valid quaternion
    // into a scale/shear that invalidates spherical pruning certificates.
    let angle = 0.7_f64;
    let nearly_unit = [angle.cos() * (1. + 2e-9), angle.sin() * (1. + 2e-9), 0., 0.];
    let corrected = Pose {
        position: [5.; 3],
        orientation: nearly_unit,
    };
    assert!(corrected.validate().is_ok());
    let origin = corrected.apply([0.; 3]);
    let unit = corrected.apply([0., 1., 0.]);
    assert!((distance_squared(origin, unit) - 1.).abs() < 4e-15);
}
