//! Physical reference tests for rigid-subset Boolean-union depletion. These
//! exercise its geometry and accepted auxiliary flow, not just implementation
//! self-consistency. All statistical fixtures have fixed seeds and analytic or
//! independently integrated overlap volumes.
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::f64::consts::PI;
use tetramer_mc::{
    depletion::GateOptions,
    geometry::{Atom, Shape, SphereTree},
    math::{Pose, cayley, matmul, norm, quaternion, rotation, sub, transpose},
    rigid_subset::{RigidSubset, carry},
    spherical::Container,
};

fn pose(position: [f64; 3]) -> Pose {
    Pose {
        position,
        orientation: [1., 0., 0., 0.],
    }
}
fn sphere(radius: f64) -> SphereTree {
    SphereTree::new(Shape {
        name: "rigid-subset sphere".into(),
        volume: 4. * PI * radius.powi(3) / 3.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius,
        }],
    })
    .unwrap()
}
fn coarse() -> GateOptions {
    GateOptions {
        max_cells: 1,
        max_depth: 0,
        min_width: 0.,
    }
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 2e-12, "{a} != {b}");
}
fn same_pose(a: Pose, b: Pose) {
    for k in 0..3 {
        close(a.position[k], b.position[k]);
    }
    let ra = rotation(a.orientation);
    let rb = rotation(b.orientation);
    for i in 0..3 {
        for j in 0..3 {
            close(ra[i][j], rb[i][j]);
        }
    }
}
fn assert_se(mean: f64, truth: f64, variance: f64, n: usize) {
    let tolerance = 6. * (variance / n as f64).sqrt() + 3e-5;
    assert!(
        (mean - truth).abs() <= tolerance,
        "mean {mean}, truth {truth}, tolerance {tolerance}"
    );
}

#[test]
fn carry_is_invertible_and_preserves_internal_relative_poses() {
    let state = vec![
        Pose {
            position: [1., 2., 3.],
            orientation: quaternion(cayley([0.1, 0.2, -0.3])),
        },
        Pose {
            position: [-2., 4., 1.],
            orientation: quaternion(cayley([0.5, -0.1, 0.2])),
        },
        pose([14., -11., 3.]),
        Pose {
            position: [4., 5., 3.],
            orientation: quaternion(cayley([-0.2, 0.8, 0.3])),
        },
    ];
    let selected = [3, 0, 1];
    let new_handle = Pose {
        position: [8., -2., -4.],
        orientation: quaternion(cayley([-0.2, 0.5, 0.7])),
    };
    let next = carry(&state, &selected, 0, new_handle).unwrap();
    assert_eq!(next[2], state[2]);
    assert_eq!(next[0], new_handle);
    let reverse = carry(&next, &selected, 0, state[0]).unwrap();
    for i in 0..state.len() {
        same_pose(state[i], reverse[i]);
    }
    for &i in &selected {
        for &j in &selected {
            close(
                norm(sub(state[i].position, state[j].position)),
                norm(sub(next[i].position, next[j].position)),
            );
            let a = matmul(
                transpose(rotation(state[i].orientation)),
                rotation(state[j].orientation),
            );
            let b = matmul(
                transpose(rotation(next[i].orientation)),
                rotation(next[j].orientation),
            );
            for k in 0..3 {
                for l in 0..3 {
                    close(a[k][l], b[k][l]);
                }
            }
        }
    }
    assert_eq!(carry(&state, &selected, 0, state[0]).unwrap(), state);
}

#[test]
fn membership_validation_and_hard_wall_checks() {
    let tree = sphere(0.5);
    let state = [pose([-1., 0., 0.]), pose([1., 0., 0.]), pose([6., 0., 0.])];
    assert!(RigidSubset::new(&tree, &state, &[], 0, state[0], 0.5).is_err());
    assert!(RigidSubset::new(&tree, &state, &[0, 0], 0, state[0], 0.5).is_err());
    assert!(RigidSubset::new(&tree, &state, &[0, 3], 0, state[0], 0.5).is_err());
    assert!(RigidSubset::new(&tree, &state, &[0, 1], 2, state[0], 0.5).is_err());
    let wall = Container::new(8., &tree).unwrap();
    let good = RigidSubset::new(&tree, &state, &[1, 0], 0, pose([1., 2., 0.]), 0.5).unwrap();
    assert_eq!(good.members(), &[1, 0]);
    assert_eq!(good.proposed_poses()[1], pose([1., 2., 0.]));
    assert!(good.hard_valid(Some(&wall), [0.; 3]));
    let clash = RigidSubset::new(&tree, &state, &[0, 1], 0, pose([4., 0., 0.]), 0.5).unwrap();
    assert!(!clash.hard_valid(Some(&wall), [0.; 3]));
    let outside = RigidSubset::new(&tree, &state, &[0, 1], 0, pose([6., 3., 0.]), 0.5).unwrap();
    assert!(!outside.hard_valid(Some(&wall), [0.; 3]));
}

#[test]
fn isolated_union_and_zero_activity_have_zero_gate_without_rng_consumption() {
    let tree = sphere(0.1);
    // The moving exclusion balls overlap strongly, although their hard cores do not.
    let state = [pose([-0.3, 0., 0.]), pose([0.3, 0., 0.])];
    let trial = RigidSubset::new(&tree, &state, &[0, 1], 0, pose([8., 1., 2.]), 0.9).unwrap();
    assert_eq!(trial.spectator_count(), 0);
    let mut rng = StdRng::seed_from_u64(4751);
    let mut reference = StdRng::seed_from_u64(4751);
    let gate = trial
        .sample(&mut rng, 0.5, 0.3, GateOptions::default())
        .unwrap();
    assert_eq!(gate.raw_points, 0);
    assert_eq!(gate.log_weight, 0.);
    assert_eq!(rng.random::<u64>(), reference.random::<u64>());
    let near = [state[0], state[1], pose([0., 0.5, 0.])];
    let trial = RigidSubset::new(&tree, &near, &[0, 1], 0, pose([8., 1., 2.]), 0.9).unwrap();
    rng = StdRng::seed_from_u64(4752);
    let mut reference = StdRng::seed_from_u64(4752);
    let gate = trial
        .sample(&mut rng, 0.5, 0., GateOptions::default())
        .unwrap();
    assert_eq!(gate.raw_points, 0);
    assert_eq!(gate.log_weight, 0.);
    assert_eq!(rng.random::<u64>(), reference.random::<u64>());
    let identity = RigidSubset::new(&tree, &near, &[0, 1], 0, near[0], 0.9).unwrap();
    assert_eq!(
        identity
            .sample(&mut rng, 0.5, 0.3, coarse())
            .unwrap()
            .raw_points,
        0
    );
}

#[test]
fn disjoint_envelope_never_prunes_changed_union_points() {
    let tree = SphereTree::new(Shape {
        name: "asymmetric".into(),
        volume: 0.,
        atoms: vec![
            Atom {
                center: [-0.5, 0., 0.],
                radius: 0.5,
            },
            Atom {
                center: [0.6, 0.1, 0.],
                radius: 0.35,
            },
        ],
    })
    .unwrap();
    let state = [
        pose([0., 0., 0.]),
        pose([0., 1.5, 0.]),
        pose([1.6, 0.5, 0.]),
    ];
    let next = Pose {
        position: [0.5, 0.2, 0.1],
        orientation: quaternion(cayley([0.15, -0.2, 0.4])),
    };
    let trial = RigidSubset::new(&tree, &state, &[0, 1], 0, next, 0.4).unwrap();
    let root = trial.root_bounds();
    for budget in [1, 31, 255] {
        let envelope = trial
            .envelope(GateOptions {
                max_cells: budget,
                max_depth: 12,
                min_width: 0.,
            })
            .unwrap();
        let mut rng = StdRng::seed_from_u64(5161);
        let mut changed = 0;
        for _ in 0..20_000 {
            let p = std::array::from_fn(|k| {
                root.lo[k] + rng.random::<f64>() * (root.hi[k] - root.lo[k])
            });
            let (old, new) = trial.overlap_indicators(p);
            if old != new {
                changed += 1;
                let cover = envelope
                    .cells
                    .iter()
                    .filter(|c| (0..3).all(|k| p[k] >= c.lo[k] && p[k] < c.hi[k]))
                    .count();
                assert_eq!(
                    cover, 1,
                    "changed point absent from disjoint envelope, budget={budget}"
                );
            }
        }
        assert!(changed > 50);
    }
}

#[test]
fn analytic_pair_depletion_acceptance_and_poisson_counts() {
    let tree = sphere(0.2);
    let rd = 0.8;
    let lambda = 0.7;
    let z = 0.3;
    let state = [pose([0., 0., 0.]), pose([1., 0., 0.])];
    let trial = RigidSubset::new(&tree, &state, &[0], 0, pose([-4., 0., 0.]), rd).unwrap();
    assert!(trial.hard_valid(None, [0.; 3]));
    // Equal exclusion spheres, radius R=1, separation d=1.
    let volume = PI * (4. + 1.) * (2.0_f64 - 1.).powi(2) / 12.;
    let expected_accept = (-z * volume).exp();
    let expected_lost = (lambda + z) * volume;
    let n = 30_000;
    let mut rng = StdRng::seed_from_u64(613154);
    let mut sum = 0.;
    let mut sumsq = 0.;
    let mut lost = 0.;
    for _ in 0..n {
        let gate = trial.sample(&mut rng, lambda, z, coarse()).unwrap();
        assert_eq!(gate.gained, 0);
        lost += gate.lost as f64;
        let accept = gate.log_weight.exp().min(1.);
        sum += accept;
        sumsq += accept * accept;
    }
    assert_se(lost / n as f64, expected_lost, expected_lost, n);
    assert_se(
        sum / n as f64,
        expected_accept,
        (sumsq / n as f64 - (sum / n as f64).powi(2)).max(0.),
        n,
    );
    let next = carry(&state, &[0], 0, pose([-4., 0., 0.])).unwrap();
    let reverse = RigidSubset::new(&tree, &next, &[0], 0, state[0], rd).unwrap();
    let mut gained = 0.;
    for _ in 0..n {
        let gate = reverse.sample(&mut rng, lambda, z, coarse()).unwrap();
        assert_eq!(gate.lost, 0);
        assert!(gate.log_weight >= 0.);
        gained += gate.gained as f64;
    }
    assert_se(gained / n as f64, lambda * volume, lambda * volume, n);
}

#[test]
fn triple_overlap_is_union_counted_and_matches_independent_axial_integral() {
    let tree = sphere(0.1);
    let rd = 0.9;
    let lambda = 0.6;
    let z = 0.25;
    let state = [
        pose([-0.6, 0., 0.]),
        pose([0.6, 0., 0.]),
        pose([0., 0., 0.]),
    ];
    let trial = RigidSubset::new(&tree, &state, &[0, 1], 0, pose([3.4, 0., 0.]), rd).unwrap();
    assert!(trial.hard_valid(None, [0.; 3]));
    // Coaxial equal balls: cross-sections are concentric disks. Their union
    // radius is max, and intersection with spectator is min. Independent
    // composite Simpson quadrature of this continuous, piecewise-quadratic area.
    let steps = 20_000;
    let dx = 2. / steps as f64;
    let area = |x: f64| {
        let moving = (1. - (x + 0.6).powi(2)).max(1. - (x - 0.6).powi(2)).max(0.);
        PI * moving.min((1. - x * x).max(0.))
    };
    let integral = (0..=steps)
        .map(|i| {
            let weight = if i == 0 || i == steps {
                1.
            } else if i % 2 == 0 {
                2.
            } else {
                4.
            };
            weight * area(-1. + i as f64 * dx)
        })
        .sum::<f64>()
        * dx
        / 3.;
    let pair_sum = 2. * PI * (4. + 0.6) * (2.0_f64 - 0.6).powi(2) / 12.;
    assert!(
        pair_sum - integral > 0.5,
        "fixture needs substantial three-body correction"
    );
    // The center belongs to both moving exclusion balls and the spectator, but
    // contributes ONE lost event predicate, not two pairwise events.
    assert_eq!(trial.overlap_indicators([0.6, 0., 0.]), (true, false));
    let expected_lost = (lambda + z) * integral;
    let expected_accept = (-z * integral).exp();
    let mut rng = StdRng::seed_from_u64(2314567);
    let n = 40_000;
    let mut lost = 0.;
    let mut sum = 0.;
    let mut sumsq = 0.;
    for _ in 0..n {
        let gate = trial.sample(&mut rng, lambda, z, coarse()).unwrap();
        assert_eq!(gate.gained, 0);
        lost += gate.lost as f64;
        let accept = gate.log_weight.exp().min(1.);
        sum += accept;
        sumsq += accept * accept;
    }
    assert_se(lost / n as f64, expected_lost, expected_lost, n);
    assert_se(
        sum / n as f64,
        expected_accept,
        (sumsq / n as f64 - (sum / n as f64).powi(2)).max(0.),
        n,
    );
    let observed = sum / n as f64;
    assert!(
        (observed - (-z * pair_sum).exp()).abs() > 0.04,
        "test must distinguish union depletion from pairwise surrogate"
    );
}
