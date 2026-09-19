//! Independent geometric and equilibrium checks for the spherical kernels.
//!
//! The statistical test uses exact inclusion-exclusion volumes, including a
//! nonzero triple intersection. It tests actual continuous PPP sampling on a
//! finite isometry orbit, not full-state ergodicity or formal FP interval bounds.
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::f64::consts::PI;
use tetramer_mc::{
    geometry::{Atom, Shape, SphereTree},
    math::*,
    spherical::{self, Container, HalfTurn},
};

fn pose(position: Vec3) -> Pose {
    Pose {
        position,
        orientation: [1., 0., 0., 0.],
    }
}

fn shape(atoms: Vec<Atom>) -> SphereTree {
    SphereTree::new(Shape {
        name: "spherical reference".into(),
        volume: 0.,
        atoms,
    })
    .unwrap()
}

fn sphere(radius: f64) -> SphereTree {
    shape(vec![Atom {
        center: [0.; 3],
        radius,
    }])
}

fn asymmetric_shape() -> SphereTree {
    shape(vec![
        Atom {
            center: [0.21, 0., 0.],
            radius: 0.13,
        },
        Atom {
            center: [-0.08, 0.18, 0.],
            radius: 0.12,
        },
        Atom {
            center: [-0.09, -0.07, 0.17],
            radius: 0.10,
        },
        Atom {
            center: [0.02, -0.05, -0.13],
            radius: 0.09,
        },
    ])
}

fn near(a: f64, b: f64, tolerance: f64) {
    assert!(
        (a - b).abs() <= tolerance,
        "{a} != {b} (tolerance {tolerance})"
    );
}

fn vector_near(a: Vec3, b: Vec3, tolerance: f64) {
    for k in 0..3 {
        near(a[k], b[k], tolerance);
    }
}

fn poses_near(a: Pose, b: Pose, tolerance: f64) {
    vector_near(a.position, b.position, tolerance);
    for k in 0..3 {
        vector_near(
            rotation(a.orientation)[k],
            rotation(b.orientation)[k],
            tolerance,
        );
    }
}

fn sorted_groups(mut groups: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    for group in &mut groups {
        group.sort_unstable();
    }
    groups.sort_unstable();
    groups
}

#[test]
fn wall_uses_oriented_atomic_spheres_and_halfturn_preserves_shape() {
    let tree = shape(vec![Atom {
        center: [1., 0., 0.],
        radius: 0.5,
    }]);
    let wall = Container::new(3., &tree).unwrap();
    let outward = pose([1.6, 0., 0.]);
    assert!(!wall.contains(outward));
    let inward = Pose {
        orientation: [0., 0., 1., 0.],
        ..outward
    };
    assert!(wall.contains(inward));
    assert!(norm(inward.position) + tree.bound > wall.radius);
    near(wall.clearance(inward), 1.9, 1e-14);
    let tangent = pose([1.5, 0., 0.]);
    assert!(wall.contains(tangent));
    near(wall.clearance(tangent), 0., 1e-14);

    let tree = asymmetric_shape();
    let wall = Container::new(7., &tree).unwrap();
    let transform = HalfTurn::new([1., 2., 3.]).unwrap();
    let original = Pose {
        position: [2., -1.1, 3.],
        orientation: quaternion(cayley([0.4, -0.2, 0.3])),
    };
    let transformed = transform.apply(original);
    poses_near(transform.apply(transformed), original, 4e-14);
    near(wall.clearance(transformed), wall.clearance(original), 4e-14);
    // This preserved body-frame radial vector is why GCA alone is not an
    // ergodic rigid-particle sampler; center shifts/local moves complement it.
    vector_near(
        matvec(transpose(rotation(original.orientation)), original.position),
        matvec(
            transpose(rotation(transformed.orientation)),
            transformed.position,
        ),
        4e-14,
    );
    for atom in &tree.shape.atoms {
        vector_near(
            transformed.apply(atom.center),
            transform.point(original.apply(atom.center)),
            4e-14,
        );
    }
    assert!(HalfTurn::new([0.; 3]).is_err());
    poses_near(
        HalfTurn::new([0., 0., 1e-320]).unwrap().apply(original),
        HalfTurn::new([0., 0., 1.]).unwrap().apply(original),
        1e-14,
    );
    assert!(Container::new(0., &tree).is_err());
    assert!(Container::new(1e300, &tree).is_err());
}

#[test]
fn every_hard_component_flip_is_valid_and_reversible_for_anisotropic_bodies() {
    let tree = asymmetric_shape();
    let wall = Container::new(7., &tree).unwrap();
    let transform = HalfTurn::new([1., 2., 3.]).unwrap();
    let a = pose([4., 0., 0.]);
    let b = Pose {
        position: [0., 4., 0.],
        orientation: quaternion(cayley([0.1, 0.3, -0.2])),
    };
    let original = vec![a, transform.apply(a), b, transform.apply(b), pose([0.; 3])];
    spherical::validate_state(&tree, &wall, &original).unwrap();
    let (groups, shadow) = spherical::hard_components(&tree, &original, transform).unwrap();
    let groups = sorted_groups(groups);
    assert_eq!(groups, vec![vec![0, 1], vec![2, 3], vec![4]]);
    for mask in 0..(1 << groups.len()) {
        let mut next = original.clone();
        for (k, group) in groups.iter().enumerate() {
            if mask & (1 << k) != 0 {
                for &i in group {
                    next[i] = shadow[i];
                }
            }
        }
        spherical::validate_state(&tree, &wall, &next).unwrap();
        let (reverse_groups, reverse_shadow) =
            spherical::hard_components(&tree, &next, transform).unwrap();
        assert_eq!(sorted_groups(reverse_groups), groups);
        for (k, group) in groups.iter().enumerate() {
            if mask & (1 << k) != 0 {
                for &i in group {
                    next[i] = reverse_shadow[i];
                }
            }
        }
        for (a, b) in next.into_iter().zip(&original) {
            poses_near(a, *b, 5e-14);
        }
    }
}

/// Direct world-frame quadratic intersection, independent of body-frame and
/// pruning code in Container. Inputs here are well clear of quadratic tangency.
fn brute_interval(tree: &SphereTree, radius: f64, state: &[Pose], direction: Vec3) -> (f64, f64) {
    let mut lower = f64::NEG_INFINITY;
    let mut upper = f64::INFINITY;
    let aa = dot(direction, direction);
    for p in state {
        for atom in &tree.shape.atoms {
            let c = p.apply(atom.center);
            let bb = dot(c, direction);
            let cc = dot(c, c) - (radius - atom.radius).powi(2);
            let discriminant = bb * bb - aa * cc;
            lower = lower.max((-bb - discriminant.sqrt()) / aa);
            upper = upper.min((-bb + discriminant.sqrt()) / aa);
        }
    }
    (lower, upper)
}

#[test]
fn common_shift_chord_matches_atomic_reference_and_reverse_interval() {
    let tree = asymmetric_shape();
    let wall = Container::new(7., &tree).unwrap();
    let mut rng = StdRng::seed_from_u64(20260919011);
    for _ in 0..150 {
        let original: Vec<_> = [[-2., 1., 0.], [2., -1., 1.], [0., 2., -2.]]
            .into_iter()
            .map(|position| {
                let mut p = uniform_pose(&mut rng, [1.; 3]);
                p.position = add(position, scale(sub(p.position, [0.5; 3]), 0.4));
                p
            })
            .collect();
        let direction = [
            rng.random_range(0.3..1.4),
            rng.random_range(-1.3..1.3),
            rng.random_range(-1.3..1.3),
        ];
        spherical::validate_state(&tree, &wall, &original).unwrap();
        let expected = brute_interval(&tree, wall.radius, &original, direction);
        let interval = wall.translation_interval(&original, direction).unwrap();
        near(interval.lower, expected.0, 3e-13);
        near(interval.upper, expected.1, 3e-13);
        let mut next = original.clone();
        let u = rng.random_range(0.05..0.95);
        let outcome = wall.center_shift(&mut next, direction, u).unwrap();
        assert!(!outcome.degenerate);
        spherical::validate_state(&tree, &wall, &next).unwrap();
        let reverse = wall.translation_interval(&next, direction).unwrap();
        let t = dot(outcome.displacement, direction) / dot(direction, direction);
        near(reverse.lower, interval.lower - t, 4e-13);
        near(reverse.upper, interval.upper - t, 4e-13);
        for j in 1..next.len() {
            vector_near(
                sub(next[j].position, next[0].position),
                sub(original[j].position, original[0].position),
                1e-14,
            );
            assert_eq!(next[j].orientation, original[j].orientation);
        }
        let return_u = (-t - reverse.lower) / (reverse.upper - reverse.lower);
        wall.center_shift(&mut next, direction, return_u).unwrap();
        for (a, b) in next.into_iter().zip(original) {
            poses_near(a, b, 8e-13);
        }
    }
    let state = [pose([6., 0., 0.]), pose([-6., 0., 0.]), pose([0., 0., 0.])];
    let interval = wall.translation_interval(&state, [1., 0., 0.]).unwrap();
    assert!(interval.bodies_skipped > 0);
    let reference = brute_interval(&tree, wall.radius, &state, [1., 0., 0.]);
    near(interval.lower, reference.0, 2e-14);
    near(interval.upper, reference.1, 2e-14);

    let tree = sphere(1.);
    let wall = Container::new(1., &tree).unwrap();
    let mut state = [pose([0.; 3])];
    assert!(
        wall.center_shift(&mut state, [1., 2., 3.], 0.5)
            .unwrap()
            .degenerate
    );
    assert!(wall.center_shift(&mut state, [1., 0., 0.], 0.).is_err());
    assert!(wall.translation_interval(&state, [0.; 3]).is_err());
}

#[test]
fn zero_activity_is_the_hard_component_coin_kernel() {
    let tree = sphere(0.3);
    let wall = Container::new(8., &tree).unwrap();
    let transform = HalfTurn::new([0., 0., 1.]).unwrap();
    let original = vec![
        pose([2., 0., 0.]),
        pose([-2., 0., 0.]),
        pose([0., 3., 0.]),
        pose([0., 0., 2.]),
    ];
    let (groups, shadow) = spherical::hard_components(&tree, &original, transform).unwrap();
    for seed in 0..50 {
        let mut reference = StdRng::seed_from_u64(seed);
        let coins: Vec<bool> = (0..original.len()).map(|_| reference.random()).collect();
        let mut expected = original.clone();
        for group in &groups {
            if coins[group[0]] {
                for &i in group {
                    expected[i] = shadow[i];
                }
            }
        }
        let mut actual = original.clone();
        let mut rng = StdRng::seed_from_u64(seed);
        let stats =
            spherical::update(&tree, &wall, &mut actual, transform, 0.7, 0., &mut rng).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(stats.poisson_probes, 0);
        assert_eq!(stats.hyperedges, 0);
        assert_eq!(stats.components, groups.len());
        assert_eq!(rng.random::<u64>(), reference.random::<u64>());
    }
}

fn overlap(radius: f64, distance: f64) -> f64 {
    if distance >= 2. * radius {
        0.
    } else {
        PI * (4. * radius + distance) * (2. * radius - distance).powi(2) / 12.
    }
}

fn orbit_state(mask: usize, transform: HalfTurn) -> Vec<Pose> {
    [1.6, 2.2, 2.8]
        .into_iter()
        .enumerate()
        .map(|(i, x)| {
            let p = pose([x, 0., 0.]);
            if mask & (1 << i) == 0 {
                p
            } else {
                transform.apply(p)
            }
        })
        .collect()
}

fn exact_union_volume(mask: usize, radius: f64, include_triple: bool) -> f64 {
    let mut result = 3. * 4. * PI * radius.powi(3) / 3.;
    for i in 0..3 {
        for j in i + 1..3 {
            if (mask >> i) & 1 == (mask >> j) & 1 {
                result -= overlap(radius, 0.6 * (j - i) as f64);
            }
        }
    }
    // The middle exclusion contains the outer pair's intersection: every
    // point within both outer balls is within the ball at their midpoint.
    if include_triple && (mask == 0 || mask == 7) {
        result += overlap(radius, 1.2);
    }
    result
}

fn probabilities(activity: f64, radius: f64, include_triple: bool) -> Vec<f64> {
    let mut p: Vec<_> = (0..8)
        .map(|mask| (-activity * exact_union_volume(mask, radius, include_triple)).exp())
        .collect();
    let total: f64 = p.iter().sum();
    for w in &mut p {
        *w /= total;
    }
    p
}

#[test]
fn implicit_gca_has_manybody_equilibrium_on_a_continuous_three_sphere_orbit() {
    let tree = sphere(0.15);
    let wall = Container::new(5., &tree).unwrap();
    let transform = HalfTurn::new([0., 0., 1.]).unwrap();
    let rd = 0.7;
    let activity = 1.4;
    let samples = 16_000;
    let mut counts = [[0_u64; 8]; 8];
    let mut triple_hyperedge = false;
    for (source, row) in counts.iter_mut().enumerate() {
        let original = orbit_state(source, transform);
        let mut rng = StdRng::seed_from_u64(20260919071 + source as u64);
        for _ in 0..samples {
            let mut state = original.clone();
            let stats =
                spherical::update(&tree, &wall, &mut state, transform, rd, activity, &mut rng)
                    .unwrap();
            triple_hyperedge |= stats.max_hyperedge_size == 3;
            let destination = state.iter().enumerate().fold(0, |mask, (i, p)| {
                mask | (usize::from(p.position[0] < 0.) << i)
            });
            row[destination] += 1;
        }
    }
    assert!(
        triple_hyperedge,
        "the test must exercise a genuine three-owner point"
    );
    let transition: Vec<Vec<f64>> = counts
        .iter()
        .map(|row| row.iter().map(|n| *n as f64 / samples as f64).collect())
        .collect();
    let weights = probabilities(activity, rd + 0.15, true);
    for i in 0..8 {
        for j in i + 1..8 {
            let forward = weights[i] * transition[i][j];
            let reverse = weights[j] * transition[j][i];
            let variance = (weights[i].powi(2) * transition[i][j] * (1. - transition[i][j])
                + weights[j].powi(2) * transition[j][i] * (1. - transition[j][i]))
                / samples as f64;
            assert!(
                (forward - reverse).abs() < 7. * variance.sqrt() + 1. / samples as f64,
                "flux {i}<->{j}: {forward} vs {reverse}, standard error {}",
                variance.sqrt()
            );
        }
    }
    for destination in 0..8 {
        let stationary: f64 = (0..8)
            .map(|i| weights[i] * transition[i][destination])
            .sum();
        let variance: f64 = (0..8)
            .map(|i| {
                weights[i].powi(2) * transition[i][destination] * (1. - transition[i][destination])
                    / samples as f64
            })
            .sum();
        assert!(
            (stationary - weights[destination]).abs() < 7. * variance.sqrt() + 1. / samples as f64
        );
    }
    // This fixture discriminates from the tempting but incorrect pairwise
    // approximation: omitting its triple inclusion-exclusion term fails the
    // same stationary test by many standard errors.
    let pairwise = probabilities(activity, rd + 0.15, false);
    let largest_wrong_z = (0..8)
        .map(|j| {
            let stationary: f64 = (0..8).map(|i| pairwise[i] * transition[i][j]).sum();
            let variance: f64 = (0..8)
                .map(|i| {
                    pairwise[i].powi(2) * transition[i][j] * (1. - transition[i][j])
                        / samples as f64
                })
                .sum();
            (stationary - pairwise[j]).abs() / variance.sqrt()
        })
        .fold(0., f64::max);
    assert!(
        largest_wrong_z > 10.,
        "triple fixture insufficiently discriminating: {largest_wrong_z} sigma"
    );
}

#[test]
fn common_shift_samples_the_one_sphere_uniform_volume() {
    // Hit-and-run with an independent isotropic chord direction. Batch means
    // allow serial correlation instead of treating saved states as independent.
    let tree = sphere(0.25);
    let wall = Container::new(2.25, &tree).unwrap();
    let mut state = [pose([0.; 3])];
    let mut rng = StdRng::seed_from_u64(20260919099);
    let mut batch_means = Vec::new();
    for batch in 0..101 {
        let mut radial_sum = 0.;
        for _ in 0..500 {
            let z = rng.random_range(-1_f64..1_f64);
            let angle = 2. * PI * rng.random::<f64>();
            let radial = (1. - z * z).sqrt();
            let d = [radial * angle.cos(), radial * angle.sin(), z];
            let u = rng.random_range(f64::EPSILON..1. - f64::EPSILON);
            wall.center_shift(&mut state, d, u).unwrap();
            radial_sum += dot(state[0].position, state[0].position);
        }
        if batch > 0 {
            batch_means.push(radial_sum / 500.);
        }
    }
    let n = batch_means.len() as f64;
    let mean = batch_means.iter().sum::<f64>() / n;
    let variance = batch_means.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.);
    let reference = 3. * 2_f64.powi(2) / 5.;
    assert!(
        (mean - reference).abs() < 6. * (variance / n).sqrt(),
        "<r²> {mean}, expected {reference}"
    );
}
