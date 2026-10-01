use super::*;
use crate::geometry::{Atom, Shape};
use rand::{RngExt, SeedableRng, rngs::StdRng};

fn tree(atoms: Vec<Atom>) -> SphereTree {
    SphereTree::new(Shape {
        name: "line-test".into(),
        volume: 0.,
        atoms,
    })
    .unwrap()
}
fn sphere(radius: f64) -> SphereTree {
    tree(vec![Atom {
        center: [0.; 3],
        radius,
    }])
}
fn pose(position: Vec3) -> Pose {
    Pose {
        position,
        orientation: [1., 0., 0., 0.],
    }
}
fn interval(lower: f64, upper: f64, lower_closed: bool, upper_closed: bool) -> Interval {
    Interval {
        lower,
        upper,
        lower_closed,
        upper_closed,
    }
}
fn set(intervals: Vec<Interval>) -> IntervalSet {
    IntervalSet::from_intervals(intervals).unwrap()
}
fn near(a: &IntervalSet, b: &IntervalSet) {
    assert_eq!(a.intervals.len(), b.intervals.len(), "{a:?} != {b:?}");
    for (x, y) in a.intervals.iter().zip(&b.intervals) {
        for (u, v) in [(x.lower, y.lower), (x.upper, y.upper)] {
            assert!(
                (u - v).abs() < 2e-11 * (1. + u.abs() + v.abs()),
                "{a:?} != {b:?}"
            );
        }
        assert_eq!(x.lower_closed, y.lower_closed);
        assert_eq!(x.upper_closed, y.upper_closed);
    }
}

#[test]
fn interval_algebra_keeps_missing_points_and_closed_singletons() {
    let a = set(vec![
        interval(-2., 0., false, false),
        interval(0., 2., false, false),
    ]);
    assert_eq!(a.intervals.len(), 2);
    assert!(!a.contains(0.));
    let point = set(vec![interval(0., 0., true, true)]);
    let joined = a.union(&point);
    assert_eq!(joined, set(vec![interval(-2., 2., false, false)]));
    assert_eq!(joined.difference(&point), a);
    assert!(a.intersection(&point).is_empty());
    assert_eq!(joined.intersection(&point), point);
    let all = IntervalSet::segment([-2., 2.]).unwrap();
    let interior = set(vec![interval(-1., 1., false, false)]);
    let outside = all.difference(&interior);
    assert_eq!(
        outside,
        set(vec![
            interval(-2., -1., true, true),
            interval(1., 2., true, true)
        ])
    );
    assert_eq!(all.difference(&all), IntervalSet::empty());
    assert_eq!(all.union(&all), all);
    assert!(IntervalSet::segment([1., -1.]).is_err());
    assert!(Interval::closed(f64::NAN, 1.).is_err());
    assert!(
        IntervalSet::from_intervals(vec![
            interval(0., 1., false, false),
            interval(2., 1., false, false)
        ])
        .is_err()
    );
}

#[test]
fn interval_algebra_matches_boolean_membership_exhaustively() {
    let mut intervals = vec![];
    for lower in -2..=2 {
        for upper in lower..=2 {
            for lo in [false, true] {
                for hi in [false, true] {
                    intervals.push(interval(lower as f64, upper as f64, lo, hi));
                }
            }
        }
    }
    for a in &intervals {
        for b in &intervals {
            let x = set(vec![*a]);
            let y = set(vec![*b]);
            let union = x.union(&y);
            let both = x.intersection(&y);
            let diff = x.difference(&y);
            for n in -6..=6 {
                let s = n as f64 / 2.;
                let px = x.contains(s);
                let py = y.contains(s);
                assert_eq!(union.contains(s), px || py, "union {a:?} {b:?} at {s}");
                assert_eq!(
                    both.contains(s),
                    px && py,
                    "intersection {a:?} {b:?} at {s}"
                );
                assert_eq!(diff.contains(s), px && !py, "difference {a:?} {b:?} at {s}");
            }
        }
    }
}

#[test]
fn spheres_have_analytic_core_contact_and_wall_intervals() {
    let t = sphere(1.);
    let q = translation_intervals(
        &t,
        pose([0.; 3]),
        [2., 0., 0.],
        &t,
        &[pose([0.; 3])],
        [-3., 3.],
        Some(1.),
        Some(SphericalWall {
            center: [0.; 3],
            radius: 5.,
        }),
    )
    .unwrap();
    assert_eq!(q.hard_overlap, set(vec![interval(-1., 1., false, false)]));
    assert_eq!(
        q.contact_overlap,
        Some(set(vec![interval(-1.5, 1.5, false, false)]))
    );
    assert_eq!(q.wall_valid, set(vec![interval(-2., 2., true, true)]));
    let free = q.wall_valid.difference(&q.hard_overlap);
    assert_eq!(
        free,
        set(vec![
            interval(-2., -1., true, true),
            interval(1., 2., true, true)
        ])
    );
    assert_eq!(q.counts.leaf_pairs_tested, 1);
    assert_eq!(q.counts.wall_atoms_tested, 1);
    let disabled = translation_intervals(
        &t,
        pose([0.; 3]),
        [2., 0., 0.],
        &t,
        &[],
        [-3., 3.],
        None,
        None,
    )
    .unwrap();
    assert!(disabled.hard_overlap.is_empty());
    assert!(disabled.contact_overlap.is_none());
    assert_eq!(
        disabled.wall_valid,
        IntervalSet::segment([-3., 3.]).unwrap()
    );
}

#[test]
fn tangent_zero_direction_zero_segment_and_input_guards() {
    let t = sphere(1.);
    let tangent = translation_intervals(
        &t,
        pose([0., 2., 0.]),
        [1., 0., 0.],
        &t,
        &[pose([0.; 3])],
        [-2., 2.],
        Some(0.),
        None,
    )
    .unwrap();
    assert!(tangent.hard_overlap.is_empty());
    assert!(tangent.contact_overlap.unwrap().is_empty());
    let wall_tangent = translation_intervals(
        &t,
        pose([0., 2., 0.]),
        [1., 0., 0.],
        &t,
        &[],
        [-2., 2.],
        None,
        Some(SphericalWall {
            center: [0.; 3],
            radius: 3.,
        }),
    )
    .unwrap();
    assert_eq!(
        wall_tangent.wall_valid,
        set(vec![interval(0., 0., true, true)])
    );
    let static_tangent = translation_intervals(
        &t,
        pose([2., 0., 0.]),
        [0.; 3],
        &t,
        &[pose([0.; 3])],
        [-2., 2.],
        Some(0.1),
        Some(SphericalWall {
            center: [0.; 3],
            radius: 3.,
        }),
    )
    .unwrap();
    assert!(static_tangent.hard_overlap.is_empty());
    assert_eq!(
        static_tangent.contact_overlap,
        Some(IntervalSet::segment([-2., 2.]).unwrap())
    );
    assert_eq!(
        static_tangent.wall_valid,
        IntervalSet::segment([-2., 2.]).unwrap()
    );
    let static_inside = translation_intervals(
        &t,
        pose([0.; 3]),
        [0.; 3],
        &t,
        &[pose([0.; 3])],
        [-2., 2.],
        None,
        None,
    )
    .unwrap();
    assert_eq!(
        static_inside.hard_overlap,
        IntervalSet::segment([-2., 2.]).unwrap()
    );
    let point = translation_intervals(
        &t,
        pose([0.; 3]),
        [1., 0., 0.],
        &t,
        &[pose([0.; 3])],
        [2., 2.],
        None,
        None,
    )
    .unwrap();
    assert!(point.hard_overlap.is_empty());
    assert!(point.wall_valid.contains(2.));
    let impossible = translation_intervals(
        &t,
        pose([0.; 3]),
        [1., 0., 0.],
        &t,
        &[],
        [-2., 2.],
        None,
        Some(SphericalWall {
            center: [0.; 3],
            radius: 0.5,
        }),
    )
    .unwrap();
    assert!(impossible.wall_valid.is_empty());
    assert!(
        translation_intervals(
            &t,
            pose([0.; 3]),
            [f64::NAN, 0., 0.],
            &t,
            &[],
            [-2., 2.],
            None,
            None
        )
        .is_err()
    );
    assert!(
        translation_intervals(
            &t,
            pose([0.; 3]),
            [0.; 3],
            &t,
            &[],
            [-2., 2.],
            Some(-1.),
            None
        )
        .is_err()
    );
    assert!(
        translation_intervals(
            &t,
            Pose {
                position: [0.; 3],
                orientation: [2., 0., 0., 0.]
            },
            [0.; 3],
            &t,
            &[],
            [-2., 2.],
            None,
            None
        )
        .is_err()
    );
}

#[test]
fn dumbbell_keeps_disconnected_contact_shell_and_transformed_wall() {
    let moving = sphere(0.5);
    let fixed = tree(vec![
        Atom {
            center: [-2., 0., 0.],
            radius: 0.5,
        },
        Atom {
            center: [2., 0., 0.],
            radius: 0.5,
        },
    ]);
    let q = translation_intervals(
        &moving,
        pose([3., 4., 5.]),
        [1., 0., 0.],
        &fixed,
        &[pose([3., 4., 5.])],
        [-5., 5.],
        Some(0.5),
        Some(SphericalWall {
            center: [3., 4., 5.],
            radius: 5.,
        }),
    )
    .unwrap();
    assert_eq!(
        q.hard_overlap,
        set(vec![
            interval(-3., -1., false, false),
            interval(1., 3., false, false)
        ])
    );
    assert_eq!(
        q.contact_overlap,
        Some(set(vec![
            interval(-3.5, -0.5, false, false),
            interval(0.5, 3.5, false, false)
        ]))
    );
    let surface = q.contact_overlap.unwrap().difference(&q.hard_overlap);
    assert_eq!(surface.intervals.len(), 4);
    assert!(surface.contains(1.));
    assert!(!surface.contains(0.5));
    assert_eq!(q.wall_valid, set(vec![interval(-4.5, 4.5, true, true)]));
    // Two touching forbidden intervals must leave the hard-valid contact point.
    let one = sphere(1.);
    let two = tree(vec![
        Atom {
            center: [-2., 0., 0.],
            radius: 1.,
        },
        Atom {
            center: [2., 0., 0.],
            radius: 1.,
        },
    ]);
    let q = translation_intervals(
        &one,
        pose([0.; 3]),
        [1., 0., 0.],
        &two,
        &[pose([0.; 3])],
        [-5., 5.],
        None,
        None,
    )
    .unwrap();
    assert_eq!(q.hard_overlap.intervals.len(), 2);
    assert!(!q.hard_overlap.contains(0.));
}

// Independent coefficient/discriminant implementation for moderate random
// geometries. It uses no BVH or analytic primitive from the tested module.
fn brute_ball(
    offset: Vec3,
    direction: Vec3,
    radius: f64,
    segment: [f64; 2],
    closed: bool,
) -> IntervalSet {
    if radius < 0. {
        return IntervalSet::empty();
    }
    let a = dot(direction, direction);
    let b = dot(offset, direction);
    let c = dot(offset, offset) - radius * radius;
    let whole = IntervalSet::segment(segment).unwrap();
    if a == 0. {
        return if c < 0. || (closed && c == 0.) {
            whole
        } else {
            IntervalSet::empty()
        };
    }
    let delta = b * b - a * c;
    if delta < 0. || (delta == 0. && !closed) {
        return IntervalSet::empty();
    }
    let lo = (-b - delta.sqrt()) / a;
    let hi = (-b + delta.sqrt()) / a;
    set(vec![interval(lo, hi, closed, closed)]).intersection(&whole)
}

fn random_pose(rng: &mut StdRng, extent: f64) -> Pose {
    let mut q = std::array::from_fn(|_| rng.random_range(-1.0..1.0));
    let n = q.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut q {
        *x /= n;
    }
    Pose {
        position: std::array::from_fn(|_| rng.random_range(-extent..extent)),
        orientation: q,
    }
}

#[test]
fn random_unions_match_brute_quadratics_and_point_membership() {
    let mut rng = StdRng::seed_from_u64(202610010711);
    for _ in 0..48 {
        let moving = tree(
            (0..13)
                .map(|_| Atom {
                    center: std::array::from_fn(|_| rng.random_range(-3.0..3.0)),
                    radius: rng.random_range(0.2..0.9),
                })
                .collect(),
        );
        let fixed = tree(
            (0..11)
                .map(|_| Atom {
                    center: std::array::from_fn(|_| rng.random_range(-3.0..3.0)),
                    radius: rng.random_range(0.2..0.9),
                })
                .collect(),
        );
        let origin = random_pose(&mut rng, 3.);
        let poses = [random_pose(&mut rng, 5.), random_pose(&mut rng, 5.)];
        let d = std::array::from_fn(|_| rng.random_range(-2.0..2.0));
        let segment = [-4., 4.];
        let wall = SphericalWall {
            center: [1., -0.5, 0.25],
            radius: 9.,
        };
        let q = translation_intervals(
            &moving,
            origin,
            d,
            &fixed,
            &poses,
            segment,
            Some(0.7),
            Some(wall),
        )
        .unwrap();
        let mut hard = IntervalSet::empty();
        let mut contact = IntervalSet::empty();
        let mut valid = IntervalSet::segment(segment).unwrap();
        for a in &moving.shape.atoms {
            let c = origin.apply(a.center);
            valid = valid.intersection(&brute_ball(
                sub(c, wall.center),
                d,
                wall.radius - a.radius,
                segment,
                true,
            ));
            for p in &poses {
                for b in &fixed.shape.atoms {
                    let offset = sub(c, p.apply(b.center));
                    hard = hard.union(&brute_ball(offset, d, a.radius + b.radius, segment, false));
                    contact = contact.union(&brute_ball(
                        offset,
                        d,
                        a.radius + b.radius + 0.7,
                        segment,
                        false,
                    ));
                }
            }
        }
        near(&q.hard_overlap, &hard);
        near(q.contact_overlap.as_ref().unwrap(), &contact);
        near(&q.wall_valid, &valid);
        for k in 0..161 {
            let s = -4. + k as f64 / 20.;
            let translated = Pose {
                position: add(origin.position, scale(d, s)),
                ..origin
            };
            let mut h = false;
            let mut c = false;
            for a in &moving.shape.atoms {
                for p in &poses {
                    for b in &fixed.shape.atoms {
                        let distance = norm(sub(translated.apply(a.center), p.apply(b.center)));
                        h |= distance < a.radius + b.radius;
                        c |= distance < a.radius + b.radius + 0.7;
                    }
                }
            }
            let w = moving.shape.atoms.iter().all(|a| {
                norm(sub(translated.apply(a.center), wall.center)) <= wall.radius - a.radius
            });
            assert_eq!(q.hard_overlap.contains(s), h);
            assert_eq!(q.contact_overlap.as_ref().unwrap().contains(s), c);
            assert_eq!(q.wall_valid.contains(s), w);
        }
    }
}

#[test]
fn bvh_prunes_far_nodes_and_certifies_full_wall_containment() {
    let t = tree(
        (0..128)
            .map(|i| Atom {
                center: [(i % 8) as f64, (i / 8) as f64, 0.],
                radius: 0.2,
            })
            .collect(),
    );
    let q = translation_intervals(
        &t,
        pose([0.; 3]),
        [1., 0., 0.],
        &t,
        &[pose([0., 10000., 0.])],
        [-2., 2.],
        Some(1.),
        Some(SphericalWall {
            center: [0.; 3],
            radius: 100.,
        }),
    )
    .unwrap();
    assert_eq!(q.counts.node_pairs_visited, 1);
    assert_eq!(q.counts.node_pairs_pruned, 1);
    assert_eq!(q.counts.leaf_pairs_tested, 0);
    assert_eq!(q.counts.wall_nodes_contained, 1);
    assert_eq!(q.counts.wall_atoms_tested, 0);
    assert!(q.hard_overlap.is_empty());
    assert!(q.contact_overlap.unwrap().is_empty());
    let moving = sphere(0.2);
    let local = translation_intervals(
        &moving,
        pose([0.; 3]),
        [1., 0., 0.],
        &t,
        &[pose([0.; 3])],
        [-1., 1.],
        Some(0.1),
        None,
    )
    .unwrap();
    assert!(local.counts.leaf_pairs_tested < 8, "{:?}", local.counts);
    assert!(local.counts.node_pairs_pruned > 0);
    let mut brute = IntervalSet::empty();
    for atom in &t.shape.atoms {
        brute = brute.union(&brute_ball(
            scale(atom.center, -1.),
            [1., 0., 0.],
            0.2 + atom.radius,
            [-1., 1.],
            false,
        ));
    }
    near(&local.hard_overlap, &brute);
    eprintln!(
        "128-atom local-line cost: {:?}; far-union cost: {:?}",
        local.counts, q.counts
    );
}

#[test]
fn distant_and_tiny_directions_do_not_lose_chords_or_mutate_inputs() {
    let t = sphere(1.);
    let origin = pose([-1e8, 0., 0.]);
    let fixed = [pose([0.; 3])];
    let q = translation_intervals(
        &t,
        origin,
        [1., 0., 0.],
        &t,
        &fixed,
        [1e8 - 3., 1e8 + 3.],
        None,
        None,
    )
    .unwrap();
    assert_eq!(
        q.hard_overlap,
        set(vec![interval(1e8 - 2., 1e8 + 2., false, false)])
    );
    assert_eq!(origin, pose([-1e8, 0., 0.]));
    assert_eq!(fixed, [pose([0.; 3])]);
    let tiny = translation_intervals(
        &t,
        pose([0.; 3]),
        [1e-300, 0., 0.],
        &t,
        &fixed,
        [-1., 1.],
        None,
        None,
    )
    .unwrap();
    assert_eq!(tiny.hard_overlap, IntervalSet::segment([-1., 1.]).unwrap());
}
