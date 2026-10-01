use super::*;
use crate::{
    contact_distances::ContactFrame,
    geometry::{Atom, Shape},
    math::*,
};
use rand::{RngExt, SeedableRng, rngs::StdRng};

fn circle(radius: f64) -> ContactCircle {
    ContactCircle {
        center: [0.; 3],
        radius,
        radii: [1., 1.],
        axis: [0., 0., 1.],
        basis: [[1., 0., 0.], [0., 1., 0.]],
    }
}
fn pose(position: Vec3) -> Pose {
    Pose {
        position,
        orientation: [1., 0., 0., 0.],
    }
}
fn tree(atoms: Vec<Atom>) -> SphereTree {
    SphereTree::new(Shape {
        name: "toy".into(),
        volume: 1.,
        atoms,
    })
    .unwrap()
}
fn near(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a:e} != {b:e}; tol={tol:e}");
}

#[test]
fn analytic_empty_full_tangent_and_axis_cases() {
    let c = circle(1.);
    assert!(
        ball_forbidden_arcs([2., 0., 0.], 1., &c)
            .unwrap()
            .is_empty()
    );
    let full = ball_forbidden_arcs([2., 0., 0.], 3.1, &c).unwrap();
    assert_eq!(full, whole().unwrap());
    let tangent = ball_forbidden_arcs([1., 0., 0.], 2., &c).unwrap();
    assert!(!tangent.contains(0.));
    assert!(tangent.contains(PI));
    let allowed = whole().unwrap().difference(&tangent);
    assert_eq!(allowed.intervals(), &[Interval::closed(0., 0.).unwrap()]);
    assert_eq!(
        azimuth_allowed_mass(&AzimuthLaw::uniform(), &allowed).unwrap(),
        0.
    );
    assert!(
        ball_forbidden_arcs([0., 0., 0.], 1., &c)
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        ball_forbidden_arcs([0., 0., 0.], 1.01, &c).unwrap(),
        whole().unwrap()
    );
    assert!(
        ball_forbidden_arcs([0., 0., 4.], 5., &circle(3.))
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        ball_forbidden_arcs([0., 0., 2.], 2.3, &c).unwrap(),
        whole().unwrap()
    );
    let fixed = circle(0.);
    assert!(
        ball_forbidden_arcs([1., 0., 0.], 1., &fixed)
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        ball_forbidden_arcs([0.9, 0., 0.], 1., &fixed).unwrap(),
        whole().unwrap()
    );
}

#[test]
fn wrapped_strict_arc_endpoints_have_direct_witnesses() {
    let c = circle(1.);
    let arcs = ball_forbidden_arcs([-1., 0., 0.], 1., &c).unwrap();
    assert_eq!(arcs.intervals().len(), 2);
    near(arcs.length(), 2. * PI / 3., 2e-15);
    assert!(arcs.contains(0.));
    assert!(!arcs.contains(PI));
    for interval in arcs.intervals() {
        for angle in [interval.lower, interval.upper] {
            if angle == 0. || angle == TAU {
                continue;
            }
            near(norm(add([-1., 0., 0.], c.point(angle).unwrap())), 1., 2e-15);
            let left = norm(add([-1., 0., 0.], c.point(angle - 1e-7).unwrap())) < 1.;
            let right = norm(add([-1., 0., 0.], c.point(angle + 1e-7).unwrap())) < 1.;
            assert_ne!(left, right);
        }
    }
    let tiny = ball_forbidden_arcs([-2., 0., 0.], 1. + 1e-12, &c).unwrap();
    assert!(tiny.length() > 0. && tiny.length() < 1e-5);
}

fn brute(
    moving: &SphereTree,
    c: &ContactCircle,
    orientation: [f64; 4],
    fixed: &SphereTree,
    poses: &[Pose],
) -> IntervalSet {
    let origin = pose(c.center);
    let origin = Pose {
        orientation,
        ..origin
    };
    let mut all = IntervalSet::empty();
    for p in poses {
        for a in &moving.shape.atoms {
            for b in &fixed.shape.atoms {
                all = all.union(
                    &ball_forbidden_arcs(
                        sub(origin.apply(a.center), p.apply(b.center)),
                        a.radius + b.radius,
                        c,
                    )
                    .unwrap(),
                );
            }
        }
    }
    all
}
fn direct_gap(
    moving: &SphereTree,
    c: &ContactCircle,
    orientation: [f64; 4],
    fixed: &SphereTree,
    poses: &[Pose],
    phi: f64,
) -> f64 {
    let mobile = Pose {
        position: c.point(phi).unwrap(),
        orientation,
    };
    let mut gap = f64::INFINITY;
    for p in poses {
        for a in &moving.shape.atoms {
            for b in &fixed.shape.atoms {
                gap = gap.min(
                    norm(sub(mobile.apply(a.center), p.apply(b.center))) - a.radius - b.radius,
                );
            }
        }
    }
    gap
}

#[test]
fn tiny_union_bvh_matches_exhaustive_pairs_and_direct_angles() {
    let mut rng = StdRng::seed_from_u64(610071001);
    let mut witnesses = 0;
    for _ in 0..48 {
        let moving = tree(
            (0..7)
                .map(|_| Atom {
                    center: std::array::from_fn(|_| rng.random_range(-1.5..1.5)),
                    radius: rng.random_range(0.08..0.3),
                })
                .collect(),
        );
        let fixed = tree(
            (0..6)
                .map(|_| Atom {
                    center: std::array::from_fn(|_| rng.random_range(-1.5..1.5)),
                    radius: rng.random_range(0.08..0.3),
                })
                .collect(),
        );
        let frame = ContactFrame::new([-0.5, 0.2, -0.1], [0.9, -0.1, 0.7]).unwrap();
        let mut c = frame.circle([1.5, 1.7]).unwrap();
        c.center = std::array::from_fn(|_| rng.random_range(-0.2..0.2));
        let orientation = quaternion(cayley(std::array::from_fn(|_| rng.random_range(-0.3..0.3))));
        let poses = [
            pose([0.; 3]),
            Pose {
                position: [2., 0.2, 0.1],
                orientation: quaternion(cayley([0.1, -0.2, 0.3])),
            },
        ];
        let actual = circle_forbidden_arcs(&moving, &c, orientation, &fixed, &poses).unwrap();
        let expected = brute(&moving, &c, orientation, &fixed, &poses);
        assert_eq!(actual.forbidden, expected);
        assert_eq!(actual.allowed, whole().unwrap().difference(&expected));
        for i in 0..257 {
            let phi = TAU * (i as f64 + 0.37) / 257.;
            let gap = direct_gap(&moving, &c, orientation, &fixed, &poses, phi);
            if gap.abs() > 1e-12 {
                assert_eq!(actual.forbidden.contains(phi), gap < 0.);
            }
        }
        for interval in actual.forbidden.intervals() {
            for phi in [interval.lower, interval.upper] {
                if phi == 0. || phi == TAU {
                    continue;
                }
                near(
                    direct_gap(&moving, &c, orientation, &fixed, &poses, phi),
                    0.,
                    3e-14,
                );
                witnesses += 1;
            }
        }
    }
    assert!(witnesses > 100);
}

#[test]
fn distant_bvh_prunes_without_leaf_work_and_inputs_remain_unchanged() {
    let t = tree(
        (0..128)
            .map(|i| Atom {
                center: [i as f64 * 0.2, 0., 0.],
                radius: 0.04,
            })
            .collect(),
    );
    let c = circle(1.);
    let p = pose([1000., 0., 0.]);
    let before = serde_json::to_vec(&t.shape).unwrap();
    let result = circle_forbidden_arcs(&t, &c, [1., 0., 0., 0.], &t, &[p]).unwrap();
    assert!(result.forbidden.is_empty());
    assert_eq!(result.allowed, whole().unwrap());
    assert_eq!(result.counts.node_pairs_visited, 1);
    assert_eq!(result.counts.node_pairs_pruned, 1);
    assert_eq!(result.counts.leaf_pairs_tested, 0);
    assert_eq!(serde_json::to_vec(&t.shape).unwrap(), before);
    let mut invalid = c;
    invalid.basis[0] = [2., 0., 0.];
    assert!(circle_forbidden_arcs(&t, &invalid, [1., 0., 0., 0.], &t, &[p]).is_err());
}

#[test]
fn azimuth_masses_normalize_split_cuts_and_match_quadrature() {
    for mode in [0., 0.7, PI, TAU - 0.01] {
        for gamma in [0.01, 0.2, 1., PI, 50.] {
            let law = AzimuthLaw::new(mode, gamma, 0.9).unwrap();
            assert_eq!(azimuth_allowed_mass(&law, &whole().unwrap()).unwrap(), 1.);
            assert_eq!(
                azimuth_allowed_mass(&law, &IntervalSet::empty()).unwrap(),
                0.
            );
            let parts = (0..97)
                .map(|i| {
                    Interval::closed(TAU * i as f64 / 97., TAU * (i + 1) as f64 / 97.).unwrap()
                })
                .collect::<Vec<_>>();
            near(
                parts
                    .iter()
                    .map(|p| azimuth_interval_mass(&law, p).unwrap())
                    .sum(),
                1.,
                8e-15,
            );
            for (a, b) in [(0., 0.3), (0.2, 4.), (3., TAU), (mode, mode + 1e-10)] {
                if b > TAU {
                    continue;
                }
                let interval = Interval::closed(a, b).unwrap();
                let actual = azimuth_interval_mass(&law, &interval).unwrap();
                // Independent composite midpoint quadrature; enough points for
                // gamma>=.01, and tiny intervals also test cancellation resistance.
                let n = 131072;
                let reference = (0..n)
                    .map(|i| {
                        law.density(a + (b - a) * (i as f64 + 0.5) / n as f64)
                            .unwrap()
                    })
                    .sum::<f64>()
                    * (b - a)
                    / n as f64;
                near(actual, reference, 2e-7 * reference.max(1e-12));
            }
        }
    }
    let law = AzimuthLaw::uniform();
    near(
        azimuth_interval_mass(&law, &Interval::closed(0.4, 1.2).unwrap()).unwrap(),
        0.8 / TAU,
        1e-16,
    );
    assert!(azimuth_interval_mass(&law, &Interval::closed(-1., 0.2).unwrap()).is_err());
}

#[test]
fn mass_complements_include_tangent_singletons_without_probability() {
    let c = circle(1.);
    for offset in [[-1., 0., 0.], [0.4, 0.8, 0.1], [1., 0., 0.]] {
        for radius in [0.1, 0.7, 1., 2., 3.] {
            let forbidden = ball_forbidden_arcs(offset, radius, &c).unwrap();
            let allowed = whole().unwrap().difference(&forbidden);
            let law = AzimuthLaw::new(0.7, 0.03, 0.9).unwrap();
            near(
                azimuth_allowed_mass(&law, &forbidden).unwrap()
                    + azimuth_allowed_mass(&law, &allowed).unwrap(),
                1.,
                2e-14,
            );
        }
    }
}

#[test]
fn one_ulp_extrema_match_exact_binary64_decimal_fixtures() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/data/contact_circle_tangent_reference.json"
    ))
    .unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let c = circle(case["circle_radius"].as_f64().unwrap());
        let offset: Vec3 = serde_json::from_value(case["fixed_offset"].clone()).unwrap();
        let forbidden =
            ball_forbidden_arcs(offset, case["radius_sum"].as_f64().unwrap(), &c).unwrap();
        let allowed = whole().unwrap().difference(&forbidden);
        let expected = case["allowed_length"].as_f64().unwrap();
        eprintln!(
            "circle fixture {}: allowed={} reference={expected}",
            case["name"],
            allowed.length()
        );
        near(allowed.length(), expected, 5e-14);
        match case["analytic_topology"].as_str().unwrap() {
            "none" => assert!(forbidden.is_empty()),
            "all" => assert!(allowed.is_empty()),
            "all_except_tangent_point" => {
                assert!(!allowed.is_empty());
                assert_eq!(allowed.length(), 0.);
            }
            "partial" => assert!(allowed.length() > 0. && forbidden.length() > 0.),
            other => panic!("Unexpected fixture topology {other}"),
        }
    }
}
