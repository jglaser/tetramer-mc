use super::*;
use crate::geometry::{Atom, Shape};

fn q(a: f64, b: f64, c: f64, segment: [f64; 2]) -> IntervalSet {
    quadratic_intervals(
        Wide::scalar(a),
        Wide::scalar(b),
        Wide::scalar(c),
        1.,
        segment,
    )
    .unwrap()
}

#[test]
fn quadratic_all_degrees_and_strict_tangencies() {
    let seg = [-3., 3.];
    let up = q(1., 0., -1., seg);
    assert!(up.contains(0.));
    assert!(!up.contains(-1.));
    assert!(!up.contains(1.));
    assert_eq!(up.intervals().len(), 1);
    let down = q(-1., 0., 1., seg);
    assert!(!down.contains(0.));
    assert!(down.contains(-3.));
    assert!(down.contains(3.));
    assert_eq!(down.intervals().len(), 2);
    assert!(q(1., 0., 0., seg).is_empty());
    let punctured = q(-1., 0., 0., seg);
    assert_eq!(punctured.intervals().len(), 2);
    assert!(!punctured.contains(0.));
    assert!(punctured.contains(-1.));
    assert!(q(0., 2., -2., seg).contains(0.));
    assert!(!q(0., 2., -2., seg).contains(1.));
    assert!(q(0., -2., 2., seg).contains(3.));
    assert!(!q(0., -2., 2., seg).contains(1.));
    assert_eq!(q(0., 0., -1., seg), IntervalSet::segment(seg).unwrap());
    assert!(q(0., 0., 0., seg).is_empty());
    assert!(q(0., 0., 1., seg).is_empty());
    assert!(q(1., 0., 1., seg).is_empty());
    assert_eq!(q(-1., 0., -1., seg), IntervalSet::segment(seg).unwrap());
    assert_eq!(
        q(0., 0., -1., [2., 2.]),
        IntervalSet::segment([2., 2.]).unwrap()
    );
}

#[test]
fn narrow_compensated_and_extreme_finite_chords() {
    // Retain the low part that ordinary 1-epsilon rounding would discard.
    let epsilon = 1e-12;
    let c = Wide::scalar(1.).sub(Wide::scalar(epsilon).mul(Wide::scalar(epsilon)));
    let intervals =
        quadratic_intervals(Wide::scalar(1.), Wide::scalar(-2.), c, 1., [-1e250, 1e250]).unwrap();
    assert!(intervals.contains(1.));
    assert!(!intervals.contains(1. - 2e-12));
    let i = intervals.intervals()[0];
    assert!((i.lower - (1. - epsilon)).abs() < 2e-16);
    assert!((i.upper - (1. + epsilon)).abs() < 2e-16);
    assert_eq!(
        q(1e-250, 0., -1e-250, [-1e300, 1e300]),
        q(1., 0., -1., [-1e300, 1e300])
    );
    let huge = quadratic_intervals(
        Wide::scalar(1.),
        Wide::scalar(0.),
        Wide::scalar(-1.),
        1e200,
        [-1e300, 1e300],
    )
    .unwrap();
    assert!(huge.contains(1e199));
    assert!(!huge.contains(1e201));
    let one_side = q(1., 0., -1., [1e200, 1e300]);
    assert!(one_side.is_empty());
}

#[test]
fn exact_single_sphere_limits_and_near_contact() {
    // Rotation about z: p(s)=((1-s²)/(1+s²), 2s/(1+s²), 0).
    let a = [1., 0., 0.];
    let v = [0.; 3];
    let seg = [-1e200, 1e200];
    let same = sphere_overlap_intervals(a, a, 1., v, 2, 1., seg).unwrap();
    let edge = 1. / 3_f64.sqrt();
    assert!((same.intervals()[0].lower + edge).abs() < 3e-16);
    assert!((same.intervals()[0].upper - edge).abs() < 3e-16);
    let opposite = sphere_overlap_intervals(a, [-1., 0., 0.], 1., v, 2, 1., seg).unwrap();
    assert!(!opposite.contains(0.));
    assert!(opposite.contains(2.));
    assert!(opposite.contains(-2.));
    // Exact linear polynomial B*s: sphere tangent to both limiting points.
    let linear =
        sphere_overlap_intervals(a, [0., 1., 0.], 2_f64.sqrt(), v, 2, 1., [-3., 3.]).unwrap();
    assert!(linear.contains(1.));
    assert!(!linear.contains(-1.));
    let at_origin = sphere_overlap_intervals([0.; 3], [0.; 3], 0.5, v, 2, 1., seg).unwrap();
    assert_eq!(at_origin, IntervalSet::segment(seg).unwrap());
    let axis_constant =
        sphere_overlap_intervals([0., 0., 1.], [0., 0., 2.], 1., v, 2, 1., seg).unwrap();
    assert!(axis_constant.is_empty());
    let tangent = sphere_overlap_intervals(a, [2., 0., 0.], 1., v, 2, 1., seg).unwrap();
    assert!(tangent.is_empty());
    let wide = sphere_overlap_intervals(a, a, 2., v, 2, 1., seg).unwrap();
    assert_eq!(wide, IntervalSet::segment(seg).unwrap()); // excluded antipode is at infinity
    let tiny = sphere_overlap_intervals(a, a, 1e-10, v, 2, 1., [-2., 2.]).unwrap();
    assert!(tiny.contains(0.));
    assert!(!tiny.contains(1e-10));
    assert!((tiny.intervals()[0].upper - 5e-11).abs() < 1e-25);
}

fn tree(atoms: &[(Vec3, f64)]) -> SphereTree {
    SphereTree::new(Shape {
        name: "fixed Cayley reference".into(),
        volume: 0.,
        atoms: atoms
            .iter()
            .map(|&(center, radius)| Atom { center, radius })
            .collect(),
    })
    .unwrap()
}

fn direct(
    moving: &SphereTree,
    axis: &CayleyAxis,
    fixed: &SphereTree,
    poses: &[Pose],
    raw: f64,
) -> bool {
    let moving_pose = axis.pose(raw).unwrap();
    poses.iter().any(|pose| {
        moving.shape.atoms.iter().any(|a| {
            fixed.shape.atoms.iter().any(|b| {
                let d = sub(moving_pose.apply(a.center), pose.apply(b.center));
                dot(d, d) < (a.radius + b.radius).powi(2)
            })
        })
    })
}

fn brute(
    moving: &SphereTree,
    axis: &CayleyAxis,
    fixed: &SphereTree,
    poses: &[Pose],
    seg: [f64; 2],
) -> IntervalSet {
    let inverse = transpose(rotation(axis.chart_orientation));
    let mut result = IntervalSet::empty();
    for pose in poses {
        for a in &moving.shape.atoms {
            for b in &fixed.shape.atoms {
                let a0 = matvec(axis.anchor_rotation, a.center);
                let b0 = matvec(inverse, sub(pose.apply(b.center), axis.world_center));
                result = result.union(
                    &sphere_overlap_intervals(
                        a0,
                        b0,
                        a.radius + b.radius,
                        axis.fixed_cayley,
                        axis.axis,
                        axis.length_scale,
                        seg,
                    )
                    .unwrap(),
                );
            }
        }
    }
    result
}

#[test]
fn all_axes_sphere_dumbbell_asymmetric_bvh_matches_brute_and_direct() {
    let shapes = [
        tree(&[([0.; 3], 0.4)]),
        tree(&[([-1., 0., 0.], 0.35), ([1., 0., 0.], 0.45)]),
        tree(&[
            ([-0.7, 0.2, 0.1], 0.31),
            ([0.9, -0.4, 0.3], 0.39),
            ([0.1, 0.8, -0.6], 0.27),
            ([0.2, -0.1, 0.9], 0.43),
        ]),
    ];
    let segment = [-12., 12.];
    for (shape_id, moving) in shapes.iter().enumerate() {
        for fixed in &shapes {
            for selected in 0..3 {
                for shifted in [false, true] {
                    let axis = CayleyAxis {
                        world_center: if shifted { [3., -2., 1.] } else { [0.; 3] },
                        chart_orientation: quaternion(cayley(if shifted {
                            [0.3, -0.2, 0.4]
                        } else {
                            [0.; 3]
                        })),
                        anchor_rotation: cayley([-0.1, 0.5, 0.15]),
                        fixed_cayley: [0.17, -0.35, 0.25],
                        axis: selected,
                        length_scale: 1.8,
                    };
                    let poses = [
                        Pose {
                            position: add(axis.world_center, [1.3, 0.2, -0.1]),
                            orientation: quaternion(cayley([0.2, 0.1, -0.3])),
                        },
                        Pose {
                            position: add(axis.world_center, [50., -20., 0.]),
                            orientation: [1., 0., 0., 0.],
                        },
                    ];
                    let got = cayley_axis_intervals(moving, &axis, fixed, &poses, segment).unwrap();
                    assert_eq!(
                        got.hard_overlap,
                        brute(moving, &axis, fixed, &poses, segment)
                    );
                    assert!(got.counts.node_pairs_pruned > 0);
                    for sample in 0..257 {
                        let raw = -12. + 24. * sample as f64 / 256.;
                        let overlap = direct(moving, &axis, fixed, &poses, raw);
                        assert_eq!(
                            got.hard_overlap.contains(raw),
                            overlap,
                            "shape {shape_id} axis {selected} shifted {shifted} raw {raw}"
                        );
                        assert_eq!(got.hard_free.contains(raw), !overlap);
                    }
                }
            }
        }
    }
}

#[test]
fn orbit_identity_and_selected_coordinate_is_ignored() {
    let a = [0.3, -0.7, 0.9];
    for selected in 0..3 {
        for v in [[0.2, -0.5, 0.3], [1e100, -2e100, 0.5e100]] {
            let orbit = Orbit::new(v, selected, 2.).unwrap();
            for raw in [-1e150, -2., 0., 3., 1e150] {
                let mut c = v;
                c[selected] = raw / 2.;
                let p = matvec(cayley(c), a);
                let t = raw / orbit.raw_scale;
                let half = 1_f64.hypot(t);
                let q = [
                    1. / half,
                    orbit.axis[0] * t / half,
                    orbit.axis[1] * t / half,
                    orbit.axis[2] * t / half,
                ];
                let expected = matvec(rotation(q), matvec(orbit.rotation_zero, a));
                assert!(norm(sub(p, expected)) < 3e-15);
            }
        }
    }
    let x =
        sphere_overlap_intervals(a, [1., 0.1, 0.2], 0.8, [0.2, 0.3, 0.], 2, 2., [-5., 5.]).unwrap();
    let y = sphere_overlap_intervals(a, [1., 0.1, 0.2], 0.8, [0.2, 0.3, 1e100], 2, 2., [-5., 5.])
        .unwrap();
    assert_eq!(x, y);
}

#[test]
fn invalid_or_unrepresentable_inputs_are_errors_not_empty_intervals() {
    assert!(sphere_overlap_intervals([0.; 3], [1.; 3], 0.5, [0.; 3], 3, 1., [-1., 1.]).is_err());
    assert!(sphere_overlap_intervals([0.; 3], [1.; 3], 0.5, [0.; 3], 1, 0., [-1., 1.]).is_err());
    assert!(sphere_overlap_intervals([0.; 3], [1.; 3], 0.5, [0.; 3], 1, 1., [1., -1.]).is_err());
    assert!(sphere_overlap_intervals([1.; 3], [0.; 3], 1e-300, [0.; 3], 1, 1., [-1., 1.]).is_err());
    assert!(
        sphere_overlap_intervals([1.; 3], [0.; 3], 1., [1e300; 3], 1, 1e300, [-1., 1.]).is_err()
    );
    let shape = tree(&[([0.; 3], 0.5)]);
    let invalid = CayleyAxis {
        world_center: [0.; 3],
        chart_orientation: [1., 0., 0., 0.],
        anchor_rotation: [[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        fixed_cayley: [0.; 3],
        axis: 0,
        length_scale: 1.,
    };
    assert!(cayley_axis_intervals(&shape, &invalid, &shape, &[], [-1., 1.]).is_err());
}
