use anyhow::Result;
use tetramer_mc::{
    depletion_surrogate::DimerDepletionSurrogate,
    geometry::{Atom, Shape, SphereTree},
    math::Pose,
};
fn pose(x: f64) -> Pose {
    Pose {
        position: [x, 0., 0.],
        orientation: [1., 0., 0., 0.],
    }
}
fn sphere() -> SphereTree {
    SphereTree::new(Shape {
        name: "unit sphere".into(),
        volume: 4. * std::f64::consts::PI / 3.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius: 1.,
        }],
    })
    .unwrap()
}

#[test]
fn boolean_shielding_internal_overlap_and_empty_limits() -> Result<()> {
    let tree = sphere();
    let points = vec![[0.; 3], [0.5, 0., 0.]];
    let no_fixed = DimerDepletionSurrogate::new(&tree, points.clone(), &[], 3., 2.)?;
    assert_eq!(no_fixed.score([pose(0.), pose(5.)])?.log_surrogate, 0.);
    let s = no_fixed.score([pose(0.), pose(0.)])?;
    assert_eq!(s.internal_unshielded, [2, 2]);
    assert_eq!(s.overlap_volume_estimate, 6.);
    let shield =
        DimerDepletionSurrogate::new(&tree, points.clone(), &[pose(0.), pose(0.)], 3., 2.)?;
    let s = shield.score([pose(0.), pose(0.)])?;
    assert_eq!(s.spectator_covered, [2, 2]);
    assert_eq!(s.internal_unshielded, [0, 0]);
    assert_eq!(s.overlap_volume_estimate, 12.); // 2v, not 3v from pair overlaps
    assert_eq!(
        DimerDepletionSurrogate::new(&tree, vec![], &[], 1., 2.)?
            .score([pose(0.), pose(0.)])?
            .log_surrogate,
        0.
    );
    assert_eq!(
        DimerDepletionSurrogate::new(&tree, points, &[pose(0.)], 1., 0.)?
            .score([pose(0.), pose(0.)])?
            .log_surrogate,
        0.
    );
    Ok(())
}

#[test]
fn prune_equals_enumeration_boundary_duplicates_and_swap() -> Result<()> {
    let tree = sphere();
    let points = vec![[1., 0., 0.], [-1., 0., 0.], [0.; 3], [0.; 3]];
    for distance in [2. - 1e-12, 2., 2. + 1e-12, 20.] {
        let g = DimerDepletionSurrogate::new(
            &tree,
            points.clone(),
            &[pose(distance), pose(50.)],
            2.,
            0.3,
        )?;
        let x = [pose(0.), pose(1.5)];
        let a = g.score(x)?;
        let b = g.score_unpruned(x)?;
        assert_eq!(a.twice_overlap_units, b.twice_overlap_units);
        assert_eq!(a.log_surrogate, b.log_surrogate);
        let swap = g.score([x[1], x[0]])?;
        assert_eq!(a.log_surrogate, swap.log_surrogate);
        assert_eq!(
            a.spectator_covered,
            [swap.spectator_covered[1], swap.spectator_covered[0]]
        );
    }
    Ok(())
}

#[test]
fn fixed_cloud_rigid_frame_covariance_and_input_errors() -> Result<()> {
    let tree = sphere();
    let pts = vec![[0.2, 0.3, 0.1], [-0.2, -0.3, -0.1]];
    let g = DimerDepletionSurrogate::new(&tree, pts.clone(), &[pose(0.2)], 1., 1.)?;
    let transform = |p: Pose| Pose {
        position: [3., 4. + p.position[0], 5.],
        orientation: [0.5, 0.5, 0.5, 0.5],
    };
    let h = DimerDepletionSurrogate::new(&tree, pts, &[transform(pose(0.2))], 1., 1.)?;
    assert_eq!(
        g.score([pose(0.), pose(1.5)])?.twice_overlap_units,
        h.score([transform(pose(0.)), transform(pose(1.5))])?
            .twice_overlap_units
    );
    assert!(DimerDepletionSurrogate::new(&tree, vec![[1.1, 0., 0.]], &[], 1., 1.).is_err());
    assert!(DimerDepletionSurrogate::new(&tree, vec![], &[], 0., 1.).is_err());
    assert!(g.score([pose(f64::INFINITY), pose(0.)]).is_err());
    Ok(())
}

#[test]
fn deterministic_box_quadrature_matches_sphere_lens() -> Result<()> {
    let tree = sphere();
    let side = 80usize;
    let step = 2. / side as f64;
    let mut points = vec![];
    for i in 0..side {
        for j in 0..side {
            for k in 0..side {
                let p = [i, j, k].map(|n| -1. + (n as f64 + 0.5) * step);
                if tree.contains(p, 0.) {
                    points.push(p);
                }
            }
        }
    }
    let g = DimerDepletionSurrogate::new(&tree, points, &[], step.powi(3), 1.)?;
    for d in [0., 0.5, 1., 1.5, 2., 2.5] {
        let exact = if d <= 2. {
            std::f64::consts::PI * (4. + d) * (2. - d).powi(2) / 12.
        } else {
            0.
        };
        assert!((g.score([pose(0.), pose(d)])?.overlap_volume_estimate - exact).abs() < 0.012);
    }
    Ok(())
}

#[test]
fn anisotropic_sphere_union_matches_direct_atom_enumeration() -> Result<()> {
    use tetramer_mc::geometry::Placed;
    let tree = SphereTree::new(Shape {
        name: "asymmetric union".into(),
        volume: 1.,
        atoms: vec![
            Atom {
                center: [-0.8, 0., 0.],
                radius: 0.8,
            },
            Atom {
                center: [0.8, 0.2, 0.],
                radius: 1.,
            },
            Atom {
                center: [0., 0.2, 0.8],
                radius: 0.5,
            },
        ],
    })?;
    let contains = |p: [f64; 3]| {
        tree.shape.atoms.iter().any(|a| {
            (0..3).map(|k| (p[k] - a.center[k]).powi(2)).sum::<f64>() <= a.radius * a.radius
        })
    };
    let mut points = vec![];
    for i in -8..=8 {
        for j in -8..=8 {
            for k in -8..=8 {
                let p = [i, j, k].map(|n| n as f64 / 4.);
                if contains(p) {
                    points.push(p);
                }
            }
        }
    }
    let fixed = [
        pose(-0.7),
        Pose {
            position: [1., 1., 0.],
            orientation: [0.5, 0.5, 0.5, 0.5],
        },
        pose(100.),
    ];
    let members = [
        Pose {
            position: [0.2, 0.1, -0.3],
            orientation: [0.5, -0.5, 0.5, -0.5],
        },
        pose(1.),
    ];
    let mut direct = 0;
    for i in 0..2 {
        for &p in &points {
            let world = Placed::new(members[i]).apply(p);
            direct += if fixed
                .iter()
                .any(|&s| contains(Placed::new(s).unapply(world)))
            {
                2
            } else if contains(Placed::new(members[1 - i]).unapply(world)) {
                1
            } else {
                0
            };
        }
    }
    let g = DimerDepletionSurrogate::new(&tree, points, &fixed, 0.125, 0.035)?;
    assert_eq!(g.score(members)?.twice_overlap_units, direct);
    assert_eq!(g.score_unpruned(members)?.twice_overlap_units, direct);
    Ok(())
}
