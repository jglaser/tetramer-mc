//! Memory support, volume measure, slot independence, and dynamic-chart checks.
//! An independently implemented analytic stationary-law test is maintained in
//! contact_memory_stationary.rs; these tests exercise the remaining interfaces.
use anyhow::Result;
use rand::{SeedableRng, rngs::StdRng};
use serde_json::json;
use std::f64::consts::PI;
use tetramer_mc::{
    contact_memory::{MemoryConfig, MemoryCounts, MemoryProposalKind, MemoryState},
    depletion::GateOptions,
    geometry::{Atom, Shape, SphereTree},
    math::*,
    proposal::FrozenRelativePoseProposal,
};

fn shape(atoms: Vec<Atom>) -> SphereTree {
    SphereTree::new(Shape {
        name: "contact-memory reference".into(),
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
fn pose(t: Vec3) -> Pose {
    Pose {
        position: t,
        orientation: [1., 0., 0., 0.],
    }
}
fn near(x: f64, y: f64, tol: f64) {
    assert!((x - y).abs() < tol, "{x} vs {y}");
}

#[test]
fn preparation_is_deterministic_and_slots_are_independent_pairs() -> Result<()> {
    let tree = sphere(0.5);
    let cfg = MemoryConfig {
        slots: 5,
        ..Default::default()
    };
    let resolved = cfg.resolve(&tree, 0., 0.)?;
    let first = MemoryState::initialize(&tree, &resolved, &mut StdRng::seed_from_u64(123))?;
    let second = MemoryState::initialize(&tree, &resolved, &mut StdRng::seed_from_u64(123))?;
    assert_eq!(first, second);
    let frozen_cfg = MemoryConfig {
        attempts_per_sweep: 0,
        ..cfg.clone()
    }
    .resolve(&tree, 0., 0.)?;
    assert_eq!(
        MemoryState::initialize(&tree, &frozen_cfg, &mut StdRng::seed_from_u64(123))?,
        first
    );
    for p in &first.poses {
        assert!(norm(p.position) > 1. + cfg.initial_gap);
        assert!(norm(p.position) < resolved.radius);
    }
    // Memory slots are separate anchored pair configurations, not M mobile
    // particles sharing one box: coincident memory entries are allowed.
    let repeated = MemoryState {
        poses: vec![first.poses[0]; resolved.slots],
    };
    repeated.validate(&tree, &resolved)?;
    let mut malformed = repeated.clone();
    malformed.poses[0] = pose([0.; 3]);
    assert!(malformed.validate(&tree, &resolved).is_err());
    malformed = repeated.clone();
    malformed.poses.pop();
    assert!(malformed.validate(&tree, &resolved).is_err());
    let override_cfg = MemoryConfig {
        depletant_radius: Some(0.8),
        depletant_activity: Some(0.2),
        radius: Some(4.),
        ..Default::default()
    };
    let resolved = override_cfg.resolve(&tree, 1.5, 0.035)?;
    assert_eq!(resolved.depletant_radius, 0.8);
    assert_eq!(resolved.depletant_activity, 0.2);
    assert_eq!(resolved.radius, 4.);
    Ok(())
}

#[test]
fn global_redraw_has_uniform_ball_measure_and_zero_activity_accepts_every_valid_endpoint()
-> Result<()> {
    let tree = sphere(0.5);
    let cfg = MemoryConfig {
        slots: 1,
        radius: Some(3.),
        global_probability: 1.,
        ..Default::default()
    }
    .resolve(&tree, 0.6, 0.)?;
    let mut state = MemoryState {
        poses: vec![pose([2., 0., 0.])],
    };
    let mut rng = StdRng::seed_from_u64(7123431);
    let mut radii2 = 0.;
    let mut traces = 0.;
    let mut hard_rejects = 0;
    let mut counts = MemoryCounts::default();
    let samples = 5000;
    for _ in 0..samples {
        let old = state.poses[0];
        let m = state.update(&tree, &cfg, GateOptions::default(), &mut rng)?;
        assert_eq!(m.kind, MemoryProposalKind::Global);
        assert_eq!(m.old_pose, old);
        assert_eq!(m.retained_pose, state.poses[0]);
        assert!(!m.outside_ball);
        let r = norm(m.proposed_pose.position);
        radii2 += r * r;
        let rotation = rotation(m.proposed_pose.orientation);
        traces += (0..3).map(|i| rotation[i][i]).sum::<f64>();
        assert_eq!(m.hard_valid, r >= 1.);
        assert_eq!(m.accepted, m.hard_valid);
        if let Some(gate) = &m.gate {
            assert_eq!(gate.log_weight, 0.);
            assert_eq!(gate.raw_points, 0);
        }
        if !m.hard_valid {
            hard_rejects += 1;
            assert_eq!(state.poses[0], old);
        }
        counts.record(&m);
    }
    near(radii2 / samples as f64, 3. * 9. / 5., 0.18);
    near(traces / samples as f64, 0., 0.12);
    assert!(hard_rejects > 80 && hard_rejects < 280);
    assert_eq!(counts.global_attempted, samples);
    assert_eq!(counts.accepted + counts.hard_rejected, counts.attempted);
    assert_eq!(counts.since(&MemoryCounts::default()), counts);
    Ok(())
}

#[test]
fn local_kernel_retains_rejected_states_and_support_includes_inner_free_pockets() -> Result<()> {
    let tree = sphere(0.5);
    let cfg = MemoryConfig {
        slots: 1,
        radius: Some(1.3),
        global_probability: 0.,
        local_translation_std_a: 0.8,
        local_small_angle_std_degrees: 12.,
        ..Default::default()
    }
    .resolve(&tree, 0., 0.1)?;
    let mut state = MemoryState {
        poses: vec![pose([1.15, 0., 0.])],
    };
    let mut rng = StdRng::seed_from_u64(166651);
    let mut counts = MemoryCounts::default();
    for _ in 0..500 {
        let old = state.poses[0];
        let outcome = state.update(
            &tree,
            &cfg,
            GateOptions {
                max_cells: 63,
                max_depth: 6,
                min_width: 0.1,
            },
            &mut rng,
        )?;
        assert_eq!(outcome.kind, MemoryProposalKind::Local);
        if !outcome.accepted {
            assert_eq!(state.poses[0], old);
        }
        if outcome.hard_valid {
            // At rd=0 two hard-free core unions have no positive-volume
            // exclusion overlap, so even nonzero bath activity is indifferent.
            assert_eq!(outcome.gate.as_ref().unwrap().log_weight, 0.);
            assert!(outcome.accepted);
        }
        counts.record(&outcome);
    }
    state.validate(&tree, &cfg)?;
    assert!(counts.outside_ball > 10 && counts.hard_rejected > 5 && counts.accepted > 5);

    let dumbbell = shape(vec![
        Atom {
            center: [-2., 0., 0.],
            radius: 0.4,
        },
        Atom {
            center: [2., 0., 0.],
            radius: 0.4,
        },
    ]);
    let cfg = MemoryConfig {
        slots: 1,
        radius: Some(5.),
        ..Default::default()
    }
    .resolve(&dumbbell, 0.5, 0.)?;
    let pocket = MemoryState {
        poses: vec![Pose {
            position: [0.; 3],
            orientation: quaternion(cayley([0., 0., 1.])),
        }],
    };
    // Two perpendicular dumbbells can share the origin while every atom pair
    // remains hard-free. No minimum-center-distance or outer-ray veto applies.
    pocket.validate(&dumbbell, &cfg)?;
    Ok(())
}

fn base() -> FrozenRelativePoseProposal {
    let mut cov = [[0.; 6]; 6];
    for i in 0..6 {
        cov[i][i] = 1.;
    }
    let model = json!({"coordinate_convention":"anchor-body-relative","shape_sha256":"a".repeat(64),
        "angular_length":4.,"anchors":[{"position":[4.,0.,0.],"rotation":IDENTITY}],
        "means":vec![[0.;6]],"covariances":[cov],"weights":[1.]});
    FrozenRelativePoseProposal::from_json_str_open(
        &model.to_string(),
        [20.; 3],
        0.1,
        &"a".repeat(64),
    )
    .unwrap()
}

fn memory_density(t: Vec3, r: Mat3, anchor: Pose, dt: f64, angle: f64, ell: f64) -> f64 {
    let rel = matmul(r, transpose(rotation(anchor.orientation)));
    let denominator = 1. + (0..3).map(|i| rel[i][i]).sum::<f64>();
    let c = [
        (rel[2][1] - rel[1][2]) / denominator,
        (rel[0][2] - rel[2][0]) / denominator,
        (rel[1][0] - rel[0][1]) / denominator,
    ];
    let angular = ell * angle.to_radians() / 2.;
    let radial = dot(sub(t, anchor.position), sub(t, anchor.position)) / (dt * dt);
    let rotation2 = ell * ell * dot(c, c) / (angular * angular);
    let log_gaussian =
        -3. * (2. * PI).ln() - 3. * dt.ln() - 3. * angular.ln() - 0.5 * (radial + rotation2);
    (log_gaussian + 3. * ell.ln() + 2. * PI.ln() + 2. * (1. + dot(c, c)).ln()).exp()
}

#[test]
fn memory_charts_have_correct_normalized_mixture_density_and_fixed_label_weights() -> Result<()> {
    let base = base();
    let mut poses = vec![
        pose([2., 0., 0.]),
        Pose {
            position: [0., 2., 0.],
            orientation: quaternion(cayley([0.1, -0.15, 0.2])),
        },
    ];
    let (mass, dt, angle) = (0.4, 0.8, 8.);
    let dynamic = base.with_contact_components(&poses, mass, dt, angle)?;
    assert_eq!(base.component_count(), 1);
    assert_eq!(dynamic.component_count(), 3);
    assert_eq!(dynamic.component_weights(), vec![0.6, 0.2, 0.2]);
    for (t, r) in [
        ([2.1, 0.2, -0.1], cayley([0.02, 0.04, -0.03])),
        ([0.2, 2.1, 0.], cayley([0.13, -0.11, 0.21])),
    ] {
        let expected = (1. - mass) * base.relative_log_density(t, r)?.exp()
            + mass / poses.len() as f64
                * poses
                    .iter()
                    .map(|p| memory_density(t, r, *p, dt, angle, base.angular_length()))
                    .sum::<f64>();
        near(
            dynamic.relative_log_density(t, r)?.exp(),
            expected,
            2e-11 * (1. + expected),
        );
    }
    poses[0].position = [3., -0.2, 0.1];
    let changed = base.with_contact_components(&poses, mass, dt, angle)?;
    assert_eq!(changed.component_weights(), dynamic.component_weights());
    assert_ne!(
        changed.relative_log_density([2., 0., 0.], IDENTITY)?,
        dynamic.relative_log_density([2., 0., 0.], IDENTITY)?
    );
    assert_eq!(
        changed.selected_components(&[2, 0, 2])?.component_weights(),
        vec![1. / 3.; 3]
    );
    assert!(base.with_contact_components(&poses, 0., dt, angle).is_err());
    assert!(base.with_contact_components(&poses, 1., dt, angle).is_err());
    assert!(
        base.with_contact_components(&poses, mass, 0., angle)
            .is_err()
    );
    assert!(base.with_contact_components(&[], mass, dt, angle).is_err());
    Ok(())
}
