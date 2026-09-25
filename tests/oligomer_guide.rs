use anyhow::Result;
use std::f64::consts::PI;
use tetramer_mc::{
    docking::{DockingMethod, DockingProposal},
    math::{IDENTITY, Pose, add, cayley, quaternion},
    oligomer_guide::{OligomerGuideConfig, anchor_pool},
    proposal::{FrozenRelativePoseProposal, GaussianComponentParameters},
    rigid_subset::carry,
};
fn pose(position: [f64; 3]) -> Pose {
    Pose {
        position,
        orientation: [1., 0., 0., 0.],
    }
}

#[test]
fn anchor_pool_is_invariant_when_oligomer_changes_external_contacts() -> Result<()> {
    let state = vec![
        pose([0., 0., 0.]),
        pose([0.9, 0., 0.]),
        pose([2., 0., 0.]),
        pose([2., 1., 0.]),
        pose([5., 0., 0.]),
        pose([2., -1., 0.]),
    ];
    let members = [0, 1];
    let next = carry(
        &state,
        &members,
        0,
        Pose {
            position: [4., 0., 0.],
            orientation: quaternion(cayley([0.2, 0.4, -0.1])),
        },
    )?;
    for primary in 2..6 {
        for count in [1, 2, 4, 20] {
            let a = anchor_pool(&state, &members, primary, count)?;
            assert_eq!(a, anchor_pool(&next, &members, primary, count)?);
            assert_eq!(a.len(), count.min(4));
            assert!(a.iter().all(|i| !members.contains(i)));
        }
    }
    assert_eq!(anchor_pool(&state, &members, 2, 3)?, vec![2, 3, 5]);
    assert!(anchor_pool(&state, &members, 0, 2).is_err());
    assert!(anchor_pool(&state, &members, 2, 0).is_err());
    Ok(())
}

#[test]
fn complete_guide_density_matches_independent_haar_gaussian_formula() -> Result<()> {
    let component = GaussianComponentParameters {
        anchor_position: [0.; 3],
        anchor_rotation: IDENTITY,
        mean: [0.; 6],
        covariance: std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1. } else { 0. })),
        weight: 1.,
    };
    let hash = "a".repeat(64);
    let model = FrozenRelativePoseProposal::from_components_open(
        vec![component],
        1.,
        [20.; 3],
        0.2,
        &hash,
        &hash,
    )?;
    let center = [3., -2., 1.];
    let p = DockingProposal::new(model, DockingMethod::PosteriorInvolution, 0.9, center)?;
    let anchors = [
        pose(add(center, [1., 0., 0.])),
        pose(add(center, [-1., 0., 0.])),
    ];
    let c = [0.2, -0.1, 0.3];
    let x = Pose {
        position: add(center, [0.4, 0.3, -0.1]),
        orientation: quaternion(cayley(c)),
    };
    let c2: f64 = c.iter().map(|a| a * a).sum();
    let mut learned = 0.;
    for a in anchors {
        let t2: f64 = x
            .position
            .iter()
            .zip(a.position)
            .map(|(u, v)| (u - v) * (u - v))
            .sum();
        learned +=
            (-0.5 * (t2 + c2)).exp() / (2. * PI).powi(3) * PI.powi(2) * (1. + c2).powi(2) / 2.;
    }
    let expected = 0.2 / 20_f64.powi(3) + 0.8 * learned;
    assert!((p.guide_log_density(x, &anchors)?.exp() - expected).abs() < 1e-14);
    // A far pose is protected by the complete uniform branch, even when every
    // Gaussian contact density is negligible.
    let far = pose(add(center, [9., 9., 9.]));
    assert!(p.guide_log_density(far, &anchors)? >= (0.2 / 20_f64.powi(3)).ln() - 1e-14);
    assert!(p.guide_log_density(x, &[]).is_err());
    Ok(())
}

#[test]
fn guide_parameters_fail_before_sampling() {
    let mut c = OligomerGuideConfig::default();
    assert!(c.validate().is_ok());
    c.anchor_count = 0;
    assert!(c.validate().is_err());
    c.anchor_count = 1;
    c.score_power = -1.;
    assert!(c.validate().is_err());
    c.score_power = f64::NAN;
    assert!(c.validate().is_err());
    c.score_power = 0.;
    c.steps = 0;
    assert!(c.validate().is_ok() && !c.enabled());
}
