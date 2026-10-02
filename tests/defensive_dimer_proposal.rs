//! Deterministic proposal/measure controls only; no physical sampling campaign.
use anyhow::Result;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde_json::json;
use std::f64::consts::PI;
use tetramer_mc::{
    defensive_dimer_proposal::{DefensiveBranch, DefensiveDimerProposal},
    dimer_tree_proposal::{tree_coordinates, tree_members},
    docking::{DockingMethod, DockingProposal},
    math::{Pose, cayley, norm, quaternion, rotation, sub},
    proposal::FrozenRelativePoseProposal,
};

const SHA: &str = "0000000000000000000000000000000000000000000000000000000000000000";
const ID: Pose = Pose {
    position: [0.; 3],
    orientation: [1., 0., 0., 0.],
};

fn model(periodic: bool, overflowing: bool) -> Result<FrozenRelativePoseProposal> {
    let covariance: [[f64; 6]; 6] =
        std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1. } else { 0. }));
    let position = if overflowing {
        [1e308, 0., 0.]
    } else {
        [0.; 3]
    };
    let mean = if overflowing {
        [1e308, 0., 0., 0., 0., 0.]
    } else {
        [0.; 6]
    };
    let raw = json!({"coordinate_convention":"anchor-body-relative","shape_sha256":SHA,
        "angular_length":1.,"weights":[1.],
        "anchors":[{"position":position,"rotation":rotation(ID.orientation)}],
        "means":[mean],"covariances":[covariance]})
    .to_string();
    if periodic {
        FrozenRelativePoseProposal::from_json_str(&raw, [80.; 3], 0.1, SHA)
    } else {
        FrozenRelativePoseProposal::from_json_str_open(&raw, [80.; 3], 0.1, SHA)
    }
}

fn proposal() -> Result<DockingProposal> {
    DockingProposal::new(
        model(false, false)?,
        DockingMethod::PosteriorInvolution,
        0.7,
        [17., -5., 3.],
    )
}

fn pose(t: [f64; 3], c: [f64; 3]) -> Pose {
    Pose {
        position: t,
        orientation: quaternion(cayley(c)),
    }
}

fn close(a: f64, b: f64, tolerance: f64) {
    assert!((a - b).abs() < tolerance, "{a} != {b}");
}
fn same(a: Pose, b: Pose) {
    assert!(norm(sub(a.position, b.position)) < 1e-10);
    let ra = rotation(a.orientation);
    let rb = rotation(b.orientation);
    assert!(
        ra.iter()
            .flatten()
            .zip(rb.iter().flatten())
            .all(|(a, b)| (a - b).abs() < 1e-10)
    );
}

// Independent one-chart standard Gaussian times inverse Cayley/Haar Jacobian.
fn learned_density(p: Pose) -> f64 {
    let square = p.orientation[0] * p.orientation[0];
    if square == 0. {
        return 0.;
    }
    let c2 = (1. - square).max(0.) / square;
    let t2 = p.position.iter().map(|v| v * v).sum::<f64>();
    (-3. * (2. * PI).ln() - 0.5 * (t2 + c2) + 2. * PI.ln() - 2. * square.ln()).exp()
}
fn full_density(p: Pose, half_width: f64, alpha: f64) -> f64 {
    let uniform = if p.position.iter().all(|x| x.abs() <= half_width) {
        (2. * half_width).powi(-3)
    } else {
        0.
    };
    alpha * uniform + (1. - alpha) * learned_density(p)
}

#[test]
fn pure_uniform_and_gaussian_limits_have_the_required_normalized_measure() -> Result<()> {
    let base = proposal()?;
    let uniform = DefensiveDimerProposal::new(&base, 2.5, 1.)?;
    for c in [[0.; 3], [0.2, -0.5, 0.8], [1., 2., 3.]] {
        let density = uniform.density(pose([1., -2., 0.3], c))?;
        close(density.log_full.exp() * 5_f64.powi(3), 1., 2e-14);
    }
    let learned = DefensiveDimerProposal::new(&base, 2.5, 0.)?;
    for p in [
        pose([0.; 3], [0.; 3]),
        pose([5., -3., 0.2], [1., -0.5, 0.3]),
    ] {
        let d = learned.density(p)?;
        assert_eq!(d.log_full, d.log_learned);
        close(d.log_full.exp(), learned_density(p), 1e-13);
    }
    // Deterministic quadrature of each normalized radial Gaussian factor after
    // cancelling the rotational Haar Jacobian. The omitted r>10 tail is <1e-20.
    let n = 1000;
    let h = 10. / n as f64;
    let integral = |rotational: bool| -> Result<f64> {
        let mut sum = 0.;
        for i in 0..=n {
            let r = i as f64 * h;
            let p = if rotational {
                pose([0.; 3], [r, 0., 0.])
            } else {
                pose([r, 0., 0.], [0.; 3])
            };
            let density = learned.density(p)?.log_full.exp();
            let center_gaussian = (2. * PI).powf(-1.5);
            let radial = if rotational {
                density / (center_gaussian * PI * PI * (1. + r * r).powi(2))
            } else {
                density / (center_gaussian * PI * PI)
            };
            let weight = if i == 0 || i == n {
                1.
            } else if i % 2 == 0 {
                2.
            } else {
                4.
            };
            sum += weight * 4. * PI * r * r * radial;
        }
        Ok(sum * h / 3.)
    };
    close(integral(false)?, 1., 2e-11);
    close(integral(true)?, 1., 2e-11);
    Ok(())
}

#[test]
fn complete_product_density_uses_both_components_at_every_endpoint() -> Result<()> {
    let base = proposal()?;
    let defensive = DefensiveDimerProposal::new(&base, 2., 0.35)?;
    let anchor = pose([3., -2., 5.], [0.3, 0.2, -0.4]);
    let old = [
        pose([0.1, -0.2, 0.3], [0.2, 0.1, -0.1]),
        pose([3., 0.1, 0.2], [0.4, -0.2, 0.1]),
    ];
    let new = [
        pose([-0.5, 0.4, 0.7], [0.3, -0.2, 0.5]),
        pose([0.4, 0.1, 0.8], [0.1, 0.3, 0.2]),
    ];
    let [root, child] = tree_members(anchor, old)?;
    let [nr, nc] = tree_members(anchor, new)?;
    let d = defensive.correction(anchor, root, child, nr, nc)?;
    let expected = old
        .map(|p| full_density(p, 2., 0.35).ln())
        .iter()
        .sum::<f64>()
        - new
            .map(|p| full_density(p, 2., 0.35).ln())
            .iter()
            .sum::<f64>();
    close(d.log_reverse_forward, expected, 2e-11);
    let backwards = defensive.correction(anchor, nr, nc, root, child)?;
    close(
        d.log_reverse_forward + backwards.log_reverse_forward,
        0.,
        2e-11,
    );
    assert_eq!(d.log_tree_coordinate_jacobian, 0.);
    assert_eq!(d.selection_log_reverse_forward, 0.);
    let mut rng = StdRng::seed_from_u64(9287);
    let mut seen = [0; 2];
    for _ in 0..32 {
        let outcome = defensive.propose(&mut rng, anchor, root, child)?;
        let candidate = outcome.candidate.as_ref().unwrap();
        for (edge, density) in outcome.edges.iter().zip(candidate.diagnostics.new_edges) {
            seen[usize::from(edge.branch == DefensiveBranch::Learned)] += 1;
            close(
                density.log_full.exp(),
                full_density(edge.proposed_relative_pose.unwrap(), 2., 0.35),
                2e-12,
            );
        }
    }
    assert!(seen.iter().all(|&n| n > 0));
    Ok(())
}

#[test]
fn relative_cube_support_floor_and_zero_reverse_density_are_explicit() -> Result<()> {
    let base = proposal()?;
    let mixed = DefensiveDimerProposal::new(&base, 50., 0.5)?;
    let remote = pose([40., 0., 0.], [0.1, 0.2, 0.3]);
    let d = mixed.density(remote)?;
    assert!(d.log_learned < -790.);
    close(d.log_full, 0.5_f64.ln() - 3. * 100_f64.ln(), 1e-13);
    let seam = Pose {
        position: [0.; 3],
        orientation: [0., 1., 0., 0.],
    };
    close(mixed.density(seam)?.log_full, d.log_full, 1e-13);
    let small = DefensiveDimerProposal::new(&base, 1., 0.5)?;
    let outside = pose([1.1, 0., 0.], [0.1, 0.2, 0.3]);
    let density = small.density(outside)?;
    assert_eq!(density.log_uniform, f64::NEG_INFINITY);
    close(density.log_full, 0.5_f64.ln() + density.log_learned, 1e-13);
    let pure = DefensiveDimerProposal::new(&base, 1., 1.)?;
    let old = tree_members(ID, [outside, ID])?;
    let mut rng = StdRng::seed_from_u64(2);
    let outcome = pure.propose(&mut rng, ID, old[0], old[1])?;
    let c = outcome.candidate.as_ref().unwrap();
    assert_eq!(c.diagnostics.log_reverse_forward, f64::NEG_INFINITY);
    assert!(outcome.null_reason.is_none());
    assert_eq!(
        serde_json::to_value(&outcome)?["candidate"]["diagnostics"]["log_reverse_forward"],
        "-inf"
    );
    assert!(pure.correction(ID, ID, ID, old[0], old[1]).is_err());
    Ok(())
}

#[test]
fn nonorigin_anchor_and_new_root_define_the_two_drawn_coordinates() -> Result<()> {
    let base = proposal()?;
    let defensive = DefensiveDimerProposal::new(&base, 2.5, 0.5)?;
    let anchor = pose([5., -8., 3.], [0.3, -0.4, 0.2]);
    let old = tree_members(
        anchor,
        [
            pose([1., 0.2, 0.3], [0.1, 0.2, 0.3]),
            pose([0.5, -0.1, 0.2], [0.2, 0.1, -0.1]),
        ],
    )?;
    let mut rng = StdRng::seed_from_u64(19);
    let outcome = defensive.propose(&mut rng, anchor, old[0], old[1])?;
    assert_eq!(outcome.spectator, anchor);
    let candidate = outcome.candidate.as_ref().unwrap();
    let coordinates = tree_coordinates(anchor, candidate.root, candidate.child)?;
    for (p, edge) in coordinates.into_iter().zip(&outcome.edges) {
        same(p, edge.proposed_relative_pose.unwrap());
    }
    let wrong_child = tree_members(
        old[0],
        [outcome.edges[1].proposed_relative_pose.unwrap(), ID],
    )?[0];
    assert!(norm(sub(wrong_child.position, candidate.child.position)) > 0.1);
    Ok(())
}

#[test]
fn draws_are_independent_of_old_state_and_fixed_seed_reproducible() -> Result<()> {
    let base = proposal()?;
    let defensive = DefensiveDimerProposal::new(&base, 2.5, 0.5)?;
    let old = tree_members(
        ID,
        [
            pose([1., 0.2, 0.3], [0.1, 0.2, 0.3]),
            pose([0.5, -0.1, 0.2], [0.2, 0.1, -0.1]),
        ],
    )?;
    let other = tree_members(
        ID,
        [
            pose([-1., 0.1, 0.2], [0.2, -0.1, 0.3]),
            pose([0.2, 1.5, 0.1], [0.1, 0.3, 0.1]),
        ],
    )?;
    let mut a = StdRng::seed_from_u64(128);
    let mut b = StdRng::seed_from_u64(128);
    let mut c = StdRng::seed_from_u64(128);
    for _ in 0..16 {
        let x = defensive.propose(&mut a, ID, old[0], old[1])?;
        let y = defensive.propose(&mut b, ID, old[0], old[1])?;
        let z = defensive.propose(&mut c, ID, other[0], other[1])?;
        assert_eq!(serde_json::to_value(&x)?, serde_json::to_value(&y)?);
        for (xe, ze) in x.edges.iter().zip(&z.edges) {
            assert_eq!(xe.trace, ze.trace);
            assert_eq!(xe.proposed_relative_pose, ze.proposed_relative_pose);
        }
        assert_eq!(
            x.candidate.as_ref().unwrap().root,
            z.candidate.as_ref().unwrap().root
        );
        assert_eq!(
            x.candidate.as_ref().unwrap().child,
            z.candidate.as_ref().unwrap().child
        );
    }
    Ok(())
}

#[test]
fn numerical_nulls_keep_both_raw_draws_without_retries() -> Result<()> {
    let broken = DockingProposal::new(
        model(false, true)?,
        DockingMethod::PosteriorInvolution,
        0.7,
        [0.; 3],
    )?;
    let defensive = DefensiveDimerProposal::new(&broken, 2.5, 0.)?;
    let mut actual = StdRng::seed_from_u64(33);
    let mut reference = StdRng::seed_from_u64(33);
    let result = defensive.propose(&mut actual, ID, ID, ID)?;
    assert!(result.candidate.is_none());
    assert!(result.null_reason.is_some());
    for edge in &result.edges {
        let coin = reference.random::<f64>();
        let (pose, mut trace) = broken.draw_singleton_independent(&mut reference, &[ID])?;
        trace["branch_uniform"] = json!(coin);
        assert_eq!(edge.trace, trace);
        assert_eq!(edge.proposed_relative_pose, pose);
        assert_eq!(edge.branch, DefensiveBranch::Learned);
        assert!(edge.null_reason.is_some());
    }
    assert_eq!(actual.random::<u64>(), reference.random::<u64>());
    Ok(())
}

#[test]
fn malformed_modes_parameters_and_old_poses_fail_before_random_draws() -> Result<()> {
    let base = proposal()?;
    for (l, a) in [
        (0., 0.5),
        (-1., 0.5),
        (f64::NAN, 0.5),
        (2., -0.1),
        (2., 1.1),
        (2., f64::NAN),
    ] {
        assert!(DefensiveDimerProposal::new(&base, l, a).is_err());
    }
    let periodic = DockingProposal::new(
        model(true, false)?,
        DockingMethod::PosteriorInvolution,
        0.7,
        [0.; 3],
    )?;
    assert!(DefensiveDimerProposal::new(&periodic, 2., 0.5).is_err());
    let incompatible =
        DockingProposal::new(model(false, false)?, DockingMethod::Mixture, 0.7, [0.; 3])?;
    assert!(DefensiveDimerProposal::new(&incompatible, 2., 0.5).is_err());
    let defensive = DefensiveDimerProposal::new(&base, 2.5, 0.5)?;
    let mut actual = StdRng::seed_from_u64(2);
    let mut expected = StdRng::seed_from_u64(2);
    let invalid = Pose {
        orientation: [0.; 4],
        ..ID
    };
    assert!(defensive.propose(&mut actual, invalid, ID, ID).is_err());
    assert_eq!(actual.random::<u64>(), expected.random::<u64>());
    Ok(())
}
