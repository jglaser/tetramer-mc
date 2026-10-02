//! The scored law must use the coefficients that actually generated the pose.
//! No physical sampling: fixed anisotropic witnesses and bounded RNG replay.
use anyhow::Result;
use rand::{SeedableRng, rngs::StdRng};
use serde_json::json;
use tetramer_mc::{
    docking::{DockingMethod, DockingProposal, MemberLabel},
    math::*,
    proposal::FrozenRelativePoseProposal,
};

const SHA: &str = "0000000000000000000000000000000000000000000000000000000000000000";
const IDENTITY: Pose = Pose {
    position: [0.; 3],
    orientation: [1., 0., 0., 0.],
};

fn model() -> Result<FrozenRelativePoseProposal> {
    // A narrow coupled translation direction gives condition ~1e10. The
    // covariance is explicit and fixed, not fitted/searched against a result.
    let mut lower = [[0.; 6]; 6];
    for (i, diagonal) in [3., 2., 0.0001, 0.5, 0.6, 0.7].into_iter().enumerate() {
        lower[i][i] = diagonal;
    }
    lower[1][0] = 1.3;
    lower[2][0] = 2.2;
    lower[2][1] = 1.1;
    lower[3][1] = 0.12;
    lower[4][0] = -0.17;
    lower[5][3] = 0.14;
    let covariance: [[f64; 6]; 6] = std::array::from_fn(|i| {
        std::array::from_fn(|j| (0..6).map(|k| lower[i][k] * lower[j][k]).sum())
    });
    let raw = json!({"schema":"reciprocal-pose-mixture-v1",
        "base_model":{"coordinate_convention":"anchor-body-relative","shape_sha256":SHA,
            "angular_length":1.7,"weights":[0.37,0.63],
            "anchors":[{"position":[0.,0.,0.],"rotation":rotation(IDENTITY.orientation)},
                       {"position":[100.,-80.,60.],"rotation":rotation(IDENTITY.orientation)}],
            "means":vec![[0.;6];2],"covariances":[covariance,covariance]},
        "reciprocal_components":[true,true]});
    FrozenRelativePoseProposal::from_json_str_open(&raw.to_string(), [400.; 3], 0.1, SHA)
}

fn relative(anchor: Pose, pose: Pose) -> Pose {
    let inverse = transpose(rotation(anchor.orientation));
    Pose {
        position: matvec(inverse, sub(pose.position, anchor.position)),
        orientation: quaternion(matmul(inverse, rotation(pose.orientation))),
    }
}

fn log_sum(values: &[f64]) -> f64 {
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if maximum == f64::NEG_INFINITY {
        maximum
    } else {
        maximum + values.iter().map(|x| (x - maximum).exp()).sum::<f64>().ln()
    }
}

fn map_logs(proposal: &DockingProposal, pose: Pose, anchor: Pose) -> Vec<f64> {
    let (map, inverted, weights) = proposal.member_chart_parts();
    let p = relative(anchor, pose);
    inverted
        .iter()
        .enumerate()
        .map(|(i, inverse)| {
            weights[i] + map.log_density(i, if *inverse { invert_relative_pose(p) } else { p })
        })
        .collect()
}

fn near_pose(a: Pose, b: Pose) {
    assert!(norm(sub(a.position, b.position)) < 2e-10);
    assert!(
        rotation(a.orientation)
            .iter()
            .flatten()
            .zip(rotation(b.orientation).iter().flatten())
            .all(|(a, b)| (a - b).abs() < 2e-12)
    );
}

#[test]
fn anisotropic_branch_and_member_scores_use_actual_map_factor_not_refactor() -> Result<()> {
    let original = model()?;
    let parameters = original.component_parameters();
    let proposal =
        DockingProposal::new(original, DockingMethod::PosteriorInvolution, 0.7, [0.; 3])?;
    let (map, inversions, weights) = proposal.member_chart_parts();
    let witness = map.decode(0, [0.7, -0.4, 2., 0.1, -0.2, 0.3])?;
    let mut legacy = parameters[0].clone();
    legacy.weight = 1.;
    let legacy = FrozenRelativePoseProposal::from_components_open(
        vec![legacy],
        1.7,
        [400.; 3],
        0.1,
        SHA,
        SHA,
    )?;
    let mapped = map.log_density(0, relative(IDENTITY, witness));
    let refactored =
        legacy.relative_log_density(witness.position, rotation(witness.orientation))?;
    assert!(
        (mapped - refactored).abs() > 1e-9,
        "fixture must expose old duplicate scorer"
    );
    for branch in 0..4 {
        let mut p = map.decode(branch, [0.7, -0.4, 2., 0.1, -0.2, 0.3])?;
        if inversions[branch] {
            p = invert_relative_pose(p);
        }
        let logs = proposal.branch_log_densities(p, IDENTITY)?;
        assert_eq!(logs, map_logs(&proposal, p, IDENTITY));
        assert_eq!(
            proposal.members_log_density(&[p], &[IDENTITY])?,
            log_sum(&logs)
        );
        let back = if inversions[branch] {
            invert_relative_pose(relative(IDENTITY, p))
        } else {
            relative(IDENTITY, p)
        };
        assert_eq!(
            logs[branch],
            weights[branch] + map.log_density(branch, back)
        );
    }
    // Complete mixture over all physical member/anchor labels, not just the
    // selected draw's branch. Reciprocal wrappers retain unit Jacobian.
    let members = [
        witness,
        Pose {
            position: [1., -2., 0.5],
            ..IDENTITY
        },
    ];
    let anchors = [
        IDENTITY,
        Pose {
            position: [-0.2, 0.7, 0.1],
            ..IDENTITY
        },
    ];
    let mut joint = Vec::new();
    for member in members {
        for anchor in anchors {
            joint.extend(
                map_logs(&proposal, member, anchor)
                    .into_iter()
                    .map(|x| x - 4_f64.ln()),
            );
        }
    }
    assert_eq!(
        proposal.members_log_density(&members, &anchors)?,
        log_sum(&joint)
    );
    Ok(())
}

#[test]
fn target_only_draw_and_singleton_posterior_full_scores_share_map_law() -> Result<()> {
    for method in [
        DockingMethod::PosteriorInvolution,
        DockingMethod::Involution,
    ] {
        let proposal = DockingProposal::new(model()?, method, 0.7, [0.; 3])?;
        let old = proposal
            .member_chart_parts()
            .0
            .decode(0, [0.7, -0.4, 2., 0.1, -0.2, 0.3])?;
        let mut rng = StdRng::seed_from_u64(6100203111);
        let mut mapped = 0;
        for _ in 0..24 {
            let (_, info) = proposal.propose(&mut rng, old, &[IDENTITY])?;
            if info["branch"] == "uniform" {
                continue;
            }
            assert!(info.get("null_reason").is_none());
            let step: Pose = serde_json::from_value(info["step"]["pose"].clone())?;
            let expected_old = log_sum(&map_logs(&proposal, old, IDENTITY));
            // The full singleton logs are evaluated at the local map endpoint.
            let (map, inverted, weights) = proposal.member_chart_parts();
            let expected_new = log_sum(
                &(0..weights.len())
                    .map(|b| {
                        weights[b]
                            + map.log_density(
                                b,
                                if inverted[b] {
                                    invert_relative_pose(step)
                                } else {
                                    step
                                },
                            )
                    })
                    .collect::<Vec<_>>(),
            );
            assert_eq!(
                info["full_old_gaussian_log_density"].as_f64().unwrap(),
                expected_old
            );
            assert_eq!(
                info["full_new_gaussian_log_density"].as_f64().unwrap(),
                expected_new
            );
            if method == DockingMethod::PosteriorInvolution {
                assert_eq!(
                    info["log_reverse_forward"].as_f64().unwrap(),
                    expected_old - expected_new
                );
                let error = (info["expanded_log_reverse_forward"].as_f64().unwrap()
                    - (expected_old - expected_new))
                    .abs();
                assert!(error < 2e-8, "expanded posterior cancellation: {error}");
            }
            mapped += 1;
        }
        assert!(mapped > 0);
    }
    let proposal =
        DockingProposal::new(model()?, DockingMethod::PosteriorInvolution, 0.7, [0.; 3])?;
    let mut rng = StdRng::seed_from_u64(6100203112);
    for _ in 0..24 {
        let (candidate, trace) = proposal.draw_singleton_independent(&mut rng, &[IDENTITY])?;
        let candidate = candidate.expect("finite fixed fixture");
        let branch = trace["target_label"]["branch"].as_u64().unwrap() as usize;
        let latent: [f64; 6] = serde_json::from_value(trace["target_latent"].clone())?;
        let (map, inverted, _) = proposal.member_chart_parts();
        let generated = map.decode(branch, latent)?;
        near_pose(
            candidate,
            if inverted[branch] {
                invert_relative_pose(generated)
            } else {
                generated
            },
        );
        assert_eq!(
            proposal.members_log_density(&[candidate], &[IDENTITY])?,
            log_sum(&map_logs(&proposal, candidate, IDENTITY))
        );
    }
    Ok(())
}

#[test]
fn exact_seam_is_zero_density_and_member_trace_keeps_complete_correction() -> Result<()> {
    let proposal =
        DockingProposal::new(model()?, DockingMethod::PosteriorInvolution, 0.7, [0.; 3])?;
    let seam = Pose {
        orientation: [0., 1., 0., 0.],
        ..IDENTITY
    };
    assert!(
        proposal
            .branch_log_densities(seam, IDENTITY)?
            .iter()
            .all(|x| *x == f64::NEG_INFINITY)
    );
    assert_eq!(
        proposal.members_log_density(&[seam], &[IDENTITY])?,
        f64::NEG_INFINITY
    );
    let old = proposal
        .member_chart_parts()
        .0
        .decode(0, [0.7, -0.4, 2., 0.1, -0.2, 0.3])?;
    let source = MemberLabel {
        member: 0,
        anchor: 0,
        branch: 0,
    };
    let target = MemberLabel {
        member: 0,
        anchor: 0,
        branch: 1,
    };
    let step = proposal.apply_member_trace(
        &[old],
        0,
        &[IDENTITY],
        source,
        target,
        [0.2, -0.1, 0.3, 0.4, -0.5, 0.6],
    )?;
    let expected = proposal.members_log_density(&[old], &[IDENTITY])?
        - proposal.members_log_density(&[step.handle], &[IDENTITY])?;
    assert_eq!(step.log_reverse_forward, expected);
    assert!((step.expanded_log_reverse_forward - expected).abs() < 2e-8);
    let backward = proposal.apply_relative_trace(step.step.pose, &step.step.inverse_trace)?;
    near_pose(old, backward.pose);
    Ok(())
}

#[test]
fn direct_mixture_control_keeps_its_own_consistent_draw_score_and_rng() -> Result<()> {
    let original = model()?;
    let proposal = DockingProposal::new(original.clone(), DockingMethod::Mixture, 0.7, [0.; 3])?;
    let old = Pose {
        position: [0.3, -0.2, 0.1],
        ..IDENTITY
    };
    let mut direct_rng = StdRng::seed_from_u64(6100203113);
    let mut wrapped_rng = StdRng::seed_from_u64(6100203113);
    for _ in 0..24 {
        let direct = original.propose(&mut direct_rng, &[old, IDENTITY], 0)?;
        let (candidate, info) = proposal.propose(&mut wrapped_rng, old, &[IDENTITY])?;
        assert_eq!(candidate, direct.candidate);
        let mut expected = serde_json::to_value(direct)?;
        expected["anchor_index"] = json!(0);
        assert_eq!(info, expected);
    }
    Ok(())
}
