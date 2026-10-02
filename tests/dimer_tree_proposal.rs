//! Proposal-only controls. No physical gate, sampling campaign or native atlas.
use anyhow::Result;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde_json::json;
use tetramer_mc::{
    basin_involution::BasinTrace,
    dimer_tree_proposal::{DimerTreeProposal, DimerTreeTrace, tree_coordinates, tree_members},
    docking::{DockingMethod, DockingProposal},
    math::{
        Pose, add, cayley, invert_relative_pose, matmul, matvec, norm, quaternion, rotation, sub,
        transpose,
    },
    proposal::FrozenRelativePoseProposal,
};

const SHA: &str = "0000000000000000000000000000000000000000000000000000000000000000";

// The same two anisotropic, translation/rotation-correlated atlas fixture as
// tests/cluster_member_charts.rs and tests/oligomer_proposal.rs.
fn model(reciprocal: bool, periodic: bool) -> Result<FrozenRelativePoseProposal> {
    let covariances: Vec<[[f64; 6]; 6]> = [
        [0.5, 0.7, 0.4, 0.6, 0.9, 0.5],
        [0.9, 0.4, 0.6, 1.1, 0.5, 0.7],
    ]
    .iter()
    .map(|scales| {
        let mut l = [[0.; 6]; 6];
        for i in 0..6 {
            l[i][i] = scales[i];
        }
        l[3][0] = 0.2;
        l[5][1] = -0.15;
        std::array::from_fn(|i| std::array::from_fn(|j| (0..6).map(|a| l[i][a] * l[j][a]).sum()))
    })
    .collect();
    let base = json!({"coordinate_convention":"anchor-body-relative","shape_sha256":SHA,
        "angular_length":1.3,"weights":[0.3,0.7],
        "anchors":[{"position":[2.4,-0.5,0.3],"rotation":cayley([0.3,-0.1,0.4])},
                   {"position":[-0.6,2.2,1.0],"rotation":cayley([-0.2,0.3,0.1])}],
        "means":vec![[0.05,0.03,-0.04,0.1,-0.05,0.04];2],"covariances":covariances});
    let raw = if reciprocal {
        json!({"schema":"reciprocal-pose-mixture-v1","base_model":base,"reciprocal_components":[true,false]})
    } else {
        base
    };
    if periodic {
        FrozenRelativePoseProposal::from_json_str(&raw.to_string(), [80.; 3], 0.1, SHA)
    } else {
        FrozenRelativePoseProposal::from_json_str_open(&raw.to_string(), [80.; 3], 0.1, SHA)
    }
}

fn proposal(reciprocal: bool, correlation: f64) -> Result<DockingProposal> {
    DockingProposal::new(
        model(reciprocal, false)?,
        DockingMethod::PosteriorInvolution,
        correlation,
        [0.; 3],
    )
}

fn compose(a: Pose, b: Pose) -> Pose {
    Pose {
        position: add(a.position, matvec(rotation(a.orientation), b.position)),
        orientation: quaternion(matmul(rotation(a.orientation), rotation(b.orientation))),
    }
}

fn pose(t: [f64; 3], c: [f64; 3]) -> Pose {
    Pose {
        position: t,
        orientation: quaternion(cayley(c)),
    }
}

fn same_pose(a: Pose, b: Pose, tolerance: f64) {
    assert!(
        norm(sub(a.position, b.position)) < tolerance,
        "positions {:?} != {:?}",
        a,
        b
    );
    let ra = rotation(a.orientation);
    let rb = rotation(b.orientation);
    let error = ra
        .iter()
        .flatten()
        .zip(rb.iter().flatten())
        .map(|(a, b)| (a - b).abs())
        .fold(0_f64, f64::max);
    assert!(error < tolerance, "orientation error {error}");
}

fn close(a: f64, b: f64, tolerance: f64) {
    assert!(
        (a - b).abs() < tolerance,
        "{a} != {b}, difference {}",
        a - b
    );
}

fn in_chart(p: &DockingProposal, branch: usize, z: [f64; 6]) -> Pose {
    let (map, inverted, _) = p.member_chart_parts();
    let value = map.decode(branch, z).unwrap();
    if inverted[branch] {
        invert_relative_pose(value)
    } else {
        value
    }
}

fn scene(p: &DockingProposal, sources: [usize; 2]) -> (Pose, Pose, Pose) {
    let a = pose([3.1, -2.4, 1.7], [0.2, -0.15, 0.1]);
    let h0 = in_chart(p, sources[0], [0.1, -0.3, 0.2, 0.15, -0.2, 0.1]);
    let h1 = in_chart(p, sources[1], [-0.2, 0.1, 0.3, -0.1, 0.2, -0.15]);
    let root = compose(a, h0);
    let child = compose(root, h1);
    (a, root, child)
}

fn trace(sources: [usize; 2], targets: [usize; 2]) -> DimerTreeTrace {
    DimerTreeTrace {
        edges: [
            BasinTrace {
                source: sources[0],
                target: targets[0],
                noise: [0.3, -0.2, 0.5, 0.1, -0.1, 0.2],
            },
            BasinTrace {
                source: sources[1],
                target: targets[1],
                noise: [-0.1, 0.4, -0.3, 0.2, 0.3, -0.2],
            },
        ],
    }
}

#[test]
fn two_edge_inverse_and_product_density_corrections_all_labels() -> Result<()> {
    for reciprocal in [false, true] {
        for correlation in [0., 0.7, 1.] {
            let p = proposal(reciprocal, correlation)?;
            let tree = DimerTreeProposal::new(&p)?;
            let reference = model(reciprocal, false)?;
            let branches = if reciprocal { 3 } else { 2 };
            for a0 in 0..branches {
                for a1 in 0..branches {
                    let (a, root, child) = scene(&p, [a0, a1]);
                    for b0 in 0..branches {
                        for b1 in 0..branches {
                            let forward_trace = trace([a0, a1], [b0, b1]);
                            let forward = tree.apply(a, root, child, &forward_trace)?;
                            let candidate = forward
                                .candidate
                                .as_ref()
                                .expect("finite fixture forward map");
                            assert_eq!(forward.spectator, a);
                            let backward = tree.apply(
                                a,
                                candidate.root,
                                candidate.child,
                                &candidate.inverse_trace,
                            )?;
                            let returned = backward
                                .candidate
                                .as_ref()
                                .expect("finite fixture inverse map");
                            same_pose(returned.root, root, 2e-9);
                            same_pose(returned.child, child, 2e-9);
                            close(
                                candidate.diagnostics.log_reverse_forward
                                    + returned.diagnostics.log_reverse_forward,
                                0.,
                                2e-8,
                            );
                            let old = tree_coordinates(a, root, child)?;
                            let new = tree_coordinates(a, candidate.root, candidate.child)?;
                            let density = |coordinates: [Pose; 2]| -> Result<f64> {
                                Ok(reference.relative_log_density(
                                    coordinates[0].position,
                                    rotation(coordinates[0].orientation),
                                )? + reference.relative_log_density(
                                    coordinates[1].position,
                                    rotation(coordinates[1].orientation),
                                )?)
                            };
                            let d = &candidate.diagnostics;
                            close(d.full_old_log_density, density(old)?, 2e-10);
                            close(d.full_new_log_density, density(new)?, 2e-9);
                            close(d.log_reverse_forward, density(old)? - density(new)?, 2e-9);
                            close(d.expanded_log_reverse_forward, d.log_reverse_forward, 2e-9);
                            close(
                                d.log_extended_jacobian
                                    + d.log_auxiliary_ratio
                                    + d.label_log_reverse_forward,
                                d.log_reverse_forward,
                                2e-9,
                            );
                            assert_eq!(d.log_tree_coordinate_jacobian, 0.);
                            assert_eq!(d.selection_log_reverse_forward, 0.);
                            for k in 0..2 {
                                assert_eq!(
                                    forward.edges[k].trace.as_ref(),
                                    Some(&forward_trace.edges[k])
                                );
                                let recovered = &returned.inverse_trace.edges[k];
                                assert_eq!(
                                    (recovered.source, recovered.target),
                                    (forward_trace.edges[k].source, forward_trace.edges[k].target)
                                );
                                for j in 0..6 {
                                    close(
                                        recovered.noise[j],
                                        forward_trace.edges[k].noise[j],
                                        2e-9,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn root_internal_and_joint_changes_have_distinct_geometry() -> Result<()> {
    let p = proposal(true, 1.)?;
    let tree = DimerTreeProposal::new(&p)?;
    let (a, root, child) = scene(&p, [0, 2]);
    let old = tree_coordinates(a, root, child)?;
    let root_only = tree
        .apply(a, root, child, &trace([0, 2], [1, 2]))?
        .candidate
        .unwrap();
    let r = tree_coordinates(a, root_only.root, root_only.child)?;
    same_pose(r[1], old[1], 1e-10);
    assert!(norm(sub(root_only.root.position, root.position)) > 0.1);
    let internal = tree
        .apply(a, root, child, &trace([0, 2], [0, 1]))?
        .candidate
        .unwrap();
    assert_eq!(internal.root, root);
    let h = tree_coordinates(a, internal.root, internal.child)?;
    same_pose(h[0], old[0], 1e-10);
    assert!(norm(sub(h[1].position, old[1].position)) > 0.1);
    let joint = tree
        .apply(a, root, child, &trace([0, 2], [1, 1]))?
        .candidate
        .unwrap();
    let j = tree_coordinates(a, joint.root, joint.child)?;
    same_pose(j[0], r[0], 1e-10);
    same_pose(j[1], h[1], 1e-10);
    let identity = tree
        .apply(a, root, child, &trace([0, 2], [0, 2]))?
        .candidate
        .unwrap();
    assert_eq!(identity.root, root);
    assert_eq!(identity.child, child);
    assert_eq!(identity.diagnostics.log_reverse_forward, 0.);
    Ok(())
}

#[test]
fn world_frame_changes_do_not_change_edge_maps_or_corrections() -> Result<()> {
    let world = pose([-7.2, 3.3, 1.4], [-0.5, 0.2, 0.3]);
    for correlation in [0., 0.7, 1.] {
        let p = proposal(true, correlation)?;
        let tree = DimerTreeProposal::new(&p)?;
        let (a, root, child) = scene(&p, [1, 2]);
        let traces = trace([1, 2], [2, 0]);
        let first = tree.apply(a, root, child, &traces)?.candidate.unwrap();
        let changed = tree.apply(
            compose(world, a),
            compose(world, root),
            compose(world, child),
            &traces,
        )?;
        let second = changed.candidate.unwrap();
        same_pose(second.root, compose(world, first.root), 2e-9);
        same_pose(second.child, compose(world, first.child), 2e-9);
        close(
            first.diagnostics.log_reverse_forward,
            second.diagnostics.log_reverse_forward,
            2e-9,
        );
        assert_eq!(changed.spectator, compose(world, a));
    }
    Ok(())
}

#[test]
fn zero_correlation_is_independent_of_old_edge_coordinates_given_targets_and_noise() -> Result<()> {
    let p = proposal(true, 0.)?;
    let tree = DimerTreeProposal::new(&p)?;
    let (a, root, child) = scene(&p, [0, 2]);
    let t = trace([0, 2], [1, 0]);
    let first = tree.apply(a, root, child, &t)?.candidate.unwrap();
    let second_root = compose(a, in_chart(&p, 0, [1.5; 6]));
    let second_child = compose(second_root, in_chart(&p, 2, [-1.1; 6]));
    let second = tree
        .apply(a, second_root, second_child, &t)?
        .candidate
        .unwrap();
    same_pose(first.root, second.root, 1e-10);
    same_pose(first.child, second.child, 1e-10);
    assert!(
        (first.diagnostics.log_reverse_forward - second.diagnostics.log_reverse_forward).abs()
            > 1e-3
    );
    Ok(())
}

#[test]
fn fixed_seed_draws_keep_old_internal_coordinates_and_replay_exact_traces() -> Result<()> {
    let p = proposal(true, 0.7)?;
    let tree = DimerTreeProposal::new(&p)?;
    let (a, root, child) = scene(&p, [0, 2]);
    let old = tree_coordinates(a, root, child)?;
    let mut rng1 = StdRng::seed_from_u64(610020001);
    let mut rng2 = StdRng::seed_from_u64(610020001);
    for _ in 0..80 {
        let first = tree.propose(&mut rng1, a, root, child)?;
        let second = tree.propose(&mut rng2, a, root, child)?;
        assert_eq!(
            serde_json::to_value(&first)?,
            serde_json::to_value(&second)?
        );
        assert_eq!(first.edges[0].old_relative_pose, old[0]);
        assert_eq!(first.edges[1].old_relative_pose, old[1]);
        let trace = DimerTreeTrace {
            edges: std::array::from_fn(|k| first.edges[k].trace.clone().unwrap()),
        };
        let replay = tree.apply(a, root, child, &trace)?;
        assert_eq!(
            serde_json::to_value(&first)?,
            serde_json::to_value(&replay)?
        );
        let candidate = first.candidate.unwrap();
        assert!(
            norm(sub(
                tree_coordinates(a, candidate.root, child)?[1].position,
                old[1].position
            )) > 1e-5
        );
    }
    assert_eq!(rng1.random::<u64>(), rng2.random::<u64>());
    Ok(())
}

#[test]
fn invalid_maps_and_numerical_nulls_are_retained_without_retries() -> Result<()> {
    for method in [
        DockingMethod::Mixture,
        DockingMethod::Involution,
        DockingMethod::Local,
    ] {
        let p = DockingProposal::new(model(false, false)?, method, 0.7, [0.; 3])?;
        assert!(DimerTreeProposal::new(&p).is_err());
    }
    let periodic = DockingProposal::new(
        model(true, true)?,
        DockingMethod::PosteriorInvolution,
        0.7,
        [0.; 3],
    )?;
    assert!(DimerTreeProposal::new(&periodic).is_err());
    let p = proposal(true, 0.7)?;
    let tree = DimerTreeProposal::new(&p)?;
    let (a, root, child) = scene(&p, [0, 2]);
    let mut t = trace([0, 2], [1, 0]);
    t.edges[0].noise = [1e308; 6];
    let null = tree.apply(a, root, child, &t)?;
    assert!(null.candidate.is_none() && null.null_reason.is_some());
    assert!(null.edges[0].step.is_none() && null.edges[0].null_reason.is_some());
    assert!(null.edges[1].step.is_some());
    assert_eq!(null.edges[0].trace.as_ref(), Some(&t.edges[0]));
    assert_eq!(null.edges[1].trace.as_ref(), Some(&t.edges[1]));
    t.edges[0].noise = [0.; 6];
    t.edges[0].source = usize::MAX;
    assert!(tree.apply(a, root, child, &t)?.candidate.is_none());
    let huge = pose([1e200, 0., 0.], [0.; 3]);
    let mut rng = StdRng::seed_from_u64(7);
    let null = tree.propose(&mut rng, pose([0.; 3], [0.; 3]), huge, huge)?;
    assert!(null.candidate.is_none());
    assert!(null.edges[0].trace.is_none());
    assert!(null.edges[1].trace.is_some() && null.edges[1].step.is_some());
    let mut malformed = a;
    malformed.orientation = [0.; 4];
    assert!(
        tree.apply(malformed, root, child, &trace([0, 2], [1, 0]))
            .is_err()
    );
    Ok(())
}

// Independent infinitesimal physical coordinates: Cartesian translations plus
// left rotation-vector increments. Their local Haar density is constant at 0.
fn perturb(mut p: Pose, index: usize, delta: f64) -> Pose {
    if index < 3 {
        p.position[index] += delta;
    } else {
        let axis = index - 3;
        let (s, c) = delta.sin_cos();
        let mut r = [[0.; 3]; 3];
        r[axis][axis] = 1.;
        let i = (axis + 1) % 3;
        let j = (axis + 2) % 3;
        r[i][i] = c;
        r[j][j] = c;
        r[i][j] = -s;
        r[j][i] = s;
        p.orientation = quaternion(matmul(r, rotation(p.orientation)));
    }
    p
}
fn local_delta(p: Pose, base: Pose) -> [f64; 6] {
    let t = sub(p.position, base.position);
    let r = matmul(
        rotation(p.orientation),
        transpose(rotation(base.orientation)),
    );
    [
        t[0],
        t[1],
        t[2],
        0.5 * (r[2][1] - r[1][2]),
        0.5 * (r[0][2] - r[2][0]),
        0.5 * (r[1][0] - r[0][1]),
    ]
}
fn determinant(mut a: [[f64; 12]; 12]) -> f64 {
    let mut result = 1.;
    for k in 0..12 {
        let pivot = (k..12)
            .max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))
            .unwrap();
        assert!(a[pivot][k].abs() > 1e-12);
        if pivot != k {
            a.swap(k, pivot);
            result = -result;
        }
        result *= a[k][k];
        for i in k + 1..12 {
            let factor = a[i][k] / a[k][k];
            for j in k + 1..12 {
                a[i][j] -= factor * a[k][j];
            }
        }
    }
    result
}
fn numerical_volume(input: [Pose; 2], map: impl Fn([Pose; 2]) -> [Pose; 2]) -> f64 {
    let baseline = map(input);
    let epsilon = 2e-5;
    let mut jacobian = [[0.; 12]; 12];
    for k in 0..12 {
        let mut plus = input;
        let mut minus = input;
        plus[k / 6] = perturb(input[k / 6], k % 6, epsilon);
        minus[k / 6] = perturb(input[k / 6], k % 6, -epsilon);
        let p = map(plus);
        let m = map(minus);
        for b in 0..2 {
            let pd = local_delta(p[b], baseline[b]);
            let md = local_delta(m[b], baseline[b]);
            for j in 0..6 {
                jacobian[6 * b + j][k] = (pd[j] - md[j]) / (2. * epsilon);
            }
        }
    }
    determinant(jacobian)
}

#[test]
fn independent_twelve_dimensional_physical_volume_jacobian_is_one() -> Result<()> {
    let p = proposal(true, 0.7)?;
    for source in [[0, 2], [1, 0], [2, 1]] {
        let (a, root, child) = scene(&p, source);
        let coordinates = tree_coordinates(a, root, child)?;
        let encoded = numerical_volume([root, child], |g| tree_coordinates(a, g[0], g[1]).unwrap());
        let decoded = numerical_volume(coordinates, |h| tree_members(a, h).unwrap());
        close(encoded, 1., 2e-7);
        close(decoded, 1., 2e-7);
        close(encoded * decoded, 1., 3e-7);
    }
    Ok(())
}
