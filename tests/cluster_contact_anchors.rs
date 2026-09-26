//! State-dependent primary-anchor selection: support, binding/detachment,
//! endpoint flow, and composition with the implemented member-chart involution.
use anyhow::Result;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde_json::json;
use tetramer_mc::{
    cluster_phase::{ContactGraph, anchor_pool},
    docking::{DockingMethod, DockingProposal, MemberLabel},
    geometry::{Atom, Shape, SphereTree},
    math::{IDENTITY, Pose, norm, sub},
    proposal::FrozenRelativePoseProposal,
    rigid_subset::transport_members,
};

fn graph(n: usize, edges: &[(usize, usize)]) -> ContactGraph {
    let mut adjacency = vec![vec![false; n]; n];
    for &(i, j) in edges {
        adjacency[i][j] = true;
        adjacency[j][i] = true;
    }
    ContactGraph { adjacency }
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-12, "{a} != {b}");
}

#[test]
fn anchor_law_normalizes_preserves_support_and_counts_member_contacts() -> Result<()> {
    let g = graph(6, &[(0, 1), (0, 2), (1, 2), (1, 3), (4, 5)]);
    for eps in [1e-8, 0.2, 1.] {
        let q = g.anchor_probabilities(&[0, 1], eps)?;
        assert_eq!(q.len(), 6);
        close(q.iter().sum(), 1.);
        assert_eq!(q[0], 0.);
        assert_eq!(q[1], 0.);
        for &x in &q[2..] {
            assert!(x >= eps / 4. && x > 0.);
        }
        // Two contacts to the same spectator carry twice the contact weight;
        // spectator-spectator contacts do not enter the normalization.
        close(q[2], eps / 4. + (1. - eps) * 2. / 3.);
        close(q[3], eps / 4. + (1. - eps) / 3.);
        close(q[4], eps / 4.);
        close(q[5], eps / 4.);
        assert_eq!(q, g.anchor_probabilities(&[1, 0], eps)?);
    }
    let none = graph(6, &[(0, 1), (2, 3), (3, 4)]);
    assert_eq!(
        none.anchor_probabilities(&[0, 1], 0.2)?,
        vec![0., 0., 0.25, 0.25, 0.25, 0.25]
    );
    for eps in [0., -0.1, 1.01, f64::NAN, f64::INFINITY] {
        assert!(g.anchor_probabilities(&[0, 1], eps).is_err());
    }
    for ids in [vec![], vec![0, 0], vec![7]] {
        assert!(g.anchor_probabilities(&ids, 0.2).is_err());
    }
    Ok(())
}

#[test]
fn binding_detachment_and_anchor_loss_use_complete_reverse_denominator() -> Result<()> {
    let eps = 0.2;
    let unbound = graph(5, &[(0, 1)]).anchor_probabilities(&[0, 1], eps)?;
    let bound = graph(5, &[(0, 1), (0, 2), (1, 2)]).anchor_probabilities(&[0, 1], eps)?;
    close(unbound[2], 1. / 3.);
    close(bound[2], eps / 3. + 1. - eps);
    let bind = bound[2].ln() - unbound[2].ln();
    let detach = unbound[2].ln() - bound[2].ln();
    assert!(bind > 0. && detach < 0.);
    close(bind + detach, 0.);
    // The selected anchor keeps exactly one contact, but a newly contacting
    // spectator changes C. Looking only at that anchor would miss this factor.
    let x = graph(5, &[(0, 1), (0, 2)]).anchor_probabilities(&[0, 1], eps)?;
    let y = graph(5, &[(0, 1), (0, 2), (1, 3)]).anchor_probabilities(&[0, 1], eps)?;
    close(y[2], eps / 3. + (1. - eps) / 2.);
    assert!(y[2].ln() - x[2].ln() < -0.5);
    // Losing the selected anchor while another interface remains still has
    // nonzero reverse support through the defensive probability.
    let z = graph(5, &[(0, 1), (1, 3)]).anchor_probabilities(&[0, 1], eps)?;
    close(z[2], eps / 3.);
    assert!((z[2].ln() - y[2].ln()).is_finite());
    assert!(z[2] / y[2] < 0.15);
    Ok(())
}

#[test]
fn exact_finite_state_flow_exposes_omitted_anchor_correction() -> Result<()> {
    // Three physical configurations; each spectator label induces its own
    // deterministic involution. q changes as the subset binds and exchanges.
    let states = [
        graph(5, &[(0, 1)]),
        graph(5, &[(0, 1), (0, 2), (1, 2)]),
        graph(5, &[(0, 1), (0, 3), (1, 4)]),
    ];
    let q: Vec<_> = states
        .iter()
        .map(|g| g.anchor_probabilities(&[0, 1], 0.2))
        .collect::<Result<_>>()?;
    let pi: [f64; 3] = [0.2, 0.5, 0.3];
    let endpoints = [[1, 0, 2], [0, 2, 1], [2, 1, 0]];
    let kernel = |correct: bool| {
        let mut p = [[0.; 3]; 3];
        for x in 0..3 {
            for a in 0..3 {
                let y = endpoints[a][x];
                let ratio = pi[y] / pi[x]
                    * if correct {
                        q[y][a + 2] / q[x][a + 2]
                    } else {
                        1.
                    };
                let accept = ratio.min(1.);
                p[x][y] += q[x][a + 2] * accept;
                p[x][x] += q[x][a + 2] * (1. - accept);
            }
        }
        p
    };
    let p = kernel(true);
    for x in 0..3 {
        close(p[x].iter().sum(), 1.);
        for y in 0..3 {
            close(pi[x] * p[x][y], pi[y] * p[y][x]);
        }
    }
    let wrong = kernel(false);
    let residual: f64 = (0..3)
        .map(|y| ((0..3).map(|x| pi[x] * wrong[x][y]).sum::<f64>() - pi[y]).abs())
        .sum();
    assert!(
        residual > 0.1,
        "control must expose omitted correction, residual={residual}"
    );
    // Rejection/residence probabilities remain in P; accepted-only samples
    // are not being substituted for the equilibrium state chain.
    assert!(p.iter().enumerate().any(|(i, row)| row[i] > 0.3));
    Ok(())
}

#[test]
fn contact_selection_composes_with_member_map_and_exact_inverse() -> Result<()> {
    let sha = "0000000000000000000000000000000000000000000000000000000000000000";
    let covariance = |scale: f64| -> [[f64; 6]; 6] {
        std::array::from_fn(|i| std::array::from_fn(|j| if i == j { scale } else { 0. }))
    };
    let frozen = FrozenRelativePoseProposal::from_json_str_open(&json!({
        "coordinate_convention":"anchor-body-relative","shape_sha256":sha,
        "angular_length":1.,"weights":[0.35,0.65],
        "anchors":[{"position":[2.1,0.,0.],"rotation":IDENTITY},{"position":[0.,3.2,0.],"rotation":IDENTITY}],
        "means":vec![[0.;6];2],"covariances":[covariance(0.4),covariance(0.9)]
    }).to_string(), [40.;3], 0.1, sha)?;
    let map = DockingProposal::new(frozen, DockingMethod::PosteriorInvolution, 0.6, [0.; 3])?;
    let pose = |position| Pose {
        position,
        orientation: [1., 0., 0., 0.],
    };
    let x = vec![
        pose([0., 0., 0.]),
        pose([2., 0., 0.]),
        pose([0., 2.2, 0.]),
        pose([4., 0., 0.]),
        pose([0., 0., 6.]),
    ];
    let tree = SphereTree::new(Shape {
        name: "exclusion balls".into(),
        volume: 0.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius: 1.5,
        }],
    })?;
    let members = [0, 1];
    let gx = ContactGraph::build(&tree, &x);
    let qx = gx.anchor_probabilities(&members, 0.2)?;
    let mut rng = StdRng::seed_from_u64(942417);
    let mut changed = 0;
    for _ in 0..180 {
        let primary = rng.random_range(2..5);
        let pool = anchor_pool(&x, &members, primary, 2)?;
        let anchors: Vec<_> = pool.iter().map(|&i| x[i]).collect();
        let label = |rng: &mut StdRng| MemberLabel {
            member: rng.random_range(0..2),
            anchor: rng.random_range(0..2),
            branch: rng.random_range(0..2),
        };
        let (source, target) = (label(&mut rng), label(&mut rng));
        let noise = std::array::from_fn(|_| StandardNormal.sample(&mut rng));
        let step = map.apply_member_trace(&x[..2], 0, &anchors, source, target, noise)?;
        let carried = transport_members(&x, &members, 0, step.handle)?;
        let mut y = x.clone();
        for (i, p) in carried.iter().enumerate() {
            y[i] = *p;
        }
        assert_eq!(anchor_pool(&y, &members, primary, 2)?, pool);
        let qy = ContactGraph::build(&tree, &y).anchor_probabilities(&members, 0.2)?;
        let c = qy[primary].ln() - qx[primary].ln();
        changed += usize::from(c.abs() > 1e-8);
        let reverse = map.apply_member_trace(
            &carried,
            0,
            &anchors,
            target,
            source,
            step.step.inverse_trace.noise,
        )?;
        assert!(norm(sub(reverse.handle.position, x[0].position)) < 1e-8);
        let combined_forward = step.log_reverse_forward + c;
        let combined_reverse = reverse.log_reverse_forward - c;
        assert!((combined_forward + combined_reverse).abs() < 1e-8);
        let expanded = step.expanded_log_reverse_forward + c;
        assert!((expanded - combined_forward).abs() < 1e-8);
    }
    assert!(
        changed > 50,
        "must exercise state-dependent contact selection"
    );
    Ok(())
}
