//! Member-chart transport for rigid subsets: exact inverse, correction algebra,
//! handle independence, spectator-only anchor pools, and exact preservation of
//! the member-chart mixture G_S by the correlated map.
use anyhow::Result;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde_json::json;
use tetramer_mc::{
    cluster_phase::anchor_pool,
    docking::{DockingMethod, DockingProposal, MemberLabel},
    math::{Pose, cayley, norm, sub},
    proposal::FrozenRelativePoseProposal,
    rigid_subset::transport_members,
};

const SHA: &str = "0000000000000000000000000000000000000000000000000000000000000000";

fn model(reciprocal: bool) -> Result<FrozenRelativePoseProposal> {
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
    FrozenRelativePoseProposal::from_json_str_open(&raw.to_string(), [60.; 3], 0.1, SHA)
}
fn proposal(reciprocal: bool, correlation: f64) -> Result<DockingProposal> {
    DockingProposal::new(
        model(reciprocal)?,
        DockingMethod::PosteriorInvolution,
        correlation,
        [0.; 3],
    )
}
fn random_pose(rng: &mut StdRng, spread: f64) -> Pose {
    let q: [f64; 4] = std::array::from_fn(|_| StandardNormal.sample(rng));
    let n = q.iter().map(|x| x * x).sum::<f64>().sqrt();
    Pose {
        position: std::array::from_fn(|_| rng.random_range(-spread..spread)),
        orientation: q.map(|x| x / n),
    }
}
fn subset(rng: &mut StdRng) -> Vec<Pose> {
    (0..3).map(|_| random_pose(rng, 2.5)).collect()
}
fn pool(rng: &mut StdRng) -> Vec<Pose> {
    (0..3).map(|_| random_pose(rng, 4.)).collect()
}
fn label(rng: &mut StdRng, branches: usize) -> MemberLabel {
    MemberLabel {
        member: rng.random_range(0..3),
        anchor: rng.random_range(0..3),
        branch: rng.random_range(0..branches),
    }
}
fn noise(rng: &mut StdRng) -> [f64; 6] {
    std::array::from_fn(|_| StandardNormal.sample(rng))
}
fn state(members: &[Pose]) -> (Vec<Pose>, Vec<usize>) {
    (members.to_vec(), (0..members.len()).collect())
}
fn same_pose(a: Pose, b: Pose, tol: f64) -> bool {
    norm(sub(a.position, b.position)) < tol
        && (a
            .orientation
            .iter()
            .zip(&b.orientation)
            .map(|(x, y)| x * y)
            .sum::<f64>()
            .abs()
            - 1.)
            .abs()
            < tol
}

#[test]
fn member_map_inverts_and_correction_matches_expanded_trace() -> Result<()> {
    for reciprocal in [false, true] {
        let p = proposal(reciprocal, 0.6)?;
        let branches = if reciprocal { 3 } else { 2 };
        let mut rng = StdRng::seed_from_u64(71 + u64::from(reciprocal));
        let mut checked = 0;
        for _ in 0..400 {
            let members = subset(&mut rng);
            let anchors = pool(&mut rng);
            let handle = rng.random_range(0..3);
            let (source, target) = (label(&mut rng, branches), label(&mut rng, branches));
            let Ok(forward) =
                p.apply_member_trace(&members, handle, &anchors, source, target, noise(&mut rng))
            else {
                continue;
            };
            let (s, m) = state(&members);
            let moved = transport_members(&s, &m, handle, forward.handle)?;
            let direct = p.members_log_density(&members, &anchors)?
                - p.members_log_density(&moved, &anchors)?;
            assert!((forward.log_reverse_forward - direct).abs() < 1e-9);
            assert!(
                (forward.expanded_log_reverse_forward - forward.log_reverse_forward).abs()
                    // Arbitrary (tail) source labels can make the chart and
                    // label terms huge and opposite; allow cancellation error.
                    < 1e-9 + 1e-12 * forward.step.log_correction.abs(),
                "label/Jacobian expansion {} != {} (old {}, new {}, chart {})",
                forward.expanded_log_reverse_forward,
                forward.log_reverse_forward,
                forward.full_old_member_log_density,
                forward.full_new_member_log_density,
                forward.step.log_correction
            );
            let reverse = p.apply_member_trace(
                &moved,
                handle,
                &anchors,
                target,
                source,
                forward.step.inverse_trace.noise,
            )?;
            assert!(same_pose(reverse.handle, members[handle], 1e-8));
            assert!((reverse.log_reverse_forward + forward.log_reverse_forward).abs() < 1e-8);
            checked += 1;
        }
        assert!(checked > 350);
    }
    Ok(())
}

#[test]
fn member_map_endpoint_does_not_depend_on_handle() -> Result<()> {
    let p = proposal(true, 0.4)?;
    let mut rng = StdRng::seed_from_u64(9);
    for _ in 0..100 {
        let members = subset(&mut rng);
        let anchors = pool(&mut rng);
        let (source, target, n) = (label(&mut rng, 3), label(&mut rng, 3), noise(&mut rng));
        let (s, m) = state(&members);
        let endpoints: Vec<Vec<Pose>> = (0..3)
            .map(|h| {
                let step = p.apply_member_trace(&members, h, &anchors, source, target, n)?;
                transport_members(&s, &m, h, step.handle)
            })
            .collect::<Result<_>>()?;
        for e in &endpoints[1..] {
            for (a, b) in e.iter().zip(&endpoints[0]) {
                assert!(same_pose(*a, *b, 1e-9));
            }
        }
    }
    Ok(())
}

#[test]
fn anchor_pool_uses_spectators_only() -> Result<()> {
    let mut rng = StdRng::seed_from_u64(3);
    let mut state: Vec<_> = (0..9).map(|_| random_pose(&mut rng, 6.)).collect();
    let members = [2, 5, 7];
    let pool = anchor_pool(&state, &members, 0, 4)?;
    assert_eq!(pool.len(), 4);
    assert_eq!(pool[0], 0);
    assert!(pool.iter().all(|i| !members.contains(i)));
    let moved = transport_members(&state, &members, 5, random_pose(&mut rng, 6.))?;
    for (&i, p) in members.iter().zip(moved) {
        state[i] = p;
    }
    assert_eq!(anchor_pool(&state, &members, 0, 4)?, pool);
    assert_eq!(anchor_pool(&state, &members, 0, 50)?.len(), 6);
    assert!(anchor_pool(&state, &members, 2, 1).is_err());
    Ok(())
}

/// With c = 0 the destination latent is pure noise, so a draw with labels
/// from the destination law is an exact G_S sample. The correlated map must
/// then return G_S samples when every trial is accepted with
/// min(1, G_S(y) q(y,x) / G_S(x) q(x,y)) = min(1, G_S(y)/G_S(x) e^corr).
#[test]
fn correlated_member_map_preserves_member_chart_mixture() -> Result<()> {
    let fresh = proposal(true, 0.)?;
    let map = proposal(true, 0.7)?;
    let mut rng = StdRng::seed_from_u64(20260926);
    let anchors = pool(&mut rng);
    let template = subset(&mut rng);
    let weights = [0.15, 0.15, 0.7];
    let draw_branch = |rng: &mut StdRng| {
        let u: f64 = rng.random();
        if u < weights[0] {
            0
        } else if u < weights[0] + weights[1] {
            1
        } else {
            2
        }
    };
    let observable = |members: &[Pose]| -> [f64; 5] {
        let c: [f64; 3] =
            std::array::from_fn(|k| members.iter().map(|p| p.position[k]).sum::<f64>() / 3.);
        [
            c[0],
            c[1],
            c[2],
            c[0] * c[0] + c[1] * c[1] + c[2] * c[2],
            members[0].orientation[0].powi(2),
        ]
    };
    let (s, m) = state(&template);
    let mut before = [(0., 0.); 5];
    let mut after = [(0., 0.); 5];
    let mut accepted = 0;
    let n = 40_000;
    let mut used = 0;
    for _ in 0..n {
        let start = MemberLabel {
            member: rng.random_range(0..3),
            anchor: rng.random_range(0..3),
            branch: draw_branch(&mut rng),
        };
        let pick = MemberLabel {
            member: rng.random_range(0..3),
            anchor: rng.random_range(0..3),
            branch: draw_branch(&mut rng),
        };
        let Ok(x) = fresh.apply_member_trace(&template, 0, &anchors, start, pick, noise(&mut rng))
        else {
            continue;
        };
        let xs = transport_members(&s, &m, 0, x.handle)?;
        // Posterior source label at x, independent destination label and noise.
        let source = posterior_label(&map, &xs, &anchors, &mut rng)?;
        let target = MemberLabel {
            member: rng.random_range(0..3),
            anchor: rng.random_range(0..3),
            branch: draw_branch(&mut rng),
        };
        let y = match map.apply_member_trace(&xs, 0, &anchors, source, target, noise(&mut rng)) {
            Ok(step) => {
                let ys = transport_members(&s, &m, 0, step.handle)?;
                let log_alpha = map.members_log_density(&ys, &anchors)?
                    - map.members_log_density(&xs, &anchors)?
                    + step.log_reverse_forward;
                assert!(
                    log_alpha.abs() < 1e-8,
                    "exact target must accept every trial"
                );
                accepted += 1;
                ys
            }
            Err(_) => xs.clone(),
        };
        used += 1;
        for (acc, v) in [(&mut before, observable(&xs)), (&mut after, observable(&y))] {
            for k in 0..5 {
                acc[k].0 += v[k];
                acc[k].1 += v[k] * v[k];
            }
        }
    }
    assert!(used > n * 9 / 10 && accepted > used * 9 / 10);
    let nf = used as f64;
    for k in 0..5 {
        let (m0, m1) = (before[k].0 / nf, after[k].0 / nf);
        let var0 = before[k].1 / nf - m0 * m0;
        let var1 = after[k].1 / nf - m1 * m1;
        // Paired draws are positively correlated; independent SE is conservative.
        let se = ((var0 + var1) / nf).sqrt();
        assert!(
            (m0 - m1).abs() < 5. * se,
            "observable {k}: {m0} vs {m1} (se {se})"
        );
    }
    Ok(())
}

/// Draw (member, anchor, branch) with probability w_b N_b(g_a^{-1} g_i) / G_S.
fn posterior_label(
    p: &DockingProposal,
    members: &[Pose],
    anchors: &[Pose],
    rng: &mut StdRng,
) -> Result<MemberLabel> {
    let mut labels = Vec::new();
    let mut logs = Vec::new();
    for (member, &m) in members.iter().enumerate() {
        for (anchor, &a) in anchors.iter().enumerate() {
            let branch_logs = p.branch_log_densities(m, a)?;
            for (branch, l) in branch_logs.into_iter().enumerate() {
                labels.push(MemberLabel {
                    member,
                    anchor,
                    branch,
                });
                logs.push(l);
            }
        }
    }
    let max = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = logs.iter().map(|l| (l - max).exp()).collect();
    let mut u = rng.random::<f64>() * weights.iter().sum::<f64>();
    for (l, w) in labels.iter().zip(&weights) {
        if u < *w {
            return Ok(*l);
        }
        u -= w;
    }
    Ok(*labels.last().unwrap())
}
