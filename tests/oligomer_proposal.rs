//! Fused two-contact oligomer charts: agreement with the member mixture when
//! nothing fuses, exact inverse and correction algebra, invariant context
//! reconstruction after a carried move, and exact preservation of G_O.
use anyhow::Result;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde_json::json;
use std::f64::consts::PI;
use tetramer_mc::{
    docking::{DockingMethod, DockingProposal},
    geometry::{Atom, Shape, SphereTree},
    math::{Pose, cayley, invert_relative_pose, matmul, matvec, norm, quaternion, rotation, sub},
    oligomer_proposal::{OligomerConfig, OligomerMixture},
    proposal::FrozenRelativePoseProposal,
    rigid_subset::transport_members,
    spherical::Container,
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
    FrozenRelativePoseProposal::from_json_str_open(&raw.to_string(), [80.; 3], 0.1, SHA)
}
fn proposal(reciprocal: bool, correlation: f64) -> Result<DockingProposal> {
    DockingProposal::new(
        model(reciprocal)?,
        DockingMethod::PosteriorInvolution,
        correlation,
        [0.; 3],
    )
}
fn compose(a: Pose, b: Pose) -> Pose {
    let ra = rotation(a.orientation);
    Pose {
        position: std::array::from_fn(|k| a.position[k] + matvec(ra, b.position)[k]),
        orientation: quaternion(matmul(ra, rotation(b.orientation))),
    }
}
fn random_pose(rng: &mut StdRng, spread: f64) -> Pose {
    let q: [f64; 4] = std::array::from_fn(|_| StandardNormal.sample(rng));
    let n = q.iter().map(|x| x * x).sum::<f64>().sqrt();
    Pose {
        position: std::array::from_fn(|_| rng.random_range(-spread..spread)),
        orientation: q.map(|x| x / n),
    }
}
fn small_sphere() -> (SphereTree, Container) {
    let tree = SphereTree::new(Shape {
        name: "small sphere".into(),
        volume: 4. * PI / 3. * 0.2_f64.powi(3),
        atoms: vec![Atom {
            center: [0.; 3],
            radius: 0.2,
        }],
    })
    .unwrap();
    let wall = Container::new(35., &tree).unwrap();
    (tree, wall)
}

/// Two-member subset and three anchors. Anchor 0 holds member 0 at branch
/// `b0`'s chart mean and anchor 1 holds member 1 at branch `b1`'s mean, so
/// that pair of labels fuses with zero mismatch. Anchor 2 is random.
fn docked_scene(p: &DockingProposal, rng: &mut StdRng) -> (Vec<Pose>, Vec<Pose>) {
    let (map, inverted, _) = p.member_chart_parts();
    let mean = |b: usize| {
        let y = map.decode(b, [0.; 6]).unwrap();
        if inverted[b] {
            invert_relative_pose(y)
        } else {
            y
        }
    };
    let g0 = random_pose(rng, 3.);
    let g1 = compose(
        g0,
        Pose {
            position: [2.3, 0.4, -0.2],
            orientation: quaternion(cayley([0.2, -0.3, 0.1])),
        },
    );
    let (b0, b1) = (0, inverted.len() - 1);
    let a0 = compose(g0, invert_relative_pose(mean(b0)));
    let a1 = compose(g1, invert_relative_pose(mean(b1)));
    let a2 = compose(g0, random_pose(rng, 6.));
    (vec![g0, g1], vec![a0, a1, a2])
}
fn loose() -> OligomerConfig {
    OligomerConfig {
        max_mismatch: 40.,
        pair_distance_a: 30.,
        pair_angle_degrees: 180.,
        ..OligomerConfig::default()
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
fn without_fused_components_the_mixture_is_the_member_mixture() -> Result<()> {
    let (tree, wall) = small_sphere();
    for reciprocal in [false, true] {
        let p = proposal(reciprocal, 0.6)?;
        let mut rng = StdRng::seed_from_u64(5);
        for _ in 0..50 {
            let members: Vec<_> = (0..3).map(|_| random_pose(&mut rng, 3.)).collect();
            let anchors: Vec<_> = (0..3).map(|_| random_pose(&mut rng, 5.)).collect();
            let none = OligomerConfig {
                pair_distance_a: 1e-12,
                ..OligomerConfig::default()
            };
            let mixture =
                OligomerMixture::build(&p, &tree, &wall, &members, &anchors, &anchors, &none)?;
            assert!(mixture.fused.is_empty());
            let a = mixture.log_density(&members);
            let b = p.members_log_density(&members, &anchors)?;
            assert!((a - b).abs() < 1e-10 * (1. + b.abs()), "{a} vs {b}");
        }
    }
    Ok(())
}

#[test]
fn docked_pair_fuses_and_context_rebuilds_identically() -> Result<()> {
    let (tree, wall) = small_sphere();
    for reciprocal in [false, true] {
        let p = proposal(reciprocal, 0.6)?;
        let mut rng = StdRng::seed_from_u64(17 + u64::from(reciprocal));
        let (members, anchors) = docked_scene(&p, &mut rng);
        let x = OligomerMixture::build(&p, &tree, &wall, &members, &anchors, &anchors, &loose())?;
        let nb = if reciprocal { 3 } else { 2 };
        // Labels are (member * anchors + anchor) * branches + branch.
        let exact = 0;
        let partner = 4 * nb + nb - 1;
        let docked = x
            .fused
            .iter()
            .find(|f| f.first == exact.min(partner) && f.second == exact.max(partner))
            .expect("docked label pair must fuse");
        assert!(docked.mismatch < 1e-8, "{}", docked.mismatch);
        assert!(same_pose(docked.center, members[0], 1e-6));
        // Rigidly carry the subset anywhere: every construction output agrees.
        let (s, m) = state(&members);
        let moved = transport_members(&s, &m, 1, random_pose(&mut rng, 4.))?;
        let y = OligomerMixture::build(&p, &tree, &wall, &moved, &anchors, &anchors, &loose())?;
        assert_eq!(x.fused.len(), y.fused.len());
        for (a, b) in x.fused.iter().zip(&y.fused) {
            assert_eq!((a.first, a.second), (b.first, b.second));
            assert!(same_pose(a.center, b.center, 1e-7));
        }
        for (a, b) in x.log_weights().iter().zip(y.log_weights()) {
            assert!((a - b).abs() < 1e-9);
        }
        for g in [members.clone(), moved.clone()] {
            assert!((x.log_density(&g) - y.log_density(&g)).abs() < 1e-7);
        }
    }
    Ok(())
}

#[test]
fn oligomer_map_inverts_and_correction_matches_expanded_trace() -> Result<()> {
    let (tree, wall) = small_sphere();
    for reciprocal in [false, true] {
        let p = proposal(reciprocal, 0.6)?;
        let mut rng = StdRng::seed_from_u64(29 + u64::from(reciprocal));
        let mut fused_checked = 0;
        let mut checked = 0;
        for _ in 0..40 {
            let (members, anchors) = docked_scene(&p, &mut rng);
            let x =
                OligomerMixture::build(&p, &tree, &wall, &members, &anchors, &anchors, &loose())?;
            assert!(!x.fused.is_empty());
            let singles = x.label_count() - x.fused.len();
            let logs = x.label_logs(members[0]);
            for _ in 0..20 {
                // Posterior-plausible source, any destination, fused-heavy.
                let source = (0..x.label_count())
                    .filter(|&l| logs[l] > logs.iter().cloned().fold(f64::MIN, f64::max) - 30.)
                    .nth(rng.random_range(0..3))
                    .unwrap_or(singles);
                let target = if rng.random::<bool>() {
                    singles + rng.random_range(0..x.fused.len())
                } else {
                    rng.random_range(0..x.label_count())
                };
                let handle = rng.random_range(0..2);
                let Ok(forward) = x.apply(&members, handle, source, target, noise(&mut rng)) else {
                    continue;
                };
                let (s, m) = state(&members);
                let moved = transport_members(&s, &m, handle, forward.handle)?;
                let direct = x.log_density(&members) - x.log_density(&moved);
                assert!((forward.log_reverse_forward - direct).abs() < 1e-9);
                assert!(
                    (forward.expanded_log_reverse_forward - forward.log_reverse_forward).abs()
                        < 1e-9 + 1e-12 * forward.step.log_correction.abs(),
                    "expansion {} != {}",
                    forward.expanded_log_reverse_forward,
                    forward.log_reverse_forward
                );
                let reverse = x.apply(
                    &moved,
                    handle,
                    target,
                    source,
                    forward.step.inverse_trace.noise,
                )?;
                assert!(same_pose(reverse.handle, members[handle], 1e-8));
                assert!((reverse.log_reverse_forward + forward.log_reverse_forward).abs() < 1e-8);
                checked += 1;
                fused_checked += usize::from(source >= singles || target >= singles);
            }
        }
        assert!(
            checked > 600 && fused_checked > 300,
            "{checked} {fused_checked}"
        );
    }
    Ok(())
}

/// With c = 0 a destination draw is an exact sample of that chart. Drawing
/// the destination label from the mixture weights gives an exact G_O sample,
/// which the correlated posterior-source map must preserve with unit
/// acceptance min(1, G_O(y)/G_O(x) e^corr).
#[test]
fn correlated_oligomer_map_preserves_the_oligomer_mixture() -> Result<()> {
    let (tree, wall) = small_sphere();
    let fresh = proposal(true, 0.)?;
    let map = proposal(true, 0.7)?;
    let mut rng = StdRng::seed_from_u64(20260926);
    let (members, anchors) = docked_scene(&fresh, &mut rng);
    let x0 = OligomerMixture::build(&fresh, &tree, &wall, &members, &anchors, &anchors, &loose())?;
    let x7 = OligomerMixture::build(&map, &tree, &wall, &members, &anchors, &anchors, &loose())?;
    assert!(x0.fused.len() >= 2);
    let weights: Vec<f64> = x0.log_weights().iter().map(|w| w.exp()).collect();
    let draw = |rng: &mut StdRng| {
        let mut u = rng.random::<f64>() * weights.iter().sum::<f64>();
        for (l, w) in weights.iter().enumerate() {
            if u < *w {
                return l;
            }
            u -= w;
        }
        weights.len() - 1
    };
    let start = {
        let logs = x0.label_logs(members[0]);
        (0..logs.len())
            .max_by(|&a, &b| logs[a].total_cmp(&logs[b]))
            .unwrap()
    };
    let observable = |m: &[Pose]| -> [f64; 5] {
        let c: [f64; 3] = std::array::from_fn(|k| 0.5 * (m[0].position[k] + m[1].position[k]));
        [
            c[0],
            c[1],
            c[2],
            norm(sub(c, members[0].position)),
            m[0].orientation[0].powi(2),
        ]
    };
    let (s, m) = state(&members);
    let mut before = [(0., 0.); 5];
    let mut after = [(0., 0.); 5];
    let (mut used, mut accepted, mut fused_moves) = (0, 0, 0);
    let singles = x7.label_count() - x7.fused.len();
    for _ in 0..30_000 {
        let pick = draw(&mut rng);
        let Ok(sample) = x0.apply(&members, 0, start, pick, noise(&mut rng)) else {
            continue;
        };
        let xs = transport_members(&s, &m, 0, sample.handle)?;
        let logs = x7.label_logs(xs[0]);
        let max = logs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut u = rng.random::<f64>() * logs.iter().map(|l| (l - max).exp()).sum::<f64>();
        let mut source = logs.len() - 1;
        for (l, v) in logs.iter().enumerate() {
            let w = (v - max).exp();
            if u < w {
                source = l;
                break;
            }
            u -= w;
        }
        let target = draw(&mut rng);
        let y = match x7.apply(&xs, 0, source, target, noise(&mut rng)) {
            Ok(step) => {
                let ys = transport_members(&s, &m, 0, step.handle)?;
                let log_alpha =
                    x7.log_density(&ys) - x7.log_density(&xs) + step.log_reverse_forward;
                assert!(
                    log_alpha.abs() < 1e-8,
                    "exact target must accept every trial"
                );
                accepted += 1;
                fused_moves += usize::from(source >= singles || target >= singles);
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
    assert!(used > 27_000 && accepted > used * 9 / 10 && fused_moves > used / 2);
    let n = used as f64;
    for k in 0..5 {
        let (m0, m1) = (before[k].0 / n, after[k].0 / n);
        let se = ((before[k].1 / n - m0 * m0 + after[k].1 / n - m1 * m1) / n).sqrt();
        assert!(
            (m0 - m1).abs() < 5. * se,
            "observable {k}: {m0} vs {m1} (se {se})"
        );
    }
    Ok(())
}
