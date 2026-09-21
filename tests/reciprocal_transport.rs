//! Independent mixture starts, joint virtual-label laws and exact inverse traces.
use anyhow::Result;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde_json::{Value, json};
use tetramer_mc::{
    basin_involution::{BasinPair, BasinTrace, FixedBasinInvolution},
    docking::{DockingMethod, DockingProposal},
    math::*,
    proposal::FrozenRelativePoseProposal,
};

const SHA: &str = "0000000000000000000000000000000000000000000000000000000000000000";
const W: [f64; 2] = [0.23, 0.77];

fn base_json() -> Value {
    let covariances: Vec<_> = (0..2)
        .map(|k| {
            let scales = if k == 0 {
                [0.6, 0.8, 0.7, 0.3, 0.5, 0.4]
            } else {
                [1.3, 0.9, 0.7, 0.6, 0.8, 0.5]
            };
            let mut l = [[0.; 6]; 6];
            for i in 0..6 {
                l[i][i] = scales[i];
            }
            l[3][0] = 0.25;
            l[4][2] = -0.13;
            l[5][1] = 0.18;
            let c: [[f64; 6]; 6] = std::array::from_fn(|i| {
                std::array::from_fn(|j| (0..6).map(|a| l[i][a] * l[j][a]).sum())
            });
            c
        })
        .collect();
    json!({"coordinate_convention":"anchor-body-relative","shape_sha256":SHA,
        "angular_length":1.4,"weights":W,
        "anchors":[{"position":[3.,-2.,1.],"rotation":cayley([0.3,-0.1,0.4])},
                   {"position":[-1.,2.5,3.],"rotation":cayley([-0.2,0.3,0.1])}],
        "means":vec![[0.05,0.03,-0.04,0.1,-0.05,0.04];2],"covariances":covariances})
}
fn models() -> Result<(FrozenRelativePoseProposal, FrozenRelativePoseProposal)> {
    let base = base_json();
    let wrapped = json!({"schema":"reciprocal-pose-mixture-v1","base_model":base,"reciprocal_components":[true,true]});
    Ok((
        FrozenRelativePoseProposal::from_json_str_open(&base.to_string(), [50.; 3], 0.12, SHA)?,
        FrozenRelativePoseProposal::from_json_str_open(&wrapped.to_string(), [50.; 3], 0.12, SHA)?,
    ))
}
fn base_map(base: &FrozenRelativePoseProposal) -> Result<FixedBasinInvolution> {
    FixedBasinInvolution::new(
        base.component_parameters(),
        1.4,
        0.,
        vec![BasinPair {
            first: 0,
            second: 1,
            weight: 1.,
        }],
    )
}
fn normal(rng: &mut StdRng) -> [f64; 6] {
    std::array::from_fn(|_| StandardNormal.sample(rng))
}
fn near(a: Pose, b: Pose) {
    assert!(norm(sub(a.position, b.position)) < 2e-10);
    assert!(
        rotation(a.orientation)
            .iter()
            .flatten()
            .zip(rotation(b.orientation).iter().flatten())
            .all(|(x, y)| (x - y).abs() < 2e-10)
    );
}
fn log_g(model: &FrozenRelativePoseProposal, p: Pose) -> Result<f64> {
    model.relative_log_density(p.position, rotation(p.orientation))
}
fn lab(p: Pose, anchor: Pose) -> Pose {
    Pose {
        position: anchor.apply(p.position),
        orientation: quaternion(matmul(
            rotation(anchor.orientation),
            rotation(p.orientation),
        )),
    }
}
fn relative(p: Pose, anchor: Pose) -> Pose {
    Pose {
        position: anchor.inverse(p.position),
        orientation: quaternion(matmul(
            transpose(rotation(anchor.orientation)),
            rotation(p.orientation),
        )),
    }
}

#[test]
fn all_virtual_label_inverses_recover_pose_noise_and_physical_jacobian() -> Result<()> {
    let (base, model) = models()?;
    let map = base_map(&base)?;
    let branches = model.virtual_branches();
    let z = [0.2, -0.3, 0.4, -0.25, 0.15, 0.1];
    for c in [-1., -0.6, 0., 0.9, 1.] {
        let proposal = DockingProposal::new(
            model.clone(),
            DockingMethod::PosteriorInvolution,
            c,
            [0.; 3],
        )?;
        for a in 0..4 {
            for b in 0..4 {
                let p = map.decode(branches[a].component_index, z)?;
                let p = if branches[a].inverted {
                    invert_relative_pose(p)
                } else {
                    p
                };
                let trace = BasinTrace {
                    source: a,
                    target: b,
                    noise: [-0.4, 0.1, 0.5, 0.3, -0.2, 0.6],
                };
                let forward = proposal.apply_relative_trace(p, &trace)?;
                let backward =
                    proposal.apply_relative_trace(forward.pose, &forward.inverse_trace)?;
                near(p, backward.pose);
                assert_eq!(backward.inverse_trace.source, a);
                assert_eq!(backward.inverse_trace.target, b);
                for (x, y) in trace.noise.iter().zip(backward.inverse_trace.noise) {
                    assert!((x - y).abs() < 3e-10);
                }
                assert!((forward.log_correction + backward.log_correction).abs() < 3e-10);
                assert!(
                    (forward.log_extended_jacobian + backward.log_extended_jacobian).abs() < 3e-10
                );
                if c == 1. && a / 2 == b / 2 && a != b {
                    near(forward.pose, invert_relative_pose(p));
                    assert!(norm(sub(p.position, forward.pose.position)) > 1.);
                }
            }
        }
    }
    Ok(())
}

#[test]
fn source_parity_is_conditional_destination_parity_is_fair_and_c0_redraws() -> Result<()> {
    let (base, model) = models()?;
    let map = base_map(&base)?;
    let old = map.decode(0, [0.; 6])?;
    let proposal = DockingProposal::new(
        model.clone(),
        DockingMethod::PosteriorInvolution,
        0.,
        [0.; 3],
    )?;
    let anchor = Pose {
        position: [4., -3., 2.],
        orientation: quaternion(cayley([0.3, 0.4, -0.2])),
    };
    let branches = model.virtual_branches();
    let components = base
        .component_parameters()
        .into_iter()
        .map(|mut p| {
            p.weight = 1.;
            FrozenRelativePoseProposal::from_components_open(vec![p], 1.4, [50.; 3], 0.12, SHA, SHA)
        })
        .collect::<Result<Vec<_>>>()?;
    let expected: Vec<_> = branches
        .iter()
        .map(|b| {
            let p = if b.inverted {
                invert_relative_pose(old)
            } else {
                old
            };
            Ok(
                (b.weight.ln() + log_g(&components[b.component_index], p)? - log_g(&model, old)?)
                    .exp(),
            )
        })
        .collect::<Result<_>>()?;
    assert!(
        (expected[1] + expected[3] - 0.5).abs() > 0.35,
        "Control must discriminate a fair source parity coin"
    );
    let mut rng = StdRng::seed_from_u64(119001010);
    let (mut sources, mut targets, mut n) = ([0.; 4], [0.; 4], 0.);
    for _ in 0..16000 {
        let (candidate, info) = proposal.propose(&mut rng, lab(old, anchor), &[anchor])?;
        if info["branch"] == "uniform" {
            continue;
        }
        let trace: BasinTrace = serde_json::from_value(info["trace"].clone())?;
        sources[trace.source] += 1.;
        targets[trace.target] += 1.;
        n += 1.;
        let target = branches[trace.target];
        assert_eq!(info["source_inverted"], branches[trace.source].inverted);
        assert_eq!(info["target_component_index"], target.component_index);
        let p = relative(candidate.unwrap(), anchor);
        let p = if target.inverted {
            invert_relative_pose(p)
        } else {
            p
        };
        let z = map.encode(target.component_index, p)?;
        for (a, b) in z.into_iter().zip(trace.noise) {
            assert!((a - b).abs() < 3e-10);
        }
        let correction = info["log_reverse_forward"].as_f64().unwrap();
        assert!(
            (correction - info["expanded_log_reverse_forward"].as_f64().unwrap()).abs() < 5e-10
        );
    }
    for i in 0..4 {
        for (count, p) in [(sources[i], expected[i]), (targets[i], branches[i].weight)] {
            assert!((count - n * p).abs() < 5.5 * (n * p * (1. - p)).sqrt().max(1.));
        }
    }
    Ok(())
}

#[test]
fn independently_drawn_reciprocal_mixture_is_stationary_under_correlated_transport() -> Result<()> {
    let (base, model) = models()?;
    let map = base_map(&base)?;
    let anchor = Pose {
        position: [1., -3., 2.],
        orientation: quaternion(cayley([0.4, 0.1, -0.2])),
    };
    let observable = |p: Pose| {
        let r = rotation(p.orientation);
        [
            p.position[0],
            p.position[0].powi(2),
            p.position[1],
            p.position[2],
            r[0][0],
            r[0][0].powi(2),
            p.position[0] * r[1][1],
            r[2][0],
        ]
    };
    for (ci, c) in [-1., -0.6, 0., 0.9, 1.].into_iter().enumerate() {
        let proposal = DockingProposal::new(
            model.clone(),
            DockingMethod::PosteriorInvolution,
            c,
            [0.; 3],
        )?;
        let mut starts = StdRng::seed_from_u64(119003028);
        let mut rng = StdRng::seed_from_u64(119004037 + ci as u64);
        let (mut sum, mut sq, mut n) = ([0.; 8], [0.; 8], 0.);
        for _ in 0..6000 {
            let k = usize::from(starts.random::<f64>() >= W[0]);
            let old = map.decode(k, normal(&mut starts))?;
            let old = if starts.random::<bool>() {
                invert_relative_pose(old)
            } else {
                old
            };
            let (candidate, info) = proposal.propose(&mut rng, lab(old, anchor), &[anchor])?;
            if info["branch"] == "uniform" {
                continue;
            }
            let new = relative(candidate.expect("finite reciprocal Gaussian draw"), anchor);
            let correction = info["log_reverse_forward"].as_f64().unwrap();
            assert!((correction + log_g(&model, new)? - log_g(&model, old)?).abs() < 3e-10);
            assert!(
                (correction - info["expanded_log_reverse_forward"].as_f64().unwrap()).abs() < 5e-10
            );
            for (j, (a, b)) in observable(new).into_iter().zip(observable(old)).enumerate() {
                sum[j] += a - b;
                sq[j] += (a - b).powi(2);
            }
            n += 1.;
        }
        let mut max_z = 0_f64;
        for j in 0..8 {
            let mean = sum[j] / n;
            let se = ((sq[j] / n - mean * mean).max(0.) / (n - 1.))
                .sqrt()
                .max(1e-14);
            max_z = max_z.max((mean / se).abs());
        }
        eprintln!("Reciprocal exact-mixture c={c}: draws={n}, maximum paired z={max_z:.3}");
        assert!(n > 4800. && max_z < 5.5);
    }
    Ok(())
}
