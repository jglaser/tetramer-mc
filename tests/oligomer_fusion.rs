//! Independent algebra and involution checks for fused oligomer chart proposals.
use anyhow::Result;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde_json::json;
use tetramer_mc::{
    basin_involution::BasinTrace,
    docking::{DockingMethod, DockingProposal},
    math::{
        Pose, add, cayley, invert_relative_pose, matmul, matvec, norm, quaternion, rotation, sub,
        transpose,
    },
    oligomer_fusion::{FusedCatalogue, FusionConfig, gaussian_product, pullback_contact},
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
fn noise(rng: &mut StdRng) -> [f64; 6] {
    std::array::from_fn(|_| StandardNormal.sample(rng))
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

type Matrix = [[f64; 6]; 6];
fn inverse_and_logdet(a: Matrix) -> (Matrix, f64) {
    // Independent row-reduction, deliberately not the production Cholesky path.
    let mut m = [[0.; 12]; 6];
    for i in 0..6 {
        for j in 0..6 {
            m[i][j] = a[i][j];
        }
        m[i][6 + i] = 1.;
    }
    let mut logdet = 0.;
    for i in 0..6 {
        let pivot = (i..6)
            .max_by(|&j, &k| m[j][i].abs().total_cmp(&m[k][i].abs()))
            .unwrap();
        m.swap(i, pivot);
        let d = m[i][i];
        assert!(d.abs() > 1e-14);
        logdet += d.abs().ln();
        for j in 0..12 {
            m[i][j] /= d;
        }
        for k in 0..6 {
            if k != i {
                let t = m[k][i];
                for j in 0..12 {
                    m[k][j] -= t * m[i][j];
                }
            }
        }
    }
    (
        std::array::from_fn(|i| std::array::from_fn(|j| m[i][6 + j])),
        logdet,
    )
}
fn log_pdf(x: [f64; 6], mean: [f64; 6], cov: Matrix) -> f64 {
    let (inv, logdet) = inverse_and_logdet(cov);
    let d: [f64; 6] = std::array::from_fn(|i| x[i] - mean[i]);
    let mut quad = 0.;
    for i in 0..6 {
        for j in 0..6 {
            quad += d[i] * inv[i][j] * d[j];
        }
    }
    -0.5 * (6. * (2. * std::f64::consts::PI).ln() + logdet + quad)
}
fn covariance(scale: f64) -> Matrix {
    let mut l = [[0.; 6]; 6];
    for i in 0..6 {
        l[i][i] = scale * (0.7 + 0.2 * i as f64);
    }
    l[3][0] = 0.31;
    l[4][1] = -0.27;
    l[5][0] = 0.16;
    l[5][3] = 0.2;
    std::array::from_fn(|i| std::array::from_fn(|j| (0..6).map(|k| l[i][k] * l[j][k]).sum()))
}
#[test]
fn gaussian_product_has_independent_mean_covariance_and_normalizer() -> Result<()> {
    let m1 = [1., -0.2, 0.3, 0.6, -0.7, 0.5];
    let m2 = [-0.4, 0.8, 0.1, -0.3, 0.2, -0.1];
    let (c1, c2) = (covariance(0.8), covariance(1.3));
    let product = gaussian_product(m1, c1, m2, c2)?;
    let (p1, _) = inverse_and_logdet(c1);
    let (p2, _) = inverse_and_logdet(c2);
    let precision = std::array::from_fn(|i| std::array::from_fn(|j| p1[i][j] + p2[i][j]));
    let (expected_cov, _) = inverse_and_logdet(precision);
    let rhs: [f64; 6] =
        std::array::from_fn(|i| (0..6).map(|j| p1[i][j] * m1[j] + p2[i][j] * m2[j]).sum());
    let expected_mean: [f64; 6] =
        std::array::from_fn(|i| (0..6).map(|j| expected_cov[i][j] * rhs[j]).sum());
    for i in 0..6 {
        assert!((product.mean[i] - expected_mean[i]).abs() < 1e-11);
        for j in 0..6 {
            assert!((product.covariance[i][j] - expected_cov[i][j]).abs() < 1e-11);
        }
    }
    let sum_cov = std::array::from_fn(|i| std::array::from_fn(|j| c1[i][j] + c2[i][j]));
    assert!((product.log_compatibility - log_pdf(m1, m2, sum_cov)).abs() < 1e-11);
    let mut rng = StdRng::seed_from_u64(41721);
    for _ in 0..100 {
        let x = noise(&mut rng);
        let lhs = log_pdf(x, m1, c1) + log_pdf(x, m2, c2);
        let rhs = product.log_compatibility + log_pdf(x, product.mean, product.covariance);
        assert!(
            (lhs - rhs).abs() < 2e-10,
            "unnormalized product mismatch {lhs} vs {rhs}"
        );
    }
    let swapped = gaussian_product(m2, c2, m1, c1)?;
    assert!((swapped.log_compatibility - product.log_compatibility).abs() < 1e-12);
    Ok(())
}
#[test]
fn fusion_config_guards_and_defensive_original_mass() -> Result<()> {
    FusionConfig::default().validate()?;
    for weight in [f64::NAN, -0.1, 1., 1.1] {
        let mut cfg = FusionConfig::default();
        cfg.fused_weight = weight;
        assert!(cfg.validate().is_err());
    }
    for zero in 0..3 {
        let mut cfg = FusionConfig::default();
        match zero {
            0 => cfg.max_fused = 0,
            1 => cfg.candidates_per_member_anchor = 0,
            _ => cfg.max_pair_candidates = 0,
        };
        assert!(cfg.validate().is_err());
    }
    let mut cfg = FusionConfig::default();
    cfg.fused_weight = 0.;
    cfg.validate()?;
    Ok(())
}

fn compose(a: Pose, b: Pose) -> Pose {
    Pose {
        position: add(a.position, matvec(rotation(a.orientation), b.position)),
        orientation: quaternion(matmul(rotation(a.orientation), rotation(b.orientation))),
    }
}
fn local_coordinates(reference: Pose, p: Pose, ell: f64) -> [f64; 6] {
    let d = sub(p.position, reference.position);
    let q = quaternion(matmul(
        rotation(p.orientation),
        transpose(rotation(reference.orientation)),
    ));
    [
        d[0],
        d[1],
        d[2],
        ell * q[1] / q[0],
        ell * q[2] / q[0],
        ell * q[3] / q[0],
    ]
}
#[test]
fn pullback_has_lever_arm_and_reciprocal_numerical_derivative() -> Result<()> {
    let parameters = model(false)?.component_parameters().remove(0);
    let mut rng = StdRng::seed_from_u64(60013);
    let anchor = random_pose(&mut rng, 2.);
    let internal = random_pose(&mut rng, 2.);
    assert!(norm(internal.position) > 0.5);
    let ell = 1.3;
    for inverted in [false, true] {
        let pulled = pullback_contact(&parameters, inverted, anchor, internal, ell)?;
        let pose_at = |x: [f64; 6]| {
            let mut relative = Pose {
                position: std::array::from_fn(|i| parameters.anchor_position[i] + x[i]),
                orientation: quaternion(matmul(
                    cayley([x[3] / ell, x[4] / ell, x[5] / ell]),
                    parameters.anchor_rotation,
                )),
            };
            if inverted {
                relative = invert_relative_pose(relative);
            }
            compose(compose(anchor, relative), invert_relative_pose(internal))
        };
        assert!(same_pose(pulled.mean_pose, pose_at(parameters.mean), 1e-11));
        let eps = 1e-6;
        let mut derivative = [[0.; 6]; 6];
        for k in 0..6 {
            let mut plus = parameters.mean;
            plus[k] += eps;
            let mut minus = parameters.mean;
            minus[k] -= eps;
            let yplus = local_coordinates(pulled.mean_pose, pose_at(plus), ell);
            let yminus = local_coordinates(pulled.mean_pose, pose_at(minus), ell);
            for j in 0..6 {
                derivative[j][k] = (yplus[j] - yminus[j]) / (2. * eps);
                assert!(
                    (derivative[j][k] - pulled.jacobian[j][k]).abs() < 2e-8,
                    "reciprocal={inverted}, derivative[{j},{k}]: {} != {}",
                    derivative[j][k],
                    pulled.jacobian[j][k]
                );
            }
        }
        assert!(
            (0..3)
                .flat_map(|i| (3..6).map(move |j| pulled.jacobian[i][j].abs()))
                .sum::<f64>()
                > 0.5,
            "nonzero rotational lever-arm contribution required"
        );
        for i in 0..6 {
            for j in 0..6 {
                let expected = (0..6)
                    .flat_map(|a| {
                        (0..6).map(move |b| {
                            derivative[i][a] * parameters.covariance[a][b] * derivative[j][b]
                        })
                    })
                    .sum::<f64>();
                assert!((expected - pulled.covariance[i][j]).abs() < 1e-7);
            }
        }
    }
    Ok(())
}
fn compact_config() -> FusionConfig {
    FusionConfig {
        max_fused: 8,
        candidates_per_member_anchor: 4,
        max_pair_candidates: 32,
        ..FusionConfig::default()
    }
}
fn two_members() -> Vec<Pose> {
    vec![
        Pose {
            position: [0., 0., 0.],
            orientation: [1., 0., 0., 0.],
        },
        Pose {
            position: [2.1, 0.3, -0.2],
            orientation: quaternion(cayley([0.2, -0.1, 0.3])),
        },
    ]
}
fn two_anchors() -> Vec<Pose> {
    vec![
        Pose {
            position: [-2.4, 0.7, 0.4],
            orientation: quaternion(cayley([-0.3, 0.2, 0.1])),
        },
        Pose {
            position: [0.4, 3.2, -0.5],
            orientation: quaternion(cayley([0.1, 0.3, -0.2])),
        },
    ]
}
#[test]
fn catalogue_is_independent_of_external_contacts_and_has_original_limit() -> Result<()> {
    let p = proposal(true, 0.6)?;
    let members = two_members();
    let anchors = two_anchors();
    let c = FusedCatalogue::build(&p, &members, &anchors, &compact_config())?;
    assert_eq!(c.original_count(), 12);
    assert!(c.chart_count() > c.original_count());
    assert!((c.log_weights().iter().map(|v| v.exp()).sum::<f64>() - 1.).abs() < 1e-12);
    let original_mass = c.log_weights()[..c.original_count()]
        .iter()
        .map(|v| v.exp())
        .sum::<f64>();
    assert!((original_mass - 0.5).abs() < 1e-12);
    let destination = Pose {
        position: [31., -22., 11.],
        orientation: quaternion(cayley([0.4, -0.6, 0.2])),
    };
    let moved = transport_members(&members, &[0, 1], 0, destination)?;
    let external_contact = |state: &[Pose]| {
        state
            .iter()
            .flat_map(|m| {
                anchors
                    .iter()
                    .map(move |a| norm(sub(m.position, a.position)) < 3.)
            })
            .filter(|v| *v)
            .count()
    };
    assert!(external_contact(&members) > 0);
    assert_eq!(external_contact(&moved), 0);
    let rebuilt = FusedCatalogue::build(&p, &moved, &anchors, &compact_config())?;
    assert_eq!(c.chart_count(), rebuilt.chart_count());
    for j in 0..c.chart_count() {
        assert!((c.log_weights()[j] - rebuilt.log_weights()[j]).abs() < 2e-9);
        for z in [[0.; 6], [0.2, -0.3, 0.4, -0.2, 0.1, 0.3]] {
            let a = c.decode(&p, j, z)?;
            let b = rebuilt.decode(&p, j, z)?;
            assert!(
                same_pose(a, b, 2e-8),
                "chart {j}, z={z:?}, old_label={}, new_label={}, old_pose={a:?}, new_pose={b:?}, old_weight={}, new_weight={}, oldG(a)={}, newG(a)={}",
                c.label(j)?,
                rebuilt.label(j)?,
                c.log_weights()[j],
                rebuilt.log_weights()[j],
                c.log_density(&p, a)?,
                rebuilt.log_density(&p, a)?
            );
        }
    }
    let mut cfg = compact_config();
    cfg.fused_weight = 0.;
    let original = FusedCatalogue::build(&p, &members, &anchors, &cfg)?;
    assert_eq!(original.chart_count(), original.original_count());
    let mut rng = StdRng::seed_from_u64(7743);
    for _ in 0..30 {
        let h = random_pose(&mut rng, 5.);
        let body: Vec<_> = (0..2)
            .map(|i| original.member_pose(h, i))
            .collect::<Result<_>>()?;
        assert!(
            (original.log_density(&p, h)? - p.members_log_density(&body, &anchors)?).abs() < 1e-9
        );
    }
    Ok(())
}
/// Same-anchor pair scores are exactly tied across spectator frames in real
/// arithmetic. Cutting either shortlist at such a tie must retain the same
/// semantic labels after an arbitrary rigid motion of the carried oligomer.
#[test]
fn tied_cutoff_catalogues_keep_semantic_labels_weights_and_reverse_maps() -> Result<()> {
    let p = proposal(true, 0.6)?;
    let members = two_members();
    let anchors = two_anchors();
    let mut rng = StdRng::seed_from_u64(49007);
    let moves: Vec<_> = (0..4).map(|_| random_pose(&mut rng, 35.)).collect();
    let probes: Vec<_> = (0..8).map(|_| random_pose(&mut rng, 5.)).collect();
    for max_fused in [1, 2, 3, 8] {
        for max_pair_candidates in [1, 2, 3, 8, 32] {
            let config = FusionConfig {
                max_fused,
                max_pair_candidates,
                ..compact_config()
            };
            let original = FusedCatalogue::build(&p, &members, &anchors, &config)?;
            let semantics = |c: &FusedCatalogue| -> Result<Vec<serde_json::Value>> {
                (0..c.chart_count())
                    .map(|j| {
                        let mut label = c.label(j)?;
                        label.as_object_mut().unwrap().remove("log_compatibility");
                        Ok(label)
                    })
                    .collect()
            };
            let labels = semantics(&original)?;
            for &h in &moves {
                let moved = transport_members(&members, &[0, 1], 0, h)?;
                let rebuilt = FusedCatalogue::build(&p, &moved, &anchors, &config)?;
                assert_eq!(
                    labels,
                    semantics(&rebuilt)?,
                    "semantic cutoff drift max_fused={max_fused}, max_pair_candidates={max_pair_candidates}"
                );
                for j in 0..original.chart_count() {
                    assert!((original.log_weights()[j] - rebuilt.log_weights()[j]).abs() < 2e-9);
                    assert!(same_pose(
                        original.decode(&p, j, [0.; 6])?,
                        rebuilt.decode(&p, j, [0.; 6])?,
                        2e-8
                    ));
                }
                for &probe in &probes {
                    let a = original.log_density(&p, probe)?;
                    let b = rebuilt.log_density(&p, probe)?;
                    assert!(
                        (a - b).abs() < 2e-8,
                        "density cutoff drift max_fused={max_fused}, max_pair_candidates={max_pair_candidates}: {a} vs {b}"
                    );
                }
            }
            let trace = BasinTrace {
                source: 0,
                target: original.chart_count() - 1,
                noise: [0.1, -0.2, 0.3, -0.2, 0.1, 0.05],
            };
            let step = original.apply(&p, members[0], &trace)?;
            let endpoint: Vec<_> = (0..members.len())
                .map(|i| original.member_pose(step.pose, i))
                .collect::<Result<_>>()?;
            let reverse_catalogue = FusedCatalogue::build(&p, &endpoint, &anchors, &config)?;
            assert_eq!(labels, semantics(&reverse_catalogue)?);
            let reverse = reverse_catalogue.apply(&p, step.pose, &step.inverse_trace)?;
            assert!(same_pose(reverse.pose, members[0], 2e-8));
            assert!((step.log_reverse_forward + reverse.log_reverse_forward).abs() < 2e-8);
        }
    }
    Ok(())
}

#[test]
fn complete_chart_densities_match_numerical_haar_jacobians() -> Result<()> {
    let p = proposal(true, 0.6)?;
    let c = FusedCatalogue::build(&p, &two_members(), &two_anchors(), &compact_config())?;
    let ell: f64 = 1.3;
    let eps = 2e-6;
    let mut rng = StdRng::seed_from_u64(97431);
    for label in 0..c.chart_count() {
        let z = noise(&mut rng).map(|v| 0.4 * v);
        let center = c.decode(&p, label, z)?;
        let roundtrip = c.encode(&p, label, center)?;
        for i in 0..6 {
            assert!((roundtrip[i] - z[i]).abs() < 1e-9);
        }
        let mut jacobian = [[0.; 6]; 6];
        for k in 0..6 {
            let mut plus = z;
            plus[k] += eps;
            let mut minus = z;
            minus[k] -= eps;
            let a = local_coordinates(center, c.decode(&p, label, plus)?, ell);
            let b = local_coordinates(center, c.decode(&p, label, minus)?, ell);
            for j in 0..6 {
                jacobian[j][k] = (a[j] - b[j]) / (2. * eps);
            }
        }
        let (_, logdet) = inverse_and_logdet(jacobian);
        let lognormal =
            -3. * (2. * std::f64::consts::PI).ln() - 0.5 * z.iter().map(|v| v * v).sum::<f64>();
        let expected = lognormal - logdet + 3. * ell.ln() + 2. * std::f64::consts::PI.ln();
        let actual = c.component_log_density(&p, label, center)?;
        assert!(
            (expected - actual).abs() < 2e-7,
            "chart{label} expected={expected}, actual={actual}"
        );
    }
    Ok(())
}
#[test]
fn mixed_fusion_maps_invert_and_match_expanded_auxiliary_flow() -> Result<()> {
    let p = proposal(true, 0.7)?;
    let members = two_members();
    let anchors = two_anchors();
    let c = FusedCatalogue::build(&p, &members, &anchors, &compact_config())?;
    let mut rng = StdRng::seed_from_u64(4601);
    let mut kinds = [0; 4];
    for i in 0..240 {
        let source = if i % 2 == 0 {
            rng.random_range(0..c.original_count())
        } else {
            rng.random_range(c.original_count()..c.chart_count())
        };
        let target = if i % 4 < 2 {
            rng.random_range(0..c.original_count())
        } else {
            rng.random_range(c.original_count()..c.chart_count())
        };
        let old = c.decode(&p, source, noise(&mut rng))?;
        let trace = BasinTrace {
            source,
            target,
            noise: noise(&mut rng),
        };
        let forward = c.apply(&p, old, &trace)?;
        let reverse = c.apply(&p, forward.pose, &forward.inverse_trace)?;
        assert!(same_pose(old, reverse.pose, 2e-8));
        assert!((forward.log_reverse_forward + reverse.log_reverse_forward).abs() < 1e-7);
        assert!((forward.expanded_log_reverse_forward - forward.log_reverse_forward).abs() < 1e-8);
        let direct = c.log_density(&p, old)? - c.log_density(&p, forward.pose)?;
        assert!((direct - forward.log_reverse_forward).abs() < 1e-10);
        for k in 0..6 {
            assert!((reverse.inverse_trace.noise[k] - trace.noise[k]).abs() < 1e-8);
        }
        let norm_before = forward.source_latent.iter().map(|v| v * v).sum::<f64>()
            + reverse
                .inverse_trace
                .noise
                .iter()
                .map(|v| v * v)
                .sum::<f64>();
        let norm_after = forward.target_latent.iter().map(|v| v * v).sum::<f64>()
            + forward
                .inverse_trace
                .noise
                .iter()
                .map(|v| v * v)
                .sum::<f64>();
        assert!((norm_before - norm_after).abs() < 1e-8);
        kinds[usize::from(source >= c.original_count()) * 2
            + usize::from(target >= c.original_count())] += 1;
    }
    assert!(
        kinds.iter().all(|&n| n > 40),
        "all four chart-family directions covered: {kinds:?}"
    );
    Ok(())
}
fn draw_log(rng: &mut StdRng, logs: &[f64]) -> usize {
    let maximum = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<_> = logs.iter().map(|v| (v - maximum).exp()).collect();
    let mut u = rng.random::<f64>() * weights.iter().sum::<f64>();
    for (i, &w) in weights.iter().enumerate() {
        if u < w {
            return i;
        }
        u -= w;
    }
    weights.len() - 1
}
#[test]
fn correlated_mixed_map_preserves_complete_normalized_catalogue() -> Result<()> {
    let p = proposal(true, 0.6)?;
    let c = FusedCatalogue::build(&p, &two_members(), &two_anchors(), &compact_config())?;
    let mut rng = StdRng::seed_from_u64(619207);
    let n = 20_000;
    let mut before = [(0., 0.); 5];
    let mut after = [(0., 0.); 5];
    let mut mixed = 0;
    let observable = |p: Pose| {
        [
            p.position[0],
            p.position[1],
            p.position[2],
            norm(p.position).powi(2),
            p.orientation[0].powi(2),
        ]
    };
    for _ in 0..n {
        // Direct sampling of a normalized catalogue component, without MCMC.
        let label = draw_log(&mut rng, c.log_weights());
        let x = c.decode(&p, label, noise(&mut rng))?;
        let source = draw_log(&mut rng, &c.label_log_densities(&p, x)?);
        let target = draw_log(&mut rng, c.log_weights());
        let step = c.apply(
            &p,
            x,
            &BasinTrace {
                source,
                target,
                noise: noise(&mut rng),
            },
        )?;
        let log_accept =
            c.log_density(&p, step.pose)? - c.log_density(&p, x)? + step.log_reverse_forward;
        assert!(
            log_accept.abs() < 1e-10,
            "catalogue target accepts every exact-map proposal"
        );
        mixed += usize::from((source < c.original_count()) != (target < c.original_count()));
        for (acc, values) in [
            (&mut before, observable(x)),
            (&mut after, observable(step.pose)),
        ] {
            for k in 0..5 {
                acc[k].0 += values[k];
                acc[k].1 += values[k] * values[k];
            }
        }
    }
    assert!(mixed > n / 4);
    for k in 0..5 {
        let a = before[k].0 / n as f64;
        let b = after[k].0 / n as f64;
        let variance = before[k].1 / n as f64 - a * a + after[k].1 / n as f64 - b * b;
        let se = (variance / n as f64).max(0.).sqrt();
        assert!(
            (a - b).abs() < 5. * se + 1e-9,
            "observable{k}: {a} vs {b}, se={se}"
        );
    }
    Ok(())
}
