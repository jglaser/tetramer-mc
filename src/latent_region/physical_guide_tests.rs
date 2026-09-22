//! Pure coordinate/density tests: no atoms, environment, overlap or Poisson calls.
use super::*;
use rand::SeedableRng;
use serde_json::json;

const SHAPE: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

fn fixture(transformed: bool, alpha: f64) -> (Vec<u8>, Vec<u8>) {
    let identity: [[f64; 6]; 6] =
        std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1. } else { 0. }));
    let fixed = if transformed {
        Pose {
            position: [3., -2., 1.],
            orientation: [0.5_f64.sqrt(), 0.5_f64.sqrt(), 0., 0.],
        }
    } else {
        Pose {
            position: [0.; 3],
            orientation: [1., 0., 0., 0.],
        }
    };
    let mut lower = identity;
    if transformed {
        for (i, scale) in [2., 0.5, 1.5, 0.2, 0.3, 0.4].iter().enumerate() {
            lower[i][i] = *scale;
        }
        lower[3][0] = 0.2;
        lower[4][1] = -0.1;
        lower[5][2] = 0.15;
    }
    let covariance: [[f64; 6]; 6] = std::array::from_fn(|i| {
        std::array::from_fn(|j| (0..6).map(|k| lower[i][k] * lower[j][k]).sum())
    });
    let anchor_rotation = if transformed {
        [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    } else {
        IDENTITY
    };
    let mean = if transformed {
        [0.2, -0.1, 0.4, 0.05, -0.03, 0.1]
    } else {
        [0.; 6]
    };
    let region = json!({"shape_sha256":SHAPE,"fixed_neighbor":fixed,"physical_fixed_neighbors":[fixed],
        "capture_center":[0.,0.,0.],"capture_radius":0.1,"mahalanobis_radius":4.,"minimum_original_q":0.,
        "gaussian_chart":{"shape_sha256":SHAPE,"coordinate_convention":"anchor-body-relative","angular_length":if transformed {2.} else {1.},
            "anchors":[{"position":if transformed {[0.1,0.2,-0.3]} else {[0.,0.,0.]},"rotation":anchor_rotation}],
            "means":[mean],"covariances":[covariance],"weights":[1.]}});
    let region_raw = serde_json::to_vec(&region).unwrap();
    let guide = json!({"schema":"defensive-latent-shell-guide-v1","region_sha256":hash_bytes(&region_raw),
        "defensive_uniform_shell_probability":alpha,"gaussian_components":[{"weight":1.,"mean":[0.,0.,0.,0.,0.,0.],"covariance":identity}]});
    (region_raw, serde_json::to_vec(&guide).unwrap())
}

fn parsed(transformed: bool, alpha: f64) -> Result<PhysicalLatentGuide> {
    let (region, guide) = fixture(transformed, alpha);
    PhysicalLatentGuide::from_bytes(&region, &guide, SHAPE)
}

#[test]
fn density_is_q_over_j_inside_and_outside_with_two_full_covariance_charts() -> Result<()> {
    for transformed in [false, true] {
        let guide = parsed(transformed, 0.5)?;
        for latent in [
            [0.2, -0.3, 0.4, 0.1, 0.2, -0.1],
            [5., 0., 0., 0.3, -0.4, 0.1],
        ] {
            let (pose, log_jacobian) = guide.decode(latent)?;
            let result = guide.evaluate(pose)?;
            let radius2: f64 = latent.iter().map(|v| v * v).sum();
            let volume = PI.powi(3) * 4_f64.powi(6) / 6.;
            let q = 0.5 * (-3. * (2. * PI).ln() - 0.5 * radius2).exp()
                + if radius2 <= 16. { 0.5 / volume } else { 0. };
            assert!((result.log_physical_density - (q.ln() - log_jacobian)).abs() < 1e-10);
            assert_eq!(result.in_reference_ball, radius2 <= 16.);
            for (a, b) in result.latent.unwrap().iter().zip(latent) {
                assert!((a - b).abs() < 1e-11);
            }
        }
    }
    Ok(())
}

#[test]
fn physical_normalization_in_two_charts_by_independent_radial_quadrature() -> Result<()> {
    // The synthetic latent law is isotropic. Integrate q_physical(x(u))*J(u)
    // over du with S5=pi³, along a direction involving translation AND rotation.
    // The uniform component is integrated analytically to avoid a ball-edge
    // quadrature discontinuity. Neither chart has a physical target predicate.
    for transformed in [false, true] {
        let guide = parsed(transformed, 0.5)?;
        let volume = PI.powi(3) * 4_f64.powi(6) / 6.;
        let f = |r: f64| -> Result<f64> {
            let (pose, logj) = guide.decode([0.6 * r, 0., 0., 0., 0.8 * r, 0.])?;
            let recovered_q = (guide.log_density(pose)? + logj).exp();
            let gaussian_part = recovered_q - if r <= 4. { 0.5 / volume } else { 0. };
            Ok(PI.powi(3) * r.powi(5) * gaussian_part)
        };
        let count = 4096;
        let step = 12. / count as f64;
        let mut sum = f(0.)? + f(12.)?;
        for i in 1..count {
            sum += if i % 2 == 0 { 2. } else { 4. } * f(i as f64 * step)?;
        }
        let mass = 0.5 + step * sum / 3.;
        assert!(
            (mass - 1.).abs() < 2e-8,
            "transformed={transformed}, mass={mass}"
        );
    }
    Ok(())
}

#[test]
fn guide_draws_remain_unconditional_and_outside_source_capture() -> Result<()> {
    let guide = parsed(true, 0.5)?;
    let mut rng = StdRng::seed_from_u64(80351);
    let mut uniform = 0;
    let mut gaussian = 0;
    let mut exterior = 0;
    let mut outside_capture = 0;
    for _ in 0..512 {
        let draw = guide.draw(&mut rng)?;
        if draw.gaussian_component.is_some() {
            gaussian += 1;
        } else {
            uniform += 1;
            assert!(draw.density.in_reference_ball);
        }
        exterior += usize::from(!draw.density.in_reference_ball);
        outside_capture += usize::from(norm(draw.pose.position) > guide.reference_capture().1);
        let evaluated = guide.log_density(draw.pose)?;
        assert!((evaluated - draw.density.log_physical_density).abs() < 1e-10);
    }
    assert!(uniform > 100 && gaussian > 100 && exterior > 0 && outside_capture > 500);
    Ok(())
}

#[test]
fn exact_seam_is_zero_but_no_nonzero_near_seam_band_is_removed() -> Result<()> {
    let guide = parsed(false, 0.5)?;
    let seam = Pose {
        position: [0.; 3],
        orientation: [0., 1., 0., 0.],
    };
    let zero = guide.evaluate(seam)?;
    assert!(zero.latent.is_none());
    assert_eq!(zero.log_physical_density, f64::NEG_INFINITY);
    for scalar in [1e-6_f64, 1e-12, 1e-50] {
        let point = Pose {
            orientation: [scalar, (1. - scalar * scalar).sqrt(), 0., 0.],
            ..seam
        };
        let value = guide.evaluate(point)?;
        assert!(value.latent.is_some() && value.log_physical_density.is_finite());
        assert!(!value.in_reference_ball);
    }
    // Finite input beyond representable Gaussian log range fails explicitly.
    let unrepresentable = Pose {
        orientation: [1e-200, 1., 0., 0.],
        ..seam
    };
    assert!(guide.evaluate(unrepresentable).is_err());
    Ok(())
}

#[test]
fn pure_uniform_has_zero_exterior_density_without_a_physical_rejection() -> Result<()> {
    let guide = parsed(false, 1.)?;
    let (outside, _) = guide.decode([5., 0., 0., 0., 0., 0.])?;
    let result = guide.evaluate(outside)?;
    assert!(result.latent.is_some());
    assert!(!result.in_reference_ball);
    assert_eq!(result.log_physical_density, f64::NEG_INFINITY);
    Ok(())
}

#[test]
fn exact_outer_half_mixture_retains_vessel_support_and_requires_both_densities() -> Result<()> {
    let actual = half_mixture_log_density(0.3_f64.ln(), 0.7_f64.ln())?;
    assert!((actual - 0.5_f64.ln()).abs() < 1e-15);
    assert!(
        (half_mixture_log_density(0.3_f64.ln(), f64::NEG_INFINITY)? - 0.15_f64.ln()).abs() < 1e-15
    );
    assert_eq!(
        half_mixture_log_density(f64::NEG_INFINITY, f64::NEG_INFINITY)?,
        f64::NEG_INFINITY
    );
    assert!(half_mixture_log_density(0., f64::NAN).is_err());
    for value in [-100., -10., 0., 100.] {
        assert!(half_mixture_log_density(-3., value)? >= -3. - 2_f64.ln());
    }
    Ok(())
}

#[test]
fn exact_source_hash_shape_and_physical_context_are_bound() -> Result<()> {
    let (region, guide_bytes) = fixture(false, 0.5);
    let guide = PhysicalLatentGuide::from_bytes(&region, &guide_bytes, SHAPE)?;
    assert_eq!(guide.region_sha256(), hash_bytes(&region));
    assert_eq!(guide.guide_sha256(), hash_bytes(&guide_bytes));
    guide.validate_physical_context(SHAPE, guide.physical_fixed_neighbors())?;
    assert!(guide.validate_physical_context(SHAPE, &[]).is_err());
    assert!(
        guide
            .validate_physical_context("different", guide.physical_fixed_neighbors())
            .is_err()
    );
    let mut changed = region.clone();
    changed.push(b' ');
    assert!(PhysicalLatentGuide::from_bytes(&changed, &guide_bytes, SHAPE).is_err());
    let mut filtered: Value = serde_json::from_slice(&region)?;
    filtered["minimum_original_q"] = json!(1.);
    assert!(
        PhysicalLatentGuide::from_bytes(&serde_json::to_vec(&filtered)?, &guide_bytes, SHAPE)
            .is_err()
    );
    Ok(())
}

/// Explicit cross-language fixture emission. This invokes only coordinate and
/// density evaluation; there is no physical model or random proposal stream.
#[test]
#[ignore = "emit a fresh math-only fixture for the independent Python evaluator"]
fn emit_independent_coordinate_reference() -> Result<()> {
    let output = std::env::var_os("TETRAMER_GUIDE_REFERENCE_OUT")
        .context("Set TETRAMER_GUIDE_REFERENCE_OUT to a fresh math-validation directory")?;
    let output = std::path::PathBuf::from(output);
    ensure!(!output.exists(), "Fresh math-validation output required");
    fs::create_dir_all(&output)?;
    let mut records = Vec::new();
    for transformed in [false, true] {
        let name = if transformed {
            "transformed"
        } else {
            "identity"
        };
        let (region_raw, guide_raw) = fixture(transformed, 0.5);
        fs::write(output.join(format!("{name}-region.json")), &region_raw)?;
        fs::write(output.join(format!("{name}-guide.json")), &guide_raw)?;
        let guide = PhysicalLatentGuide::from_bytes(&region_raw, &guide_raw, SHAPE)?;
        let mut poses = vec![
            (
                "inside".to_owned(),
                guide.decode([0.2, -0.3, 0.4, 0.1, 0.2, -0.1])?.0,
            ),
            (
                "exterior".to_owned(),
                guide.decode([5., 0., 0., 0.3, -0.4, 0.1])?.0,
            ),
        ];
        if !transformed {
            poses.push((
                "exact-seam".into(),
                Pose {
                    position: [0.; 3],
                    orientation: [0., 1., 0., 0.],
                },
            ));
            for scalar in [1e-6_f64, 1e-12, 1e-50] {
                poses.push((
                    format!("near-seam-{scalar}"),
                    Pose {
                        position: [0.; 3],
                        orientation: [scalar, (1. - scalar * scalar).sqrt(), 0., 0.],
                    },
                ));
            }
        }
        for (case, pose) in poses {
            let value = guide.evaluate(pose)?;
            records.push(json!({"chart":name,"case":case,"pose":pose,"latent":value.latent,
                "in_reference_ball":value.in_reference_ball,"log_latent_density":value.log_latent_density,
                "log_physical_jacobian":value.log_physical_jacobian,
                "log_physical_density":value.log_physical_density.is_finite().then_some(value.log_physical_density),
                "zero_density":value.log_physical_density==f64::NEG_INFINITY}));
        }
    }
    fs::write(
        output.join("rust-reference.json"),
        serde_json::to_vec_pretty(&json!({
        "schema":"physical-latent-guide-math-reference-v1","shape_sha256":SHAPE,
        "physical_samples":0,"poisson_clouds":0,"records":records}))?,
    )?;
    Ok(())
}
