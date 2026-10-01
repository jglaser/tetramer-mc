//! Independent support/coordinate references for the world-pose wrapper.
use super::*;
use crate::geometry::Shape;
use rand::{RngExt, SeedableRng};
use serde_json::json;

const SHAPE: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

fn fixture(
    alpha: f64,
    beta: f64,
    axes: &[usize],
    transformed: bool,
) -> (Vec<u8>, Vec<u8>, DockingConfig, SphereTree) {
    let fixed = Pose {
        position: if transformed { [3., -2., 1.] } else { [0.; 3] },
        orientation: if transformed {
            quaternion(cayley([0.2, -0.3, 0.1]))
        } else {
            [1., 0., 0., 0.]
        },
    };
    let mut lower = [[0.; 6]; 6];
    for i in 0..6 {
        lower[i][i] = 1.;
    }
    if transformed {
        lower[3][0] = 0.2;
        lower[4][1] = -0.1;
        lower[0][0] = 1.3;
    }
    let covariance: [[f64; 6]; 6] = std::array::from_fn(|i| {
        std::array::from_fn(|j| (0..6).map(|k| lower[i][k] * lower[j][k]).sum())
    });
    let identity: [[f64; 6]; 6] =
        std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1. } else { 0. }));
    let region = json!({"shape_sha256":SHAPE,"fixed_neighbor":fixed,"physical_fixed_neighbors":[fixed],
        "capture_center":fixed.position,"capture_radius":1.5,"mahalanobis_radius":4.,"minimum_original_q":0.,
        "gaussian_chart":{"shape_sha256":SHAPE,"coordinate_convention":"anchor-body-relative","angular_length":1.,
        "anchors":[{"position":[0.,0.,0.],"rotation":IDENTITY}],"means":[[0.,0.,0.,0.,0.,0.]],"covariances":[covariance],"weights":[1.]}});
    let region_raw = serde_json::to_vec(&region).unwrap();
    let guide = json!({"schema":"defensive-hard-free-line-guide-v1","region_sha256":hash_bytes(&region_raw),
        "defensive_uniform_shell_probability":alpha,"conditional_probability":beta,"minimum_conditional_mass":1e-12,
        "raw_translation_axes":axes,"gaussian_components":[
            {"weight":0.4,"mean":[0.,0.,0.,0.,0.,0.],"covariance":identity},
            {"weight":0.6,"mean":[4.5,0.,0.,0.,0.,0.],"covariance":identity}]});
    let config:DockingConfig=serde_json::from_value(json!({"shape":"unused.json","fixed_poses":[fixed],
        "initial_pose":fixed,"capture_center":[0.,0.,0.],"capture_radius":20.,"depletant_radius":0.4,"reservoir_density":2.,"poisson_lambda_ratio":64.,
        "translation_steps":[0.1],"rotation_steps_deg":[1.],"rotation_probability":0.5,"local_attempts_per_cycle":1,
        "uniform_probability":0.1,"seed":1})).unwrap();
    let shape: Shape =
        serde_json::from_value(json!({"atoms":[{"center":[0.,0.,0.],"radius":0.3}]})).unwrap();
    (
        region_raw,
        serde_json::to_vec(&guide).unwrap(),
        config,
        SphereTree::new(shape).unwrap(),
    )
}
fn parsed(alpha: f64, beta: f64, axes: &[usize], transformed: bool) -> PhysicalLatentGuide {
    let (r, g, c, t) = fixture(alpha, beta, axes, transformed);
    PhysicalLatentGuide::from_bytes_with_geometry(&r, &g, SHAPE, &c, &t).unwrap()
}
fn gaussian_bytes(bytes: &[u8]) -> Vec<u8> {
    let mut value: Value = serde_json::from_slice(bytes).unwrap();
    value["schema"] = json!("defensive-latent-shell-guide-v1");
    for key in [
        "conditional_probability",
        "minimum_conditional_mass",
        "raw_translation_axes",
    ] {
        value.as_object_mut().unwrap().remove(key);
    }
    serde_json::to_vec(&value).unwrap()
}

#[test]
fn physical_hard_free_source_capture_and_context_are_separate_from_vessel() -> Result<()> {
    let (r, g, mut c, t) = fixture(0.5, 1., &[0], true);
    assert!(PhysicalLatentGuide::from_bytes(&r, &g, SHAPE).is_err());
    let a = PhysicalLatentGuide::from_bytes_with_geometry(&r, &g, SHAPE, &c, &t)?;
    c.capture_radius = 1000.;
    c.capture_center = [-99., 12., 40.];
    let b = PhysicalLatentGuide::from_bytes_with_geometry(&r, &g, SHAPE, &c, &t)?;
    assert_eq!(a.reference_capture(), ([3., -2., 1.], 1.5));
    for u in [[0.; 6], [1., 0., 0., 0.2, 0., 0.], [5., 0., 0., 0., 0., 0.]] {
        let (pose, j) = a.decode(u)?;
        let x = a.evaluate(pose)?;
        let y = b.evaluate(pose)?;
        assert_eq!(x.log_physical_density, y.log_physical_density);
        assert_eq!(x.hard_free_line_density, y.hard_free_line_density);
        if !x.structural_zero {
            assert!((x.log_physical_density - (x.log_latent_density.unwrap() - j)).abs() < 1e-12);
        }
    }
    c.fixed_poses[0].position[0] += 1.;
    assert!(PhysicalLatentGuide::from_bytes_with_geometry(&r, &g, SHAPE, &c, &t).is_err());
    Ok(())
}

#[test]
fn physical_hard_free_structural_zero_fallback_and_near_seam_are_distinct() -> Result<()> {
    let g = parsed(0.5, 1., &[0], false);
    let (p, _) = g.decode([5., 0., 0., 0., 0., 0.])?;
    let q = g.evaluate(p)?;
    assert!(q.structural_zero && q.log_physical_density == f64::NEG_INFINITY && q.latent.is_some());
    assert!(
        (half_mixture_log_density(-3., q.log_physical_density)? - (-3. - 2_f64.ln())).abs() < 1e-14
    );
    let (p, _) = g.decode([0., 0., 0., 5., 0., 0.])?;
    let q = g.evaluate(p)?;
    assert!(!q.structural_zero && q.log_physical_density.is_finite());
    assert_eq!(
        q.hard_free_line_density.as_ref().unwrap()["fallback_component_branches"],
        2
    );
    for scalar in [1e-6_f64, 1e-12, 1e-50] {
        let p = Pose {
            position: [2., 0., 0.],
            orientation: [scalar, (1. - scalar * scalar).sqrt(), 0., 0.],
        };
        assert!(g.evaluate(p)?.log_physical_density.is_finite());
    }
    let p = Pose {
        position: [2., 0., 0.],
        orientation: [1e-200, 1., 0., 0.],
    };
    assert!(g.evaluate(p).is_err());
    let p = Pose {
        orientation: [0., 1., 0., 0.],
        ..p
    };
    let q = g.evaluate(p)?;
    assert!(q.structural_zero && q.latent.is_none());
    Ok(())
}

#[test]
fn physical_hard_free_disabled_and_uniform_preserve_legacy_rng_and_density() -> Result<()> {
    for (alpha, beta) in [(0.5, 0.), (1., 1.)] {
        let (r, g, c, t) = fixture(alpha, beta, &[0, 1, 2], true);
        let new = PhysicalLatentGuide::from_bytes_with_geometry(&r, &g, SHAPE, &c, &t)?;
        let old = PhysicalLatentGuide::from_bytes(&r, &gaussian_bytes(&g), SHAPE)?;
        let mut a = StdRng::seed_from_u64(7711);
        let mut b = StdRng::seed_from_u64(7711);
        for _ in 0..128 {
            let x = new.draw_only(&mut a)?;
            let y = old.draw(&mut b)?;
            assert_eq!(x.latent, y.latent);
            assert_eq!(x.pose, y.pose);
            assert_eq!(x.gaussian_component, y.gaussian_component);
            assert_eq!(
                new.evaluate(x.pose)?.log_physical_density,
                old.evaluate(y.pose)?.log_physical_density
            );
        }
        assert_eq!(a.random::<u64>(), b.random::<u64>());
    }
    Ok(())
}

#[test]
fn physical_hard_free_axes_are_complete_arithmetic_mixture() -> Result<()> {
    let all = parsed(0.5, 1., &[0, 1, 2], false);
    let each = [
        parsed(0.5, 1., &[0], false),
        parsed(0.5, 1., &[1], false),
        parsed(0.5, 1., &[2], false),
    ];
    for u in [
        [5., 0., 0., 0., 0., 0.],
        [1., 0.2, 0.1, 0.1, 0., 0.],
        [0., 0., 0., 5., 0., 0.],
        [0.; 6],
    ] {
        let (pose, _) = all.decode(u)?;
        let expected = each
            .iter()
            .map(|g| g.log_density(pose))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .fold(f64::NEG_INFINITY, log_add)
            - 3_f64.ln();
        assert!((all.log_density(pose)? - expected).abs() < 1e-12);
    }
    Ok(())
}

#[test]
fn physical_hard_free_draw_only_retains_five_raw_coordinates_and_trace() -> Result<()> {
    let g = parsed(0.5, 1., &[0, 1, 2], true);
    let mut rng = StdRng::seed_from_u64(7712);
    let mut conditioned = 0;
    for _ in 0..256 {
        let draw = g.draw_only(&mut rng)?;
        let trace = draw.hard_free_line_draw.unwrap();
        if trace["conditional"] == true {
            conditioned += 1;
            let axis = trace["axis"].as_u64().unwrap() as usize;
            let original: [f64; 6] = serde_json::from_value(trace["original_latent"].clone())?;
            let a = g.chart.coordinates(original);
            let b = g.chart.coordinates(draw.latent);
            for i in 0..6 {
                if i != axis {
                    assert!((a[i] - b[i]).abs() < 1e-12);
                }
            }
        }
        let scored = g.evaluate(draw.pose)?;
        assert!(scored.log_physical_density.is_finite());
        if let PhysicalGuideLaw::HardFreeLine(line) = &g.guide {
            assert_eq!(trace, *line.last_draw.borrow());
        }
    }
    assert!(conditioned > 60);
    Ok(())
}

#[test]
fn physical_hard_free_conditional_global_normalization_including_fallback() -> Result<()> {
    let g = parsed(0.5, 1., &[0], false);
    let volume = PI.powi(3) * 4_f64.powi(6) / 6.;
    for angle in [0., 5.] {
        let integrate = |n: usize| -> Result<f64> {
            let mut integral = 0.;
            let cuts = [-16., -4., -1.5, -0.6, 0.6, 1.5, 4., 20.];
            for cut in cuts.windows(2) {
                let step = (cut[1] - cut[0]) / n as f64;
                for i in 0..n {
                    let u = [cut[0] + (i as f64 + 0.5) * step, 0., 0., angle, 0., 0.];
                    let (p, j) = g.decode(u)?;
                    integral += (g.log_density(p)? + j).exp() * step;
                }
            }
            Ok(integral)
        };
        // Split every discontinuity, then remove the midpoint O(h²) error
        // independently of the guide's Normal/CDF implementation.
        let coarse = integrate(4096)?;
        let fine = integrate(8192)?;
        let integral = (4. * fine - coarse) / 3.;
        let expected = if angle == 0. { 0.5 * 8. / volume } else { 0. }
            + 0.5 * (2. * PI).powf(-2.5) * (-0.5_f64 * angle * angle).exp();
        assert!(
            (integral - expected).abs() < 1e-11 * expected.max(1e-5),
            "{angle}: {integral} vs {expected}"
        );
    }
    Ok(())
}
