//! Independent physical-density checks for exact shared-chart initialization.
use tetramer_mc::{
    conditional::{ConditionalConfig, ConditionalEngine, ConditionalState},
    math::{Pose, cayley, quaternion},
    proposal::{FrozenRelativePoseProposal, GaussianComponentParameters},
};

const SHA: &str = "0000000000000000000000000000000000000000000000000000000000000000";

fn setup() -> (ConditionalEngine, Vec<Pose>) {
    let config = ConditionalConfig {
        k_max: 3,
        pair_cutoff_a: Some(12.),
        ..Default::default()
    };
    let engine = ConditionalEngine::new(1., 20., SHA, config).unwrap();
    let poses = vec![
        Pose {
            position: [0., 0., 0.],
            orientation: [1., 0., 0., 0.],
        },
        Pose {
            position: [3., 1., 0.],
            orientation: [1., 0., 0., 0.],
        },
        Pose {
            position: [-2., 3., 1.],
            orientation: [1., 0., 0., 0.],
        },
    ];
    (engine, poses)
}

fn learned_parameters(
    mut components: Vec<GaussianComponentParameters>,
) -> Vec<GaussianComponentParameters> {
    for (k, component) in components.iter_mut().enumerate() {
        component.weight = [0.2, 0.3, 0.5][k];
        component.mean = std::array::from_fn(|d| (k as f64 - 1.) * (d as f64 + 1.) * 0.15);
        // Coupled full covariance independently constructed in proposal units.
        let lower: [[f64; 6]; 6] = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                if i == j {
                    8. + i as f64 + k as f64
                } else if j < i {
                    0.3 * (1. + (i - j) as f64)
                } else {
                    0.
                }
            })
        });
        component.covariance = std::array::from_fn(|i| {
            std::array::from_fn(|j| (0..6).map(|d| lower[i][d] * lower[j][d]).sum())
        });
    }
    components
}

#[test]
fn shared_chart_import_recovers_full_learned_density_and_transports_residual() {
    let (engine, poses) = setup();
    let fit = engine.fit(&poses).unwrap();
    let parameters = learned_parameters(fit.counts[3].fitted_components.clone());
    let imported = FrozenRelativePoseProposal::from_components_open(
        parameters,
        engine.config.angular_length_a,
        [42.; 3],
        engine.config.uniform_weight,
        SHA,
        SHA,
    )
    .unwrap();
    let state = engine.state_from_model(&fit, &imported).unwrap();
    assert_eq!(state.k, 3);
    assert_eq!(state.eta.len(), 83);
    assert!(state.eta.iter().any(|x| x.abs() > 0.1));
    let recovered = engine.model(&fit, &state).unwrap();
    for n in 0..40 {
        let x = n as f64 / 10.;
        let t = [x - 2., 0.3 * x, 1. - 0.2 * x];
        let r = cayley([0.05 * x, -0.03 * x, 0.02 * x]);
        let expected = imported.relative_log_density(t, r).unwrap();
        let actual = recovered.relative_log_density(t, r).unwrap();
        assert!(
            (expected - actual).abs() < 2e-12,
            "{n}: {expected} versus {actual}"
        );
        let moving = Pose {
            position: t,
            orientation: quaternion(r),
        };
        let expected = imported.log_density(&moving, &poses[0]).unwrap();
        let actual = recovered.log_density(&moving, &poses[0]).unwrap();
        assert!(
            (expected - actual).abs() < 2e-12,
            "full proposal {n}: {expected} versus {actual}"
        );
    }
    let imported_components = imported.component_parameters();
    let recovered_components = recovered.component_parameters();
    for (a, b) in imported_components.iter().zip(&recovered_components) {
        assert!((a.weight - b.weight).abs() < 1e-14);
        for i in 0..6 {
            assert!((a.mean[i] - b.mean[i]).abs() < 1e-12);
            for j in 0..6 {
                assert!((a.covariance[i][j] - b.covariance[i][j]).abs() < 2e-11);
            }
        }
    }
    // Future reconstruction uses current X and retained eta; it does not
    // retain the imported law as a permanent hidden fitting reference.
    let mut next = poses;
    next[1].position[0] += 0.7;
    let next_fit = engine.fit(&next).unwrap();
    let before = fit
        .coordinates(&state, engine.config.residual_scale)
        .unwrap();
    let after = next_fit
        .coordinates(&state, engine.config.residual_scale)
        .unwrap();
    for i in 0..state.eta.len() {
        let old_residual =
            (before[i] - fit.counts[3].coordinates[i]) / engine.config.residual_scale;
        let new_residual =
            (after[i] - next_fit.counts[3].coordinates[i]) / engine.config.residual_scale;
        assert!((old_residual - new_residual).abs() < 2e-13);
    }
    engine.model(&next_fit, &state).unwrap();
}

#[test]
fn import_rejects_incompatible_charts_invalid_covariances_and_coordinate_bounds() {
    let (engine, poses) = setup();
    let fit = engine.fit(&poses).unwrap();
    let source = learned_parameters(fit.counts[3].fitted_components.clone());
    let check = |components: Vec<GaussianComponentParameters>| {
        assert!(engine.state_from_components(&fit, &components).is_err());
    };
    let mut changed = source.clone();
    changed[0].anchor_position[0] += 1.;
    check(changed);
    let mut changed = source.clone();
    changed[0].anchor_rotation = cayley([0.01, 0., 0.]);
    check(changed);
    let mut changed = source.clone();
    changed[0].covariance = [[0.; 6]; 6];
    check(changed);
    let mut changed = source.clone();
    changed[0].covariance[0][1] += 1.;
    check(changed);
    let mut changed = source.clone();
    let ridge_variance =
        engine.config.covariance_floor / 16. * engine.config.translation_scale_a.powi(2);
    changed[0].covariance =
        std::array::from_fn(|i| std::array::from_fn(|j| if i == j { ridge_variance } else { 0. }));
    check(changed);
    let mut changed = source.clone();
    changed[0].mean[0] = 64. * engine.config.translation_scale_a;
    check(changed);
    let mut changed = source.clone();
    changed[0].weight = 0.;
    check(changed);
    let mut changed = source.clone();
    changed[0].weight += 0.1;
    check(changed);
    let mut changed = source.clone();
    changed.push(source[0].clone());
    check(changed);
    let mut malformed_fit = fit.clone();
    malformed_fit.counts.truncate(1);
    assert!(
        engine
            .state_from_components(&malformed_fit, &source)
            .is_err()
    );

    let zero = engine.state_from_components(&fit, &[]).unwrap();
    assert_eq!(zero, ConditionalState { k: 0, eta: vec![] });
    assert_eq!(engine.model(&fit, &zero).unwrap().component_count(), 0);
    let density = engine
        .model(&fit, &zero)
        .unwrap()
        .log_density(&poses[0], &poses[1])
        .unwrap();
    assert!((density + 3. * 42_f64.ln()).abs() < 1e-14);
}

#[test]
fn model_import_rejects_source_shape_metric_and_periodic_metadata_mismatch() {
    let (engine, poses) = setup();
    let fit = engine.fit(&poses).unwrap();
    let source = learned_parameters(fit.counts[3].fitted_components.clone());
    let wrong_sha = "1111111111111111111111111111111111111111111111111111111111111111";
    for (angular_length, shape) in [
        (engine.config.angular_length_a * 2., SHA),
        (engine.config.angular_length_a, wrong_sha),
    ] {
        let imported = FrozenRelativePoseProposal::from_components_open(
            source.clone(),
            angular_length,
            [42.; 3],
            engine.config.uniform_weight,
            shape,
            shape,
        )
        .unwrap();
        assert!(engine.state_from_model(&fit, &imported).is_err());
    }
    for (cube, uniform_weight) in [([40.; 3], engine.config.uniform_weight), ([42.; 3], 0.2)] {
        let imported = FrozenRelativePoseProposal::from_components_open(
            source.clone(),
            engine.config.angular_length_a,
            cube,
            uniform_weight,
            SHA,
            SHA,
        )
        .unwrap();
        assert!(engine.state_from_model(&fit, &imported).is_err());
    }
    let json = serde_json::json!({
        "angular_length":engine.config.angular_length_a,
        "shape_sha256":SHA,
        "coordinate_convention":"anchor-body-relative",
        "anchors":source.iter().map(|c| serde_json::json!({
            "position":c.anchor_position,"rotation":c.anchor_rotation})).collect::<Vec<_>>(),
        "means":source.iter().map(|c| c.mean).collect::<Vec<_>>(),
        "covariances":source.iter().map(|c| c.covariance).collect::<Vec<_>>(),
        "weights":source.iter().map(|c| c.weight).collect::<Vec<_>>()
    });
    let periodic = FrozenRelativePoseProposal::from_json_str(
        &json.to_string(),
        [42.; 3],
        engine.config.uniform_weight,
        SHA,
    )
    .unwrap();
    assert!(engine.state_from_model(&fit, &periodic).is_err());
}
