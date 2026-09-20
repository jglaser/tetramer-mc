use rand::{SeedableRng, rngs::StdRng};
use tetramer_mc::{
    atlas_transport::{
        AtlasInitialization, AtlasTransportConfig, AtlasTransportEngine, AtlasTransportState,
        parameter_dimension,
    },
    math::{IDENTITY, Pose, cayley, matmul, quaternion},
    proposal::{FrozenRelativePoseProposal, GaussianComponentParameters},
};

const HASH: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn tiny_atlas() -> FrozenRelativePoseProposal {
    let components = [0.7, 0.3]
        .into_iter()
        .enumerate()
        .map(|(k, weight)| GaussianComponentParameters {
            anchor_position: [0.; 3],
            anchor_rotation: IDENTITY,
            mean: [if k == 0 { 2. } else { -2. }, 0., 0., 0., 0., 0.],
            covariance: std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    if i == j {
                        if i < 3 { 0.0004 } else { 0.01 }
                    } else {
                        0.
                    }
                })
            }),
            weight,
        })
        .collect();
    FrozenRelativePoseProposal::from_components_open(components, 1., [20.; 3], 0.1, HASH, HASH)
        .unwrap()
}

fn protein_atlas() -> FrozenRelativePoseProposal {
    let raw = include_str!("../examples/frozen-relative-mixture.json");
    let value: serde_json::Value = serde_json::from_str(raw).unwrap();
    FrozenRelativePoseProposal::from_json_str_open(
        raw,
        [200.; 3],
        0.1,
        value["shape_sha256"].as_str().unwrap(),
    )
    .unwrap()
}

fn poses() -> Vec<Pose> {
    vec![
        Pose {
            position: [0.; 3],
            orientation: [1., 0., 0., 0.],
        },
        Pose {
            position: [2.025, 0.01, 0.],
            orientation: quaternion(cayley([0.01, 0., 0.])),
        },
    ]
}

fn atlas_mean_poses(model: &FrozenRelativePoseProposal) -> Vec<Pose> {
    let p = &model.component_parameters()[0];
    vec![
        Pose {
            position: [0.; 3],
            orientation: [1., 0., 0., 0.],
        },
        Pose {
            position: std::array::from_fn(|i| p.anchor_position[i] + p.mean[i]),
            orientation: quaternion(matmul(
                cayley(std::array::from_fn(|i| {
                    p.mean[i + 3] / model.angular_length()
                })),
                p.anchor_rotation,
            )),
        },
    ]
}

#[test]
fn zero_limit_preserves_actual_frozen_atlas_density_draws_and_refreshed_state() {
    let base = protein_atlas();
    let engine = AtlasTransportEngine::new(
        base.clone(),
        AtlasTransportConfig {
            mean_gain: 0.,
            covariance_gain: 0.,
            weight_gain: 0.,
            mean_noise: 0.,
            covariance_noise: 0.,
            weight_noise: 0.,
            ..Default::default()
        },
    )
    .unwrap();
    let poses = atlas_mean_poses(&base);
    let fit = engine.fit(&poses).unwrap();
    let mut state = engine
        .initialize(&fit, &mut StdRng::seed_from_u64(5))
        .unwrap();
    engine
        .refresh(&mut state, &mut StdRng::seed_from_u64(6))
        .unwrap();
    assert!(state.eta.iter().any(|v| *v != 0.));
    let model = engine.model(&fit, &state).unwrap();
    assert_eq!(base.component_parameters(), model.component_parameters());
    assert_eq!(
        base.log_density(&poses[1], &poses[0]).unwrap(),
        model.log_density(&poses[1], &poses[0]).unwrap()
    );
    let mut a = StdRng::seed_from_u64(55);
    let mut b = StdRng::seed_from_u64(55);
    for _ in 0..200 {
        let first = base.propose(&mut a, &poses, 1).unwrap();
        let second = model.propose(&mut b, &poses, 1).unwrap();
        assert_eq!(
            serde_json::to_value(first).unwrap(),
            serde_json::to_value(second).unwrap()
        );
    }
}

#[test]
fn reference_initialization_recovers_all_original_charts_and_narrow_covariances() {
    for base in [tiny_atlas(), protein_atlas()] {
        let engine =
            AtlasTransportEngine::new(base.clone(), AtlasTransportConfig::default()).unwrap();
        let poses = if base.component_count() == 2 {
            poses()
        } else {
            atlas_mean_poses(&base)
        };
        let fit = engine.fit(&poses).unwrap();
        let state = engine
            .initialize(&fit, &mut StdRng::seed_from_u64(8))
            .unwrap();
        let recovered = engine.model(&fit, &state).unwrap();
        for (old, new) in base
            .component_parameters()
            .iter()
            .zip(recovered.component_parameters())
        {
            assert_eq!(old.anchor_position, new.anchor_position);
            assert_eq!(old.anchor_rotation, new.anchor_rotation);
            assert!(
                old.mean
                    .iter()
                    .zip(new.mean)
                    .all(|(a, b)| (a - b).abs() < 2e-12 * (1. + a.abs()))
            );
            assert!(
                old.covariance
                    .iter()
                    .flatten()
                    .zip(new.covariance.iter().flatten())
                    .all(|(a, b)| (a - b).abs() < 2e-12 * (1. + a.abs()))
            );
            assert!((old.weight - new.weight).abs() < 2e-15);
        }
        let delta = recovered.log_density(&poses[1], &poses[0]).unwrap()
            - base.log_density(&poses[1], &poses[0]).unwrap();
        assert!(
            delta.abs() < 1e-8,
            "relative pose log-density mismatch {delta}"
        );
    }
}

#[test]
fn empty_data_stays_at_reference_and_geometry_fit_is_reproducible() {
    let engine = AtlasTransportEngine::new(tiny_atlas(), AtlasTransportConfig::default()).unwrap();
    for data in [vec![], vec![poses()[0]]] {
        let fit = engine.fit(&data).unwrap();
        assert_eq!(fit.data_count, 0);
        assert_eq!(fit.assigned_count, 0);
        assert!(fit.coordinates.iter().all(|v| *v == 0.));
    }
    let fit = engine.fit(&poses()).unwrap();
    assert_eq!(fit.data_count, 2);
    assert_eq!(fit.assigned_count, 2);
    assert_eq!(fit.component_counts, vec![1, 1]);
    assert_eq!(fit.coordinates, engine.fit(&poses()).unwrap().coordinates);
    assert!(fit.coordinates.iter().any(|v| *v != 0.));
    let remote = [
        poses()[0],
        Pose {
            position: [100.; 3],
            ..poses()[1]
        },
    ];
    assert_eq!(engine.fit(&remote).unwrap().assigned_count, 0);
}

#[test]
fn all_parameter_blocks_move_but_no_global_width_floor_is_introduced() {
    let base = tiny_atlas();
    let engine = AtlasTransportEngine::new(
        base.clone(),
        AtlasTransportConfig {
            mean_gain: 0.,
            covariance_gain: 0.,
            weight_gain: 0.,
            ..Default::default()
        },
    )
    .unwrap();
    let fit = engine.fit(&[]).unwrap();
    let state = AtlasTransportState {
        eta: vec![0.7; parameter_dimension(2)],
    };
    let changed = engine.model(&fit, &state).unwrap().component_parameters();
    let original = base.component_parameters();
    for k in 0..2 {
        assert_eq!(changed[k].anchor_position, original[k].anchor_position);
        assert_eq!(changed[k].anchor_rotation, original[k].anchor_rotation);
        assert_ne!(changed[k].mean, original[k].mean);
        assert_ne!(changed[k].covariance, original[k].covariance);
        assert_ne!(changed[k].weight, original[k].weight);
        assert!(changed[k].covariance[1][0] != 0.);
        // 0.02 A reference widths remain small, instead of gaining an absolute
        // 2 A floor as in the independent conditional-closure baseline.
        for d in 0..3 {
            assert!(changed[k].covariance[d][d].sqrt() < 0.04);
        }
    }
    assert!((changed.iter().map(|p| p.weight).sum::<f64>() - 1.).abs() < 1e-14);
}

#[test]
fn disabled_blocks_and_full_atlas_stay_valid_under_extreme_latents() {
    let base = protein_atlas();
    let engine = AtlasTransportEngine::new(
        base.clone(),
        AtlasTransportConfig {
            mean_gain: 0.,
            covariance_gain: 0.,
            weight_gain: 0.,
            covariance_noise: 0.,
            weight_noise: 0.,
            ..Default::default()
        },
    )
    .unwrap();
    let fit = engine.fit(&[]).unwrap();
    let state = AtlasTransportState {
        eta: vec![0.2; parameter_dimension(base.component_count())],
    };
    let changed = engine.model(&fit, &state).unwrap().component_parameters();
    for (old, new) in base.component_parameters().iter().zip(changed) {
        assert_ne!(old.mean, new.mean);
        for (a, b) in old
            .covariance
            .iter()
            .flatten()
            .zip(new.covariance.iter().flatten())
        {
            assert!((a - b).abs() < 1e-12 * (1. + a.abs()));
        }
        assert!((old.weight - new.weight).abs() < 1e-15);
    }
    let engine = AtlasTransportEngine::new(base.clone(), AtlasTransportConfig::default()).unwrap();
    for magnitude in [1., -1., 1e200, -1e200] {
        let state = AtlasTransportState {
            eta: vec![magnitude; parameter_dimension(base.component_count())],
        };
        let model = engine.model(&fit, &state).unwrap();
        assert_eq!(model.component_count(), 28);
        assert!(model.component_parameters().iter().all(|p| {
            p.mean
                .iter()
                .chain(p.covariance.iter().flatten())
                .all(|x| x.is_finite())
        }));
        let poses = atlas_mean_poses(&base);
        assert!(model.log_density(&poses[1], &poses[0]).unwrap().is_finite());
    }
}

#[test]
fn initialization_and_auxiliary_validation_are_explicit() {
    let base = tiny_atlas();
    let mut config = AtlasTransportConfig {
        mean_noise: 0.,
        ..Default::default()
    };
    let engine = AtlasTransportEngine::new(base.clone(), config.clone()).unwrap();
    let fit = engine.fit(&poses()).unwrap();
    assert!(
        engine
            .initialize(&fit, &mut StdRng::seed_from_u64(4))
            .is_err()
    );
    config.initialization = AtlasInitialization::Random;
    let engine = AtlasTransportEngine::new(base, config).unwrap();
    let first = engine
        .initialize(&fit, &mut StdRng::seed_from_u64(4))
        .unwrap();
    let second = engine
        .initialize(&fit, &mut StdRng::seed_from_u64(4))
        .unwrap();
    assert_eq!(first, second);
    assert_eq!(first.eta.len(), 55);
    assert!(
        AtlasTransportState { eta: vec![0.; 54] }
            .validate(2)
            .is_err()
    );
    assert!(
        AtlasTransportState {
            eta: vec![f64::NAN; 55]
        }
        .validate(2)
        .is_err()
    );
    let mut invalid = AtlasTransportConfig::default();
    invalid.shrinkage = 0.;
    assert!(invalid.validate().is_err());
}
