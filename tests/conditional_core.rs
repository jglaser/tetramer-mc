use rand::{SeedableRng, rngs::StdRng};
use tetramer_mc::{
    conditional::{
        ConditionalConfig, ConditionalEngine, ConditionalInitialization, ConditionalState,
        covariance_penalty, parameter_dimension,
    },
    math::{IDENTITY, Pose, cayley, quaternion},
    proposal::{FrozenRelativePoseProposal, GaussianComponentParameters},
};

const HASH: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn engine(config: ConditionalConfig) -> ConditionalEngine {
    ConditionalEngine::new(1., 30., HASH, config).unwrap()
}
fn config() -> ConditionalConfig {
    ConditionalConfig {
        k_max: 3,
        pair_cutoff_a: Some(15.),
        translation_scale_a: 2.,
        angular_length_a: 2.,
        ..Default::default()
    }
}
fn poses() -> Vec<Pose> {
    vec![
        Pose {
            position: [0.; 3],
            orientation: [1., 0., 0., 0.],
        },
        Pose {
            position: [2., 1., 0.5],
            orientation: quaternion(cayley([0.1, 0.2, 0.])),
        },
        Pose {
            position: [-1., -2., -0.2],
            orientation: quaternion(cayley([0.3, -0.1, 0.2])),
        },
    ]
}

#[test]
fn count_scores_are_normalized_and_empty_data_has_explicit_s_zero_law() {
    let mut c = config();
    c.covariance_exponent = 0.;
    c.basin_activity = 1.7;
    let engine = engine(c);
    let fit = engine.fit(&[]).unwrap();
    let total: f64 = fit.log_probabilities.iter().map(|p| p.exp()).sum();
    assert!((total - 1.).abs() < 1e-14);
    assert_eq!(fit.data_count, 0);
    assert_eq!(fit.covariance_penalties, vec![0.; 4]);
    let mut fact = 0.;
    for k in 0..=3 {
        if k > 0 {
            fact += (k as f64).ln();
        }
        assert!((fit.log_scores[k] - (k as f64 * 1.7_f64.ln() - fact)).abs() < 1e-14);
    }
}

#[test]
fn covariance_penalty_matches_volume_formula_and_disable_semantics() {
    let component = GaussianComponentParameters {
        anchor_position: [0.; 3],
        anchor_rotation: IDENTITY,
        mean: [0.; 6],
        covariance: std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 4. } else { 0. })),
        weight: 1.,
    };
    assert!((covariance_penalty(&[component.clone()], 1., 6.).unwrap() - 1. / 64.).abs() < 1e-14);
    assert_eq!(
        covariance_penalty(&[component.clone()], 1., 0.).unwrap(),
        0.
    );
    assert_eq!(covariance_penalty(&[], 1., 6.).unwrap(), 0.);
    assert_eq!(covariance_penalty(&[], 1., 0.).unwrap(), 0.);
    assert!((covariance_penalty(&[component], 2., 6.).unwrap() - 1.).abs() < 1e-14);
}

#[test]
fn fit_is_deterministic_uses_all_ordered_neighbors_and_handles_empty_clusters() {
    let engine = engine(config());
    let a = engine.fit(&poses()).unwrap();
    let b = engine.fit(&poses()).unwrap();
    assert_eq!(a.data_count, 6);
    assert_eq!(a.log_scores, b.log_scores);
    for k in 0..=3 {
        assert_eq!(a.counts[k].coordinates, b.counts[k].coordinates);
        assert_eq!(a.counts[k].fitted_components, b.counts[k].fitted_components);
    }
    for data in [vec![], vec![poses()[0]], poses()[..2].to_vec()] {
        let fit = engine.fit(&data).unwrap();
        assert!(fit.log_probabilities.iter().all(|x| x.is_finite()));
        for k in 0..=3 {
            let state = ConditionalState {
                k,
                eta: vec![0.; parameter_dimension(k)],
            };
            assert_eq!(engine.model(&fit, &state).unwrap().component_count(), k);
        }
    }
}

#[test]
fn k_zero_is_uniform_without_model_file_and_constructor_checks_shape() {
    let model = FrozenRelativePoseProposal::uniform_only_open([60.; 3], HASH).unwrap();
    assert_eq!(model.component_count(), 0);
    assert_eq!(model.uniform_weight(), 1.);
    assert!(!model.is_periodic());
    let poses = poses();
    let logp = model.log_density(&poses[0], &poses[1]).unwrap();
    assert!((logp + 3. * 60_f64.ln()).abs() < 1e-14);
    let mut rng = StdRng::seed_from_u64(123);
    let proposal = model.propose(&mut rng, &poses, 0).unwrap();
    assert!(proposal.candidate.is_some());
    assert_eq!(proposal.log_reverse_forward, Some(0.));
    assert!(
        FrozenRelativePoseProposal::from_components_open(
            vec![],
            1.,
            [60.; 3],
            1.,
            HASH,
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        )
        .is_err()
    );
}

#[test]
fn all_means_full_covariances_and_weights_fluctuate() {
    let engine = engine(config());
    let fit = engine.fit(&poses()).unwrap();
    let zero = ConditionalState {
        k: 2,
        eta: vec![0.; parameter_dimension(2)],
    };
    let nonzero = ConditionalState {
        k: 2,
        eta: vec![0.4; parameter_dimension(2)],
    };
    let a = engine.model(&fit, &zero).unwrap().component_parameters();
    let b = engine.model(&fit, &nonzero).unwrap().component_parameters();
    for k in 0..2 {
        assert_ne!(a[k].mean, b[k].mean);
        assert_ne!(a[k].covariance, b[k].covariance);
        assert_ne!(a[k].weight, b[k].weight);
        assert!((0..6).any(|i| (0..i).any(|j| b[k].covariance[i][j] != 0.)));
    }
    assert!((b.iter().map(|c| c.weight).sum::<f64>() - 1.).abs() < 1e-14);
}

#[test]
fn extreme_finite_residuals_decode_finite_spd_models() {
    let engine = engine(config());
    let fit = engine.fit(&poses()).unwrap();
    for sign in [-1., 1.] {
        let state = ConditionalState {
            k: 3,
            eta: vec![sign * 1e300; parameter_dimension(3)],
        };
        let model = engine.model(&fit, &state).unwrap();
        assert!(model.component_parameters().iter().all(|c| {
            c.mean
                .iter()
                .chain(c.covariance.iter().flatten())
                .all(|x| x.is_finite())
        }));
        assert!(
            model
                .log_density(&poses()[0], &poses()[1])
                .unwrap()
                .is_finite()
        );
    }
}

#[test]
fn fixed_eta_coordinate_density_cancels_but_reverse_density_must_refit() {
    let engine = engine(config());
    let x = poses();
    let mut y = x.clone();
    y[0].position = [7., -2., 1.];
    y[0].orientation = quaternion(cayley([0.7, -0.2, 0.1]));
    let fx = engine.fit(&x).unwrap();
    let fy = engine.fit(&y).unwrap();
    let state = ConditionalState {
        k: 2,
        eta: vec![0.2; parameter_dimension(2)],
    };
    let vx = fx
        .coordinates(&state, engine.config.residual_scale)
        .unwrap();
    let vy = fy
        .coordinates(&state, engine.config.residual_scale)
        .unwrap();
    let rx = fx
        .coordinate_log_density(2, &vx, engine.config.residual_scale)
        .unwrap();
    let ry = fy
        .coordinate_log_density(2, &vy, engine.config.residual_scale)
        .unwrap();
    assert!((rx - ry).abs() < 1e-11);
    let qx = engine.model(&fx, &state).unwrap();
    let qy = engine.model(&fy, &state).unwrap();
    let correct = qy.log_density(&x[0], &y[1]).unwrap() - qx.log_density(&y[0], &x[1]).unwrap();
    let stale = qx.log_density(&x[0], &y[1]).unwrap() - qx.log_density(&y[0], &x[1]).unwrap();
    assert!((correct - stale).abs() > 1e-5);
    let forward = correct + fy.log_probabilities[2] - fx.log_probabilities[2];
    let reverse = qx.log_density(&y[0], &x[1]).unwrap() - qy.log_density(&x[0], &y[1]).unwrap()
        + fx.log_probabilities[2]
        - fy.log_probabilities[2];
    assert!((forward + reverse).abs() < 1e-12);
}

#[test]
fn initializations_and_checkpoint_validation() {
    let mut rng = StdRng::seed_from_u64(9);
    for (init, k) in [
        (ConditionalInitialization::Zero, 0),
        (ConditionalInitialization::FixedKZero, 2),
    ] {
        let mut cfg = config();
        cfg.initialization = init;
        cfg.initial_k = Some(2);
        let engine = engine(cfg);
        let fit = engine.fit(&poses()).unwrap();
        let state = engine.initialize(&fit, &mut rng).unwrap();
        assert_eq!(state.k, k);
        assert!(state.eta.iter().all(|x| *x == 0.));
        let encoded = serde_json::to_string(&state).unwrap();
        let decoded: ConditionalState = serde_json::from_str(&encoded).unwrap();
        assert_eq!(state, decoded);
        decoded.validate(&engine.config).unwrap();
    }
    let mut c = config();
    c.initialization = ConditionalInitialization::FixedKZero;
    assert!(c.validate().is_err());
    assert!(
        ConditionalState {
            k: 0,
            eta: vec![0.]
        }
        .validate(&config())
        .is_err()
    );
    assert!(
        ConditionalState {
            k: 4,
            eta: vec![0.; parameter_dimension(4)]
        }
        .validate(&config())
        .is_err()
    );
}

#[test]
fn exact_pi_seam_still_has_finite_normalized_scores() {
    let engine = engine(config());
    let data = vec![
        Pose {
            position: [0.; 3],
            orientation: [1., 0., 0., 0.],
        },
        Pose {
            position: [1., 0., 0.],
            orientation: [0., 1., 0., 0.],
        },
        Pose {
            position: [0., 1., 0.],
            orientation: [0., 0., 1., 0.],
        },
    ];
    let fit = engine.fit(&data).unwrap();
    assert!(fit.log_scores.iter().all(|x| x.is_finite()));
    assert!((fit.log_probabilities.iter().map(|x| x.exp()).sum::<f64>() - 1.).abs() < 1e-13);
}

#[test]
fn defensive_cube_covers_allowed_displaced_body_origins() {
    let engine = engine(config());
    let fit = engine.fit(&[]).unwrap();
    let model = engine
        .model(&fit, &ConditionalState { k: 0, eta: vec![] })
        .unwrap();
    assert_eq!(model.box_lengths(), [62.; 3]);
    let anchor = Pose {
        position: [0.; 3],
        orientation: [1., 0., 0., 0.],
    };
    let displaced = Pose {
        position: [30.5, 0., 0.],
        orientation: anchor.orientation,
    };
    assert!(model.log_density(&displaced, &anchor).unwrap().is_finite());
}
