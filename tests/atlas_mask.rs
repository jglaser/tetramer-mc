use rand::{RngExt, SeedableRng, rngs::StdRng};
use tetramer_mc::atlas_mask::{
    AtlasMaskConfig, AtlasMaskEngine, AtlasMaskLabelLaw, AtlasMaskState,
};

fn state(bits: usize, j: usize) -> AtlasMaskState {
    AtlasMaskState {
        labels: (0..j).filter(|&i| bits & (1 << i) != 0).collect(),
    }
}

fn poisson_counts(j: usize, min: usize, max: usize, activity: f64) -> Vec<f64> {
    let mut values = vec![0.; j + 1];
    let mut value = 1.;
    for (k, slot) in values.iter_mut().enumerate() {
        if k > 0 {
            value *= activity / k as f64;
        }
        if (min..=max).contains(&k) {
            *slot = value;
        }
    }
    let total: f64 = values.iter().sum();
    values.iter_mut().for_each(|v| *v /= total);
    values
}

#[test]
fn enumerated_subsets_normalize_separately_at_every_cardinality() {
    for j in 0usize..=6 {
        let weights: Vec<_> = (0..j).map(|i| (i + 1).pow(2) as f64).collect();
        for law in [AtlasMaskLabelLaw::Uniform, AtlasMaskLabelLaw::AtlasWeight] {
            for min in 0..=j {
                let max = j;
                let engine = AtlasMaskEngine::new(
                    &weights,
                    AtlasMaskConfig {
                        activity: 2.3,
                        min_components: min,
                        max_components: Some(max),
                        label_law: law,
                        ..Default::default()
                    },
                )
                .unwrap();
                let expected = poisson_counts(j, min, max, 2.3);
                let mut measured = vec![0.; j + 1];
                for bits in 0..1 << j {
                    let mask = state(bits, j);
                    if mask.labels.len() < min {
                        assert!(engine.log_probability(&mask).is_err());
                    } else {
                        measured[mask.labels.len()] += engine.log_probability(&mask).unwrap().exp();
                    }
                }
                for k in 0..=j {
                    assert!((measured[k] - expected[k]).abs() < 5e-14);
                    assert!(
                        (engine.log_count_probabilities()[k].exp() - expected[k]).abs() < 5e-14
                    );
                }
                assert!((measured.iter().sum::<f64>() - 1.).abs() < 5e-14);
            }
        }
    }
}

#[test]
fn fixed_cardinality_weights_match_products_and_exact_subset_samples() {
    let weights = [1., 2., 3., 5.];
    let engine = AtlasMaskEngine::new(
        &weights,
        AtlasMaskConfig {
            min_components: 2,
            max_components: Some(2),
            ..Default::default()
        },
    )
    .unwrap();
    let denominator: f64 = (0..weights.len())
        .flat_map(|i| (i + 1..weights.len()).map(move |j| weights[i] * weights[j]))
        .sum();
    let mut expected = [0.; 16];
    for (bits, probability) in expected.iter_mut().enumerate() {
        let mask = state(bits, weights.len());
        if mask.labels.len() == 2 {
            *probability = mask.labels.iter().map(|&i| weights[i]).product::<f64>() / denominator;
            assert!((engine.log_probability(&mask).unwrap().exp() - *probability).abs() < 1e-14);
        }
    }
    let samples = 100_000;
    let mut rng = StdRng::seed_from_u64(9981);
    let mut counts = [0usize; 16];
    let mut mask = engine.initialize(&mut rng).unwrap();
    for _ in 0..samples {
        engine.refresh(&mut mask, &mut rng).unwrap();
        let bits = mask.labels.iter().map(|&i| 1 << i).sum::<usize>();
        counts[bits] += 1;
    }
    for bits in 0..16 {
        let observed = counts[bits] as f64 / samples as f64;
        let probability = expected[bits];
        let sigma = (probability * (1. - probability) / samples as f64).sqrt();
        assert!((observed - probability).abs() < 6. * sigma + 1. / samples as f64);
    }
}

#[test]
fn count_law_and_empirical_mean_are_independent_of_label_weight_bias() {
    let config = AtlasMaskConfig {
        activity: 2.3,
        min_components: 1,
        max_components: Some(5),
        initial_full: false,
        ..Default::default()
    };
    let weighted = AtlasMaskEngine::new(&[1e-30, 1., 2., 8., 100., 1e30], config.clone()).unwrap();
    let uniform = AtlasMaskEngine::new(
        &[1.; 6],
        AtlasMaskConfig {
            label_law: AtlasMaskLabelLaw::Uniform,
            ..config
        },
    )
    .unwrap();
    assert_eq!(
        weighted.log_count_probabilities(),
        uniform.log_count_probabilities()
    );
    let expected = poisson_counts(6, 1, 5, 2.3);
    let mean: f64 = expected.iter().enumerate().map(|(k, p)| k as f64 * p).sum();
    let variance: f64 = expected
        .iter()
        .enumerate()
        .map(|(k, p)| (k as f64 - mean).powi(2) * p)
        .sum();
    let samples = 30_000;
    let mut sum = 0.;
    let mut rng = StdRng::seed_from_u64(8234);
    let mut mask = weighted.initialize(&mut rng).unwrap();
    for _ in 0..samples {
        weighted.refresh(&mut mask, &mut rng).unwrap();
        sum += mask.labels.len() as f64;
    }
    assert!((sum / samples as f64 - mean).abs() < 6. * (variance / samples as f64).sqrt());
}

#[test]
fn common_weight_scaling_cancels_and_uniform_law_ignores_relative_weights() {
    let config = AtlasMaskConfig {
        min_components: 2,
        max_components: Some(2),
        ..Default::default()
    };
    let first = AtlasMaskEngine::new(&[1., 2., 3., 4.], config.clone()).unwrap();
    let scaled = AtlasMaskEngine::new(&[1e-200, 2e-200, 3e-200, 4e-200], config.clone()).unwrap();
    let uniform = AtlasMaskEngine::new(
        &[1e-200, 2e-100, 1., 1e100],
        AtlasMaskConfig {
            label_law: AtlasMaskLabelLaw::Uniform,
            ..config
        },
    )
    .unwrap();
    for bits in 0..16 {
        let mask = state(bits, 4);
        if mask.labels.len() == 2 {
            assert!(
                (first.log_probability(&mask).unwrap() - scaled.log_probability(&mask).unwrap())
                    .abs()
                    < 1e-12
            );
            assert!((uniform.log_probability(&mask).unwrap().exp() - 1. / 6.).abs() < 1e-14);
        }
    }
}

#[test]
fn empty_and_full_fixed_supports_are_exact_and_consume_no_random_numbers() {
    for j in [0, 1, 6] {
        for k in [0, j] {
            let engine = AtlasMaskEngine::new(
                &vec![0.2; j],
                AtlasMaskConfig {
                    min_components: k,
                    max_components: Some(k),
                    initial_full: false,
                    ..Default::default()
                },
            )
            .unwrap();
            let mut rng = StdRng::seed_from_u64(141);
            let mut control = StdRng::seed_from_u64(141);
            let mut mask = engine.initialize(&mut rng).unwrap();
            for _ in 0..5 {
                assert_eq!(mask.labels, (0..k).collect::<Vec<_>>());
                assert_eq!(engine.log_probability(&mask).unwrap(), 0.);
                engine.refresh(&mut mask, &mut rng).unwrap();
            }
            assert_eq!(rng.random::<u64>(), control.random::<u64>());
        }
    }
}

#[test]
fn prepared_full_initialization_requires_support_and_is_not_a_gibbs_draw() {
    let engine = AtlasMaskEngine::new(&[1., 2., 3., 4.], Default::default()).unwrap();
    let mut rng = StdRng::seed_from_u64(55);
    let mut control = StdRng::seed_from_u64(55);
    let mask = engine.initialize(&mut rng).unwrap();
    assert_eq!(mask.labels, vec![0, 1, 2, 3]);
    assert!(engine.log_probability(&mask).unwrap() < 0.);
    assert_eq!(rng.random::<u64>(), control.random::<u64>());
    let restricted = AtlasMaskEngine::new(
        &[1., 2., 3., 4.],
        AtlasMaskConfig {
            max_components: Some(2),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(restricted.initialize(&mut rng).unwrap().labels.len() <= 2);
}

#[test]
fn invalid_configuration_and_checkpoint_states_are_rejected() {
    for activity in [0., -1., f64::NAN, f64::INFINITY] {
        assert!(
            AtlasMaskEngine::new(
                &[1., 2.],
                AtlasMaskConfig {
                    activity,
                    ..Default::default()
                }
            )
            .is_err()
        );
    }
    for weights in [vec![0., 1.], vec![-1.], vec![f64::NAN], vec![f64::INFINITY]] {
        assert!(AtlasMaskEngine::new(&weights, Default::default()).is_err());
    }
    for config in [
        AtlasMaskConfig {
            min_components: 2,
            max_components: Some(1),
            ..Default::default()
        },
        AtlasMaskConfig {
            max_components: Some(3),
            ..Default::default()
        },
        AtlasMaskConfig {
            min_components: 3,
            ..Default::default()
        },
        AtlasMaskConfig {
            refresh_probability: -0.1,
            ..Default::default()
        },
        AtlasMaskConfig {
            refresh_probability: f64::NAN,
            ..Default::default()
        },
    ] {
        assert!(AtlasMaskEngine::new(&[1., 2.], config).is_err());
    }
    let engine = AtlasMaskEngine::new(
        &[1., 2.],
        AtlasMaskConfig {
            min_components: 1,
            ..Default::default()
        },
    )
    .unwrap();
    for labels in [vec![], vec![2], vec![0, 0], vec![1, 0]] {
        let mask = AtlasMaskState { labels };
        assert!(engine.validate_state(&mask).is_err());
        assert!(engine.log_probability(&mask).is_err());
    }
}

#[test]
fn fixed_count_masks_require_reverse_coverage_despite_exact_equilibrium_balance() {
    // Equal-probability physical basins A,B. A retained one-component mask
    // imposes componentwise MH, which mixes much more slowly than MH with the
    // full mixture even though both obey detailed balance exactly.
    let q = [[0.99_f64, 0.01_f64], [0.01_f64, 0.99_f64]];
    let transition = |from: usize, to: usize| {
        q.iter()
            .map(|c| 0.5 * c[to] * (c[from] / c[to]).min(1.))
            .sum::<f64>()
    };
    let ab = transition(0, 1);
    let ba = transition(1, 0);
    assert!((ab - 0.01).abs() < 1e-15);
    assert_eq!(ab, ba);
    let full = 0.5;
    assert!((full / ab - 50.).abs() < 1e-12);
}
