//! Geometry, proposal-density and transactional checks for the conditional-axis
//! wrapper. Physical stationarity is tested independently in the companion file.
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::f64::consts::PI;
use tetramer_mc::{
    conditional_axis::{ConditionalAxis, ConditionalAxisConfig},
    geometry::{Atom, Shape, SphereTree},
    math::*,
    spherical::{self, Container, HalfTurn},
};
fn sphere() -> SphereTree {
    SphereTree::new(Shape {
        name: "unit sphere".into(),
        volume: 4. * PI / 3.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius: 1.,
        }],
    })
    .unwrap()
}
fn pose(position: Vec3) -> Pose {
    Pose {
        position,
        orientation: [1., 0., 0., 0.],
    }
}
fn near(a: f64, b: f64, tolerance: f64) {
    assert!((a - b).abs() <= tolerance, "{a} != {b} (+/- {tolerance})");
}
fn axis(z: f64, phi: f64) -> Vec3 {
    let r = (1. - z * z).sqrt();
    [r * phi.cos(), r * phi.sin(), z]
}

#[test]
fn configuration_defaults_validation_and_unknown_fields() {
    let c: ConditionalAxisConfig = serde_json::from_str("{}").unwrap();
    assert_eq!(c.score_floor, 0.02);
    assert_eq!(c.max_candidates, 64);
    assert_eq!(c.uniform_axis_weight, 0.25);
    assert!(serde_json::from_str::<ConditionalAxisConfig>(r#"{"misspelling":1}"#).is_err());
    for value in [0., -1., f64::NAN, f64::INFINITY, 1.1] {
        assert!(
            ConditionalAxisConfig {
                score_floor: value,
                ..c
            }
            .validate()
            .is_err()
        );
        assert!(
            ConditionalAxisConfig {
                uniform_axis_weight: value,
                ..c
            }
            .validate()
            .is_err()
        );
    }
    assert!(
        ConditionalAxisConfig {
            max_candidates: 0,
            ..c
        }
        .validate()
        .is_err()
    );
}

#[test]
fn full_sphere_band_jacobian_is_normalized_and_axis_even() {
    let tree = sphere();
    let sampler = ConditionalAxis::new(&tree, 0.5, ConditionalAxisConfig::default()).unwrap();
    let base = sampler
        .base(&[pose([0., 0., 2.]), pose([0.; 3])], 0)
        .unwrap();
    near(base.band_area(), 4. * PI, 1e-14);
    for k in 0..101 {
        let u = axis((k as f64 + 0.5) / 101., 0.73);
        near(base.density(u).unwrap(), 0.25 + 1.5 * u[2].abs(), 1e-14);
        near(
            base.density(u).unwrap(),
            base.density(scale(u, -1.)).unwrap(),
            1e-14,
        );
    }
    let integral = (0..1000)
        .map(|k| base.density(axis((k as f64 + 0.5) / 1000., 0.)).unwrap())
        .sum::<f64>()
        / 1000.;
    near(integral, 1., 1e-14);
}

#[test]
fn overlapping_bands_count_the_complete_mixture() {
    let tree = sphere();
    let sampler = ConditionalAxis::new(&tree, 0.5, ConditionalAxisConfig::default()).unwrap();
    let state = [pose([0., 0., 4.]), pose([4., 0., 0.])];
    let single = sampler.base(&state, 0).unwrap();
    let double = sampler.base(&[state[0], state[1], state[1]], 0).unwrap();
    let u = [2_f64.sqrt().recip(), 0., 2_f64.sqrt().recip()];
    near(single.band_area(), 2. * PI * (1. - 23. / 32.), 1e-14);
    near(double.band_area(), 2. * single.band_area(), 1e-14);
    assert_eq!(single.multiplicity(u).unwrap(), 1);
    assert_eq!(double.multiplicity(u).unwrap(), 2);
    near(
        single.density(u).unwrap(),
        double.density(u).unwrap(),
        1e-14,
    );
    for k in 0..100 {
        let v = axis(-1. + 2. * (k as f64 + 0.5) / 100., k as f64);
        near(
            single.density(v).unwrap(),
            double.density(v).unwrap(),
            1e-13,
        );
    }
}

#[test]
fn cap_density_normalization_and_draw_moments_agree() {
    let tree = sphere();
    let sampler = ConditionalAxis::new(&tree, 0.5, ConditionalAxisConfig::default()).unwrap();
    let base = sampler
        .base(&[pose([0., 0., 4.]), pose([4., 0., 0.])], 0)
        .unwrap();
    // Uniform solid-angle quadrature independently reconstructs normalization.
    let mut integral = 0.;
    for iz in 0..400 {
        for iphi in 0..800 {
            integral += base
                .density(axis(
                    -1. + 2. * (iz as f64 + 0.5) / 400.,
                    2. * PI * (iphi as f64 + 0.5) / 800.,
                ))
                .unwrap();
        }
    }
    near(integral / 320000., 1., 0.002);
    let mut rng = StdRng::seed_from_u64(135);
    let mut mean = [0.; 3];
    for _ in 0..40000 {
        let u = base.draw(&mut rng).unwrap();
        near(norm(u), 1., 5e-15);
        let endpoint = HalfTurn::new(u).unwrap().point([0., 0., 1.]);
        mean = add(mean, scale(endpoint, 1. / 40000.));
    }
    // Uniform axes give E[T r]=-r/3; the cap part has mean (1+a)/2 along x.
    near(mean[0], 0.75 * (1. + 23. / 32.) / 2., 0.012);
    near(mean[1], 0., 0.012);
    near(mean[2], -0.25 / 3., 0.012);
}

#[test]
fn empty_centered_and_tangent_bands_have_defined_limits() {
    let tree = sphere();
    let sampler = ConditionalAxis::new(&tree, 0.5, ConditionalAxisConfig::default()).unwrap();
    for state in [
        vec![pose([0.; 3]), pose([4., 0., 0.])],
        vec![pose([4., 0., 0.]), pose([10., 0., 0.])],
        vec![pose([4., 0., 0.]), pose([7., 0., 0.])],
        vec![pose([4., 0., 0.])],
    ] {
        let base = sampler.base(&state, 0).unwrap();
        assert!(base.is_uniform());
        near(base.density([1., 0., 0.]).unwrap(), 1., 0.);
    }
    // A centered neighbor at exactly the cutoff covers every direction.
    let full = sampler
        .base(&[pose([3., 0., 0.]), pose([0.; 3])], 0)
        .unwrap();
    near(full.band_area(), 4. * PI, 1e-14);
    assert!(full.density([0.; 3]).is_err());
}

#[test]
fn contact_score_detects_exchange_and_rejects_core_collision() {
    let tree = sphere();
    let sampler = ConditionalAxis::new(&tree, 0.5, ConditionalAxisConfig::default()).unwrap();
    let state = [
        pose([4., 0., 0.]),
        pose([4., 2.2, 0.]),
        pose([-4., 2.2, 0.]),
    ];
    assert_eq!(sampler.contacts(&state, 0).unwrap(), vec![1]);
    let score = sampler.score(&state, 0, [0., 0., 1.]).unwrap();
    assert!(score.predicate);
    assert_eq!(score.value, 1.);
    assert_eq!(score.hard_valid, Some(true));
    assert_eq!(score.lost, vec![1]);
    assert_eq!(score.gained, vec![2]);
    let mut invalid = state;
    invalid[2].position[1] = 1.9;
    let collision = sampler.score(&invalid, 0, [0., 0., 1.]).unwrap();
    assert_eq!(collision.hard_valid, Some(false));
    assert!(!collision.predicate);
    assert_eq!(collision.value, 0.02);
    let transformed: Vec<_> = state
        .iter()
        .map(|p| HalfTurn::new([0., 0., 1.]).unwrap().apply(*p))
        .collect();
    for tag in 0..3 {
        assert_eq!(
            sampler.contacts(&state, tag).unwrap(),
            sampler.contacts(&transformed, tag).unwrap()
        );
    }
}

#[test]
fn partial_half_turn_preserves_band_areas_without_assuming_density_symmetry() {
    let tree = sphere();
    let sampler = ConditionalAxis::new(&tree, 0.5, ConditionalAxisConfig::default()).unwrap();
    let state = [pose([0., 0., 4.]), pose([4., 0., 0.]), pose([0., 3., 2.])];
    let turn = HalfTurn::new([1., 1., 1.]).unwrap();
    let mut changed = state;
    changed[0] = turn.apply(changed[0]);
    changed[2] = turn.apply(changed[2]);
    for tag in 0..3 {
        near(
            sampler.base(&state, tag).unwrap().band_area(),
            sampler.base(&changed, tag).unwrap().band_area(),
            1e-13,
        );
    }
}

#[test]
fn uniform_limit_replays_the_unchanged_physical_kernel() {
    let tree = sphere();
    let wall = Container::new(15., &tree).unwrap();
    let sampler = ConditionalAxis::new(
        &tree,
        0.5,
        ConditionalAxisConfig {
            score_floor: 1.,
            uniform_axis_weight: 1.,
            ..Default::default()
        },
    )
    .unwrap();
    let initial = vec![
        pose([4., 0., 0.]),
        pose([4., 2.2, 0.]),
        pose([-4., 2.2, 0.]),
    ];
    for seed in 0..24 {
        let mut guided = initial.clone();
        let mut control = initial.clone();
        let mut guide_rng = StdRng::seed_from_u64(seed);
        let mut physical_rng = StdRng::seed_from_u64(seed + 1000);
        let mut control_rng = StdRng::seed_from_u64(seed + 1000);
        let outcome = sampler
            .update(&wall, &mut guided, 0.3, &mut guide_rng, &mut physical_rng)
            .unwrap();
        assert!(outcome.accepted);
        assert_eq!(outcome.stats.log_acceptance, Some(0.));
        assert_eq!(outcome.stats.forward.attempts, 1);
        assert_eq!(outcome.stats.reverse.attempts, 1);
        spherical::update(
            &tree,
            &wall,
            &mut control,
            HalfTurn::new(outcome.stats.forward.selected_axis.unwrap()).unwrap(),
            0.5,
            0.3,
            &mut control_rng,
        )
        .unwrap();
        assert_eq!(guided, control);
        assert_eq!(physical_rng.random::<u64>(), control_rng.random::<u64>());
    }
}

#[test]
fn forward_cap_failure_retains_state_and_physical_rng() {
    let tree = sphere();
    let wall = Container::new(15., &tree).unwrap();
    let sampler = ConditionalAxis::new(
        &tree,
        0.,
        ConditionalAxisConfig {
            score_floor: 1e-100,
            max_candidates: 1,
            uniform_axis_weight: 1.,
        },
    )
    .unwrap();
    let mut state = vec![pose([4., 0., 0.])];
    let initial = state.clone();
    let mut guide = StdRng::seed_from_u64(63);
    let mut physical = StdRng::seed_from_u64(71);
    let mut unchanged = StdRng::seed_from_u64(71);
    let outcome = sampler
        .update(&wall, &mut state, 0., &mut guide, &mut physical)
        .unwrap();
    assert!(!outcome.accepted);
    assert!(outcome.gca.is_none());
    assert!(outcome.stats.forward.capped_failure);
    assert_eq!(outcome.stats.forward.attempts, 1);
    assert_eq!(outcome.stats.reverse.attempts, 0);
    assert_eq!(
        outcome.stats.failure_reason.as_deref(),
        Some("forward_search_cap")
    );
    assert_eq!(state, initial);
    assert_eq!(physical.random::<u64>(), unchanged.random::<u64>());
}

#[test]
fn reverse_cap_failure_rolls_back_physical_endpoint() {
    let tree = sphere();
    let wall = Container::new(15., &tree).unwrap();
    let sampler = ConditionalAxis::new(
        &tree,
        0.,
        ConditionalAxisConfig {
            score_floor: 0.3,
            max_candidates: 1,
            uniform_axis_weight: 1.,
        },
    )
    .unwrap();
    let initial = vec![pose([4., 0., 0.])];
    let mut found = false;
    for seed in 0..100 {
        let mut state = initial.clone();
        let outcome = sampler
            .update(
                &wall,
                &mut state,
                0.,
                &mut StdRng::seed_from_u64(seed),
                &mut StdRng::seed_from_u64(seed + 31),
            )
            .unwrap();
        if outcome.stats.reverse.capped_failure {
            found = true;
            assert!(!outcome.accepted);
            assert!(outcome.gca.is_some());
            assert_eq!(outcome.stats.forward.attempts, 1);
            assert_eq!(outcome.stats.reverse.attempts, 1);
            assert_eq!(
                outcome.stats.failure_reason.as_deref(),
                Some("reverse_search_cap")
            );
            assert_eq!(state, initial);
            assert!(
                outcome.stats.accepted_lost.is_empty() && outcome.stats.accepted_gained.is_empty()
            );
            break;
        }
    }
    assert!(found, "reference stream should exercise reverse failure");
}

#[test]
fn repeatability_and_recorded_four_density_ratio() {
    let tree = sphere();
    let wall = Container::new(15., &tree).unwrap();
    let sampler = ConditionalAxis::new(&tree, 0.5, ConditionalAxisConfig::default()).unwrap();
    let initial = vec![
        pose([4., 0., 0.]),
        pose([4., 2.2, 0.]),
        pose([-4., 2.2, 0.]),
    ];
    let mut saw_ratio = false;
    for seed in 0..24 {
        let mut a = initial.clone();
        let mut b = initial.clone();
        let run = |state: &mut Vec<Pose>| {
            sampler
                .update(
                    &wall,
                    state,
                    0.1,
                    &mut StdRng::seed_from_u64(seed),
                    &mut StdRng::seed_from_u64(seed + 45),
                )
                .unwrap()
        };
        let first = run(&mut a);
        let second = run(&mut b);
        assert_eq!(a, b);
        assert_eq!(first.accepted, second.accepted);
        let mut x = serde_json::to_value(&first.stats).unwrap();
        let mut y = serde_json::to_value(&second.stats).unwrap();
        for value in [&mut x, &mut y] {
            value.as_object_mut().unwrap().remove("selection_seconds");
            value.as_object_mut().unwrap().remove("total_seconds");
        }
        assert_eq!(x, y);
        let s = first.stats;
        if let Some(ratio) = s.log_selector_ratio {
            saw_ratio = true;
            let independent = (s.cross_score_y_u.unwrap() * s.cross_score_x_v.unwrap()
                / (s.forward.selected_score.unwrap() * s.reverse.selected_score.unwrap()))
            .ln()
                + (s.cross_base_y_u.unwrap() * s.cross_base_x_v.unwrap()
                    / (s.forward.selected_base_density.unwrap()
                        * s.reverse.selected_base_density.unwrap()))
                .ln();
            near(ratio, independent, 2e-14);
        }
        spherical::validate_state(&tree, &wall, &a).unwrap();
    }
    assert!(saw_ratio);
}
