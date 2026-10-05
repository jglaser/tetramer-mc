//! Reference checks independent of the depletion acceptance kernel. The equal
//! sphere lens gives an analytic C, including exact Poisson first/second moments.
use anyhow::Result;
use rand::{SeedableRng, rngs::StdRng};
use std::f64::consts::PI;
use tetramer_mc::{
    depletion::GateOptions,
    geometry::{Atom, Environment, Placed, Shape, SphereTree},
    math::Pose,
    overlap_weight::{self, OverlapEnvelope},
};

fn sphere() -> Result<SphereTree> {
    SphereTree::new(Shape {
        name: "positive overlap estimator reference".into(),
        volume: 4. * PI / 3.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius: 1.,
        }],
    })
}

fn pose(position: [f64; 3]) -> Pose {
    Pose {
        position,
        orientation: [1., 0., 0., 0.],
    }
}

fn environment<'a>(tree: &'a SphereTree, positions: &[[f64; 3]]) -> Environment<'a> {
    Environment {
        tree,
        fixed: positions
            .iter()
            .copied()
            .map(|p| Placed::new(pose(p)))
            .collect(),
        labels: Vec::new(),
        rd: 0.6,
    }
}

fn options(max_cells: usize) -> GateOptions {
    GateOptions {
        max_cells,
        max_depth: 40,
        min_width: 0.,
    }
}

fn lens(radius: f64, distance: f64) -> f64 {
    if distance >= 2. * radius {
        0.
    } else {
        PI * (4. * radius + distance) * (2. * radius - distance).powi(2) / 12.
    }
}

/// Independent theoretical moments, not sample-estimated error bars.
fn check_moments(env: &Environment, c: f64, budget: usize, seed: u64, n: usize) -> Result<()> {
    let old = pose([0.; 3]);
    let envelope = OverlapEnvelope::build(env, old, options(budget))?;
    assert!(envelope.lower_volume <= c + 1e-12);
    assert!(envelope.upper_volume() >= c - 1e-12);
    let z = 0.35;
    let lambda = 1.4;
    let residual = c - envelope.lower_volume;
    // Moment of W / exp(z*C); first moment is exactly one.
    let moment = |order: i32| {
        (lambda * residual * ((1. + z / lambda).powi(order) - 1.) - order as f64 * z * residual)
            .exp()
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let mut sums = [0.; 2];
    let mut count_sum = 0.;
    for _ in 0..n {
        let result =
            overlap_weight::sample_with_envelope(&mut rng, env, old, lambda, z, &envelope)?;
        assert!(result.raw_points >= result.overlap_points);
        assert_eq!(result.created_cells, envelope.created);
        let normalized = (result.log_weight - z * c).exp();
        sums[0] += normalized;
        sums[1] += normalized * normalized;
        count_sum += result.overlap_points as f64;
    }
    for (j, observed) in sums.iter().enumerate() {
        let order = (j + 1) as i32;
        let expected = moment(order);
        let se = ((moment(2 * order) - expected * expected) / n as f64).sqrt();
        let standardized = (observed / n as f64 - expected) / se;
        eprintln!(
            "budget={budget}, moment={order}, z={standardized:.3}, L={}, U={}",
            envelope.lower_volume,
            envelope.upper_volume()
        );
        assert!(
            standardized.abs() < 5.5,
            "analytic moment mismatch: {standardized}"
        );
    }
    let count_mean = lambda * residual;
    assert!((count_sum / n as f64 - count_mean).abs() < 5.5 * (count_mean / n as f64).sqrt());
    Ok(())
}

#[test]
fn analytic_lens_bounds_and_budget_sensitive_variance() -> Result<()> {
    let tree = sphere()?;
    let env = environment(&tree, &[[2., 0., 0.]]);
    let c = lens(1.6, 2.);
    let mut previous_lower = 0.;
    let mut previous_upper = f64::INFINITY;
    for budget in [1, 31, 255, 2047, 16383] {
        let envelope = OverlapEnvelope::build(&env, pose([0.; 3]), options(budget))?;
        assert!(envelope.lower_volume <= c && c <= envelope.upper_volume());
        assert!(envelope.lower_volume >= previous_lower - 1e-12);
        assert!(envelope.upper_volume() <= previous_upper + 1e-12);
        assert!(envelope.created <= budget);
        previous_lower = envelope.lower_volume;
        previous_upper = envelope.upper_volume();
    }
    assert!(previous_lower > 0.5 * c);
    assert!(previous_upper < 1.5 * c);
    for (j, budget) in [1, 511, 8191].into_iter().enumerate() {
        check_moments(&env, c, budget, 91405 + j as u64, 30_000)?;
    }
    Ok(())
}

#[test]
fn fixed_exclusion_union_is_not_a_sum_over_neighbors() -> Result<()> {
    let tree = sphere()?;
    let single = environment(&tree, &[[2., 0., 0.]]);
    let duplicate = environment(&tree, &[[2., 0., 0.], [2., 0., 0.]]);
    let old = pose([0.; 3]);
    let a = OverlapEnvelope::build(&single, old, options(2047))?;
    let b = OverlapEnvelope::build(&duplicate, old, options(2047))?;
    assert_eq!(a.lower_volume, b.lower_volume);
    assert_eq!(a.uncertain_volume, b.uncertain_volume);
    let mut rng_a = StdRng::seed_from_u64(7280);
    let mut rng_b = StdRng::seed_from_u64(7280);
    for _ in 0..1000 {
        let a = overlap_weight::sample_with_envelope(&mut rng_a, &single, old, 1.4, 0.35, &a)?;
        let b = overlap_weight::sample_with_envelope(&mut rng_b, &duplicate, old, 1.4, 0.35, &b)?;
        assert_eq!(a.raw_points, b.raw_points);
        assert_eq!(a.overlap_points, b.overlap_points);
        assert_eq!(a.log_weight, b.log_weight);
    }
    // Two disjoint fixed exclusion balls, one duplicated. Known union overlap
    // is two lenses, whereas a pair sum would spuriously count three lenses.
    let union = environment(&tree, &[[2., 0., 0.], [-2., 0., 0.], [2., 0., 0.]]);
    check_moments(&union, 2. * lens(1.6, 2.), 4095, 819409, 40_000)?;
    Ok(())
}

#[test]
fn rotated_sphere_union_uses_body_and_laboratory_frames_consistently() -> Result<()> {
    // A separated dumbbell against a perpendicular one has exactly one lens:
    // moving atom (3,0,0) against fixed atom (3,2,0). All other atom pairs
    // are more than two exclusion radii apart. This is a hard-valid contact.
    let tree = SphereTree::new(Shape {
        name: "rotated independent dumbbells".into(),
        volume: 8. * PI / 3.,
        atoms: vec![
            Atom {
                center: [-3., 0., 0.],
                radius: 1.,
            },
            Atom {
                center: [3., 0., 0.],
                radius: 1.,
            },
        ],
    })?;
    let env = Environment {
        tree: &tree,
        fixed: vec![Placed::new(Pose {
            position: [3., 5., 0.],
            orientation: [
                std::f64::consts::FRAC_1_SQRT_2,
                0.,
                0.,
                std::f64::consts::FRAC_1_SQRT_2,
            ],
        })],
        labels: Vec::new(),
        rd: 0.6,
    };
    check_moments(&env, lens(1.6, 2.), 4095, 883718, 30_000)?;
    Ok(())
}

#[test]
fn empty_overlap_zero_activity_and_boundary_reference_limits() -> Result<()> {
    let tree = sphere()?;
    let old = pose([0.; 3]);
    let mut rng = StdRng::seed_from_u64(3817);
    for positions in [vec![], vec![[4., 0., 0.]], vec![[3.2, 0., 0.]]] {
        let env = environment(&tree, &positions);
        let envelope = OverlapEnvelope::build(&env, old, options(2047))?;
        assert_eq!(envelope.lower_volume, 0.);
        for _ in 0..50 {
            let result =
                overlap_weight::sample_with_envelope(&mut rng, &env, old, 1., 0.3, &envelope)?;
            assert_eq!(result.log_weight, 0.);
            assert_eq!(result.overlap_points, 0);
        }
    }
    let env = environment(&tree, &[[2., 0., 0.]]);
    let result = overlap_weight::sample(&mut rng, &env, old, 1., 0., options(1))?;
    assert_eq!(result.log_weight, 0.);
    assert_eq!(result.raw_points, 0);
    assert_eq!(result.created_cells, 0);
    assert_eq!(
        overlap_weight::sample(&mut rng, &env, old, 0., 0., options(1))?.log_weight,
        0.
    );
    assert!(overlap_weight::sample(&mut rng, &env, old, 0., 0.3, options(1)).is_err());
    assert!(overlap_weight::sample(&mut rng, &env, old, 1., -0.3, options(1)).is_err());
    let envelope = OverlapEnvelope::build(&env, old, options(1))?;
    assert!(
        overlap_weight::sample_with_envelope(
            &mut rng,
            &env,
            pose([0.1, 0., 0.]),
            1.,
            0.3,
            &envelope,
        )
        .is_err()
    );
    Ok(())
}

// The compatibility oracle below is the pre-observer sampling loop, frozen in
// this test. Unlike a bounded-vs-public-wrapper comparison alone, it can catch
// an accidental RNG/arithmetic change in their now-shared private core.
fn legacy_positive_cloud(
    rng: &mut StdRng,
    env: &Environment,
    old: Pose,
    lambda: f64,
    z: f64,
    envelope: &OverlapEnvelope,
) -> overlap_weight::OverlapWeight {
    use rand::RngExt;
    use rand_distr::{Distribution, Poisson};
    let mut result = overlap_weight::OverlapWeight {
        lower_volume: envelope.lower_volume,
        upper_volume: envelope.upper_volume(),
        uncertain_volume: envelope.uncertain_volume,
        retained_cells: envelope.cells.len(),
        created_cells: envelope.created,
        certified_cells: envelope.certified_cells,
        ..Default::default()
    };
    result.log_weight = z * envelope.lower_volume;
    result.raw_points = Poisson::<f64>::new(lambda * envelope.uncertain_volume)
        .unwrap()
        .sample(rng) as u64;
    let moving = Placed::new(old);
    for _ in 0..result.raw_points {
        let target = rng.random::<f64>() * envelope.uncertain_volume;
        let k = envelope
            .cumulative
            .partition_point(|&v| v <= target)
            .min(envelope.cells.len() - 1);
        let cell = envelope.cells[k];
        let p =
            std::array::from_fn(|j| cell.lo[j] + rng.random::<f64>() * (cell.hi[j] - cell.lo[j]));
        if env.tree.contains(p, env.rd) && env.contains(moving.apply(p)) {
            result.overlap_points += 1;
        }
    }
    let ratio = z / lambda;
    let coefficient = if ratio.is_finite() {
        ratio.ln_1p()
    } else {
        (lambda + z).ln() - lambda.ln()
    };
    result.log_weight += result.overlap_points as f64 * coefficient;
    result
}

fn cloud_limits(interval: u64) -> overlap_weight::CloudLimits {
    overlap_weight::CloudLimits {
        raw_points: 1_000_000,
        processed_points: 1_000_000,
        callback_interval: interval,
    }
}

#[test]
fn bounded_cloud_matches_legacy_and_original_bitwise_with_rng_continuation() -> Result<()> {
    use overlap_weight::{CloudEvent, CloudProgress};
    use rand::RngExt;
    let tree = sphere()?;
    let old = pose([0.; 3]);
    for positions in [
        vec![[2., 0., 0.]],
        vec![[2., 0., 0.], [-2., 0., 0.], [2., 0., 0.]],
    ] {
        let env = environment(&tree, &positions);
        for cells in [1, 255] {
            let envelope = OverlapEnvelope::build(&env, old, options(cells))?;
            assert!(envelope.uncertain_volume > 0.);
            for seed in [19, 721, 20261005] {
                for interval in [1, 7, 1024] {
                    let mut bounded_rng = StdRng::seed_from_u64(seed);
                    let mut original_rng = StdRng::seed_from_u64(seed);
                    let mut legacy_rng = StdRng::seed_from_u64(seed);
                    let mut progress = CloudProgress::default();
                    let mut events = Vec::new();
                    let bounded = overlap_weight::sample_with_envelope_bounded(
                        &mut bounded_rng,
                        &env,
                        old,
                        1.4,
                        0.35,
                        &envelope,
                        cloud_limits(interval),
                        &mut progress,
                        |event, state| {
                            assert!(!state.complete && state.log_weight.is_none());
                            events.push((event, *state));
                            Ok(())
                        },
                    )?;
                    let original = overlap_weight::sample_with_envelope(
                        &mut original_rng,
                        &env,
                        old,
                        1.4,
                        0.35,
                        &envelope,
                    )?;
                    let legacy =
                        legacy_positive_cloud(&mut legacy_rng, &env, old, 1.4, 0.35, &envelope);
                    assert_eq!(
                        serde_json::to_value(bounded)?,
                        serde_json::to_value(original)?
                    );
                    assert_eq!(
                        serde_json::to_value(bounded)?,
                        serde_json::to_value(legacy)?
                    );
                    assert_eq!(bounded.log_weight.to_bits(), legacy.log_weight.to_bits());
                    assert_eq!(progress.planned_points, Some(bounded.raw_points));
                    assert_eq!(progress.processed_points, bounded.raw_points);
                    assert_eq!(progress.overlap_points, bounded.overlap_points);
                    assert!(progress.complete && progress.log_weight == Some(bounded.log_weight));
                    assert_eq!(events[0].0, CloudEvent::Begun);
                    assert_eq!(events[1].0, CloudEvent::CountDrawn);
                    assert_eq!(events[1].1.processed_points, 0);
                    assert_eq!(events.last().unwrap().0, CloudEvent::Finishing);
                    for _ in 0..4 {
                        let next = legacy_rng.random::<u64>();
                        assert_eq!(original_rng.random::<u64>(), next);
                        assert_eq!(bounded_rng.random::<u64>(), next);
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn bounded_cloud_zero_activity_and_empty_envelope_consume_no_rng() -> Result<()> {
    use overlap_weight::{CloudEvent, CloudProgress};
    use rand::RngExt;
    let tree = sphere()?;
    let old = pose([0.; 3]);
    for (positions, lambda, z) in [(vec![], 1., 0.3), (vec![[2., 0., 0.]], 0., 0.)] {
        let env = environment(&tree, &positions);
        let envelope = OverlapEnvelope::build(&env, old, options(255))?;
        let mut rng = StdRng::seed_from_u64(9126);
        let mut baseline = StdRng::seed_from_u64(9126);
        let mut original_rng = StdRng::seed_from_u64(9126);
        let mut progress = CloudProgress::default();
        let mut events = Vec::new();
        let mut caps = cloud_limits(7);
        caps.raw_points = 0;
        caps.processed_points = 0;
        let result = overlap_weight::sample_with_envelope_bounded(
            &mut rng,
            &env,
            old,
            lambda,
            z,
            &envelope,
            caps,
            &mut progress,
            |event, _| {
                events.push(event);
                Ok(())
            },
        )?;
        let original = overlap_weight::sample_with_envelope(
            &mut original_rng,
            &env,
            old,
            lambda,
            z,
            &envelope,
        )?;
        assert_eq!(
            serde_json::to_value(result)?,
            serde_json::to_value(original)?
        );
        assert_eq!(result.log_weight, 0.);
        assert_eq!(progress.planned_points, None);
        assert_eq!(progress.processed_points, 0);
        assert!(progress.complete);
        assert_eq!(events, [CloudEvent::Begun, CloudEvent::Finishing]);
        let next = baseline.random::<u64>();
        assert_eq!(rng.random::<u64>(), next);
        assert_eq!(original_rng.random::<u64>(), next);
    }
    Ok(())
}

#[test]
fn bounded_cloud_count_is_visible_before_both_budget_failures() -> Result<()> {
    use overlap_weight::{CloudEvent, CloudProgress};
    use rand::RngExt;
    use rand_distr::{Distribution, Poisson};
    let tree = sphere()?;
    let env = environment(&tree, &[[2., 0., 0.]]);
    let old = pose([0.; 3]);
    let envelope = OverlapEnvelope::build(&env, old, options(1))?;
    for raw_failure in [true, false] {
        let mut rng = StdRng::seed_from_u64(173);
        let mut reference = StdRng::seed_from_u64(173);
        let planned =
            Poisson::<f64>::new(2. * envelope.uncertain_volume)?.sample(&mut reference) as u64;
        assert!(planned > 0);
        let mut caps = cloud_limits(7);
        if raw_failure {
            caps.raw_points = planned - 1;
        } else {
            caps.processed_points = planned - 1;
        }
        let mut progress = CloudProgress::default();
        let mut journal = Vec::new();
        let error = overlap_weight::sample_with_envelope_bounded(
            &mut rng,
            &env,
            old,
            2.,
            0.3,
            &envelope,
            caps,
            &mut progress,
            |event, state| {
                journal.push(serde_json::to_string(&(event, state))?);
                Ok(())
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains(if raw_failure {
            "raw-point"
        } else {
            "processed-point"
        }));
        assert_eq!(journal.len(), 2);
        let count_event: (CloudEvent, CloudProgress) = serde_json::from_str(&journal[1])?;
        assert_eq!(count_event.0, CloudEvent::CountDrawn);
        assert_eq!(count_event.1.planned_points, Some(planned));
        assert_eq!(progress.planned_points, Some(planned));
        assert_eq!(progress.processed_points, 0);
        assert!(!progress.complete && progress.log_weight.is_none());
        assert_eq!(rng.random::<u64>(), reference.random::<u64>());
    }
    Ok(())
}

#[test]
fn bounded_cloud_callback_abort_retains_exact_processed_prefix() -> Result<()> {
    use overlap_weight::{CloudEvent, CloudProgress};
    use rand::RngExt;
    use rand_distr::{Distribution, Poisson};
    let tree = sphere()?;
    let env = environment(&tree, &[[2., 0., 0.]]);
    let old = pose([0.; 3]);
    let envelope = OverlapEnvelope::build(&env, old, options(1))?;
    let mut rng = StdRng::seed_from_u64(6418);
    let mut reference = StdRng::seed_from_u64(6418);
    let planned =
        Poisson::<f64>::new(2. * envelope.uncertain_volume)?.sample(&mut reference) as u64;
    assert!(planned >= 3);
    let mut overlaps = 0;
    for _ in 0..3 {
        let _cell_uniform = reference.random::<f64>(); // one-cell envelope
        let cell = envelope.cells[0];
        let point = std::array::from_fn(|j| {
            cell.lo[j] + reference.random::<f64>() * (cell.hi[j] - cell.lo[j])
        });
        overlaps += u64::from(env.tree.contains(point, env.rd) && env.contains(point));
    }
    let mut progress = CloudProgress::default();
    let mut events = Vec::new();
    let result = overlap_weight::sample_with_envelope_bounded(
        &mut rng,
        &env,
        old,
        2.,
        0.3,
        &envelope,
        cloud_limits(3),
        &mut progress,
        |event, state| {
            events.push((event, *state));
            if event == CloudEvent::Progress {
                anyhow::bail!("synthetic CPU/wall guard");
            }
            Ok(())
        },
    );
    assert!(result.is_err());
    assert_eq!(
        events.iter().map(|x| x.0).collect::<Vec<_>>(),
        [
            CloudEvent::Begun,
            CloudEvent::CountDrawn,
            CloudEvent::Progress
        ]
    );
    assert_eq!(progress.planned_points, Some(planned));
    assert_eq!(progress.processed_points, 3);
    assert_eq!(progress.overlap_points, overlaps);
    assert!(!progress.complete && progress.log_weight.is_none());
    assert_eq!(rng.random::<u64>(), reference.random::<u64>());
    Ok(())
}

#[test]
fn bounded_cloud_final_guard_failure_never_publishes_weight() -> Result<()> {
    use overlap_weight::{CloudEvent, CloudProgress};
    use rand::RngExt;
    let tree = sphere()?;
    let env = environment(&tree, &[[2., 0., 0.]]);
    let old = pose([0.; 3]);
    let envelope = OverlapEnvelope::build(&env, old, options(31))?;
    let mut rng = StdRng::seed_from_u64(963);
    let mut reference = StdRng::seed_from_u64(963);
    let original =
        overlap_weight::sample_with_envelope(&mut reference, &env, old, 1.4, 0.35, &envelope)?;
    let mut progress = CloudProgress::default();
    let result = overlap_weight::sample_with_envelope_bounded(
        &mut rng,
        &env,
        old,
        1.4,
        0.35,
        &envelope,
        cloud_limits(7),
        &mut progress,
        |event, _| {
            if event == CloudEvent::Finishing {
                anyhow::bail!("final journal/CPU failure");
            }
            Ok(())
        },
    );
    assert!(result.is_err());
    assert_eq!(progress.processed_points, original.raw_points);
    assert_eq!(progress.overlap_points, original.overlap_points);
    assert!(!progress.complete && progress.log_weight.is_none());
    assert_eq!(rng.random::<u64>(), reference.random::<u64>());
    Ok(())
}

#[test]
fn bounded_cloud_start_validation_and_reused_progress_do_not_draw() -> Result<()> {
    use overlap_weight::{CloudEvent, CloudProgress};
    use rand::RngExt;
    let tree = sphere()?;
    let env = environment(&tree, &[[2., 0., 0.]]);
    let old = pose([0.; 3]);
    let envelope = OverlapEnvelope::build(&env, old, options(1))?;
    for case in 0..5 {
        let mut rng = StdRng::seed_from_u64(320);
        let mut baseline = StdRng::seed_from_u64(320);
        let mut progress = CloudProgress::default();
        if case == 4 {
            progress.begun = true;
        }
        let mut caps = cloud_limits(7);
        if case == 0 {
            caps.callback_interval = 0;
        }
        let p = if case == 1 { pose([0.1, 0., 0.]) } else { old };
        let lambda = if case == 2 { 0. } else { 1. };
        let mut events = Vec::new();
        let result = overlap_weight::sample_with_envelope_bounded(
            &mut rng,
            &env,
            p,
            lambda,
            0.3,
            &envelope,
            caps,
            &mut progress,
            |event, _| {
                events.push(event);
                if case == 3 {
                    anyhow::bail!("begin journal failure");
                }
                Ok(())
            },
        );
        assert!(result.is_err());
        assert_eq!(
            events,
            if case == 0 || case == 4 {
                vec![]
            } else {
                vec![CloudEvent::Begun]
            }
        );
        assert_eq!(progress.planned_points, None);
        assert_eq!(progress.processed_points, 0);
        assert!(!progress.complete && progress.log_weight.is_none());
        assert_eq!(rng.random::<u64>(), baseline.random::<u64>());
    }
    Ok(())
}

#[test]
fn bounded_cloud_sampled_zero_is_distinct_from_skipped_poisson() -> Result<()> {
    use overlap_weight::{CloudEvent, CloudProgress};
    let tree = sphere()?;
    let env = environment(&tree, &[[2., 0., 0.]]);
    let old = pose([0.; 3]);
    let envelope = OverlapEnvelope::build(&env, old, options(1))?;
    let mut rng = StdRng::seed_from_u64(3817);
    let mut progress = CloudProgress::default();
    let mut events = Vec::new();
    let mut caps = cloud_limits(7);
    caps.raw_points = 0;
    caps.processed_points = 0;
    let result = overlap_weight::sample_with_envelope_bounded(
        &mut rng,
        &env,
        old,
        1e-12,
        0.3,
        &envelope,
        caps,
        &mut progress,
        |event, _| {
            events.push(event);
            Ok(())
        },
    )?;
    assert_eq!(result.raw_points, 0);
    assert_eq!(progress.planned_points, Some(0));
    assert!(progress.complete);
    assert_eq!(
        events,
        [
            CloudEvent::Begun,
            CloudEvent::CountDrawn,
            CloudEvent::Finishing
        ]
    );
    Ok(())
}

#[test]
fn bounded_cloud_remaining_campaign_caps_charge_each_attempt_once() -> Result<()> {
    use overlap_weight::CloudProgress;
    use rand_distr::{Distribution, Poisson};
    let tree = sphere()?;
    let env = environment(&tree, &[[2., 0., 0.]]);
    let old = pose([0.; 3]);
    let envelope = OverlapEnvelope::build(&env, old, options(1))?;
    let seeds = [914, 1519];
    let planned: Vec<u64> = seeds
        .iter()
        .map(|&seed| {
            Poisson::<f64>::new(2. * envelope.uncertain_volume)
                .unwrap()
                .sample(&mut StdRng::seed_from_u64(seed)) as u64
        })
        .collect();
    assert!(planned.iter().all(|&n| n > 0));
    let campaign_cap = planned[0] + planned[1] - 1;
    let mut total_planned = 0;
    let mut total_processed = 0;
    for (index, seed) in seeds.into_iter().enumerate() {
        let mut caps = cloud_limits(3);
        caps.raw_points = caps.raw_points.min(campaign_cap - total_planned);
        caps.processed_points = caps.processed_points.min(campaign_cap - total_processed);
        let mut progress = CloudProgress::default();
        let mut callbacks = 0;
        let result = overlap_weight::sample_with_envelope_bounded(
            &mut StdRng::seed_from_u64(seed),
            &env,
            old,
            2.,
            0.3,
            &envelope,
            caps,
            &mut progress,
            |_, _| {
                callbacks += 1;
                Ok(())
            },
        );
        // Accumulate once after the call, never once per progress notification.
        total_planned += progress.planned_points.unwrap_or(0);
        total_processed += progress.processed_points;
        assert_eq!(progress.planned_points, Some(planned[index]));
        if index == 0 {
            assert!(result.is_ok() && progress.complete);
            assert!(callbacks >= 3);
            assert_eq!(total_planned, planned[0]);
            assert_eq!(total_processed, planned[0]);
        } else {
            assert!(result.is_err() && !progress.complete);
            assert_eq!(callbacks, 2);
            assert_eq!(progress.processed_points, 0);
            assert!(progress.log_weight.is_none());
        }
    }
    // The oversized sampled request is recorded in full; only its predecessor
    // was processed. A caller must stop here, not subtract again or retry.
    assert_eq!(total_planned, campaign_cap + 1);
    assert_eq!(total_processed, planned[0]);
    Ok(())
}
