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
