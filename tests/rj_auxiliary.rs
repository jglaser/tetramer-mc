//! Exact count-law balance and the continuous/discrete auxiliary prior.
use anyhow::Result;
use rand::{SeedableRng, rngs::StdRng};
use tetramer_mc::rj::{RjConfig, RjState};

#[test]
fn truncated_count_chain_has_exact_poisson_flux_including_boundaries() -> Result<()> {
    for (lo, hi, lambda) in [(1, 1, 3.), (1, 6, 0.4), (2, 13, 8.), (8, 10, 20.)] {
        let c = RjConfig {
            min_components: lo,
            max_components: hi,
            initial_components: lo,
            poisson_mean: lambda,
            attempts_per_sweep: 1,
        };
        c.validate()?;
        let mut p = vec![1.];
        for k in lo..hi {
            p.push(p.last().unwrap() * lambda / (k + 1) as f64);
        }
        let z = p.iter().sum::<f64>();
        for x in &mut p {
            *x /= z;
        }
        let mut evolved = vec![0.; p.len()];
        for k in lo..=hi {
            let birth = if k == hi {
                0.
            } else {
                0.5 * c.birth_log_ratio(k).exp().min(1.)
            };
            let death = if k == lo {
                0.
            } else {
                0.5 * c.death_log_ratio(k).exp().min(1.)
            };
            evolved[k - lo] += p[k - lo] * (1. - birth - death);
            if k < hi {
                evolved[k + 1 - lo] += p[k - lo] * birth;
                let reverse = 0.5 * c.death_log_ratio(k + 1).exp().min(1.);
                assert!((p[k - lo] * birth - p[k + 1 - lo] * reverse).abs() < 1e-14);
            }
            if k > lo {
                evolved[k - 1 - lo] += p[k - lo] * death;
            }
        }
        for (x, y) in p.iter().zip(evolved) {
            assert!((x - y).abs() < 1e-14);
        }
    }
    Ok(())
}

#[test]
fn birth_death_chain_preserves_label_gaussian_priors_and_visits_count_limits() -> Result<()> {
    let c = RjConfig {
        min_components: 1,
        max_components: 6,
        initial_components: 3,
        poisson_mean: 3.,
        attempts_per_sweep: 4,
    };
    let weights = [0.1, 0.3, 0.6];
    let mut rng = StdRng::seed_from_u64(881771);
    let mut state = RjState::new(&c, &weights, &mut rng)?;
    let mut histogram = [0usize; 6];
    let mut labels = [0usize; 3];
    let (mut n, mut first, mut second) = (0usize, 0., 0.);
    let (mut births, mut deaths, mut nulls) = (0, 0, 0);
    for sample in 0..101_000 {
        // Fixed thinning gives independent-enough samples for a generous moment
        // smoke check; the stationary count law is checked algebraically above.
        for _ in 0..16 {
            let jump = state.update(&c, &weights, &mut rng)?;
            births += usize::from(jump.birth && jump.accepted);
            deaths += usize::from(!jump.birth && jump.accepted);
            nulls += usize::from(jump.boundary_null);
        }
        state.refresh_eta(&mut rng);
        if sample < 1000 {
            continue;
        }
        histogram[state.labels.len() - 1] += 1;
        for &label in &state.labels {
            labels[label] += 1;
        }
        for &v in state.eta.iter().flatten() {
            n += 1;
            first += v;
            second += v * v;
        }
    }
    let mut p = vec![1.];
    for k in 1..6 {
        p.push(p.last().unwrap() * 3. / (k + 1) as f64);
    }
    let z = p.iter().sum::<f64>();
    for (i, count) in histogram.iter().enumerate() {
        assert!((*count as f64 / 100_000. - p[i] / z).abs() < 0.012);
    }
    let label_total = labels.iter().sum::<usize>() as f64;
    for (count, expected) in labels.iter().zip(weights) {
        assert!((*count as f64 / label_total - expected).abs() < 0.01);
    }
    assert!((first / n as f64).abs() < 0.01);
    assert!((second / n as f64 - 1.).abs() < 0.01);
    assert!(births > 1000 && deaths > 1000 && nulls > 1000);
    assert!(state.validate(&c, 0).is_err());
    assert!(RjState::new(&c, &[0., 1.], &mut rng).is_err());
    let mut invalid = c.clone();
    invalid.initial_components = 0;
    assert!(invalid.validate().is_err());
    Ok(())
}
