//! Independent physical reference for the geometry-dependent axis selector.
//!
//! Two sphere centers have fixed radius R from the vessel center. Their
//! unconstrained separation measure is proportional to d dd. With hard core r
//! and ideal-depletant exclusion radius a, the exact marginal is therefore
//! p(d) ∝ d exp[z V_overlap(a,d)], 2r <= d <= 2R. This tests the production
//! geometric band sampler, Jacobian, four-density correction and physical GCA
//! against a reference that does not use any of those implementations.
//!
//! Rejected and zero/full-flip states are all retained. Separate guide/physical
//! streams, two initial separations and disjoint 1000-step block means provide
//! reproducible, conservative statistical diagnostics; this is not an
//! ergodicity proof or a protein efficiency benchmark.
use rand::{SeedableRng, rngs::StdRng};
use std::f64::consts::PI;
use tetramer_mc::{
    conditional_axis::{ConditionalAxis, ConditionalAxisConfig},
    geometry::{Atom, Shape, SphereTree},
    math::{Pose, norm, sub},
    spherical::{self, Container},
};

const CORE: f64 = 0.3;
const DEPLETANT: f64 = 0.6;
const RADIAL: f64 = 1.5;
const ACTIVITY: f64 = 1.5;
const BURN: usize = 2_000;
const BLOCK: usize = 1_000;
const BLOCKS: usize = 20;

fn reference() -> [f64; 2] {
    let a = CORE + DEPLETANT;
    let integrate = |lo: f64, hi: f64| {
        let n = 50_000;
        let step = (hi - lo) / n as f64;
        let mut total = [0.; 2];
        for k in 0..n {
            let d = lo + (k as f64 + 0.5) * step;
            let overlap = if d < 2. * a {
                PI * (4. * a + d) * (2. * a - d).powi(2) / 12.
            } else { 0. };
            let weight = d * (ACTIVITY * overlap).exp() * step;
            total[0] += weight;
            total[1] += d * weight;
        }
        total
    };
    let contact = integrate(2. * CORE, 2. * a);
    let unbound = integrate(2. * a, 2. * RADIAL);
    let z = contact[0] + unbound[0];
    [(contact[1] + unbound[1]) / z, contact[0] / z]
}

fn initial(distance: f64) -> Vec<Pose> {
    let c = 1. - distance.powi(2) / (2. * RADIAL.powi(2));
    [[RADIAL, 0., 0.], [RADIAL * c, RADIAL * (1. - c * c).sqrt(), 0.]]
        .into_iter().map(|position| Pose { position, orientation: [1., 0., 0., 0.] })
        .collect()
}

#[derive(Debug)]
struct Diagnostics {
    blocks: Vec<[f64; 2]>,
    accepted: usize,
    selector_rejected: usize,
}
fn mean_error(blocks: &[[f64; 2]]) -> ([f64; 2], [f64; 2]) {
    let n = blocks.len() as f64;
    let mean = std::array::from_fn(|j| blocks.iter().map(|b| b[j]).sum::<f64>() / n);
    let error = std::array::from_fn(|j| {
        (blocks.iter().map(|b| (b[j] - mean[j]).powi(2)).sum::<f64>() / (n * (n - 1.))).sqrt()
    });
    (mean, error)
}
fn sample(tree: &SphereTree, wall: &Container, eta: f64, distance: f64, seed: u64) -> Diagnostics {
    // For two particles, exchanging partner identities is impossible; h is
    // constant. Floor=1 removes futile rejection trials while retaining the
    // complete state-dependent band density and exchange auxiliary correction.
    let selector = ConditionalAxis::new(tree, DEPLETANT, ConditionalAxisConfig {
        score_floor: 1., max_candidates: 1, uniform_axis_weight: eta,
    }).unwrap();
    let mut poses = initial(distance);
    spherical::validate_state(tree, wall, &poses).unwrap();
    let mut guide = StdRng::seed_from_u64(seed);
    let mut physical = StdRng::seed_from_u64(seed ^ 0xc831_9576_a932_b06d);
    let mut result = Diagnostics { blocks: Vec::new(), accepted: 0, selector_rejected: 0 };
    let mut block = [0.; 2];
    for step in 0..BURN + BLOCK * BLOCKS {
        let outcome = selector.update(wall, &mut poses, ACTIVITY, &mut guide, &mut physical).unwrap();
        assert!(!outcome.stats.forward.capped_failure && !outcome.stats.reverse.capped_failure);
        if outcome.accepted { result.accepted += 1; } else { result.selector_rejected += 1; }
        if eta == 1. { assert!(outcome.accepted); }
        for p in &poses {
            assert!((norm(p.position) - RADIAL).abs() < 1e-9);
        }
        if step >= BURN {
            let d = norm(sub(poses[0].position, poses[1].position));
            assert!(d >= 2. * CORE - 1e-12 && d <= 2. * RADIAL + 1e-9);
            block[0] += d;
            block[1] += f64::from(d < 2. * (CORE + DEPLETANT));
            if (step - BURN + 1) % BLOCK == 0 {
                result.blocks.push(block.map(|v| v / BLOCK as f64));
                block = [0.; 2];
            }
        }
    }
    assert_eq!(result.blocks.len(), BLOCKS);
    result
}

#[test]
fn conditional_bands_match_exact_two_sphere_depletion_marginal_from_both_starts() {
    let tree = SphereTree::new(Shape {
        name: "independent two-sphere physical reference".into(), volume: 4. * PI * CORE.powi(3) / 3.,
        atoms: vec![Atom { center: [0.; 3], radius: CORE }],
    }).unwrap();
    let wall = Container::new(4., &tree).unwrap();
    let exact = reference();
    assert!((exact[0] - 1.681720516910517).abs() < 2e-9);
    assert!((exact[1] - 0.5502974837944473).abs() < 2e-9);
    let mut arm_results = Vec::new();
    for (arm, eta) in [1., 0.25].into_iter().enumerate() {
        let mut pooled = Vec::new();
        let mut preparations = Vec::new();
        for (preparation, distance) in [0.65, 2.8].into_iter().enumerate() {
            let seed = 202609250310 + 1000 * arm as u64 + 10 * preparation as u64;
            let result = sample(&tree, &wall, eta, distance, seed);
            let (mean, error) = mean_error(&result.blocks);
            eprintln!("sphere axis eta={eta}, start={distance}: mean/contact={mean:?}, block_SE={error:?}, accepted={}, rejected={}", result.accepted, result.selector_rejected);
            if eta < 1. { assert!(result.selector_rejected > 0); }
            for j in 0..2 {
                assert!(error[j] < 0.035, "insufficient reference-test precision: {error:?}");
                assert!((mean[j] - exact[j]).abs() <= 6. * error[j] + 0.003,
                    "arm {arm} preparation {preparation}, observable {j}: {} vs {} ± {}", mean[j], exact[j], error[j]);
            }
            pooled.extend_from_slice(&result.blocks);
            preparations.push((mean, error));
        }
        for j in 0..2 {
            let se = preparations[0].1[j].hypot(preparations[1].1[j]);
            assert!((preparations[0].0[j] - preparations[1].0[j]).abs() <= 6. * se + 0.003);
        }
        let (mean, error) = mean_error(&pooled);
        eprintln!("sphere axis eta={eta} pooled: {mean:?} ± {error:?}; exact={exact:?}");
        for j in 0..2 {
            assert!((mean[j] - exact[j]).abs() <= 6. * error[j] + 0.003);
        }
        arm_results.push((mean, error));
    }
    for j in 0..2 {
        let se = arm_results[0].1[j].hypot(arm_results[1].1[j]);
        assert!((arm_results[0].0[j] - arm_results[1].0[j]).abs() <= 6. * se + 0.003);
    }
}
