//! Set-volume checks for the frozen GCA diagnostic. No production Markov
//! transition uses these noisy estimates as a plug-in bond probability.
use tetramer_mc::{
    gca_overlap_diagnostic::{PairEstimate, estimate_pair},
    geometry::{Atom, Placed, Shape, SphereTree},
    math::Pose,
};

fn sphere() -> SphereTree {
    SphereTree::new(Shape {
        name: "analytic unit sphere".into(),
        volume: 4. * std::f64::consts::PI / 3.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius: 1.,
        }],
    })
    .unwrap()
}
fn placed(position: [f64; 3]) -> Placed {
    Placed::new(Pose {
        position,
        orientation: [1., 0., 0., 0.],
    })
}
fn exchange() -> (Vec<Placed>, Vec<Placed>) {
    // Half-turn about z: old/cross lenses have equal volumes but are disjoint.
    // Every pair within each endpoint is hard-core valid (distance 2.2 > 2).
    (
        vec![placed([1.1, 1.1, 0.]), placed([1.1, -1.1, 0.])],
        vec![placed([-1.1, -1.1, 0.]), placed([-1.1, 1.1, 0.])],
    )
}
fn lens(radius: f64, separation: f64) -> f64 {
    std::f64::consts::PI * (4. * radius + separation) * (2. * radius - separation).powi(2) / 12.
}
fn assert_volume(estimate: &PairEstimate, hits: usize, expected: f64) {
    let draws: usize = estimate.populations.iter().map(|p| p.draws).sum();
    let p = expected / estimate.envelope_volume;
    let standard_error = estimate.envelope_volume * (p * (1. - p) / draws as f64).sqrt();
    let measured = estimate.envelope_volume * hits as f64 / draws as f64;
    assert!(
        (measured - expected).abs() < 7. * standard_error + 1e-11,
        "measured {measured}, analytic {expected}, standard error {standard_error}"
    );
}

#[test]
fn equal_pair_volumes_can_have_nonzero_pointwise_recruitment() {
    let tree = sphere();
    let (old, shadow) = exchange();
    let e = estimate_pair(&tree, &old, &shadow, 0, 1, 0.5, 16_384, 4, 381_771, 1023).unwrap();
    let expected = lens(1.5, 2.2);
    assert_volume(&e, e.populations.iter().map(|p| p.old_hits).sum(), expected);
    assert_volume(
        &e,
        e.populations.iter().map(|p| p.cross_hits).sum(),
        expected,
    );
    for p in &e.populations {
        assert!(p.old_hits > 0 && p.cross_hits > 0);
        assert_eq!(p.lost_hits, p.old_hits);
        assert_eq!(p.reverse_hits, p.cross_hits);
        assert_eq!(p.shielded_lost_hits, p.lost_hits);
        assert_eq!(p.triple_old_hits, 0);
    }
    let activity = 0.7_f64;
    let pair_volume_probability = -(-activity * (expected - expected).max(0.)).exp_m1();
    let pointwise_probability = -(-activity * expected).exp_m1();
    assert_eq!(pair_volume_probability, 0.);
    assert!(pointwise_probability > 0.5);
    assert!(e.created_cells <= 1023);
}

#[test]
fn identity_move_has_no_lost_or_reverse_volume() {
    let tree = sphere();
    let (old, _) = exchange();
    let e = estimate_pair(&tree, &old, &old, 0, 1, 0.5, 4096, 2, 900, 255).unwrap();
    for p in &e.populations {
        assert_eq!(p.old_hits, p.cross_hits);
        assert!(p.old_hits > 0);
        assert_eq!(p.lost_hits + p.reverse_hits + p.shielded_lost_hits, 0);
    }
}

#[test]
fn separated_pairs_preserve_declared_denominators() {
    let tree = sphere();
    let old = vec![placed([0.; 3]), placed([100., 0., 0.])];
    let shadow = vec![placed([0., 50., 0.]), placed([100., 50., 0.])];
    let e = estimate_pair(&tree, &old, &shadow, 0, 1, 0.5, 1000, 3, 111, 255).unwrap();
    assert_eq!(e.envelope_volume, 0.);
    assert_eq!(e.retained_cells, 0);
    assert_eq!(e.created_cells, 1);
    assert_eq!(e.populations.len(), 3);
    for p in &e.populations {
        assert_eq!(p.draws, 1000);
        assert_eq!(
            p.old_hits
                + p.cross_hits
                + p.lost_hits
                + p.reverse_hits
                + p.shielded_lost_hits
                + p.triple_old_hits,
            0
        );
    }
    assert!(e.populations.windows(2).all(|p| p[0].seed != p[1].seed));
}

#[test]
fn extra_bodies_can_shield_lost_points_and_cover_old_triples() {
    let tree = sphere();
    let (mut old, mut shadow) = exchange();
    // Deliberate duplicated bodies isolate set-mask logic, not a physical
    // hard-sphere state. All shadows still obey the same z half-turn.
    old.extend([old[1], shadow[1]]);
    shadow.extend([shadow[1], old[1]]);
    let e = estimate_pair(&tree, &old, &shadow, 0, 1, 0.5, 8192, 2, 745, 511).unwrap();
    for p in &e.populations {
        assert!(p.lost_hits > 0);
        assert_eq!(p.shielded_lost_hits, 0);
        assert_eq!(p.triple_old_hits, p.old_hits);
    }
}

#[test]
fn root_only_and_refined_envelopes_both_recover_analytic_volume() {
    let tree = sphere();
    let (old, shadow) = exchange();
    for budget in [1, 3, 31, 1023] {
        let e = estimate_pair(
            &tree,
            &old,
            &shadow,
            0,
            1,
            0.5,
            16_384,
            2,
            801_951 + budget as u64,
            budget,
        )
        .unwrap();
        assert!(e.created_cells <= budget);
        assert_volume(
            &e,
            e.populations.iter().map(|p| p.old_hits).sum(),
            lens(1.5, 2.2),
        );
        assert_volume(
            &e,
            e.populations.iter().map(|p| p.cross_hits).sum(),
            lens(1.5, 2.2),
        );
    }
}

#[test]
fn replay_is_deterministic_and_populations_extend_without_changing_prefix() {
    let tree = sphere();
    let (old, shadow) = exchange();
    let a = estimate_pair(&tree, &old, &shadow, 0, 1, 0.5, 2048, 2, 914, 511).unwrap();
    let b = estimate_pair(&tree, &old, &shadow, 0, 1, 0.5, 2048, 2, 914, 511).unwrap();
    let c = estimate_pair(&tree, &old, &shadow, 0, 1, 0.5, 2048, 4, 914, 511).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.populations, c.populations[..2]);
    assert_ne!(c.populations[0].seed, c.populations[1].seed);
}

#[test]
fn invalid_allocations_and_geometry_fail_before_sampling() {
    let tree = sphere();
    let (old, shadow) = exchange();
    for radius in [-0.1, f64::NAN, f64::INFINITY] {
        assert!(estimate_pair(&tree, &old, &shadow, 0, 1, radius, 20, 1, 0, 31).is_err());
    }
    assert!(estimate_pair(&tree, &old, &shadow, 0, 0, 0.5, 20, 1, 0, 31).is_err());
    assert!(estimate_pair(&tree, &old, &shadow, 0, 2, 0.5, 20, 1, 0, 31).is_err());
    assert!(estimate_pair(&tree, &old, &shadow[..1], 0, 1, 0.5, 20, 1, 0, 31).is_err());
    assert!(estimate_pair(&tree, &[], &[], 0, 1, 0.5, 20, 1, 0, 31).is_err());
    assert!(estimate_pair(&tree, &old, &shadow, 0, 1, 0.5, 0, 1, 0, 31).is_err());
    assert!(estimate_pair(&tree, &old, &shadow, 0, 1, 0.5, 20, 0, 0, 31).is_err());
    assert!(estimate_pair(&tree, &old, &shadow, 0, 1, 0.5, 20, 1, 0, 0).is_err());
    let mut invalid = old;
    invalid[0].position[0] = f64::NAN;
    assert!(estimate_pair(&tree, &invalid, &shadow, 0, 1, 0.5, 20, 1, 0, 31).is_err());
}
