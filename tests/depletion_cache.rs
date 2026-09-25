//! Regression oracle for cached body geometry: every envelope, Poisson result,
//! and subsequent RNG state must agree bit-for-bit with the uncached code.
//! These tests compare deterministic calculations; they are not sampling runs.
use rand::{RngExt, SeedableRng, rngs::StdRng};
use tetramer_mc::{
    depletion::{self, BodyEnvelopeCache, Envelope, GateOptions, GateResult},
    geometry::{Atom, Environment, Placed, Shape, SphereTree},
    math::{Pose, add, cayley, matmul, quaternion, rotation, uniform_pose},
};

fn pose(position: [f64; 3]) -> Pose {
    Pose {
        position,
        orientation: [1., 0., 0., 0.],
    }
}

fn shape(dumbbell: bool) -> SphereTree {
    let atoms = if dumbbell {
        vec![
            Atom {
                center: [-0.7, 0., 0.],
                radius: 1.,
            },
            Atom {
                center: [0.9, 0.2, -0.1],
                radius: 0.8,
            },
        ]
    } else {
        vec![Atom {
            center: [0.; 3],
            radius: 1.,
        }]
    };
    SphereTree::new(Shape {
        name: "cache reference".into(),
        volume: 0.,
        atoms,
    })
    .unwrap()
}

fn assert_float(a: f64, b: f64, field: &str) {
    assert_eq!(
        a.to_bits(),
        b.to_bits(),
        "bitwise mismatch in {field}: {a:?} vs {b:?}"
    );
}

fn assert_envelopes(a: &Envelope, b: &Envelope) {
    assert_eq!(a.created, b.created, "created-cell budget changed");
    assert_eq!(a.cells.len(), b.cells.len(), "retained-cell count changed");
    assert_eq!(a.cumulative.len(), b.cumulative.len());
    for (left, right) in a.cells.iter().zip(&b.cells) {
        for k in 0..3 {
            assert_float(left.lo[k], right.lo[k], "cell.lo");
            assert_float(left.hi[k], right.hi[k], "cell.hi");
        }
    }
    for (&left, &right) in a.cumulative.iter().zip(&b.cumulative) {
        assert_float(left, right, "cumulative volume");
    }
    assert_float(a.volume, b.volume, "envelope volume");
}

fn assert_gates(a: GateResult, b: GateResult) {
    assert_eq!(a.gained, b.gained);
    assert_eq!(a.lost, b.lost);
    assert_eq!(a.raw_points, b.raw_points);
    assert_eq!(a.retained_points, b.retained_points);
    assert_eq!(a.retained_cells, b.retained_cells);
    assert_eq!(a.created_cells, b.created_cells);
    assert_float(a.log_weight, b.log_weight, "gate log weight");
    assert_float(a.envelope_volume, b.envelope_volume, "gate envelope volume");
}

fn check_gate(
    cache: &mut BodyEnvelopeCache<'_>,
    env: &Environment<'_>,
    old: Pose,
    new: Pose,
    opts: GateOptions,
    seed: u64,
    lambda: f64,
    z: f64,
) -> GateResult {
    let mut baseline_rng = StdRng::seed_from_u64(seed);
    let mut cached_rng = StdRng::seed_from_u64(seed);
    let baseline = depletion::sample(&mut baseline_rng, env, old, new, lambda, z, opts).unwrap();
    let cached = cache
        .sample(&mut cached_rng, env, old, new, lambda, z, opts)
        .unwrap();
    assert_gates(baseline, cached);
    // Equal outcomes alone do not establish reproducible continuation.
    for _ in 0..4 {
        assert_eq!(
            baseline_rng.random::<u64>(),
            cached_rng.random::<u64>(),
            "cache changed RNG consumption"
        );
    }
    cached
}

#[test]
fn cached_envelopes_and_gates_match_across_budgets_and_endpoint_swaps() {
    let mut raw = 0;
    let mut retained = 0;
    for dumbbell in [false, true] {
        let tree = shape(dumbbell);
        let mut cache = BodyEnvelopeCache::new(&tree, 0.6).unwrap();
        let old = pose([12., 12., 12.]);
        let new = Pose {
            position: [12.35, 12.12, 11.9],
            orientation: quaternion(cayley([0.09, -0.04, 0.02])),
        };
        let poses = [old, pose([14.3, 12., 12.]), pose([11.4, 14.1, 12.3])];
        for (from, to) in [(old, new), (new, old)] {
            let env = Environment::new(&tree, &poses, 0, from, to, [48.; 3], 0.6).unwrap();
            assert!(!env.fixed.is_empty());
            for max_cells in [1, 3, 31, 2047] {
                for min_width in [0., 0.5, 8.] {
                    let opts = GateOptions {
                        max_cells,
                        max_depth: 14,
                        min_width,
                    };
                    let reference = Envelope::build(&env, from, to, opts).unwrap();
                    let cached = cache.build(&env, from, to, opts).unwrap();
                    assert_envelopes(&reference, &cached);
                    let nodes = cache.node_count();
                    assert_envelopes(&cached, &cache.build(&env, from, to, opts).unwrap());
                    assert_eq!(
                        nodes,
                        cache.node_count(),
                        "identical warm query grew the cache"
                    );
                    for seed in [72417, 86193] {
                        let result = check_gate(&mut cache, &env, from, to, opts, seed, 0.4, 0.1);
                        raw += result.raw_points;
                        retained += result.retained_points;
                    }
                }
            }
        }
        assert!(cache.node_count() > 0);
    }
    assert!(
        raw > 0 && retained > 0,
        "fixture must exercise nontrivial Poisson thinning"
    );
}

#[test]
fn changing_orientations_and_environments_preserves_the_uncached_result() {
    let tree = shape(true);
    let mut cache = BodyEnvelopeCache::new(&tree, 0.35).unwrap();
    let mut fixture_rng = StdRng::seed_from_u64(936020);
    for case in 0..12 {
        let mut old = uniform_pose(&mut fixture_rng, [48.; 3]);
        old.position = [12., 12., 12.];
        let mut new = uniform_pose(&mut fixture_rng, [48.; 3]);
        new.position =
            std::array::from_fn(|k| old.position[k] + fixture_rng.random_range(-0.7..0.7));
        let mut spectator = uniform_pose(&mut fixture_rng, [48.; 3]);
        spectator.position = [14.2, 12.1 + 0.03 * case as f64, 12.];
        let poses = [old, spectator, pose([12., 14.6, 12.4])];
        let env = Environment::new(&tree, &poses, 0, old, new, [48.; 3], 0.35).unwrap();
        let opts = GateOptions {
            max_cells: [3, 31, 2047][case % 3],
            max_depth: [0, 5, 14][case % 3],
            min_width: 0.15,
        };
        assert_envelopes(
            &Envelope::build(&env, old, new, opts).unwrap(),
            &cache.build(&env, old, new, opts).unwrap(),
        );
        check_gate(
            &mut cache,
            &env,
            old,
            new,
            opts,
            555000 + case as u64,
            0.25,
            0.06,
        );
    }
}

#[test]
fn periodic_crossing_uses_the_same_image_union_and_rng_stream() {
    let tree = shape(true);
    let old = pose([0.2, 12., 12.]);
    let new = Pose {
        position: [47.8, 12.2, 12.],
        orientation: quaternion(cayley([0., 0., 0.07])),
    };
    let poses = [old, pose([46.2, 12.3, 12.]), pose([1.9, 12., 12.2])];
    let mut cache = BodyEnvelopeCache::new(&tree, 0.6).unwrap();
    let opts = GateOptions::default();
    for (from, to) in [(old, new), (new, old)] {
        let env = Environment::new(&tree, &poses, 0, from, to, [48.; 3], 0.6).unwrap();
        assert!(env.labels.iter().any(|(_, image)| *image != [0; 3]));
        assert_envelopes(
            &Envelope::build(&env, from, to, opts).unwrap(),
            &cache.build(&env, from, to, opts).unwrap(),
        );
        check_gate(&mut cache, &env, from, to, opts, 94025, 0.3, 0.075);
    }
}

#[test]
fn empty_identity_and_zero_activity_paths_are_bitwise_unchanged() {
    let tree = shape(false);
    let mut cache = BodyEnvelopeCache::new(&tree, 0.6).unwrap();
    let old = pose([12.; 3]);
    let new = pose([12.3, 12., 12.]);
    let poses = [old, pose([14.4, 12., 12.])];
    let env = Environment::new(&tree, &poses, 0, old, new, [48.; 3], 0.6).unwrap();
    let empty = Environment::new(&tree, &[old], 0, old, new, [48.; 3], 0.6).unwrap();
    for (environment, from, to, z) in [
        (&env, old, new, 0.),
        (&env, old, old, 0.1),
        (&empty, old, new, 0.1),
    ] {
        let opts = GateOptions::default();
        assert_envelopes(
            &Envelope::build(environment, from, to, opts).unwrap(),
            &cache.build(environment, from, to, opts).unwrap(),
        );
        let result = check_gate(&mut cache, environment, from, to, opts, 6160, 0.4, z);
        assert_eq!(result.raw_points, 0);
        assert_float(result.log_weight, 0., "deterministic gate");
        if z == 0. {
            assert_gates(result, GateResult::default());
        }
    }
}

#[test]
fn cache_rejects_a_different_shape_or_depletant_radius_even_on_fast_paths() {
    let tree = shape(false);
    let other = shape(false); // Equal geometry still has a different owning tree.
    let old = pose([12.; 3]);
    let new = pose([12.3, 12., 12.]);
    let mut cache = BodyEnvelopeCache::new(&tree, 0.6).unwrap();
    for (foreign_tree, rd) in [(&other, 0.6), (&tree, 0.61)] {
        let env = Environment::new(foreign_tree, &[old], 0, old, new, [48.; 3], rd).unwrap();
        assert!(cache.build(&env, old, new, GateOptions::default()).is_err());
        assert!(cache.build(&env, old, old, GateOptions::default()).is_err());
        for z in [0., 0.1] {
            let mut rng = StdRng::seed_from_u64(330);
            let mut untouched = StdRng::seed_from_u64(330);
            assert!(
                cache
                    .sample(&mut rng, &env, old, new, 1., z, GateOptions::default())
                    .is_err()
            );
            assert_eq!(rng.random::<u64>(), untouched.random::<u64>());
        }
    }
    for rd in [-0.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(BodyEnvelopeCache::new(&tree, rd).is_err());
    }
    assert!(BodyEnvelopeCache::new(&tree, 0.).is_ok());
}

#[test]
fn invalid_queries_do_not_escape_through_cached_fast_paths() {
    let tree = shape(false);
    let old = pose([12.; 3]);
    let new = pose([12.3, 12., 12.]);
    let env = Environment::new(&tree, &[old], 0, old, new, [48.; 3], 0.6).unwrap();
    let mut cache = BodyEnvelopeCache::new(&tree, 0.6).unwrap();
    let bad_opts = [
        GateOptions {
            max_cells: 0,
            ..GateOptions::default()
        },
        GateOptions {
            min_width: f64::NAN,
            ..GateOptions::default()
        },
        GateOptions {
            min_width: -0.1,
            ..GateOptions::default()
        },
    ];
    for opts in bad_opts {
        assert!(cache.build(&env, old, new, opts).is_err());
        let mut rng = StdRng::seed_from_u64(950);
        assert!(
            cache
                .sample(&mut rng, &env, old, new, 1., 0., opts)
                .is_err()
        );
    }
    for invalid in [
        Pose {
            position: [f64::NAN, 0., 0.],
            ..old
        },
        Pose {
            orientation: [2., 0., 0., 0.],
            ..old
        },
    ] {
        assert!(
            cache
                .build(&env, old, invalid, GateOptions::default())
                .is_err()
        );
        let mut rng = StdRng::seed_from_u64(951);
        assert!(
            cache
                .sample(&mut rng, &env, old, invalid, 1., 0., GateOptions::default())
                .is_err()
        );
    }
    for (lambda, z) in [
        (0., 0.),
        (f64::NAN, 0.),
        (1., -0.1),
        (1., f64::NAN),
        (f64::MAX, f64::MAX),
    ] {
        let mut rng = StdRng::seed_from_u64(952);
        let mut untouched = StdRng::seed_from_u64(952);
        assert!(
            cache
                .sample(&mut rng, &env, old, new, lambda, z, GateOptions::default())
                .is_err()
        );
        assert_eq!(rng.random::<u64>(), untouched.random::<u64>());
    }
}

#[test]
fn actual_repaired_tetramer_envelopes_match_for_native_neighbor_geometry() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let config: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("examples/spherical-reciprocal-seeded.json")).unwrap(),
    )
    .unwrap();
    let shape_path = root
        .join("examples")
        .join(config["shape"].as_str().unwrap());
    let protein: Shape = serde_json::from_slice(&std::fs::read(shape_path).unwrap()).unwrap();
    assert!(
        protein.atoms.len() > 1000,
        "fixture must be the real sphere-union tetramer"
    );
    let tree = SphereTree::new(protein).unwrap();
    let poses: Vec<Pose> = serde_json::from_value(config["initial_poses"].clone()).unwrap();
    let old = poses[0];
    let env = Environment {
        tree: &tree,
        fixed: poses[1..8].iter().copied().map(Placed::new).collect(),
        labels: (1..8).map(|i| (i, [0; 3])).collect(),
        rd: 1.5,
    };
    let mut cache = BodyEnvelopeCache::new(&tree, 1.5).unwrap();
    for displacement in [[0.05, -0.03, 0.02], [0.4, 0.1, -0.2]] {
        let new = Pose {
            position: add(old.position, displacement),
            orientation: quaternion(matmul(
                cayley([0.002, -0.003, 0.001]),
                rotation(old.orientation),
            )),
        };
        for max_cells in [31, 255, 2047] {
            let opts = GateOptions {
                max_cells,
                ..GateOptions::default()
            };
            for (from, to) in [(old, new), (new, old)] {
                assert_envelopes(
                    &Envelope::build(&env, from, to, opts).unwrap(),
                    &cache.build(&env, from, to, opts).unwrap(),
                );
            }
        }
    }
    // Real geometry gate at deliberately tiny test intensity, not a physical
    // sampling campaign. This also checks exact point predicates and RNG use.
    let new = Pose {
        position: add(old.position, [0.05, -0.03, 0.02]),
        ..old
    };
    check_gate(
        &mut cache,
        &env,
        old,
        new,
        GateOptions::default(),
        704980,
        1e-5,
        2.5e-6,
    );
}

#[test]
fn bounded_cache_falls_back_without_changing_envelopes_or_rng() {
    let tree = shape(true);
    assert!(BodyEnvelopeCache::with_node_limit(&tree, 0.6, 0).is_err());
    let old = pose([12.; 3]);
    let poses = [old, pose([14.4, 12., 12.]), pose([11.5, 14.2, 12.2])];
    for cap in [1, 3, 17] {
        let mut cache = BodyEnvelopeCache::with_node_limit(&tree, 0.6, cap).unwrap();
        assert!(cache.node_count() <= cap);
        for turn in 0..3 {
            let new = Pose {
                position: [12.15 + 0.2 * turn as f64, 12.1, 11.9],
                orientation: quaternion(cayley([0.07 * (turn + 1) as f64, -0.03, 0.02])),
            };
            let env = Environment::new(&tree, &poses, 0, old, new, [48.; 3], 0.6).unwrap();
            for budget in [1, 31, 2047, 3] {
                let opts = GateOptions {
                    max_cells: budget,
                    min_width: 0.,
                    ..GateOptions::default()
                };
                for (from, to) in [(old, new), (new, old)] {
                    assert_envelopes(
                        &Envelope::build(&env, from, to, opts).unwrap(),
                        &cache.build(&env, from, to, opts).unwrap(),
                    );
                    check_gate(
                        &mut cache,
                        &env,
                        from,
                        to,
                        opts,
                        31370 + turn as u64,
                        0.3,
                        0.075,
                    );
                    assert!(
                        cache.node_count() <= cap,
                        "persistent cache exceeded node limit"
                    );
                }
            }
        }
    }
}
