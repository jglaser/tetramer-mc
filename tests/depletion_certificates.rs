//! Reusing conservative cell certificates must preserve the original Poisson
//! gate and its exact RNG continuation. These are fixed-geometry tests only.
use rand::{RngExt, SeedableRng, rngs::StdRng};
use tetramer_mc::{
    depletion::{self, BodyEnvelopeCache, ContainmentQueries, Envelope, GateOptions, GateResult},
    geometry::{Atom, Coverage, Environment, Placed, Shape, SphereTree},
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
        name: "cell certificate oracle".into(),
        volume: 0.,
        atoms,
    })
    .unwrap()
}

fn assert_gates(a: GateResult, b: GateResult) {
    assert_eq!(a.gained, b.gained);
    assert_eq!(a.lost, b.lost);
    assert_eq!(a.raw_points, b.raw_points);
    assert_eq!(a.retained_points, b.retained_points);
    assert_eq!(a.retained_cells, b.retained_cells);
    assert_eq!(a.created_cells, b.created_cells);
    assert_eq!(
        a.log_weight.to_bits(),
        b.log_weight.to_bits(),
        "log weight changed"
    );
    assert_eq!(
        a.envelope_volume.to_bits(),
        b.envelope_volume.to_bits(),
        "envelope changed"
    );
}

#[allow(clippy::too_many_arguments)]
fn compare(
    cache: &mut BodyEnvelopeCache<'_>,
    env: &Environment<'_>,
    old: Pose,
    new: Pose,
    opts: GateOptions,
    seed: u64,
    lambda: f64,
    z: f64,
) -> (GateResult, ContainmentQueries) {
    let mut reference_rng = StdRng::seed_from_u64(seed);
    let mut unclassified_rng = StdRng::seed_from_u64(seed);
    let mut certified_rng = StdRng::seed_from_u64(seed);
    let mut profiled_rng = StdRng::seed_from_u64(seed);
    let reference = depletion::sample(&mut reference_rng, env, old, new, lambda, z, opts).unwrap();
    let unclassified = cache
        .sample_unclassified(&mut unclassified_rng, env, old, new, lambda, z, opts)
        .unwrap();
    let certified = cache
        .sample(&mut certified_rng, env, old, new, lambda, z, opts)
        .unwrap();
    let (profiled, queries) = cache
        .sample_profiled(&mut profiled_rng, env, old, new, lambda, z, opts)
        .unwrap();
    for result in [unclassified, certified, profiled] {
        assert_gates(reference, result);
    }
    for _ in 0..8 {
        let expected = reference_rng.random::<u64>();
        assert_eq!(expected, unclassified_rng.random::<u64>());
        assert_eq!(expected, certified_rng.random::<u64>());
        assert_eq!(expected, profiled_rng.random::<u64>());
    }
    assert_eq!(
        queries.body_queries + queries.body_skipped,
        reference.raw_points
    );
    assert_eq!(
        queries.old_queries + queries.old_skipped,
        queries.new_queries + queries.new_skipped
    );
    assert!(queries.old_queries + queries.old_skipped <= reference.raw_points);
    (reference, queries)
}

#[test]
fn certified_sphere_and_dumbbell_gates_preserve_counts_and_rng() {
    let mut skipped = [0_u64; 3];
    let mut gained = 0;
    let mut lost = 0;
    for dumbbell in [false, true] {
        let tree = shape(dumbbell);
        let old = pose([-0.6, 0., 0.]);
        let new = Pose {
            position: [0.6, 0.1, 0.],
            orientation: quaternion(cayley([0.04, -0.07, 0.02])),
        };
        // Deliberate geometric overlap makes all certificate branches visible;
        // these fixed endpoints do not represent hard-particle production moves.
        let env = Environment {
            tree: &tree,
            fixed: vec![Placed::new(pose([1.2, 0., 0.]))],
            labels: vec![(1, [0; 3])],
            rd: 0.6,
        };
        let mut cache = BodyEnvelopeCache::new(&tree, env.rd).unwrap();
        for (from, to) in [(old, new), (new, old)] {
            for budget in [1, 3, 31, 2047, 8191] {
                let opts = GateOptions {
                    max_cells: budget,
                    max_depth: 18,
                    min_width: 0.05,
                };
                for seed in [673521, 940638] {
                    let (gate, q) = compare(&mut cache, &env, from, to, opts, seed, 8., 2.);
                    skipped[0] += q.body_skipped;
                    skipped[1] += q.old_skipped;
                    skipped[2] += q.new_skipped;
                    gained += gate.gained;
                    lost += gate.lost;
                }
            }
        }
    }
    assert!(
        skipped.into_iter().all(|n| n > 0),
        "fixture must exercise all three certificate shortcuts"
    );
    assert!(
        gained > 0 && lost > 0,
        "both retained-point branches must execute"
    );
}

#[test]
fn random_rotations_and_bounded_cache_fallback_preserve_gate() {
    let tree = shape(true);
    let mut fixture_rng = StdRng::seed_from_u64(790320);
    for cap in [1, 3, 127, 65535] {
        let mut cache = BodyEnvelopeCache::with_node_limit(&tree, 0.5, cap).unwrap();
        for case in 0..8 {
            let mut old = uniform_pose(&mut fixture_rng, [48.; 3]);
            old.position = [12.; 3];
            let mut new = uniform_pose(&mut fixture_rng, [48.; 3]);
            new.position =
                std::array::from_fn(|k| old.position[k] + fixture_rng.random_range(-0.7..0.7));
            let mut spectator = uniform_pose(&mut fixture_rng, [48.; 3]);
            spectator.position = [14.1, 12.2, 11.9];
            let poses = [old, spectator, pose([11.7, 14.3, 12.])];
            let env = Environment::new(&tree, &poses, 0, old, new, [48.; 3], 0.5).unwrap();
            let opts = GateOptions {
                max_cells: [31, 2047][case % 2],
                max_depth: [5, 14][case % 2],
                min_width: 0.15,
            };
            compare(
                &mut cache,
                &env,
                old,
                new,
                opts,
                50190 + case as u64,
                1.,
                0.25,
            );
            assert!(cache.node_count() <= cap);
        }
    }
}

#[test]
fn periodic_image_and_endpoint_swaps_preserve_gate() {
    let tree = shape(true);
    let old = pose([0.2, 12., 12.]);
    let new = Pose {
        position: [47.8, 12.2, 12.],
        orientation: quaternion(cayley([0., 0., 0.07])),
    };
    let poses = [old, pose([46.2, 12.3, 12.]), pose([1.9, 12., 12.2])];
    let mut cache = BodyEnvelopeCache::new(&tree, 0.6).unwrap();
    for (from, to) in [(old, new), (new, old)] {
        let env = Environment::new(&tree, &poses, 0, from, to, [48.; 3], 0.6).unwrap();
        assert!(env.labels.iter().any(|(_, image)| *image != [0; 3]));
        compare(
            &mut cache,
            &env,
            from,
            to,
            GateOptions::default(),
            509850,
            3.,
            0.75,
        );
    }
}

#[test]
fn empty_identity_and_zero_activity_skip_all_point_queries() {
    let tree = shape(false);
    let old = pose([12.; 3]);
    let new = pose([12.3, 12., 12.]);
    let poses = [old, pose([14.4, 12., 12.])];
    let env = Environment::new(&tree, &poses, 0, old, new, [48.; 3], 0.6).unwrap();
    let empty = Environment::new(&tree, &[old], 0, old, new, [48.; 3], 0.6).unwrap();
    let mut cache = BodyEnvelopeCache::new(&tree, 0.6).unwrap();
    for (e, to, z) in [(&env, new, 0.), (&env, old, 0.1), (&empty, new, 0.1)] {
        let (gate, q) = compare(
            &mut cache,
            e,
            old,
            to,
            GateOptions::default(),
            70931,
            0.4,
            z,
        );
        assert_eq!(gate.raw_points, 0);
        assert_eq!(
            q.body_queries
                + q.body_skipped
                + q.old_queries
                + q.old_skipped
                + q.new_queries
                + q.new_skipped,
            0
        );
    }
}

fn assert_certificate(status: Coverage, contains: bool) {
    match status {
        Coverage::Inside => assert!(contains, "inside certificate contains an outside point"),
        Coverage::Outside => assert!(!contains, "outside certificate contains an inside point"),
        Coverage::Unknown => {}
    }
}

#[test]
fn retained_cell_certificates_cover_faces_and_near_faces_after_rotation() {
    let mut certified_inside = 0;
    let mut certified_outside = 0;
    let mut uncertain = 0;
    for translation in [0., 1e8] {
        let tree = shape(true);
        let old = Pose {
            position: [translation; 3],
            orientation: quaternion(cayley([0.12, -0.09, 0.05])),
        };
        let new = Pose {
            position: add(old.position, [0.55, 0.2, -0.1]),
            orientation: quaternion(matmul(
                cayley([-0.05, 0.08, 0.02]),
                rotation(old.orientation),
            )),
        };
        let env = Environment {
            tree: &tree,
            fixed: vec![Placed::new(Pose {
                position: add(old.position, [1.2, -0.1, 0.]),
                orientation: quaternion(cayley([0.04, 0.1, -0.02])),
            })],
            labels: vec![(1, [0; 3])],
            rd: 0.6,
        };
        let from = Placed::new(old);
        let to = Placed::new(new);
        let opts = GateOptions {
            max_cells: 4095,
            max_depth: 18,
            min_width: 0.05,
        };
        let envelope = Envelope::build(&env, old, new, opts).unwrap();
        for cell in &envelope.cells {
            let center = cell.center();
            let radius = cell.radius();
            let statuses = [
                tree.classify_ball(center, radius, env.rd),
                env.classify_ball(from.apply(center), radius),
                env.classify_ball(to.apply(center), radius),
            ];
            for status in statuses {
                match status {
                    Coverage::Inside => certified_inside += 1,
                    Coverage::Outside => certified_outside += 1,
                    Coverage::Unknown => uncertain += 1,
                }
            }
            // Include exact corners, all faces, and immediately interior floats.
            let axes: [[f64; 5]; 3] = std::array::from_fn(|k| {
                [
                    cell.lo[k],
                    cell.lo[k].next_up(),
                    center[k],
                    cell.hi[k].next_down(),
                    cell.hi[k],
                ]
            });
            for &x in &axes[0] {
                for &y in &axes[1] {
                    for &z in &axes[2] {
                        let point = [x, y, z];
                        assert_certificate(statuses[0], tree.contains(point, env.rd));
                        assert_certificate(statuses[1], env.contains(from.apply(point)));
                        assert_certificate(statuses[2], env.contains(to.apply(point)));
                    }
                }
            }
        }
    }
    assert!(certified_inside > 0 && certified_outside > 0 && uncertain > 0);
}

#[test]
fn tangent_geometry_is_not_misclassified_as_a_certified_intersection() {
    let tree = shape(false);
    let rd = 0.6;
    for point in [
        [1.6, 0., 0.],
        [1.6_f64.next_up(), 0., 0.],
        [1.6_f64.next_down(), 0., 0.],
    ] {
        assert_eq!(tree.classify_ball(point, 0., rd), Coverage::Unknown);
        // The exact sphere predicate retains its boundary convention.
        let contained = tree.contains(point, rd);
        assert_eq!(contained, point[0] * point[0] <= (1.0 + rd).powi(2));
    }
    let old = pose([0.; 3]);
    for distance in [3.2, 3.2_f64.next_up(), 3.2_f64.next_down()] {
        let new = pose([0.05, 0., 0.]);
        let env = Environment {
            tree: &tree,
            fixed: vec![Placed::new(pose([distance, 0., 0.]))],
            labels: vec![(1, [0; 3])],
            rd,
        };
        let mut cache = BodyEnvelopeCache::new(&tree, rd).unwrap();
        compare(
            &mut cache,
            &env,
            old,
            new,
            GateOptions::default(),
            975093,
            2.,
            0.5,
        );
    }
}

#[test]
fn actual_repaired_tetramer_preserves_certified_gate_and_rng() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let config: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("examples/spherical-reciprocal-seeded.json")).unwrap(),
    )
    .unwrap();
    let shape_path = root
        .join("examples")
        .join(config["shape"].as_str().unwrap());
    let protein: Shape = serde_json::from_slice(&std::fs::read(shape_path).unwrap()).unwrap();
    assert!(protein.atoms.len() > 1000);
    let tree = SphereTree::new(protein).unwrap();
    let poses: Vec<Pose> = serde_json::from_value(config["initial_poses"].clone()).unwrap();
    let old = poses[0];
    let new = Pose {
        position: add(old.position, [0.4, 0.1, -0.2]),
        orientation: quaternion(matmul(
            cayley([0.002, -0.003, 0.001]),
            rotation(old.orientation),
        )),
    };
    let env = Environment {
        tree: &tree,
        fixed: poses[1..8].iter().copied().map(Placed::new).collect(),
        labels: (1..8).map(|i| (i, [0; 3])).collect(),
        rd: 1.5,
    };
    let mut cache = BodyEnvelopeCache::new(&tree, env.rd).unwrap();
    let mut points = 0;
    let mut skipped = 0;
    // Deliberately reduced intensity bounds test runtime; this is not a physical
    // simulation or a statement about production sampling efficiency.
    for (from, to) in [(old, new), (new, old)] {
        let (gate, q) = compare(
            &mut cache,
            &env,
            from,
            to,
            GateOptions::default(),
            943293,
            0.001,
            0.00025,
        );
        points += gate.raw_points;
        skipped += q.body_skipped + q.old_skipped + q.new_skipped;
    }
    assert!(points > 0);
    assert!(
        skipped > 0,
        "protein fixture must use at least one geometric certificate"
    );
}
