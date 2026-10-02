//! Bounded geometry/count-law tests for the proposal-independent flexible gate.
//! No assembly kernel, physical campaign or convergence claim is involved.
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::f64::consts::PI;
use tetramer_mc::{
    depletion::{Envelope, GateOptions},
    flexible_subset::FlexibleSubset,
    geometry::{Atom, Cell, Shape, SphereTree},
    math::{Pose, cayley, norm, quaternion, sub},
    spherical::Container,
};

fn pose(position: [f64; 3]) -> Pose {
    Pose {
        position,
        orientation: [1., 0., 0., 0.],
    }
}

fn sphere(radius: f64) -> SphereTree {
    SphereTree::new(Shape {
        name: "flexible-subset sphere".into(),
        volume: 4. * PI * radius.powi(3) / 3.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius,
        }],
    })
    .unwrap()
}

fn coarse() -> GateOptions {
    GateOptions {
        max_cells: 1,
        max_depth: 0,
        min_width: 0.,
    }
}

fn next_state(state: &[Pose], members: &[usize], proposed: &[Pose]) -> Vec<Pose> {
    let mut result = state.to_vec();
    for (&label, &p) in members.iter().zip(proposed) {
        result[label] = p;
    }
    result
}

fn brute_union(tree: &SphereTree, state: &[Pose], rd: f64, point: [f64; 3]) -> bool {
    state.iter().any(|p| {
        tree.shape
            .atoms
            .iter()
            .any(|a| norm(sub(point, p.apply(a.center))) <= a.radius + rd)
    })
}

fn same_cell(a: Cell, b: Cell) {
    assert_eq!(a.lo.map(f64::to_bits), b.lo.map(f64::to_bits));
    assert_eq!(a.hi.map(f64::to_bits), b.hi.map(f64::to_bits));
}

fn same_envelope(a: &Envelope, b: &Envelope) {
    assert_eq!(a.created, b.created);
    assert_eq!(a.volume.to_bits(), b.volume.to_bits());
    assert_eq!(
        a.cumulative.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        b.cumulative.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(a.cells.len(), b.cells.len());
    for (&a, &b) in a.cells.iter().zip(&b.cells) {
        same_cell(a, b);
    }
}

#[test]
fn label_pose_and_numeric_failures_are_errors_before_random_draws() {
    let tree = sphere(0.2);
    let state = [pose([0.; 3]), pose([1., 0., 0.]), pose([5., 0., 0.])];
    let proposed = [state[1], state[0]];
    let good = FlexibleSubset::new(&tree, &state, &[1, 0], &proposed, 0.8).unwrap();
    assert_eq!(good.members(), &[1, 0]);
    assert_eq!(good.proposed_poses(), &proposed);
    assert_eq!(good.spectator_count(), 1);
    assert!(FlexibleSubset::new(&tree, &state, &[], &[], 0.8).is_err());
    assert!(FlexibleSubset::new(&tree, &state, &[0, 0], &proposed, 0.8).is_err());
    assert!(FlexibleSubset::new(&tree, &state, &[0, 3], &proposed, 0.8).is_err());
    assert!(FlexibleSubset::new(&tree, &state, &[0], &proposed, 0.8).is_err());
    for rd in [-0.1, f64::NAN, f64::INFINITY, 1e308] {
        assert!(FlexibleSubset::new(&tree, &state, &[0, 1], &proposed, rd).is_err());
    }
    for bad in [
        pose([f64::NAN, 0., 0.]),
        pose([f64::INFINITY, 0., 0.]),
        pose([1e308, 0., 0.]),
        Pose {
            orientation: [0.; 4],
            ..state[0]
        },
        Pose {
            orientation: [1., 0.01, 0., 0.],
            ..state[0]
        },
    ] {
        assert!(FlexibleSubset::new(&tree, &state, &[0], &[bad], 0.8).is_err());
    }
    let mut invalid_spectator = state;
    invalid_spectator[2].position[0] = f64::NAN;
    assert!(FlexibleSubset::new(&tree, &invalid_spectator, &[0], &[state[0]], 0.8).is_err());
    let moving = FlexibleSubset::new(&tree, &state, &[1], &[pose([3., 0., 0.])], 0.8).unwrap();
    let mut rng = StdRng::seed_from_u64(808201);
    let mut untouched = StdRng::seed_from_u64(808201);
    for (lambda, z) in [
        (0., 0.),
        (-1., 0.2),
        (0.5, -0.1),
        (0.5, f64::NAN),
        (f64::INFINITY, 0.2),
        (1e308, 1e308),
        (1e16, 0.2),
    ] {
        assert!(moving.sample(&mut rng, lambda, z, coarse()).is_err());
    }
    for opts in [
        GateOptions {
            max_cells: 0,
            ..coarse()
        },
        GateOptions {
            min_width: f64::NAN,
            ..coarse()
        },
    ] {
        assert!(moving.envelope(opts).is_err());
        assert!(moving.sample(&mut rng, 0.5, 0.2, opts).is_err());
    }
    assert_eq!(rng.random::<u64>(), untouched.random::<u64>());
}

#[test]
fn endpoint_checks_include_internal_pairs_spectators_and_full_wall() {
    let tree = sphere(0.2);
    let state = [pose([0.; 3]), pose([1., 0., 0.]), pose([4., 0., 0.])];
    let wall = Container::new(5., &tree).unwrap();
    let good = FlexibleSubset::new(
        &tree,
        &state,
        &[1, 0],
        &[pose([2., 1., 0.]), pose([1., 1., 0.])],
        0.8,
    )
    .unwrap();
    assert!(good.hard_valid(Some(&wall), [0.; 3]));
    assert!(!good.hard_valid(Some(&wall), [f64::NAN, 0., 0.]));
    let internal = FlexibleSubset::new(
        &tree,
        &state,
        &[0, 1],
        &[pose([1., 0., 0.]), pose([1.1, 0., 0.])],
        0.8,
    )
    .unwrap();
    assert!(!internal.hard_valid(None, [0.; 3]));
    let spectator = FlexibleSubset::new(
        &tree,
        &state,
        &[0, 1],
        &[pose([0., 1., 0.]), pose([4.1, 0., 0.])],
        0.8,
    )
    .unwrap();
    assert!(!spectator.hard_valid(None, [0.; 3]));
    let outside = FlexibleSubset::new(
        &tree,
        &state,
        &[0, 1],
        &[pose([5., 1., 0.]), pose([6., 1., 0.])],
        0.8,
    )
    .unwrap();
    assert!(outside.hard_valid(None, [0.; 3]));
    assert!(!outside.hard_valid(Some(&wall), [0.; 3]));
    assert!(outside.hard_valid(Some(&wall), [4., 0., 0.]));
    // The wall affects only endpoint validity; bath points outside it still
    // belong to the new excluded union and must remain in the world cover.
    assert_eq!(outside.excluded_indicators([6.8, 1., 0.]), (false, true));
    let mut rng = StdRng::seed_from_u64(808202);
    assert!(
        outside
            .sample(&mut rng, 0.5, 0.2, coarse())
            .unwrap()
            .envelope_volume
            > 0.
    );
}

#[test]
fn identity_empty_cover_and_zero_activity_leave_rng_unchanged() {
    let tree = sphere(0.2);
    let state = [pose([0.; 3]), pose([1., 0., 0.])];
    let identity = FlexibleSubset::new(&tree, &state, &[1, 0], &[state[1], state[0]], 0.8).unwrap();
    let empty = identity.envelope(coarse()).unwrap();
    assert!(empty.cells.is_empty());
    assert_eq!(empty.volume, 0.);
    assert_eq!(empty.created, 0);
    let changed =
        FlexibleSubset::new(&tree, &state, &[0, 1], &[state[0], pose([3., 0., 0.])], 0.8).unwrap();
    assert_eq!(changed.spectator_count(), 0);
    assert!(
        changed.envelope(coarse()).unwrap().volume > 0.,
        "an isolated flexible union must not take the rigid isolated-subset shortcut"
    );
    let mut rng = StdRng::seed_from_u64(808203);
    let mut untouched = StdRng::seed_from_u64(808203);
    for gate in [
        identity.sample(&mut rng, 0.5, 0.3, coarse()).unwrap(),
        changed.sample(&mut rng, 0.5, 0., coarse()).unwrap(),
    ] {
        assert_eq!(gate.raw_points, 0);
        assert_eq!(gate.retained_points, 0);
        assert_eq!(gate.log_weight, 0.);
    }
    assert_eq!(rng.random::<u64>(), untouched.random::<u64>());
}

#[test]
fn brute_world_union_preserves_triple_overlap_and_spectator_shielding() {
    let tree = sphere(0.1);
    let rd = 0.9;
    let state = [pose([-0.6, 0., 0.]), pose([0.6, 0., 0.]), pose([0.; 3])];
    let members = [1, 0];
    let proposed = [pose([3.6, 0., 0.]), pose([2.4, 0., 0.])];
    let trial = FlexibleSubset::new(&tree, &state, &members, &proposed, rd).unwrap();
    let next = next_state(&state, &members, &proposed);
    assert_eq!(
        trial.excluded_indicators([0.; 3]),
        (true, true),
        "the spectator shields a point covered by both old selected bodies"
    );
    assert_eq!(
        trial.excluded_indicators([3., 0., 0.]),
        (false, true),
        "two selected exclusions produce one lost-solvent predicate"
    );
    let mut changed = 0;
    for i in -9..=24 {
        for j in -7..=7 {
            for k in -7..=7 {
                let p = [
                    i as f64 * 0.2 + 0.013,
                    j as f64 * 0.2 + 0.017,
                    k as f64 * 0.2 + 0.019,
                ];
                let expected = (
                    brute_union(&tree, &state, rd, p),
                    brute_union(&tree, &next, rd, p),
                );
                assert_eq!(trial.excluded_indicators(p), expected);
                changed += usize::from(expected.0 != expected.1);
            }
        }
    }
    assert!(changed > 100);
    // The old and new selected copies coincide with different fixed spectators.
    // Their hard-invalid intermediate geometry is still a well-defined bath
    // union, and complete shielding gives zero counts without conditioning/retry.
    let shielded = [pose([0.; 3]), pose([0.; 3]), pose([2., 0., 0.])];
    let trial = FlexibleSubset::new(&tree, &shielded, &[0], &[pose([2., 0., 0.])], rd).unwrap();
    assert!(!trial.hard_valid(None, [0.; 3]));
    let mut rng = StdRng::seed_from_u64(808204);
    for _ in 0..16 {
        let gate = trial.sample(&mut rng, 0.7, 0.3, coarse()).unwrap();
        assert_eq!((gate.gained, gate.lost, gate.log_weight), (0, 0, 0.));
    }
}

#[test]
fn swapped_envelopes_match_bitwise_and_partial_traversal_covers_brute_changes() {
    let tree = SphereTree::new(Shape {
        name: "flexible asymmetric union".into(),
        volume: 0.,
        atoms: vec![
            Atom {
                center: [-0.5, 0., 0.],
                radius: 0.4,
            },
            Atom {
                center: [0.6, 0.1, 0.2],
                radius: 0.25,
            },
            Atom {
                center: [0.2, -0.6, 0.1],
                radius: 0.3,
            },
        ],
    })
    .unwrap();
    let members = [1, 0];
    let rd = 0.35;
    for shift in [[0.; 3], [1e6, -2e6, 3e6]] {
        let translated = |p: [f64; 3], c| Pose {
            position: std::array::from_fn(|k| p[k] + shift[k]),
            orientation: quaternion(cayley(c)),
        };
        let state = [
            translated([0.; 3], [0.1, 0.2, -0.3]),
            translated([0.8, 1.3, 0.1], [-0.2, 0.1, 0.3]),
            translated([2.3, 0., 0.], [0.2, -0.1, 0.2]),
        ];
        let proposed = [
            translated([1.4, -0.5, 0.1], [-0.1, 0.4, 0.2]),
            translated([-0.4, 0.8, 0.2], [0.3, -0.2, 0.1]),
        ];
        let next = next_state(&state, &members, &proposed);
        let forward = FlexibleSubset::new(&tree, &state, &members, &proposed, rd).unwrap();
        let reverse =
            FlexibleSubset::new(&tree, &next, &members, &[state[1], state[0]], rd).unwrap();
        same_cell(forward.root_bounds(), reverse.root_bounds());
        for opts in [
            coarse(),
            GateOptions {
                max_cells: 7,
                max_depth: 12,
                min_width: 0.,
            },
            GateOptions {
                max_cells: 255,
                max_depth: 2,
                min_width: 0.,
            },
            GateOptions {
                max_cells: 255,
                max_depth: 12,
                min_width: 0.1,
            },
        ] {
            let envelope = forward.envelope(opts).unwrap();
            same_envelope(&envelope, &reverse.envelope(opts).unwrap());
            assert!(envelope.created <= opts.max_cells);
            for (i, a) in envelope.cells.iter().enumerate() {
                for b in &envelope.cells[..i] {
                    assert!(
                        (0..3).any(|k| a.hi[k] <= b.lo[k] || b.hi[k] <= a.lo[k]),
                        "cover cell interiors overlap"
                    );
                }
            }
            let root = forward.root_bounds();
            let mut rng = StdRng::seed_from_u64(808205);
            let mut changed = 0;
            for _ in 0..6000 {
                let p = std::array::from_fn(|k| {
                    root.lo[k] + rng.random::<f64>() * (root.hi[k] - root.lo[k])
                });
                let expected = (
                    brute_union(&tree, &state, rd, p),
                    brute_union(&tree, &next, rd, p),
                );
                assert_eq!(forward.excluded_indicators(p), expected);
                assert_eq!(reverse.excluded_indicators(p), (expected.1, expected.0));
                if expected.0 != expected.1 {
                    changed += 1;
                    assert_eq!(
                        envelope
                            .cells
                            .iter()
                            .filter(|c| (0..3).all(|k| c.lo[k] <= p[k] && p[k] < c.hi[k]))
                            .count(),
                        1,
                        "partial traversal pruned a brute-force changed point"
                    );
                }
            }
            assert!(changed > 100);
        }
    }
}

#[derive(Default)]
struct Moments {
    sum: f64,
    squares: f64,
}
impl Moments {
    fn add(&mut self, x: f64) {
        self.sum += x;
        self.squares += x * x;
    }
    fn mean(&self, n: usize) -> f64 {
        self.sum / n as f64
    }
    fn variance(&self, n: usize) -> f64 {
        ((self.squares - self.sum * self.sum / n as f64) / (n - 1) as f64).max(0.)
    }
}

#[test]
fn isolated_internal_separation_has_analytic_poisson_counts_and_accepted_flow_balance() {
    let tree = sphere(0.2);
    let rd = 0.8; // Unit exclusion radius; hard core remains radius 0.2.
    let state = [pose([0.; 3]), pose([1., 0., 0.])];
    let next = [state[0], pose([3., 0., 0.])];
    let forward = FlexibleSubset::new(&tree, &state, &[0, 1], &next, rd).unwrap();
    let reverse = FlexibleSubset::new(&tree, &next, &[0, 1], &state, rd).unwrap();
    assert!(forward.hard_valid(None, [0.; 3]) && reverse.hard_valid(None, [0.; 3]));
    assert_eq!(forward.spectator_count(), 0);
    same_envelope(
        &forward.envelope(coarse()).unwrap(),
        &reverse.envelope(coarse()).unwrap(),
    );
    // B0 stays fixed. The old and new moving balls touch at distance 2;
    // the old ball overlaps B0 at d=1, the new ball is disjoint from B0.
    let sphere_volume = 4. * PI / 3.;
    let lens = PI * (4. + 1.) * (2.0_f64 - 1.).powi(2) / 12.;
    let gained_volume = sphere_volume - lens;
    let lost_volume = sphere_volume;
    let lambda = 0.35;
    let z = 0.45;
    let target_ratio = (-z * lens).exp();
    let n = 12_000;
    let mut rng = StdRng::seed_from_u64(808206);
    let mut accepts = [Moments::default(), Moments::default()];
    let mut gained = [Moments::default(), Moments::default()];
    let mut lost = [Moments::default(), Moments::default()];
    let mut raw_points = 0;
    for (index, trial) in [&forward, &reverse].into_iter().enumerate() {
        for _ in 0..n {
            let gate = trial.sample(&mut rng, lambda, z, coarse()).unwrap();
            assert_eq!(gate.retained_points, gate.gained + gate.lost);
            assert!(gate.retained_points <= gate.raw_points);
            assert_eq!((gate.created_cells, gate.retained_cells), (1, 1));
            assert_eq!(
                gate.log_weight,
                (z / lambda).ln_1p() * (gate.gained as f64 - gate.lost as f64)
            );
            raw_points += gate.raw_points;
            gained[index].add(gate.gained as f64);
            lost[index].add(gate.lost as f64);
            accepts[index].add(gate.log_weight.exp().min(1.));
        }
    }
    for (observed, expected) in [
        (&gained[0], lambda * gained_volume),
        (&lost[0], (lambda + z) * lost_volume),
        (&gained[1], lambda * lost_volume),
        (&lost[1], (lambda + z) * gained_volume),
    ] {
        let tolerance = 6. * (expected / n as f64).sqrt();
        assert!(
            (observed.mean(n) - expected).abs() < tolerance,
            "count mean {} differs from analytic {expected} by more than {tolerance}",
            observed.mean(n)
        );
    }
    let flow_error = accepts[0].mean(n) - target_ratio * accepts[1].mean(n);
    let flow_se = ((accepts[0].variance(n) + target_ratio.powi(2) * accepts[1].variance(n))
        / n as f64)
        .sqrt();
    assert!(
        flow_error.abs() < 6. * flow_se + 1e-5,
        "forward flow {}, weighted reverse flow {}, SE {flow_se}",
        accepts[0].mean(n),
        target_ratio * accepts[1].mean(n)
    );
    assert!(raw_points > 100_000);
    assert!(
        accepts[0].mean(n) < 0.7 && accepts[1].mean(n) < 0.95,
        "the world gate has nontrivial gain/loss fluctuations, not a rigid zero gate"
    );
}
