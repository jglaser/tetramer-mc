//! Bounded auxiliary-path controls, separate from production and protein runs.
use anyhow::Result;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde_json::json;
use std::f64::consts::PI;
use tetramer_mc::{
    basin_involution::BasinTrace,
    depletion::{GateOptions, GateResult},
    dimer_tree_proposal::{DimerTreeProposal, DimerTreeTrace},
    docking::{DockingMethod, DockingProposal},
    flexible_subset::FlexibleSubset,
    geometry::{Atom, Shape, SphereTree},
    math::{Pose, norm, sub},
    proposal::FrozenRelativePoseProposal,
    singleton_path::{SingletonOrder, SingletonPath, SingletonPathError, SingletonPathResult},
    spherical::Container,
};

const ORDERS: [SingletonOrder; 2] = [
    SingletonOrder::FirstThenSecond,
    SingletonOrder::SecondThenFirst,
];
fn pose(position: [f64; 3]) -> Pose {
    Pose {
        position,
        orientation: [1., 0., 0., 0.],
    }
}
fn sphere() -> SphereTree {
    SphereTree::new(Shape {
        name: "singleton path sphere".into(),
        volume: 4. * PI * 0.2_f64.powi(3) / 3.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius: 0.2,
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
fn same_pose_bits(a: Pose, b: Pose) {
    assert_eq!(a.position.map(f64::to_bits), b.position.map(f64::to_bits));
    assert_eq!(
        a.orientation.map(f64::to_bits),
        b.orientation.map(f64::to_bits)
    );
}
fn assert_sum(result: &SingletonPathResult, lambda: f64, z: f64) {
    let [a, b] = result.legs;
    let s = result.aggregate;
    assert_eq!(s.gained, a.gained + b.gained);
    assert_eq!(s.lost, a.lost + b.lost);
    assert_eq!(s.raw_points, a.raw_points + b.raw_points);
    assert_eq!(s.retained_points, a.retained_points + b.retained_points);
    assert_eq!(s.retained_points, s.gained + s.lost);
    assert!(s.retained_points <= s.raw_points);
    assert_eq!(s.created_cells, a.created_cells + b.created_cells);
    assert_eq!(s.retained_cells, a.retained_cells + b.retained_cells);
    assert_eq!(s.envelope_volume, a.envelope_volume + b.envelope_volume);
    assert_eq!(s.log_weight, a.log_weight + b.log_weight);
    assert!(
        (s.log_weight - (z / lambda).ln_1p() * (s.gained as f64 - s.lost as f64)).abs() < 1e-12
    );
}
fn empty(gate: GateResult) {
    assert_eq!(
        serde_json::to_value(gate).unwrap(),
        serde_json::to_value(GateResult::default()).unwrap()
    );
}

#[test]
fn invalid_inputs_and_sampling_options_fail_before_order_randomness() {
    let tree = sphere();
    let state = [pose([0.; 3]), pose([1., 0., 0.]), pose([6., 0., 0.])];
    let next = [state[0], pose([3., 0., 0.])];
    for members in [vec![], vec![0], vec![0, 1, 2], vec![0, 0], vec![0, 3]] {
        assert!(SingletonPath::new(&tree, &state, &members, &next, 0.8).is_err());
    }
    assert!(SingletonPath::new(&tree, &state, &[0, 1], &next[..1], 0.8).is_err());
    for rd in [-1., f64::NAN, f64::INFINITY, 1e308] {
        assert!(SingletonPath::new(&tree, &state, &[0, 1], &next, rd).is_err());
    }
    let mut bad = next;
    bad[0].orientation = [0.; 4];
    assert!(SingletonPath::new(&tree, &state, &[0, 1], &bad, 0.8).is_err());
    let mut bad_state = state;
    bad_state[2].position[0] = f64::NAN;
    assert!(SingletonPath::new(&tree, &bad_state, &[0, 1], &next, 0.8).is_err());
    let path = SingletonPath::new(&tree, &state, &[0, 1], &next, 0.8).unwrap();
    let mut rng = StdRng::seed_from_u64(808210);
    let mut untouched = StdRng::seed_from_u64(808210);
    for (lambda, z) in [
        (0., 0.),
        (-1., 0.4),
        (f64::NAN, 0.4),
        (1., -0.1),
        (1., f64::NAN),
        (f64::MAX, f64::MAX),
    ] {
        assert!(path.sample(&mut rng, lambda, z, coarse()).is_err());
        assert!(
            path.sample_with_order(&mut rng, lambda, z, coarse(), ORDERS[0])
                .is_err()
        );
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
        assert!(path.sample(&mut rng, 0.5, 0.4, opts).is_err());
    }
    assert_eq!(rng.random::<u64>(), untouched.random::<u64>());
}

#[test]
fn copied_intermediates_match_reversed_paths_with_noncanonical_label_order() {
    let tree = sphere();
    let old = [pose([0.; 3]), pose([7., 0., 0.]), pose([1., 0., 0.])];
    let members = [2, 0];
    let proposed = [pose([3., 0.5, 0.]), pose([0., 0.5, 0.])];
    let new = [proposed[1], old[1], proposed[0]];
    let forward = SingletonPath::new(&tree, &old, &members, &proposed, 0.8).unwrap();
    let reverse = SingletonPath::new(&tree, &new, &members, &[old[2], old[0]], 0.8).unwrap();
    assert_eq!(forward.members(), &members);
    let mut rng = StdRng::seed_from_u64(808211);
    for order in ORDERS {
        let f = forward
            .sample_with_order(&mut rng, 0.5, 0., coarse(), order)
            .unwrap();
        let r = reverse
            .sample_with_order(&mut rng, 0.5, 0., coarse(), order.reversed())
            .unwrap();
        assert_eq!(
            f.ordered_members,
            [r.ordered_members[1], r.ordered_members[0]]
        );
        let expected = if order == ORDERS[0] {
            [proposed[0], old[0]]
        } else {
            [old[2], proposed[1]]
        };
        for i in 0..2 {
            same_pose_bits(f.intermediate_selected[i], expected[i]);
            same_pose_bits(f.intermediate_selected[i], r.intermediate_selected[i]);
            same_pose_bits(
                f.intermediate_selected[i],
                forward.intermediate_selected(order)[i],
            );
        }
    }
}

#[test]
fn identity_and_zero_activity_keep_coin_but_draw_no_clouds() {
    let tree = sphere();
    let old = [pose([0.; 3]), pose([1., 0., 0.])];
    let identity = SingletonPath::new(&tree, &old, &[0, 1], &old, 0.8).unwrap();
    let changed =
        SingletonPath::new(&tree, &old, &[0, 1], &[old[0], pose([3., 0., 0.])], 0.8).unwrap();
    let mut rng = StdRng::seed_from_u64(808212);
    let mut reference = StdRng::seed_from_u64(808212);
    let mut seen = [0; 2];
    for _ in 0..32 {
        for (path, z) in [(&identity, 0.4), (&changed, 0.)] {
            let index = usize::from(reference.random::<bool>());
            let result = path.sample(&mut rng, 0.5, z, coarse()).unwrap();
            assert_eq!(result.order, ORDERS[index]);
            seen[index] += 1;
            for gate in result.legs {
                empty(gate);
            }
            empty(result.aggregate);
        }
    }
    assert!(seen.iter().all(|&n| n > 0));
    for order in ORDERS {
        empty(
            changed
                .sample_with_order(&mut rng, 0.5, 0., coarse(), order)
                .unwrap()
                .aggregate,
        );
    }
    assert_eq!(rng.random::<u64>(), reference.random::<u64>());
}

#[test]
fn valid_endpoints_allow_core_invalid_intermediates_and_wall_checks_remain_external() {
    let tree = sphere();
    let old = [pose([0.; 3]), pose([1., 0., 0.])];
    let swapped = [old[1], old[0]];
    let path = SingletonPath::new(&tree, &old, &[0, 1], &swapped, 0.8).unwrap();
    let wall = Container::new(2., &tree).unwrap();
    assert!(path.hard_valid(Some(&wall), [0.; 3]));
    let mut rng = StdRng::seed_from_u64(808213);
    for order in ORDERS {
        let middle = path.intermediate_selected(order);
        assert!(
            !FlexibleSubset::new(&tree, &old, &[0, 1], &middle, 0.8)
                .unwrap()
                .hard_valid(None, [0.; 3])
        );
        let mut raw = 0;
        let mut nonzero = 0;
        for _ in 0..32 {
            let result = path
                .sample_with_order(&mut rng, 0.5, 0.4, coarse(), order)
                .unwrap();
            assert_sum(&result, 0.5, 0.4);
            raw += result.aggregate.raw_points;
            nonzero += usize::from(result.aggregate.log_weight != 0.);
        }
        assert!(
            raw > 0 && nonzero > 0,
            "intermediate was silently hard-filtered or bath skipped"
        );
    }
    let outside = [pose([5., 0., 0.]), pose([6., 0., 0.])];
    let path = SingletonPath::new(&tree, &old, &[0, 1], &outside, 0.8).unwrap();
    assert!(path.hard_valid(None, [0.; 3]));
    assert!(!path.hard_valid(Some(&wall), [0.; 3]));
    assert!(path.hard_valid(Some(&wall), [5., 0., 0.]));
    assert!(!path.hard_valid(Some(&wall), [f64::NAN, 0., 0.]));
    let result = path
        .sample_with_order(&mut rng, 0.5, 0.4, coarse(), ORDERS[0])
        .unwrap();
    assert_sum(&result, 0.5, 0.4);
}

#[test]
fn later_leg_errors_retain_completed_diagnostics_without_retry() {
    let tree = sphere();
    let old = [pose([0.; 3]), pose([4., 0., 0.])];
    let next = [pose([10., 0., 0.]), pose([11., 0., 0.])];
    let path = SingletonPath::new(&tree, &old, &[0, 1], &next, 0.8).unwrap();
    // First leg is isolated at both ends and consumes no cloud. The second
    // leg's unsupported Poisson mean is an error with that first result intact.
    let mut rng = StdRng::seed_from_u64(808214);
    let mut untouched = StdRng::seed_from_u64(808214);
    let error = path
        .sample_with_order(&mut rng, 1e16, 0.4, coarse(), ORDERS[0])
        .unwrap_err();
    let detail = error.downcast_ref::<SingletonPathError>().unwrap();
    assert_eq!(detail.order, ORDERS[0]);
    assert_eq!(detail.failed_leg, Some(1));
    assert_eq!(detail.completed_legs.len(), 1);
    empty(detail.completed_legs[0]);
    assert!(detail.reason.contains("unsupported Poisson mean"));
    same_pose_bits(detail.intermediate_selected[0], next[0]);
    same_pose_bits(detail.intermediate_selected[1], old[1]);
    assert_eq!(rng.random::<u64>(), untouched.random::<u64>());
}

fn actual_map_witness() -> Result<([Pose; 2], [Pose; 2], f64, f64)> {
    let sha = "0000000000000000000000000000000000000000000000000000000000000000";
    let covariance: [[f64; 6]; 6] = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            if i == j {
                if i < 3 { 9. } else { 1. }
            } else {
                0.
            }
        })
    });
    let model = json!({"coordinate_convention":"anchor-body-relative","shape_sha256":sha,
        "angular_length":1.,"weights":[1.],"anchors":[{"position":[0.,0.,0.],"rotation":[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]}],
        "means":vec![[0.;6]],"covariances":[covariance]});
    let frozen =
        FrozenRelativePoseProposal::from_json_str_open(&model.to_string(), [20.; 3], 0.1, sha)?;
    let docking = DockingProposal::new(frozen, DockingMethod::PosteriorInvolution, 0., [0.; 3])?;
    let proposal = DimerTreeProposal::new(&docking)?;
    let old = [pose([0.; 3]), pose([1., 0., 0.])];
    let trace = DimerTreeTrace {
        edges: [
            BasinTrace {
                source: 0,
                target: 0,
                noise: [0., 1. / 6., 0., 0., 0., 0.],
            },
            BasinTrace {
                source: 0,
                target: 0,
                noise: [1., 0., 0., 0., 0., 0.],
            },
        ],
    };
    let candidate = proposal
        .apply(pose([0.; 3]), old[0], old[1], &trace)?
        .candidate
        .unwrap();
    let new = [candidate.root, candidate.child];
    let returned = proposal
        .apply(pose([0.; 3]), new[0], new[1], &candidate.inverse_trace)?
        .candidate
        .unwrap();
    assert!(norm(sub(returned.root.position, old[0].position)) < 1e-13);
    assert!(norm(sub(returned.child.position, old[1].position)) < 1e-13);
    let log_r = candidate.diagnostics.log_reverse_forward;
    let reverse_r = returned.diagnostics.log_reverse_forward;
    assert!((log_r - 11. / 24.).abs() < 1e-12 && (log_r + reverse_r).abs() < 1e-12);
    Ok((old, new, log_r, reverse_r))
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
fn actual_nonunit_tree_map_has_balanced_accepted_flow_for_each_reversed_order() -> Result<()> {
    let tree = sphere();
    let (old, new, log_r, reverse_r) = actual_map_witness()?;
    let forward = SingletonPath::new(&tree, &old, &[0, 1], &new, 0.8)?;
    let reverse = SingletonPath::new(&tree, &new, &[0, 1], &old, 0.8)?;
    assert!(forward.hard_valid(None, [0.; 3]) && reverse.hard_valid(None, [0.; 3]));
    assert!((norm(sub(old[0].position, old[1].position)) - 1.).abs() < 1e-13);
    assert!((norm(sub(new[0].position, new[1].position)) - 3.).abs() < 1e-13);
    let lambda = 0.35;
    let z = 0.45;
    // Unit exclusion spheres: old overlap=5pi/12, new overlap=0. The two
    // singleton gates must include this INTERNAL union-volume change even
    // though no physical spectator is present outside the selected pair.
    let weighted_reverse = (log_r - z * 5. * PI / 12.).exp();
    let n = 6_000;
    let mut total_raw = 0;
    for (index, order) in ORDERS.into_iter().enumerate() {
        let mut moments = [Moments::default(), Moments::default()];
        for (direction, path, used_order, correction) in [
            (0, &forward, order, log_r),
            (1, &reverse, order.reversed(), reverse_r),
        ] {
            let mut rng = StdRng::seed_from_u64(808215 + 2 * index as u64 + direction as u64);
            let zero = path.sample_with_order(&mut rng, lambda, 0., coarse(), used_order)?;
            empty(zero.aggregate);
            assert_eq!(
                (correction + zero.aggregate.log_weight).min(0.).exp(),
                correction.min(0.).exp()
            );
            for _ in 0..n {
                let result = path.sample_with_order(&mut rng, lambda, z, coarse(), used_order)?;
                assert_sum(&result, lambda, z);
                total_raw += result.aggregate.raw_points;
                moments[direction].add((correction + result.aggregate.log_weight).min(0.).exp());
            }
        }
        let error = moments[0].mean(n) - weighted_reverse * moments[1].mean(n);
        let se = ((moments[0].variance(n) + weighted_reverse.powi(2) * moments[1].variance(n))
            / n as f64)
            .sqrt();
        assert!(
            error.abs() < 6. * se + 1e-5,
            "{order:?}: flow residual {error}, SE {se}"
        );
        assert!(
            moments[0].mean(n) < 0.85,
            "internal bath change was omitted"
        );
    }
    assert!(total_raw > 50_000);
    Ok(())
}
