use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde::Deserialize;
use serde_json::{Value, json};
use tetramer_mc::math::{
    Mat3, Pose, cayley, invert_relative_pose, matmul, quaternion, rotation, transpose,
};
use tetramer_mc::proposal::{FrozenRelativePoseProposal, ProposalBranch};

const SHA: &str = "0000000000000000000000000000000000000000000000000000000000000000";
const CUBE: [f64; 3] = [12., 14., 16.];

fn close(a: f64, b: f64, tolerance: f64) {
    assert!(
        a.is_finite() && b.is_finite() && (a - b).abs() <= tolerance,
        "actual={a:.16e}, expected={b:.16e}, tolerance={tolerance:.2e}"
    );
}

fn pose(position: [f64; 3], c: [f64; 3]) -> Pose {
    Pose {
        position,
        orientation: quaternion(cayley(c)),
    }
}

fn base_model() -> Value {
    let mut lower = [[0.; 6]; 6];
    for (i, d) in [0.4, 0.6, 0.3, 0.14, 0.18, 0.16].into_iter().enumerate() {
        lower[i][i] = d;
    }
    lower[2][0] = 0.05;
    lower[3][0] = 0.06;
    lower[4][1] = -0.09;
    lower[5][2] = 0.07;
    let covariance: [[f64; 6]; 6] = std::array::from_fn(|i| {
        std::array::from_fn(|j| (0..6).map(|k| lower[i][k] * lower[j][k]).sum())
    });
    json!({"angular_length":2.5,"weights":[0.7,0.3],
        "anchors":[
            {"position":[3.,-1.,0.5],"rotation":cayley([0.3,-0.15,0.07])},
            {"position":[-2.,2.,1.],"rotation":cayley([-0.1,0.25,0.2])}],
        "means":[[0.15,-0.1,0.2,0.08,-0.04,0.03],[-0.1,0.2,0.1,-0.04,0.03,0.08]],
        "covariances":[covariance,covariance],"shape_sha256":SHA,
        "coordinate_convention":"anchor-body-relative"})
}

fn envelope(base: Value, flags: &[bool]) -> Value {
    json!({"schema":"reciprocal-pose-mixture-v1","base_model":base,
        "reciprocal_components":flags})
}

fn open(model: &Value) -> FrozenRelativePoseProposal {
    FrozenRelativePoseProposal::from_json_str_open(&model.to_string(), CUBE, 0.2, SHA).unwrap()
}

fn log_g(model: &FrozenRelativePoseProposal, p: Pose) -> f64 {
    model
        .relative_log_density(p.position, rotation(p.orientation))
        .unwrap()
}

fn log_sum(values: &[f64]) -> f64 {
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    maximum + values.iter().map(|x| (x - maximum).exp()).sum::<f64>().ln()
}

fn compose(anchor: Pose, relative: Pose) -> Pose {
    Pose {
        position: anchor.apply(relative.position),
        orientation: quaternion(matmul(
            rotation(anchor.orientation),
            rotation(relative.orientation),
        )),
    }
}

fn relative(anchor: Pose, other: Pose) -> Pose {
    Pose {
        position: anchor.inverse(other.position),
        orientation: quaternion(matmul(
            transpose(rotation(anchor.orientation)),
            rotation(other.orientation),
        )),
    }
}

fn same_pose(a: Pose, b: Pose, tolerance: f64) {
    for (x, y) in a.position.into_iter().zip(b.position) {
        close(x, y, tolerance);
    }
    for (x, y) in rotation(a.orientation)
        .iter()
        .flatten()
        .zip(rotation(b.orientation).iter().flatten())
    {
        close(*x, *y, tolerance);
    }
}

#[test]
fn reciprocal_pose_is_nonlinear_rigid_inverse_and_involution() {
    for p in [
        pose([3., -7., 2.], [0.4, -0.2, 0.7]),
        Pose {
            position: [-4., 2., 9.],
            orientation: [0., 1., 0., 0.],
        },
        pose([100., -20., 7.], [1e6, -2e6, 1e6]),
    ] {
        let inverse = invert_relative_pose(p);
        same_pose(invert_relative_pose(inverse), p, 2e-13);
        same_pose(compose(p, inverse), pose([0.; 3], [0.; 3]), 2e-13);
        let point = [1., -3., 4.];
        for (actual, expected) in inverse.apply(p.apply(point)).into_iter().zip(point) {
            close(actual, expected, 2e-13);
        }
    }
    let p = pose([3., -7., 2.], [0.4, -0.2, 0.7]);
    let q = invert_relative_pose(p);
    assert!(
        q.position
            .into_iter()
            .zip(p.position)
            .any(|(a, b)| (a + b).abs() > 1.)
    );
    let mut scaled = p;
    scaled.orientation = p.orientation.map(|x| x * (1_f64 + 2e-9).sqrt());
    scaled.validate().unwrap();
    same_pose(invert_relative_pose(scaled), q, 2e-14);
}

fn determinant(mut a: [[f64; 6]; 6]) -> f64 {
    let mut d = 1.;
    for j in 0..6 {
        let pivot = (j..6)
            .max_by(|&i, &k| a[i][j].abs().total_cmp(&a[k][j].abs()))
            .unwrap();
        if pivot != j {
            a.swap(pivot, j);
            d = -d;
        }
        d *= a[j][j];
        for i in j + 1..6 {
            let f = a[i][j] / a[j][j];
            for k in j + 1..6 {
                a[i][k] -= f * a[j][k];
            }
        }
    }
    d
}

fn inverse_in_chart(x: [f64; 6], chart: Mat3) -> [f64; 6] {
    let p = Pose {
        position: [x[0], x[1], x[2]],
        orientation: quaternion(matmul(cayley([x[3], x[4], x[5]]), chart)),
    };
    let inverse = invert_relative_pose(p);
    let q = quaternion(matmul(rotation(inverse.orientation), transpose(chart)));
    [
        inverse.position[0],
        inverse.position[1],
        inverse.position[2],
        q[1] / q[0],
        q[2] / q[0],
        q[3] / q[0],
    ]
}

#[test]
fn inverse_preserves_physical_measure_in_a_nonidentity_cayley_chart() {
    let chart = cayley([0.3, -0.2, 0.1]);
    let x = [3., -2., 5., 0.6, 0.1, -0.4];
    let y = inverse_in_chart(x, chart);
    let h = 1e-6;
    let jacobian: [[f64; 6]; 6] = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let mut plus = x;
            let mut minus = x;
            plus[j] += h;
            minus[j] -= h;
            (inverse_in_chart(plus, chart)[i] - inverse_in_chart(minus, chart)[i]) / (2. * h)
        })
    });
    let euclidean = determinant(jacobian).abs();
    let haar = |v: [f64; 6]| (1. + v[3..].iter().map(|a| a * a).sum::<f64>()).powi(-2);
    assert!(
        (euclidean - 1.).abs() > 0.1,
        "This fixture must exercise the Haar correction"
    );
    close(euclidean * haar(y) / haar(x), 1., 2e-8);
}

#[test]
fn full_reciprocal_density_is_exactly_symmetrized_and_base_parameters_stay_base() {
    let base = open(&base_model());
    let sym = open(&envelope(base_model(), &[true, true]));
    assert!(sym.has_reciprocal_components());
    assert_eq!(sym.component_count(), 2);
    assert_eq!(sym.component_parameters(), base.component_parameters());
    assert_eq!(sym.component_weights(), vec![0.7, 0.3]);
    assert_eq!(sym.reciprocal_components(), vec![true, true]);
    let points = [
        pose([3.1, -1.1, 0.7], [0.3, -0.15, 0.07]),
        pose([-2.2, 2.1, 0.8], [-0.1, 0.25, 0.2]),
        pose([0.1, -0.2, 0.3], [0.7, 0.2, -0.4]),
        pose([5., 3., -1.], [-0.5, 0.4, 0.2]),
    ];
    for p in points {
        let inverse = invert_relative_pose(p);
        let expected = log_sum(&[log_g(&base, p), log_g(&base, inverse)]) - 2_f64.ln();
        close(log_g(&sym, p), expected, 2e-11);
        close(log_g(&sym, inverse), expected, 2e-11);
    }
    assert!((log_g(&base, points[0]) - log_g(&base, invert_relative_pose(points[0]))).abs() > 20.);
}

#[test]
fn partial_reciprocity_preserves_unsplit_mass_and_virtual_order() {
    let model = open(&envelope(base_model(), &[true, false]));
    let branches = model.virtual_branches();
    assert_eq!(branches.len(), 3);
    for (branch, (index, inverted, weight)) in
        branches
            .iter()
            .zip([(0, false, 0.35), (0, true, 0.35), (1, false, 0.3)])
    {
        assert_eq!((branch.component_index, branch.inverted), (index, inverted));
        close(branch.weight, weight, 0.);
    }
    let raw = base_model();
    let single = |index: usize| {
        let mut value = raw.clone();
        for field in ["anchors", "means", "covariances"] {
            value[field] = json!([raw[field][index].clone()]);
        }
        value["weights"] = json!([1.]);
        open(&value)
    };
    let (g0, g1) = (single(0), single(1));
    for p in [
        pose([3., -1., 0.5], [0.3, -0.15, 0.07]),
        pose([-2., 2., 1.], [-0.1, 0.25, 0.2]),
        pose([1., 0., -1.], [0.2, 0.3, 0.1]),
    ] {
        let expected = log_sum(&[
            0.35_f64.ln() + log_g(&g0, p),
            0.35_f64.ln() + log_g(&g0, invert_relative_pose(p)),
            0.3_f64.ln() + log_g(&g1, p),
        ]);
        close(log_g(&model, p), expected, 1e-11);
    }
}

#[test]
fn capture_full_q_keeps_lab_uniform_support_and_hastings_balance() {
    let model = open(&envelope(base_model(), &[true, false]));
    let anchor = pose([20., -3., 2.], [0.4, 0.1, -0.2]);
    let points = [
        pose([0.5, -2., 1.], [0.1, -0.3, 0.2]),
        compose(anchor, pose([3.1, -1.1, 0.7], [0.3, -0.15, 0.07])),
        Pose {
            position: [-6., 0., 0.],
            orientation: [1., 0., 0., 0.],
        },
        Pose {
            position: [6., 0., 0.],
            orientation: [1., 0., 0., 0.],
        },
    ];
    for p in points {
        let learned = 0.8_f64.ln() + log_g(&model, relative(anchor, p));
        let expected =
            if (0..3).all(|k| p.position[k] >= -CUBE[k] / 2. && p.position[k] < CUBE[k] / 2.) {
                log_sum(&[learned, (0.2 / CUBE.iter().product::<f64>()).ln()])
            } else {
                learned
            };
        close(model.log_density(&p, &anchor).unwrap(), expected, 2e-11);
    }
    let target = |p: Pose| {
        -0.01 * p.position.iter().map(|v| v * v).sum::<f64>() + 0.7 * rotation(p.orientation)[0][1]
    };
    let (x, y) = (points[0], points[1]);
    let (qx, qy) = (
        model.log_density(&x, &anchor).unwrap(),
        model.log_density(&y, &anchor).unwrap(),
    );
    let (tx, ty) = (target(x), target(y));
    close(
        tx + qy + (ty - tx + qx - qy).min(0.),
        ty + qx + (tx - ty + qy - qx).min(0.),
        2e-12,
    );
}

#[test]
fn capture_draws_correct_branch_masses_and_inverse_pushforward_gaussians() {
    let model = open(&envelope(base_model(), &[true, false]));
    let anchor = pose([20., -3., 2.], [0.4, 0.1, -0.2]);
    let moving = compose(anchor, pose([3., -1., 0.5], [0.3, -0.15, 0.07]));
    let mut rng = StdRng::seed_from_u64(841145);
    let mut counts = [0usize; 4];
    let mut sums = [[0.; 6]; 3];
    let mut squares = [[[0.; 6]; 6]; 3];
    let mut uniform_sum = [0.; 3];
    let draws = 30_000;
    for _ in 0..draws {
        let draw = model.propose(&mut rng, &[moving, anchor], 0).unwrap();
        assert!(draw.null_reason.is_none());
        let candidate = draw.candidate.unwrap();
        close(
            draw.log_reverse_forward.unwrap(),
            model.log_density(&moving, &anchor).unwrap()
                - model.log_density(&candidate, &anchor).unwrap(),
            1e-12,
        );
        if draw.branch == ProposalBranch::Uniform {
            assert_eq!(draw.component_inverted, None);
            counts[3] += 1;
            for k in 0..3 {
                assert!(
                    candidate.position[k] >= -CUBE[k] / 2. && candidate.position[k] < CUBE[k] / 2.
                );
                uniform_sum[k] += candidate.position[k];
            }
            continue;
        }
        let index = draw.component_index.unwrap();
        let inverted = draw.component_inverted.unwrap();
        let slot = match (index, inverted) {
            (0, false) => 0,
            (0, true) => 1,
            (1, false) => 2,
            _ => panic!("Impossible virtual branch"),
        };
        let mut base_pose = relative(anchor, candidate);
        if inverted {
            base_pose = invert_relative_pose(base_pose);
        }
        let residual = model
            .component_residual(base_pose.position, rotation(base_pose.orientation), index)
            .unwrap();
        counts[slot] += 1;
        for i in 0..6 {
            sums[slot][i] += residual[i];
            for j in 0..6 {
                squares[slot][i][j] += residual[i] * residual[j];
            }
        }
    }
    for (count, probability) in counts.iter().zip([0.28, 0.28, 0.24, 0.2]) {
        let expected = draws as f64 * probability;
        assert!((*count as f64 - expected).abs() < 6. * (expected * (1. - probability)).sqrt());
    }
    for slot in 0..3 {
        for i in 0..6 {
            close(sums[slot][i] / counts[slot] as f64, 0., 0.07);
            for j in 0..6 {
                close(
                    squares[slot][i][j] / counts[slot] as f64,
                    if i == j { 1. } else { 0. },
                    0.1,
                );
            }
        }
    }
    for k in 0..3 {
        close(uniform_sum[k] / counts[3] as f64, 0., 0.15);
    }
}

#[test]
fn allfalse_open_envelope_preserves_legacy_density_draws_rng_and_serialization() {
    let base = base_model();
    let legacy = open(&base);
    let wrapped = open(&envelope(base, &[false, false]));
    assert!(!wrapped.has_reciprocal_components());
    assert_eq!(
        legacy.component_parameters(),
        wrapped.component_parameters()
    );
    let mut old_rng = StdRng::seed_from_u64(675991);
    let mut new_rng = StdRng::seed_from_u64(675991);
    let poses = [
        pose([0.3, -0.5, 0.7], [0.1, 0.2, -0.1]),
        pose([1., 2., -3.], [0.2, -0.3, 0.4]),
    ];
    for _ in 0..512 {
        let old = serde_json::to_value(legacy.propose(&mut old_rng, &poses, 0).unwrap()).unwrap();
        let new = serde_json::to_value(wrapped.propose(&mut new_rng, &poses, 0).unwrap()).unwrap();
        assert_eq!(old, new);
        assert!(new.get("component_inverted").is_none());
    }
    assert_eq!(old_rng.random::<u64>(), new_rng.random::<u64>());
    for p in poses {
        assert_eq!(log_g(&legacy, p).to_bits(), log_g(&wrapped, p).to_bits());
    }
}

#[test]
fn reciprocal_envelope_fails_closed_for_malformed_models_in_either_boundary() {
    let model = envelope(base_model(), &[true, false]);
    #[derive(Deserialize)]
    struct LegacyRequired {
        #[serde(rename = "angular_length")]
        _angular_length: f64,
        #[serde(rename = "anchors")]
        _anchors: Vec<Value>,
    }
    assert!(serde_json::from_value::<LegacyRequired>(model.clone()).is_err());
    let bad = |v: Value| {
        assert!(
            FrozenRelativePoseProposal::from_json_str_open(&v.to_string(), CUBE, 0.2, SHA).is_err(),
            "Accepted open {v}"
        );
        assert!(
            FrozenRelativePoseProposal::from_json_str(&v.to_string(), CUBE, 0.2, SHA).is_err(),
            "Accepted periodic {v}"
        )
    };
    for flags in [&[true, false][..], &[false, false][..]] {
        assert!(
            FrozenRelativePoseProposal::from_json_str(
                &envelope(base_model(), flags).to_string(),
                CUBE,
                0.2,
                SHA
            )
            .is_ok()
        );
    }
    bad(envelope(model.clone(), &[true, false]));
    bad(envelope(base_model(), &[]));
    bad(envelope(base_model(), &[true]));
    bad(envelope(base_model(), &[true, false, true]));
    let mut unknown = model.clone();
    unknown["schema"] = json!("reciprocal-pose-mixture-v2");
    bad(unknown);
    let mut extra = model.clone();
    extra["angular_length"] = json!(2.5);
    bad(extra);
    let mut wrong_type = model.clone();
    wrong_type["reciprocal_components"] = json!([1, false]);
    bad(wrong_type);
    let mut absent = model.clone();
    absent
        .as_object_mut()
        .unwrap()
        .remove("reciprocal_components");
    bad(absent);
    let mut unwrapped = base_model();
    unwrapped["reciprocal_components"] = json!([true, false]);
    bad(unwrapped);
    let mut unwrapped = base_model();
    unwrapped["base_model"] = base_model();
    bad(unwrapped.clone());
    bad(envelope(unwrapped, &[true, false]));
    let mut empty = base_model();
    for field in ["anchors", "means", "covariances", "weights"] {
        empty[field] = json!([]);
    }
    bad(envelope(empty, &[]));
    let mut tiny = base_model();
    tiny["weights"] = json!([1., f64::from_bits(1)]);
    bad(envelope(tiny, &[false, true]));
    assert!(
        FrozenRelativePoseProposal::from_json_str_open(
            &model.to_string(),
            CUBE,
            0.2,
            &"a".repeat(64)
        )
        .is_err()
    );
}

#[test]
fn unsupported_parameter_memory_and_dictionary_changes_reject_active_reciprocity() {
    let active = open(&envelope(base_model(), &[true, false]));
    assert!(
        active
            .with_component_parameters(active.component_parameters())
            .is_err()
    );
    assert!(active.with_whitened_mean_offsets(&[[0.; 6]; 2]).is_err());
    assert!(
        active
            .with_contact_components(&[pose([0.; 3], [0.; 3])], 0.1, 0.2, 1.)
            .is_err()
    );
    for labels in [&[][..], &[0][..], &[0, 1][..]] {
        assert!(active.weighted_subset(labels).is_err());
    }
    assert!(active.selected_components(&[0, 1]).is_err());
    let base = open(&envelope(base_model(), &[false, false]));
    let changed = [
        base.with_component_parameters(base.component_parameters())
            .unwrap(),
        base.with_whitened_mean_offsets(&[[0.; 6]; 2]).unwrap(),
        base.with_contact_components(&[pose([0.; 3], [0.; 3])], 0.1, 0.2, 1.)
            .unwrap(),
        base.weighted_subset(&[]).unwrap(),
        base.weighted_subset(&[1]).unwrap(),
        base.selected_components(&[1, 0, 1]).unwrap(),
    ];
    for model in changed {
        assert_eq!(
            model.reciprocal_components(),
            vec![false; model.component_count()]
        );
        assert_eq!(model.virtual_branches().len(), model.component_count());
    }
}
