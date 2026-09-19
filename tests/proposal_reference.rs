use rand::{SeedableRng, rngs::StdRng};
use serde::Deserialize;
use serde_json::{Value, json};
use tetramer_mc::math::{
    IDENTITY, Mat3, Pose, Vec3, matmul, minimum_image, quaternion, rotation, transpose,
    uniform_pose,
};
use tetramer_mc::proposal::{FrozenRelativePoseProposal, ProposalBranch};

const MODEL: &str = include_str!("fixtures/proposal_model.json");
const SYNTHETIC: &str = include_str!("fixtures/proposal_synthetic_model.json");
const REFERENCE: &str = include_str!("fixtures/proposal_reference.json");
const ZERO_SHA: &str = "0000000000000000000000000000000000000000000000000000000000000000";

#[derive(Deserialize)]
struct Case {
    label: String,
    old: Pose,
    candidate: Pose,
    anchor: Pose,
    periodic_old: Pose,
    periodic_anchor: Pose,
    expected_old_log_density: f64,
    expected_new_log_density: f64,
    expected_log_reverse_forward: f64,
    relative_position: Vec3,
    relative_rotation: Mat3,
    expected_relative_log_density: f64,
}

#[derive(Deserialize)]
struct Reference {
    shape_sha256: String,
    box_lengths: Vec3,
    uniform_weight: f64,
    cases: Vec<Case>,
}

fn reference() -> Reference {
    serde_json::from_str(REFERENCE).unwrap()
}

fn close(actual: f64, expected: f64, tolerance: f64, label: &str) {
    assert!(
        actual.is_finite() && expected.is_finite(),
        "Nonfinite comparison: {label}"
    );
    assert!(
        (actual - expected).abs() <= tolerance,
        "{label}: actual={actual:0.16e}, expected={expected:0.16e}, error={:0.6e}",
        (actual - expected).abs()
    );
}

fn identity_pose() -> Pose {
    Pose {
        position: [0.; 3],
        orientation: [1., 0., 0., 0.],
    }
}

fn toy_model(mean_x: f64, variance: f64) -> Value {
    let covariance: Vec<Vec<f64>> = (0..6)
        .map(|i| (0..6).map(|j| if i == j { variance } else { 0. }).collect())
        .collect();
    json!({"angular_length":1., "weights":[1.],
        "anchors":[{"position":[0.,0.,0.],"rotation":IDENTITY}],
        "means":[[mean_x,0.,0.,0.,0.,0.]],"covariances":[covariance],
        "shape_sha256":ZERO_SHA,"coordinate_convention":"anchor-body-relative"})
}

#[test]
fn ingress_quaternion_tolerance_matches_geometry() {
    let proposal = FrozenRelativePoseProposal::from_json_str(
        &toy_model(0., 1.).to_string(),
        [20.; 3],
        0.1,
        ZERO_SHA,
    )
    .unwrap();
    let mut pose = identity_pose();
    pose.position = [1., 2., 3.];
    let expected = proposal.log_density(&pose, &identity_pose()).unwrap();
    pose.orientation[0] = (1_f64 + 4e-9).sqrt();
    pose.validate().unwrap();
    close(
        proposal.log_density(&pose, &identity_pose()).unwrap(),
        expected,
        1e-12,
        "normalized ingress quaternion",
    );
}

#[test]
fn python_reference_density_hastings_and_periodic_images() {
    let data = reference();
    let proposal = FrozenRelativePoseProposal::from_json_str(
        MODEL,
        data.box_lengths,
        data.uniform_weight,
        &data.shape_sha256,
    )
    .unwrap();
    assert_eq!(proposal.component_count(), 28);
    let mut maximum_density_error = 0.0_f64;
    for case in &data.cases {
        let old = proposal.log_density(&case.old, &case.anchor).unwrap();
        let new = proposal.log_density(&case.candidate, &case.anchor).unwrap();
        let relative = proposal
            .relative_log_density(case.relative_position, case.relative_rotation)
            .unwrap();
        maximum_density_error =
            maximum_density_error.max((old - case.expected_old_log_density).abs());
        close(old, case.expected_old_log_density, 2e-6, &case.label);
        close(new, case.expected_new_log_density, 2e-6, &case.label);
        close(
            relative,
            case.expected_relative_log_density,
            2e-6,
            &case.label,
        );
        close(
            old - new,
            case.expected_log_reverse_forward,
            4e-6,
            &case.label,
        );
        close(
            proposal
                .log_density(&case.periodic_old, &case.periodic_anchor)
                .unwrap(),
            old,
            2e-6,
            &case.label,
        );
        // An independently specified nonuniform target checks the accepted
        // flux using Python's forward/reverse densities and Rust's correction.
        let target = |p: &Pose| 0.001 * p.position[0] + 0.3 * rotation(p.orientation)[0][0];
        let (px, py) = (target(&case.old), target(&case.candidate));
        let correction = old - new;
        let forward = px + case.expected_new_log_density + (py - px + correction).min(0.0);
        let reverse = py + case.expected_old_log_density + (px - py - correction).min(0.0);
        close(forward, reverse, 4e-6, &format!("flux {}", case.label));
    }
    println!(
        "{} independent Python cases; maximum old-density error {maximum_density_error:e}",
        data.cases.len()
    );
}

#[test]
fn reject_students_incompatible_geometry_and_invalid_models() {
    let data = reference();
    let raw: Value = serde_json::from_str(MODEL).unwrap();
    let make = |value: &Value| {
        FrozenRelativePoseProposal::from_json_str(
            &value.to_string(),
            data.box_lengths,
            0.1,
            &data.shape_sha256,
        )
    };
    assert!(
        FrozenRelativePoseProposal::from_json_str(MODEL, data.box_lengths, 0.1, ZERO_SHA).is_err()
    );
    let mut student = raw.clone();
    let mut dfs = vec![Value::Null; student["weights"].as_array().unwrap().len()];
    dfs[0] = json!(1.);
    student["dfs"] = json!(dfs);
    assert!(make(&student).unwrap_err().to_string().contains("Student"));
    let mut wrong_frame = raw.clone();
    wrong_frame["coordinate_convention"] = json!("world");
    assert!(make(&wrong_frame).is_err());
    let mut zero_weight = raw.clone();
    zero_weight["weights"][0] = json!(0.);
    assert!(make(&zero_weight).is_err());
    let mut indefinite = raw.clone();
    indefinite["covariances"][0][0][0] = json!(-1.);
    assert!(make(&indefinite).is_err());
    let mut asymmetric = raw.clone();
    asymmetric["covariances"][0][0][1] = json!(1e9);
    assert!(make(&asymmetric).is_err());
    let mut scales = raw.clone();
    let covariances = scales
        .as_object_mut()
        .unwrap()
        .remove("covariances")
        .unwrap();
    scales["scales"] = covariances;
    scales["dfs"] = json!(vec![
        Value::Null;
        scales["weights"].as_array().unwrap().len()
    ]);
    let migrated = make(&scales).unwrap();
    let case = &data.cases[0];
    close(
        migrated.log_density(&case.old, &case.anchor).unwrap(),
        case.expected_old_log_density,
        2e-6,
        "Gaussian scale schema",
    );
    scales["covariances"] = scales["scales"].clone();
    assert!(make(&scales).is_err());
    assert!(
        FrozenRelativePoseProposal::from_json_str(MODEL, [0., 1., 1.], 0.1, &data.shape_sha256)
            .is_err()
    );
    assert!(
        FrozenRelativePoseProposal::from_json_str(MODEL, data.box_lengths, 0., &data.shape_sha256)
            .is_err()
    );
}

#[test]
fn half_open_periodic_boundaries_are_null_without_raw_wrapping() {
    let poses = [identity_pose(); 2];
    for (center, expect_null) in [(1., true), (-1., false), (3.1, true), (0.999, false)] {
        // The nonzero SPD variance is so small that adding the sampled x noise
        // rounds exactly to the specified face. No stub replaces Rust's draw.
        let proposal = FrozenRelativePoseProposal::from_json_str(
            &toy_model(center, 1e-40).to_string(),
            [2.; 3],
            1e-6,
            ZERO_SHA,
        )
        .unwrap();
        let mut rng = StdRng::seed_from_u64(18841);
        let mut learned = 0;
        for _ in 0..100 {
            let result = proposal.propose(&mut rng, &poses, 0).unwrap();
            if result.branch == ProposalBranch::Learned {
                learned += 1;
                assert_eq!(
                    result.candidate.is_none(),
                    expect_null,
                    "raw center {center}"
                );
                if expect_null {
                    assert_eq!(
                        result.null_reason.as_deref(),
                        Some("learned_outside_unique_image_cube")
                    );
                    assert!(
                        result.new_log_density.is_none() && result.log_reverse_forward.is_none()
                    );
                }
            }
        }
        assert!(learned >= 99);
    }
}

#[test]
fn sampling_normalization_uniform_anchor_and_null_mass() {
    let fixture: Value = serde_json::from_str(REFERENCE).unwrap();
    let analytic = &fixture["unit_sphere_gaussian_sampling"];
    let samples = analytic["samples"].as_u64().unwrap() as usize;
    let expected_null = analytic["expected_null_mass"].as_f64().unwrap();
    let proposal = FrozenRelativePoseProposal::from_json_str(
        &toy_model(0., 1.).to_string(),
        [2.; 3],
        0.1,
        ZERO_SHA,
    )
    .unwrap();
    let poses = [identity_pose(); 5];
    let mut rng = StdRng::seed_from_u64(331784);
    let (mut nulls, mut uniforms) = (0_usize, 0_usize);
    let mut anchor_counts = [0_usize; 4];
    for _ in 0..samples {
        let result = proposal.propose(&mut rng, &poses, 0).unwrap();
        assert_ne!(result.anchor_index, 0);
        anchor_counts[result.anchor_index - 1] += 1;
        if result.branch == ProposalBranch::Uniform {
            uniforms += 1;
        }
        match result.candidate {
            None => {
                nulls += 1;
                assert_eq!(result.branch, ProposalBranch::Learned);
            }
            Some(candidate) => {
                assert!(candidate.position.iter().all(|v| *v >= 0. && *v < 2.));
                let old = proposal
                    .log_density(&poses[0], &poses[result.anchor_index])
                    .unwrap();
                let new = proposal
                    .log_density(&candidate, &poses[result.anchor_index])
                    .unwrap();
                close(
                    result.log_reverse_forward.unwrap(),
                    old - new,
                    1e-12,
                    "sample correction",
                );
            }
        }
    }
    let null_rate = nulls as f64 / samples as f64;
    let null_se = (expected_null * (1. - expected_null) / samples as f64).sqrt();
    close(
        null_rate,
        expected_null,
        6. * null_se,
        "analytic null probability",
    );
    close(
        uniforms as f64 / samples as f64,
        0.1,
        6. * (0.09 / samples as f64).sqrt(),
        "uniform branch probability",
    );
    for count in anchor_counts {
        close(
            count as f64 / samples as f64,
            0.25,
            6. * (0.1875 / samples as f64).sqrt(),
            "uniform other anchor",
        );
    }
    // Independent uniform-box/Haar quadrature of the continuous subdensity.
    let n = 16000;
    let mut sum = 0.;
    let mut squares = 0.;
    for _ in 0..n {
        let x = uniform_pose(&mut rng, [2.; 3]);
        let value = 8. * proposal.log_density(&x, &identity_pose()).unwrap().exp();
        sum += value;
        squares += value * value;
    }
    let integrated = sum / n as f64;
    let integration_se = ((squares / n as f64 - integrated * integrated) / (n - 1) as f64).sqrt();
    close(
        integrated,
        1. - expected_null,
        6. * integration_se,
        "offdiagonal subdensity integral",
    );
    close(
        integrated + null_rate,
        1.,
        6. * (integration_se * integration_se + null_se * null_se).sqrt(),
        "offdiagonal plus null normalization",
    );
}

#[test]
fn full_covariance_samples_and_uniform_haar_moments() {
    let raw: Value = serde_json::from_str(SYNTHETIC).unwrap();
    let mean: [f64; 6] = serde_json::from_value(raw["means"][0].clone()).unwrap();
    let covariance: [[f64; 6]; 6] = serde_json::from_value(raw["covariances"][0].clone()).unwrap();
    let chart_position: Vec3 =
        serde_json::from_value(raw["anchors"][0]["position"].clone()).unwrap();
    let chart_rotation: Mat3 =
        serde_json::from_value(raw["anchors"][0]["rotation"].clone()).unwrap();
    let ell = raw["angular_length"].as_f64().unwrap();
    let proposal =
        FrozenRelativePoseProposal::from_json_str(SYNTHETIC, [1000.; 3], 0.1, ZERO_SHA).unwrap();
    let poses = [identity_pose(); 2];
    let mut rng = StdRng::seed_from_u64(731728);
    let mut sums = [0.; 6];
    let mut products = [[0.; 6]; 6];
    let mut rotations_sum = [[0.; 3]; 3];
    let mut positions_sum = [0.; 3];
    let (mut learned, mut uniform) = (0_usize, 0_usize);
    for _ in 0..40000 {
        let trial = proposal.propose(&mut rng, &poses, 0).unwrap();
        let candidate = trial
            .candidate
            .expect("All synthetic Gaussian draws should lie well inside this very large box");
        if trial.branch == ProposalBranch::Uniform {
            uniform += 1;
            let r = rotation(candidate.orientation);
            for i in 0..3 {
                positions_sum[i] += candidate.position[i];
                for j in 0..3 {
                    rotations_sum[i][j] += r[i][j];
                }
            }
        } else {
            learned += 1;
            let t = minimum_image(candidate.position, [1000.; 3]);
            let q = quaternion(matmul(
                rotation(candidate.orientation),
                transpose(chart_rotation),
            ));
            let latent: [f64; 6] = std::array::from_fn(|i| {
                if i < 3 {
                    t[i] - chart_position[i]
                } else {
                    ell * q[i - 2] / q[0]
                }
            });
            for i in 0..6 {
                sums[i] += latent[i];
                for j in 0..6 {
                    products[i][j] += latent[i] * latent[j];
                }
            }
        }
    }
    for i in 0..6 {
        let estimated_mean = sums[i] / learned as f64;
        close(
            estimated_mean,
            mean[i],
            6. * (covariance[i][i] / learned as f64).sqrt(),
            "Gaussian sampled mean",
        );
        for j in 0..6 {
            let observed =
                (products[i][j] - sums[i] * sums[j] / learned as f64) / (learned - 1) as f64;
            let se = ((covariance[i][j] * covariance[i][j] + covariance[i][i] * covariance[j][j])
                / (learned - 1) as f64)
                .sqrt();
            close(
                observed,
                covariance[i][j],
                6. * se,
                "Gaussian sampled full covariance",
            );
        }
    }
    for i in 0..3 {
        close(
            positions_sum[i] / uniform as f64,
            500.,
            6. * 1000. / (12. * uniform as f64).sqrt(),
            "Uniform center mean",
        );
        for j in 0..3 {
            close(
                rotations_sum[i][j] / uniform as f64,
                0.,
                6. / (3. * uniform as f64).sqrt(),
                "Haar rotation mean",
            );
        }
    }
}
