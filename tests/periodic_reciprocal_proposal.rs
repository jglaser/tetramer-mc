//! Periodic reciprocal capture is a subdensity plus its unretried null atom.
//! These tests use synthetic Gaussians only; no protein or bath is sampled.

use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde_json::{Value, json};
use std::f64::consts::PI;
use tetramer_mc::math::{
    IDENTITY, Mat3, Pose, cayley, matmul, matvec, quaternion, rotation, transpose,
};
use tetramer_mc::proposal::{FrozenRelativePoseProposal, ProposalBranch};

const SHA: &str = "0000000000000000000000000000000000000000000000000000000000000000";

fn close(a: f64, b: f64, tolerance: f64) {
    assert!(
        a.is_finite() && b.is_finite() && (a - b).abs() <= tolerance,
        "actual={a:.16e}, expected={b:.16e}, tolerance={tolerance:.3e}"
    );
}

fn pose(position: [f64; 3], c: [f64; 3]) -> Pose {
    Pose {
        position,
        orientation: quaternion(cayley(c)),
    }
}

fn model(center: [f64; 3], lower: [[f64; 6]; 6], ell: f64, chart: Mat3, mean: [f64; 6]) -> Value {
    let covariance: [[f64; 6]; 6] = std::array::from_fn(|i| {
        std::array::from_fn(|j| (0..6).map(|k| lower[i][k] * lower[j][k]).sum())
    });
    json!({"angular_length":ell, "weights":[1.],
        "anchors":[{"position":center,"rotation":chart}], "means":[mean],
        "covariances":[covariance], "shape_sha256":SHA,
        "coordinate_convention":"anchor-body-relative"})
}

fn diagonal(values: [f64; 6]) -> [[f64; 6]; 6] {
    std::array::from_fn(|i| std::array::from_fn(|j| if i == j { values[i] } else { 0. }))
}

fn envelope(base: &Value, reciprocal: bool) -> Value {
    json!({"schema":"reciprocal-pose-mixture-v1", "base_model":base,
           "reciprocal_components":[reciprocal]})
}

fn periodic(value: &Value, lengths: [f64; 3], eta: f64) -> FrozenRelativePoseProposal {
    FrozenRelativePoseProposal::from_json_str(&value.to_string(), lengths, eta, SHA).unwrap()
}

fn log_sum(a: f64, b: f64) -> f64 {
    let high = a.max(b);
    if high == f64::NEG_INFINITY {
        return high;
    }
    high + ((a - high).exp() + (b - high).exp()).ln()
}

// Independent formula in physical translation × normalized Haar measure.
// This does not call proposal density, latent inversion or Cholesky helpers.
fn gaussian_log_density(
    t: [f64; 3],
    r: Mat3,
    center: [f64; 3],
    chart: Mat3,
    mean: [f64; 6],
    lower: [[f64; 6]; 6],
    ell: f64,
) -> f64 {
    let delta = matmul(r, transpose(chart));
    let denominator = 1. + delta[0][0] + delta[1][1] + delta[2][2];
    assert!(
        denominator > 1e-8,
        "The independent formula avoids its pi seam"
    );
    let c = [
        (delta[2][1] - delta[1][2]) / denominator,
        (delta[0][2] - delta[2][0]) / denominator,
        (delta[1][0] - delta[0][1]) / denominator,
    ];
    let u: [f64; 6] = std::array::from_fn(|i| {
        if i < 3 {
            t[i] - center[i]
        } else {
            ell * c[i - 3]
        }
    });
    let mut z = [0.; 6];
    for i in 0..6 {
        z[i] = (u[i] - mean[i] - (0..i).map(|j| lower[i][j] * z[j]).sum::<f64>()) / lower[i][i];
    }
    -3. * (2. * PI).ln()
        - (0..6).map(|i| lower[i][i].ln()).sum::<f64>()
        - 0.5 * z.iter().map(|x| x * x).sum::<f64>()
        + 3. * ell.ln()
        + 2. * PI.ln()
        + 2. * (1. + c.iter().map(|x| x * x).sum::<f64>()).ln()
}

#[test]
fn unique_image_density_matches_independent_reciprocal_formula_with_rotated_anchor() {
    let lengths = [5., 7., 9.];
    let eta = 0.23;
    let center = [1.2, -0.4, 0.7];
    let chart = cayley([0.2, -0.1, 0.15]);
    let mean = [0.1, -0.2, 0.05, 0.2, -0.15, 0.1];
    let ell = 2.3;
    let mut lower = diagonal([0.8, 0.6, 0.9, 0.4, 0.7, 0.5]);
    lower[2][0] = 0.2;
    lower[3][1] = -0.1;
    lower[4][0] = 0.15;
    lower[5][2] = -0.25;
    let base = model(center, lower, ell, chart, mean);
    let proposal = periodic(&envelope(&base, true), lengths, eta);
    let anchor = pose([4.6, 6.8, 0.2], [0.4, -0.3, 0.1]);
    for world in [[1.7, -2.1, 0.8], [-2.3, 3.2, -4.], [0.1, 0.2, -0.3]] {
        let candidate = pose(
            std::array::from_fn(|i| anchor.position[i] + world[i]),
            [0.1, 0.3, -0.25],
        );
        let inverse_anchor = transpose(rotation(anchor.orientation));
        let t = matvec(inverse_anchor, world);
        let r = matmul(inverse_anchor, rotation(candidate.orientation));
        let ri = transpose(r);
        let ti = matvec(ri, t).map(|x| -x);
        let learned = log_sum(
            gaussian_log_density(t, r, center, chart, mean, lower, ell),
            gaussian_log_density(ti, ri, center, chart, mean, lower, ell),
        ) - 2_f64.ln();
        let expected = log_sum(
            (eta / lengths.iter().product::<f64>()).ln(),
            (1. - eta).ln() + learned,
        );
        close(
            proposal.log_density(&candidate, &anchor).unwrap(),
            expected,
            2e-12,
        );
        let mut image = candidate;
        image.position =
            std::array::from_fn(|i| candidate.position[i] + [2., -3., 1.][i] * lengths[i]);
        image.orientation = image.orientation.map(|x| -x);
        close(
            proposal.log_density(&image, &anchor).unwrap(),
            expected,
            2e-12,
        );
    }
}

#[test]
fn normalized_reciprocal_gaussian_keeps_analytic_null_mass_without_conditioning() {
    let lengths = [2., 3., 4.];
    let eta = 0.17;
    // Isotropic N(0,I) translations remain N(0,I) after -R^T t for any R.
    // The inverse Cayley orientation has the same centered isotropic law.
    let base = model(
        [0.; 3],
        diagonal([1., 1., 1., 0.9, 0.9, 0.9]),
        1.,
        IDENTITY,
        [0.; 6],
    );
    let proposal = periodic(&envelope(&base, true), lengths, eta);
    let anchor = pose([1.8, 0.1, 3.7], [0.3, -0.2, 0.4]);
    let poses = [pose([0.1, 0.2, 0.3], [0.; 3]), anchor];
    // Exact standard-normal interval probabilities for |x|<1,1.5,2.
    let inside = 0.6826894921370859 * 0.8663855974622838 * 0.9544997361036416;
    let expected_null = (1. - eta) * (1. - inside);
    let mut rng = StdRng::seed_from_u64(718293);
    let draws = 24_000;
    let (mut nulls, mut uniform, mut inverted, mut direct) = (0, 0, 0, 0);
    for _ in 0..draws {
        let result = proposal.propose(&mut rng, &poses, 0).unwrap();
        if result.branch == ProposalBranch::Uniform {
            uniform += 1;
        } else if result.component_inverted == Some(true) {
            inverted += 1;
        } else {
            direct += 1;
        }
        if let Some(candidate) = result.candidate {
            assert!(
                (0..3).all(|i| candidate.position[i] >= 0. && candidate.position[i] < lengths[i])
            );
            close(
                result.log_reverse_forward.unwrap(),
                proposal.log_density(&poses[0], &anchor).unwrap()
                    - proposal.log_density(&candidate, &anchor).unwrap(),
                1e-12,
            );
        } else {
            nulls += 1;
            assert_eq!(result.branch, ProposalBranch::Learned);
            assert_eq!(
                result.null_reason.as_deref(),
                Some("learned_outside_unique_image_cube")
            );
            assert!(result.new_log_density.is_none() && result.log_reverse_forward.is_none());
        }
    }
    for (count, probability) in [
        (nulls, expected_null),
        (uniform, eta),
        (inverted, (1. - eta) / 2.),
        (direct, (1. - eta) / 2.),
    ] {
        close(
            count as f64 / draws as f64,
            probability,
            6. * (probability * (1. - probability) / draws as f64).sqrt(),
        );
    }
    // Independent box × Haar integration measures the continuous mass alone.
    let mut quadrature_rng = StdRng::seed_from_u64(847512);
    let samples = 32_000;
    let (mut sum, mut squares) = (0., 0.);
    for _ in 0..samples {
        let position = std::array::from_fn(|i| quadrature_rng.random::<f64>() * lengths[i]);
        let q: [f64; 4] = std::array::from_fn(|_| StandardNormal.sample(&mut quadrature_rng));
        let norm = q.iter().map(|x| x * x).sum::<f64>().sqrt();
        let p = Pose {
            position,
            orientation: q.map(|x| x / norm),
        };
        let value =
            lengths.iter().product::<f64>() * proposal.log_density(&p, &anchor).unwrap().exp();
        sum += value;
        squares += value * value;
    }
    let integral = sum / samples as f64;
    let variance = (squares / samples as f64 - integral * integral) / (samples - 1) as f64;
    close(integral, 1. - expected_null, 6. * variance.sqrt());
    close(
        integral + nulls as f64 / draws as f64,
        1.,
        6. * (variance + expected_null * (1. - expected_null) / draws as f64).sqrt(),
    );
    assert!(
        nulls > draws / 4,
        "Missing cube mass must not be redrawn away"
    );
}

#[test]
fn cutoff_is_after_reciprocal_inversion_and_world_rotation() {
    let anchor = Pose {
        position: [0.; 3],
        orientation: [(0.5_f64).sqrt(), 0., 0., (0.5_f64).sqrt()],
    };
    let poses = [pose([0.; 3], [0.; 3]), anchor];
    let lengths = [2., 4., 6.];
    for (center, null) in [([1.5, 0., 0.], false), ([0., 1.5, 0.], true)] {
        let base = model(center, diagonal([1e-20; 6]), 1., IDENTITY, [0.; 6]);
        let proposal = periodic(&envelope(&base, true), lengths, 1e-6);
        let mut rng = StdRng::seed_from_u64(99881);
        let mut parity = [0; 2];
        for _ in 0..128 {
            let result = proposal.propose(&mut rng, &poses, 0).unwrap();
            if result.branch == ProposalBranch::Learned {
                parity[usize::from(result.component_inverted.unwrap())] += 1;
                assert_eq!(result.candidate.is_none(), null);
            }
        }
        assert!(parity.iter().all(|count| *count > 30));
    }
}

#[test]
fn half_open_antipodal_faces_keep_opposite_parity_nulls_and_pi_seam_is_uniform() {
    let eta = 1e-6;
    let base = model([1., 0., 0.], diagonal([1e-20; 6]), 1., IDENTITY, [0.; 6]);
    let proposal = periodic(&envelope(&base, true), [2.; 3], eta);
    let origin = pose([0.; 3], [0.; 3]);
    let mut rng = StdRng::seed_from_u64(71322);
    let mut counts = [0; 2];
    for _ in 0..128 {
        let result = proposal.propose(&mut rng, &[origin; 2], 0).unwrap();
        if result.branch == ProposalBranch::Learned {
            let inverse = result.component_inverted.unwrap();
            counts[usize::from(inverse)] += 1;
            assert_eq!(result.candidate.is_none(), !inverse);
        }
    }
    assert!(counts.iter().all(|count| *count > 30));
    let plus = pose([1., 0., 0.], [0.; 3]);
    let minus = pose([-1., 0., 0.], [0.; 3]);
    close(
        proposal.log_density(&plus, &origin).unwrap(),
        proposal.log_density(&minus, &origin).unwrap(),
        0.,
    );
    let seam = Pose {
        position: [0.2, 0.3, 0.4],
        orientation: [0., 1., 0., 0.],
    };
    close(
        proposal.log_density(&seam, &origin).unwrap(),
        (eta / 8.).ln(),
        1e-14,
    );
}

#[test]
fn exterior_peaks_are_nulls_not_lattice_images_added_to_the_density() {
    let lengths = [4., 8., 12.];
    let eta = 0.2;
    let base = model(
        [4., 0., 0.],
        diagonal([0.15, 0.15, 0.15, 0.02, 0.02, 0.02]),
        1.,
        IDENTITY,
        [0.; 6],
    );
    let proposal = periodic(&envelope(&base, true), lengths, eta);
    let origin = pose([0.; 3], [0.; 3]);
    let unique = proposal.log_density(&origin, &origin).unwrap();
    close(unique, (eta / lengths.iter().product::<f64>()).ln(), 1e-14);
    let unwrapped_peak = proposal
        .relative_log_density([4., 0., 0.], IDENTITY)
        .unwrap();
    assert!(
        unwrapped_peak - unique > 10.,
        "A naive wrapped image sum must differ measurably"
    );
    let mut rng = StdRng::seed_from_u64(872631);
    let mut learned = 0;
    for _ in 0..128 {
        let result = proposal.propose(&mut rng, &[origin; 2], 0).unwrap();
        if result.branch == ProposalBranch::Learned {
            learned += 1;
            assert!(result.candidate.is_none());
        }
    }
    assert!(learned > 80);
}

#[test]
fn all_false_periodic_envelope_keeps_legacy_draws_rng_and_serialization() {
    let base = model(
        [1., -0.5, 0.3],
        diagonal([1., 0.7, 0.8, 0.4, 0.5, 0.6]),
        1.,
        IDENTITY,
        [0.; 6],
    );
    let legacy = periodic(&base, [3., 4., 5.], 0.2);
    let wrapped = periodic(&envelope(&base, false), [3., 4., 5.], 0.2);
    let poses = [
        pose([0.1, 0.2, 0.3], [0.1, -0.2, 0.3]),
        pose([2.8, 3.9, 4.7], [0.3, 0.2, -0.1]),
    ];
    let mut old_rng = StdRng::seed_from_u64(283974);
    let mut new_rng = StdRng::seed_from_u64(283974);
    for _ in 0..128 {
        let old = legacy.propose(&mut old_rng, &poses, 0).unwrap();
        let new = wrapped.propose(&mut new_rng, &poses, 0).unwrap();
        assert_eq!(
            serde_json::to_value(old).unwrap(),
            serde_json::to_value(new).unwrap()
        );
    }
    assert_eq!(old_rng.random::<u64>(), new_rng.random::<u64>());
}
