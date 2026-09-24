//! Independent periodic references. No protein model, depletion, or assembly.
//! Out-of-image Gaussian outcomes are retained as self-loops, never redrawn.
use anyhow::Result;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde_json::{Value, json};
use tetramer_mc::{
    basin_involution::BasinTrace,
    docking::{DockingMethod, DockingProposal},
    math::*,
    proposal::FrozenRelativePoseProposal,
};

const SHA: &str = "0000000000000000000000000000000000000000000000000000000000000000";
const WEIGHTS: [f64; 2] = [0.31, 0.69];
const CENTERS: [Vec3; 2] = [[0.45, -0.30, 0.20], [-0.35, 0.25, -0.15]];
const SCALES: [[f64; 6]; 2] = [
    [0.20, 0.25, 0.20, 0.15, 0.20, 0.18],
    [0.28, 0.20, 0.23, 0.21, 0.14, 0.24],
];

fn raw_model(centers: [Vec3; 2], scales: [[f64; 6]; 2], reciprocal: bool) -> Value {
    let covariance: Vec<[[f64; 6]; 6]> = scales
        .iter()
        .map(|s| {
            std::array::from_fn(|i| std::array::from_fn(|j| if i == j { s[i] * s[i] } else { 0. }))
        })
        .collect();
    let base = json!({"coordinate_convention":"anchor-body-relative", "shape_sha256":SHA,
        "angular_length":1., "weights":WEIGHTS,
        "anchors":[{"position":centers[0],"rotation":IDENTITY},
                   {"position":centers[1],"rotation":IDENTITY}],
        "means":vec![[0_f64;6];2], "covariances":covariance});
    if reciprocal {
        json!({"schema":"reciprocal-pose-mixture-v1","base_model":base,
               "reciprocal_components":[true,true]})
    } else {
        base
    }
}

fn periodic_model(raw: &Value, lengths: Vec3, uniform: f64) -> Result<FrozenRelativePoseProposal> {
    FrozenRelativePoseProposal::from_json_str(&raw.to_string(), lengths, uniform, SHA)
}

fn image(v: Vec3, lengths: Vec3) -> Vec3 {
    std::array::from_fn(|i| v[i] - lengths[i] * (v[i] / lengths[i] + 0.5).floor())
}

fn periodic_lab(p: Pose, anchor: Pose, lengths: Vec3) -> Pose {
    let raw = anchor.apply(p.position);
    Pose {
        position: std::array::from_fn(|i| raw[i].rem_euclid(lengths[i])),
        orientation: quaternion(matmul(
            rotation(anchor.orientation),
            rotation(p.orientation),
        )),
    }
}

fn canonical_relative(p: Pose, anchor: Pose, lengths: Vec3) -> Pose {
    let inverse = transpose(rotation(anchor.orientation));
    Pose {
        position: matvec(inverse, image(sub(p.position, anchor.position), lengths)),
        orientation: quaternion(matmul(inverse, rotation(p.orientation))),
    }
}

fn reciprocal(p: Pose) -> Pose {
    let inverse = transpose(rotation(p.orientation));
    Pose {
        position: scale(matvec(inverse, p.position), -1.),
        orientation: quaternion(inverse),
    }
}

fn decode_independent(center: Vec3, scale: [f64; 6], z: [f64; 6]) -> Pose {
    Pose {
        position: std::array::from_fn(|i| center[i] + scale[i] * z[i]),
        orientation: quaternion(cayley(std::array::from_fn(|i| scale[i + 3] * z[i + 3]))),
    }
}

fn in_image(relative: Pose, anchor: Pose, lengths: Vec3) -> bool {
    let d = matvec(rotation(anchor.orientation), relative.position);
    (0..3).all(|i| d[i] >= -0.5 * lengths[i] && d[i] < 0.5 * lengths[i])
}

fn near(a: Pose, b: Pose) {
    assert!(norm(sub(a.position, b.position)) < 2e-10);
    for (x, y) in rotation(a.orientation)
        .iter()
        .flatten()
        .zip(rotation(b.orientation).iter().flatten())
    {
        assert!((x - y).abs() < 2e-10);
    }
}

#[test]
fn every_periodic_reciprocal_trace_recovers_image_noise_and_physical_jacobian() -> Result<()> {
    let lengths = [2.6, 3.4, 4.2];
    let model = periodic_model(&raw_model(CENTERS, SCALES, true), lengths, 0.1)?;
    let anchor = Pose {
        position: [2.55, 3.35, 4.15],
        orientation: quaternion(cayley([0.3, -0.2, 0.4])),
    };
    let z: [f64; 6] = [0.2, -0.3, 0.4, -0.25, 0.15, 0.1];
    let noise: [f64; 6] = [-0.4, 0.1, 0.5, 0.3, -0.2, 0.6];
    let mut crossed = 0;
    for rho in [-1_f64, -0.6, 0., 0.9, 1.] {
        let proposal = DockingProposal::new(
            model.clone(),
            DockingMethod::PosteriorInvolution,
            rho,
            [0.; 3],
        )?;
        for source in 0..4 {
            for target in 0..4 {
                let mut relative = decode_independent(CENTERS[source / 2], SCALES[source / 2], z);
                if source % 2 == 1 {
                    relative = reciprocal(relative);
                }
                assert!(in_image(relative, anchor, lengths));
                let old = periodic_lab(relative, anchor, lengths);
                near(canonical_relative(old, anchor, lengths), relative);
                let trace = BasinTrace {
                    source,
                    target,
                    noise,
                };
                let step = proposal.apply_relative_trace(relative, &trace)?;
                assert!(in_image(step.pose, anchor, lengths));
                let raw = anchor.apply(step.pose.position);
                crossed += usize::from((0..3).any(|i| raw[i] < 0. || raw[i] >= lengths[i]));
                let new = periodic_lab(step.pose, anchor, lengths);
                let canonical_new = canonical_relative(new, anchor, lengths);
                near(canonical_new, step.pose);
                let back = proposal.apply_relative_trace(canonical_new, &step.inverse_trace)?;
                near(back.pose, relative);
                near(periodic_lab(back.pose, anchor, lengths), old);
                let sine = (1. - rho * rho).sqrt();
                let target_z: [f64; 6] = std::array::from_fn(|i| rho * z[i] + sine * noise[i]);
                let inverse_noise: [f64; 6] = std::array::from_fn(|i| sine * z[i] - rho * noise[i]);
                for i in 0..6 {
                    assert!((step.source_latent[i] - z[i]).abs() < 2e-10);
                    assert!((step.target_latent[i] - target_z[i]).abs() < 2e-10);
                    assert!((step.inverse_trace.noise[i] - inverse_noise[i]).abs() < 2e-10);
                    assert!((back.inverse_trace.noise[i] - noise[i]).abs() < 2e-10);
                }
                let old_u2: f64 = (3..6).map(|i| (SCALES[source / 2][i] * z[i]).powi(2)).sum();
                let new_u2: f64 = (3..6)
                    .map(|i| (SCALES[target / 2][i] * target_z[i]).powi(2))
                    .sum();
                let determinant_ratio: f64 = (0..6)
                    .map(|i| (SCALES[target / 2][i] / SCALES[source / 2][i]).ln())
                    .sum();
                let jacobian =
                    determinant_ratio - 2. * (1. + new_u2).ln() + 2. * (1. + old_u2).ln();
                assert!((step.log_extended_jacobian - jacobian).abs() < 2e-10);
                assert!((step.log_extended_jacobian + back.log_extended_jacobian).abs() < 2e-10);
                assert!((step.log_correction + back.log_correction).abs() < 2e-10);
            }
        }
    }
    assert!(
        crossed > 20,
        "Reference must actually cross coordinate cell faces"
    );
    Ok(())
}

#[test]
fn rotated_anchor_uses_world_image_cube_and_records_null_without_redrawing() -> Result<()> {
    let lengths = [2., 6., 8.];
    let centers = [[0., 2., 0.], [2., 0., 0.]];
    let scales = [[1e-6; 6]; 2];
    let model = periodic_model(&raw_model(centers, scales, false), lengths, 1e-9)?;
    let proposal = DockingProposal::new(model, DockingMethod::PosteriorInvolution, 0., [0.; 3])?;
    let anchor = Pose {
        position: [1.8, 5.8, 7.8],
        orientation: [0.5_f64.sqrt(), 0., 0., 0.5_f64.sqrt()],
    };
    let relative = decode_independent(centers[1], scales[1], [0.; 6]);
    let old = periodic_lab(relative, anchor, lengths);
    assert!(relative.position[0] > lengths[0] / 2. && in_image(relative, anchor, lengths));
    let mut rng = StdRng::seed_from_u64(172240111);
    let (mut valid, mut nulls) = (0, 0);
    for _ in 0..512 {
        let (candidate, info) = proposal.propose(&mut rng, old, &[anchor])?;
        assert_ne!(info["branch"], "uniform");
        let trace: BasinTrace = serde_json::from_value(info["trace"].clone())?;
        let step =
            proposal.apply_relative_trace(canonical_relative(old, anchor, lengths), &trace)?;
        if trace.target == 0 {
            assert!(!in_image(step.pose, anchor, lengths));
            assert!(candidate.is_none());
            assert_eq!(info["null_reason"], "outside_unique_image_cube");
            nulls += 1;
        } else {
            let new = candidate.expect("Body-frame-outside but world-frame-inside output is valid");
            near(new, periodic_lab(step.pose, anchor, lengths));
            near(canonical_relative(new, anchor, lengths), step.pose);
            valid += 1;
        }
    }
    assert!(
        nulls > 110 && valid > 250,
        "Both geometry controls must occur without resampling"
    );
    Ok(())
}

#[test]
fn periodic_uniform_branch_is_normalized_over_cell_and_haar_rotation() -> Result<()> {
    let lengths = [2.6, 3.4, 4.2];
    let model = periodic_model(&raw_model(CENTERS, SCALES, true), lengths, 0.8)?;
    let proposal = DockingProposal::new(model, DockingMethod::PosteriorInvolution, 0.9, [0.; 3])?;
    let anchor = Pose {
        position: [2.5, 3.3, 4.1],
        orientation: quaternion(cayley([0.3, -0.2, 0.4])),
    };
    let old = periodic_lab(
        decode_independent(CENTERS[0], SCALES[0], [0.; 6]),
        anchor,
        lengths,
    );
    let mut rng = StdRng::seed_from_u64(172241120);
    let mut sum = [0.; 8];
    let mut square = [0.; 8];
    let mut n = 0.;
    for _ in 0..16000 {
        let (candidate, info) = proposal.propose(&mut rng, old, &[anchor])?;
        if info["branch"] != "uniform" {
            continue;
        }
        let p = candidate.expect("Uniform torus draw has no geometric null");
        assert_eq!(info["log_reverse_forward"].as_f64(), Some(0.));
        assert!((0..3).all(|i| p.position[i] >= 0. && p.position[i] < lengths[i]));
        let x: Vec3 = std::array::from_fn(|i| p.position[i] / lengths[i]);
        let r = rotation(p.orientation);
        let values = [
            x[0],
            x[1],
            x[2],
            x[0] * x[0],
            x[1] * x[1],
            x[2] * x[2],
            r[0][0],
            r[0][0] * r[0][0],
        ];
        for i in 0..8 {
            sum[i] += values[i];
            square[i] += values[i] * values[i];
        }
        n += 1.;
    }
    assert!(n > 12000. && n < 13600.);
    let expected = [0.5, 0.5, 0.5, 1. / 3., 1. / 3., 1. / 3., 0., 1. / 3.];
    for i in 0..8 {
        let mean = sum[i] / n;
        let se = ((square[i] / n - mean * mean) / (n - 1.)).sqrt();
        assert!(
            (mean - expected[i]).abs() < 5.5 * se,
            "Uniform observable {i}: {mean} versus {}",
            expected[i]
        );
    }
    Ok(())
}

#[test]
fn integer_cell_gauges_preserve_capture_and_posterior_proposals() -> Result<()> {
    let lengths = [2.6, 3.4, 4.2];
    let model = periodic_model(&raw_model(CENTERS, SCALES, true), lengths, 0.35)?;
    let anchor = Pose {
        position: [2.5, 3.3, 4.1],
        orientation: quaternion(cayley([0.3, -0.2, 0.4])),
    };
    let old = periodic_lab(
        decode_independent(CENTERS[0], SCALES[0], [0.; 6]),
        anchor,
        lengths,
    );
    let shifted_anchor = Pose {
        position: add(
            anchor.position,
            [-2. * lengths[0], 3. * lengths[1], -lengths[2]],
        ),
        ..anchor
    };
    let shifted_old = Pose {
        position: add(old.position, [lengths[0], -lengths[1], 2. * lengths[2]]),
        ..old
    };
    for method in [DockingMethod::Mixture, DockingMethod::PosteriorInvolution] {
        let first = DockingProposal::new(model.clone(), method, 0.9, [0.; 3])?;
        let second = DockingProposal::new(
            model.clone(),
            method,
            0.9,
            [2. * lengths[0], -lengths[1], 3. * lengths[2]],
        )?;
        let mut left_rng = StdRng::seed_from_u64(172246165);
        let mut right_rng = StdRng::seed_from_u64(172246165);
        for _ in 0..64 {
            let (left, left_info) = first.propose(&mut left_rng, old, &[anchor])?;
            let (right, right_info) =
                second.propose(&mut right_rng, shifted_old, &[shifted_anchor])?;
            assert_eq!(left.is_some(), right.is_some());
            if let (Some(left), Some(right)) = (left, right) {
                near(left, right);
                assert!(
                    (left_info["log_reverse_forward"].as_f64().unwrap()
                        - right_info["log_reverse_forward"].as_f64().unwrap())
                    .abs()
                        < 2e-9
                );
            }
        }
    }
    Ok(())
}

fn integrate_normal_moment(mean: f64, sigma: f64, half: f64, power: i32) -> f64 {
    // Independent deterministic composite Simpson integration, including the
    // normalization constant needed for mixture component truncation masses.
    let panels = 2048;
    let step = 2. * half / panels as f64;
    let mut sum = 0.;
    for i in 0..=panels {
        let x = -half + step * i as f64;
        let density = (-0.5 * ((x - mean) / sigma).powi(2)).exp()
            / (sigma * (2. * std::f64::consts::PI).sqrt());
        let coefficient = if i == 0 || i == panels {
            1.
        } else if i % 2 == 0 {
            2.
        } else {
            4.
        };
        sum += coefficient * density * x.powi(power);
    }
    sum * step / 3.
}

#[derive(Default)]
struct Moments {
    sum: [f64; 6],
    square: [f64; 6],
    count: f64,
}
impl Moments {
    fn add(&mut self, x: [f64; 6]) {
        for i in 0..6 {
            self.sum[i] += x[i];
            self.square[i] += x[i] * x[i];
        }
        self.count += 1.;
    }
    fn check(&self, expected: [f64; 6], label: &str) {
        for i in 0..6 {
            let mean = self.sum[i] / self.count;
            let se = ((self.square[i] / self.count - mean * mean).max(0.) / (self.count - 1.))
                .sqrt()
                .max(1e-14);
            assert!(
                (mean - expected[i]).abs() < 6. * se + 2e-10,
                "{label} observable {i}: mean {mean}, expected {}, SE {se}",
                expected[i]
            );
        }
    }
}

#[test]
fn truncated_periodic_gaussian_mixture_preserves_independent_quadrature_reference() -> Result<()> {
    let lengths = [2., 2.8, 3.6];
    let centers = [[0.8, -0.6, 0.3], [-0.3, 0.7, -0.4]];
    let scales = [
        [0.8, 0.6, 1.0, 0.3, 0.4, 0.25],
        [0.6, 0.9, 0.8, 0.45, 0.25, 0.35],
    ];
    let raw = raw_model(centers, scales, false);
    let model = periodic_model(&raw, lengths, 0.15)?;
    let anchor = Pose {
        position: [1.9, 2.7, 3.5],
        orientation: [1., 0., 0., 0.],
    };
    let masses: [[f64; 3]; 2] = std::array::from_fn(|k| {
        std::array::from_fn(|i| {
            integrate_normal_moment(centers[k][i], scales[k][i], lengths[i] / 2., 0)
        })
    });
    let component_mass: [f64; 2] =
        std::array::from_fn(|k| WEIGHTS[k] * masses[k].iter().product::<f64>());
    let normalizer: f64 = component_mass.iter().sum();
    assert!(
        normalizer > 0.3 && normalizer < 0.8,
        "Reference must have consequential clipped mass"
    );
    let expected: [f64; 6] = std::array::from_fn(|j| {
        let i = j % 3;
        let power = if j < 3 { 1 } else { 2 };
        (0..2)
            .map(|k| {
                component_mass[k]
                    * integrate_normal_moment(centers[k][i], scales[k][i], lengths[i] / 2., power)
                    / masses[k][i]
            })
            .sum::<f64>()
            / normalizer
    });
    let raw_mean_x = WEIGHTS[0] * centers[0][0] + WEIGHTS[1] * centers[1][0];
    assert!(
        (expected[0] - raw_mean_x).abs() > 0.08,
        "Untruncated moments must not accidentally be a valid reference"
    );
    for (index, rho) in [0., 0.7, 0.95].into_iter().enumerate() {
        let proposal = DockingProposal::new(
            model.clone(),
            DockingMethod::PosteriorInvolution,
            rho,
            [0.; 3],
        )?;
        let mut reference = StdRng::seed_from_u64(172242129);
        let mut rng = StdRng::seed_from_u64(172243138 + index as u64);
        let mut before = Moments::default();
        let mut after = Moments::default();
        let mut differences = Moments::default();
        let (mut nulls, mut uniform, mut gaussians) = (0, 0, 0);
        for _ in 0..12000 {
            // Independent exact toy-target sampler: ordinary Gaussian mixture
            // followed by direct geometric rejection into the image cube.
            // Rejection is used only for reference initialization, never for
            // the transition kernel being tested.
            let old_relative = loop {
                let k = usize::from(reference.random::<f64>() >= WEIGHTS[0]);
                let z = std::array::from_fn(|_| StandardNormal.sample(&mut reference));
                let draw = decode_independent(centers[k], scales[k], z);
                if in_image(draw, anchor, lengths) {
                    break draw;
                }
            };
            let old = periodic_lab(old_relative, anchor, lengths);
            let (candidate, info) = proposal.propose(&mut rng, old, &[anchor])?;
            let new_relative = if info["branch"] == "uniform" {
                uniform += 1;
                let next = canonical_relative(candidate.expect("Uniform branch"), anchor, lengths);
                let target_ratio = model
                    .relative_log_density(next.position, rotation(next.orientation))?
                    - model.relative_log_density(
                        old_relative.position,
                        rotation(old_relative.orientation),
                    )?;
                if rng.random::<f64>().ln() < target_ratio.min(0.) {
                    next
                } else {
                    old_relative
                }
            } else {
                gaussians += 1;
                match candidate {
                    Some(next) => {
                        let next = canonical_relative(next, anchor, lengths);
                        let ratio = model.relative_log_density(
                            old_relative.position,
                            rotation(old_relative.orientation),
                        )? - model
                            .relative_log_density(next.position, rotation(next.orientation))?;
                        assert!(
                            (info["log_reverse_forward"].as_f64().unwrap() - ratio).abs() < 2e-9
                        );
                        assert!(
                            (info["expanded_log_reverse_forward"].as_f64().unwrap() - ratio).abs()
                                < 2e-9
                        );
                        next // target G cancels the tested posterior correction
                    }
                    None => {
                        assert_eq!(info["null_reason"], "outside_unique_image_cube");
                        nulls += 1;
                        old_relative
                    }
                }
            };
            let values = |p: Pose| {
                [
                    p.position[0],
                    p.position[1],
                    p.position[2],
                    p.position[0].powi(2),
                    p.position[1].powi(2),
                    p.position[2].powi(2),
                ]
            };
            let (a, b) = (values(old_relative), values(new_relative));
            before.add(a);
            after.add(b);
            differences.add(std::array::from_fn(|i| b[i] - a[i]));
        }
        assert!(
            uniform > 1500 && gaussians > 9500 && nulls > 400,
            "Actual periodic null and uniform branches must be exercised: {uniform} {gaussians} {nulls}"
        );
        before.check(expected, "Independent truncated input");
        after.check(expected, "After full periodic transition");
        differences.check([0.; 6], "Paired stationarity difference");
    }
    Ok(())
}

#[test]
fn clipped_reciprocal_mixture_preserves_joint_orientation_translation_reference() -> Result<()> {
    let lengths = [2., 2.8, 3.6];
    let centers = [[0.8, -0.6, 0.3], [-0.3, 0.7, -0.4]];
    let scales = [
        [0.8, 0.6, 1.0, 0.3, 0.4, 0.25],
        [0.6, 0.9, 0.8, 0.45, 0.25, 0.35],
    ];
    let model = periodic_model(&raw_model(centers, scales, true), lengths, 0.15)?;
    let anchor = Pose {
        position: [1.9, 2.7, 3.5],
        orientation: quaternion(cayley([0.3, -0.2, 0.4])),
    };
    for (index, rho) in [-0.7, 0., 0.9, 1.].into_iter().enumerate() {
        let proposal = DockingProposal::new(
            model.clone(),
            DockingMethod::PosteriorInvolution,
            rho,
            [0.; 3],
        )?;
        let mut reference = StdRng::seed_from_u64(172244147);
        let mut rng = StdRng::seed_from_u64(172245156 + index as u64);
        let mut differences = Moments::default();
        let (mut nulls, mut inversions) = (0, 0);
        for _ in 0..5000 {
            let old_relative = loop {
                let k = usize::from(reference.random::<f64>() >= WEIGHTS[0]);
                let z = std::array::from_fn(|_| StandardNormal.sample(&mut reference));
                let mut draw = decode_independent(centers[k], scales[k], z);
                if reference.random::<bool>() {
                    draw = reciprocal(draw);
                }
                if in_image(draw, anchor, lengths) {
                    break draw;
                }
            };
            let old = periodic_lab(old_relative, anchor, lengths);
            let (candidate, info) = proposal.propose(&mut rng, old, &[anchor])?;
            let next = match candidate {
                Some(candidate) => {
                    let candidate = canonical_relative(candidate, anchor, lengths);
                    let old_g = model.relative_log_density(
                        old_relative.position,
                        rotation(old_relative.orientation),
                    )?;
                    let new_g = model.relative_log_density(
                        candidate.position,
                        rotation(candidate.orientation),
                    )?;
                    if info["branch"] == "uniform" {
                        if rng.random::<f64>().ln() < (new_g - old_g).min(0.) {
                            candidate
                        } else {
                            old_relative
                        }
                    } else {
                        assert!(
                            (info["log_reverse_forward"].as_f64().unwrap() - (old_g - new_g)).abs()
                                < 3e-9
                        );
                        assert!(
                            (info["expanded_log_reverse_forward"].as_f64().unwrap()
                                - (old_g - new_g))
                                .abs()
                                < 3e-9
                        );
                        inversions += usize::from(
                            info["source_inverted"] == true || info["target_inverted"] == true,
                        );
                        candidate
                    }
                }
                None => {
                    assert_eq!(info["null_reason"], "outside_unique_image_cube");
                    nulls += 1;
                    old_relative
                }
            };
            let values = |p: Pose| {
                let r = rotation(p.orientation);
                [
                    p.position[0],
                    p.position[1],
                    p.position[2],
                    dot(p.position, p.position),
                    r[0][0],
                    p.position[0] * r[1][1],
                ]
            };
            let (a, b) = (values(old_relative), values(next));
            differences.add(std::array::from_fn(|i| b[i] - a[i]));
        }
        assert!(
            nulls > 100 && inversions > 1000,
            "Reciprocal and clipped outcomes must both occur"
        );
        differences.check(
            [0.; 6],
            "Joint reciprocal clipped-target paired stationarity",
        );
    }
    Ok(())
}
