use anyhow::Result;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use std::f64::consts::PI;
use tetramer_mc::{
    basin_involution::{BasinPair, BasinTrace, FixedBasinInvolution},
    math::*,
    proposal::{FrozenRelativePoseProposal, GaussianComponentParameters},
};

const SHA: &str = "0000000000000000000000000000000000000000000000000000000000000000";

fn parameters() -> Vec<GaussianComponentParameters> {
    (0..2)
        .map(|j| {
            let scales = if j == 0 {
                [0.7, 1.1, 0.9, 0.35, 0.5, 0.4]
            } else {
                [1.4, 0.8, 1.3, 0.9, 0.65, 1.1]
            };
            let mut lower = [[0.; 6]; 6];
            for i in 0..6 {
                lower[i][i] = scales[i];
            }
            lower[1][0] = 0.2;
            lower[3][0] = 0.17;
            lower[5][2] = -0.15;
            lower[4][3] = 0.12;
            GaussianComponentParameters {
                anchor_position: if j == 0 {
                    [-0.5, 0.1, 0.]
                } else {
                    [0.7, -0.2, 0.3]
                },
                anchor_rotation: if j == 0 {
                    cayley([0.1, -0.2, 0.3])
                } else {
                    cayley([-0.5, 0.7, 0.2])
                },
                mean: [0.1, 0., -0.1, 0.05, -0.1, 0.02],
                covariance: std::array::from_fn(|i| {
                    std::array::from_fn(|k| (0..6).map(|d| lower[i][d] * lower[k][d]).sum())
                }),
                weight: 0.5,
            }
        })
        .collect()
}

fn pairs() -> Vec<BasinPair> {
    vec![
        BasinPair {
            first: 0,
            second: 0,
            weight: 0.4,
        },
        BasinPair {
            first: 0,
            second: 1,
            weight: 2.,
        },
        BasinPair {
            first: 1,
            second: 1,
            weight: 0.6,
        },
    ]
}

fn engine(correlation: f64) -> Result<FixedBasinInvolution> {
    FixedBasinInvolution::new(parameters(), 1.3, correlation, pairs())
}

fn normal<const N: usize>(rng: &mut StdRng) -> [f64; N] {
    std::array::from_fn(|_| StandardNormal.sample(rng))
}

fn pose_error(a: Pose, b: Pose) -> f64 {
    a.position
        .iter()
        .zip(b.position)
        .map(|(x, y)| (x - y).abs())
        .chain(
            rotation(a.orientation)
                .into_iter()
                .flatten()
                .zip(rotation(b.orientation).into_iter().flatten())
                .map(|(x, y)| (x - y).abs()),
        )
        .fold(0., f64::max)
}

#[test]
fn extended_map_is_its_explicit_inverse_and_matches_component_density_ratio() -> Result<()> {
    let models = parameters()
        .into_iter()
        .map(|mut component| {
            component.weight = 1.;
            FrozenRelativePoseProposal::from_components_open(
                vec![component],
                1.3,
                [10.; 3],
                0.1,
                SHA,
                SHA,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let mut rng = StdRng::seed_from_u64(630279);
    let mut max_roundtrip = 0_f64;
    let mut max_ratio = 0_f64;
    for c in [-1., -0.6, 0., 0.6, 1.] {
        let map = engine(c)?;
        for _ in 0..150 {
            let trace = map.draw_trace(&mut rng);
            let pose = map.decode(trace.source, normal(&mut rng))?;
            let forward = map.apply(pose, &trace)?;
            let backward = map.apply(forward.pose, &forward.inverse_trace)?;
            let error = pose_error(pose, backward.pose).max(
                trace
                    .noise
                    .iter()
                    .zip(backward.inverse_trace.noise)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0., f64::max),
            );
            max_roundtrip = max_roundtrip.max(error);
            assert_eq!(backward.inverse_trace.source, trace.source);
            assert_eq!(backward.inverse_trace.target, trace.target);
            assert!((forward.log_correction + backward.log_correction).abs() < 2e-10);
            assert!((forward.log_extended_jacobian + backward.log_extended_jacobian).abs() < 2e-10);
            let independent = models[trace.source]
                .relative_log_density(pose.position, rotation(pose.orientation))?
                - models[trace.target].relative_log_density(
                    forward.pose.position,
                    rotation(forward.pose.orientation),
                )?;
            max_ratio = max_ratio.max((forward.log_correction - independent).abs());
        }
    }
    eprintln!(
        "Involution max pose/noise error={max_roundtrip:.3e}, independent density ratio error={max_ratio:.3e}"
    );
    assert!(max_roundtrip < 2e-11 && max_ratio < 2e-10);
    Ok(())
}

fn pose_from_coordinates(x: [f64; 12]) -> Pose {
    Pose {
        position: [x[0], x[1], x[2]],
        orientation: quaternion(cayley([x[3], x[4], x[5]])),
    }
}

fn coordinates_from_pose(pose: Pose) -> [f64; 6] {
    let q = pose.orientation;
    [
        pose.position[0],
        pose.position[1],
        pose.position[2],
        q[1] / q[0],
        q[2] / q[0],
        q[3] / q[0],
    ]
}

fn log_haar(u: [f64; 3]) -> f64 {
    -2. * PI.ln() - 2. * (1. + u.iter().map(|x| x * x).sum::<f64>()).ln()
}

fn log_abs_det(mut matrix: [[f64; 12]; 12]) -> f64 {
    let mut value = 0.;
    for i in 0..12 {
        let pivot = (i..12)
            .max_by(|&a, &b| matrix[a][i].abs().total_cmp(&matrix[b][i].abs()))
            .unwrap();
        matrix.swap(i, pivot);
        assert!(
            matrix[i][i].abs() > 1e-12,
            "Augmented map must remain nonsingular even at c=0"
        );
        value += matrix[i][i].abs().ln();
        for j in i + 1..12 {
            let factor = matrix[j][i] / matrix[i][i];
            for k in i + 1..12 {
                matrix[j][k] -= factor * matrix[i][k];
            }
        }
    }
    value
}

#[test]
fn finite_difference_augmented_jacobian_includes_rotational_haar_measure() -> Result<()> {
    let input = [
        0.2, -0.7, 0.4, 0.25, -0.3, 0.15, 0.6, -0.2, 0.5, 0.1, -0.8, 0.3,
    ];
    for c in [0., 0.6, 1., -0.8] {
        let map = engine(c)?;
        let transform = |x: [f64; 12]| -> Result<[f64; 12]> {
            let out = map.apply(
                pose_from_coordinates(x),
                &BasinTrace {
                    source: 0,
                    target: 1,
                    noise: std::array::from_fn(|j| x[j + 6]),
                },
            )?;
            let pose = coordinates_from_pose(out.pose);
            Ok(std::array::from_fn(|j| {
                if j < 6 {
                    pose[j]
                } else {
                    out.inverse_trace.noise[j - 6]
                }
            }))
        };
        let output = transform(input)?;
        let mut matrix = [[0.; 12]; 12];
        let step = 1e-6;
        for k in 0..12 {
            let mut plus = input;
            let mut minus = input;
            plus[k] += step;
            minus[k] -= step;
            let a = transform(plus)?;
            let b = transform(minus)?;
            for j in 0..12 {
                matrix[j][k] = (a[j] - b[j]) / (2. * step);
            }
        }
        let finite = log_abs_det(matrix) + log_haar([output[3], output[4], output[5]])
            - log_haar([input[3], input[4], input[5]]);
        let exact = map
            .apply(
                pose_from_coordinates(input),
                &BasinTrace {
                    source: 0,
                    target: 1,
                    noise: std::array::from_fn(|j| input[j + 6]),
                },
            )?
            .log_extended_jacobian;
        eprintln!("c={c}: finite-difference extended logJ={finite:.9}, chart formula={exact:.9}");
        assert!((finite - exact).abs() < 3e-7);
    }
    Ok(())
}

#[derive(Clone, Copy, Default)]
struct Moments {
    n: usize,
    sum: f64,
    sum2: f64,
}
impl Moments {
    fn add(&mut self, x: f64) {
        self.n += 1;
        self.sum += x;
        self.sum2 += x * x;
    }
    fn mean(self) -> f64 {
        self.sum / self.n as f64
    }
    fn z(self, mean: f64) -> f64 {
        let se = ((self.sum2 - self.sum * self.sum / self.n as f64).max(0.)
            / (self.n - 1) as f64
            / self.n as f64)
            .sqrt();
        (self.mean() - mean) / se.max(1e-15)
    }
}

fn observables(pose: Pose) -> [f64; 6] {
    let r = rotation(pose.orientation);
    [
        dot(pose.position, pose.position),
        pose.position[0],
        pose.position[0].powi(2),
        r[0][0],
        r[0][0].powi(2),
        pose.position[0] * r[0][0],
    ]
}

#[test]
fn actual_pose_kernel_preserves_uniform_ball_and_haar_and_detects_missing_jacobian() -> Result<()> {
    let map = engine(0.35)?;
    let radius = 3.;
    let reference = [
        3. * radius * radius / 5.,
        0.,
        radius * radius / 5.,
        0.,
        1. / 3.,
        0.,
    ];
    let mut good = [Moments::default(); 6];
    let mut bad = [Moments::default(); 6];
    let mut differences = [Moments::default(); 6];
    let mut initial = [Moments::default(); 6];
    let mut accepted = [0; 2];
    let mut rng = StdRng::seed_from_u64(719903);
    const N: usize = 16000;
    for _ in 0..N {
        let direction: Vec3 = normal(&mut rng);
        let q: [f64; 4] = normal(&mut rng);
        let qnorm = q.iter().map(|x| x * x).sum::<f64>().sqrt();
        let start = Pose {
            position: scale(
                direction,
                radius * rng.random::<f64>().cbrt() / norm(direction),
            ),
            orientation: q.map(|x| x / qnorm),
        };
        let mut states = [start; 2];
        for _ in 0..5 {
            let trace = map.draw_trace(&mut rng);
            let log_u = rng.random::<f64>().max(f64::MIN_POSITIVE).ln();
            for variant in 0..2 {
                let step = map.apply(states[variant], &trace)?;
                let correction = if variant == 0 {
                    step.log_correction
                } else {
                    step.log_auxiliary_ratio
                };
                if norm(step.pose.position) < radius && log_u < correction.min(0.) {
                    states[variant] = step.pose;
                    accepted[variant] += 1;
                }
            }
        }
        let before = observables(start);
        let a = observables(states[0]);
        let b = observables(states[1]);
        for j in 0..6 {
            initial[j].add(before[j]);
            good[j].add(a[j]);
            bad[j].add(b[j]);
            differences[j].add(a[j] - before[j]);
        }
    }
    let mut negative_power = 0_f64;
    for j in 0..6 {
        eprintln!(
            "Pose control {j}: expected={:.6} correct={:.6} paired_z={:.3} omittedJ={:.6} bad_z={:.3}",
            reference[j],
            good[j].mean(),
            differences[j].z(0.),
            bad[j].mean(),
            bad[j].z(reference[j])
        );
        assert!(initial[j].z(reference[j]).abs() < 5.5);
        assert!(good[j].z(reference[j]).abs() < 5.5);
        assert!(differences[j].z(0.).abs() < 5.5);
        negative_power = negative_power.max(bad[j].z(reference[j]).abs());
    }
    eprintln!(
        "Actual pose accepted with/without J: {accepted:?}; strongest omitted-J deviation {negative_power:.2} SE"
    );
    assert!(accepted[0] > 3000);
    assert!(
        negative_power > 8.,
        "Omitted-J negative control must have power"
    );
    Ok(())
}

#[test]
fn pair_law_enforces_exchange_symmetry_and_invalid_inputs_fail_safely() -> Result<()> {
    let map = engine(0.)?;
    let mut rng = StdRng::seed_from_u64(637123);
    let mut counts = [[0usize; 2]; 2];
    for _ in 0..30000 {
        let trace = map.draw_trace(&mut rng);
        counts[trace.source][trace.target] += 1;
    }
    let probabilities: [[f64; 2]; 2] = [[0.4 / 3., 1. / 3.], [1. / 3., 0.6 / 3.]];
    for i in 0..2 {
        for j in 0..2 {
            let expected = 30000. * probabilities[i][j];
            let se = (expected * (1. - probabilities[i][j])).sqrt();
            assert!((counts[i][j] as f64 - expected).abs() < 5.5 * se);
        }
    }
    assert!(
        FixedBasinInvolution::new(
            parameters(),
            1.,
            0.2,
            vec![BasinPair {
                first: 1,
                second: 0,
                weight: 1.
            }]
        )
        .is_err()
    );
    assert!(FixedBasinInvolution::new(parameters(), 1., 1.1, pairs()).is_err());
    let mut broken = parameters();
    broken[0].covariance[0][0] = -1.;
    assert!(FixedBasinInvolution::new(broken, 1., 0., pairs()).is_err());
    let pose = map.decode(0, [0.; 6])?;
    assert!(
        map.apply(
            pose,
            &BasinTrace {
                source: 0,
                target: 2,
                noise: [0.; 6]
            }
        )
        .is_err()
    );
    assert!(
        map.apply(
            pose,
            &BasinTrace {
                source: 0,
                target: 1,
                noise: [f64::NAN; 6]
            }
        )
        .is_err()
    );
    let seam = Pose {
        position: [0.; 3],
        orientation: quaternion(matmul(
            rotation([0., 1., 0., 0.]),
            parameters()[0].anchor_rotation,
        )),
    };
    // Use an identity chart for an exactly representable pi seam.
    let mut identity = parameters();
    identity[0].anchor_rotation = IDENTITY;
    let seam_map = FixedBasinInvolution::new(identity, 1., 0., pairs())?;
    assert!(
        seam_map
            .encode(
                0,
                Pose {
                    position: seam.position,
                    orientation: [0., 1., 0., 0.]
                }
            )
            .is_err()
    );
    Ok(())
}
