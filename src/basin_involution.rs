//! A reversible Gaussian-chart map with an explicit inverse auxiliary trace.
//!
//! Charts and the unordered pair law are immutable during a physical move.
//! They may be built from spectators, but never from the moving pose. In each
//! chart x=mean+L*z with Cayley rotational coordinates. The extended update
//! (z,xi)->(c*z+s*xi,s*z-c*xi), s=sqrt(1-c*c), is an orthogonal involution;
//! swapping source/target charts supplies the exact inverse construction.
//!
//! This is not a symmetric ordinary pose proposal or a rejection-free move.
//! Its MH correction includes the auxiliary Gaussian ratio and the extended
//! translation/Haar volume ratio. No separately coded reverse density exists.
use anyhow::{Result, ensure};
use rand::{RngExt, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use crate::{
    math::{Pose, cayley, matmul, quaternion, rotation, transpose},
    proposal::GaussianComponentParameters,
};

type Vec6 = [f64; 6];
type Mat6 = [[f64; 6]; 6];

/// One unordered pair. A fair direction coin makes its directed law symmetric.
/// Self pairs are allowed and need no direction coin. Pairs must be unique and
/// canonical (first <= second); weights need not already sum to one.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BasinPair {
    pub first: usize,
    pub second: usize,
    pub weight: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BasinTrace {
    pub source: usize,
    pub target: usize,
    pub noise: Vec6,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BasinStep {
    pub pose: Pose,
    /// Applying the same map to pose and this trace recovers the input state.
    pub inverse_trace: BasinTrace,
    pub source_latent: Vec6,
    pub target_latent: Vec6,
    /// Jacobian of the full (pose, noise) map, with normalized Haar orientation.
    /// This is not the fixed-noise pose derivative (which is singular at c=0).
    pub log_extended_jacobian: f64,
    pub log_auxiliary_ratio: f64,
    /// Add to the exact physical/depletion acceptance log factor.
    pub log_correction: f64,
}

#[derive(Clone, Debug)]
struct Chart {
    parameters: GaussianComponentParameters,
    lower: Mat6,
    log_determinant: f64,
}

#[derive(Clone, Debug)]
pub struct FixedBasinInvolution {
    charts: Vec<Chart>,
    angular_length: f64,
    correlation: f64,
    sine: f64,
    pairs: Vec<BasinPair>,
    cumulative: Vec<f64>,
    /// Certified by the constructor's range/canonical/uniqueness checks and
    /// the cardinality of all unordered pairs, including self pairs.
    complete_pair_support: bool,
}

fn complete_pair_count(charts: usize) -> Option<usize> {
    // Divide the even factor first: n(n+1) may overflow even when its half fits.
    let next = charts.checked_add(1)?;
    if charts % 2 == 0 {
        (charts / 2).checked_mul(next)
    } else {
        charts.checked_mul(next / 2)
    }
}

fn cholesky(covariance: Mat6) -> Result<(Mat6, f64)> {
    ensure!(
        covariance.iter().flatten().all(|x| x.is_finite()),
        "Nonfinite chart covariance"
    );
    let magnitude = covariance
        .iter()
        .flatten()
        .fold(0_f64, |a, b| a.max(b.abs()));
    ensure!(magnitude > 0., "Chart covariance is not positive definite");
    let mut symmetric = [[0.; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            ensure!(
                (covariance[i][j] - covariance[j][i]).abs()
                    <= 1e-12 * (magnitude + covariance[i][j].abs()),
                "Asymmetric chart covariance"
            );
            symmetric[i][j] = 0.5 * (covariance[i][j] + covariance[j][i]);
        }
    }
    let mut lower = [[0.; 6]; 6];
    for i in 0..6 {
        for j in 0..=i {
            let residual = symmetric[i][j] - (0..j).map(|k| lower[i][k] * lower[j][k]).sum::<f64>();
            lower[i][j] = if i == j {
                ensure!(
                    residual > 0. && residual.is_finite(),
                    "Chart covariance is not positive definite"
                );
                residual.sqrt()
            } else {
                residual / lower[j][j]
            };
            ensure!(
                lower[i][j].is_finite(),
                "Unrepresentable chart Cholesky factor"
            );
        }
    }
    let log_determinant = (0..6).map(|i| lower[i][i].ln()).sum::<f64>();
    ensure!(
        log_determinant.is_finite(),
        "Unrepresentable chart determinant"
    );
    Ok((lower, log_determinant))
}

fn chart_parameters_valid(p: &GaussianComponentParameters) -> Result<()> {
    ensure!(
        p.anchor_position
            .iter()
            .chain(p.mean.iter())
            .all(|x| x.is_finite())
            && p.anchor_rotation.iter().flatten().all(|x| x.is_finite()),
        "Nonfinite chart metadata"
    );
    let reconstructed = rotation(quaternion(p.anchor_rotation));
    ensure!(
        reconstructed
            .iter()
            .flatten()
            .zip(p.anchor_rotation.iter().flatten())
            .all(|(a, b)| (a - b).abs() <= 1e-10),
        "Chart anchor must be a proper rotation"
    );
    Ok(())
}

impl FixedBasinInvolution {
    /// All poses and chart anchors must use one fixed common coordinate frame.
    /// Gaussian component weights are metadata only: pair selection uses pairs.
    pub fn new(
        parameters: Vec<GaussianComponentParameters>,
        angular_length: f64,
        correlation: f64,
        pairs: Vec<BasinPair>,
    ) -> Result<Self> {
        ensure!(
            !parameters.is_empty(),
            "At least one fixed chart is required"
        );
        ensure!(
            angular_length.is_finite() && angular_length > 0.,
            "Invalid angular length"
        );
        ensure!(
            correlation.is_finite() && (-1. ..=1.).contains(&correlation),
            "Correlation must lie in [-1,1]"
        );
        let mut charts = Vec::with_capacity(parameters.len());
        for parameters in parameters {
            chart_parameters_valid(&parameters)?;
            let (lower, log_determinant) = cholesky(parameters.covariance)?;
            charts.push(Chart {
                parameters,
                lower,
                log_determinant,
            });
        }
        ensure!(
            !pairs.is_empty(),
            "At least one unordered chart pair is required"
        );
        let mut seen = std::collections::BTreeSet::new();
        for pair in &pairs {
            ensure!(
                pair.first <= pair.second && pair.second < charts.len(),
                "Invalid unordered chart pair"
            );
            ensure!(
                seen.insert((pair.first, pair.second)),
                "Duplicate unordered chart pair"
            );
            ensure!(
                pair.weight.is_finite() && pair.weight > 0.,
                "Pair weights must be finite and positive"
            );
        }
        // A subset of the legal unordered pairs with the full cardinality is
        // the whole set. This implication requires ALL checks above, including
        // duplicate rejection; pair count alone would not certify support.
        let complete_pair_support = complete_pair_count(charts.len()) == Some(pairs.len());
        // Scale first, so harmless common weight magnitude cannot overflow.
        let maximum = pairs.iter().map(|p| p.weight).fold(0., f64::max);
        let total = pairs.iter().map(|p| p.weight / maximum).sum::<f64>();
        let mut accumulated = 0.;
        let mut cumulative = Vec::with_capacity(pairs.len());
        for pair in &pairs {
            let next = accumulated + (pair.weight / maximum) / total;
            ensure!(
                next > accumulated && next.is_finite(),
                "Pair probabilities lose floating-point support"
            );
            accumulated = next;
            cumulative.push(next);
        }
        ensure!(
            cumulative[..cumulative.len() - 1].iter().all(|&p| p < 1.),
            "Pair probabilities lose terminal floating-point support"
        );
        *cumulative.last_mut().unwrap() = 1.;
        Ok(Self {
            charts,
            angular_length,
            correlation,
            sine: ((1. - correlation) * (1. + correlation)).sqrt(),
            pairs,
            cumulative,
            complete_pair_support,
        })
    }

    fn supports_pair(&self, source: usize, target: usize) -> bool {
        if source >= self.charts.len() || target >= self.charts.len() {
            return false;
        }
        if self.complete_pair_support {
            return true;
        }
        let pair = (source.min(target), source.max(target));
        self.pairs.iter().any(|p| (p.first, p.second) == pair)
    }

    pub fn chart_count(&self) -> usize {
        self.charts.len()
    }

    /// This is the only random direction selection in the kernel. Positive
    /// direction weights cannot accidentally differ: the same coin is used.
    pub fn draw_trace(&self, rng: &mut StdRng) -> BasinTrace {
        let target = rng.random::<f64>();
        let index = self
            .cumulative
            .partition_point(|&p| p <= target)
            .min(self.pairs.len() - 1);
        let pair = &self.pairs[index];
        let reverse = pair.first != pair.second && rng.random::<bool>();
        BasinTrace {
            source: if reverse { pair.second } else { pair.first },
            target: if reverse { pair.first } else { pair.second },
            noise: std::array::from_fn(|_| StandardNormal.sample(rng)),
        }
    }

    pub fn encode(&self, chart_index: usize, pose: Pose) -> Result<Vec6> {
        pose.validate()?;
        let chart = self
            .charts
            .get(chart_index)
            .ok_or_else(|| anyhow::anyhow!("Chart index out of range"))?;
        let q = quaternion(matmul(
            rotation(pose.orientation),
            transpose(chart.parameters.anchor_rotation),
        ));
        ensure!(q[0] != 0., "Pose lies on the Cayley half-turn seam");
        let coordinates: Vec6 = std::array::from_fn(|i| {
            if i < 3 {
                pose.position[i] - chart.parameters.anchor_position[i]
            } else {
                self.angular_length * q[i - 2] / q[0]
            }
        });
        ensure!(
            coordinates.iter().all(|x| x.is_finite()),
            "Unrepresentable Cayley chart coordinates"
        );
        let mut z = [0.; 6];
        for i in 0..6 {
            z[i] = (coordinates[i]
                - chart.parameters.mean[i]
                - (0..i).map(|j| chart.lower[i][j] * z[j]).sum::<f64>())
                / chart.lower[i][i];
        }
        ensure!(
            z.iter().all(|x| x.is_finite()),
            "Unrepresentable standardized pose"
        );
        Ok(z)
    }

    fn decode_and_log_volume(&self, chart_index: usize, z: Vec6) -> Result<(Pose, f64)> {
        let chart = self
            .charts
            .get(chart_index)
            .ok_or_else(|| anyhow::anyhow!("Chart index out of range"))?;
        ensure!(
            z.iter().all(|x| x.is_finite()),
            "Nonfinite standardized pose"
        );
        let coordinates: Vec6 = std::array::from_fn(|i| {
            chart.parameters.mean[i] + (0..=i).map(|j| chart.lower[i][j] * z[j]).sum::<f64>()
        });
        ensure!(
            coordinates.iter().all(|x| x.is_finite()),
            "Unrepresentable chart output"
        );
        let u = std::array::from_fn(|i| coordinates[i + 3] / self.angular_length);
        ensure!(
            u.iter().all(|x| x.is_finite()),
            "Unrepresentable rotational chart output"
        );
        let pose = Pose {
            position: std::array::from_fn(|i| coordinates[i] + chart.parameters.anchor_position[i]),
            orientation: quaternion(matmul(cayley(u), chart.parameters.anchor_rotation)),
        };
        pose.validate()?;
        // dmu_SO3/du = 1/[pi^2(1+|u|^2)^2]. Rotation-chart anchors do not
        // change Haar measure. Scaling three angular coordinates gives ell^-3.
        let norm = u.iter().fold(1_f64, |a, b| a.hypot(*b));
        let log_volume =
            chart.log_determinant - 3. * self.angular_length.ln() - 2. * PI.ln() - 4. * norm.ln();
        ensure!(log_volume.is_finite(), "Unrepresentable chart volume");
        Ok((pose, log_volume))
    }

    pub fn decode(&self, chart_index: usize, z: Vec6) -> Result<Pose> {
        Ok(self.decode_and_log_volume(chart_index, z)?.0)
    }

    pub fn correlation(&self) -> f64 {
        self.correlation
    }

    /// Normalized chart density in translation times Haar measure:
    /// standard normal of the latent over the chart volume element. Poses on
    /// the Cayley seam or outside representable range have zero density.
    pub fn log_density(&self, chart_index: usize, pose: Pose) -> f64 {
        let Ok(z) = self.encode(chart_index, pose) else {
            return f64::NEG_INFINITY;
        };
        let Ok((_, log_volume)) = self.decode_and_log_volume(chart_index, z) else {
            return f64::NEG_INFINITY;
        };
        -0.5 * z.iter().map(|x| x * x).sum::<f64>() - 3. * (2. * PI).ln() - log_volume
    }

    /// Strict chart density for callers that must retain numerical failures.
    /// Only the exact Cayley half-turn seam has legitimate zero density;
    /// invalid inputs and unrepresentable chart arithmetic are errors.
    pub fn checked_log_density(&self, chart_index: usize, pose: Pose) -> Result<f64> {
        pose.validate()?;
        let chart = self
            .charts
            .get(chart_index)
            .ok_or_else(|| anyhow::anyhow!("Chart index out of range"))?;
        let q = quaternion(matmul(
            rotation(pose.orientation),
            transpose(chart.parameters.anchor_rotation),
        ));
        ensure!(
            q.iter().all(|x| x.is_finite()),
            "Unrepresentable relative chart rotation"
        );
        if q[0] == 0. {
            return Ok(f64::NEG_INFINITY);
        }
        let z = self.encode(chart_index, pose)?;
        let (_, log_volume) = self.decode_and_log_volume(chart_index, z)?;
        let square = z.iter().map(|x| x * x).sum::<f64>();
        ensure!(square.is_finite(), "Unrepresentable chart quadratic");
        let density = -0.5 * square - 3. * (2. * PI).ln() - log_volume;
        ensure!(density.is_finite(), "Unrepresentable chart log density");
        Ok(density)
    }

    /// Pure deterministic transformation. Numerical seams and invalid traces
    /// return errors; callers must never retry until a valid pose is obtained.
    pub fn apply(&self, old: Pose, trace: &BasinTrace) -> Result<BasinStep> {
        ensure!(
            self.supports_pair(trace.source, trace.target),
            "Trace pair is outside the fixed law"
        );
        ensure!(
            trace.noise.iter().all(|x| x.is_finite()),
            "Nonfinite auxiliary noise"
        );
        let source_latent = self.encode(trace.source, old)?;
        let target_latent = std::array::from_fn(|i| {
            self.correlation * source_latent[i] + self.sine * trace.noise[i]
        });
        let inverse_noise: Vec6 = std::array::from_fn(|i| {
            self.sine * source_latent[i] - self.correlation * trace.noise[i]
        });
        let (pose, new_volume) = self.decode_and_log_volume(trace.target, target_latent)?;
        let (_, old_volume) = self.decode_and_log_volume(trace.source, source_latent)?;
        let old_square = trace.noise.iter().map(|x| x * x).sum::<f64>();
        let new_square = inverse_noise.iter().map(|x| x * x).sum::<f64>();
        ensure!(
            old_square.is_finite() && new_square.is_finite(),
            "Unrepresentable auxiliary norm"
        );
        let log_auxiliary_ratio = 0.5 * (old_square - new_square);
        let log_extended_jacobian = new_volume - old_volume;
        let log_correction = log_extended_jacobian + log_auxiliary_ratio;
        ensure!(
            log_correction.is_finite(),
            "Unrepresentable involution correction"
        );
        Ok(BasinStep {
            pose,
            inverse_trace: BasinTrace {
                source: trace.target,
                target: trace.source,
                noise: inverse_noise,
            },
            source_latent,
            target_latent,
            log_extended_jacobian,
            log_auxiliary_ratio,
            log_correction,
        })
    }
}

/// The correlated latent map between a chart of `source` and a chart of
/// `target`, which may be different chart sets on the same pose space. Uses
/// the source set's correlation. The inverse applies the same function with
/// the roles swapped and `inverse_trace.noise`; `inverse_trace` indices refer
/// to (target chart, source chart).
pub fn cross_chart_step(
    source: (&FixedBasinInvolution, usize),
    target: (&FixedBasinInvolution, usize),
    old: Pose,
    noise: Vec6,
) -> Result<BasinStep> {
    let (from, s) = source;
    let (to, t) = target;
    ensure!(
        noise.iter().all(|x| x.is_finite()),
        "Nonfinite auxiliary noise"
    );
    ensure!(
        from.correlation == to.correlation && from.angular_length == to.angular_length,
        "Cross-chart maps need one correlation and angular length"
    );
    let source_latent = from.encode(s, old)?;
    let target_latent: Vec6 =
        std::array::from_fn(|i| from.correlation * source_latent[i] + from.sine * noise[i]);
    let inverse_noise: Vec6 =
        std::array::from_fn(|i| from.sine * source_latent[i] - from.correlation * noise[i]);
    let (pose, new_volume) = to.decode_and_log_volume(t, target_latent)?;
    let (_, old_volume) = from.decode_and_log_volume(s, source_latent)?;
    let old_square = noise.iter().map(|x| x * x).sum::<f64>();
    let new_square = inverse_noise.iter().map(|x| x * x).sum::<f64>();
    let log_auxiliary_ratio = 0.5 * (old_square - new_square);
    let log_extended_jacobian = new_volume - old_volume;
    let log_correction = log_extended_jacobian + log_auxiliary_ratio;
    ensure!(
        log_correction.is_finite(),
        "Unrepresentable involution correction"
    );
    Ok(BasinStep {
        pose,
        inverse_trace: BasinTrace {
            source: t,
            target: s,
            noise: inverse_noise,
        },
        source_latent,
        target_latent,
        log_extended_jacobian,
        log_auxiliary_ratio,
        log_correction,
    })
}

#[cfg(test)]
mod checked_density_tests {
    use super::*;
    use crate::math::IDENTITY;

    fn map() -> FixedBasinInvolution {
        FixedBasinInvolution::new(
            vec![GaussianComponentParameters {
                anchor_position: [0.; 3],
                anchor_rotation: IDENTITY,
                mean: [0.; 6],
                covariance: std::array::from_fn(|i| {
                    std::array::from_fn(|j| if i == j { 1. } else { 0. })
                }),
                weight: 1.,
            }],
            1.,
            0.,
            vec![BasinPair {
                first: 0,
                second: 0,
                weight: 1.,
            }],
        )
        .unwrap()
    }

    fn pose(position: [f64; 3]) -> Pose {
        Pose {
            position,
            orientation: [1., 0., 0., 0.],
        }
    }

    #[test]
    fn two_neighbor_singleton_checked_density_agrees_for_finite_scores() {
        let map = map();
        for p in [
            pose([0.; 3]),
            pose([1e150, -2., 3.]),
            map.decode(0, [0.4, -1.1, 2., 0.3, 0.7, -0.2]).unwrap(),
        ] {
            let checked = map.checked_log_density(0, p).unwrap();
            assert!(checked.is_finite());
            assert_eq!(checked, map.log_density(0, p));
        }
    }

    #[test]
    fn two_neighbor_singleton_checked_density_only_exact_seam_is_zero() {
        let map = map();
        let mut p = pose([0.; 3]);
        p.orientation = [0., 1., 0., 0.];
        assert_eq!(map.checked_log_density(0, p).unwrap(), f64::NEG_INFINITY);
        // No epsilon seam: a representable nearby tail is scored, while an
        // unrepresentable quadratic at a still nonzero scalar is fatal.
        p.orientation[0] = 1e-8;
        assert!(map.checked_log_density(0, p).unwrap().is_finite());
        p.orientation[0] = 1e-200;
        assert!(map.checked_log_density(0, p).is_err());
    }

    #[test]
    fn two_neighbor_singleton_checked_density_rejects_inputs_and_quadratic_overflow() {
        let map = map();
        assert!(map.checked_log_density(1, pose([0.; 3])).is_err());
        assert!(
            map.checked_log_density(0, pose([f64::NAN, 0., 0.]))
                .is_err()
        );
        let mut invalid = pose([0.; 3]);
        invalid.orientation = [0.; 4];
        assert!(map.checked_log_density(0, invalid).is_err());
        let huge = pose([1e200, 0., 0.]);
        assert!(map.encode(0, huge).is_ok());
        assert_eq!(map.log_density(0, huge), f64::NEG_INFINITY);
        assert!(map.checked_log_density(0, huge).is_err());
    }

    #[test]
    fn two_neighbor_singleton_checked_density_propagates_encode_and_decode_errors() {
        let mut map = map();
        map.charts[0].parameters.anchor_position[0] = -1e308;
        assert!(map.checked_log_density(0, pose([1e308, 0., 0.])).is_err());
        map.charts[0].parameters.anchor_position[0] = 0.;
        // Fault injection isolates decoder error propagation after encoding
        // succeeds; ordinary constructors forbid this invalid determinant.
        map.charts[0].log_determinant = f64::NAN;
        assert!(map.encode(0, pose([0.; 3])).is_ok());
        assert!(map.checked_log_density(0, pose([0.; 3])).is_err());
    }
}

#[cfg(test)]
mod pair_support_tests {
    use super::*;
    use crate::math::cayley;

    fn parameters(count: usize) -> Vec<GaussianComponentParameters> {
        (0..count)
            .map(|index| {
                let shift = (index % 7) as f64 * 0.03;
                let mut lower = [[0.; 6]; 6];
                for (i, row) in lower.iter_mut().enumerate() {
                    row[i] = 0.7 + 0.08 * i as f64;
                }
                lower[3][0] = 0.12;
                lower[4][1] = -0.09;
                GaussianComponentParameters {
                    anchor_position: [shift, -0.2, 0.3],
                    anchor_rotation: cayley([0.13, -0.07 + shift, 0.11]),
                    mean: [0.03, -0.01, 0.02, 0.04, -0.02, 0.01],
                    covariance: std::array::from_fn(|i| {
                        std::array::from_fn(|j| (0..6).map(|k| lower[i][k] * lower[j][k]).sum())
                    }),
                    weight: 1.,
                }
            })
            .collect()
    }

    fn dense_pairs(count: usize) -> Vec<BasinPair> {
        (0..count)
            .flat_map(|first| {
                (first..count).map(move |second| BasinPair {
                    first,
                    second,
                    weight: 1. + ((first + second) % 5) as f64 * 0.1,
                })
            })
            .collect()
    }

    fn dense_map(count: usize, correlation: f64) -> FixedBasinInvolution {
        FixedBasinInvolution::new(parameters(count), 1.3, correlation, dense_pairs(count)).unwrap()
    }

    #[inline(never)]
    fn legacy_support(map: &FixedBasinInvolution, source: usize, target: usize) -> bool {
        let pair = (source.min(target), source.max(target));
        map.pairs.iter().any(|p| (p.first, p.second) == pair)
    }

    fn step_bits(step: &BasinStep) -> Vec<u64> {
        step.pose
            .position
            .iter()
            .chain(step.pose.orientation.iter())
            .chain(step.inverse_trace.noise.iter())
            .chain(step.source_latent.iter())
            .chain(step.target_latent.iter())
            .copied()
            .chain([
                step.log_extended_jacobian,
                step.log_auxiliary_ratio,
                step.log_correction,
            ])
            .map(f64::to_bits)
            .collect()
    }

    #[test]
    fn checked_triangular_count_handles_boundaries_without_intermediate_overflow() {
        for count in [0, 1, 2, 64, 256, 2048, 4096, usize::MAX / 2, usize::MAX] {
            let wide = count as u128;
            let expected = usize::try_from(wide * (wide + 1) / 2).ok();
            assert_eq!(complete_pair_count(count), expected);
        }
        // This count's product n(n+1) overflows a 64-bit usize, while its
        // triangular count fits. The same property holds on 32-bit targets.
        let count = 1usize << (usize::BITS / 2);
        assert!(count.checked_mul(count + 1).is_none());
        assert!(complete_pair_count(count).is_some());
    }

    #[test]
    fn dense_support_matches_exhaustive_scan_for_all_ordered_pairs() {
        for count in [1, 2, 3, 8, 32] {
            let map = dense_map(count, 0.65);
            assert!(map.complete_pair_support);
            for source in 0..count + 2 {
                for target in 0..count + 2 {
                    assert_eq!(
                        map.supports_pair(source, target),
                        legacy_support(&map, source, target)
                    );
                    assert_eq!(
                        map.supports_pair(source, target),
                        source < count && target < count
                    );
                }
            }
            assert!(!map.supports_pair(usize::MAX, 0));
            assert!(!map.supports_pair(0, usize::MAX));
        }
    }

    #[test]
    fn sparse_holes_and_reversed_labels_keep_legacy_semantics() {
        let pairs = vec![
            BasinPair {
                first: 0,
                second: 0,
                weight: 1.,
            },
            BasinPair {
                first: 0,
                second: 2,
                weight: 2.,
            },
            BasinPair {
                first: 1,
                second: 1,
                weight: 1.,
            },
        ];
        let map = FixedBasinInvolution::new(parameters(3), 1.3, 0.65, pairs).unwrap();
        assert!(!map.complete_pair_support);
        let old = map.decode(0, [0.2, -0.1, 0.3, 0.1, 0.2, -0.2]).unwrap();
        for source in 0..5 {
            for target in 0..5 {
                let allowed = legacy_support(&map, source, target);
                assert_eq!(map.supports_pair(source, target), allowed);
                assert_eq!(
                    map.apply(
                        old,
                        &BasinTrace {
                            source,
                            target,
                            noise: [0.1; 6]
                        }
                    )
                    .is_ok(),
                    allowed
                );
            }
        }
        assert!(map.supports_pair(2, 0));
        assert!(!map.supports_pair(2, 2));
    }

    #[test]
    fn constructor_rejects_duplicate_range_and_noncanonical_pairs_before_certifying() {
        let mut duplicate = dense_pairs(3);
        duplicate[1] = duplicate[0].clone();
        assert!(FixedBasinInvolution::new(parameters(3), 1.3, 0.65, duplicate).is_err());
        let mut outside = dense_pairs(3);
        outside.last_mut().unwrap().second = 3;
        assert!(FixedBasinInvolution::new(parameters(3), 1.3, 0.65, outside).is_err());
        let mut noncanonical = dense_pairs(3);
        noncanonical[1].first = 2;
        noncanonical[1].second = 0;
        assert!(FixedBasinInvolution::new(parameters(3), 1.3, 0.65, noncanonical).is_err());
    }

    #[test]
    fn dense_apply_rejects_out_of_range_labels_without_touching_the_map() {
        let map = dense_map(3, 0.65);
        let old = map.decode(0, [0.; 6]).unwrap();
        for (source, target) in [(3, 0), (0, 3), (3, 3), (usize::MAX, 0), (0, usize::MAX)] {
            assert!(
                map.apply(
                    old,
                    &BasinTrace {
                        source,
                        target,
                        noise: [0.; 6]
                    }
                )
                .is_err()
            );
        }
    }

    #[test]
    fn complete_apply_outputs_are_bit_identical_to_forced_legacy_scan() {
        for correlation in [-1., -0.65, 0., 0.65, 1.] {
            let fast = dense_map(8, correlation);
            let mut legacy = fast.clone();
            legacy.complete_pair_support = false;
            for source in 0..8 {
                let old = fast
                    .decode(source, [0.2, -0.3, 0.1, 0.4, -0.2, 0.3])
                    .unwrap();
                for target in 0..8 {
                    let trace = BasinTrace {
                        source,
                        target,
                        noise: [-0.3, 0.1, 0.4, 0.2, 0.5, -0.1],
                    };
                    let a = fast.apply(old, &trace).unwrap();
                    let b = legacy.apply(old, &trace).unwrap();
                    assert_eq!(a.inverse_trace.source, b.inverse_trace.source);
                    assert_eq!(a.inverse_trace.target, b.inverse_trace.target);
                    assert_eq!(step_bits(&a), step_bits(&b));
                }
            }
        }
    }

    #[test]
    fn pair_draws_and_cumulative_weights_ignore_the_support_certificate() {
        use rand::{SeedableRng, rngs::StdRng};
        let fast = dense_map(8, 0.65);
        let mut legacy = fast.clone();
        legacy.complete_pair_support = false;
        assert_eq!(fast.cumulative, legacy.cumulative);
        let mut a = StdRng::seed_from_u64(202610040701);
        let mut b = StdRng::seed_from_u64(202610040701);
        for _ in 0..256 {
            assert_eq!(fast.draw_trace(&mut a), legacy.draw_trace(&mut b));
        }
    }

    /// Source-only until a separate isolated build/execution allocation is
    /// authorized. Exactly 256 deterministic queries per method/size; no RNG,
    /// pose transformation, geometry, MC update or adaptive repetition count.
    #[test]
    #[ignore = "Requires a frozen bounded support-only benchmark allocation"]
    fn complete_pair_support_microbenchmark() -> Result<()> {
        use crate::simulation::cpu_seconds;
        use serde_json::json;
        use std::{fs, hint::black_box, io::Write, path::PathBuf, time::Instant};

        let output = PathBuf::from(std::env::var("BASIN_PAIR_SUPPORT_BENCHMARK_OUT")?);
        ensure!(
            output.is_absolute() && !output.exists(),
            "Fresh absolute benchmark output required"
        );
        fs::create_dir(&output)?;
        let mut journal = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(output.join("events.jsonl"))?;
        let mut emit = |value: &serde_json::Value| -> Result<()> {
            serde_json::to_writer(&mut journal, value)?;
            writeln!(&mut journal)?;
            journal.flush()?;
            Ok(())
        };
        let started = Instant::now();
        let total_cpu = cpu_seconds();
        let mut rows = Vec::new();
        for count in [64usize, 256, 2048] {
            emit(&json!({"kind":"size_begun","charts":count,"queries_per_method":256}))?;
            let tick = Instant::now();
            let cpu = cpu_seconds();
            let map = dense_map(count, 0.65);
            let constructor_cpu = cpu_seconds() - cpu;
            let constructor_wall = tick.elapsed().as_secs_f64();
            ensure!(
                map.complete_pair_support,
                "Expected complete support certificate"
            );
            let queries: Vec<_> = (0..256)
                .map(|k| match k {
                    0 => (0, 0),
                    1 => (count - 1, count - 1),
                    2 => (0, count - 1),
                    3 => (count - 1, 0),
                    _ => ((k * 37 + 7) % count, (k * 101 + 3) % count),
                })
                .collect();
            emit(
                &json!({"kind":"constructed","charts":count,"pairs":map.pairs.len(),
                "constructor_cpu_seconds":constructor_cpu,"constructor_wall_seconds":constructor_wall,
                "queries":queries}),
            )?;
            let mut measurements = Vec::new();
            let mut old_answers = None;
            for legacy in [true, false] {
                let method = if legacy {
                    "legacy_scan"
                } else {
                    "certified_complete_support"
                };
                emit(&json!({"kind":"measurement_begun","charts":count,"method":method}))?;
                let tick = Instant::now();
                let cpu = cpu_seconds();
                let answers: Vec<_> = queries
                    .iter()
                    .map(|&(source, target)| {
                        let answer = if legacy {
                            legacy_support(black_box(&map), black_box(source), black_box(target))
                        } else {
                            black_box(&map).supports_pair(black_box(source), black_box(target))
                        };
                        black_box(answer)
                    })
                    .collect();
                let cpu_seconds = cpu_seconds() - cpu;
                let wall_seconds = tick.elapsed().as_secs_f64();
                ensure!(
                    answers.len() == 256 && answers.iter().all(|a| *a),
                    "Support result differs"
                );
                if legacy {
                    old_answers = Some(answers.clone());
                } else {
                    ensure!(
                        old_answers.as_ref() == Some(&answers),
                        "Fast/old support differs"
                    );
                }
                let measurement = json!({"kind":"measurement_complete","charts":count,"method":method,
                    "calls":256,"accepted":answers.iter().filter(|v| **v).count(),"answers":answers,
                    "cpu_seconds":cpu_seconds,"wall_seconds":wall_seconds});
                emit(&measurement)?;
                measurements.push(measurement);
            }
            rows.push(json!({"charts":count,"pairs":map.pairs.len(),"queries":queries,
                "constructor_cpu_seconds":constructor_cpu,"constructor_wall_seconds":constructor_wall,
                "measurements":measurements}));
            ensure!(
                cpu_seconds() - total_cpu < 30. && started.elapsed().as_secs_f64() < 60.,
                "Fixed benchmark budget exceeded"
            );
        }
        drop(emit);
        journal.sync_all()?;
        let mut file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(output.join("report.json"))?;
        serde_json::to_writer_pretty(
            &mut file,
            &json!({"schema":"basin-pair-support-benchmark-v1",
            "complete":true,"passed":true,"sizes":[64,256,2048],"queries_per_method":256,
            "total_support_calls":1536,"random_draws":0,"pose_updates":0,"geometry_queries":0,
            "cpu_seconds":cpu_seconds()-total_cpu,"wall_seconds":started.elapsed().as_secs_f64(),"rows":rows,
            "scope":"Support membership only. Constructor cost is separate. This does not measure complete apply, physical MC, assembly efficiency or production speedup. Timer/allocation overhead can dominate the short certified check; no timing-dependent repetition or outcome filtering."}),
        )?;
        writeln!(file)?;
        file.sync_all()?;
        Ok(())
    }
}
