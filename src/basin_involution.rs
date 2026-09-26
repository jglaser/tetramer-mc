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
        })
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

    /// Pure deterministic transformation. Numerical seams and invalid traces
    /// return errors; callers must never retry until a valid pose is obtained.
    pub fn apply(&self, old: Pose, trace: &BasinTrace) -> Result<BasinStep> {
        let pair = (
            trace.source.min(trace.target),
            trace.source.max(trace.target),
        );
        ensure!(
            self.pairs.iter().any(|p| (p.first, p.second) == pair),
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
