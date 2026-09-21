//! Reversible conditional fluctuations around an immutable learned atlas.
//!
//! The retained target is `pi(X) phi(eta)`: the component count, chart anchors,
//! reference means and full covariances remain fixed. Current pair geometry
//! determines a reproducible coordinate fit F(X); the actual proposal decodes
//! `v = gain * F(X) + noise * eta`. A physical proposal holds eta fixed and must
//! evaluate its reverse proposal from F(Y), not reuse the proposal at X. There
//! is no fit-score, determinant, or component-count factor in physical
//! acceptance. Refreshing eta from independent standard normals is Gibbs.
//!
//! Each component's reference Cholesky B defines its own units. Means become
//! mu+B*a and covariance becomes B*T*T^T*B^T, with bounded smooth coordinates
//! for a and the lower triangular T. Consequently there is no global covariance
//! floor to erase narrow learned registration basins. Fits use clipped nearest
//! reference-chart residuals with explicit shrinkage toward that reference.
//!
//! This finite-dimensional conditional law is deliberately an atlas-preserving
//! control, not a posterior on mixture parameters or a discovery algorithm.
use anyhow::{Result, ensure};
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};

use crate::{
    math::{Pose, matmul, matvec, norm, rotation, sub, transpose},
    proposal::{FrozenRelativePoseProposal, GaussianComponentParameters},
};

type Mat6 = [[f64; 6]; 6];
const MEAN_BOUND: f64 = 3.;
const LOG_DIAGONAL_BOUND: f64 = std::f64::consts::LN_2;
const OFF_DIAGONAL_BOUND: f64 = 0.5;
const LOGIT_BOUND: f64 = 4.;

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AtlasInitialization {
    /// Choose eta to recover the reference atlas at the initial configuration.
    #[default]
    Reference,
    /// Center the unconstrained conditional model on its current deterministic fit.
    Zero,
    /// Draw the retained auxiliary coordinates from their normalized target.
    Random,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AtlasTransportConfig {
    pub mean_gain: f64,
    pub covariance_gain: f64,
    pub weight_gain: f64,
    pub mean_noise: f64,
    pub covariance_noise: f64,
    pub weight_noise: f64,
    /// Maximum Euclidean norm in a reference component's whitened coordinates.
    pub assignment_cutoff: f64,
    /// Coordinatewise clipping after deterministic nearest-component assignment.
    pub residual_clip: f64,
    /// Pseudocount for reference means, covariance, and component probabilities.
    pub shrinkage: f64,
    pub refresh_probability: f64,
    pub initialization: AtlasInitialization,
}

impl Default for AtlasTransportConfig {
    fn default() -> Self {
        Self {
            mean_gain: 0.25,
            covariance_gain: 0.25,
            weight_gain: 0.25,
            mean_noise: 0.1,
            covariance_noise: 0.1,
            weight_noise: 0.1,
            assignment_cutoff: 6.,
            residual_clip: 3.,
            shrinkage: 8.,
            refresh_probability: 1.,
            initialization: AtlasInitialization::Reference,
        }
    }
}

impl AtlasTransportConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            [
                self.mean_gain,
                self.covariance_gain,
                self.weight_gain,
                self.mean_noise,
                self.covariance_noise,
                self.weight_noise
            ]
            .iter()
            .all(|x| x.is_finite() && (0. ..=100.).contains(x)),
            "Atlas gains and noises must be finite in [0,100]"
        );
        ensure!(
            [self.assignment_cutoff, self.residual_clip, self.shrinkage]
                .iter()
                .all(|x| x.is_finite() && (1e-8..=1e8).contains(x)),
            "Atlas cutoff, clipping, and shrinkage must lie in [1e-8,1e8]"
        );
        ensure!(
            self.refresh_probability.is_finite() && (0. ..=1.).contains(&self.refresh_probability),
            "Atlas refresh_probability must lie in [0,1]"
        );
        Ok(())
    }
}

/// Six means, 21 lower-triangular covariance coordinates per component,
/// followed by K-1 reference logits. The count itself is immutable.
pub fn parameter_dimension(k: usize) -> usize {
    if k == 0 { 0 } else { 28 * k - 1 }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AtlasTransportState {
    pub eta: Vec<f64>,
}

impl AtlasTransportState {
    pub fn validate(&self, k: usize) -> Result<()> {
        ensure!(
            self.eta.len() == parameter_dimension(k) && self.eta.iter().all(|x| x.is_finite()),
            "Invalid atlas transport auxiliary state"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AtlasTransportFit {
    /// F(X), before multiplying separate mean/covariance/weight gains.
    pub coordinates: Vec<f64>,
    pub data_count: usize,
    pub assigned_count: usize,
    pub component_counts: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct AtlasTransportEngine {
    base: FrozenRelativePoseProposal,
    config: AtlasTransportConfig,
    parameters: Vec<GaussianComponentParameters>,
    lower: Vec<Mat6>,
}

fn cholesky(covariance: Mat6) -> Result<Mat6> {
    let mut lower = [[0.; 6]; 6];
    for i in 0..6 {
        for j in 0..=i {
            let residual =
                covariance[i][j] - (0..j).map(|d| lower[i][d] * lower[j][d]).sum::<f64>();
            if i == j {
                ensure!(
                    residual.is_finite() && residual > 0.,
                    "Atlas covariance is not numerically positive definite"
                );
                lower[i][j] = residual.sqrt();
            } else {
                lower[i][j] = residual / lower[j][j];
            }
        }
    }
    Ok(lower)
}

fn bounded(value: f64, bound: f64) -> f64 {
    bound * (value / bound).tanh()
}

impl AtlasTransportEngine {
    pub fn new(base: FrozenRelativePoseProposal, config: AtlasTransportConfig) -> Result<Self> {
        config.validate()?;
        ensure!(
            !base.is_periodic(),
            "Atlas transport currently requires open spherical boundaries"
        );
        ensure!(
            !base.has_reciprocal_components(),
            "Adaptive atlas transport does not support reciprocal charts"
        );
        let parameters = base.component_parameters();
        ensure!(
            !parameters.is_empty(),
            "Atlas transport requires at least one learned component"
        );
        let lower = parameters
            .iter()
            .map(|p| cholesky(p.covariance))
            .collect::<Result<_>>()?;
        Ok(Self {
            base,
            config,
            parameters,
            lower,
        })
    }

    pub fn component_count(&self) -> usize {
        self.parameters.len()
    }

    pub fn fit(&self, poses: &[Pose]) -> Result<AtlasTransportFit> {
        for pose in poses {
            pose.validate()?;
        }
        let k = self.component_count();
        let mut observations = vec![Vec::<[f64; 6]>::new(); k];
        let mut data_count = 0;
        for (i, anchor) in poses.iter().enumerate() {
            let inverse_anchor = transpose(rotation(anchor.orientation));
            for (j, pose) in poses.iter().enumerate() {
                if i == j {
                    continue;
                }
                data_count += 1;
                let relative_t = matvec(inverse_anchor, sub(pose.position, anchor.position));
                let relative_r = matmul(inverse_anchor, rotation(pose.orientation));
                let mut nearest: Option<(usize, [f64; 6], f64)> = None;
                for component in 0..k {
                    if let Some(residual) = self
                        .base
                        .component_residual(relative_t, relative_r, component)
                    {
                        // Nested hypot avoids squared-distance overflow for remote pairs.
                        let distance = norm([
                            norm([residual[0], residual[1], residual[2]]),
                            norm([residual[3], residual[4], residual[5]]),
                            0.,
                        ]);
                        if distance <= self.config.assignment_cutoff
                            && nearest.as_ref().is_none_or(|old| distance < old.2)
                        {
                            nearest = Some((component, residual, distance));
                        }
                    }
                }
                if let Some((component, residual, _)) = nearest {
                    observations[component].push(
                        residual.map(|x| {
                            x.clamp(-self.config.residual_clip, self.config.residual_clip)
                        }),
                    );
                }
            }
        }
        let component_counts: Vec<_> = observations.iter().map(Vec::len).collect();
        let assigned_count = component_counts.iter().sum();
        let mut coordinates = Vec::with_capacity(parameter_dimension(k));
        for values in &observations {
            if values.is_empty() {
                coordinates.extend([0.; 27]);
                continue;
            }
            let denominator = values.len() as f64 + self.config.shrinkage;
            let mean: [f64; 6] =
                std::array::from_fn(|i| values.iter().map(|v| v[i]).sum::<f64>() / denominator);
            // Prior pseudo-observations have mean zero and covariance identity.
            // Centered accumulation retains its positive definiteness even for
            // a large collection of identical observations.
            let covariance: Mat6 = std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    let prior = self.config.shrinkage * (f64::from(i == j) + mean[i] * mean[j]);
                    (prior
                        + values
                            .iter()
                            .map(|v| (v[i] - mean[i]) * (v[j] - mean[j]))
                            .sum::<f64>())
                        / denominator
                })
            });
            let lower = cholesky(covariance)?;
            coordinates.extend(mean);
            for (i, row) in lower.iter().enumerate() {
                for (j, value) in row.iter().enumerate().take(i + 1) {
                    coordinates.push(if i == j { value.ln() } else { *value });
                }
            }
        }
        // Dirichlet-like shrinkage gives the reference probabilities when no
        // observations were assigned, and never removes an atlas component.
        let reference = (component_counts[k - 1] as f64 / self.parameters[k - 1].weight
            + self.config.shrinkage)
            .ln();
        for (component, count) in component_counts.iter().enumerate().take(k - 1) {
            coordinates.push(
                (*count as f64 / self.parameters[component].weight + self.config.shrinkage).ln()
                    - reference,
            );
        }
        ensure!(
            coordinates.iter().all(|v| v.is_finite()),
            "Nonfinite atlas fit"
        );
        Ok(AtlasTransportFit {
            coordinates,
            data_count,
            assigned_count,
            component_counts,
        })
    }

    fn coefficient(&self, index: usize) -> (f64, f64) {
        if index >= 27 * self.component_count() {
            (self.config.weight_gain, self.config.weight_noise)
        } else if index % 27 < 6 {
            (self.config.mean_gain, self.config.mean_noise)
        } else {
            (self.config.covariance_gain, self.config.covariance_noise)
        }
    }

    fn validate_fit(&self, fit: &AtlasTransportFit) -> Result<()> {
        ensure!(
            fit.coordinates.len() == parameter_dimension(self.component_count())
                && fit.coordinates.iter().all(|x| x.is_finite())
                && fit.component_counts.len() == self.component_count(),
            "Invalid atlas transport fit"
        );
        Ok(())
    }

    pub fn initialize(
        &self,
        fit: &AtlasTransportFit,
        rng: &mut StdRng,
    ) -> Result<AtlasTransportState> {
        self.validate_fit(fit)?;
        let eta = match self.config.initialization {
            AtlasInitialization::Zero => vec![0.; fit.coordinates.len()],
            AtlasInitialization::Random => (0..fit.coordinates.len())
                .map(|_| StandardNormal.sample(rng)).collect(),
            AtlasInitialization::Reference => fit.coordinates.iter().enumerate().map(|(i, f)| {
                let (gain, noise) = self.coefficient(i);
                let value = gain * f;
                if noise > 0. { Ok(-value / noise) } else {
                    ensure!(value == 0.,
                        "Reference atlas initialization needs nonzero noise wherever the initial fit is nonzero");
                    Ok(0.)
                }
            }).collect::<Result<Vec<_>>>()?,
        };
        let state = AtlasTransportState { eta };
        state.validate(self.component_count())?;
        Ok(state)
    }

    pub fn refresh(&self, state: &mut AtlasTransportState, rng: &mut StdRng) -> Result<()> {
        state.validate(self.component_count())?;
        for value in &mut state.eta {
            *value = StandardNormal.sample(rng);
        }
        Ok(())
    }

    pub fn model(
        &self,
        fit: &AtlasTransportFit,
        state: &AtlasTransportState,
    ) -> Result<FrozenRelativePoseProposal> {
        self.validate_fit(fit)?;
        state.validate(self.component_count())?;
        let values: Vec<_> = fit
            .coordinates
            .iter()
            .zip(&state.eta)
            .enumerate()
            .map(|(i, (f, e))| {
                let (gain, noise) = self.coefficient(i);
                gain * f + noise * e
            })
            .collect();
        ensure!(
            values.iter().all(|x| x.is_finite()),
            "Unrepresentable atlas coordinates"
        );
        // This exact path preserves the full frozen model, RNG choices, and
        // trajectory in the controlled zero-gain/zero-noise limit.
        if values.iter().all(|x| *x == 0.) {
            return Ok(self.base.clone());
        }
        let k = self.component_count();
        let mut parameters = self.parameters.clone();
        for component in 0..k {
            let coordinates = &values[27 * component..27 * (component + 1)];
            let shift: [f64; 6] = std::array::from_fn(|i| bounded(coordinates[i], MEAN_BOUND));
            for i in 0..6 {
                parameters[component].mean[i] += (0..=i)
                    .map(|j| self.lower[component][i][j] * shift[j])
                    .sum::<f64>();
            }
            // Preserve reference covariance bit-for-bit when only other model
            // coordinates move; no unnecessary refactorization floor appears.
            if coordinates[6..].iter().any(|v| *v != 0.) {
                let mut transform = [[0.; 6]; 6];
                let mut index = 6;
                for (i, row) in transform.iter_mut().enumerate() {
                    for (j, value) in row.iter_mut().enumerate().take(i + 1) {
                        *value = if i == j {
                            bounded(coordinates[index], LOG_DIAGONAL_BOUND).exp()
                        } else {
                            bounded(coordinates[index], OFF_DIAGONAL_BOUND)
                        };
                        index += 1;
                    }
                }
                let lower: Mat6 = std::array::from_fn(|i| {
                    std::array::from_fn(|j| {
                        (j..=i)
                            .map(|d| self.lower[component][i][d] * transform[d][j])
                            .sum::<f64>()
                    })
                });
                parameters[component].covariance = std::array::from_fn(|i| {
                    std::array::from_fn(|j| (0..=i.min(j)).map(|d| lower[i][d] * lower[j][d]).sum())
                });
            }
        }
        if values[27 * k..].iter().any(|x| *x != 0.) {
            let logits: Vec<_> = parameters
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    p.weight.ln()
                        + if i + 1 < k {
                            bounded(values[27 * k + i], LOGIT_BOUND)
                        } else {
                            0.
                        }
                })
                .collect();
            let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let total: f64 = logits.iter().map(|v| (v - max).exp()).sum();
            for (p, logit) in parameters.iter_mut().zip(logits) {
                p.weight = (logit - max).exp() / total;
            }
        }
        self.base.with_component_parameters(parameters)
    }
}
