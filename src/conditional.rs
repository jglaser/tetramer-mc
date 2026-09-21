//! A normalized, history-free conditional closure for a variable-size GMM.
//!
//! The target in retained coordinates is pi(X) p_K(D(X)) phi(eta). Every
//! physical move retains (K,eta), refits at its candidate, reconstructs the
//! reverse proposal there, and includes log p_K(D(Y))-log p_K(D(X)). An exact
//! conditional refresh draws K from its finite softmax and eta from independent
//! standard normals. No fitted objective is a physical energy.
//!
//! F_K is a finite-budget deterministic approximation: farthest-point seeds,
//! fixed hard-assignment/refit iterations, bounded local Cayley residuals and
//! a covariance regularizer. D contains every ordered pair inside one fixed
//! center-distance cutoff. Empty clusters retain their deterministic seed with
//! a broad covariance; empty D uses identity charts. There is no warm start.
//!
//! For count scoring, q_0 is uniform on the relative-translation cutoff ball
//! times normalized Haar. q_K=(1-u)G_K+u*q_0 is likewise normalized, and evaluated
//! with the exact Cayley/Haar Jacobian, including at seams via q_0. The physical
//! proposal instead uses the existing absolute-position uniform cube defense.
//! This explicitly chosen score law is not an inferred Bayesian posterior.
//!
//! For K>0, pack each component's six dimensionless local means followed by
//! its 21 lower-Cholesky entries in row-major order, then K-1 reference logits.
//! The decoded dimensionless covariance is L L^T + (floor/16) I; coordinates
//! parameterize L for this fixed-ridge covariance. Smooth bounded maps and the
//! ridge keep it numerically strictly SPD even for extreme finite residuals. A
//! Gaussian with fixed diagonal L=residual_scale*I lives on these unconstrained
//! coordinates, v=F_K(D)+L*eta. The chart anchors belong to F_K(D), not eta; they
//! can change across a physical move. This is valid in the stated latent target
//! but is not an affine transport of a density on a fixed physical chart.

use anyhow::{Result, ensure};
use rand::{RngExt, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use crate::{
    math::*,
    proposal::{FrozenRelativePoseProposal, GaussianComponentParameters},
};

const MEAN_BOUND: f64 = 64.;
const LOGIT_BOUND: f64 = 24.;
type Mat6 = [[f64; 6]; 6];

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConditionalInitialization {
    #[default]
    Random,
    ModeZero,
    Zero,
    FixedKZero,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ConditionalConfig {
    pub k_max: usize,
    pub basin_activity: f64,
    pub score_temperature: f64,
    pub penalty_strength: f64,
    pub covariance_exponent: f64,
    #[serde(rename = "pair_cutoff_A", alias = "pair_cutoff_a")]
    pub pair_cutoff_a: Option<f64>,
    #[serde(rename = "translation_scale_A", alias = "translation_scale_a")]
    pub translation_scale_a: f64,
    #[serde(rename = "angular_length_A", alias = "angular_length_a")]
    pub angular_length_a: f64,
    pub fit_iterations: usize,
    /// Eigenvalue bounds on the dimensionless fitted covariance.
    pub covariance_floor: f64,
    pub covariance_ceiling: f64,
    pub residual_scale: f64,
    pub uniform_weight: f64,
    pub refresh_probability: f64,
    pub initialization: ConditionalInitialization,
    pub initial_k: Option<usize>,
}

impl Default for ConditionalConfig {
    fn default() -> Self {
        Self {
            k_max: 4,
            basin_activity: 2.,
            score_temperature: 1.,
            penalty_strength: 1.,
            covariance_exponent: 6.,
            pair_cutoff_a: None,
            translation_scale_a: 10.,
            angular_length_a: 10.,
            fit_iterations: 3,
            covariance_floor: 0.04,
            covariance_ceiling: 4.,
            residual_scale: 0.05,
            uniform_weight: 0.1,
            refresh_probability: 1.,
            initialization: ConditionalInitialization::Random,
            initial_k: None,
        }
    }
}

impl ConditionalConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.k_max <= 64, "Conditional K_max must be at most 64");
        ensure!(
            self.fit_iterations > 0 && self.fit_iterations <= 100,
            "Conditional fit_iterations must lie in 1..=100"
        );
        ensure!(
            [
                self.basin_activity,
                self.translation_scale_a,
                self.angular_length_a,
                self.residual_scale
            ]
            .iter()
            .all(|x| x.is_finite() && *x > 0.),
            "Conditional activity, metric scales and residual scale must be positive finite"
        );
        ensure!(
            (1e-6..=1e6).contains(&self.translation_scale_a)
                && (1e-6..=1e6).contains(&self.angular_length_a)
                && self.residual_scale <= 1.,
            "Conditional scales exceed numerical bounds"
        );
        ensure!(
            [
                self.score_temperature,
                self.penalty_strength,
                self.covariance_exponent
            ]
            .iter()
            .all(|x| x.is_finite() && *x >= 0. && *x <= 1e6),
            "Conditional score settings must be finite and lie in [0,1e6]"
        );
        ensure!(
            self.covariance_exponent <= 64.,
            "Conditional covariance exponent exceeds 64"
        );
        ensure!(
            self.covariance_floor.is_finite()
                && self.covariance_ceiling.is_finite()
                && self.covariance_floor >= 1e-4
                && self.covariance_floor < self.covariance_ceiling
                && self.covariance_ceiling <= 1e4,
            "Conditional covariance bounds require 1e-4 <= floor < ceiling <= 1e4"
        );
        ensure!(
            self.uniform_weight.is_finite()
                && self.uniform_weight > 0.
                && self.uniform_weight <= 1.,
            "Conditional uniform_weight must lie in (0,1]"
        );
        ensure!(
            self.refresh_probability.is_finite() && (0. ..=1.).contains(&self.refresh_probability),
            "Conditional refresh_probability must lie in [0,1]"
        );
        ensure!(
            self.pair_cutoff_a
                .is_none_or(|x| x.is_finite() && x > 0. && x <= 1e12),
            "Conditional pair cutoff must be positive and at most 1e12"
        );
        ensure!(
            self.initial_k.is_none_or(|k| k <= self.k_max),
            "Initial K exceeds K_max"
        );
        ensure!(
            self.initialization != ConditionalInitialization::FixedKZero
                || self.initial_k.is_some(),
            "fixed_k_zero requires initial_k"
        );
        Ok(())
    }
}

/// Number of independent unconstrained parameters; K=0 has no residual vector.
pub fn parameter_dimension(k: usize) -> usize {
    if k == 0 { 0 } else { 28 * k - 1 }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ConditionalState {
    pub k: usize,
    pub eta: Vec<f64>,
}
impl ConditionalState {
    pub fn validate(&self, config: &ConditionalConfig) -> Result<()> {
        ensure!(
            self.k <= config.k_max
                && self.eta.len() == parameter_dimension(self.k)
                && self.eta.iter().all(|x| x.is_finite()),
            "Invalid conditional closure state"
        );
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ConditionalCountFit {
    pub anchors: Vec<Pose>,
    pub coordinates: Vec<f64>,
    pub fitted_components: Vec<GaussianComponentParameters>,
}

#[derive(Clone, Debug)]
pub struct ConditionalFit {
    pub log_probabilities: Vec<f64>,
    pub log_scores: Vec<f64>,
    pub covariance_penalties: Vec<f64>,
    pub mean_log_densities: Vec<f64>,
    pub data_count: usize,
    pub counts: Vec<ConditionalCountFit>,
}

impl ConditionalFit {
    pub fn coordinates(&self, state: &ConditionalState, scale: f64) -> Result<Vec<f64>> {
        let f = self
            .counts
            .get(state.k)
            .ok_or_else(|| anyhow::anyhow!("K outside fit"))?;
        ensure!(
            state.eta.len() == f.coordinates.len() && scale.is_finite() && scale > 0.,
            "Invalid conditional coordinate dimensions or scale"
        );
        let values: Vec<_> = f
            .coordinates
            .iter()
            .zip(&state.eta)
            .map(|(v, e)| v + scale * e)
            .collect();
        ensure!(
            values.iter().all(|x| x.is_finite()),
            "Nonfinite conditional coordinates"
        );
        Ok(values)
    }

    /// Normalized Gaussian log density in the unconstrained v coordinates.
    pub fn coordinate_log_density(&self, k: usize, coordinates: &[f64], scale: f64) -> Result<f64> {
        let f = self
            .counts
            .get(k)
            .ok_or_else(|| anyhow::anyhow!("K outside fit"))?;
        ensure!(
            coordinates.len() == f.coordinates.len()
                && scale.is_finite()
                && scale > 0.
                && coordinates.iter().all(|x| x.is_finite()),
            "Invalid conditional coordinates"
        );
        Ok(
            -(coordinates.len() as f64) * (scale.ln() + 0.5 * (2. * PI).ln())
                - 0.5
                    * coordinates
                        .iter()
                        .zip(&f.coordinates)
                        .map(|(a, b)| ((a - b) / scale).powi(2))
                        .sum::<f64>(),
        )
    }
}

#[derive(Clone, Debug)]
pub struct ConditionalEngine {
    pub config: ConditionalConfig,
    pub pair_cutoff_a: f64,
    cube_lengths: Vec3,
    shape_sha256: String,
}

impl ConditionalEngine {
    pub fn new(
        body_bound: f64,
        wall_radius: f64,
        shape_sha256: &str,
        config: ConditionalConfig,
    ) -> Result<Self> {
        config.validate()?;
        ensure!(
            body_bound.is_finite()
                && body_bound > 0.
                && wall_radius.is_finite()
                && wall_radius > body_bound
                && wall_radius <= 1e12,
            "Invalid conditional container dimensions"
        );
        let pair_cutoff_a = config.pair_cutoff_a.unwrap_or(2. * body_bound + 10.);
        ensure!(
            pair_cutoff_a.is_finite() && pair_cutoff_a > 0. && pair_cutoff_a <= 1e12,
            "Invalid resolved conditional pair cutoff"
        );
        // The body-coordinate origin need not lie inside the physical sphere
        // union; its admissible center can extend one body bound beyond R.
        let cube_lengths = [2. * (wall_radius + body_bound); 3];
        FrozenRelativePoseProposal::uniform_only_open(cube_lengths, shape_sha256)?;
        Ok(Self {
            config,
            pair_cutoff_a,
            cube_lengths,
            shape_sha256: shape_sha256.into(),
        })
    }

    /// Ordered pairs are retained in stable particle-index order. No periodic image is used.
    pub fn observations(&self, poses: &[Pose]) -> Result<Vec<Pose>> {
        for p in poses {
            p.validate()?;
        }
        let rotations: Vec<_> = poses.iter().map(|p| rotation(p.orientation)).collect();
        let mut data = Vec::new();
        for (i, p) in poses.iter().enumerate() {
            for (j, q) in poses.iter().enumerate() {
                if i == j {
                    continue;
                }
                let delta = sub(p.position, q.position);
                if norm(delta) <= self.pair_cutoff_a {
                    let inverse = transpose(rotations[j]);
                    data.push(Pose {
                        position: matvec(inverse, delta),
                        orientation: quaternion(matmul(inverse, rotations[i])),
                    });
                }
            }
        }
        Ok(data)
    }

    pub fn fit(&self, poses: &[Pose]) -> Result<ConditionalFit> {
        let data = self.observations(poses)?;
        let mut result = ConditionalFit {
            log_probabilities: Vec::new(),
            log_scores: Vec::new(),
            covariance_penalties: Vec::new(),
            mean_log_densities: Vec::new(),
            data_count: data.len(),
            counts: Vec::new(),
        };
        let uniform_log = -(4. * PI / 3.).ln() - 3. * self.pair_cutoff_a.ln();
        let mut log_factorial = 0.;
        for k in 0..=self.config.k_max {
            let fit = self.fit_count(&data, k)?;
            let penalty = covariance_penalty(
                &fit.fitted_components,
                self.config.translation_scale_a,
                self.config.covariance_exponent,
            )?;
            let mean_log_density = if data.is_empty() {
                0.
            } else if k == 0 {
                uniform_log
            } else {
                let model = self.components_model(fit.fitted_components.clone())?;
                let mut total = 0.;
                for observation in &data {
                    let learned = model.relative_log_density(
                        observation.position,
                        rotation(observation.orientation),
                    )?;
                    total += log_add(
                        self.config.uniform_weight.ln() + uniform_log,
                        if self.config.uniform_weight == 1. {
                            f64::NEG_INFINITY
                        } else {
                            (-self.config.uniform_weight).ln_1p() + learned
                        },
                    );
                }
                total / data.len() as f64
            };
            if k > 0 {
                log_factorial += (k as f64).ln();
            }
            let score = k as f64 * self.config.basin_activity.ln() - log_factorial
                + self.config.score_temperature * mean_log_density
                - self.config.penalty_strength * penalty;
            ensure!(
                score.is_finite() && penalty.is_finite() && mean_log_density.is_finite(),
                "Nonfinite conditional count score"
            );
            result.log_scores.push(score);
            result.covariance_penalties.push(penalty);
            result.mean_log_densities.push(mean_log_density);
            result.counts.push(fit);
        }
        let log_normalizer = log_sum_exp(&result.log_scores);
        result.log_probabilities = result
            .log_scores
            .iter()
            .map(|x| x - log_normalizer)
            .collect();
        Ok(result)
    }

    pub fn initialize(&self, fit: &ConditionalFit, rng: &mut StdRng) -> Result<ConditionalState> {
        ensure!(
            fit.log_probabilities.len() == self.config.k_max + 1
                && fit.log_probabilities.iter().all(|x| x.is_finite()),
            "Invalid fitted count probabilities"
        );
        let k = match self.config.initialization {
            ConditionalInitialization::Random => sample_count(&fit.log_probabilities, rng),
            ConditionalInitialization::ModeZero => {
                fit.log_probabilities
                    .iter()
                    .enumerate()
                    .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
                    .unwrap()
                    .0
            }
            ConditionalInitialization::Zero => 0,
            ConditionalInitialization::FixedKZero => self.config.initial_k.unwrap(),
        };
        let eta = if self.config.initialization == ConditionalInitialization::Random {
            (0..parameter_dimension(k))
                .map(|_| StandardNormal.sample(rng))
                .collect()
        } else {
            vec![0.; parameter_dimension(k)]
        };
        let state = ConditionalState { k, eta };
        state.validate(&self.config)?;
        Ok(state)
    }

    /// One exact independent Gibbs update; scheduling/probability belongs to the caller.
    pub fn refresh(
        &self,
        fit: &ConditionalFit,
        state: &mut ConditionalState,
        rng: &mut StdRng,
    ) -> Result<()> {
        ensure!(
            fit.log_probabilities.len() == self.config.k_max + 1,
            "Fit count range mismatch"
        );
        let k = sample_count(&fit.log_probabilities, rng);
        *state = ConditionalState {
            k,
            eta: (0..parameter_dimension(k))
                .map(|_| StandardNormal.sample(rng))
                .collect(),
        };
        state.validate(&self.config)
    }

    pub fn model(
        &self,
        fit: &ConditionalFit,
        state: &ConditionalState,
    ) -> Result<FrozenRelativePoseProposal> {
        state.validate(&self.config)?;
        if state.k == 0 {
            return FrozenRelativePoseProposal::uniform_only_open(
                self.cube_lengths,
                &self.shape_sha256,
            );
        }
        let coordinates = fit.coordinates(state, self.config.residual_scale)?;
        let count = fit
            .counts
            .get(state.k)
            .ok_or_else(|| anyhow::anyhow!("K outside fit"))?;
        ensure!(
            count.anchors.len() == state.k,
            "Invalid fitted chart dimensions"
        );
        let components = self.decode(&count.anchors, &coordinates)?;
        self.components_model(components)
    }

    fn components_model(
        &self,
        components: Vec<GaussianComponentParameters>,
    ) -> Result<FrozenRelativePoseProposal> {
        FrozenRelativePoseProposal::from_components_open(
            components,
            self.config.angular_length_a,
            self.cube_lengths,
            self.config.uniform_weight,
            &self.shape_sha256,
            &self.shape_sha256,
        )
    }

    fn fit_count(&self, data: &[Pose], k: usize) -> Result<ConditionalCountFit> {
        if k == 0 {
            return Ok(ConditionalCountFit {
                anchors: Vec::new(),
                coordinates: Vec::new(),
                fitted_components: Vec::new(),
            });
        }
        let identity = Pose {
            position: [0.; 3],
            orientation: [1., 0., 0., 0.],
        };
        let mut centers = vec![data.first().copied().unwrap_or(identity)];
        while centers.len() < k {
            let next = data
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let distance = centers
                        .iter()
                        .map(|q| self.metric_distance(p, q))
                        .fold(f64::INFINITY, f64::min);
                    (i, distance)
                })
                .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
                .map(|(i, _)| data[i])
                .unwrap_or(identity);
            centers.push(next);
        }
        let mut assignments = vec![0; data.len()];
        for _ in 0..self.config.fit_iterations {
            for (a, p) in assignments.iter_mut().zip(data) {
                *a = centers
                    .iter()
                    .enumerate()
                    .map(|(j, q)| (j, self.metric_distance(p, q)))
                    .min_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ia.cmp(ib)))
                    .unwrap()
                    .0;
            }
            for (j, center) in centers.iter_mut().enumerate() {
                let members: Vec<_> = data
                    .iter()
                    .zip(&assignments)
                    .filter_map(|(p, a)| (*a == j).then_some(*p))
                    .collect();
                if members.is_empty() {
                    continue;
                }
                let position = std::array::from_fn(|d| {
                    members.iter().map(|p| p.position[d]).sum::<f64>() / members.len() as f64
                });
                // A bounded deterministic quaternion average, aligned to the preceding seed.
                let mut sum = [0.; 4];
                for p in &members {
                    let dot: f64 = p
                        .orientation
                        .iter()
                        .zip(center.orientation)
                        .map(|(a, b)| a * b)
                        .sum();
                    let sign = if dot < 0. { -1. } else { 1. };
                    for (d, v) in sum.iter_mut().enumerate() {
                        *v += sign * p.orientation[d];
                    }
                }
                let length = sum.iter().fold(0_f64, |a, b| a.hypot(*b));
                if length > 1e-12 {
                    center.orientation = sum.map(|x| x / length);
                }
                center.position = position;
            }
        }
        // Final assignments use the final centers rather than the preceding iteration.
        for (a, p) in assignments.iter_mut().zip(data) {
            *a = centers
                .iter()
                .enumerate()
                .map(|(j, q)| (j, self.metric_distance(p, q)))
                .min_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ia.cmp(ib)))
                .unwrap()
                .0;
        }
        let mut components = Vec::new();
        for (j, center) in centers.iter().enumerate() {
            let residuals: Vec<_> = data
                .iter()
                .zip(&assignments)
                .filter_map(|(p, a)| (*a == j).then(|| self.fit_residual(p, center)))
                .collect();
            let n = residuals.len();
            let mean: [f64; 6] = if n == 0 {
                [0.; 6]
            } else {
                std::array::from_fn(|d| residuals.iter().map(|x| x[d]).sum::<f64>() / n as f64)
            };
            // Six unit-variance pseudo-observations make one/empty samples broad.
            let mut covariance: Mat6 = std::array::from_fn(|i| {
                std::array::from_fn(|l| {
                    (if i == l { 6. } else { 0. }
                        + residuals
                            .iter()
                            .map(|x| (x[i] - mean[i]) * (x[l] - mean[l]))
                            .sum::<f64>())
                        / (n as f64 + 6.)
                })
            });
            covariance = eigen_regularize(
                covariance,
                self.config.covariance_floor,
                self.config.covariance_ceiling,
            );
            components.push(GaussianComponentParameters {
                anchor_position: center.position,
                anchor_rotation: rotation(center.orientation),
                mean: mean.map(|x| x * self.config.translation_scale_a),
                covariance: covariance
                    .map(|row| row.map(|x| x * self.config.translation_scale_a.powi(2))),
                weight: (n as f64 + 1.) / (data.len() as f64 + k as f64),
            });
        }
        let coordinates = self.encode(&components)?;
        // Score exactly the decoded center, including deterministic bounded logit handling.
        let fitted_components = self.decode(&centers, &coordinates)?;
        Ok(ConditionalCountFit {
            anchors: centers,
            coordinates,
            fitted_components,
        })
    }

    fn metric_distance(&self, a: &Pose, b: &Pose) -> f64 {
        let t = norm(sub(a.position, b.position)) / self.config.translation_scale_a;
        let dot: f64 = a
            .orientation
            .iter()
            .zip(b.orientation)
            .map(|(x, y)| x * y)
            .sum();
        let angle = 2. * dot.abs().min(1.).acos();
        t * t + (self.config.angular_length_a * angle / self.config.translation_scale_a).powi(2)
    }

    fn fit_residual(&self, p: &Pose, anchor: &Pose) -> [f64; 6] {
        let q = quaternion(matmul(
            rotation(p.orientation),
            transpose(rotation(anchor.orientation)),
        ));
        // This clipping defines the approximate fit only, never its scored density.
        std::array::from_fn(|d| {
            let value = if d < 3 {
                (p.position[d] - anchor.position[d]) / self.config.translation_scale_a
            } else {
                self.config.angular_length_a / self.config.translation_scale_a * q[d - 2]
                    / q[0].max(1e-6)
            };
            value.clamp(-MEAN_BOUND / 4., MEAN_BOUND / 4.)
        })
    }

    fn diagonal_bounds(&self) -> (f64, f64) {
        (
            (0.5 * self.config.covariance_floor.sqrt()).ln(),
            (2. * self.config.covariance_ceiling.sqrt()).ln(),
        )
    }

    fn encode(&self, components: &[GaussianComponentParameters]) -> Result<Vec<f64>> {
        let mut coordinates = Vec::with_capacity(parameter_dimension(components.len()));
        let (lo, hi) = self.diagonal_bounds();
        let off_bound = 2. * self.config.covariance_ceiling.sqrt();
        for c in components {
            for mean in c.mean {
                coordinates.push(
                    MEAN_BOUND
                        * inverse_bounded(mean / self.config.translation_scale_a / MEAN_BOUND)?,
                );
            }
            let dimensionless = std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    c.covariance[i][j] / self.config.translation_scale_a.powi(2)
                        - if i == j {
                            self.config.covariance_floor / 16.
                        } else {
                            0.
                        }
                })
            });
            let lower = cholesky(dimensionless)?;
            for (i, row) in lower.iter().enumerate() {
                for (j, value) in row.iter().enumerate().take(i + 1) {
                    coordinates.push(if i == j {
                        0.5 * (hi - lo) * inverse_bounded((2. * value.ln() - lo - hi) / (hi - lo))?
                    } else {
                        off_bound * inverse_bounded(value / off_bound)?
                    });
                }
            }
        }
        if let Some(reference) = components.last() {
            for c in components.iter().take(components.len() - 1) {
                coordinates.push(
                    LOGIT_BOUND
                        * inverse_bounded((c.weight.ln() - reference.weight.ln()) / LOGIT_BOUND)?,
                );
            }
        }
        Ok(coordinates)
    }

    fn decode(
        &self,
        anchors: &[Pose],
        coordinates: &[f64],
    ) -> Result<Vec<GaussianComponentParameters>> {
        ensure!(
            coordinates.len() == parameter_dimension(anchors.len())
                && coordinates.iter().all(|x| x.is_finite()),
            "Invalid packed conditional coordinates"
        );
        let (lo, hi) = self.diagonal_bounds();
        let off_bound = 2. * self.config.covariance_ceiling.sqrt();
        let mut logits: Vec<_> = coordinates[27 * anchors.len()..]
            .iter()
            .map(|x| LOGIT_BOUND * (x / LOGIT_BOUND).tanh())
            .collect();
        logits.push(0.);
        let normalizer = log_sum_exp(&logits);
        let mut components = Vec::new();
        for (k, anchor) in anchors.iter().enumerate() {
            let v = &coordinates[27 * k..27 * (k + 1)];
            let mean = std::array::from_fn(|d| {
                self.config.translation_scale_a * MEAN_BOUND * (v[d] / MEAN_BOUND).tanh()
            });
            let mut lower = [[0.; 6]; 6];
            let mut index = 6;
            for (i, row) in lower.iter_mut().enumerate() {
                for (j, value) in row.iter_mut().enumerate().take(i + 1) {
                    *value = self.config.translation_scale_a
                        * if i == j {
                            (0.5 * (lo + hi) + 0.5 * (hi - lo) * (2. * v[index] / (hi - lo)).tanh())
                                .exp()
                        } else {
                            off_bound * (v[index] / off_bound).tanh()
                        };
                    index += 1;
                }
            }
            let covariance = std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    (0..=i.min(j))
                        .map(|d| lower[i][d] * lower[j][d])
                        .sum::<f64>()
                        + if i == j {
                            self.config.translation_scale_a.powi(2) * self.config.covariance_floor
                                / 16.
                        } else {
                            0.
                        }
                })
            });
            components.push(GaussianComponentParameters {
                anchor_position: anchor.position,
                anchor_rotation: rotation(anchor.orientation),
                mean,
                covariance,
                weight: (logits[k] - normalizer).exp(),
            });
        }
        Ok(components)
    }

    /// Exact import only for the fitted chart anchors and configured metric.
    /// Rotational recharting of a Cayley Gaussian is not generally Gaussian.
    pub fn state_from_components(
        &self,
        fit: &ConditionalFit,
        components: &[GaussianComponentParameters],
    ) -> Result<ConditionalState> {
        let k = components.len();
        ensure!(
            k <= self.config.k_max,
            "Imported component count exceeds K_max"
        );
        if k == 0 {
            return Ok(ConditionalState { k, eta: Vec::new() });
        }
        let target = fit
            .counts
            .get(k)
            .ok_or_else(|| anyhow::anyhow!("Imported K outside fit"))?;
        ensure!(
            target.anchors.len() == k && target.coordinates.len() == parameter_dimension(k),
            "Invalid fitted chart dimensions"
        );
        for (c, anchor) in components.iter().zip(&target.anchors) {
            ensure!(
                c.anchor_position == anchor.position
                    && c.anchor_rotation == rotation(anchor.orientation),
                "Exact conditional import requires identical fitted chart anchors"
            );
        }
        // Validate SPD and normalized weights. The caller must validate the source
        // file's shape digest and angular metric before extracting its components.
        self.components_model(components.to_vec())?;
        let coordinates = self.encode(components)?;
        let state = ConditionalState {
            k,
            eta: coordinates
                .iter()
                .zip(&target.coordinates)
                .map(|(a, b)| (a - b) / self.config.residual_scale)
                .collect(),
        };
        state.validate(&self.config)?;
        Ok(state)
    }

    /// Import a validated frozen model, including its original identity metadata.
    pub fn state_from_model(
        &self,
        fit: &ConditionalFit,
        model: &FrozenRelativePoseProposal,
    ) -> Result<ConditionalState> {
        ensure!(
            !model.has_reciprocal_components(),
            "Conditional state import requires unwrapped Gaussian charts"
        );
        ensure!(
            model.shape_sha256() == self.shape_sha256 && !model.is_periodic(),
            "Imported conditional model must be open and match physical shape SHA256"
        );
        ensure!(
            model.component_count() == 0 || model.angular_length() == self.config.angular_length_a,
            "Imported conditional model angular metric differs"
        );
        ensure!(
            model.box_lengths() == self.cube_lengths
                && model.uniform_weight()
                    == if model.component_count() == 0 {
                        1.
                    } else {
                        self.config.uniform_weight
                    },
            "Imported conditional model defensive uniform law differs"
        );
        self.state_from_components(fit, &model.component_parameters())
    }
}

/// R_s = sum_k w_k exp[-s log det(C_k / translation_scale^2)/(2d)], d=6.
/// s=0 explicitly disables the entire penalty, including its K=0 value.
pub fn covariance_penalty(
    components: &[GaussianComponentParameters],
    translation_scale: f64,
    exponent: f64,
) -> Result<f64> {
    ensure!(
        translation_scale.is_finite()
            && translation_scale > 0.
            && exponent.is_finite()
            && exponent >= 0.,
        "Invalid covariance penalty metric or exponent"
    );
    if exponent == 0. || components.is_empty() {
        return Ok(0.);
    }
    let mut value = 0.;
    for c in components {
        let covariance = c
            .covariance
            .map(|row| row.map(|x| x / translation_scale.powi(2)));
        let lower = cholesky(covariance)?;
        let logdet = 2. * (0..6).map(|i| lower[i][i].ln()).sum::<f64>();
        value += c.weight * (-exponent * logdet / 12.).exp();
    }
    ensure!(value.is_finite(), "Unrepresentable covariance penalty");
    Ok(value)
}

fn inverse_bounded(x: f64) -> Result<f64> {
    ensure!(
        x.is_finite() && x.abs() < 1.,
        "Imported parameter lies outside conditional coordinate bounds"
    );
    Ok(x.atanh())
}

fn cholesky(covariance: Mat6) -> Result<Mat6> {
    let mut lower = [[0.; 6]; 6];
    for i in 0..6 {
        for j in 0..=i {
            let r = covariance[i][j] - (0..j).map(|d| lower[i][d] * lower[j][d]).sum::<f64>();
            if i == j {
                ensure!(r.is_finite() && r > 0., "Conditional covariance is not SPD");
                lower[i][j] = r.sqrt();
            } else {
                lower[i][j] = r / lower[j][j];
            }
        }
    }
    Ok(lower)
}

/// Fixed 20 Jacobi sweeps, followed by deterministic spectral clipping.
fn eigen_regularize(mut a: Mat6, floor: f64, ceiling: f64) -> Mat6 {
    let mut vectors: Mat6 =
        std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1. } else { 0. }));
    for _ in 0..20 {
        for p in 0..6 {
            for q in p + 1..6 {
                if a[p][q].abs() <= 1e-15 {
                    continue;
                }
                let angle = 0.5 * (2. * a[p][q]).atan2(a[q][q] - a[p][p]);
                let (s, c) = angle.sin_cos();
                let app = a[p][p];
                let aqq = a[q][q];
                let apq = a[p][q];
                for r in 0..6 {
                    if r == p || r == q {
                        continue;
                    }
                    let arp = a[r][p];
                    let arq = a[r][q];
                    a[r][p] = c * arp - s * arq;
                    a[p][r] = a[r][p];
                    a[r][q] = s * arp + c * arq;
                    a[q][r] = a[r][q];
                }
                a[p][p] = c * c * app - 2. * s * c * apq + s * s * aqq;
                a[q][q] = s * s * app + 2. * s * c * apq + c * c * aqq;
                a[p][q] = 0.;
                a[q][p] = 0.;
                for row in &mut vectors {
                    let vp = row[p];
                    let vq = row[q];
                    row[p] = c * vp - s * vq;
                    row[q] = s * vp + c * vq;
                }
            }
        }
    }
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            (0..6)
                .map(|d| vectors[i][d] * a[d][d].clamp(floor, ceiling) * vectors[j][d])
                .sum()
        })
    })
}

fn log_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}
fn log_sum_exp(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::NEG_INFINITY, log_add)
}
fn sample_count(log_probabilities: &[f64], rng: &mut StdRng) -> usize {
    let draw = rng.random::<f64>();
    let mut sum = 0.;
    for (k, logp) in log_probabilities.iter().enumerate() {
        sum += logp.exp();
        if draw < sum {
            return k;
        }
    }
    log_probabilities.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn singleton_observation_and_extra_components_have_finite_fits() {
        let engine = ConditionalEngine::new(
            1.,
            10.,
            "0000000000000000000000000000000000000000000000000000000000000000",
            ConditionalConfig::default(),
        )
        .unwrap();
        let observation = Pose {
            position: [1., 2., 3.],
            orientation: [0., 1., 0., 0.],
        };
        for k in 1..=4 {
            let fit = engine.fit_count(&[observation], k).unwrap();
            assert!(fit.coordinates.iter().all(|x| x.is_finite()));
            assert_eq!(fit.fitted_components.len(), k);
            assert!(
                covariance_penalty(&fit.fitted_components, 10., 6.)
                    .unwrap()
                    .is_finite()
            );
        }
    }
}
