//! Exact active-subset sampling for an immutable learned proposal dictionary.
//!
//! The component count follows a truncated Poisson law. Conditional on its
//! count, a subset is sampled with probability proportional to the product of
//! its fixed label weights. Log-space elementary symmetric polynomials provide
//! both the exact normalizer and an O(J^2) sampler without enumerating subsets.
//!
//! The retained target is `pi(X) rho(mask) phi(eta)`. The mask law uses reference
//! weights only and is independent of X and eta. Physical moves retain both
//! auxiliaries and need the candidate-configuration reverse proposal, but no
//! mask-density correction. This reduces the active proposal dictionary; it is
//! not dimension-changing parameter storage or discovery of new basins.
use anyhow::{Result, ensure};
use rand::{RngExt, rngs::StdRng};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AtlasMaskLabelLaw {
    Uniform,
    #[default]
    AtlasWeight,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AtlasMaskConfig {
    /// Poisson activity for K, before truncation to the permitted count range.
    pub activity: f64,
    pub min_components: usize,
    pub max_components: Option<usize>,
    pub label_law: AtlasMaskLabelLaw,
    /// Scheduling probability, applied by the runner to exact Gibbs refreshes.
    pub refresh_probability: f64,
    /// Begin at the full dictionary if the configured count support permits it.
    pub initial_full: bool,
}

impl Default for AtlasMaskConfig {
    fn default() -> Self {
        Self {
            activity: 8.,
            min_components: 0,
            max_components: None,
            label_law: AtlasMaskLabelLaw::AtlasWeight,
            refresh_probability: 1.,
            initial_full: true,
        }
    }
}

impl AtlasMaskConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.activity.is_finite() && self.activity > 0.,
            "Atlas mask activity must be finite and positive"
        );
        ensure!(
            self.max_components
                .is_none_or(|max| max >= self.min_components),
            "Atlas mask maximum count must be at least its minimum"
        );
        ensure!(
            self.refresh_probability.is_finite() && (0. ..=1.).contains(&self.refresh_probability),
            "Atlas mask refresh_probability must lie in [0,1]"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AtlasMaskState {
    /// Sorted unique indices into the complete immutable reference atlas.
    pub labels: Vec<usize>,
}

impl AtlasMaskState {
    pub fn validate(&self, component_count: usize) -> Result<()> {
        ensure!(
            self.labels.iter().all(|&j| j < component_count)
                && self.labels.windows(2).all(|pair| pair[0] < pair[1]),
            "Atlas mask labels must be sorted, unique, and inside the reference atlas"
        );
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct AtlasMaskEngine {
    config: AtlasMaskConfig,
    j: usize,
    min_components: usize,
    max_components: usize,
    log_weights: Vec<f64>,
    log_count_probabilities: Vec<f64>,
    /// suffix[i][k] = log e_k(b_i, ..., b_{J-1}).
    suffix: Vec<Vec<f64>>,
}

fn log_add_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let high = a.max(b);
    high + (-(a - b).abs()).exp().ln_1p()
}

impl AtlasMaskEngine {
    pub fn new(reference_weights: &[f64], config: AtlasMaskConfig) -> Result<Self> {
        config.validate()?;
        ensure!(
            reference_weights.iter().all(|w| w.is_finite() && *w > 0.),
            "Atlas mask reference weights must be finite and positive"
        );
        let j = reference_weights.len();
        let max_components = config.max_components.unwrap_or(j);
        ensure!(
            config.min_components <= max_components && max_components <= j,
            "Atlas mask count bounds must lie inside the complete reference atlas"
        );
        let log_weights: Vec<f64> = match config.label_law {
            AtlasMaskLabelLaw::Uniform => vec![0.; j],
            AtlasMaskLabelLaw::AtlasWeight => {
                // A common rescaling cancels separately at every cardinality.
                // Centering logs keeps very small valid reference weights safe.
                let max_log = reference_weights
                    .iter()
                    .map(|w| w.ln())
                    .fold(f64::NEG_INFINITY, f64::max);
                reference_weights.iter().map(|w| w.ln() - max_log).collect()
            }
        };
        let mut suffix = vec![vec![f64::NEG_INFINITY; j + 1]; j + 1];
        suffix[j][0] = 0.;
        for i in (0..j).rev() {
            suffix[i][0] = 0.;
            for k in 1..=j - i {
                suffix[i][k] = log_add_exp(suffix[i + 1][k], log_weights[i] + suffix[i + 1][k - 1]);
            }
        }
        let mut log_count_probabilities = vec![f64::NEG_INFINITY; j + 1];
        let mut log_factorial = 0.;
        let mut normalization = f64::NEG_INFINITY;
        for k in 0..=j {
            if k > 0 {
                log_factorial += (k as f64).ln();
            }
            if (config.min_components..=max_components).contains(&k) {
                let value = k as f64 * config.activity.ln() - log_factorial;
                log_count_probabilities[k] = value;
                normalization = log_add_exp(normalization, value);
            }
        }
        for value in &mut log_count_probabilities {
            *value -= normalization;
        }
        Ok(Self {
            min_components: config.min_components,
            max_components,
            config,
            j,
            log_weights,
            log_count_probabilities,
            suffix,
        })
    }

    pub fn component_count(&self) -> usize {
        self.j
    }

    /// Length J+1. Unsupported component counts have log probability -infinity.
    pub fn log_count_probabilities(&self) -> &[f64] {
        &self.log_count_probabilities
    }

    pub fn validate_state(&self, state: &AtlasMaskState) -> Result<()> {
        state.validate(self.j)?;
        ensure!(
            (self.min_components..=self.max_components).contains(&state.labels.len()),
            "Atlas mask count lies outside its configured support"
        );
        Ok(())
    }

    pub fn log_probability(&self, state: &AtlasMaskState) -> Result<f64> {
        self.validate_state(state)?;
        let k = state.labels.len();
        // These subset laws are deterministic and avoiding cancellation gives
        // exactly zero log conditional probability at both support endpoints.
        if k == 0 || k == self.j {
            return Ok(self.log_count_probabilities[k]);
        }
        Ok(self.log_count_probabilities[k]
            + state
                .labels
                .iter()
                .map(|&j| self.log_weights[j])
                .sum::<f64>()
            - self.suffix[0][k])
    }

    pub fn initialize(&self, rng: &mut StdRng) -> Result<AtlasMaskState> {
        if self.config.initial_full && self.max_components == self.j {
            return Ok(AtlasMaskState {
                labels: (0..self.j).collect(),
            });
        }
        Ok(self.draw(rng))
    }

    /// Exact Gibbs redraw. Refresh scheduling is the caller's responsibility.
    pub fn refresh(&self, state: &mut AtlasMaskState, rng: &mut StdRng) -> Result<()> {
        self.validate_state(state)?;
        *state = self.draw(rng);
        Ok(())
    }

    fn draw(&self, rng: &mut StdRng) -> AtlasMaskState {
        let k = if self.min_components == self.max_components {
            self.min_components
        } else {
            // The last supported count absorbs floating-point cumulative error.
            let target = rng.random::<f64>();
            let mut sum = 0.;
            let mut choice = self.max_components;
            for k in self.min_components..self.max_components {
                sum += self.log_count_probabilities[k].exp();
                if target < sum {
                    choice = k;
                    break;
                }
            }
            choice
        };
        let mut labels = Vec::with_capacity(k);
        let mut remaining = k;
        for i in 0..self.j {
            if remaining == 0 {
                break;
            }
            if remaining == self.j - i {
                labels.extend(i..self.j);
                break;
            }
            let log_inclusion =
                self.log_weights[i] + self.suffix[i + 1][remaining - 1] - self.suffix[i][remaining];
            if rng.random::<f64>() < log_inclusion.exp().min(1.) {
                labels.push(i);
                remaining -= 1;
            }
        }
        debug_assert_eq!(labels.len(), k);
        AtlasMaskState { labels }
    }
}
