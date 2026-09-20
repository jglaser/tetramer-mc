//! Reversible births/deaths of labelled Gaussian proposal components.
//!
//! Target in stored coordinates: pi(X) p(K) product[p(c_j) phi_6(eta_j)].
//! p(K) is a Poisson law truncated to min..=max; label probabilities are the
//! fixed atlas weights. Component means are reconstructed by auxiliary::model.
//! Birth inserts a prior draw into a uniform slot; death deletes a uniform
//! slot. The latent Jacobian is one; label/normal/slot densities cancel.
//! Equal-probability birth/death choices include NULL attempts at boundaries.
//! This changes dimension but does not learn p(K) or accumulate training data.
use anyhow::{Result, ensure};
use rand::{RngExt, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};

fn one() -> usize {
    1
}
fn maximum() -> usize {
    24
}
fn initial() -> usize {
    8
}
fn mean() -> f64 {
    8.
}
fn attempts() -> usize {
    4
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RjConfig {
    #[serde(default = "one")]
    pub min_components: usize,
    #[serde(default = "maximum")]
    pub max_components: usize,
    #[serde(default = "initial")]
    pub initial_components: usize,
    #[serde(default = "mean")]
    pub poisson_mean: f64,
    #[serde(default = "attempts")]
    pub attempts_per_sweep: usize,
}
impl Default for RjConfig {
    fn default() -> Self {
        Self {
            min_components: one(),
            max_components: maximum(),
            initial_components: initial(),
            poisson_mean: mean(),
            attempts_per_sweep: attempts(),
        }
    }
}
impl RjConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.min_components >= 1
                && self.min_components <= self.initial_components
                && self.initial_components <= self.max_components
                && self.max_components <= 1024,
            "Invalid RJ component-count limits"
        );
        ensure!(
            self.poisson_mean.is_finite()
                && self.poisson_mean > 0.
                && self.attempts_per_sweep > 0
                && self.attempts_per_sweep <= 1024,
            "Invalid RJ prior or sweep budget"
        );
        Ok(())
    }
    pub fn birth_log_ratio(&self, k: usize) -> f64 {
        self.poisson_mean.ln() - ((k + 1) as f64).ln()
    }
    pub fn death_log_ratio(&self, k: usize) -> f64 {
        (k as f64).ln() - self.poisson_mean.ln()
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RjState {
    pub labels: Vec<usize>,
    pub eta: Vec<[f64; 6]>,
}
fn normal(rng: &mut StdRng) -> [f64; 6] {
    std::array::from_fn(|_| StandardNormal.sample(rng))
}
fn validate_weights(weights: &[f64]) -> Result<()> {
    ensure!(
        !weights.is_empty()
            && weights.iter().all(|v| v.is_finite() && *v > 0.)
            && (weights.iter().sum::<f64>() - 1.).abs() < 1e-10,
        "RJ label prior must be normalized and strictly positive"
    );
    Ok(())
}
fn draw_label(weights: &[f64], rng: &mut StdRng) -> usize {
    let u = rng.random::<f64>();
    let mut cumulative = 0.;
    for (i, w) in weights.iter().enumerate() {
        cumulative += w;
        if u < cumulative {
            return i;
        }
    }
    weights.len() - 1
}
impl RjState {
    pub fn new(config: &RjConfig, weights: &[f64], rng: &mut StdRng) -> Result<Self> {
        config.validate()?;
        validate_weights(weights)?;
        Ok(Self {
            labels: (0..config.initial_components)
                .map(|_| draw_label(weights, rng))
                .collect(),
            eta: (0..config.initial_components)
                .map(|_| normal(rng))
                .collect(),
        })
    }
    pub fn validate(&self, config: &RjConfig, dictionary_size: usize) -> Result<()> {
        config.validate()?;
        ensure!(
            self.labels.len() >= config.min_components
                && self.labels.len() <= config.max_components
                && self.eta.len() == self.labels.len()
                && self.labels.iter().all(|&i| i < dictionary_size)
                && self.eta.iter().flatten().all(|v| v.is_finite()),
            "Invalid RJ auxiliary state"
        );
        Ok(())
    }
    pub fn refresh_eta(&mut self, rng: &mut StdRng) {
        for value in &mut self.eta {
            *value = normal(rng);
        }
    }
    pub fn update(&mut self, config: &RjConfig, weights: &[f64], rng: &mut StdRng) -> Result<Jump> {
        self.validate(config, weights.len())?;
        validate_weights(weights)?;
        let k = self.labels.len();
        let birth = rng.random::<bool>();
        let mut jump = Jump {
            birth,
            accepted: false,
            boundary_null: false,
            before: k,
            after: k,
            slot: None,
            label: None,
            log_ratio: None,
        };
        if (birth && k == config.max_components) || (!birth && k == config.min_components) {
            jump.boundary_null = true;
            return Ok(jump);
        }
        let (slot, label, eta, ratio) = if birth {
            let slot = rng.random_range(0..=k);
            (
                slot,
                draw_label(weights, rng),
                normal(rng),
                config.birth_log_ratio(k),
            )
        } else {
            let slot = rng.random_range(0..k);
            (
                slot,
                self.labels[slot],
                self.eta[slot],
                config.death_log_ratio(k),
            )
        };
        let accepted = rng.random::<f64>().max(f64::MIN_POSITIVE).ln() < ratio.min(0.);
        if accepted {
            if birth {
                self.labels.insert(slot, label);
                self.eta.insert(slot, eta);
            } else {
                self.labels.remove(slot);
                self.eta.remove(slot);
            }
        }
        jump.accepted = accepted;
        jump.after = self.labels.len();
        jump.slot = Some(slot);
        jump.label = Some(label);
        jump.log_ratio = Some(ratio);
        Ok(jump)
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Jump {
    pub birth: bool,
    pub accepted: bool,
    pub boundary_null: bool,
    pub before: usize,
    pub after: usize,
    pub slot: Option<usize>,
    pub label: Option<usize>,
    pub log_ratio: Option<f64>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct JumpCounts {
    pub births: u64,
    pub deaths: u64,
    pub accepted_births: u64,
    pub accepted_deaths: u64,
    pub boundary_nulls: u64,
}
impl JumpCounts {
    pub fn record(&mut self, j: &Jump) {
        if j.birth {
            self.births += 1;
            self.accepted_births += u64::from(j.accepted);
        } else {
            self.deaths += 1;
            self.accepted_deaths += u64::from(j.accepted);
        }
        self.boundary_nulls += u64::from(j.boundary_null);
    }
    pub fn since(&self, old: &Self) -> Self {
        Self {
            births: self.births - old.births,
            deaths: self.deaths - old.deaths,
            accepted_births: self.accepted_births - old.accepted_births,
            accepted_deaths: self.accepted_deaths - old.accepted_deaths,
            boundary_nulls: self.boundary_nulls - old.boundary_nulls,
        }
    }
}
