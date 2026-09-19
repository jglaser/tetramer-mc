//! Normalized, history-free Gaussian law for fixed-K proposal mixture means.
//!
//! Coordinates a_k = f_k(X) + s_k(X) eta_k, eta_k ~ N(0,I), specify
//! mean_k = frozen_mean_k + frozen_Cholesky_k a_k. All other model parameters
//! are fixed. In (X,eta), the target is pi(X) phi(eta). Physical proposals
//! retain eta and MUST use the reconstructed model at Y for their reverse q.
//! GCA and common shifts independent of eta need no additional acceptance.
use crate::{math::*, proposal::FrozenRelativePoseProposal};
use anyhow::{Result, ensure};
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};

fn gain() -> f64 {
    0.25
}
fn noise() -> f64 {
    0.1
}
fn cutoff() -> f64 {
    6.0
}
fn clip() -> f64 {
    3.0
}
fn shrinkage() -> f64 {
    1.0
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuxiliaryConfig {
    #[serde(default = "gain")]
    pub gain: f64,
    #[serde(default = "noise")]
    pub noise: f64,
    #[serde(default = "cutoff")]
    pub cutoff: f64,
    #[serde(default = "clip")]
    pub clip: f64,
    #[serde(default = "shrinkage")]
    pub shrinkage: f64,
}
impl Default for AuxiliaryConfig {
    fn default() -> Self {
        Self {
            gain: gain(),
            noise: noise(),
            cutoff: cutoff(),
            clip: clip(),
            shrinkage: shrinkage(),
        }
    }
}
impl AuxiliaryConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.gain.is_finite() && self.gain >= 0.,
            "Invalid auxiliary gain"
        );
        ensure!(
            [self.noise, self.cutoff, self.clip, self.shrinkage]
                .iter()
                .all(|x| x.is_finite() && *x > 0.),
            "Auxiliary noise/cutoff/clip/shrinkage must be positive finite"
        );
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConditionalFit {
    pub center: Vec<[f64; 6]>,
    pub scale: Vec<f64>,
    pub counts: Vec<u64>,
}
impl ConditionalFit {
    pub fn coordinates(&self, eta: &[[f64; 6]]) -> Result<Vec<[f64; 6]>> {
        ensure!(
            eta.len() == self.center.len() && eta.iter().flatten().all(|x| x.is_finite()),
            "Invalid auxiliary residual"
        );
        Ok(self
            .center
            .iter()
            .zip(&self.scale)
            .zip(eta)
            .map(|((f, s), e)| std::array::from_fn(|d| f[d] + s * e[d]))
            .collect())
    }
    /// Density with respect to whitened mean coordinates a. The constant base
    /// Cholesky determinants can be included for actual means, but cancel.
    pub fn log_density(&self, coordinates: &[[f64; 6]]) -> Result<f64> {
        ensure!(
            coordinates.len() == self.center.len()
                && coordinates.iter().flatten().all(|x| x.is_finite()),
            "Invalid auxiliary coordinates"
        );
        let mut value = -3. * self.center.len() as f64 * (2. * std::f64::consts::PI).ln();
        for ((a, f), s) in coordinates.iter().zip(&self.center).zip(&self.scale) {
            value -= 6. * s.ln();
            value -= 0.5 * (0..6).map(|d| ((a[d] - f[d]) / s).powi(2)).sum::<f64>();
        }
        Ok(value)
    }
    pub fn transport_log_jacobian(&self, next: &Self) -> Result<f64> {
        ensure!(
            self.scale.len() == next.scale.len(),
            "Transport dimension differs"
        );
        Ok(6.
            * self
                .scale
                .iter()
                .zip(&next.scale)
                .map(|(a, b)| b.ln() - a.ln())
                .sum::<f64>())
    }
}

pub fn fit(
    base: &FrozenRelativePoseProposal,
    poses: &[Pose],
    options: &AuxiliaryConfig,
) -> Result<ConditionalFit> {
    options.validate()?;
    ensure!(
        !base.is_periodic(),
        "Auxiliary prototype currently requires the spherical/open proposal"
    );
    let k = base.component_count();
    let mut center = vec![[0.; 6]; k];
    let mut counts = vec![0u64; k];
    let transforms: Vec<_> = poses.iter().map(|p| rotation(p.orientation)).collect();
    for (i, p) in poses.iter().enumerate() {
        p.validate()?;
        for (j, q) in poses.iter().enumerate() {
            if i == j {
                continue;
            }
            let inverse = transpose(transforms[j]);
            let t = matvec(inverse, sub(p.position, q.position));
            let r = matmul(inverse, transforms[i]);
            let mut best: Option<(usize, [f64; 6], f64)> = None;
            for c in 0..k {
                if let Some(residual) = base.component_residual(t, r, c) {
                    let d2: f64 = residual.iter().map(|x| x * x).sum();
                    if d2.is_finite() && best.as_ref().is_none_or(|v| d2 < v.2) {
                        best = Some((c, residual, d2));
                    }
                }
            }
            if let Some((c, residual, d2)) = best {
                if d2 <= options.cutoff.powi(2) {
                    counts[c] += 1;
                    let clip = (options.clip / d2.sqrt().max(1e-300)).min(1.);
                    for d in 0..6 {
                        center[c][d] += clip * residual[d];
                    }
                }
            }
        }
    }
    let scale: Vec<f64> = counts
        .iter()
        .map(|n| options.noise / (options.shrinkage + *n as f64).sqrt())
        .collect();
    for (v, n) in center.iter_mut().zip(&counts) {
        for x in v {
            *x *= options.gain / (options.shrinkage + *n as f64);
        }
    }
    ensure!(
        scale.iter().all(|s| s.is_finite() && *s > 0.)
            && center.iter().flatten().all(|f| f.is_finite()),
        "Unrepresentable auxiliary conditional fit"
    );
    Ok(ConditionalFit {
        center,
        scale,
        counts,
    })
}

pub fn draw_eta(base: &FrozenRelativePoseProposal, rng: &mut StdRng) -> Vec<[f64; 6]> {
    (0..base.component_count())
        .map(|_| std::array::from_fn(|_| StandardNormal.sample(rng)))
        .collect()
}

pub fn model(
    base: &FrozenRelativePoseProposal,
    poses: &[Pose],
    eta: &[[f64; 6]],
    options: &AuxiliaryConfig,
) -> Result<FrozenRelativePoseProposal> {
    base.with_whitened_mean_offsets(&fit(base, poses, options)?.coordinates(eta)?)
}
