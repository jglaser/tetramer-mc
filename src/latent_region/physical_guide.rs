//! A frozen latent guide as a normalized physical-pose proposal.
//!
//! Density is `q_D(u(x)) / J_D(u(x))` relative to d^3t times normalized
//! SO(3) Haar. The radius-4 indicator restricts only the uniform branch:
//! Gaussian or conditional fallback mass outside R4 remains present. The
//! optional hard-free law uses source capture/core geometry only to define its
//! normalized proposal, never as a physical-target or wall predicate.
use super::{Chart, ImportanceGuide, PI, contact_line::ContactLineGuide, log_add};
use crate::{docking::DockingConfig, geometry::SphereTree, math::*, simulation::hash_bytes};
use anyhow::{Context, Result, ensure};
use rand::rngs::StdRng;
use serde_json::Value;
use std::{fs, path::Path};

/// A density evaluation in world coordinates. `latent == None` denotes the
/// exact measure-zero Cayley seam, where this component density is defined as
/// zero. No finite band around that seam is removed.
#[derive(Clone, Debug)]
pub struct PhysicalGuideDensity {
    pub latent: Option<[f64; 6]>,
    pub in_reference_ball: bool,
    pub log_latent_density: Option<f64>,
    pub log_physical_jacobian: Option<f64>,
    pub log_physical_density: f64,
    /// Complete conditional-density trace, present only for the hard-free law.
    pub hard_free_line_density: Option<Value>,
    /// An analytic support zero, never a substituted numerical underflow.
    pub structural_zero: bool,
}

/// A single unconditional proposal; exterior points are ordinary outcomes.
#[derive(Clone, Debug)]
pub struct PhysicalGuideDraw {
    pub pose: Pose,
    pub latent: [f64; 6],
    pub latent_radius: f64,
    pub gaussian_component: Option<usize>,
    pub density: PhysicalGuideDensity,
}

/// Generation only: scoring this world pose remains a separate operation.
/// This avoids a discarded all-axis BVH traversal in the vessel normalizer.
#[derive(Clone, Debug)]
pub struct PhysicalGuidePoseDraw {
    pub pose: Pose,
    pub latent: [f64; 6],
    pub latent_radius: f64,
    pub gaussian_component: Option<usize>,
    pub hard_free_line_draw: Option<Value>,
}

enum PhysicalGuideLaw {
    Gaussian(ImportanceGuide),
    HardFreeLine(ContactLineGuide),
}

/// Wraps the existing checked chart and full Gaussian-mixture implementation.
/// Source capture metadata is retained for provenance, never used as a target.
pub struct PhysicalLatentGuide {
    chart: Chart,
    guide: PhysicalGuideLaw,
    region_sha256: String,
    guide_sha256: String,
    shape_sha256: String,
    physical_fixed_neighbors: Vec<Pose>,
    reference_capture_center: Vec3,
    reference_capture_radius: f64,
    log_volume: f64,
}

impl PhysicalLatentGuide {
    pub const REFERENCE_RADIUS: f64 = 4.;
    pub const DENSITY_MEASURE: &'static str =
        "Lebesgue center volume times normalized SO(3) Haar measure";

    pub fn from_files(region: &Path, guide: &Path, expected_shape_sha256: &str) -> Result<Self> {
        Self::from_bytes(&fs::read(region)?, &fs::read(guide)?, expected_shape_sha256)
    }

    /// Bind the exact archived bytes. Only the Gaussian guide schema is
    /// accepted; shell/ray conditionals are separate proposal laws.
    pub fn from_bytes(
        region_raw: &[u8],
        guide_raw: &[u8],
        expected_shape_sha256: &str,
    ) -> Result<Self> {
        Self::from_bytes_inner(region_raw, guide_raw, expected_shape_sha256, None)
    }

    /// Geometry-aware constructor. The vessel capture is deliberately replaced
    /// by the archived SOURCE capture before constructing the conditional law.
    /// The tree must come from the shape bytes bound by expected_shape_sha256.
    pub fn from_bytes_with_geometry(
        region_raw: &[u8],
        guide_raw: &[u8],
        expected_shape_sha256: &str,
        vessel_config: &DockingConfig,
        tree: &SphereTree,
    ) -> Result<Self> {
        Self::from_bytes_inner(
            region_raw,
            guide_raw,
            expected_shape_sha256,
            Some((vessel_config, tree)),
        )
    }

    fn from_bytes_inner(
        region_raw: &[u8],
        guide_raw: &[u8],
        expected_shape_sha256: &str,
        geometry: Option<(&DockingConfig, &SphereTree)>,
    ) -> Result<Self> {
        let region: Value = serde_json::from_slice(region_raw)?;
        ensure!(
            region["shape_sha256"].as_str() == Some(expected_shape_sha256),
            "Physical guide shape differs"
        );
        ensure!(
            region["mahalanobis_radius"].as_f64() == Some(Self::REFERENCE_RADIUS)
                && region
                    .get("minimum_mahalanobis_radius")
                    .map_or(Some(0.), Value::as_f64)
                    == Some(0.),
            "Physical guide requires the complete radius-4 reference ball"
        );
        ensure!(
            region["minimum_original_q"].as_f64() == Some(0.)
                && region
                    .get("minimum_original_q_inclusive")
                    .map_or(Some(true), Value::as_bool)
                    == Some(true)
                && region.get("maximum_original_q").is_none(),
            "Physical guide source must retain full R4 without a q filter"
        );
        let fixed: Pose = serde_json::from_value(region["fixed_neighbor"].clone())?;
        fixed.validate()?;
        let physical_fixed_neighbors: Vec<Pose> = region
            .get("physical_fixed_neighbors")
            .map(|v| serde_json::from_value(v.clone()))
            .transpose()?
            .unwrap_or_else(|| vec![fixed]);
        ensure!(
            !physical_fixed_neighbors.is_empty() && physical_fixed_neighbors.contains(&fixed),
            "Missing physical chart anchor"
        );
        for pose in &physical_fixed_neighbors {
            pose.validate()?;
        }
        let reference_capture_center: Vec3 =
            serde_json::from_value(region["capture_center"].clone())?;
        let reference_capture_radius = region["capture_radius"]
            .as_f64()
            .context("Missing source capture radius")?;
        ensure!(
            reference_capture_center.iter().all(|v| v.is_finite())
                && reference_capture_radius.is_finite()
                && reference_capture_radius > 0.,
            "Invalid source capture metadata"
        );
        let chart = Chart::new(
            &region,
            expected_shape_sha256,
            reference_capture_radius,
            fixed,
        )?;
        let region_sha256 = hash_bytes(region_raw);
        let guide_data: Value = serde_json::from_slice(guide_raw)?;
        let guide = if guide_data["schema"] == "defensive-hard-free-line-guide-v1" {
            let (vessel, tree) =
                geometry.context("Hard-free physical guide requires bound geometry")?;
            ensure!(
                vessel.fixed_poses == physical_fixed_neighbors,
                "Vessel and hard-free guide physical neighbors differ"
            );
            let mut source = vessel.clone();
            source.capture_center = reference_capture_center;
            source.capture_radius = reference_capture_radius;
            PhysicalGuideLaw::HardFreeLine(ContactLineGuide::from_bytes(
                guide_raw,
                &region_sha256,
                &chart,
                0.,
                &source,
                tree,
            )?)
        } else {
            PhysicalGuideLaw::Gaussian(ImportanceGuide::from_bytes(guide_raw, &region_sha256)?)
        };
        Ok(Self {
            chart,
            guide,
            region_sha256,
            guide_sha256: hash_bytes(guide_raw),
            shape_sha256: expected_shape_sha256.to_owned(),
            physical_fixed_neighbors,
            reference_capture_center,
            reference_capture_radius,
            log_volume: 3. * PI.ln() - 6_f64.ln() + 6. * Self::REFERENCE_RADIUS.ln(),
        })
    }

    pub fn region_sha256(&self) -> &str {
        &self.region_sha256
    }
    pub fn guide_sha256(&self) -> &str {
        &self.guide_sha256
    }
    pub fn shape_sha256(&self) -> &str {
        &self.shape_sha256
    }
    pub fn uniform_probability(&self) -> f64 {
        self.base().alpha
    }
    pub fn gaussian_component_count(&self) -> usize {
        self.base().components.len()
    }
    fn base(&self) -> &ImportanceGuide {
        match &self.guide {
            PhysicalGuideLaw::Gaussian(g) => g,
            PhysicalGuideLaw::HardFreeLine(g) => &g.base,
        }
    }
    pub fn is_hard_free_line(&self) -> bool {
        matches!(self.guide, PhysicalGuideLaw::HardFreeLine(_))
    }
    pub fn physical_fixed_neighbors(&self) -> &[Pose] {
        &self.physical_fixed_neighbors
    }
    pub fn reference_capture(&self) -> (Vec3, f64) {
        (self.reference_capture_center, self.reference_capture_radius)
    }

    /// The enclosing vessel capture radius may differ from the guide archive.
    /// Physical shape and every fixed body must retain their reviewed identity.
    pub fn validate_physical_context(&self, shape_sha256: &str, fixed: &[Pose]) -> Result<()> {
        ensure!(
            shape_sha256 == self.shape_sha256,
            "Vessel and latent guide shapes differ"
        );
        ensure!(
            fixed == self.physical_fixed_neighbors,
            "Vessel and latent guide physical neighbors differ"
        );
        Ok(())
    }

    fn density_at_latent(
        &self,
        latent: [f64; 6],
        log_jacobian: f64,
    ) -> Result<PhysicalGuideDensity> {
        ensure!(
            latent.iter().all(|v| v.is_finite()) && log_jacobian.is_finite(),
            "Unrepresentable physical guide coordinates/Jacobian"
        );
        // Avoid squared-norm overflow for exterior coordinates.
        let inside = latent.iter().all(|v| v.abs() <= Self::REFERENCE_RADIUS)
            && latent.iter().map(|v| v * v).sum::<f64>() <= Self::REFERENCE_RADIUS.powi(2);
        let (log_q, hard_free_line_density, structural_zero) = match &self.guide {
            PhysicalGuideLaw::Gaussian(g) => (
                g.log_density(latent, inside, self.log_volume),
                None,
                g.alpha == 1. && !inside,
            ),
            PhysicalGuideLaw::HardFreeLine(g) => {
                let (q, detail, zero) = g.density_compact_checked(
                    latent,
                    inside,
                    self.log_volume,
                    &self.chart,
                    Self::REFERENCE_RADIUS,
                )?;
                (q, Some(detail), zero)
            }
        };
        // A finite non-seam point has positive Gaussian density. If its log
        // density is beyond FP64 range, stop rather than declare it a zero.
        let legitimate_zero = structural_zero;
        ensure!(
            log_q.is_finite() || (legitimate_zero && log_q == f64::NEG_INFINITY),
            "Unrepresentable non-seam Gaussian guide log density; no finite band is censored"
        );
        let log_physical = log_q - log_jacobian;
        ensure!(
            log_physical.is_finite() || (legitimate_zero && log_physical == f64::NEG_INFINITY),
            "Unrepresentable non-seam physical guide log density"
        );
        Ok(PhysicalGuideDensity {
            latent: Some(latent),
            in_reference_ball: inside,
            log_latent_density: Some(log_q),
            log_physical_jacobian: Some(log_jacobian),
            log_physical_density: log_physical,
            hard_free_line_density,
            structural_zero,
        })
    }

    /// Decode any finite latent point, with no R4/capture/wall rejection.
    pub fn decode(&self, latent: [f64; 6]) -> Result<(Pose, f64)> {
        ensure!(
            latent.iter().all(|v| v.is_finite()),
            "Nonfinite latent guide coordinate"
        );
        let (pose, log_jacobian) = self.chart.decode(latent);
        pose.validate()?;
        ensure!(
            log_jacobian.is_finite(),
            "Unrepresentable decoded physical Jacobian"
        );
        Ok((pose, log_jacobian))
    }

    /// Evaluate the complete proposal at an arbitrary WORLD pose, including
    /// outside the source reference ball and source capture sphere.
    pub fn evaluate(&self, pose: Pose) -> Result<PhysicalGuideDensity> {
        pose.validate()?;
        let relative = matmul(
            transpose(rotation(self.chart.fixed.orientation)),
            rotation(pose.orientation),
        );
        let quaternion = quaternion(matmul(relative, transpose(self.chart.anchor_rotation)));
        if quaternion[0] == 0. {
            return Ok(PhysicalGuideDensity {
                latent: None,
                in_reference_ball: false,
                log_latent_density: None,
                log_physical_jacobian: None,
                log_physical_density: f64::NEG_INFINITY,
                hard_free_line_density: None,
                structural_zero: true,
            });
        }
        let latent = self.chart.encode(pose)?;
        // 1 + |c|² = 1 / w² for the normalized relative quaternion. This form
        // remains stable near the seam without squaring enormous Cayley values.
        let log_jacobian = self.chart.log_det - 3. * self.chart.ell.ln() - 2. * PI.ln()
            + 4. * quaternion[0].abs().ln();
        self.density_at_latent(latent, log_jacobian)
    }

    pub fn log_density(&self, pose: Pose) -> Result<f64> {
        Ok(self.evaluate(pose)?.log_physical_density)
    }

    /// One draw from the original complete latent mixture. No physical checks,
    /// retries, or normalization by the probability of entering any domain.
    pub fn draw(&self, rng: &mut StdRng) -> Result<PhysicalGuideDraw> {
        let value = self.draw_only(rng)?;
        let (_, log_jacobian) = self.decode(value.latent)?;
        let density = self.density_at_latent(value.latent, log_jacobian)?;
        Ok(PhysicalGuideDraw {
            pose: value.pose,
            latent: value.latent,
            latent_radius: value.latent_radius,
            gaussian_component: value.gaussian_component,
            density,
        })
    }

    pub fn draw_only(&self, rng: &mut StdRng) -> Result<PhysicalGuidePoseDraw> {
        let (latent, latent_radius, gaussian_component, trace) = match &self.guide {
            PhysicalGuideLaw::Gaussian(g) => {
                let (u, r, c) = g.draw(rng, Self::REFERENCE_RADIUS, 0., 1.)?;
                (u, r, c, None)
            }
            PhysicalGuideLaw::HardFreeLine(g) => {
                let (u, r, c, _) = g.draw(rng, &self.chart, Self::REFERENCE_RADIUS, 0., 1.)?;
                (u, r, c, Some(g.last_draw.borrow().clone()))
            }
        };
        let (pose, _) = self.decode(latent)?;
        Ok(PhysicalGuidePoseDraw {
            pose,
            latent,
            latent_radius,
            gaussian_component,
            hard_free_line_draw: trace,
        })
    }
}

/// Exact outer 50/50 density algebra. The caller must evaluate both complete
/// component laws at the SAME world pose; a branch label is never a density.
/// In particular, `log_vessel` must include reciprocal branches, every selected
/// anchor, and the original cube/Haar defensive floor. The density is zero
/// (`-inf` in logs) if both laws have zero support at an arbitrary query pose.
/// A generated-row weight caller must separately require a positive density.
pub fn half_mixture_log_density(log_vessel: f64, log_latent_physical: f64) -> Result<f64> {
    ensure!(
        [log_vessel, log_latent_physical]
            .iter()
            .all(|v| v.is_finite() || *v == f64::NEG_INFINITY),
        "Invalid outer-mixture component density"
    );
    let result = log_add(log_vessel, log_latent_physical) - 2_f64.ln();
    ensure!(
        result.is_finite() || result == f64::NEG_INFINITY,
        "Unrepresentable outer-mixture density"
    );
    Ok(result)
}

#[cfg(test)]
#[path = "physical_guide_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "physical_hard_free_tests.rs"]
mod hard_free_tests;
