//! Ephemeral, root-body overlap guidance for a fixed-cloud pose block.
//!
//! The caller supplies a cloud drawn independently of the current internal pose
//! (a fixed cloud is also allowed). Never redraw it to obtain favorable overlap.
//! With K overlapping points, the maximum of m uniform integers in 0..=K has
//! probability ((k+1)^m-k^m)/(K+1)^m. Hold this threshold and cloud fixed across
//! all capped retries. A successful endpoint needs the SEPARATE auxiliary log
//! correction m[log(K_old+1)-log(K_new+1)] as well as full F and the physical
//! bath. No cloud generation, persistent state or physical acceptance is here.
use crate::{
    geometry::{Placed, SphereTree},
    math::{Pose, Vec3, norm},
};
use anyhow::{Context, Result, ensure};
use rand::{
    distr::{Distribution, Uniform},
    rngs::StdRng,
};
use serde::Serialize;

#[derive(Clone, Debug, Default, Serialize, PartialEq, Eq)]
pub struct OverlapCountFrame {
    pub relative_count: Option<usize>,
    pub recovered_count: Option<usize>,
    pub world_count: Option<usize>,
    pub reconstructed_world_count: Option<usize>,
}

/// Partial diagnostics survive a fatal frame/count/density error. None means
/// that quantity was not calculated, not a zero count or a zero correction.
#[derive(Clone, Debug, Serialize)]
pub struct OverlapGuidanceDiagnostics {
    pub point_count: usize,
    pub m: usize,
    pub old_count: Option<usize>,
    pub threshold: Option<usize>,
    pub integer_draws: Vec<usize>,
    pub new_count: Option<usize>,
    pub aux_log_correction: Option<f64>,
    pub count_queries: usize,
    pub point_tests: usize,
}

pub struct AuxiliaryOverlapThreshold<'a> {
    exclusion: &'a SphereTree,
    points: Vec<Vec3>,
    m: usize,
}

impl<'a> AuxiliaryOverlapThreshold<'a> {
    /// The tree is ALREADY inflated; no further radius is added. Points on its
    /// boundary are included, matching SphereTree::contains's <= convention.
    /// Repeated points are retained as distinct count observations.
    pub fn new(exclusion: &'a SphereTree, points: Vec<Vec3>, m: usize) -> Result<Self> {
        // Exact integer-to-f64 conversion for the log correction. Real Vec
        // capacities impose a much smaller bound on ordinary machines.
        const MAX_EXACT: u64 = (1u64 << 53) - 1;
        ensure!(
            m >= 1 && (m as u128) <= MAX_EXACT as u128,
            "invalid threshold multiplicity"
        );
        ensure!(
            (points.len() as u128) <= MAX_EXACT as u128,
            "too many guidance points"
        );
        ensure!(
            (8. * (1. + exclusion.bound)).powi(2).is_finite(),
            "unsupported guidance shape magnitude"
        );
        for (index, &point) in points.iter().enumerate() {
            ensure!(
                point.iter().all(|v| v.is_finite()),
                "nonfinite guidance point {index}"
            );
            ensure!(
                (8. * (1. + norm(point) + exclusion.bound))
                    .powi(2)
                    .is_finite(),
                "unsupported guidance point magnitude {index}"
            );
            ensure!(
                exclusion.contains(point, 0.),
                "guidance point {index} is outside inflated root"
            );
        }
        Ok(Self {
            exclusion,
            points,
            m,
        })
    }

    pub fn points(&self) -> &[Vec3] {
        &self.points
    }
    pub fn m(&self) -> usize {
        self.m
    }

    pub(crate) fn validate_shape(&self, tree: &SphereTree) -> Result<()> {
        ensure!(
            self.exclusion.shape.atoms.len() == tree.shape.atoms.len()
                && self
                    .exclusion
                    .shape
                    .atoms
                    .iter()
                    .zip(&tree.shape.atoms)
                    .all(|(a, b)| a.center == b.center && a.radius == b.radius),
            "guidance exclusion shape differs from fixed dimer context"
        );
        Ok(())
    }

    pub(crate) fn diagnostics(&self) -> OverlapGuidanceDiagnostics {
        OverlapGuidanceDiagnostics {
            point_count: self.points.len(),
            m: self.m,
            old_count: None,
            threshold: None,
            integer_draws: vec![],
            new_count: None,
            aux_log_correction: None,
            count_queries: 0,
            point_tests: 0,
        }
    }

    fn check_pose(&self, pose: Pose) -> Result<()> {
        pose.validate()?;
        ensure!(
            (8. * (1. + norm(pose.position) + self.exclusion.bound))
                .powi(2)
                .is_finite(),
            "unsupported guidance transform magnitude"
        );
        Ok(())
    }

    /// Boolean child-exclusion membership of each fixed root-body point.
    pub fn count_relative(&self, child_relative: Pose) -> Result<usize> {
        self.check_pose(child_relative)?;
        let child = Placed::new(child_relative);
        Ok(self
            .points
            .iter()
            .filter(|&&point| self.exclusion.contains(child.unapply(point), 0.))
            .count())
    }

    /// Independently transform root points to world and then to child body.
    pub fn count_world(&self, members: [Pose; 2]) -> Result<usize> {
        for pose in members {
            self.check_pose(pose)?;
        }
        let [root, child] = members.map(Placed::new);
        Ok(self
            .points
            .iter()
            .filter(|&&point| {
                self.exclusion
                    .contains(child.unapply(root.apply(point)), 0.)
            })
            .count())
    }

    fn record_query(&self, d: &mut OverlapGuidanceDiagnostics) -> Result<()> {
        d.count_queries = d
            .count_queries
            .checked_add(1)
            .context("guidance query counter overflow")?;
        d.point_tests = d
            .point_tests
            .checked_add(self.points.len())
            .context("guidance point counter overflow")?;
        Ok(())
    }

    pub(crate) fn recorded_relative(
        &self,
        pose: Pose,
        d: &mut OverlapGuidanceDiagnostics,
    ) -> Result<usize> {
        let count = self.count_relative(pose)?;
        self.record_query(d)?;
        Ok(count)
    }

    /// Write each completed count before checking equality, preserving evidence
    /// on failure. This is a strict predicate check, never a tolerance or retry.
    pub(crate) fn check_frame(
        &self,
        relative: Pose,
        recovered: Pose,
        members: [Pose; 2],
        reconstructed: [Pose; 2],
        d: &mut OverlapGuidanceDiagnostics,
        out: &mut OverlapCountFrame,
    ) -> Result<()> {
        out.relative_count = Some(self.recorded_relative(relative, d)?);
        out.recovered_count = Some(self.recorded_relative(recovered, d)?);
        out.world_count = Some(self.count_world(members)?);
        self.record_query(d)?;
        out.reconstructed_world_count = Some(self.count_world(reconstructed)?);
        self.record_query(d)?;
        ensure!(
            out.relative_count == out.recovered_count
                && out.relative_count == out.world_count
                && out.relative_count == out.reconstructed_world_count,
            "fatal guidance count frame mismatch"
        );
        Ok(())
    }

    pub(crate) fn draw_threshold(
        &self,
        rng: &mut StdRng,
        d: &mut OverlapGuidanceDiagnostics,
    ) -> Result<()> {
        let old = d.old_count.context("missing old guidance count")?;
        ensure!(
            old <= self.points.len() && d.integer_draws.is_empty() && d.threshold.is_none(),
            "invalid/repeated guidance threshold refresh"
        );
        d.integer_draws
            .try_reserve_exact(self.m)
            .context("threshold draw allocation failed")?;
        // Prepared Uniform uses unbiased rejection sampling in rand 0.10.
        // random_range's single-sample implementation need not be unbiased.
        let uniform = Uniform::new_inclusive(0usize, old)?;
        for _ in 0..self.m {
            d.integer_draws.push(uniform.sample(rng));
        }
        d.threshold = d.integer_draws.iter().copied().max();
        Ok(())
    }

    pub fn log_correction(&self, old_count: usize, new_count: usize) -> Result<f64> {
        ensure!(
            old_count <= self.points.len() && new_count <= self.points.len(),
            "guidance count exceeds cloud size"
        );
        let result = (self.m as f64) * ((old_count as f64).ln_1p() - (new_count as f64).ln_1p());
        ensure!(result.is_finite(), "nonfinite guidance correction");
        Ok(result)
    }
}
