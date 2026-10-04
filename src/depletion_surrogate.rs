//! A fixed quadrature surrogate for the two-mobile-body depletion potential.
//!
//! This is NOT a physical energy or acceptance estimator. With fixed spectators
//! S and identical exclusion bodies E0,E1, the score estimates
//! z (|E0 intersect E1| + |S intersect (E0 union E1)|).
//! An arbitrary frozen cloud defines a deterministic surrogate; correcting a
//! reversible inner chain requires S(old)-S(new) at the outer physical gate.
//! Never refresh the cloud conditional on the current moving poses.
use crate::{
    geometry::{Placed, SphereTree},
    math::{Pose, Vec3, norm, sub},
};
use anyhow::{Result, ensure};
use serde::Serialize;

#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct SurrogateScore {
    pub points_per_body: usize,
    pub spectator_covered: [usize; 2],
    pub internal_unshielded: [usize; 2],
    pub nearby_spectators: [usize; 2],
    pub spectator_membership_queries: usize,
    pub internal_membership_queries: usize,
    pub twice_overlap_units: usize,
    pub overlap_volume_estimate: f64,
    pub log_surrogate: f64,
}

pub struct DimerDepletionSurrogate<'a> {
    exclusion: &'a SphereTree,
    points: Vec<Vec3>,
    spectators: Vec<Placed>,
    point_volume: f64,
    activity: f64,
}

fn validate_pose(p: Pose, bound: f64) -> Result<()> {
    p.validate()?;
    ensure!(
        (8. * (1. + norm(p.position) + bound)).powi(2).is_finite(),
        "unsupported surrogate coordinate magnitude"
    );
    Ok(())
}

impl<'a> DimerDepletionSurrogate<'a> {
    /// `exclusion` is already inflated. `points` are retained body-frame points
    /// from a fixed raw box quadrature, with weight box_volume/raw_count. The
    /// caller supplies ONLY the fixed spectators, excluding both mobile labels.
    /// Empty retained clouds are allowed. Their score is identically zero.
    pub fn new(
        exclusion: &'a SphereTree,
        points: Vec<Vec3>,
        spectators: &[Pose],
        point_volume: f64,
        activity: f64,
    ) -> Result<Self> {
        ensure!(
            point_volume.is_finite() && point_volume > 0.,
            "invalid point volume"
        );
        ensure!(activity.is_finite() && activity >= 0., "invalid activity");
        ensure!(
            points.len() <= (1usize << 50),
            "cloud counter range exceeded"
        );
        ensure!(
            (2. * point_volume * points.len() as f64 * activity).is_finite(),
            "surrogate score magnitude overflow"
        );
        for &point in &points {
            ensure!(
                point.iter().all(|x| x.is_finite()) && exclusion.contains(point, 0.),
                "surrogate point outside inflated body"
            );
        }
        for &pose in spectators {
            validate_pose(pose, exclusion.bound)?;
        }
        Ok(Self {
            exclusion,
            points,
            spectators: spectators.iter().copied().map(Placed::new).collect(),
            point_volume,
            activity,
        })
    }

    /// Conservative center pruning only changes work. `score_unpruned` is the
    /// independent enumeration control for this optimization, not a new model.
    pub fn score(&self, members: [Pose; 2]) -> Result<SurrogateScore> {
        self.evaluate(members, true)
    }
    pub fn score_unpruned(&self, members: [Pose; 2]) -> Result<SurrogateScore> {
        self.evaluate(members, false)
    }
    fn evaluate(&self, members: [Pose; 2], prune: bool) -> Result<SurrogateScore> {
        for pose in members {
            validate_pose(pose, self.exclusion.bound)?;
        }
        let placed = members.map(Placed::new);
        let mut result = SurrogateScore {
            points_per_body: self.points.len(),
            ..Default::default()
        };
        for i in 0..2 {
            let nearby: Vec<_> = self
                .spectators
                .iter()
                .filter(|p| {
                    let guard = 4096.
                        * f64::EPSILON
                        * (1.
                            + norm(p.position)
                            + norm(members[i].position)
                            + self.exclusion.bound);
                    !prune
                        || norm(sub(p.position, members[i].position))
                            <= 2. * self.exclusion.bound + guard
                })
                .collect();
            result.nearby_spectators[i] = nearby.len();
            for &point in &self.points {
                let world = placed[i].apply(point);
                let shielded = nearby.iter().any(|spectator| {
                    result.spectator_membership_queries += 1;
                    self.exclusion.contains(spectator.unapply(world), 0.)
                });
                if shielded {
                    result.spectator_covered[i] += 1;
                } else {
                    result.internal_membership_queries += 1;
                    if self.exclusion.contains(placed[1 - i].unapply(world), 0.) {
                        result.internal_unshielded[i] += 1;
                    }
                }
            }
        }
        result.twice_overlap_units = 2 * result.spectator_covered.iter().sum::<usize>()
            + result.internal_unshielded.iter().sum::<usize>();
        result.overlap_volume_estimate =
            0.5 * self.point_volume * result.twice_overlap_units as f64;
        result.log_surrogate = self.activity * result.overlap_volume_estimate;
        ensure!(
            result.log_surrogate.is_finite(),
            "nonfinite surrogate score"
        );
        Ok(result)
    }
}
