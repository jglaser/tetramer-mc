//! Reversible contact-conditioned selection of spherical GCA half-turns.
//!
//! The fixed-axis physical kernel is unchanged. Two independent capped draws
//! from g_X(u) h_X(u) / Z_X supply an auxiliary-exchange correction; both unknown
//! normalizers and both cap-success probabilities cancel. A failed search is a
//! retained self-loop. See docs/conditional-half-turn.md for the full argument.
use crate::{
    geometry::{Placed, SphereTree},
    math::*,
    spherical::{self, Container, GcaStats, HalfTurn},
};
use anyhow::{Result, ensure};
use rand::{RngExt, rngs::StdRng};
use serde::{Deserialize, Serialize};
use std::{f64::consts::PI, time::Instant};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ConditionalAxisConfig {
    pub score_floor: f64,
    pub max_candidates: usize,
    pub uniform_axis_weight: f64,
}
impl Default for ConditionalAxisConfig {
    fn default() -> Self {
        Self {
            score_floor: 0.02,
            max_candidates: 64,
            uniform_axis_weight: 0.25,
        }
    }
}
impl ConditionalAxisConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.score_floor.is_finite() && self.score_floor > 0. && self.score_floor <= 1.,
            "conditional-axis score_floor must be in (0,1]"
        );
        ensure!(
            self.max_candidates > 0,
            "conditional-axis max_candidates must be positive"
        );
        ensure!(
            self.uniform_axis_weight.is_finite()
                && self.uniform_axis_weight > 0.
                && self.uniform_axis_weight <= 1.,
            "conditional-axis uniform_axis_weight must be in (0,1]"
        );
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct Band {
    direction: Vec3,
    lower: f64,
    area: f64,
}

/// Normalized defensive spherical-band density, relative to uniform unoriented
/// axis probability. Both signs of an axis have the same density. Overlapping
/// endpoint bands are a mixture, so all covering bands enter the density.
#[derive(Clone, Debug)]
pub struct AxisBase {
    tag_direction: Option<Vec3>,
    bands: Vec<Band>,
    area: f64,
    uniform_weight: f64,
}
impl AxisBase {
    fn new(state: &[Pose], tag: usize, distance: f64, uniform_weight: f64) -> Result<Self> {
        validate_poses(state, tag)?;
        ensure!(
            distance.is_finite() && distance > 0.,
            "invalid axis-guide contact bound"
        );
        let radius = norm(state[tag].position);
        ensure!(radius.is_finite(), "nonfinite tag radius");
        let mut base = Self {
            tag_direction: None,
            bands: Vec::new(),
            area: 0.,
            uniform_weight,
        };
        if radius == 0. || uniform_weight == 1. {
            return Ok(base);
        }
        base.tag_direction = Some(state[tag].position.map(|v| v / radius));
        for (j, pose) in state.iter().enumerate() {
            if j == tag {
                continue;
            }
            let neighbor_radius = norm(pose.position);
            ensure!(neighbor_radius.is_finite(), "nonfinite neighboring radius");
            if neighbor_radius == 0. {
                if radius > distance {
                    continue;
                }
            } else if (radius - neighbor_radius).abs() >= distance {
                continue;
            }
            let direction = if neighbor_radius == 0. {
                [0., 0., 1.]
            } else {
                pose.position.map(|v| v / neighbor_radius)
            };
            let lower = if radius + neighbor_radius <= distance || neighbor_radius == 0. {
                -1.
            } else {
                // Only the partially covered case reaches this expression.
                // Scaling prevents overflow in squared radii and products.
                let unit = radius.max(neighbor_radius).max(distance);
                let a = radius / unit;
                let b = neighbor_radius / unit;
                let d = distance / unit;
                let lower = ((a - b).powi(2) - d * d) / (2. * a * b) + 1.;
                ensure!(lower.is_finite(), "unresolved spherical-band arithmetic");
                lower.clamp(-1., 1.)
            };
            let area = 2. * PI * (1. - lower);
            if area > 0. {
                base.bands.push(Band {
                    direction,
                    lower,
                    area,
                });
                base.area += area;
            }
        }
        ensure!(base.area.is_finite(), "nonfinite total band area");
        Ok(base)
    }
    pub fn band_area(&self) -> f64 {
        self.area
    }
    pub fn is_uniform(&self) -> bool {
        self.tag_direction.is_none() || self.area == 0. || self.uniform_weight == 1.
    }
    /// Number of endpoint bands covering the center reached by this axis.
    pub fn multiplicity(&self, axis: Vec3) -> Result<usize> {
        let axis = unit_axis(axis)?;
        let Some(r) = self.tag_direction else {
            return Ok(0);
        };
        let v = sub(scale(axis, 2. * dot(axis, r)), r);
        Ok(self
            .bands
            .iter()
            .filter(|b| dot(v, b.direction) >= b.lower)
            .count())
    }
    pub fn density(&self, axis: Vec3) -> Result<f64> {
        let axis = unit_axis(axis)?;
        if self.is_uniform() {
            return Ok(1.);
        }
        let r = self.tag_direction.unwrap();
        let multiplicity = self.multiplicity(axis)? as f64;
        let density = self.uniform_weight
            + (1. - self.uniform_weight) * 8. * PI * dot(axis, r).abs() * multiplicity / self.area;
        ensure!(
            density.is_finite() && density > 0.,
            "invalid axis-guide density"
        );
        Ok(density)
    }
    pub fn draw(&self, rng: &mut StdRng) -> Result<Vec3> {
        if self.is_uniform() || rng.random::<f64>() < self.uniform_weight {
            return Ok(uniform_axis(rng));
        }
        let target = rng.random::<f64>() * self.area;
        let mut cumulative = 0.;
        let band = self
            .bands
            .iter()
            .find(|b| {
                cumulative += b.area;
                target < cumulative
            })
            .unwrap_or_else(|| self.bands.last().unwrap());
        let z = band.lower + (1. - band.lower) * rng.random::<f64>();
        let azimuth = 2. * PI * rng.random::<f64>();
        let transverse = (1. - z * z).max(0.).sqrt();
        let reference = if band.direction[0].abs() < 0.8 {
            [1., 0., 0.]
        } else {
            [0., 1., 0.]
        };
        let first = unit_axis(cross(band.direction, reference))?;
        let second = cross(band.direction, first);
        let endpoint = add(
            scale(band.direction, z),
            add(
                scale(first, transverse * azimuth.cos()),
                scale(second, transverse * azimuth.sin()),
            ),
        );
        // An exactly antipodal finite-precision draw is an error, never a silent
        // redraw from a changed law. In exact arithmetic it has measure zero.
        unit_axis(add(self.tag_direction.unwrap(), endpoint))
    }
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn unit_axis(axis: Vec3) -> Result<Vec3> {
    let length = norm(axis);
    ensure!(
        axis.iter().all(|x| x.is_finite()) && length.is_finite() && length > 0.,
        "invalid or unresolved antipodal conditional axis"
    );
    Ok(axis.map(|v| v / length))
}
fn uniform_axis(rng: &mut StdRng) -> Vec3 {
    let z = 2. * rng.random::<f64>() - 1.;
    let phi = 2. * PI * rng.random::<f64>();
    let r = (1. - z * z).max(0.).sqrt();
    [r * phi.cos(), r * phi.sin(), z]
}
fn validate_poses(state: &[Pose], tag: usize) -> Result<()> {
    ensure!(
        tag < state.len(),
        "conditional-axis tag out of range or empty state"
    );
    for pose in state {
        pose.validate()?;
    }
    Ok(())
}
fn difference(a: &[usize], b: &[usize]) -> Vec<usize> {
    a.iter()
        .copied()
        .filter(|i| b.binary_search(i).is_err())
        .collect()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContactScore {
    pub value: f64,
    pub predicate: bool,
    /// False when the score is known without trial geometry (floor=1 or no old contacts).
    pub geometry_evaluated: bool,
    pub hard_valid: Option<bool>,
    pub lost: Vec<usize>,
    pub gained: Vec<usize>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AxisSearchStats {
    pub attempts: usize,
    pub predicate_candidates: usize,
    /// Trial geometry evaluated during this capped search only. Context/contact
    /// construction and the two cross-scores are outside this counter.
    pub geometry_evaluations: usize,
    pub selected_axis: Option<Vec3>,
    pub selected_score: Option<f64>,
    pub selected_base_density: Option<f64>,
    pub capped_failure: bool,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ConditionalAxisStats {
    pub tag: usize,
    pub forward: AxisSearchStats,
    pub reverse: AxisSearchStats,
    pub cross_score_y_u: Option<f64>,
    pub cross_score_x_v: Option<f64>,
    pub cross_base_y_u: Option<f64>,
    pub cross_base_x_v: Option<f64>,
    pub log_selector_ratio: Option<f64>,
    pub log_acceptance: Option<f64>,
    pub failure_reason: Option<String>,
    pub proposed_tag_flipped: bool,
    pub proposed_lost: Vec<usize>,
    pub proposed_gained: Vec<usize>,
    pub accepted_lost: Vec<usize>,
    pub accepted_gained: Vec<usize>,
    pub initial_contacts: Vec<usize>,
    pub proposed_contacts: Vec<usize>,
    pub accepted_contacts: Vec<usize>,
    pub selection_seconds: f64,
    pub total_seconds: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConditionalAxisOutcome {
    pub accepted: bool,
    pub gca: Option<GcaStats>,
    pub stats: ConditionalAxisStats,
}
struct Context<'a> {
    state: &'a [Pose],
    placed: Vec<Placed>,
    tag: usize,
    contacts: Vec<usize>,
    base: AxisBase,
}

pub struct ConditionalAxis<'a> {
    tree: &'a SphereTree,
    contact_tree: SphereTree,
    rd: f64,
    config: ConditionalAxisConfig,
}
impl<'a> ConditionalAxis<'a> {
    pub fn new(tree: &'a SphereTree, rd: f64, config: ConditionalAxisConfig) -> Result<Self> {
        config.validate()?;
        ensure!(
            rd.is_finite() && rd >= 0.,
            "invalid conditional-axis depletant radius"
        );
        let mut contact_shape = tree.shape.clone();
        for atom in &mut contact_shape.atoms {
            atom.radius += rd;
        }
        let contact_tree = SphereTree::new(contact_shape)?;
        ensure!(
            (2. * (tree.bound + rd)).is_finite(),
            "nonfinite exclusion contact bound"
        );
        Ok(Self {
            tree,
            contact_tree,
            rd,
            config,
        })
    }
    pub fn base(&self, state: &[Pose], tag: usize) -> Result<AxisBase> {
        AxisBase::new(
            state,
            tag,
            2. * (self.tree.bound + self.rd),
            self.config.uniform_axis_weight,
        )
    }
    pub fn contacts(&self, state: &[Pose], tag: usize) -> Result<Vec<usize>> {
        validate_poses(state, tag)?;
        let placed: Vec<_> = state.iter().copied().map(Placed::new).collect();
        Ok(self.placed_contacts(&placed, tag, &placed[tag]))
    }
    fn placed_contacts(&self, placed: &[Placed], tag: usize, trial: &Placed) -> Vec<usize> {
        placed
            .iter()
            .enumerate()
            .filter_map(|(j, other)| {
                (j != tag && self.contact_tree.overlaps(trial, other)).then_some(j)
            })
            .collect()
    }
    fn context<'s>(&self, state: &'s [Pose], tag: usize) -> Result<Context<'s>> {
        let base = self.base(state, tag)?;
        let placed: Vec<_> = state.iter().copied().map(Placed::new).collect();
        let contacts = self.placed_contacts(&placed, tag, &placed[tag]);
        Ok(Context {
            state,
            placed,
            tag,
            contacts,
            base,
        })
    }
    pub fn score(&self, state: &[Pose], tag: usize, axis: Vec3) -> Result<ContactScore> {
        self.score_context(&self.context(state, tag)?, axis)
    }
    fn score_context(&self, context: &Context<'_>, axis: Vec3) -> Result<ContactScore> {
        let mut result = ContactScore {
            value: self.config.score_floor,
            predicate: false,
            geometry_evaluated: false,
            hard_valid: None,
            lost: Vec::new(),
            gained: Vec::new(),
        };
        if self.config.score_floor == 1. || context.contacts.is_empty() {
            return Ok(result);
        }
        let trial = Placed::new(HalfTurn::new(axis)?.apply(context.state[context.tag]));
        result.geometry_evaluated = true;
        let valid = context
            .placed
            .iter()
            .enumerate()
            .all(|(j, other)| j == context.tag || !self.tree.overlaps(&trial, other));
        result.hard_valid = Some(valid);
        if !valid {
            return Ok(result);
        }
        let contacts = self.placed_contacts(&context.placed, context.tag, &trial);
        result.lost = difference(&context.contacts, &contacts);
        result.gained = difference(&contacts, &context.contacts);
        result.predicate = !result.lost.is_empty() && !result.gained.is_empty();
        if result.predicate {
            result.value = 1.;
        }
        Ok(result)
    }
    fn search(&self, context: &Context<'_>, rng: &mut StdRng) -> Result<AxisSearchStats> {
        let mut stats = AxisSearchStats::default();
        for _ in 0..self.config.max_candidates {
            stats.attempts += 1;
            let axis = context.base.draw(rng)?;
            let score = self.score_context(context, axis)?;
            stats.predicate_candidates += usize::from(score.predicate);
            stats.geometry_evaluations += usize::from(score.geometry_evaluated);
            if rng.random::<f64>() < score.value {
                stats.selected_axis = Some(axis);
                stats.selected_score = Some(score.value);
                stats.selected_base_density = Some(context.base.density(axis)?);
                return Ok(stats);
            }
        }
        stats.capped_failure = true;
        Ok(stats)
    }
    pub fn update(
        &self,
        wall: &Container,
        state: &mut [Pose],
        activity: f64,
        guide_rng: &mut StdRng,
        physical_rng: &mut StdRng,
    ) -> Result<ConditionalAxisOutcome> {
        ensure!(!state.is_empty(), "empty conditional-axis state");
        let tag = guide_rng.random_range(0..state.len());
        self.update_with_tag(wall, state, activity, tag, guide_rng, physical_rng)
    }
    /// A retained tag may be selected by the caller using a state-independent law.
    /// Selecting it from contacts would require another forward/reverse correction.
    pub fn update_with_tag(
        &self,
        wall: &Container,
        state: &mut [Pose],
        activity: f64,
        tag: usize,
        guide_rng: &mut StdRng,
        physical_rng: &mut StdRng,
    ) -> Result<ConditionalAxisOutcome> {
        let start = Instant::now();
        ensure!(
            activity.is_finite() && activity >= 0.,
            "invalid conditional-axis activity"
        );
        let old = self.context(state, tag)?;
        let mut stats = ConditionalAxisStats {
            tag,
            initial_contacts: old.contacts.clone(),
            proposed_contacts: old.contacts.clone(),
            accepted_contacts: old.contacts.clone(),
            ..Default::default()
        };
        stats.forward = self.search(&old, guide_rng)?;
        let Some(axis) = stats.forward.selected_axis else {
            stats.failure_reason = Some("forward_search_cap".into());
            stats.total_seconds = start.elapsed().as_secs_f64();
            stats.selection_seconds = stats.total_seconds;
            return Ok(ConditionalAxisOutcome {
                accepted: false,
                gca: None,
                stats,
            });
        };
        let mut proposed = state.to_vec();
        let before_physical = Instant::now();
        let gca = spherical::update(
            self.tree,
            wall,
            &mut proposed,
            HalfTurn::new(axis)?,
            self.rd,
            activity,
            physical_rng,
        )?;
        let physical_seconds = before_physical.elapsed().as_secs_f64();
        let new = self.context(&proposed, tag)?;
        stats.proposed_tag_flipped = gca.flipped_indices.contains(&tag);
        stats.proposed_contacts = new.contacts.clone();
        stats.proposed_lost = difference(&old.contacts, &new.contacts);
        stats.proposed_gained = difference(&new.contacts, &old.contacts);
        stats.reverse = self.search(&new, guide_rng)?;
        let Some(reverse_axis) = stats.reverse.selected_axis else {
            stats.failure_reason = Some("reverse_search_cap".into());
            stats.total_seconds = start.elapsed().as_secs_f64();
            stats.selection_seconds = (stats.total_seconds - physical_seconds).max(0.);
            return Ok(ConditionalAxisOutcome {
                accepted: false,
                gca: Some(gca),
                stats,
            });
        };
        let h_y_u = self.score_context(&new, axis)?.value;
        let h_x_v = self.score_context(&old, reverse_axis)?.value;
        let g_y_u = new.base.density(axis)?;
        let g_x_v = old.base.density(reverse_axis)?;
        stats.cross_score_y_u = Some(h_y_u);
        stats.cross_score_x_v = Some(h_x_v);
        stats.cross_base_y_u = Some(g_y_u);
        stats.cross_base_x_v = Some(g_x_v);
        let ratio = h_y_u.ln() + h_x_v.ln() + g_y_u.ln() + g_x_v.ln()
            - stats.forward.selected_score.unwrap().ln()
            - stats.reverse.selected_score.unwrap().ln()
            - stats.forward.selected_base_density.unwrap().ln()
            - stats.reverse.selected_base_density.unwrap().ln();
        ensure!(ratio.is_finite(), "nonfinite conditional-axis log ratio");
        stats.log_selector_ratio = Some(ratio);
        stats.log_acceptance = Some(ratio.min(0.));
        let accepted = guide_rng.random::<f64>().ln() < ratio.min(0.);
        if accepted {
            stats.accepted_contacts = new.contacts.clone();
            stats.accepted_lost = stats.proposed_lost.clone();
            stats.accepted_gained = stats.proposed_gained.clone();
            // All geometry, physical sampling and selector work precede commit.
            state.copy_from_slice(&proposed);
        } else {
            stats.failure_reason = Some("selector_rejection".into());
        }
        stats.total_seconds = start.elapsed().as_secs_f64();
        stats.selection_seconds = (stats.total_seconds - physical_seconds).max(0.);
        Ok(ConditionalAxisOutcome {
            accepted,
            gca: Some(gca),
            stats,
        })
    }
}
