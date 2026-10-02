//! Fixed-context, capped independent redraw conditioned on a feasible dimer.
//!
//! D is full selected-body hard/wall validity AND internal exclusion contact.
//! Only finite, successfully scored whole-joint draws outside D may be retried.
//! The source must be hard-valid; a source outside contact D is a self-loop.
//! For x,y in D, q_K(x,dy)=F(y)1_D(y) sum_{j<K}(1-p)^j dy. Fixed spectators,
//! anchor, labels, atlas and cap make p context-only, so the existing full F
//! ratio is unchanged. No unknown conditioning normalizer is estimated.
//!
//! Unexpected arithmetic/density errors are FATAL, with every completed raw
//! trial attached. A caller must persist that record and stop, never convert it
//! into another retry or an ordinary MH self-loop. This distinction is needed:
//! a source-dependent computational failure is not covered by the balance proof.
//! No physical bath, acceptance, adaptation or state-selection law lives here.
use crate::{
    defensive_dimer_proposal::{
        DefensiveDimerCandidate, DefensiveDimerOutcome, DefensiveDimerProposal,
    },
    geometry::{Placed, SphereTree},
    math::{Pose, Vec3, norm, sub},
    spherical::Container,
};
use anyhow::{Result, ensure};
use rand::rngs::StdRng;
use serde::Serialize;
use std::fmt;

/// Immutable spectator snapshot. Rebuild after another kernel moves spectators.
/// Their mutual physical validity is the caller's state invariant.
pub struct FixedDimerContext<'a> {
    core: &'a SphereTree,
    exclusion: &'a SphereTree,
    spectators: Vec<(usize, Placed)>,
    members: [usize; 2],
    anchor_label: usize,
    anchor: Pose,
    wall: Option<Container>,
    wall_center: Vec3,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct DimerFeasibility {
    pub internal_core_overlap: bool,
    pub spectator_core_collisions: [Vec<usize>; 2],
    pub wall_valid: [bool; 2],
    pub internal_exclusion_contact: bool,
}
impl DimerFeasibility {
    pub fn hard_valid(&self) -> bool {
        !self.internal_core_overlap
            && self.spectator_core_collisions.iter().all(Vec::is_empty)
            && self.wall_valid.iter().all(|&x| x)
    }
    pub fn feasible(&self) -> bool {
        self.hard_valid() && self.internal_exclusion_contact
    }
}

impl<'a> FixedDimerContext<'a> {
    pub fn new(
        core: &'a SphereTree,
        exclusion: &'a SphereTree,
        state: &[Pose],
        members: [usize; 2],
        anchor_label: usize,
        wall_radius: Option<f64>,
        wall_center: Vec3,
    ) -> Result<Self> {
        ensure!(
            members[0] != members[1] && members.iter().all(|&i| i < state.len()),
            "invalid fixed dimer labels"
        );
        ensure!(
            anchor_label < state.len() && !members.contains(&anchor_label),
            "anchor must be a fixed spectator"
        );
        ensure!(
            wall_center.iter().all(|x| x.is_finite()),
            "invalid wall center"
        );
        // The contact predicate must be an inflation of this same hard shape.
        // This is an auxiliary geometric condition, not a pairwise energy.
        ensure!(
            core.shape.atoms.len() == exclusion.shape.atoms.len()
                && core
                    .shape
                    .atoms
                    .iter()
                    .zip(&exclusion.shape.atoms)
                    .all(|(a, b)| a.center == b.center && b.radius >= a.radius),
            "exclusion tree must enclose the corresponding core atoms"
        );
        for &pose in state {
            Self::check_arithmetic(pose, exclusion.bound)?;
        }
        Ok(Self {
            core,
            exclusion,
            spectators: state
                .iter()
                .enumerate()
                .filter(|(i, _)| !members.contains(i))
                .map(|(i, &p)| (i, Placed::new(p)))
                .collect(),
            members,
            anchor_label,
            anchor: state[anchor_label],
            wall: wall_radius.map(|r| Container::new(r, core)).transpose()?,
            wall_center,
        })
    }
    fn check_arithmetic(pose: Pose, bound: f64) -> Result<()> {
        pose.validate()?;
        ensure!(
            (8. * (1. + norm(pose.position) + bound))
                .powi(2)
                .is_finite(),
            "unsupported geometry arithmetic magnitude"
        );
        Ok(())
    }
    pub fn anchor(&self) -> Pose {
        self.anchor
    }
    pub fn anchor_label(&self) -> usize {
        self.anchor_label
    }
    pub fn members(&self) -> [usize; 2] {
        self.members
    }
    /// Evaluate every recorded hard predicate, even after a collision is found.
    pub fn evaluate(&self, members: [Pose; 2]) -> Result<DimerFeasibility> {
        for p in members {
            Self::check_arithmetic(p, self.exclusion.bound)?;
            if self.wall.is_some() {
                Self::check_arithmetic(
                    Pose {
                        position: sub(p.position, self.wall_center),
                        ..p
                    },
                    self.core.bound,
                )?;
            }
        }
        let placed = members.map(Placed::new);
        let wall_valid = members.map(|p| {
            self.wall.as_ref().is_none_or(|wall| {
                wall.contains(Pose {
                    position: sub(p.position, self.wall_center),
                    ..p
                })
            })
        });
        let spectator_core_collisions = std::array::from_fn(|i| {
            self.spectators
                .iter()
                .filter_map(|(label, p)| self.core.overlaps(&placed[i], p).then_some(*label))
                .collect()
        });
        Ok(DimerFeasibility {
            internal_core_overlap: self.core.overlaps(&placed[0], &placed[1]),
            spectator_core_collisions,
            wall_valid,
            internal_exclusion_contact: self.exclusion.overlaps(&placed[0], &placed[1]),
        })
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CappedDimerTrial {
    /// One-based raw-trial index, including the final successful trial.
    pub index: usize,
    pub draw: Option<DefensiveDimerOutcome>,
    pub feasibility: Option<DimerFeasibility>,
}
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CappedDimerStatus {
    Candidate,
    SourceOutsideContact,
    CapExhausted,
}
#[derive(Clone, Debug, Serialize)]
pub struct CappedDimerOutcome {
    pub status: CappedDimerStatus,
    pub trial_cap: usize,
    pub members: [usize; 2],
    pub anchor_label: usize,
    pub old: [Pose; 2],
    pub source_feasibility: DimerFeasibility,
    pub trials: Vec<CappedDimerTrial>,
    pub candidate: Option<DefensiveDimerCandidate>,
}
/// A failed calculation, NOT an MCMC outcome. The attached raw trials must be
/// saved by the runner before it terminates. Serialization contains no NaNs.
#[derive(Debug, Serialize)]
pub struct CappedDimerFailure {
    pub fatal_error: String,
    pub source_feasibility: Option<DimerFeasibility>,
    pub trials: Vec<CappedDimerTrial>,
}
impl fmt::Display for CappedDimerFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.fatal_error)
    }
}
impl std::error::Error for CappedDimerFailure {}

pub struct CappedDimerProposal<'a> {
    raw: DefensiveDimerProposal<'a>,
    trial_cap: usize,
}
impl<'a> CappedDimerProposal<'a> {
    /// K=0 is the exact identity limit; no raw draw occurs.
    pub fn new(raw: DefensiveDimerProposal<'a>, trial_cap: usize) -> Self {
        Self { raw, trial_cap }
    }
    pub fn propose(
        &self,
        rng: &mut StdRng,
        context: &FixedDimerContext<'_>,
        old: [Pose; 2],
    ) -> std::result::Result<CappedDimerOutcome, CappedDimerFailure> {
        let source = context.evaluate(old).map_err(|e| CappedDimerFailure {
            fatal_error: e.to_string(),
            source_feasibility: None,
            trials: vec![],
        })?;
        if !source.hard_valid() {
            return Err(CappedDimerFailure {
                fatal_error: "source is not hard-valid".into(),
                source_feasibility: Some(source),
                trials: vec![],
            });
        }
        let mut outcome = CappedDimerOutcome {
            status: CappedDimerStatus::CapExhausted,
            trial_cap: self.trial_cap,
            members: context.members(),
            anchor_label: context.anchor_label(),
            old,
            source_feasibility: source.clone(),
            trials: vec![],
            candidate: None,
        };
        if !source.internal_exclusion_contact {
            outcome.status = CappedDimerStatus::SourceOutsideContact;
            return Ok(outcome);
        }
        for index in 1..=self.trial_cap {
            let draw = match self.raw.propose(rng, context.anchor(), old[0], old[1]) {
                Ok(draw) => draw,
                Err(e) => {
                    outcome.trials.push(CappedDimerTrial {
                        index,
                        draw: None,
                        feasibility: None,
                    });
                    return Err(CappedDimerFailure {
                        fatal_error: e.to_string(),
                        source_feasibility: Some(source),
                        trials: outcome.trials,
                    });
                }
            };
            // Nulls are not geometric failures and cannot enter the renewal sum.
            let candidate = match &draw.candidate {
                Some(candidate) => candidate.clone(),
                None => {
                    let error = draw
                        .null_reason
                        .clone()
                        .unwrap_or("unclassified proposal null".into());
                    outcome.trials.push(CappedDimerTrial {
                        index,
                        draw: Some(draw),
                        feasibility: None,
                    });
                    return Err(CappedDimerFailure {
                        fatal_error: error,
                        source_feasibility: Some(source),
                        trials: outcome.trials,
                    });
                }
            };
            let feasibility = match context.evaluate([candidate.root, candidate.child]) {
                Ok(f) => f,
                Err(e) => {
                    outcome.trials.push(CappedDimerTrial {
                        index,
                        draw: Some(draw),
                        feasibility: None,
                    });
                    return Err(CappedDimerFailure {
                        fatal_error: e.to_string(),
                        source_feasibility: Some(source),
                        trials: outcome.trials,
                    });
                }
            };
            let feasible = feasibility.feasible();
            outcome.trials.push(CappedDimerTrial {
                index,
                draw: Some(draw),
                feasibility: Some(feasibility),
            });
            if feasible {
                outcome.status = CappedDimerStatus::Candidate;
                outcome.candidate = Some(candidate);
                return Ok(outcome);
            }
        }
        Ok(outcome)
    }
}
