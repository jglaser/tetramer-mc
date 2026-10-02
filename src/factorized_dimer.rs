//! Independent finite-cap edge filters followed by a complete endpoint check.
//!
//! Fixed spectators, labels, anchor, edge law, caps and stage order define C.
//! Root filtering uses fixed spectators/wall; internal filtering uses identity
//! root-frame hard validity and exclusion contact. Each failed edge redraws the
//! complete mixture. Edge exhaustion consumes a joint attempt and skips the
//! other stage if necessary. Final failure discards BOTH successful edges.
//! Thus the successful subdensity is a(C) s_J(q(C)) F0 F1 1_D, and its common
//! factor cancels. Full F scoring happens only at a geometrically feasible end.
//!
//! This is not a physical kernel. Source/endpoint frame-predicate disagreement,
//! numerical null or scoring failure is FATAL with the partial outcome retained,
//! never another retry or an ordinary MH self-loop. These checks do not prove
//! floating-point reversibility. Rebuild the context if a spectator moves.
use crate::{
    capped_dimer::{
        DimerFeasibility, FixedBodyFeasibility, FixedDimerContext, InternalDimerFeasibility,
    },
    defensive_dimer_proposal::{
        DefensiveDimerCandidate, DefensiveDimerEdge, DefensiveDimerProposal,
    },
    dimer_tree_proposal::{tree_coordinates, tree_members},
    math::Pose,
};
use anyhow::{Context, Result, ensure};
use rand::rngs::StdRng;
use serde::Serialize;
use std::fmt;

const IDENTITY: Pose = Pose {
    position: [0.; 3],
    orientation: [1., 0., 0., 0.],
};

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
pub struct FactorizedDimerCaps {
    pub root: usize,
    pub internal: usize,
    pub joint: usize,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FactorizedDimerOrder {
    RootFirst,
    InternalFirst,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FactorizedDimerStatus {
    InProgress,
    Candidate,
    SourceOutsideDomain,
    CapExhausted,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FactorizedAttemptStatus {
    InProgress,
    Candidate,
    RootCapExhausted,
    InternalCapExhausted,
    FinalRejected,
}

#[derive(Clone, Debug, Serialize)]
pub struct FactorizedRootDraw {
    pub index: usize,
    pub draw: DefensiveDimerEdge,
    pub world_pose: Option<Pose>,
    pub feasibility: Option<FixedBodyFeasibility>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FactorizedInternalDraw {
    pub index: usize,
    pub draw: DefensiveDimerEdge,
    pub feasibility: Option<InternalDimerFeasibility>,
}

/// Retain actual recomposition and every predicate used to detect frame changes.
/// Pose values need not be bitwise equal after a round trip. Predicates must be.
#[derive(Clone, Debug, Serialize)]
pub struct FactorizedFrameCheck {
    pub recovered_edges: [Pose; 2],
    pub reconstructed_members: [Pose; 2],
    pub reconstructed_feasibility: DimerFeasibility,
    pub reconstructed_root: FixedBodyFeasibility,
    pub internal_relative: InternalDimerFeasibility,
}

impl FactorizedFrameCheck {
    fn evaluate(context: &FixedDimerContext<'_>, members: [Pose; 2]) -> Result<Self> {
        let recovered_edges = tree_coordinates(context.anchor(), members[0], members[1])?;
        let reconstructed_members = tree_members(context.anchor(), recovered_edges)?;
        Ok(Self {
            recovered_edges,
            reconstructed_members,
            reconstructed_feasibility: context.evaluate(reconstructed_members)?,
            reconstructed_root: context.evaluate_fixed_body(reconstructed_members[0])?,
            internal_relative: context.evaluate_internal_relative(recovered_edges[1])?,
        })
    }

    fn validate(
        &self,
        world: &DimerFeasibility,
        root: Option<&FixedBodyFeasibility>,
        internal: Option<&InternalDimerFeasibility>,
    ) -> Result<()> {
        ensure!(
            self.reconstructed_feasibility == *world,
            "fatal frame predicate mismatch: endpoint round trip"
        );
        let world_root = FixedBodyFeasibility {
            spectator_core_collisions: world.spectator_core_collisions[0].clone(),
            wall_valid: world.wall_valid[0],
        };
        let world_internal = InternalDimerFeasibility {
            internal_core_overlap: world.internal_core_overlap,
            internal_exclusion_contact: world.internal_exclusion_contact,
        };
        ensure!(
            self.reconstructed_root == world_root,
            "fatal frame predicate mismatch: recovered root"
        );
        ensure!(
            self.internal_relative == world_internal,
            "fatal frame predicate mismatch: recovered internal edge"
        );
        if let Some(root) = root {
            ensure!(
                *root == world_root,
                "fatal frame predicate mismatch: raw root filter"
            );
        }
        if let Some(internal) = internal {
            ensure!(
                *internal == world_internal,
                "fatal frame predicate mismatch: raw internal filter"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct FactorizedDimerAttempt {
    pub index: usize,
    pub status: FactorizedAttemptStatus,
    pub root_draws: Vec<FactorizedRootDraw>,
    pub internal_draws: Vec<FactorizedInternalDraw>,
    pub proposed: Option<[Pose; 2]>,
    pub final_feasibility: Option<DimerFeasibility>,
    pub frame: Option<FactorizedFrameCheck>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FactorizedDimerOutcome {
    pub status: FactorizedDimerStatus,
    pub caps: FactorizedDimerCaps,
    pub order: FactorizedDimerOrder,
    pub members: [usize; 2],
    pub anchor_label: usize,
    pub old: [Pose; 2],
    pub source_feasibility: DimerFeasibility,
    pub source_frame: Option<FactorizedFrameCheck>,
    pub attempts: Vec<FactorizedDimerAttempt>,
    pub candidate: Option<DefensiveDimerCandidate>,
}

/// Persist this error and stop. None means the source failed arithmetic checks
/// before any draw. InProgress fields belong to this incomplete fatal event.
#[derive(Debug, Serialize)]
pub struct FactorizedDimerFailure {
    pub fatal_error: String,
    pub outcome: Option<FactorizedDimerOutcome>,
}
impl fmt::Display for FactorizedDimerFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.fatal_error)
    }
}
impl std::error::Error for FactorizedDimerFailure {}

pub struct FactorizedDimerProposal<'a> {
    raw: DefensiveDimerProposal<'a>,
    caps: FactorizedDimerCaps,
    order: FactorizedDimerOrder,
}

impl<'a> FactorizedDimerProposal<'a> {
    pub fn new(
        raw: DefensiveDimerProposal<'a>,
        caps: FactorizedDimerCaps,
        order: FactorizedDimerOrder,
    ) -> Self {
        Self { raw, caps, order }
    }

    fn root_stage(
        &self,
        rng: &mut StdRng,
        context: &FixedDimerContext<'_>,
        old: Pose,
        attempt: &mut FactorizedDimerAttempt,
    ) -> Result<bool> {
        for index in 1..=self.caps.root {
            let draw = self.raw.draw_edge(rng, old);
            attempt.root_draws.push(FactorizedRootDraw {
                index,
                draw,
                world_pose: None,
                feasibility: None,
            });
            let record = attempt.root_draws.last_mut().unwrap();
            let relative = record.draw.proposed_relative_pose.context(
                record
                    .draw
                    .null_reason
                    .clone()
                    .unwrap_or_else(|| "null root edge".into()),
            )?;
            ensure!(
                record.draw.null_reason.is_none(),
                "root draw has a numerical error"
            );
            let root = tree_members(context.anchor(), [relative, IDENTITY])?[0];
            record.world_pose = Some(root);
            let feasibility = context.evaluate_fixed_body(root)?;
            let pass = feasibility.hard_valid();
            record.feasibility = Some(feasibility);
            if pass {
                return Ok(true);
            }
        }
        attempt.status = FactorizedAttemptStatus::RootCapExhausted;
        Ok(false)
    }

    fn internal_stage(
        &self,
        rng: &mut StdRng,
        context: &FixedDimerContext<'_>,
        old: Pose,
        attempt: &mut FactorizedDimerAttempt,
    ) -> Result<bool> {
        for index in 1..=self.caps.internal {
            let draw = self.raw.draw_edge(rng, old);
            attempt.internal_draws.push(FactorizedInternalDraw {
                index,
                draw,
                feasibility: None,
            });
            let record = attempt.internal_draws.last_mut().unwrap();
            let relative = record.draw.proposed_relative_pose.context(
                record
                    .draw
                    .null_reason
                    .clone()
                    .unwrap_or_else(|| "null internal edge".into()),
            )?;
            ensure!(
                record.draw.null_reason.is_none(),
                "internal draw has a numerical error"
            );
            let feasibility = context.evaluate_internal_relative(relative)?;
            let pass = feasibility.feasible();
            record.feasibility = Some(feasibility);
            if pass {
                return Ok(true);
            }
        }
        attempt.status = FactorizedAttemptStatus::InternalCapExhausted;
        Ok(false)
    }

    fn run(
        &self,
        rng: &mut StdRng,
        context: &FixedDimerContext<'_>,
        outcome: &mut FactorizedDimerOutcome,
    ) -> Result<()> {
        ensure!(
            outcome.source_feasibility.hard_valid(),
            "source is not hard-valid"
        );
        outcome.source_frame = Some(FactorizedFrameCheck::evaluate(context, outcome.old)?);
        let source_frame = outcome.source_frame.as_ref().unwrap();
        source_frame.validate(&outcome.source_feasibility, None, None)?;
        if !outcome.source_feasibility.internal_exclusion_contact {
            outcome.status = FactorizedDimerStatus::SourceOutsideDomain;
            return Ok(());
        }
        // Any zero cap makes the channel identity. Do not spend RNG on a stage
        // whose partner is known never to return a successful edge.
        if self.caps.root == 0 || self.caps.internal == 0 || self.caps.joint == 0 {
            outcome.status = FactorizedDimerStatus::CapExhausted;
            return Ok(());
        }
        let old_edges = source_frame.recovered_edges;
        for index in 1..=self.caps.joint {
            outcome.attempts.push(FactorizedDimerAttempt {
                index,
                status: FactorizedAttemptStatus::InProgress,
                root_draws: vec![],
                internal_draws: vec![],
                proposed: None,
                final_feasibility: None,
                frame: None,
            });
            let attempt = outcome.attempts.last_mut().unwrap();
            let success = match self.order {
                FactorizedDimerOrder::RootFirst => {
                    self.root_stage(rng, context, old_edges[0], attempt)?
                        && self.internal_stage(rng, context, old_edges[1], attempt)?
                }
                FactorizedDimerOrder::InternalFirst => {
                    self.internal_stage(rng, context, old_edges[1], attempt)?
                        && self.root_stage(rng, context, old_edges[0], attempt)?
                }
            };
            if !success {
                continue;
            }
            let root_record = attempt.root_draws.last().unwrap();
            let internal_record = attempt.internal_draws.last().unwrap();
            let edges = [
                root_record.draw.proposed_relative_pose.unwrap(),
                internal_record.draw.proposed_relative_pose.unwrap(),
            ];
            let members = tree_members(context.anchor(), edges)?;
            attempt.proposed = Some(members);
            ensure!(
                Some(members[0]) == root_record.world_pose,
                "root composition is not identical"
            );
            attempt.final_feasibility = Some(context.evaluate(members)?);
            attempt.frame = Some(FactorizedFrameCheck::evaluate(context, members)?);
            let feasibility = attempt.final_feasibility.as_ref().unwrap();
            attempt.frame.as_ref().unwrap().validate(
                feasibility,
                root_record.feasibility.as_ref(),
                internal_record.feasibility.as_ref(),
            )?;
            if !feasibility.feasible() {
                attempt.status = FactorizedAttemptStatus::FinalRejected;
                continue;
            }
            // Complete mixture density is deliberately deferred until every
            // geometry/frame check has passed. No failed-edge score is needed.
            let diagnostics = self.raw.correction(
                context.anchor(),
                outcome.old[0],
                outcome.old[1],
                members[0],
                members[1],
            )?;
            outcome.candidate = Some(DefensiveDimerCandidate {
                root: members[0],
                child: members[1],
                diagnostics,
            });
            attempt.status = FactorizedAttemptStatus::Candidate;
            outcome.status = FactorizedDimerStatus::Candidate;
            return Ok(());
        }
        outcome.status = FactorizedDimerStatus::CapExhausted;
        Ok(())
    }

    pub fn propose(
        &self,
        rng: &mut StdRng,
        context: &FixedDimerContext<'_>,
        old: [Pose; 2],
    ) -> std::result::Result<FactorizedDimerOutcome, FactorizedDimerFailure> {
        let source_feasibility = context.evaluate(old).map_err(|e| FactorizedDimerFailure {
            fatal_error: e.to_string(),
            outcome: None,
        })?;
        let mut outcome = FactorizedDimerOutcome {
            status: FactorizedDimerStatus::InProgress,
            caps: self.caps,
            order: self.order,
            members: context.members(),
            anchor_label: context.anchor_label(),
            old,
            source_feasibility,
            source_frame: None,
            attempts: vec![],
            candidate: None,
        };
        match self.run(rng, context, &mut outcome) {
            Ok(()) => Ok(outcome),
            Err(e) => Err(FactorizedDimerFailure {
                fatal_error: e.to_string(),
                outcome: Some(outcome),
            }),
        }
    }
}
