//! Fixed-label, learned-only joint docking/internal-pose proposal.
//!
//! The spectator A is fixed, h0=A^-1 g_root and h1=g_root^-1 g_child.
//! Two independent posterior-source maps act on the OLD h0 and h1, followed by
//! g_root'=A h0', g_child'=g_root' h1'. Product translation/Haar measure has
//! unit Jacobian in these tree coordinates. The two existing complete proposal
//! corrections therefore add. Labels, spectator and tree order are fixed:
//! there is no selection correction in this isolated building block.
//!
//! This is NOT a physical MC kernel. It makes no hard, wall or depletion
//! decision, selects no particles, and has no production integration. In
//! particular RigidSubset's depletion gate is INVALID for this proposal because
//! changing h1 changes the moving exclusion union's volume. Tetramers themselves
//! remain rigid. The uniform world-cube branch is deliberately not transplanted
//! into these relative coordinates; ordinary defensive/local kernels must be
//! composed separately under a valid physical target.
use crate::{
    basin_involution::BasinTrace,
    docking::{DockingMethod, DockingProposal, MemberLabel, MemberStep, draw_log_category},
    math::{Pose, add, matmul, matvec, quaternion, rotation, sub, transpose},
};
use anyhow::{Context, Result, ensure};
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};

const IDENTITY: Pose = Pose {
    position: [0.; 3],
    orientation: [1., 0., 0., 0.],
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DimerTreeTrace {
    /// Edge 0: spectator -> root. Edge 1: root -> child.
    pub edges: [BasinTrace; 2],
}

#[derive(Clone, Debug, Serialize)]
pub struct DimerTreeEdge {
    pub old_relative_pose: Pose,
    /// None only when there was no finite source density. Never redrawn.
    pub trace: Option<BasinTrace>,
    pub step: Option<MemberStep>,
    pub null_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct DimerTreeDiagnostics {
    pub full_old_log_density: f64,
    pub full_new_log_density: f64,
    pub log_reverse_forward: f64,
    pub log_extended_jacobian: f64,
    pub log_auxiliary_ratio: f64,
    pub label_log_reverse_forward: f64,
    pub expanded_log_reverse_forward: f64,
    /// Root/child <-> tree coordinates preserve product physical pose measure.
    pub log_tree_coordinate_jacobian: f64,
    /// Exactly zero only because all three ordered labels are fixed here.
    pub selection_log_reverse_forward: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct DimerTreeCandidate {
    pub root: Pose,
    pub child: Pose,
    pub inverse_trace: DimerTreeTrace,
    pub diagnostics: DimerTreeDiagnostics,
}

#[derive(Clone, Debug, Serialize)]
pub struct DimerTreeOutcome {
    /// Copied unchanged, including quaternion sign and floating-point bits.
    pub spectator: Pose,
    pub old_root: Pose,
    pub old_child: Pose,
    /// Both attempts are retained even if either makes the joint proposal null.
    pub edges: [DimerTreeEdge; 2],
    pub candidate: Option<DimerTreeCandidate>,
    pub null_reason: Option<String>,
}

fn relative(anchor: Pose, pose: Pose) -> Pose {
    let inverse = transpose(rotation(anchor.orientation));
    Pose {
        position: matvec(inverse, sub(pose.position, anchor.position)),
        orientation: quaternion(matmul(inverse, rotation(pose.orientation))),
    }
}

fn compose(anchor: Pose, pose: Pose) -> Pose {
    let r = rotation(anchor.orientation);
    Pose {
        position: add(anchor.position, matvec(r, pose.position)),
        orientation: quaternion(matmul(r, rotation(pose.orientation))),
    }
}

/// Open-space physical tree coordinates; no wrapping or minimum image.
pub fn tree_coordinates(spectator: Pose, root: Pose, child: Pose) -> Result<[Pose; 2]> {
    for pose in [spectator, root, child] {
        pose.validate()?;
    }
    let coordinates = [relative(spectator, root), relative(root, child)];
    for pose in coordinates {
        pose.validate()?;
    }
    Ok(coordinates)
}

/// Decode tree coordinates to the root and child; spectator is unchanged.
pub fn tree_members(spectator: Pose, coordinates: [Pose; 2]) -> Result<[Pose; 2]> {
    for pose in [spectator, coordinates[0], coordinates[1]] {
        pose.validate()?;
    }
    let root = compose(spectator, coordinates[0]);
    let child = compose(root, coordinates[1]);
    root.validate()?;
    child.validate()?;
    Ok([root, child])
}

/// Both edges borrow exactly the same immutable atlas and map correlation.
pub struct DimerTreeProposal<'a> {
    proposal: &'a DockingProposal,
}

impl<'a> DimerTreeProposal<'a> {
    pub fn new(proposal: &'a DockingProposal) -> Result<Self> {
        ensure!(
            proposal.method() == DockingMethod::PosteriorInvolution && !proposal.is_periodic(),
            "Dimer tree needs an open posterior-involution proposal"
        );
        Ok(Self { proposal })
    }

    fn draw_trace(&self, rng: &mut StdRng, old: Pose) -> Result<Option<BasinTrace>> {
        let source_logs = self.proposal.branch_log_densities(old, IDENTITY)?;
        let Some(source) = draw_log_category(rng, &source_logs)? else {
            return Ok(None);
        };
        let (_, _, weights) = self.proposal.member_chart_parts();
        let target = draw_log_category(rng, weights)?.context("No dimer edge destination label")?;
        Ok(Some(BasinTrace {
            source,
            target,
            noise: std::array::from_fn(|_| StandardNormal.sample(rng)),
        }))
    }

    fn apply_edge(&self, old: Pose, trace: Option<BasinTrace>) -> DimerTreeEdge {
        let result = trace.as_ref().map(|trace| {
            self.proposal.apply_member_trace(
                &[old],
                0,
                &[IDENTITY],
                MemberLabel {
                    member: 0,
                    anchor: 0,
                    branch: trace.source,
                },
                MemberLabel {
                    member: 0,
                    anchor: 0,
                    branch: trace.target,
                },
                trace.noise,
            )
        });
        let (step, null_reason) = match result {
            Some(Ok(step)) => (Some(step), None),
            Some(Err(error)) => (None, Some(error.to_string())),
            None => (None, Some("No finite posterior source density".into())),
        };
        DimerTreeEdge {
            old_relative_pose: old,
            trace,
            step,
            null_reason,
        }
    }

    fn finish(
        &self,
        spectator: Pose,
        root: Pose,
        child: Pose,
        old: [Pose; 2],
        traces: [Option<BasinTrace>; 2],
    ) -> DimerTreeOutcome {
        let [trace0, trace1] = traces;
        let edges = [
            self.apply_edge(old[0], trace0),
            self.apply_edge(old[1], trace1),
        ];
        let assembled = (|| -> Result<DimerTreeCandidate> {
            let a = edges[0]
                .step
                .as_ref()
                .context("Docking edge proposal is null")?;
            let b = edges[1]
                .step
                .as_ref()
                .context("Internal edge proposal is null")?;
            let mut members = tree_members(spectator, [a.handle, b.handle])?;
            if a.identity {
                members[0] = root;
                members[1] = if b.identity {
                    child
                } else {
                    compose(root, b.handle)
                };
            }
            for pose in members {
                pose.validate()?;
            }
            let diagnostics = DimerTreeDiagnostics {
                full_old_log_density: a.full_old_member_log_density + b.full_old_member_log_density,
                full_new_log_density: a.full_new_member_log_density + b.full_new_member_log_density,
                log_reverse_forward: a.log_reverse_forward + b.log_reverse_forward,
                log_extended_jacobian: a.step.log_extended_jacobian + b.step.log_extended_jacobian,
                log_auxiliary_ratio: a.step.log_auxiliary_ratio + b.step.log_auxiliary_ratio,
                label_log_reverse_forward: a.label_log_reverse_forward
                    + b.label_log_reverse_forward,
                expanded_log_reverse_forward: a.expanded_log_reverse_forward
                    + b.expanded_log_reverse_forward,
                log_tree_coordinate_jacobian: 0.,
                selection_log_reverse_forward: 0.,
            };
            ensure!(
                [
                    diagnostics.full_old_log_density,
                    diagnostics.full_new_log_density,
                    diagnostics.log_reverse_forward,
                    diagnostics.log_extended_jacobian,
                    diagnostics.log_auxiliary_ratio,
                    diagnostics.label_log_reverse_forward,
                    diagnostics.expanded_log_reverse_forward
                ]
                .iter()
                .all(|x| x.is_finite()),
                "Nonfinite joint dimer proposal diagnostics"
            );
            Ok(DimerTreeCandidate {
                root: members[0],
                child: members[1],
                inverse_trace: DimerTreeTrace {
                    edges: [a.step.inverse_trace.clone(), b.step.inverse_trace.clone()],
                },
                diagnostics,
            })
        })();
        let (candidate, null_reason) = match assembled {
            Ok(candidate) => (Some(candidate), None),
            Err(error) => (None, Some(error.to_string())),
        };
        DimerTreeOutcome {
            spectator,
            old_root: root,
            old_child: child,
            edges,
            candidate,
            null_reason,
        }
    }

    /// Apply both retained traces. Numerical nulls retain both attempts; no retry.
    pub fn apply(
        &self,
        spectator: Pose,
        root: Pose,
        child: Pose,
        trace: &DimerTreeTrace,
    ) -> Result<DimerTreeOutcome> {
        let old = tree_coordinates(spectator, root, child)?;
        Ok(self.finish(spectator, root, child, old, trace.edges.clone().map(Some)))
    }

    /// Draw independent edge traces from OLD tree coordinates before either map.
    /// The learned-only source/target law is the existing posterior map law.
    pub fn propose(
        &self,
        rng: &mut StdRng,
        spectator: Pose,
        root: Pose,
        child: Pose,
    ) -> Result<DimerTreeOutcome> {
        let old = tree_coordinates(spectator, root, child)?;
        // Evaluate both draw operations before propagating an invalid-law error.
        let [first, second] = [self.draw_trace(rng, old[0]), self.draw_trace(rng, old[1])];
        Ok(self.finish(spectator, root, child, old, [first?, second?]))
    }
}
