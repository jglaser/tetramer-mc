//! Fixed-label independent redraw of both dimer tree edges.
//!
//! Each edge has normalized physical-pose density
//! F(h) = alpha U_L(h) + (1-alpha) G(h), where U_L is the centered relative
//! translation cube [-L,L]^3 times normalized Haar measure and G is the frozen
//! learned singleton atlas. Both draws are independent of the old edges. The
//! child is reconstructed relative to the NEW root. Product translation/Haar
//! tree coordinates have unit Jacobian, so the full correction is
//! log F(h0_old)+log F(h1_old)-log F(h0_new)-log F(h1_new).
//!
//! Both mixture contributions enter every density, regardless of drawn branch.
//! This is an independent proposal, not a modification of the correlated map
//! and not an invertible auxiliary trace. Numerical nulls are retained without
//! redraw. There is no wall, hard/depletion gate, native label or state selection.
//! Selection correction is zero ONLY for fixed spectator/root/child labels.
//! Alpha=0 and alpha=1 are supported reference limits. A pure-uniform proposal
//! from an old pose outside its cube has zero reverse density and correction
//! -infinity: a valid always-reject candidate, not a numerical error. Serialized
//! logarithms encode this zero density as the string "-inf"; NaN/+inf are errors.
use crate::{
    dimer_tree_proposal::{tree_coordinates, tree_members},
    docking::{DockingMethod, DockingProposal},
    math::Pose,
};
use anyhow::{Context, Result, ensure};
use rand::{RngExt, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde::{Serialize, Serializer};
use serde_json::{Value, json};

const IDENTITY: Pose = Pose {
    position: [0.; 3],
    orientation: [1., 0., 0., 0.],
};

fn serialize_log<S: Serializer>(x: &f64, serializer: S) -> Result<S::Ok, S::Error> {
    if *x == f64::NEG_INFINITY {
        serializer.serialize_str("-inf")
    } else if !x.is_finite() {
        Err(serde::ser::Error::custom("invalid log density"))
    } else {
        serializer.serialize_f64(*x)
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct DefensiveDensity {
    #[serde(serialize_with = "serialize_log")]
    pub log_uniform: f64,
    #[serde(serialize_with = "serialize_log")]
    pub log_learned: f64,
    #[serde(serialize_with = "serialize_log")]
    pub log_full: f64,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DefensiveBranch {
    Uniform,
    Learned,
}

#[derive(Clone, Debug, Serialize)]
pub struct DefensiveDimerEdge {
    pub old_relative_pose: Pose,
    pub branch: DefensiveBranch,
    /// Uniform raw coordinates/normals, or the existing learned label/latent.
    /// A generation record only: no inverse-map interpretation is claimed.
    pub trace: Value,
    pub proposed_relative_pose: Option<Pose>,
    pub null_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct DefensiveDimerDiagnostics {
    pub old_edges: [DefensiveDensity; 2],
    pub new_edges: [DefensiveDensity; 2],
    #[serde(serialize_with = "serialize_log")]
    pub full_old_log_density: f64,
    #[serde(serialize_with = "serialize_log")]
    pub full_new_log_density: f64,
    #[serde(serialize_with = "serialize_log")]
    pub log_reverse_forward: f64,
    pub log_tree_coordinate_jacobian: f64,
    pub selection_log_reverse_forward: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct DefensiveDimerCandidate {
    pub root: Pose,
    pub child: Pose,
    pub diagnostics: DefensiveDimerDiagnostics,
}

#[derive(Clone, Debug, Serialize)]
pub struct DefensiveDimerOutcome {
    pub spectator: Pose,
    pub old_root: Pose,
    pub old_child: Pose,
    /// Both independently attempted draws are retained, including on null.
    pub edges: [DefensiveDimerEdge; 2],
    pub candidate: Option<DefensiveDimerCandidate>,
    pub null_reason: Option<String>,
}

pub struct DefensiveDimerProposal<'a> {
    learned: &'a DockingProposal,
    half_width: f64,
    uniform_probability: f64,
    log_uniform: f64,
}

impl<'a> DefensiveDimerProposal<'a> {
    pub fn new(
        learned: &'a DockingProposal,
        half_width: f64,
        uniform_probability: f64,
    ) -> Result<Self> {
        ensure!(
            learned.method() == DockingMethod::PosteriorInvolution && !learned.is_periodic(),
            "Defensive dimer redraw needs an open posterior member atlas"
        );
        ensure!(
            half_width.is_finite() && half_width > 0.,
            "invalid relative cube half-width"
        );
        ensure!(
            uniform_probability.is_finite() && (0. ..=1.).contains(&uniform_probability),
            "invalid defensive probability"
        );
        Ok(Self {
            learned,
            half_width,
            uniform_probability,
            log_uniform: -3. * (std::f64::consts::LN_2 + half_width.ln()),
        })
    }

    pub fn half_width(&self) -> f64 {
        self.half_width
    }
    pub fn uniform_probability(&self) -> f64 {
        self.uniform_probability
    }

    /// Complete density with respect to translation times normalized Haar.
    /// The atlas model's own world-cube weight is deliberately not reused.
    pub fn density(&self, relative: Pose) -> Result<DefensiveDensity> {
        relative.validate()?;
        let log_uniform = if relative.position.iter().all(|x| x.abs() <= self.half_width) {
            self.log_uniform
        } else {
            f64::NEG_INFINITY
        };
        let log_learned = self.learned.members_log_density(&[relative], &[IDENTITY])?;
        ensure!(
            log_learned.is_finite() || log_learned == f64::NEG_INFINITY,
            "nonfinite learned density"
        );
        let a = if self.uniform_probability == 0. {
            f64::NEG_INFINITY
        } else {
            self.uniform_probability.ln() + log_uniform
        };
        let b = if self.uniform_probability == 1. {
            f64::NEG_INFINITY
        } else {
            (-self.uniform_probability).ln_1p() + log_learned
        };
        let maximum = a.max(b);
        let log_full = if maximum == f64::NEG_INFINITY {
            maximum
        } else {
            maximum + ((a - maximum).exp() + (b - maximum).exp()).ln()
        };
        ensure!(
            log_full.is_finite() || log_full == f64::NEG_INFINITY,
            "invalid full defensive density"
        );
        Ok(DefensiveDensity {
            log_uniform,
            log_learned,
            log_full,
        })
    }

    fn diagnostics(&self, old: [Pose; 2], new: [Pose; 2]) -> Result<DefensiveDimerDiagnostics> {
        let old_edges = [self.density(old[0])?, self.density(old[1])?];
        let new_edges = [self.density(new[0])?, self.density(new[1])?];
        let full_old_log_density = old_edges.iter().map(|d| d.log_full).sum::<f64>();
        let full_new_log_density = new_edges.iter().map(|d| d.log_full).sum::<f64>();
        ensure!(
            full_new_log_density.is_finite(),
            "sampled endpoint has zero or invalid proposal density"
        );
        let log_reverse_forward = full_old_log_density - full_new_log_density;
        ensure!(
            log_reverse_forward.is_finite() || log_reverse_forward == f64::NEG_INFINITY,
            "invalid independent proposal correction"
        );
        Ok(DefensiveDimerDiagnostics {
            old_edges,
            new_edges,
            full_old_log_density,
            full_new_log_density,
            log_reverse_forward,
            log_tree_coordinate_jacobian: 0.,
            selection_log_reverse_forward: 0.,
        })
    }

    /// Deterministic density-ratio evaluation for fixed open-space endpoints.
    /// New endpoints must have positive F density; old zero density is legal.
    pub fn correction(
        &self,
        spectator: Pose,
        root: Pose,
        child: Pose,
        new_root: Pose,
        new_child: Pose,
    ) -> Result<DefensiveDimerDiagnostics> {
        self.diagnostics(
            tree_coordinates(spectator, root, child)?,
            tree_coordinates(spectator, new_root, new_child)?,
        )
    }

    fn draw(&self, rng: &mut StdRng, old: Pose) -> DefensiveDimerEdge {
        // Keep an explicit branch coin in both reference limits as well.
        let branch_uniform = rng.random::<f64>();
        let branch = if branch_uniform < self.uniform_probability {
            DefensiveBranch::Uniform
        } else {
            DefensiveBranch::Learned
        };
        let (proposed_relative_pose, mut trace, null_reason) = match branch {
            DefensiveBranch::Uniform => {
                let translation_uniforms: [f64; 3] = std::array::from_fn(|_| rng.random());
                let normals: [f64; 4] = std::array::from_fn(|_| StandardNormal.sample(rng));
                let length = normals.iter().map(|x| x * x).sum::<f64>().sqrt();
                let pose = Pose {
                    position: translation_uniforms.map(|u| (2. * u - 1.) * self.half_width),
                    orientation: normals.map(|x| x / length),
                };
                let trace = json!({"translation_uniforms":translation_uniforms,
                    "quaternion_normals":normals});
                match pose.validate() {
                    Ok(()) => (Some(pose), trace, None),
                    Err(error) => (None, trace, Some(error.to_string())),
                }
            }
            DefensiveBranch::Learned => {
                match self.learned.draw_singleton_independent(rng, &[IDENTITY]) {
                    Ok((pose, trace)) => {
                        let reason = if pose.is_none() {
                            Some(
                                trace
                                    .get("null_reason")
                                    .and_then(Value::as_str)
                                    .unwrap_or("Null learned draw")
                                    .to_string(),
                            )
                        } else {
                            None
                        };
                        (pose, trace, reason)
                    }
                    Err(error) => (None, json!({}), Some(error.to_string())),
                }
            }
        };
        trace["branch_uniform"] = json!(branch_uniform);
        DefensiveDimerEdge {
            old_relative_pose: old,
            branch,
            trace,
            proposed_relative_pose,
            null_reason,
        }
    }

    /// Draw one complete independent edge without scoring either endpoint.
    ///
    /// This is exactly the edge drawer used by `propose`, including its branch
    /// coin and null trace. `old_relative_pose` is recorded metadata only; a
    /// caller must validate its source before consuming RNG. Geometric retries
    /// must redraw this WHOLE mixture, never retain a failed component label.
    /// A numerical null is not a geometric rejection and must remain visible.
    pub fn draw_edge(&self, rng: &mut StdRng, old_relative_pose: Pose) -> DefensiveDimerEdge {
        self.draw(rng, old_relative_pose)
    }

    pub fn propose(
        &self,
        rng: &mut StdRng,
        spectator: Pose,
        root: Pose,
        child: Pose,
    ) -> Result<DefensiveDimerOutcome> {
        let old = tree_coordinates(spectator, root, child)?;
        // Neither draw depends on the old pose; always attempt BOTH exactly once.
        let edges = [self.draw(rng, old[0]), self.draw(rng, old[1])];
        let result = (|| -> Result<DefensiveDimerCandidate> {
            let new = [
                edges[0]
                    .proposed_relative_pose
                    .context("Docking edge redraw is null")?,
                edges[1]
                    .proposed_relative_pose
                    .context("Internal edge redraw is null")?,
            ];
            let members = tree_members(spectator, new)?;
            // Evaluate the stored physical endpoint, including its coordinate
            // round trip; raw drawn coordinates remain in the edge record.
            let diagnostics =
                self.diagnostics(old, tree_coordinates(spectator, members[0], members[1])?)?;
            Ok(DefensiveDimerCandidate {
                root: members[0],
                child: members[1],
                diagnostics,
            })
        })();
        let (candidate, null_reason) = match result {
            Ok(candidate) => (Some(candidate), None),
            Err(error) => (None, Some(error.to_string())),
        };
        Ok(DefensiveDimerOutcome {
            spectator,
            old_root: root,
            old_child: child,
            edges,
            candidate,
            null_reason,
        })
    }
}
