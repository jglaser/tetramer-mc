//! Fixed-context, two-neighbor independent singleton redraw.
//!
//! The catalogue is built with an exact identity member, the fixed pool
//! [other mobile, anchor], and ALL nonmoving bodies as hard spectators. The
//! source pose is never a catalogue input. Rebuild this snapshot whenever any
//! spectator moves. Label selection and the subsequent many-body bath/MH gate
//! belong to the caller; neither native labels nor contact preservation do.
//!
//! Every capped trial redraws G = .5 U + .5 M, where U is the cube in the fixed
//! external anchor frame times normalized Haar and M is the complete oligomer
//! mixture. Only hard/wall exclusions may be retried. For fixed context C,
//! q_K(y|x,C) = G(y) 1_D(y) sum_{r=0}^{K-1}(1-p_C)^r, so the cap factor
//! cancels and the correction is log G(old) - log G(new). Decoder and scoring
//! failures are fatal and retain the begun trial; they are never rejections.
use crate::{
    capped_dimer::FixedBodyFeasibility,
    defensive_dimer_proposal::{DefensiveBranch, DefensiveDensity},
    docking::{DockingMethod, DockingProposal},
    geometry::{Placed, SphereTree},
    math::{Pose, add, matmul, matvec, norm, quaternion, rotation, sub, transpose},
    oligomer_proposal::{OligomerConfig, OligomerMixture},
    spherical::Container,
};
use anyhow::{Result, ensure};
use rand::{RngExt, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde::Serialize;
use serde_json::{Value, json};
use std::{f64::consts::LN_2, fmt};

const IDENTITY: Pose = Pose {
    position: [0.; 3],
    orientation: [1., 0., 0., 0.],
};

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SingletonStatus {
    Candidate,
    CapZero,
    SourceZeroReverseFlow,
    CapExhausted,
    Fatal,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrialDisposition {
    Begun,
    GeometricRejection,
    Candidate,
    Fatal,
}

#[derive(Clone, Debug, Serialize)]
pub struct SingletonTrial {
    /// One-based index, including every geometric rejection and fatal trial.
    pub index: usize,
    pub branch: DefensiveBranch,
    pub branch_uniform: f64,
    /// Raw translation uniforms/quaternion normals or learned label/latent.
    pub trace: Value,
    pub proposed_pose: Option<Pose>,
    pub density: Option<DefensiveDensity>,
    pub feasibility: Option<FixedBodyFeasibility>,
    pub disposition: TrialDisposition,
}

#[derive(Clone, Debug, Default, Serialize, PartialEq, Eq)]
pub struct SingletonCounts {
    pub uniform: usize,
    pub learned: usize,
    pub geometric_rejections: usize,
    /// Both counters increment if one trial violates both predicates.
    pub hard_rejections: usize,
    pub wall_rejections: usize,
    pub candidates: usize,
    pub fatal_trials: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct SingletonOutcome {
    pub status: SingletonStatus,
    pub moving: usize,
    pub neighbors: [usize; 2],
    pub old: Pose,
    pub trial_cap: usize,
    pub uniform_frame: Pose,
    pub uniform_center: [f64; 3],
    pub uniform_half_width: f64,
    pub uniform_probability: f64,
    pub source_feasibility: Option<FixedBodyFeasibility>,
    pub old_density: Option<DefensiveDensity>,
    pub new_density: Option<DefensiveDensity>,
    pub candidate: Option<Pose>,
    /// Full G(old)/G(new), present only for a successful candidate.
    pub full_log_reverse_forward: Option<f64>,
    pub counts: SingletonCounts,
    pub trials: Vec<SingletonTrial>,
}
impl SingletonOutcome {
    pub fn complete_log_correction(&self) -> Result<f64> {
        ensure!(
            self.status == SingletonStatus::Candidate,
            "no singleton candidate"
        );
        self.full_log_reverse_forward
            .filter(|x| x.is_finite())
            .ok_or_else(|| anyhow::anyhow!("missing finite singleton correction"))
    }
}

/// This is a failed calculation, never an MCMC self-loop. Persist and stop.
#[derive(Debug, Serialize)]
pub struct SingletonFailure {
    pub fatal_error: String,
    pub outcome: SingletonOutcome,
}
impl fmt::Display for SingletonFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.fatal_error)
    }
}
impl std::error::Error for SingletonFailure {}

pub struct TwoNeighborSingleton<'a> {
    core: &'a SphereTree,
    wall: Container,
    spectators: Vec<(usize, Placed)>,
    mixture: OligomerMixture<'a>,
    moving: usize,
    neighbors: [usize; 2],
    uniform_frame: Pose,
    half_width: f64,
    log_uniform: f64,
    trial_cap: usize,
}

fn check_pose(pose: Pose, bound: f64) -> Result<()> {
    pose.validate()?;
    ensure!(
        (8. * (1. + norm(pose.position) + bound))
            .powi(2)
            .is_finite(),
        "unsupported singleton geometry arithmetic magnitude"
    );
    Ok(())
}

fn relative(frame: Pose, pose: Pose) -> Pose {
    let inverse = transpose(rotation(frame.orientation));
    Pose {
        position: matvec(inverse, sub(pose.position, frame.position)),
        orientation: quaternion(matmul(inverse, rotation(pose.orientation))),
    }
}

fn compose(frame: Pose, pose: Pose) -> Pose {
    let r = rotation(frame.orientation);
    Pose {
        position: add(frame.position, matvec(r, pose.position)),
        orientation: quaternion(matmul(r, rotation(pose.orientation))),
    }
}

/// Validate every component before reduction: a log-sum helper can otherwise
/// silently discard NaN terms when a different finite component is present.
fn checked_log_sum(values: &[f64]) -> Result<f64> {
    ensure!(
        values
            .iter()
            .all(|v| v.is_finite() || *v == f64::NEG_INFINITY),
        "malformed singleton density component"
    );
    let largest = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if largest == f64::NEG_INFINITY {
        return Ok(largest);
    }
    let result = largest + values.iter().map(|v| (v - largest).exp()).sum::<f64>().ln();
    ensure!(result.is_finite(), "malformed singleton density sum");
    Ok(result)
}

impl<'a> TwoNeighborSingleton<'a> {
    /// `state[moving]` is deliberately never read. `neighbors` is the fixed
    /// ordered pool [other mobile, anchor]; every other label is a spectator.
    /// The wall radius is reused with this exact core shape. The caller must
    /// keep the spectator snapshot mutually valid and rebuild after it changes.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        proposal: &'a DockingProposal,
        core: &'a SphereTree,
        wall: &Container,
        state: &[Pose],
        moving: usize,
        neighbors: [usize; 2],
        uniform_half_width: f64,
        trial_cap: usize,
        config: &OligomerConfig,
    ) -> Result<Self> {
        ensure!(
            proposal.method() == DockingMethod::PosteriorInvolution && !proposal.is_periodic(),
            "singleton redraw needs an open posterior member atlas"
        );
        ensure!(moving < state.len(), "invalid moving singleton label");
        ensure!(
            neighbors[0] != neighbors[1]
                && neighbors.iter().all(|&i| i < state.len() && i != moving),
            "singleton neighbors must be distinct fixed labels"
        );
        ensure!(
            uniform_half_width.is_finite() && uniform_half_width > 0.,
            "invalid singleton cube half-width"
        );
        // All possible cube draws must have safe geometry arithmetic.
        check_pose(
            Pose {
                position: [uniform_half_width; 3],
                ..IDENTITY
            },
            core.bound,
        )?;
        let fixed: Vec<Pose> = state
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != moving)
            .map(|(_, &p)| p)
            .collect();
        for &pose in &fixed {
            check_pose(pose, core.bound)?;
        }
        let uniform_frame = state[neighbors[1]];
        // Rotation preserves radius; include the fixed frame translation in
        // the representability bound for every possible defensive cube draw.
        check_pose(
            Pose {
                position: [norm(uniform_frame.position) + 3_f64.sqrt() * uniform_half_width; 3],
                ..IDENTITY
            },
            core.bound,
        )?;
        let wall = Container::new(wall.radius, core)?;
        let mixture = OligomerMixture::build(
            proposal,
            core,
            &wall,
            &[IDENTITY],
            &fixed,
            &neighbors.map(|i| state[i]),
            config,
        )?;
        let log_total = checked_log_sum(mixture.log_weights())?;
        ensure!(
            log_total.is_finite() && log_total.abs() < 1e-10,
            "singleton catalogue weights are not normalized"
        );
        Ok(Self {
            core,
            wall,
            spectators: state
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != moving)
                .map(|(i, &p)| (i, Placed::new(p)))
                .collect(),
            mixture,
            moving,
            neighbors,
            uniform_frame,
            half_width: uniform_half_width,
            log_uniform: -3. * (LN_2 + uniform_half_width.ln()),
            trial_cap,
        })
    }

    pub fn mixture(&self) -> &OligomerMixture<'a> {
        &self.mixture
    }

    /// Full physical-pose density, with respect to translation and normalized
    /// Haar. The cube is expressed in `neighbors[1]`'s fixed frame; this rigid
    /// change of frame has unit Jacobian. The atlas's world-uniform kernel is
    /// not part of M.
    pub fn density(&self, pose: Pose) -> Result<DefensiveDensity> {
        check_pose(pose, self.core.bound)?;
        let in_frame = relative(self.uniform_frame, pose);
        check_pose(in_frame, self.core.bound)?;
        let log_uniform = if in_frame.position.iter().all(|x| x.abs() <= self.half_width) {
            self.log_uniform
        } else {
            f64::NEG_INFINITY
        };
        let log_learned = checked_log_sum(&self.mixture.checked_label_logs(pose)?)?;
        let log_full = checked_log_sum(&[log_uniform - LN_2, log_learned - LN_2])?;
        Ok(DefensiveDensity {
            log_uniform,
            log_learned,
            log_full,
        })
    }

    /// No contact/native condition: detachment remains in the target support.
    pub fn evaluate(&self, pose: Pose) -> Result<FixedBodyFeasibility> {
        check_pose(pose, self.core.bound)?;
        let placed = Placed::new(pose);
        Ok(FixedBodyFeasibility {
            spectator_core_collisions: self
                .spectators
                .iter()
                .filter_map(|(label, p)| self.core.overlaps(&placed, p).then_some(*label))
                .collect(),
            wall_valid: self.wall.contains(pose),
        })
    }

    fn draw(&self, rng: &mut StdRng, trial: &mut SingletonTrial) -> Result<Pose> {
        let pose = match trial.branch {
            DefensiveBranch::Uniform => {
                let translation_uniforms: [f64; 3] = std::array::from_fn(|_| rng.random());
                let quaternion_normals: [f64; 4] =
                    std::array::from_fn(|_| StandardNormal.sample(rng));
                trial.trace = json!({"translation_uniforms":translation_uniforms,
                    "quaternion_normals":quaternion_normals});
                let length = quaternion_normals.iter().map(|x| x * x).sum::<f64>().sqrt();
                ensure!(
                    length.is_finite() && length > 0.,
                    "invalid Haar quaternion norm"
                );
                compose(
                    self.uniform_frame,
                    Pose {
                        position: translation_uniforms.map(|u| (2. * u - 1.) * self.half_width),
                        orientation: quaternion_normals.map(|x| x / length),
                    },
                )
            }
            DefensiveBranch::Learned => {
                let (pose, trace) = self.mixture.draw_singleton_independent(rng)?;
                trial.trace = trace;
                pose.ok_or_else(|| anyhow::anyhow!("singleton decoder null: {}", trial.trace))?
            }
        };
        trial.proposed_pose = Some(pose);
        check_pose(pose, self.core.bound)?;
        Ok(pose)
    }

    /// K=0 consumes no RNG. A zero-density source also self-loops without RNG.
    /// All begun trials are retained in either the outcome or the fatal error.
    pub fn propose(
        &self,
        rng: &mut StdRng,
        old: Pose,
    ) -> std::result::Result<SingletonOutcome, SingletonFailure> {
        let mut outcome = SingletonOutcome {
            status: SingletonStatus::CapExhausted,
            moving: self.moving,
            neighbors: self.neighbors,
            old,
            trial_cap: self.trial_cap,
            uniform_frame: self.uniform_frame,
            uniform_center: self.uniform_frame.position,
            uniform_half_width: self.half_width,
            uniform_probability: 0.5,
            source_feasibility: None,
            old_density: None,
            new_density: None,
            candidate: None,
            full_log_reverse_forward: None,
            counts: SingletonCounts::default(),
            trials: Vec::new(),
        };
        match self.run(rng, old, &mut outcome) {
            Ok(()) => Ok(outcome),
            Err(error) => {
                outcome.status = SingletonStatus::Fatal;
                if let Some(last) = outcome.trials.last_mut() {
                    last.disposition = TrialDisposition::Fatal;
                    outcome.counts.fatal_trials += 1;
                }
                Err(SingletonFailure {
                    fatal_error: error.to_string(),
                    outcome,
                })
            }
        }
    }

    fn run(&self, rng: &mut StdRng, old: Pose, outcome: &mut SingletonOutcome) -> Result<()> {
        let source = self.evaluate(old)?;
        let source_valid = source.hard_valid();
        outcome.source_feasibility = Some(source);
        ensure!(source_valid, "source singleton is not hard/wall valid");
        if self.trial_cap == 0 {
            outcome.status = SingletonStatus::CapZero;
            return Ok(());
        }
        let old_density = self.density(old)?;
        outcome.old_density = Some(old_density);
        if old_density.log_full == f64::NEG_INFINITY {
            outcome.status = SingletonStatus::SourceZeroReverseFlow;
            return Ok(());
        }
        for index in 1..=self.trial_cap {
            let branch_uniform = rng.random::<f64>();
            let branch = if branch_uniform < 0.5 {
                outcome.counts.uniform += 1;
                DefensiveBranch::Uniform
            } else {
                outcome.counts.learned += 1;
                DefensiveBranch::Learned
            };
            outcome.trials.push(SingletonTrial {
                index,
                branch,
                branch_uniform,
                trace: Value::Null,
                proposed_pose: None,
                density: None,
                feasibility: None,
                disposition: TrialDisposition::Begun,
            });
            let trial = outcome.trials.last_mut().unwrap();
            let pose = self.draw(rng, trial)?;
            let density = self.density(pose)?;
            trial.density = Some(density);
            ensure!(
                density.log_full.is_finite(),
                "drawn singleton has zero full density"
            );
            let feasibility = self.evaluate(pose)?;
            let valid = feasibility.hard_valid();
            if !valid {
                outcome.counts.geometric_rejections += 1;
                outcome.counts.hard_rejections +=
                    usize::from(!feasibility.spectator_core_collisions.is_empty());
                outcome.counts.wall_rejections += usize::from(!feasibility.wall_valid);
            }
            trial.feasibility = Some(feasibility);
            if valid {
                let correction = old_density.log_full - density.log_full;
                ensure!(correction.is_finite(), "nonfinite singleton correction");
                trial.disposition = TrialDisposition::Candidate;
                outcome.status = SingletonStatus::Candidate;
                outcome.candidate = Some(pose);
                outcome.new_density = Some(density);
                outcome.full_log_reverse_forward = Some(correction);
                outcome.counts.candidates += 1;
                return Ok(());
            }
            trial.disposition = TrialDisposition::GeometricRejection;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::checked_log_sum;
    #[test]
    fn malformed_components_cannot_hide_under_finite_density() {
        for malformed in [f64::NAN, f64::INFINITY] {
            assert!(checked_log_sum(&[0., malformed]).is_err());
        }
        assert_eq!(
            checked_log_sum(&[f64::NEG_INFINITY]).unwrap(),
            f64::NEG_INFINITY
        );
        assert_eq!(checked_log_sum(&[0., f64::NEG_INFINITY]).unwrap(), 0.);
    }
}
