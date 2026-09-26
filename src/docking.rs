//! One moving rigid body in a fixed neighborhood and a fixed center-capture ball.
//! This conditional docking experiment uses the full immutable learned atlas.
//! An optional explicit original-q window restricts the physical target for
//! conditional contact diagnostics; it never changes the proposal charts.
use crate::{
    basin_involution::{BasinPair, BasinStep, BasinTrace, FixedBasinInvolution},
    depletion::{self, GateOptions},
    geometry::{Environment, Placed, Shape, SphereTree},
    math::*,
    native_region::{NativeMetric, QWindow},
    proposal::{FrozenRelativePoseProposal, RelativePoseBranch},
    simulation::{cpu_seconds, hash_bytes, hash_file, save},
};
use anyhow::{Context, Result, ensure};
use rand::{RngExt, SeedableRng, distr::Open01, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DockingConfig {
    pub shape: PathBuf,
    pub fixed_poses: Vec<Pose>,
    pub initial_pose: Pose,
    pub capture_center: Vec3,
    pub capture_radius: f64,
    pub depletant_radius: f64,
    pub reservoir_density: f64,
    pub poisson_lambda_ratio: f64,
    pub translation_steps: Vec<f64>,
    pub rotation_steps_deg: Vec<f64>,
    pub rotation_probability: f64,
    pub local_attempts_per_cycle: usize,
    pub uniform_probability: f64,
    pub seed: u64,
    #[serde(default)]
    pub endpoint_gate: GateOptions,
    #[serde(default)]
    pub metadata: Value,
    /// Fixed spectator index used only to define proposal coordinates. Every
    /// fixed pose remains present in the physical hard/depletion environment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proposal_anchor_index: Option<usize>,
    /// An explicit conditional target, never a change to proposal chart widths.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_region: Option<DockingRegion>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DockingRegion {
    pub metric: NativeMetric,
    pub window: QWindow,
}
impl DockingRegion {
    pub fn validate(&self) -> Result<()> {
        self.window.validate()?;
        ensure!(
            !self.metric.native_poses.is_empty() && !self.metric.rigid_members.is_empty(),
            "Conditional region needs native reference poses and rigid members"
        );
        ensure!(
            self.metric.member_error_scale.is_finite()
                && self.metric.member_error_scale > 0.
                && self.metric.angle_error_scale_deg.is_finite()
                && self.metric.angle_error_scale_deg > 0.
                && self.metric.angle_error_scale_deg <= 180.,
            "Invalid original registration metric"
        );
        for pose in self
            .metric
            .native_poses
            .iter()
            .chain(&self.metric.rigid_members)
        {
            pose.validate()?;
        }
        Ok(())
    }
    pub fn contains(&self, pose: Pose) -> bool {
        self.window.contains(self.metric.q(pose))
    }
}

impl DockingConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.fixed_poses.is_empty(),
            "At least one fixed neighbor is required"
        );
        self.initial_pose.validate()?;
        for p in &self.fixed_poses {
            p.validate()?;
        }
        if let Some(index) = self.proposal_anchor_index {
            ensure!(
                index < self.fixed_poses.len(),
                "Proposal anchor index out of range"
            );
        }
        if let Some(region) = &self.target_region {
            region.validate()?;
        }
        ensure!(
            self.capture_center.iter().all(|x| x.is_finite())
                && self.capture_radius.is_finite()
                && self.capture_radius > 0.,
            "Invalid capture ball"
        );
        ensure!(
            self.depletant_radius.is_finite()
                && self.depletant_radius >= 0.
                && self.reservoir_density.is_finite()
                && self.reservoir_density >= 0.
                && self.poisson_lambda_ratio.is_finite()
                && self.poisson_lambda_ratio > 0.,
            "Invalid bath"
        );
        ensure!(
            !self.translation_steps.is_empty()
                && !self.rotation_steps_deg.is_empty()
                && self
                    .translation_steps
                    .iter()
                    .chain(&self.rotation_steps_deg)
                    .all(|x| x.is_finite() && *x >= 0.),
            "Local step lists must be finite and nonnegative"
        );
        ensure!(
            self.rotation_probability.is_finite()
                && (0. ..=1.).contains(&self.rotation_probability)
                && self.uniform_probability.is_finite()
                && self.uniform_probability > 0.
                && self.uniform_probability < 1.,
            "Invalid proposal probabilities"
        );
        self.endpoint_gate.validate()
    }
    pub fn contains(&self, pose: Pose) -> bool {
        norm(sub(pose.position, self.capture_center)) <= self.capture_radius
    }
    pub fn region_contains(&self, pose: Pose) -> bool {
        self.target_region
            .as_ref()
            .is_none_or(|region| region.contains(pose))
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum DockingMethod {
    Mixture,
    Involution,
    /// Posterior source chart, independent weight-distributed destination.
    /// At correlation zero this is the branch-separated independent redraw.
    PosteriorInvolution,
    /// Keep the common local slots and replace the final global slot by one
    /// additional local attempt with exactly the same translation/rotation law.
    Local,
}

#[derive(Clone, Debug)]
pub struct DockingOptions {
    pub config: PathBuf,
    pub model: PathBuf,
    pub out: PathBuf,
    pub cycles: u64,
    pub sample_every: u64,
    pub method: DockingMethod,
    pub correlation: f64,
    pub resume: Option<PathBuf>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct DockingCounts {
    pub attempted: u64,
    pub numerical_nulls: u64,
    pub capture_rejected: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub region_rejected: u64,
    pub hard_rejected: u64,
    pub hard_valid: u64,
    pub accepted: u64,
    pub accepted_pose_changes: u64,
    pub gate_raw_points: u64,
}

fn is_zero(value: &u64) -> bool {
    *value == 0
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DockingCheckpoint {
    pub protocol: String,
    pub config_sha256: String,
    pub model_sha256: String,
    pub shape_sha256: String,
    pub method: DockingMethod,
    pub correlation: f64,
    pub completed_cycles: u64,
    pub pose: Pose,
    pub counts: [DockingCounts; 2],
}

/// Frozen atlas controls with a separately labeled uniform branch for transport.
pub struct DockingProposal {
    model: FrozenRelativePoseProposal,
    map: FixedBasinInvolution,
    components: Vec<FrozenRelativePoseProposal>,
    branches: Vec<RelativePoseBranch>,
    log_component_weights: Vec<f64>,
    method: DockingMethod,
    correlation: f64,
    center: Vec3,
    cube: Vec3,
    proposal_anchor_index: Option<usize>,
}
impl DockingProposal {
    pub fn new(
        model: FrozenRelativePoseProposal,
        method: DockingMethod,
        correlation: f64,
        capture_center: Vec3,
    ) -> Result<Self> {
        ensure!(model.component_count() > 0, "Need a nonempty frozen atlas");
        // The stored atlas contains each base Gaussian once. The numerical map
        // has one chart per virtual branch; its reciprocal wrappers preserve
        // translation volume times rotational Haar measure exactly.
        let branches = model.virtual_branches();
        let weights: Vec<_> = branches.iter().map(|b| b.weight).collect();
        let base_parameters = model.component_parameters();
        let parameters: Vec<_> = branches
            .iter()
            .map(|b| {
                let mut p = base_parameters[b.component_index].clone();
                p.weight = b.weight;
                p
            })
            .collect();
        let mut pairs = Vec::new();
        for a in 0..weights.len() {
            for b in a..weights.len() {
                pairs.push(BasinPair {
                    first: a,
                    second: b,
                    weight: weights[a] * weights[b] * if a == b { 1. } else { 2. },
                });
            }
        }
        let map = FixedBasinInvolution::new(
            parameters.clone(),
            model.angular_length(),
            correlation,
            pairs,
        )?;
        let components = parameters
            .into_iter()
            .map(|mut p| {
                p.weight = 1.;
                FrozenRelativePoseProposal::from_components_open(
                    vec![p],
                    model.angular_length(),
                    model.box_lengths(),
                    model.uniform_weight(),
                    model.shape_sha256(),
                    model.shape_sha256(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            cube: model.box_lengths(),
            model,
            map,
            components,
            branches,
            log_component_weights: weights.iter().map(|w| w.ln()).collect(),
            method,
            correlation,
            center: capture_center,
            proposal_anchor_index: None,
        })
    }
    pub fn with_anchor_index(mut self, index: Option<usize>) -> Self {
        self.proposal_anchor_index = index;
        self
    }
    /// Apply a virtual-chart trace in one fixed anchor-relative frame.
    /// The inverse trace swaps virtual labels as well as Gaussian noise. Each
    /// reciprocal wrapper has unit absolute Jacobian in physical pose measure.
    pub fn apply_relative_trace(&self, relative: Pose, trace: &BasinTrace) -> Result<BasinStep> {
        relative.validate()?;
        let source = self
            .branches
            .get(trace.source)
            .context("Invalid virtual source chart")?;
        let target = self
            .branches
            .get(trace.target)
            .context("Invalid virtual target chart")?;
        let input = if source.inverted {
            invert_relative_pose(relative)
        } else {
            relative
        };
        let mut step = self.map.apply(input, trace)?;
        if target.inverted {
            step.pose = invert_relative_pose(step.pose);
        }
        step.pose.validate()?;
        Ok(step)
    }
    fn component_log_density(&self, index: usize, relative: Pose) -> Result<f64> {
        let pose = if self.branches[index].inverted {
            invert_relative_pose(relative)
        } else {
            relative
        };
        self.components[index].relative_log_density(pose.position, rotation(pose.orientation))
    }
    pub fn propose(
        &self,
        rng: &mut StdRng,
        old: Pose,
        fixed: &[Pose],
    ) -> Result<(Option<Pose>, Value)> {
        ensure!(!fixed.is_empty(), "No proposal anchor");
        ensure!(
            self.method != DockingMethod::Local,
            "Local control has no atlas proposal"
        );
        if let Some(index) = self.proposal_anchor_index {
            ensure!(index < fixed.len(), "Proposal anchor index out of range");
        }
        if self.method == DockingMethod::Mixture {
            let centered = |p: Pose| Pose {
                position: sub(p.position, self.center),
                orientation: p.orientation,
            };
            let mut poses = vec![centered(old)];
            if let Some(index) = self.proposal_anchor_index {
                poses.push(centered(fixed[index]));
            } else {
                poses.extend(fixed.iter().copied().map(centered));
            }
            let result = self.model.propose(rng, &poses, 0)?;
            let candidate = result.candidate.map(|p| Pose {
                position: if self.model.is_periodic() {
                    wrap(add(p.position, self.center), self.cube)
                } else {
                    add(p.position, self.center)
                },
                orientation: p.orientation,
            });
            let mut log = serde_json::to_value(result)?;
            // Anchor indices in this module index fixed_poses directly.
            log["anchor_index"] = json!(
                self.proposal_anchor_index
                    .map(|index| index as u64)
                    .unwrap_or(log["anchor_index"].as_u64().unwrap() - 1)
            );
            return Ok((candidate, log));
        }
        let j = self
            .proposal_anchor_index
            .unwrap_or_else(|| rng.random_range(0..fixed.len()));
        if rng.random::<f64>() < self.model.uniform_weight() {
            let mut p = uniform_pose(rng, self.cube);
            p.position = if self.model.is_periodic() {
                wrap(add(p.position, self.center), self.cube)
            } else {
                add(sub(p.position, scale(self.cube, 0.5)), self.center)
            };
            return Ok((
                Some(p),
                json!({"branch":"uniform","anchor_index":j,"log_reverse_forward":0.}),
            ));
        }
        let anchor = fixed[j];
        let ar = rotation(anchor.orientation);
        // A periodic chart uses exactly one world-frame relative image. It is
        // selected before rotation into the anchor body frame; an anisotropic
        // world box must not be imposed in that rotating frame.
        let displacement = sub(old.position, anchor.position);
        let displacement = if self.model.is_periodic() {
            minimum_image(displacement, self.cube)
        } else {
            displacement
        };
        let relative = Pose {
            position: matvec(transpose(ar), displacement),
            orientation: quaternion(matmul(transpose(ar), rotation(old.orientation))),
        };
        let posterior = self.method == DockingMethod::PosteriorInvolution;
        let source_logs = if posterior {
            (0..self.branches.len())
                .map(|index| self.component_log_density(index, relative))
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let trace = if posterior {
            let probabilities: Vec<_> = source_logs
                .iter()
                .zip(&self.log_component_weights)
                .map(|(q, w)| q + w)
                .collect();
            let Some(source) = draw_log_category(rng, &probabilities)? else {
                return Ok((
                    None,
                    json!({"branch":"involution","anchor_index":j,
                    "source_law":"posterior","null_reason":"No finite Gaussian source density"}),
                ));
            };
            BasinTrace {
                source,
                target: draw_log_category(rng, &self.log_component_weights)?
                    .context("No destination component")?,
                noise: std::array::from_fn(|_| StandardNormal.sample(rng)),
            }
        } else {
            self.map.draw_trace(rng)
        };
        let result = self.apply_relative_trace(relative, &trace);
        let step = match result {
            Ok(step) => step,
            Err(error) => {
                return Ok((
                    None,
                    json!({"branch":"involution","anchor_index":j,
                "trace":trace,"null_reason":error.to_string()}),
                ));
            }
        };
        // These are virtual indices: the same base Gaussian with opposite
        // inversion labels is a nonidentity reciprocal move, including c=1.
        let identity = self.correlation == 1. && trace.source == trace.target;
        let proposed_displacement = matvec(ar, step.pose.position);
        // Restrict both ends of the invertible open-chart map to the same
        // canonical image domain. Exterior destinations are self-loops, never
        // redrawn or wrapped into a different relative image. Thus no truncated
        // Gaussian normalization or image sum enters the density correction.
        // An exact identity retains the existing point, including at a face.
        if self.model.is_periodic()
            && !identity
            && !(0..3).all(|k| {
                proposed_displacement[k] >= -0.5 * self.cube[k]
                    && proposed_displacement[k] < 0.5 * self.cube[k]
            })
        {
            return Ok((
                None,
                json!({"branch":"involution","anchor_index":j,"trace":trace,
                    "step":step,"source_law":if posterior {"posterior"} else {"static_pair"},
                    "null_reason":"outside_unique_image_cube"}),
            ));
        }
        let proposed = if identity {
            if self.model.is_periodic() {
                Pose {
                    position: wrap(old.position, self.cube),
                    ..old
                }
            } else {
                old
            }
        } else {
            Pose {
                position: if self.model.is_periodic() {
                    wrap(add(anchor.position, proposed_displacement), self.cube)
                } else {
                    add(anchor.position, proposed_displacement)
                },
                orientation: quaternion(matmul(ar, rotation(step.pose.orientation))),
            }
        };
        proposed.validate()?;
        let source = self.component_log_density(trace.source, relative)?;
        let target = self.component_log_density(trace.target, step.pose)?;
        let full_old = self
            .model
            .relative_log_density(relative.position, rotation(relative.orientation))?;
        let full_new = self
            .model
            .relative_log_density(step.pose.position, rotation(step.pose.orientation))?;
        let mut label_correction = None;
        let mut source_probability = None;
        let mut inverse_source_probability = None;
        let mut expanded_correction = None;
        let correction = if posterior {
            // The selected-source responsibility and the independent destination
            // law cancel the component-specific Gaussian/Jacobian correction.
            // G excludes the separate uniform branch. The spectator anchor is
            // retained as a state-independent move label in both directions.
            ensure!(
                full_old.is_finite() && full_new.is_finite(),
                "Nonfinite posterior mixture density"
            );
            let forward_source = self.log_component_weights[trace.source] + source - full_old;
            let reverse_source = self.log_component_weights[trace.target] + target - full_new;
            let labels = reverse_source + self.log_component_weights[trace.source]
                - forward_source
                - self.log_component_weights[trace.target];
            source_probability = Some(forward_source);
            inverse_source_probability = Some(reverse_source);
            label_correction = Some(labels);
            expanded_correction = Some(step.log_correction + labels);
            full_old - full_new
        } else {
            step.log_correction
        };
        let mut info = json!({"branch":"involution","anchor_index":j,"trace":trace,"step":step,"identity":identity,
            "source_law":if posterior {"posterior"} else {"static_pair"},
            "selected_source_log_density":source,"selected_target_log_density":target,
            "full_old_gaussian_log_density":full_old,"full_new_gaussian_log_density":full_new,
            "source_log_probability":source_probability,"inverse_source_log_probability":inverse_source_probability,
            "label_log_reverse_forward":label_correction,"expanded_log_reverse_forward":expanded_correction,
            "log_reverse_forward":if identity {0.} else {correction}});
        if self.model.has_reciprocal_components() {
            let source = self.branches[trace.source];
            let target = self.branches[trace.target];
            info["source_component_index"] = json!(source.component_index);
            info["target_component_index"] = json!(target.component_index);
            info["source_inverted"] = json!(source.inverted);
            info["target_inverted"] = json!(target.inverted);
        }
        Ok((Some(proposed), info))
    }

    /// Fixed single-member charts for derived member-chart mixtures:
    /// the virtual-branch map, inversion labels and log branch weights.
    pub fn member_chart_parts(&self) -> (&FixedBasinInvolution, Vec<bool>, &[f64]) {
        (
            &self.map,
            self.branches.iter().map(|b| b.inverted).collect(),
            &self.log_component_weights,
        )
    }
    pub fn angular_length(&self) -> f64 {
        self.model.angular_length()
    }
    pub fn correlation(&self) -> f64 {
        self.correlation
    }

    pub fn method(&self) -> DockingMethod {
        self.method
    }
    pub fn is_periodic(&self) -> bool {
        self.model.is_periodic()
    }

    /// Joint label log weights w_b N_b(g_a^{-1} g_i) / (|S| |A|), flattened
    /// member-major, then anchor, then virtual branch. Their log-sum is the
    /// normalized member-chart density log G_S of the carried handle pose.
    fn member_label_logs(&self, members: &[Pose], pool: &[Pose]) -> Result<Vec<f64>> {
        let offset = ((members.len() * pool.len()) as f64).ln();
        let mut out = Vec::with_capacity(members.len() * pool.len() * self.branches.len());
        for &m in members {
            for &a in pool {
                let relative = anchor_relative(a, m);
                for (b, w) in self.log_component_weights.iter().enumerate() {
                    out.push(w + self.component_log_density(b, relative)? - offset);
                }
            }
        }
        Ok(out)
    }

    /// log w_b N_b(g_a^{-1} g) for every virtual branch b.
    pub fn branch_log_densities(&self, pose: Pose, anchor: Pose) -> Result<Vec<f64>> {
        let relative = anchor_relative(anchor, pose);
        self.log_component_weights
            .iter()
            .enumerate()
            .map(|(b, w)| Ok(w + self.component_log_density(b, relative)?))
            .collect()
    }

    /// Learned member-chart density of a rigid subset, excluding the uniform
    /// branch. Right-multiplication by a fixed internal offset preserves
    /// translation volume times Haar measure, so each term is a normalized
    /// density of the handle pose; the average over members and anchors is too.
    /// It is independent of which member serves as handle.
    pub fn members_log_density(&self, members: &[Pose], pool: &[Pose]) -> Result<f64> {
        ensure!(
            !members.is_empty() && !pool.is_empty(),
            "Member charts need members and anchors"
        );
        Ok(log_sum(&self.member_label_logs(members, pool)?))
    }

    /// State-independent weight of the defensive uniform member proposal.
    /// Callers may choose this branch before selecting a state-dependent
    /// primary anchor; the uniform branch has no anchor-selection factor.
    pub fn member_uniform_weight(&self) -> f64 {
        self.model.uniform_weight()
    }

    /// Draw the defensive uniform branch without choosing an anchor or drawing
    /// a branch coin. Its rigid-subset forward/reverse correction is zero.
    pub fn draw_member_uniform(&self, rng: &mut StdRng) -> Result<(Option<Pose>, Value)> {
        ensure!(
            self.method == DockingMethod::PosteriorInvolution && !self.model.is_periodic(),
            "Member charts need a nonperiodic posterior-involution model"
        );
        let mut p = uniform_pose(rng, self.cube);
        p.position = add(sub(p.position, scale(self.cube, 0.5)), self.center);
        Ok((
            Some(p),
            json!({"branch":"uniform","charts":"members","log_reverse_forward":0.}),
        ))
    }

    /// Posterior-source involution over joint (member, anchor, branch) charts.
    /// Any carried member can be the docking interface, and rotations act
    /// about that member instead of the handle. The anchor pool must be fixed
    /// by spectators alone, so it is identical at both endpoints. Returns the
    /// proposed handle pose. Nonperiodic frozen posterior models only.
    pub fn propose_members(
        &self,
        rng: &mut StdRng,
        members: &[Pose],
        handle: usize,
        pool: &[Pose],
    ) -> Result<(Option<Pose>, Value)> {
        ensure!(
            self.method == DockingMethod::PosteriorInvolution && !self.model.is_periodic(),
            "Member charts need a nonperiodic posterior-involution model"
        );
        ensure!(
            handle < members.len() && !pool.is_empty(),
            "Invalid member-chart handle or empty anchor pool"
        );
        if rng.random::<f64>() < self.member_uniform_weight() {
            return self.draw_member_uniform(rng);
        }
        self.propose_members_learned(rng, members, handle, pool)
    }

    /// Draw only the learned member-chart branch, without a branch coin.
    /// A caller selecting its primary anchor with a state-dependent law must
    /// separately include that same primary label's reverse/forward probability.
    /// The spectator-only pool is frozen across the paired forward/reverse move.
    pub fn propose_members_learned(
        &self,
        rng: &mut StdRng,
        members: &[Pose],
        handle: usize,
        pool: &[Pose],
    ) -> Result<(Option<Pose>, Value)> {
        ensure!(
            self.method == DockingMethod::PosteriorInvolution && !self.model.is_periodic(),
            "Member charts need a nonperiodic posterior-involution model"
        );
        ensure!(
            handle < members.len() && !pool.is_empty(),
            "Invalid member-chart handle or empty anchor pool"
        );
        let old_logs = self.member_label_logs(members, pool)?;
        let Some(source) = draw_log_category(rng, &old_logs)? else {
            return Ok((
                None,
                json!({"branch":"involution","charts":"members",
                    "null_reason":"No finite Gaussian source density"}),
            ));
        };
        let nb = self.branches.len();
        let na = pool.len();
        let source = MemberLabel {
            member: source / (na * nb),
            anchor: source / nb % na,
            branch: source % nb,
        };
        let target_member = rng.random_range(0..members.len());
        let target_anchor = rng.random_range(0..na);
        let target = MemberLabel {
            member: target_member,
            anchor: target_anchor,
            branch: draw_log_category(rng, &self.log_component_weights)?
                .context("No destination component")?,
        };
        let noise = std::array::from_fn(|_| StandardNormal.sample(rng));
        let labels = json!({"source":source,"target":target});
        match self.apply_member_trace(members, handle, pool, source, target, noise) {
            Ok(step) => {
                let mut info = serde_json::to_value(&step)?;
                info["branch"] = json!("involution");
                info["charts"] = json!("members");
                info["source_law"] = json!("posterior");
                info["labels"] = labels;
                Ok((Some(step.handle), info))
            }
            Err(error) => Ok((
                None,
                json!({"branch":"involution","charts":"members","labels":labels,
                    "noise":noise,"null_reason":error.to_string()}),
            )),
        }
    }

    /// Deterministic member-chart map for given labels and noise. The inverse
    /// uses the swapped labels and `step.inverse_trace.noise` from the carried
    /// endpoint with the same pool.
    pub fn apply_member_trace(
        &self,
        members: &[Pose],
        handle: usize,
        pool: &[Pose],
        source: MemberLabel,
        target: MemberLabel,
        noise: [f64; 6],
    ) -> Result<MemberStep> {
        let nb = self.branches.len();
        let na = pool.len();
        let index = |l: MemberLabel| (l.member * na + l.anchor) * nb + l.branch;
        ensure!(
            handle < members.len()
                && [source, target]
                    .iter()
                    .all(|l| l.member < members.len() && l.anchor < na && l.branch < nb),
            "Invalid member-chart label"
        );
        let old_logs = self.member_label_logs(members, pool)?;
        let full_old = log_sum(&old_logs);
        ensure!(full_old.is_finite(), "Nonfinite member-chart density");
        let trace = BasinTrace {
            source: source.branch,
            target: target.branch,
            noise,
        };
        let relative = anchor_relative(pool[source.anchor], members[source.member]);
        let step = self.apply_relative_trace(relative, &trace)?;
        if self.correlation == 1. && source == target {
            return Ok(MemberStep {
                handle: members[handle],
                step,
                identity: true,
                full_old_member_log_density: full_old,
                full_new_member_log_density: full_old,
                label_log_reverse_forward: 0.,
                expanded_log_reverse_forward: 0.,
                log_reverse_forward: 0.,
            });
        }
        let anchor = pool[target.anchor];
        let ar = rotation(anchor.orientation);
        let moved = Pose {
            position: add(anchor.position, matvec(ar, step.pose.position)),
            orientation: quaternion(matmul(ar, rotation(step.pose.orientation))),
        };
        moved.validate()?;
        // Common rigid motion H = g_i' g_i^{-1}, applied to every member.
        let reference = members[target.member];
        let delta = matmul(
            rotation(moved.orientation),
            transpose(rotation(reference.orientation)),
        );
        let carry = |p: Pose| Pose {
            position: add(
                moved.position,
                matvec(delta, sub(p.position, reference.position)),
            ),
            orientation: quaternion(matmul(delta, rotation(p.orientation))),
        };
        let proposed = carry(members[handle]);
        proposed.validate()?;
        let new_members: Vec<_> = members.iter().map(|&p| carry(p)).collect();
        let new_logs = self.member_label_logs(&new_members, pool)?;
        let full_new = log_sum(&new_logs);
        ensure!(full_new.is_finite(), "Nonfinite member-chart density");
        // Same algebra as the single-handle posterior map: responsibilities,
        // destination law and chart Jacobian combine to log G_S(old)-log G_S(new).
        let labels = (new_logs[index(target)] - full_new) - (old_logs[index(source)] - full_old)
            + self.log_component_weights[source.branch]
            - self.log_component_weights[target.branch];
        Ok(MemberStep {
            handle: proposed,
            identity: false,
            full_old_member_log_density: full_old,
            full_new_member_log_density: full_new,
            label_log_reverse_forward: labels,
            expanded_log_reverse_forward: step.log_correction + labels,
            log_reverse_forward: full_old - full_new,
            step,
        })
    }
}

/// Joint chart label: carried member, pool anchor, virtual atlas branch.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemberLabel {
    pub member: usize,
    pub anchor: usize,
    pub branch: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct MemberStep {
    pub handle: Pose,
    pub step: BasinStep,
    pub identity: bool,
    pub full_old_member_log_density: f64,
    pub full_new_member_log_density: f64,
    pub label_log_reverse_forward: f64,
    pub expanded_log_reverse_forward: f64,
    pub log_reverse_forward: f64,
}

/// Body-frame pose of `pose` relative to `anchor`, without periodic images.
fn anchor_relative(anchor: Pose, pose: Pose) -> Pose {
    let ar = transpose(rotation(anchor.orientation));
    Pose {
        position: matvec(ar, sub(pose.position, anchor.position)),
        orientation: quaternion(matmul(ar, rotation(pose.orientation))),
    }
}

fn log_sum(values: &[f64]) -> f64 {
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !maximum.is_finite() {
        return maximum;
    }
    maximum + values.iter().map(|v| (v - maximum).exp()).sum::<f64>().ln()
}

/// Gumbel-max avoids exponentiating tiny component responsibilities or silently
/// imposing a responsibility cutoff. The finite RNG still has finite precision.
pub(crate) fn draw_log_category(rng: &mut StdRng, log_weights: &[f64]) -> Result<Option<usize>> {
    let mut best = f64::NEG_INFINITY;
    let mut index = None;
    for (i, &weight) in log_weights.iter().enumerate() {
        ensure!(
            weight.is_finite() || weight == f64::NEG_INFINITY,
            "Invalid log category weight"
        );
        if weight == f64::NEG_INFINITY {
            continue;
        }
        let u: f64 = Open01.sample(rng);
        let score = weight - (-u.ln()).ln();
        if score > best {
            best = score;
            index = Some(i);
        }
    }
    Ok(index)
}

fn stream(seed: u64, cycle: u64, attempt: usize, kind: &str) -> StdRng {
    let mut h = Sha256::new();
    h.update(b"tetramer-docking-rng-v1");
    h.update(seed.to_le_bytes());
    h.update(cycle.to_le_bytes());
    h.update((attempt as u64).to_le_bytes());
    h.update(kind.as_bytes());
    StdRng::from_seed(h.finalize().into())
}
fn jsonline(writer: &mut impl Write, value: &Value) -> Result<()> {
    serde_json::to_writer(&mut *writer, value)?;
    writer.write_all(b"\n")?;
    Ok(())
}
pub(crate) fn local(rng: &mut StdRng, old: Pose, cfg: &DockingConfig) -> Pose {
    if rng.random::<f64>() < cfg.rotation_probability {
        let step = cfg.rotation_steps_deg[rng.random_range(0..cfg.rotation_steps_deg.len())]
            .to_radians()
            * 0.5;
        let c = std::array::from_fn(|_| {
            let v: f64 = StandardNormal.sample(rng);
            v * step
        });
        Pose {
            orientation: quaternion(matmul(cayley(c), rotation(old.orientation))),
            ..old
        }
    } else {
        let step = cfg.translation_steps[rng.random_range(0..cfg.translation_steps.len())];
        let d = std::array::from_fn(|_| {
            let v: f64 = StandardNormal.sample(rng);
            v * step
        });
        Pose {
            position: add(old.position, d),
            ..old
        }
    }
}

pub fn run(options: DockingOptions) -> Result<Value> {
    ensure!(
        options.cycles > 0 && options.sample_every > 0,
        "Run lengths must be positive"
    );
    ensure!(
        options.correlation.is_finite() && (-1. ..=1.).contains(&options.correlation),
        "Invalid correlation"
    );
    let raw = fs::read(&options.config)?;
    let mut cfg: DockingConfig = serde_json::from_slice(&raw)?;
    cfg.validate()?;
    if cfg.shape.is_relative() {
        cfg.shape = options.config.parent().unwrap().join(&cfg.shape);
    }
    let shape_raw = fs::read(&cfg.shape)?;
    let shape_sha = hash_bytes(&shape_raw);
    let config_sha = hash_bytes(&raw);
    let model_raw = fs::read(&options.model)?;
    let model_sha = hash_bytes(&model_raw);
    let tree = SphereTree::new(serde_json::from_slice::<Shape>(&shape_raw)?)?;
    // Contact is an observable only. Overlap of the two inflated sphere unions
    // distinguishes competing adsorbed poses from locally unbound excursions.
    let mut contact_shape = tree.shape.clone();
    for atom in &mut contact_shape.atoms {
        atom.radius += cfg.depletant_radius;
    }
    let contact_tree = SphereTree::new(contact_shape)?;
    let env = Environment {
        tree: &tree,
        fixed: cfg.fixed_poses.iter().copied().map(Placed::new).collect(),
        labels: (0..cfg.fixed_poses.len()).map(|i| (i, [0; 3])).collect(),
        rd: cfg.depletant_radius,
    };
    for i in 0..env.fixed.len() {
        for j in 0..i {
            ensure!(
                !tree.overlaps(&env.fixed[i], &env.fixed[j]),
                "Fixed neighbors overlap"
            );
        }
    }
    let model = FrozenRelativePoseProposal::from_json_str_open(
        std::str::from_utf8(&model_raw)?,
        [2. * cfg.capture_radius; 3],
        cfg.uniform_probability,
        &shape_sha,
    )?;
    let proposer = DockingProposal::new(
        model,
        options.method,
        options.correlation,
        cfg.capture_center,
    )?
    .with_anchor_index(cfg.proposal_anchor_index);
    let mut pose = cfg.initial_pose;
    let mut completed = 0;
    let mut counts: [DockingCounts; 2] = Default::default();
    if let Some(path) = &options.resume {
        let saved: DockingCheckpoint = serde_json::from_slice(&fs::read(path)?)?;
        ensure!(
            saved.protocol == "tetramer-docking-rng-v1"
                && saved.config_sha256 == config_sha
                && saved.model_sha256 == model_sha
                && saved.shape_sha256 == shape_sha
                && saved.method == options.method
                && saved.correlation == options.correlation,
            "Checkpoint inputs/method mismatch"
        );
        pose = saved.pose;
        completed = saved.completed_cycles;
        counts = saved.counts;
    }
    pose.validate()?;
    ensure!(
        completed < options.cycles,
        "Requested cycles already complete"
    );
    ensure!(
        cfg.contains(pose) && env.hard_valid(pose),
        "Initial/checkpoint pose violates target support"
    );
    ensure!(
        cfg.region_contains(pose),
        "Initial/checkpoint pose violates conditional target region"
    );
    if options.out.exists() {
        ensure!(
            fs::read_dir(&options.out)?.next().is_none(),
            "Output directory must be empty"
        );
    }
    fs::create_dir_all(options.out.join("provenance"))?;
    fs::write(options.out.join("provenance/input-config.json"), &raw)?;
    fs::write(options.out.join("provenance/shape.json"), &shape_raw)?;
    fs::write(options.out.join("provenance/model.json"), &model_raw)?;
    let source = include_str!(concat!(env!("OUT_DIR"), "/source-bundle.json"));
    fs::write(options.out.join("provenance/source-bundle.json"), source)?;
    save(&options.out.join("config.json"), &cfg)?;
    let mut manifest = json!({"schema":1,"config_sha256":config_sha,"shape_sha256":shape_sha,
        "model_sha256":model_sha,"source_bundle_sha256":hash_bytes(source.as_bytes()),"executable_sha256":hash_file(&std::env::current_exe()?)?,
        "method":options.method,"correlation":options.correlation,"cycles":options.cycles,"sample_every":options.sample_every,
        "rng_protocol":"tetramer-docking-rng-v1","physical_target":"hard(x,S) indicator[|x-center|<=R] exp[z |E(x) intersect union E(S)|]",
        "pair_law":"static directed w_a*w_b including self; unordered offdiagonal 2*w_a*w_b followed by fair direction",
        "scope":if cfg.target_region.is_some() {
            "One moving rigid body conditional on the explicit original-q window, all fixed neighbors and capture; not full-target native escape or assembly"
        } else {"One moving rigid body, fixed neighbors and capture ball; immutable full atlas; native labels used only by external diagnostics"}});
    if let Some(region) = &cfg.target_region {
        manifest["target_region"] = json!(region);
        manifest["physical_target"] = json!(
            "hard(x,S) indicator[|x-center|<=R] indicator[original q in configured window] exp[z |E(x) intersect union E(S)|]"
        );
    }
    if let Some(index) = cfg.proposal_anchor_index {
        manifest["proposal_anchor_index"] = json!(index);
    }
    if options.method == DockingMethod::Local {
        manifest["pair_law"] = json!(
            "No chart pairs: all local_attempts_per_cycle+1 slots use the unchanged local proposal law"
        );
    } else if cfg.target_region.is_some() && options.method == DockingMethod::PosteriorInvolution {
        manifest["pair_law"] = json!(
            "Posterior source w_a*q_a(x)/G(x), independent destination w_b, including self; separate fixed-probability uniform branch"
        );
    }
    save(&options.out.join("manifest.json"), &manifest)?;
    let start = Instant::now();
    let cpu_start = cpu_seconds();
    let initial_cycle = completed;
    let initial_counts = counts.clone();
    let contact = |p: Pose| {
        let moving = Placed::new(p);
        env.fixed.iter().any(|f| contact_tree.overlaps(&moving, f))
    };
    let mut depletion_contact = contact(pose);
    let mut diagnostic_cpu = 0.;
    let mut proposal_cpu = 0.;
    let mut geometry_cpu = 0.;
    let mut gate_cpu = 0.;
    let mut frames = BufWriter::new(File::create(options.out.join("trajectory.jsonl"))?);
    let mut moves = BufWriter::new(File::create(options.out.join("moves.jsonl"))?);
    jsonline(
        &mut frames,
        &json!({"cycle":completed,"pose":pose,"counts":counts,"depletion_contact":depletion_contact,"sampler_cpu_seconds":0.}),
    )?;
    for cycle in completed + 1..=options.cycles {
        for attempt in 0..=cfg.local_attempts_per_cycle {
            let is_global =
                attempt == cfg.local_attempts_per_cycle && options.method != DockingMethod::Local;
            let kind = if is_global { "global" } else { "local" };
            let index = usize::from(is_global);
            counts[index].attempted += 1;
            let old = pose;
            let contact_before = depletion_contact;
            let before = cpu_seconds();
            let mut rng = stream(cfg.seed, cycle, attempt, "proposal");
            let (candidate, info) = if is_global {
                proposer.propose(&mut rng, old, &cfg.fixed_poses)?
            } else {
                (
                    Some(local(&mut rng, old, &cfg)),
                    json!({"branch":"local","log_reverse_forward":0.}),
                )
            };
            proposal_cpu += cpu_seconds() - before;
            let mut capture_valid = false;
            let mut region_valid = false;
            let mut hard_valid = false;
            let mut accepted = false;
            let mut gate = None;
            let mut log_acceptance = None;
            if let Some(new) = candidate {
                new.validate()?;
                let before = cpu_seconds();
                capture_valid = cfg.contains(new);
                region_valid = cfg.region_contains(new);
                if capture_valid && region_valid {
                    hard_valid = env.hard_valid(new);
                }
                geometry_cpu += cpu_seconds() - before;
                if !capture_valid {
                    counts[index].capture_rejected += 1;
                } else if !region_valid {
                    counts[index].region_rejected += 1;
                } else if !hard_valid {
                    counts[index].hard_rejected += 1;
                } else {
                    counts[index].hard_valid += 1;
                    let correction = info["log_reverse_forward"]
                        .as_f64()
                        .context("Nonfinite/missing proposal correction")?;
                    let before = cpu_seconds();
                    let intensity = if cfg.reservoir_density > 0. {
                        cfg.poisson_lambda_ratio * cfg.reservoir_density
                    } else {
                        1.
                    };
                    let sampled = depletion::sample(
                        &mut stream(cfg.seed, cycle, attempt, "gate"),
                        &env,
                        old,
                        new,
                        intensity,
                        cfg.reservoir_density,
                        cfg.endpoint_gate,
                    )?;
                    gate_cpu += cpu_seconds() - before;
                    counts[index].gate_raw_points += sampled.raw_points;
                    let alpha = (correction + sampled.log_weight).min(0.);
                    ensure!(alpha.is_finite(), "Nonfinite acceptance");
                    accepted = stream(cfg.seed, cycle, attempt, "accept")
                        .random::<f64>()
                        .ln()
                        < alpha;
                    if accepted {
                        pose = new;
                        counts[index].accepted += 1;
                        let matrix_difference = rotation(old.orientation)
                            .into_iter()
                            .flatten()
                            .zip(rotation(new.orientation).into_iter().flatten())
                            .map(|(a, b)| (a - b).abs())
                            .fold(0., f64::max);
                        if norm(sub(old.position, new.position)) > 1e-10
                            || matrix_difference > 1e-12
                        {
                            counts[index].accepted_pose_changes += 1;
                        }
                        let before = cpu_seconds();
                        depletion_contact = contact(pose);
                        diagnostic_cpu += cpu_seconds() - before;
                    }
                    gate = Some(sampled);
                    log_acceptance = Some(alpha);
                }
            } else {
                counts[index].numerical_nulls += 1;
            }
            let mut record = json!({"cycle":cycle,"attempt":attempt,"kind":kind,"branch":info["branch"],
                "old_pose":old,"proposed_pose":candidate,"retained_pose":pose,"proposal":info,"accepted":accepted,
                "capture_valid":capture_valid,"hard_valid":hard_valid,"gate":gate,"log_acceptance":log_acceptance,
                "depletion_contact_before":contact_before,"depletion_contact":depletion_contact,
                "sampler_cpu_seconds":cpu_seconds()-cpu_start});
            if let Some(region) = &cfg.target_region {
                record["region_valid"] = json!(region_valid);
                record["old_q"] = json!(region.metric.q(old));
                record["proposed_q"] = json!(candidate.map(|p| region.metric.q(p)));
                record["retained_q"] = json!(region.metric.q(pose));
            }
            jsonline(&mut moves, &record)?;
        }
        completed = cycle;
        if cycle % options.sample_every == 0 || cycle == options.cycles {
            jsonline(
                &mut frames,
                &json!({"cycle":cycle,"pose":pose,"counts":counts,"depletion_contact":depletion_contact,"sampler_cpu_seconds":cpu_seconds()-cpu_start}),
            )?;
            frames.flush()?;
        }
        if cycle % 100 == 0 || cycle == options.cycles {
            moves.flush()?;
            save(
                &options.out.join("checkpoint.json"),
                &DockingCheckpoint {
                    protocol: "tetramer-docking-rng-v1".into(),
                    config_sha256: config_sha.clone(),
                    shape_sha256: shape_sha.clone(),
                    model_sha256: model_sha.clone(),
                    method: options.method,
                    correlation: options.correlation,
                    completed_cycles: completed,
                    pose,
                    counts: counts.clone(),
                },
            )?;
            save(
                &options.out.join("progress.json"),
                &json!({"complete":cycle==options.cycles,"completed_cycles":cycle,
                "requested_cycles":options.cycles,"counts":counts,"sampler_cpu_seconds":cpu_seconds()-cpu_start}),
            )?;
        }
    }
    let summary = json!({"complete":true,"completed_cycles":completed,"initial_cycle":initial_cycle,"counts":counts,
        "initial_counts":initial_counts,"method":options.method,"correlation":options.correlation,
        "sampler_cpu_seconds":cpu_seconds()-cpu_start,"wall_seconds":start.elapsed().as_secs_f64(),
        "cost":{"proposal_cpu_seconds":proposal_cpu,"geometry_cpu_seconds":geometry_cpu,"gate_cpu_seconds":gate_cpu,"diagnostic_cpu_seconds":diagnostic_cpu},
        "config_sha256":config_sha,"shape_sha256":shape_sha,"model_sha256":model_sha});
    save(&options.out.join("summary.json"), &summary)?;
    Ok(summary)
}
