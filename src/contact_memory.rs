//! Reversible memory of pair poses, independent of the physical production fluid.
//!
//! Fixed M slots target a product of identical anchored-pair AO distributions:
//! rho(A) ∝ product_m [H_pair(A_m) 1_(|t_m|<R) exp(-z |E_0 union E_A_m|)].
//! Slots do not interact. The finite support makes the law proper; its unknown
//! normalization is independent of production X and never enters its marginal.
//! Radial initialization is a preparation, not an equilibrium draw. Subsequent
//! moves explore the entire allowed ball, including internal contact pockets.
use crate::{
    depletion::{self, GateOptions, GateResult},
    geometry::{Environment, Placed, SphereTree},
    math::*,
};
use anyhow::{Result, ensure};
use rand::{RngExt, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};

fn slots() -> usize {
    16
}
fn attempts() -> usize {
    2
}
fn lambda() -> f64 {
    16.
}
fn global_probability() -> f64 {
    0.1
}
fn local_translation() -> f64 {
    0.3
}
fn angle() -> f64 {
    3.
}
fn mass() -> f64 {
    0.5
}
fn translation() -> f64 {
    1.
}
fn gap() -> f64 {
    0.25
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryConfig {
    #[serde(default = "slots")]
    pub slots: usize,
    #[serde(default = "attempts")]
    pub attempts_per_sweep: usize,
    #[serde(default)]
    pub radius: Option<f64>,
    #[serde(default)]
    pub depletant_radius: Option<f64>,
    #[serde(default)]
    pub depletant_activity: Option<f64>,
    #[serde(default = "lambda")]
    pub lambda_ratio: f64,
    #[serde(default = "global_probability")]
    pub global_probability: f64,
    #[serde(default = "local_translation", rename = "local_translation_std_A")]
    pub local_translation_std_a: f64,
    #[serde(default = "angle")]
    pub local_small_angle_std_degrees: f64,
    #[serde(default = "mass")]
    pub proposal_mass: f64,
    #[serde(default = "translation", rename = "proposal_translation_std_A")]
    pub proposal_translation_std_a: f64,
    #[serde(default = "angle")]
    pub proposal_small_angle_std_degrees: f64,
    #[serde(default = "gap")]
    pub initial_gap: f64,
}
impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            slots: slots(),
            attempts_per_sweep: attempts(),
            radius: None,
            depletant_radius: None,
            depletant_activity: None,
            lambda_ratio: lambda(),
            global_probability: global_probability(),
            local_translation_std_a: local_translation(),
            local_small_angle_std_degrees: angle(),
            proposal_mass: mass(),
            proposal_translation_std_a: translation(),
            proposal_small_angle_std_degrees: angle(),
            initial_gap: gap(),
        }
    }
}
impl MemoryConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.slots > 0 && self.slots <= 1024 && self.attempts_per_sweep <= 1024,
            "Invalid contact-memory size or update budget"
        );
        ensure!(
            self.radius
                .is_none_or(|x| x.is_finite() && x > 0. && (x * x).is_finite()),
            "Invalid contact-memory radius"
        );
        ensure!(
            [self.depletant_radius, self.depletant_activity]
                .iter()
                .all(|v| v.is_none_or(|x| x.is_finite() && x >= 0.)),
            "Invalid contact-memory bath"
        );
        ensure!(
            self.lambda_ratio.is_finite() && self.lambda_ratio > 0.,
            "Invalid contact-memory lambda ratio"
        );
        ensure!(
            self.global_probability.is_finite() && (0. ..=1.).contains(&self.global_probability),
            "Invalid contact-memory global probability"
        );
        ensure!(
            [
                self.local_translation_std_a,
                self.local_small_angle_std_degrees,
                self.initial_gap
            ]
            .iter()
            .all(|x| x.is_finite() && *x >= 0.),
            "Invalid contact-memory local step or preparation gap"
        );
        ensure!(
            self.proposal_mass.is_finite() && self.proposal_mass > 0. && self.proposal_mass < 1.,
            "Contact-memory proposal mass must lie strictly between zero and one"
        );
        ensure!(
            [
                self.proposal_translation_std_a,
                self.proposal_small_angle_std_degrees
            ]
            .iter()
            .all(|x| x.is_finite() && *x > 0.),
            "Contact-memory proposal widths must be positive finite"
        );
        Ok(())
    }
    pub fn resolve(
        &self,
        tree: &SphereTree,
        production_rd: f64,
        production_activity: f64,
    ) -> Result<ResolvedMemoryConfig> {
        self.validate()?;
        ensure!(
            production_rd.is_finite()
                && production_rd >= 0.
                && production_activity.is_finite()
                && production_activity >= 0.,
            "Invalid production bath defaults"
        );
        let rd = self.depletant_radius.unwrap_or(production_rd);
        let activity = self.depletant_activity.unwrap_or(production_activity);
        let guard = 4096. * f64::EPSILON * (1. + tree.bound + rd);
        let result = ResolvedMemoryConfig {
            slots: self.slots,
            attempts_per_sweep: self.attempts_per_sweep,
            radius: self
                .radius
                .unwrap_or(2. * (tree.bound + rd) + self.initial_gap + 2. * guard),
            depletant_radius: rd,
            depletant_activity: activity,
            lambda_ratio: self.lambda_ratio,
            global_probability: self.global_probability,
            local_translation_std_a: self.local_translation_std_a,
            local_small_angle_std_degrees: self.local_small_angle_std_degrees,
            proposal_mass: self.proposal_mass,
            proposal_translation_std_a: self.proposal_translation_std_a,
            proposal_small_angle_std_degrees: self.proposal_small_angle_std_degrees,
            initial_gap: self.initial_gap,
        };
        result.validate()?;
        Ok(result)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ResolvedMemoryConfig {
    pub slots: usize,
    pub attempts_per_sweep: usize,
    pub radius: f64,
    pub depletant_radius: f64,
    pub depletant_activity: f64,
    pub lambda_ratio: f64,
    pub global_probability: f64,
    #[serde(rename = "local_translation_std_A")]
    pub local_translation_std_a: f64,
    pub local_small_angle_std_degrees: f64,
    pub proposal_mass: f64,
    #[serde(rename = "proposal_translation_std_A")]
    pub proposal_translation_std_a: f64,
    pub proposal_small_angle_std_degrees: f64,
    pub initial_gap: f64,
}
impl ResolvedMemoryConfig {
    pub fn validate(&self) -> Result<()> {
        MemoryConfig {
            slots: self.slots,
            attempts_per_sweep: self.attempts_per_sweep,
            radius: Some(self.radius),
            depletant_radius: Some(self.depletant_radius),
            depletant_activity: Some(self.depletant_activity),
            lambda_ratio: self.lambda_ratio,
            global_probability: self.global_probability,
            local_translation_std_a: self.local_translation_std_a,
            local_small_angle_std_degrees: self.local_small_angle_std_degrees,
            proposal_mass: self.proposal_mass,
            proposal_translation_std_a: self.proposal_translation_std_a,
            proposal_small_angle_std_degrees: self.proposal_small_angle_std_degrees,
            initial_gap: self.initial_gap,
        }
        .validate()?;
        ensure!(
            self.depletant_activity == 0.
                || (self.lambda_ratio * self.depletant_activity).is_finite()
                    && self.lambda_ratio * self.depletant_activity > 0.,
            "Unrepresentable contact-memory Poisson intensity"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MemoryState {
    pub poses: Vec<Pose>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryProposalKind {
    Local,
    Global,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryMove {
    pub slot: usize,
    pub kind: MemoryProposalKind,
    pub old_pose: Pose,
    pub proposed_pose: Pose,
    pub retained_pose: Pose,
    pub outside_ball: bool,
    pub hard_valid: bool,
    pub accepted: bool,
    pub gate: Option<GateResult>,
    pub log_acceptance: Option<f64>,
}

fn identity() -> Pose {
    Pose {
        position: [0.; 3],
        orientation: [1., 0., 0., 0.],
    }
}
fn direction(rng: &mut StdRng) -> Vec3 {
    loop {
        let x: Vec3 = std::array::from_fn(|_| StandardNormal.sample(rng));
        let n = norm(x);
        if n.is_finite() && n > 0. {
            return x.map(|v| v / n);
        }
    }
}
fn haar(rng: &mut StdRng) -> [f64; 4] {
    loop {
        let q: [f64; 4] = std::array::from_fn(|_| StandardNormal.sample(rng));
        let n = q.iter().fold(0_f64, |a, x| a.hypot(*x));
        if n.is_finite() && n > 0. {
            return q.map(|v| v / n);
        }
    }
}
fn pair_environment<'a>(tree: &'a SphereTree, rd: f64) -> Environment<'a> {
    Environment {
        tree,
        fixed: vec![Placed::new(identity())],
        labels: vec![(0, [0; 3])],
        rd,
    }
}
fn inside(pose: Pose, radius: f64) -> bool {
    norm(pose.position) < radius
}

impl MemoryState {
    /// Geometry-only nonequilibrium initialization. No production poses, native
    /// templates, or accepted-history archive are consumed by this constructor.
    pub fn initialize(
        tree: &SphereTree,
        config: &ResolvedMemoryConfig,
        rng: &mut StdRng,
    ) -> Result<Self> {
        config.validate()?;
        let mut poses = Vec::with_capacity(config.slots);
        let fixed = Placed::new(identity());
        let gap =
            config.initial_gap + 4096. * f64::EPSILON * (1. + tree.bound + config.depletant_radius);
        for _ in 0..(10_000 * config.slots) {
            let q = haar(rng);
            let d = direction(rng);
            let Some(contact) = tree.outermost_radial_contact(rotation(q), d)? else {
                continue;
            };
            let pose = Pose {
                position: scale(contact.direction, contact.distance + gap),
                orientation: contact.orientation,
            };
            if !inside(pose, config.radius) {
                continue;
            }
            ensure!(
                !tree.overlaps(&Placed::new(pose), &fixed),
                "Numerical overlap in contact-memory preparation"
            );
            poses.push(pose);
            if poses.len() == config.slots {
                break;
            }
        }
        ensure!(
            poses.len() == config.slots,
            "Could not prepare all contact-memory slots: radius may be too small for radial entry; no stationary initialization is assumed"
        );
        let state = Self { poses };
        state.validate(tree, config)?;
        Ok(state)
    }

    pub fn validate(&self, tree: &SphereTree, config: &ResolvedMemoryConfig) -> Result<()> {
        config.validate()?;
        ensure!(
            self.poses.len() == config.slots,
            "Contact-memory slot count mismatch"
        );
        let fixed = Placed::new(identity());
        for (i, &pose) in self.poses.iter().enumerate() {
            pose.validate()?;
            ensure!(
                inside(pose, config.radius),
                "Contact-memory slot {i} outside support ball"
            );
            ensure!(
                !tree.overlaps(&Placed::new(pose), &fixed),
                "Contact-memory slot {i} has a hard pair overlap"
            );
        }
        Ok(())
    }

    /// One uniformly selected slot, followed by a state-independent mixture of
    /// symmetric local and uniform-volume/Haar global proposals. Endpoint
    /// rejection enforces the ball and hard cores; there are no redraw retries.
    /// Other memory slots and production particles are not physical spectators.
    pub fn update(
        &mut self,
        tree: &SphereTree,
        config: &ResolvedMemoryConfig,
        gate_options: GateOptions,
        rng: &mut StdRng,
    ) -> Result<MemoryMove> {
        config.validate()?;
        gate_options.validate()?;
        ensure!(
            self.poses.len() == config.slots,
            "Contact-memory slot count mismatch"
        );
        let slot = rng.random_range(0..self.poses.len());
        let old = self.poses[slot];
        old.validate()?;
        let env = pair_environment(tree, config.depletant_radius);
        ensure!(
            inside(old, config.radius) && env.hard_valid(old),
            "Invalid retained contact-memory pose"
        );
        let (kind, new) = if rng.random::<f64>() < config.global_probability {
            let radius = config.radius * rng.random::<f64>().cbrt();
            (
                MemoryProposalKind::Global,
                Pose {
                    position: scale(direction(rng), radius),
                    orientation: haar(rng),
                },
            )
        } else {
            let displacement: Vec3 = std::array::from_fn(|_| {
                let z: f64 = StandardNormal.sample(rng);
                config.local_translation_std_a * z
            });
            let c: Vec3 = std::array::from_fn(|_| {
                let z: f64 = StandardNormal.sample(rng);
                0.5 * config.local_small_angle_std_degrees.to_radians() * z
            });
            (
                MemoryProposalKind::Local,
                Pose {
                    position: add(old.position, displacement),
                    orientation: quaternion(matmul(cayley(c), rotation(old.orientation))),
                },
            )
        };
        new.validate()?;
        let outside_ball = !inside(new, config.radius);
        let hard_valid = !outside_ball && env.hard_valid(new);
        let mut result = MemoryMove {
            slot,
            kind,
            old_pose: old,
            proposed_pose: new,
            retained_pose: old,
            outside_ball,
            hard_valid,
            accepted: false,
            gate: None,
            log_acceptance: None,
        };
        if hard_valid {
            let lambda = if config.depletant_activity > 0. {
                config.lambda_ratio * config.depletant_activity
            } else {
                1.
            };
            let gate = depletion::sample(
                rng,
                &env,
                old,
                new,
                lambda,
                config.depletant_activity,
                gate_options,
            )?;
            let log_acceptance = gate.log_weight.min(0.);
            let accepted = rng.random::<f64>().max(f64::MIN_POSITIVE).ln() < log_acceptance;
            if accepted {
                self.poses[slot] = new;
                result.retained_pose = new;
            }
            result.accepted = accepted;
            result.gate = Some(gate);
            result.log_acceptance = Some(log_acceptance);
        }
        Ok(result)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct MemoryCounts {
    pub attempted: u64,
    pub accepted: u64,
    pub local_attempted: u64,
    pub global_attempted: u64,
    pub outside_ball: u64,
    pub hard_rejected: u64,
    pub raw_points: u64,
}
impl MemoryCounts {
    pub fn record(&mut self, m: &MemoryMove) {
        self.attempted += 1;
        self.accepted += u64::from(m.accepted);
        self.local_attempted += u64::from(m.kind == MemoryProposalKind::Local);
        self.global_attempted += u64::from(m.kind == MemoryProposalKind::Global);
        self.outside_ball += u64::from(m.outside_ball);
        self.hard_rejected += u64::from(!m.outside_ball && !m.hard_valid);
        self.raw_points += m.gate.as_ref().map_or(0, |g| g.raw_points);
    }
    pub fn since(&self, old: &Self) -> Self {
        Self {
            attempted: self.attempted - old.attempted,
            accepted: self.accepted - old.accepted,
            local_attempted: self.local_attempted - old.local_attempted,
            global_attempted: self.global_attempted - old.global_attempted,
            outside_ball: self.outside_ball - old.outside_ball,
            hard_rejected: self.hard_rejected - old.hard_rejected,
            raw_points: self.raw_points - old.raw_points,
        }
    }
}
