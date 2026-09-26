//! Native-blind offline contact optimization and exact anchored-pair refinement.
//!
//! The projected, greedy search is a heuristic, not an equilibrium trajectory.
//! Two independent endpoint-validation clouds are never consulted by search or
//! refinement. Local refinement reuses the exact contact-memory transition;
//! its finite preparation/burn budget does not establish stationarity.
use crate::{
    contact_memory::{MemoryConfig, MemoryCounts, MemoryMove, MemoryState, ResolvedMemoryConfig},
    depletion::GateOptions,
    geometry::{Environment, Placed, SphereTree},
    math::*,
    overlap_weight::OverlapEnvelope,
};
use anyhow::{Result, ensure};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DiscoveryConfig {
    pub starts: usize,
    pub search_steps: usize,
    pub search_points: usize,
    pub validation_points: usize,
    pub refine_steps: usize,
    pub burn: usize,
    pub save_every: usize,
    pub rd: f64,
    pub activity: f64,
    pub seed: u64,
    pub gap: f64,
    pub refine_translation_std_a: f64,
    pub refine_angle_std_degrees: f64,
    pub lambda_ratio: f64,
    pub envelope_max_cells: usize,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            starts: 8,
            search_steps: 128,
            search_points: 2048,
            validation_points: 8192,
            refine_steps: 256,
            burn: 64,
            save_every: 4,
            rd: 1.4,
            activity: 0.0275,
            seed: 2026092601,
            gap: 0.25,
            refine_translation_std_a: 0.2,
            refine_angle_std_degrees: 1.,
            lambda_ratio: 64.,
            envelope_max_cells: 2047,
        }
    }
}

impl DiscoveryConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.starts > 0 && self.starts <= 1024,
            "starts must lie in 1..=1024"
        );
        ensure!(
            self.search_points > 0 && self.validation_points > 0,
            "positive fixed score allocations required"
        );
        ensure!(
            self.refine_steps > self.burn
                && self.save_every > 0
                && self.save_every <= self.refine_steps - self.burn,
            "refinement must retain at least one fixed-stride post-burn sample"
        );
        ensure!(
            [
                self.rd,
                self.activity,
                self.gap,
                self.refine_translation_std_a,
                self.refine_angle_std_degrees
            ]
            .iter()
            .all(|x| x.is_finite() && *x >= 0.),
            "invalid discovery bath, gap, or refinement step"
        );
        ensure!(
            self.lambda_ratio.is_finite() && self.lambda_ratio > 0.,
            "invalid lambda ratio"
        );
        self.gate_options().validate()?;
        Ok(())
    }

    pub fn gate_options(&self) -> GateOptions {
        GateOptions {
            max_cells: self.envelope_max_cells,
            ..GateOptions::default()
        }
    }

    pub fn memory_config(&self, tree: &SphereTree) -> Result<ResolvedMemoryConfig> {
        MemoryConfig {
            slots: 1,
            attempts_per_sweep: 1,
            depletant_radius: Some(self.rd),
            depletant_activity: Some(self.activity),
            lambda_ratio: self.lambda_ratio,
            global_probability: 0.,
            local_translation_std_a: self.refine_translation_std_a,
            local_small_angle_std_degrees: self.refine_angle_std_degrees,
            initial_gap: self.gap,
            ..MemoryConfig::default()
        }
        .resolve(tree, self.rd, self.activity)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct VolumeEstimate {
    pub seed: u64,
    pub volume: f64,
    pub standard_error: f64,
    /// Wilson 95% interval, conditional on the conservative geometric envelope.
    pub confidence_95: [f64; 2],
    pub lower_volume: f64,
    pub uncertain_volume: f64,
    pub draws: usize,
    pub sampled_points: usize,
    pub hits: usize,
    pub created_cells: usize,
    pub retained_cells: usize,
}

fn identity() -> Pose {
    Pose {
        position: [0.; 3],
        orientation: [1., 0., 0., 0.],
    }
}

/// Fixed-size union-overlap estimate L + U*k/n. It is a diagnostic score, not
/// a Boltzmann weight or physical acceptance probability. Repeated atom spheres
/// and overlaps within either molecule do not multiply point membership.
pub fn estimate_pair_overlap(
    tree: &SphereTree,
    pose: Pose,
    rd: f64,
    draws: usize,
    seed: u64,
    opts: GateOptions,
) -> Result<VolumeEstimate> {
    ensure!(draws > 0, "positive fixed score allocation required");
    let env = Environment {
        tree,
        fixed: vec![Placed::new(identity())],
        labels: vec![],
        rd,
    };
    let envelope = OverlapEnvelope::build(&env, pose, opts)?;
    let moving = Placed::new(pose);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut hits = 0;
    let sampled_points = if envelope.uncertain_volume > 0. {
        draws
    } else {
        0
    };
    for _ in 0..sampled_points {
        let target = rng.random::<f64>() * envelope.uncertain_volume;
        let index = envelope
            .cumulative
            .partition_point(|&v| v <= target)
            .min(envelope.cells.len() - 1);
        let cell = envelope.cells[index];
        let point =
            std::array::from_fn(|k| cell.lo[k] + rng.random::<f64>() * (cell.hi[k] - cell.lo[k]));
        hits += usize::from(tree.contains(point, rd) && env.contains(moving.apply(point)));
    }
    let n = draws as f64;
    let p = hits as f64 / n;
    let u = envelope.uncertain_volume;
    let l = envelope.lower_volume;
    let z = 1.959963984540054;
    let denominator = 1. + z * z / n;
    let middle = (p + z * z / (2. * n)) / denominator;
    let half = z * (p * (1. - p) / n + z * z / (4. * n * n)).sqrt() / denominator;
    Ok(VolumeEstimate {
        seed,
        volume: l + u * p,
        standard_error: u * (p * (1. - p) / n).sqrt(),
        confidence_95: [
            l + u * (middle - half).max(0.),
            l + u * (middle + half).min(1.),
        ],
        lower_volume: l,
        uncertain_volume: u,
        draws,
        sampled_points,
        hits,
        created_cells: envelope.created,
        retained_cells: envelope.cells.len(),
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiscoverySeeds {
    pub initialization: u64,
    pub search: u64,
    pub search_score: u64,
    pub validation_initial: [u64; 2],
    pub validation_optimized: [u64; 2],
    pub refinement: u64,
}

/// Domain-separated fixed stream keys avoid budget-dependent shifts and avoid
/// sharing keys between adjacent master seeds used by independent jobs.
pub fn slot_seeds(seed: u64, slot: usize) -> DiscoverySeeds {
    let key = |stream: u64| {
        let mut hash = Sha256::new();
        hash.update(b"native-blind-depletion-contact-discovery-stream-v1");
        hash.update(seed.to_le_bytes());
        hash.update((slot as u64).to_le_bytes());
        hash.update(stream.to_le_bytes());
        u64::from_le_bytes(hash.finalize()[..8].try_into().unwrap())
    };
    DiscoverySeeds {
        initialization: key(0),
        search: key(1),
        search_score: key(2),
        validation_initial: [key(3), key(4)],
        validation_optimized: [key(5), key(6)],
        refinement: key(7),
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContactWitness {
    pub surface_point: Vec3,
    pub mobile_surface_lever: Vec3,
    pub radial_contact: serde_json::Value,
}

fn witness(tree: &SphereTree, pose: Pose) -> Result<ContactWitness> {
    let distance = norm(pose.position);
    ensure!(distance > 0., "undefined radial contact direction");
    let direction = scale(pose.position, 1. / distance);
    let contact = tree
        .outermost_radial_contact(rotation(pose.orientation), direction)?
        .ok_or_else(|| anyhow::anyhow!("saved contact ray no longer hits the core"))?;
    let r = rotation(contact.orientation);
    let contact_position = scale(contact.direction, contact.distance);
    let a = &tree.shape.atoms[contact.moving_atom];
    let b = &tree.shape.atoms[contact.fixed_atom];
    let delta = sub(add(contact_position, matvec(r, a.center)), b.center);
    ensure!(norm(delta) > 0., "undefined contact witness normal");
    let surface_point = add(b.center, scale(delta, b.radius / norm(delta)));
    Ok(ContactWitness {
        surface_point,
        mobile_surface_lever: sub(surface_point, contact_position),
        radial_contact: serde_json::to_value(contact)?,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchAttempt {
    pub step: usize,
    pub mode: usize,
    pub angle_degrees: f64,
    pub proposed_pose: Option<Pose>,
    pub hard_valid: bool,
    pub accepted: bool,
    pub score: Option<VolumeEstimate>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RefinementSample {
    pub step: usize,
    pub pose: Pose,
    pub accepted: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct SlotDiscovery {
    pub slot: usize,
    pub seeds: DiscoverySeeds,
    pub initial_pose: Pose,
    pub initial_contact: ContactWitness,
    pub optimized_pose: Pose,
    pub optimized_contact: ContactWitness,
    pub search_initial_score: VolumeEstimate,
    pub search_optimized_score: VolumeEstimate,
    pub initial_validation: [VolumeEstimate; 2],
    pub optimized_validation: [VolumeEstimate; 2],
    pub search_attempts: Vec<SearchAttempt>,
    pub refinement_samples: Vec<RefinementSample>,
    pub refinement_trace: Vec<MemoryMove>,
    pub refinement_counts: MemoryCounts,
    pub resolved_memory_config: ResolvedMemoryConfig,
}

#[derive(Clone, Debug, Serialize)]
pub struct DiscoveryResult {
    pub schema: &'static str,
    pub config: DiscoveryConfig,
    pub gate_options: GateOptions,
    pub slots: Vec<SlotDiscovery>,
}

fn perturbation(rng: &mut StdRng, angle_degrees: f64) -> Mat3 {
    cayley(std::array::from_fn(|_| {
        let z: f64 = StandardNormal.sample(rng);
        0.5 * angle_degrees.to_radians() * z
    }))
}

/// One independent start. All slots are preserved; no validation statistic or
/// supplied template can enter endpoint selection. Mode 0 changes orientation,
/// mode 1 the separation ray, mode 2 both, then projects to the outermost contact.
pub fn discover_slot(
    tree: &SphereTree,
    config: &DiscoveryConfig,
    slot: usize,
) -> Result<SlotDiscovery> {
    config.validate()?;
    ensure!(slot < config.starts, "slot index outside configured starts");
    let seeds = slot_seeds(config.seed, slot);
    let resolved_memory_config = config.memory_config(tree)?;
    let mut initialization_rng = StdRng::seed_from_u64(seeds.initialization);
    let mut state =
        MemoryState::initialize(tree, &resolved_memory_config, &mut initialization_rng)?;
    let initial_pose = state.poses[0];
    let initial_contact = witness(tree, initial_pose)?;
    let opts = config.gate_options();
    let score = |pose, seed, draws| estimate_pair_overlap(tree, pose, config.rd, draws, seed, opts);
    let search_initial_score = score(initial_pose, seeds.search_score, config.search_points)?;
    let mut search_optimized_score = search_initial_score.clone();
    let mut optimized_pose = initial_pose;
    let mut rng = StdRng::seed_from_u64(seeds.search);
    let fixed = Placed::new(identity());
    let numerical_gap = 4096. * f64::EPSILON * (1. + tree.bound + config.rd);
    let mut search_attempts = Vec::with_capacity(config.search_steps);
    for step in 1..=config.search_steps {
        let mode = rng.random_range(0..3);
        let angle_degrees = [1., 4., 12.][rng.random_range(0..3)];
        let mut orientation = rotation(optimized_pose.orientation);
        let mut direction = scale(optimized_pose.position, 1. / norm(optimized_pose.position));
        if mode != 1 {
            orientation = matmul(perturbation(&mut rng, angle_degrees), orientation);
        }
        if mode != 0 {
            direction = matvec(perturbation(&mut rng, angle_degrees), direction);
        }
        let mut attempt = SearchAttempt {
            step,
            mode,
            angle_degrees,
            proposed_pose: None,
            hard_valid: false,
            accepted: false,
            score: None,
        };
        if let Some(contact) = tree.outermost_radial_contact(orientation, direction)? {
            let pose = Pose {
                position: scale(
                    contact.direction,
                    contact.distance + config.gap + numerical_gap,
                ),
                orientation: contact.orientation,
            };
            pose.validate()?;
            attempt.proposed_pose = Some(pose);
            attempt.hard_valid = norm(pose.position) < resolved_memory_config.radius
                && !tree.overlaps(&Placed::new(pose), &fixed);
            if attempt.hard_valid {
                let estimate = score(pose, seeds.search_score, config.search_points)?;
                attempt.accepted = estimate.volume > search_optimized_score.volume;
                if attempt.accepted {
                    optimized_pose = pose;
                    search_optimized_score = estimate.clone();
                }
                attempt.score = Some(estimate);
            }
        }
        search_attempts.push(attempt);
    }
    let optimized_contact = witness(tree, optimized_pose)?;
    // The validation allocations and seeds never feed the search/refinement RNGs.
    let initial_validation = [
        score(
            initial_pose,
            seeds.validation_initial[0],
            config.validation_points,
        )?,
        score(
            initial_pose,
            seeds.validation_initial[1],
            config.validation_points,
        )?,
    ];
    let optimized_validation = [
        score(
            optimized_pose,
            seeds.validation_optimized[0],
            config.validation_points,
        )?,
        score(
            optimized_pose,
            seeds.validation_optimized[1],
            config.validation_points,
        )?,
    ];
    state.poses[0] = optimized_pose;
    state.validate(tree, &resolved_memory_config)?;
    let mut rng = StdRng::seed_from_u64(seeds.refinement);
    let mut refinement_trace = Vec::with_capacity(config.refine_steps);
    let mut refinement_samples = Vec::new();
    let mut refinement_counts = MemoryCounts::default();
    for step in 1..=config.refine_steps {
        let result = state.update(tree, &resolved_memory_config, opts, &mut rng)?;
        refinement_counts.record(&result);
        if step > config.burn && (step - config.burn) % config.save_every == 0 {
            refinement_samples.push(RefinementSample {
                step,
                pose: result.retained_pose,
                accepted: result.accepted,
            });
        }
        refinement_trace.push(result);
    }
    Ok(SlotDiscovery {
        slot,
        seeds,
        initial_pose,
        initial_contact,
        optimized_pose,
        optimized_contact,
        search_initial_score,
        search_optimized_score,
        initial_validation,
        optimized_validation,
        search_attempts,
        refinement_samples,
        refinement_trace,
        refinement_counts,
        resolved_memory_config,
    })
}

pub fn discover(tree: &SphereTree, config: &DiscoveryConfig) -> Result<DiscoveryResult> {
    config.validate()?;
    let slots = (0..config.starts)
        .map(|slot| discover_slot(tree, config, slot))
        .collect::<Result<_>>()?;
    Ok(DiscoveryResult {
        schema: "native-blind-depletion-contact-discovery-v1",
        config: config.clone(),
        gate_options: config.gate_options(),
        slots,
    })
}
