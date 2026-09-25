//! Reproducible, non-equilibrium free-body preparations around an optional seed.
//!
//! Concentration refers to tetramers per full geometric vessel volume. The
//! conservative separation test excludes depletion contacts of new free bodies; it
//! does not constrain subsequent Monte Carlo moves or change the target.
use crate::{
    geometry::{Environment, Placed, SphereTree},
    math::{Pose, Vec3, add, minimum_image, norm, scale, sub, uniform_pose, wrap},
    simulation::{Boundary, Config, hash_bytes},
    spherical::Container,
};
use anyhow::{Result, ensure};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::f64::consts::PI;

/// Exact SI Avogadro constant, converted from micromolar to particles / A^3.
pub const NUMBER_DENSITY_PER_MICROMOLAR: f64 = 6.022_140_76e-10;
const MAX_PLACEMENT_ATTEMPTS: usize = 100_000;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct FreeTetramerStart {
    pub count: usize,
    pub concentration_um: f64,
    /// Defaults to the input master seed. Preparation has its own RNG domain.
    pub seed: Option<u64>,
}

impl FreeTetramerStart {
    /// Volume when no seed is supplied; apply() includes the retained seed count.
    pub fn volume(&self) -> Result<f64> {
        self.volume_for_total(self.count)
    }

    fn volume_for_total(&self, total: usize) -> Result<f64> {
        ensure!(
            total >= 2,
            "at least two total tetramers (seed plus free) are required"
        );
        ensure!(
            self.count <= MAX_PLACEMENT_ATTEMPTS,
            "--free-tetramers exceeds the {MAX_PLACEMENT_ATTEMPTS}-attempt preparation budget"
        );
        ensure!(
            self.concentration_um.is_finite() && self.concentration_um > 0.,
            "tetramer concentration must be finite and positive (in micromolar)"
        );
        let volume = total as f64 / (self.concentration_um * NUMBER_DENSITY_PER_MICROMOLAR);
        ensure!(
            volume.is_finite() && volume > 0.,
            "unrepresentable vessel volume"
        );
        Ok(volume)
    }

    pub fn apply(&self, config: &mut Config, tree: &SphereTree) -> Result<()> {
        let mut source_indices = config.seed_labels.clone();
        source_indices.sort_unstable();
        ensure!(
            source_indices.windows(2).all(|w| w[0] != w[1]),
            "duplicate seed labels"
        );
        ensure!(
            source_indices
                .iter()
                .all(|&i| i < config.initial_poses.len()),
            "seed label out of range"
        );
        let seed_count = source_indices.len();
        let total = self
            .count
            .checked_add(seed_count)
            .ok_or_else(|| anyhow::anyhow!("tetramer count overflow"))?;
        let volume = self.volume_for_total(total)?;
        let seed = self.seed.unwrap_or(config.seed);
        let exclusion_bound = tree.bound + config.depletant_radius;
        let (boundary, lengths, inner_radius) = match config.boundary {
            Boundary::Spherical { .. } => {
                let radius = (volume / (4. * PI / 3.)).cbrt();
                ensure!(
                    radius.is_finite() && radius > 0. && (self.count == 0 || radius > tree.bound),
                    "requested concentration leaves no room for a dispersed body inside the spherical wall; lower it"
                );
                (
                    Boundary::Spherical { radius },
                    [2. * radius; 3],
                    Some(radius - tree.bound),
                )
            }
            Boundary::Periodic => {
                let length = volume.cbrt();
                ensure!(
                    length.is_finite() && length > 4. * exclusion_bound,
                    "requested concentration violates periodic L > 4*(body bound + depletant radius); lower it or increase the body count"
                );
                (Boundary::Periodic, [length; 3], None)
            }
        };
        let guard = 1024. * f64::EPSILON * (1. + lengths[0] + exclusion_bound);
        let separation = 2. * exclusion_bound + guard;
        if let Some(inner) = inner_radius.filter(|_| seed_count == 0) {
            ensure!(
                2. * inner > separation,
                "requested concentration cannot fit two separated exclusion bounds inside the wall; lower it"
            );
        }
        let mut digest = Sha256::new();
        digest.update(b"tetramer-mc-free-preparation-v1");
        digest.update(seed.to_le_bytes());
        let mut rng = StdRng::from_seed(digest.finalize().into());
        let mut poses: Vec<Pose> = Vec::with_capacity(total);
        poses.extend(source_indices.iter().map(|&i| config.initial_poses[i]));
        let mut seed_translation = [0.; 3];
        if boundary == Boundary::Periodic && seed_count > 0 {
            seed_translation = preserve_periodic_seed(&mut poses, config.box_lengths, lengths)?;
        }
        let wall = boundary
            .radius()
            .map(|r| Container::new(r, tree))
            .transpose()?;
        for (i, &pose) in poses.iter().enumerate() {
            pose.validate()?;
            ensure!(
                wall.as_ref().is_none_or(|w| w.contains(pose)),
                "preserved seed body {} does not fit the resized spherical wall; lower the concentration (seed positions are not rescaled)",
                source_indices[i]
            );
            if boundary == Boundary::Periodic {
                let env = Environment::new(
                    tree,
                    &poses,
                    i,
                    pose,
                    pose,
                    lengths,
                    config.depletant_radius,
                )?;
                ensure!(
                    env.hard_valid(pose),
                    "hard overlap in preserved periodic seed"
                );
            } else {
                for &other in &poses[..i] {
                    ensure!(
                        !tree.overlaps(&Placed::new(pose), &Placed::new(other)),
                        "hard overlap in preserved seed"
                    );
                }
            }
        }
        let mut attempts = 0;
        let mut min_distance = f64::INFINITY;
        while poses.len() < total && attempts < MAX_PLACEMENT_ATTEMPTS {
            attempts += 1;
            let mut candidate = uniform_pose(&mut rng, lengths);
            if let Some(inner) = inner_radius {
                let direction: Vec3 = std::array::from_fn(|_| StandardNormal.sample(&mut rng));
                let length = norm(direction);
                ensure!(
                    length.is_finite() && length > 0.,
                    "unrepresentable random direction"
                );
                candidate.position = scale(direction, inner * rng.random::<f64>().cbrt() / length);
            }
            candidate.validate()?;
            let distance = poses
                .iter()
                .map(|other| {
                    let delta = sub(candidate.position, other.position);
                    norm(if boundary == Boundary::Periodic {
                        minimum_image(delta, lengths)
                    } else {
                        delta
                    })
                })
                .fold(f64::INFINITY, f64::min);
            if distance > separation {
                min_distance = min_distance.min(distance);
                poses.push(candidate);
            }
        }
        ensure!(
            poses.len() == total,
            "could place only {} of {} free tetramers in {MAX_PLACEMENT_ATTEMPTS} attempts at {} uM; lower the concentration or try another --seed. This conservative dispersed preparation requires separated exclusion bounds, not just non-overlapping atomic cores",
            poses.len() - seed_count,
            self.count,
            self.concentration_um
        );
        let mut fragments: Vec<Vec<usize>> = Vec::new();
        if seed_count > 0 {
            fragments.push((0..seed_count).collect());
        }
        fragments.extend((seed_count..total).map(|i| vec![i]));
        let original_metadata = std::mem::take(&mut config.metadata);
        let native_motifs = original_metadata.get("native_pair_motifs").cloned();
        config.metadata = json!({
            "arm": if seed_count == 0 { "dispersed-free" } else { "seeded-free" },
            "initial_free_tetramers": self.count,
            "initial_seed_tetramers": seed_count,
            "total_tetramers": total,
            "initial_fragments": fragments,
            "preparation_equilibrated": false,
            "template_metadata": original_metadata,
            "initial_poses_sha256": hash_bytes(&serde_json::to_vec(&poses)?),
            "free_tetramer_initialization": {
                "protocol": if seed_count == 0 { "separated-exclusion-bounds-v1" } else { "seed-preserving-separated-free-v2" },
                "count": self.count,
                "seed_count": seed_count,
                "total_count": total,
                "preserved_seed_source_indices": source_indices,
                "seed_translation_A": seed_translation,
                "seed_geometry": "spherical poses unchanged; periodic compact seeds may be unwrapped and translated without rotation or rescaling",
                "seed_remains_mobile": true,
                "tetramer_concentration_uM": self.concentration_um,
                "number_density_A_minus3": self.concentration_um * NUMBER_DENSITY_PER_MICROMOLAR,
                "vessel_volume_A3": volume,
                "concentration_convention": "tetramers / full geometric vessel volume, not solvent-accessible or center-accessible volume",
                "master_seed": seed,
                "preparation_rng": "SHA256(tetramer-mc-free-preparation-v1 || little-endian u64 seed); rand pinned by Cargo.lock",
                "attempts": attempts,
                "attempt_limit": MAX_PLACEMENT_ATTEMPTS,
                "body_bound_A": tree.bound,
                "exclusion_bound_A": exclusion_bound,
                "minimum_center_distance_A": if min_distance.is_finite() { Some(min_distance) } else { None },
                "required_separation_A": separation,
                "separation_scope": "new free bodies versus all seed and earlier free bodies; supplied seed-seed contacts are retained",
                "orientation_law": "new free bodies: independent Haar SO(3); seed orientations retained",
                "preparation_law": "sequential uniform inner-ball or periodic-cube placements conditioned on bound separation; not equilibrium"
            }
        });
        if let Some(motifs) = native_motifs {
            config.metadata["native_pair_motifs"] = motifs;
        }
        config.initial_poses = poses;
        config.seed_labels = (0..seed_count).collect();
        config.seed = seed;
        config.boundary = boundary;
        config.box_lengths = lengths;
        config.validate()?;
        Ok(())
    }
}

/// Embed a compact periodic seed before changing its box. All pair displacement
/// checks are essential: anchor-only unwrapping can tear a winding scaffold.
fn preserve_periodic_seed(
    poses: &mut [Pose],
    old_lengths: Vec3,
    new_lengths: Vec3,
) -> Result<Vec3> {
    let original: Vec<_> = poses.iter().map(|p| p.position).collect();
    let anchor = original[0];
    for (pose, &position) in poses.iter_mut().zip(&original) {
        pose.position = std::array::from_fn(|k| {
            let image = ((position[k] - anchor[k]) / old_lengths[k] + 0.5).floor();
            if image == 0. {
                position[k]
            } else {
                position[k] - old_lengths[k] * image
            }
        });
    }
    let scale_max = old_lengths
        .into_iter()
        .chain(new_lengths)
        .chain(original.iter().map(|p| norm(*p)))
        .fold(1_f64, f64::max);
    let tolerance = 4096. * f64::EPSILON * scale_max;
    for i in 0..poses.len() {
        for j in 0..i {
            let delta = sub(poses[i].position, poses[j].position);
            let expected = minimum_image(sub(original[i], original[j]), old_lengths);
            ensure!(
                norm(sub(delta, expected)) <= tolerance,
                "periodic seed is not a compact, consistently unwrapped fragment; cannot resize its cell without changing seed geometry"
            );
            ensure!(
                norm(sub(minimum_image(delta, new_lengths), expected)) <= tolerance,
                "preserved seed is too large for the requested periodic cell; lower the concentration"
            );
        }
    }
    let mut translation = [0.; 3];
    if poses
        .iter()
        .any(|p| (0..3).any(|k| p.position[k] < 0. || p.position[k] >= new_lengths[k]))
    {
        for k in 0..3 {
            let lo = poses
                .iter()
                .map(|p| p.position[k])
                .fold(f64::INFINITY, f64::min);
            let hi = poses
                .iter()
                .map(|p| p.position[k])
                .fold(f64::NEG_INFINITY, f64::max);
            translation[k] = 0.5 * new_lengths[k] - (0.5 * lo + 0.5 * hi);
        }
    }
    for pose in poses.iter_mut() {
        if translation != [0.; 3]
            || (0..3).any(|k| pose.position[k] < 0. || pose.position[k] >= new_lengths[k])
        {
            pose.position = wrap(add(pose.position, translation), new_lengths);
        }
    }
    for i in 0..poses.len() {
        for j in 0..i {
            let actual = minimum_image(sub(poses[i].position, poses[j].position), new_lengths);
            let expected = minimum_image(sub(original[i], original[j]), old_lengths);
            ensure!(
                norm(sub(actual, expected)) <= tolerance,
                "resized cell would change periodic seed geometry"
            );
        }
    }
    Ok(translation)
}
