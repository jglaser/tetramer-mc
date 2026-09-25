//! Independent frozen-state attempts, not a trajectory or equilibrium estimate.
//! All selector failures and rejected endpoints remain in the denominator.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs, io::Write, path::PathBuf, time::Instant};
use tetramer_mc::{
    conditional_axis::{ConditionalAxis, ConditionalAxisConfig},
    geometry::{Placed, Shape, SphereTree},
    math::{Pose, Vec3, norm, scale},
    simulation::{Boundary, Config, cpu_seconds, hash_bytes, save},
    spherical::{self, Container, HalfTurn},
};

#[derive(Parser, Serialize)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    out: PathBuf,
    /// Independent attempts per population PER ARM; every attempt resets to X.
    #[arg(long, default_value_t = 128)]
    attempts: usize,
    #[arg(long, default_value_t = 4)]
    populations: usize,
    #[arg(long, default_value_t = 2026092801)]
    seed: u64,
    #[arg(long, default_value_t = 0.02)]
    score_floor: f64,
    #[arg(long, default_value_t = 64)]
    max_candidates: usize,
    #[arg(long, default_value_t = 0.25)]
    uniform_axis_weight: f64,
}

#[derive(Clone, Copy)]
enum Arm {
    Isotropic,
    ConditionedUniform,
    ConditionedBands,
}
impl Arm {
    fn name(self) -> &'static str {
        match self {
            Self::Isotropic => "isotropic",
            Self::ConditionedUniform => "conditioned_uniform",
            Self::ConditionedBands => "conditioned_bands",
        }
    }
}
const ARMS: [Arm; 3] = [
    Arm::Isotropic,
    Arm::ConditionedUniform,
    Arm::ConditionedBands,
];

fn named_seed(seed: u64, arm: Arm, population: usize, attempt: usize, label: &str) -> u64 {
    let mut h = Sha256::new();
    h.update(b"tetramer-mc-conditional-axis-benchmark-v1");
    h.update(seed.to_le_bytes());
    h.update(arm.name().as_bytes());
    h.update((population as u64).to_le_bytes());
    h.update((attempt as u64).to_le_bytes());
    h.update(label.as_bytes());
    u64::from_le_bytes(h.finalize()[..8].try_into().unwrap())
}

type ContactSet = BTreeSet<(usize, usize)>;
fn contact_pairs(tree: &SphereTree, poses: &[Pose]) -> ContactSet {
    let placed: Vec<_> = poses.iter().copied().map(Placed::new).collect();
    let mut pairs = BTreeSet::new();
    for i in 0..placed.len() {
        for j in i + 1..placed.len() {
            if tree.overlaps(&placed[i], &placed[j]) {
                pairs.insert((i, j));
            }
        }
    }
    pairs
}
fn tag_contacts(pairs: &ContactSet, tag: usize) -> BTreeSet<usize> {
    pairs
        .iter()
        .filter_map(|&(i, j)| {
            if i == tag {
                Some(j)
            } else if j == tag {
                Some(i)
            } else {
                None
            }
        })
        .collect()
}
fn difference<T: Copy + Ord>(a: &BTreeSet<T>, b: &BTreeSet<T>) -> Vec<T> {
    a.difference(b).copied().collect()
}

#[derive(Default, Serialize)]
struct Totals {
    attempted: u64,
    successful_calls: u64,
    errors: u64,
    accepted: u64,
    gca_executed: u64,
    tagged_flips: u64,
    tagged_exchanges: u64,
    any_contact_changes: u64,
    pair_loss_and_gain: u64,
    lost_tag_contacts: u64,
    gained_tag_contacts: u64,
    lost_pairs: u64,
    gained_pairs: u64,
    sampler_cpu_seconds: f64,
    sampler_wall_seconds: f64,
    diagnostics_cpu_seconds: f64,
}
impl Totals {
    fn report(&self) -> Value {
        let mut value = serde_json::to_value(self).unwrap();
        value["tagged_exchange_probability"] = json!(if self.attempted > 0 {
            Some(self.tagged_exchanges as f64 / self.attempted as f64)
        } else {
            None
        });
        value["tagged_exchanges_per_sampler_cpu_second"] =
            json!(if self.sampler_cpu_seconds > 0. {
                Some(self.tagged_exchanges as f64 / self.sampler_cpu_seconds)
            } else {
                None
            });
        value["tagged_exchanges_per_cpu_second_including_diagnostics"] =
            json!(
                if self.sampler_cpu_seconds + self.diagnostics_cpu_seconds > 0. {
                    Some(
                        self.tagged_exchanges as f64
                            / (self.sampler_cpu_seconds + self.diagnostics_cpu_seconds),
                    )
                } else {
                    None
                }
            );
        value
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        (1..=65_536).contains(&args.attempts) && (1..=32).contains(&args.populations),
        "invalid fixed benchmark allocation"
    );
    let manifest_path = args.out.with_extension("manifest.json");
    let attempts_path = args.out.with_extension("attempts.jsonl");
    ensure!(
        [&args.out, &manifest_path, &attempts_path]
            .iter()
            .all(|p| !p.exists()),
        "output already exists; choose a fresh path"
    );
    let start_cpu = cpu_seconds();
    let start_wall = Instant::now();
    let config_path = args.config.canonicalize()?;
    let config_raw = fs::read(&config_path)?;
    let config: Config = serde_json::from_slice(&config_raw)?;
    config.validate()?;
    let radius = match config.boundary {
        Boundary::Spherical { radius } => radius,
        _ => anyhow::bail!("the contact-conditioned GCA requires spherical boundaries"),
    };
    ensure!(
        config.fixed_body_indices.is_empty(),
        "benchmark requires all-mobile bodies"
    );
    let shape_path = if config.shape.is_absolute() {
        config.shape.clone()
    } else {
        config_path
            .parent()
            .context("config parent")?
            .join(&config.shape)
    }
    .canonicalize()?;
    let shape_raw = fs::read(&shape_path)?;
    let tree = SphereTree::new(serde_json::from_slice::<Shape>(&shape_raw)?)?;
    let wall = Container::new(radius, &tree)?;
    spherical::validate_state(&tree, &wall, &config.initial_poses)?;
    let mut contact_shape = tree.shape.clone();
    for atom in &mut contact_shape.atoms {
        atom.radius += config.depletant_radius;
    }
    let contact_tree = SphereTree::new(contact_shape)?;
    let old_pairs = contact_pairs(&contact_tree, &config.initial_poses);
    let uniform_config = ConditionalAxisConfig {
        score_floor: args.score_floor,
        max_candidates: args.max_candidates,
        uniform_axis_weight: 1.,
    };
    let band_config = ConditionalAxisConfig {
        score_floor: args.score_floor,
        max_candidates: args.max_candidates,
        uniform_axis_weight: args.uniform_axis_weight,
    };
    let uniform = ConditionalAxis::new(&tree, config.depletant_radius, uniform_config)?;
    let bands = ConditionalAxis::new(&tree, config.depletant_radius, band_config)?;
    let setup_cpu = cpu_seconds() - start_cpu;
    let mut report = json!({
        "schema":"conditional-axis-frozen-attempt-benchmark-v1",
        "args":args,
        "scope":"Independent one-step attempts from the same frozen X; pure physical many-body GCA plus corrected axis selector. Not a trajectory, equilibrium population, mixing-time, assembly-stability or contact-ESS measurement.",
        "physical_target":"Hard atomic wall and pair cores, exp(-z times exclusion-union volume); ideal bath permeates wall. Proposal-model auxiliary and assembly-bias gates are not part of this diagnostic.",
        "contact_definition":"Strict overlap of atom sphere unions inflated by depletant_radius; no native labels used",
        "tag_rule":"One uniformly chosen label per attempt, retained throughout the conditional-axis exchange; identical law in all arms, independent streams",
        "allocation_protocol":"Fixed populations and attempts in each of three arms, declared before sampling. Cyclic arm execution order; independent named guide and physical seeds per attempt. No conditional retries of a complete attempt.",
        "failure_protocol":"Forward/reverse search caps, selector rejections, zero flips and failed exchanges all remain in denominators. Numerical errors are streamed, remaining allocation is drained, and final completion is false.",
        "timing_protocol":"Sampler process CPU includes tag selection, axis search, physical GCA and selector correction. Diagnostic all-pair contact classification is separately timed. In-memory result-to-JSON conversion is included; setup and streamed JSONL encoding/I/O are excluded from kernel throughput and reported separately/overall.",
        "config_path":config_path,"config_sha256":hash_bytes(&config_raw),
        "config":serde_json::from_slice::<Value>(&config_raw)?,
        "shape_path":shape_path,"shape_sha256":hash_bytes(&shape_raw),
        "shape_atoms":tree.shape.atoms.len(),
        "source_bundle_sha256":hash_bytes(include_bytes!(concat!(env!("OUT_DIR"),"/source-bundle.json"))),
        "executable_sha256":hash_bytes(&fs::read(std::env::current_exe()?)?),
        "runner_source":include_str!("conditional-axis-benchmark.rs"),
        "selector_source":include_str!("../conditional_axis.rs"),
        "initial_contact_pairs":old_pairs,
        "arms":[{"name":"isotropic"},{"name":"conditioned_uniform","selector":uniform_config},
            {"name":"conditioned_bands","selector":band_config}],
        "total_attempts":3*args.populations*args.attempts,
        "setup_cpu_seconds":setup_cpu,
        "finished_allocation":false,"completed":false,
    });
    if let Some(parent) = args.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    save(&manifest_path, &report)?;
    let mut stream = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&attempts_path)?;
    let mut totals: Vec<Vec<Totals>> = (0..3)
        .map(|_| (0..args.populations).map(|_| Totals::default()).collect())
        .collect();
    let mut error_count = 0u64;
    for population in 0..args.populations {
        for attempt in 0..args.attempts {
            for offset in 0..3 {
                let arm_index = (population + attempt + offset) % 3;
                let arm = ARMS[arm_index];
                let guide_seed = named_seed(args.seed, arm, population, attempt, "guide");
                let physical_seed = named_seed(args.seed, arm, population, attempt, "physical");
                let mut guide_rng = StdRng::seed_from_u64(guide_seed);
                let mut physical_rng = StdRng::seed_from_u64(physical_seed);
                let mut endpoint = config.initial_poses.clone();
                let sampler_start = cpu_seconds();
                let sampler_wall = Instant::now();
                let result: Result<(usize, bool, Option<spherical::GcaStats>, Value)> =
                    (|| match arm {
                        Arm::Isotropic => {
                            let tag = guide_rng.random_range(0..endpoint.len());
                            let vector: Vec3 =
                                std::array::from_fn(|_| StandardNormal.sample(&mut guide_rng));
                            let axis = scale(vector, 1. / norm(vector));
                            let gca = spherical::update(
                                &tree,
                                &wall,
                                &mut endpoint,
                                HalfTurn::new(axis)?,
                                config.depletant_radius,
                                config.reservoir_density,
                                &mut physical_rng,
                            )?;
                            Ok((tag, true, Some(gca), json!({"tag":tag,"axis":axis})))
                        }
                        Arm::ConditionedUniform | Arm::ConditionedBands => {
                            let selector = if matches!(arm, Arm::ConditionedUniform) {
                                &uniform
                            } else {
                                &bands
                            };
                            let outcome = selector.update(
                                &wall,
                                &mut endpoint,
                                config.reservoir_density,
                                &mut guide_rng,
                                &mut physical_rng,
                            )?;
                            Ok((
                                outcome.stats.tag,
                                outcome.accepted,
                                outcome.gca,
                                serde_json::to_value(outcome.stats)?,
                            ))
                        }
                    })();
                let sampler_cpu = cpu_seconds() - sampler_start;
                let sampler_wall = sampler_wall.elapsed().as_secs_f64();
                let total = &mut totals[arm_index][population];
                total.attempted += 1;
                total.sampler_cpu_seconds += sampler_cpu;
                total.sampler_wall_seconds += sampler_wall;
                let mut record = json!({"arm":arm.name(),"population":population,"attempt":attempt,
                    "guide_seed":guide_seed,"physical_seed":physical_seed,
                    "sampler_cpu_seconds":sampler_cpu,"sampler_wall_seconds":sampler_wall});
                match result {
                    Ok((tag, accepted, gca, selector)) => {
                        let diagnostics_start = cpu_seconds();
                        // Reclassify even rejected endpoints so a rollback bug
                        // cannot be hidden by copying the initial contact labels.
                        let new_pairs = contact_pairs(&contact_tree, &endpoint);
                        let rollback_valid = accepted || new_pairs == old_pairs;
                        let before = tag_contacts(&old_pairs, tag);
                        let after = tag_contacts(&new_pairs, tag);
                        let tag_lost = difference(&before, &after);
                        let tag_gained = difference(&after, &before);
                        let pairs_lost = difference(&old_pairs, &new_pairs);
                        let pairs_gained = difference(&new_pairs, &old_pairs);
                        let exchange = !tag_lost.is_empty() && !tag_gained.is_empty();
                        let tag_flipped = accepted
                            && gca
                                .as_ref()
                                .is_some_and(|g| g.flipped_indices.contains(&tag));
                        let diagnostics_cpu = cpu_seconds() - diagnostics_start;
                        total.successful_calls += u64::from(rollback_valid);
                        if !rollback_valid {
                            total.errors += 1;
                            error_count += 1;
                        }
                        total.accepted += u64::from(accepted);
                        total.gca_executed += u64::from(gca.is_some());
                        total.tagged_flips += u64::from(tag_flipped);
                        total.tagged_exchanges += u64::from(exchange);
                        total.any_contact_changes +=
                            u64::from(!pairs_lost.is_empty() || !pairs_gained.is_empty());
                        total.pair_loss_and_gain +=
                            u64::from(!pairs_lost.is_empty() && !pairs_gained.is_empty());
                        total.lost_tag_contacts += tag_lost.len() as u64;
                        total.gained_tag_contacts += tag_gained.len() as u64;
                        total.lost_pairs += pairs_lost.len() as u64;
                        total.gained_pairs += pairs_gained.len() as u64;
                        total.diagnostics_cpu_seconds += diagnostics_cpu;
                        record["status"] = json!(if rollback_valid { "ok" } else { "error" });
                        if !rollback_valid {
                            record["error"] = json!("rejected endpoint changed physical contacts");
                        }
                        record["tag"] = json!(tag);
                        record["accepted"] = json!(accepted);
                        record["tag_flipped"] = json!(tag_flipped);
                        record["tagged_exchange"] = json!(exchange);
                        record["tag_contacts_before"] = json!(before);
                        record["tag_contacts_after"] = json!(after);
                        record["tag_contacts_lost"] = json!(tag_lost);
                        record["tag_contacts_gained"] = json!(tag_gained);
                        record["pairs_lost"] = json!(pairs_lost);
                        record["pairs_gained"] = json!(pairs_gained);
                        record["selector"] = selector;
                        record["gca"] = serde_json::to_value(gca)?;
                        record["diagnostics_cpu_seconds"] = json!(diagnostics_cpu);
                    }
                    Err(error) => {
                        total.errors += 1;
                        error_count += 1;
                        record["status"] = json!("error");
                        record["error"] = json!(format!("{error:#}"));
                    }
                }
                serde_json::to_writer(&mut stream, &record)?;
                stream.write_all(b"\n")?;
                stream.flush()?;
            }
        }
        stream.sync_data()?;
        eprintln!(
            "Completed population {}/{}: {} attempts per arm",
            population + 1,
            args.populations,
            args.attempts
        );
    }
    report["populations"] = json!(
        ARMS.iter()
            .enumerate()
            .map(|(a, arm)| {
                json!({"arm":arm.name(),"populations":totals[a].iter().enumerate().map(|(p,t)| {
            json!({"population":p,"statistics":t.report()})
        }).collect::<Vec<_>>()})
            })
            .collect::<Vec<_>>()
    );
    report["attempts_path"] = json!(attempts_path);
    report["attempts_sha256"] = json!(hash_bytes(&fs::read(&attempts_path)?));
    report["finished_allocation"] = json!(true);
    report["completed"] = json!(error_count == 0);
    report["errors"] = json!(error_count);
    report["overall_cpu_seconds"] = json!(cpu_seconds() - start_cpu);
    report["overall_wall_seconds"] = json!(start_wall.elapsed().as_secs_f64());
    save(&args.out, &report)?;
    ensure!(
        error_count == 0,
        "{error_count} attempts failed; all allocated attempts drained and recorded"
    );
    println!("{}", args.out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn contact_changes_require_both_loss_and_gain() {
        let old = BTreeSet::from([(0, 1), (1, 2), (3, 4)]);
        let new = BTreeSet::from([(0, 2), (1, 2), (3, 4)]);
        assert_eq!(tag_contacts(&old, 0), BTreeSet::from([1]));
        assert_eq!(
            difference(&tag_contacts(&new, 0), &tag_contacts(&old, 0)),
            vec![2]
        );
        assert_eq!(difference(&old, &new), vec![(0, 1)]);
        assert_eq!(difference(&new, &old), vec![(0, 2)]);
        assert!(difference(&tag_contacts(&old, 3), &tag_contacts(&new, 3)).is_empty());
    }
    #[test]
    fn random_streams_are_named_independently() {
        let seed = named_seed(1, Arm::Isotropic, 0, 0, "guide");
        assert_eq!(seed, named_seed(1, Arm::Isotropic, 0, 0, "guide"));
        assert_ne!(seed, named_seed(1, Arm::Isotropic, 0, 0, "physical"));
        assert_ne!(seed, named_seed(1, Arm::ConditionedUniform, 0, 0, "guide"));
        assert_ne!(seed, named_seed(1, Arm::Isotropic, 1, 0, "guide"));
        assert_ne!(seed, named_seed(1, Arm::Isotropic, 0, 1, "guide"));
    }
}
