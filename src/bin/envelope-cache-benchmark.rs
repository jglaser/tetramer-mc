//! Bounded fixed-endpoint microbenchmark; never evolves a Monte Carlo state.
//! Local endpoints are conditioned on hard validity only to time actual gates.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde_json::{Value, json};
use std::{fs, path::PathBuf};
use tetramer_mc::{
    depletion::{self, BodyEnvelopeCache, Envelope, GateResult},
    geometry::{Environment, Placed, Shape, SphereTree},
    math::{Pose, Vec3, add, cayley, matmul, norm, quaternion, rotation, sub, wrap},
    simulation::{Boundary, Config, cpu_seconds, hash_bytes, save},
    spherical::Container,
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 16)]
    cases: usize,
    #[arg(long, default_value_t = 3)]
    repeats: usize,
    #[arg(long, default_value_t = 2026092601)]
    seed: u64,
    #[arg(long, default_value_t = 128)]
    max_attempts: usize,
    /// Cap expected Poisson points across three timed arms and two profiling calls.
    #[arg(long, default_value_t = 100_000_000.0)]
    max_expected_points: f64,
}

fn environment<'a>(
    tree: &'a SphereTree,
    cfg: &Config,
    i: usize,
    new: Pose,
) -> Result<Environment<'a>> {
    let old = cfg.initial_poses[i];
    if cfg.boundary == Boundary::Periodic {
        return Environment::new(
            tree,
            &cfg.initial_poses,
            i,
            old,
            new,
            cfg.box_lengths,
            cfg.depletant_radius,
        );
    }
    // Same conservative open-bath neighborhood and ordering as production.
    let reach = 2. * (tree.bound + cfg.depletant_radius);
    let guard = 1024. * f64::EPSILON * (1. + reach + norm(old.position) + norm(new.position));
    let mut fixed = Vec::new();
    let mut labels = Vec::new();
    for (j, &pose) in cfg.initial_poses.iter().enumerate() {
        if j != i
            && (norm(sub(pose.position, old.position)) <= reach + guard
                || norm(sub(pose.position, new.position)) <= reach + guard)
        {
            fixed.push(Placed::new(pose));
            labels.push((j, [0; 3]));
        }
    }
    Ok(Environment {
        tree,
        fixed,
        labels,
        rd: cfg.depletant_radius,
    })
}

fn candidate(rng: &mut StdRng, cfg: &Config, old: Pose) -> Pose {
    let displacement: Vec3 = std::array::from_fn(|_| {
        let x: f64 = StandardNormal.sample(rng);
        x * cfg.local_translation_std_a
    });
    let c: Vec3 = std::array::from_fn(|_| {
        let x: f64 = StandardNormal.sample(rng);
        x * 0.5 * cfg.local_small_angle_std_degrees.to_radians()
    });
    let position = add(old.position, displacement);
    Pose {
        position: if cfg.boundary == Boundary::Periodic {
            wrap(position, cfg.box_lengths)
        } else {
            position
        },
        orientation: quaternion(matmul(cayley(c), rotation(old.orientation))),
    }
}

fn same_envelope(a: &Envelope, b: &Envelope) -> bool {
    a.created == b.created
        && a.volume.to_bits() == b.volume.to_bits()
        && a.cells.len() == b.cells.len()
        && a.cumulative.len() == b.cumulative.len()
        && a.cells.iter().zip(&b.cells).all(|(x, y)| {
            x.lo.iter()
                .chain(&x.hi)
                .zip(y.lo.iter().chain(&y.hi))
                .all(|(p, q)| p.to_bits() == q.to_bits())
        })
        && a.cumulative
            .iter()
            .zip(&b.cumulative)
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

fn same_gate(a: &GateResult, b: &GateResult) -> bool {
    a.gained == b.gained
        && a.lost == b.lost
        && a.raw_points == b.raw_points
        && a.retained_points == b.retained_points
        && a.retained_cells == b.retained_cells
        && a.created_cells == b.created_cells
        && a.log_weight.to_bits() == b.log_weight.to_bits()
        && a.envelope_volume.to_bits() == b.envelope_volume.to_bits()
}

fn query_summary(counts: [u64; 6]) -> Value {
    let calls = counts[0] + counts[2] + counts[4];
    let skipped = counts[1] + counts[3] + counts[5];
    let baseline_calls = calls + skipped;
    json!({
        "body_queries":counts[0],"body_skipped":counts[1],
        "old_queries":counts[2],"old_skipped":counts[3],
        "new_queries":counts[4],"new_skipped":counts[5],
        "baseline_point_containment_calls":baseline_calls,
        "remaining_point_containment_calls":calls,
        "avoided_point_containment_calls":skipped,
        "avoided_fraction":if baseline_calls > 0 { Some(skipped as f64 / baseline_calls as f64) } else { None },
        "denominator":"point containment calls the unclassified gate executes on the same Poisson cloud; excludes envelope construction"
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        (1..=64).contains(&args.cases) && (1..=10).contains(&args.repeats),
        "bounded cases/repeats required"
    );
    ensure!(
        (1..=1024).contains(&args.max_attempts),
        "bounded positive max-attempts required"
    );
    ensure!(
        args.max_expected_points.is_finite() && args.max_expected_points > 0.,
        "invalid expected-point budget"
    );
    ensure!(
        !args.out.exists(),
        "output already exists; choose a fresh file"
    );
    let config_path = args.config.canonicalize()?;
    let config_bytes = fs::read(&config_path)?;
    let cfg: Config = serde_json::from_slice(&config_bytes)?;
    cfg.validate()?;
    let shape_path = if cfg.shape.is_absolute() {
        cfg.shape.clone()
    } else {
        config_path
            .parent()
            .context("config parent")?
            .join(&cfg.shape)
    }
    .canonicalize()?;
    let shape_bytes = fs::read(&shape_path)?;
    let start = cpu_seconds();
    let tree = SphereTree::new(serde_json::from_slice::<Shape>(&shape_bytes)?)?;
    let shape_tree_cpu = cpu_seconds() - start;
    let wall = match cfg.boundary {
        Boundary::Periodic => None,
        Boundary::Spherical { radius } => Some(Container::new(radius, &tree)?),
    };
    let lambda = cfg.poisson_lambda_ratio * cfg.reservoir_density;
    ensure!(
        lambda > 0.,
        "use a positive-activity configuration for the gate benchmark"
    );
    let start = cpu_seconds();
    let mut cache = BodyEnvelopeCache::new(&tree, cfg.depletant_radius)?;
    let cache_initialization_cpu = cpu_seconds() - start;
    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut records = Vec::<Value>::new();
    let mut total_expected_points = 0.;
    let mut total_reference_envelope_cpu = 0.;
    let mut total_cached_envelope_cpu = 0.;
    let mut total_reference_gate_cpu = 0.;
    let mut total_cached_gate_cpu = 0.;
    let mut total_certified_gate_cpu = 0.;
    let mut total_queries = [0_u64; 6];
    let mut eligible_cases = 0;
    let mut nonempty_cases = 0;
    for case in 0..args.cases {
        // Visit all saved body labels before revisiting any label.
        let i = case % cfg.initial_poses.len();
        let old = cfg.initial_poses[i];
        let mut attempts = Vec::new();
        let mut selected = None;
        let setup_start = cpu_seconds();
        for attempt in 0..args.max_attempts {
            let new = candidate(&mut rng, &cfg, old);
            let env = environment(&tree, &cfg, i, new)?;
            ensure!(
                env.hard_valid(old) && wall.as_ref().is_none_or(|w| w.contains(old)),
                "saved old pose is hard invalid"
            );
            let valid = env.hard_valid(new) && wall.as_ref().is_none_or(|w| w.contains(new));
            attempts.push(json!({"attempt":attempt,"new_pose":new,"hard_valid":valid}));
            if valid {
                selected = Some(new);
                break;
            }
        }
        let endpoint_setup_cpu = cpu_seconds() - setup_start;
        let Some(new) = selected else {
            records.push(json!({"case":case,"moving_index":i,"old_pose":old,"attempts":attempts,
                "status":"no_hard_valid_endpoint_within_declared_budget","endpoint_setup_cpu_seconds":endpoint_setup_cpu}));
            continue;
        };
        let env = environment(&tree, &cfg, i, new)?;
        let start = cpu_seconds();
        let reference = Envelope::build(&env, old, new, cfg.endpoint_gate)?;
        let reference_first_cpu = cpu_seconds() - start;
        let start = cpu_seconds();
        let mut cold_cache = BodyEnvelopeCache::new(&tree, cfg.depletant_radius)?;
        let cold_envelope = cold_cache.build(&env, old, new, cfg.endpoint_gate)?;
        let cold_cpu = cpu_seconds() - start;
        ensure!(
            same_envelope(&reference, &cold_envelope),
            "cold envelope mismatch in case {case}"
        );
        let start = cpu_seconds();
        let warmup = cache.build(&env, old, new, cfg.endpoint_gate)?;
        let shared_cache_first_visit_cpu = cpu_seconds() - start;
        ensure!(
            same_envelope(&reference, &warmup),
            "warmup envelope mismatch in case {case}"
        );
        let expected_points = (lambda + cfg.reservoir_density) * reference.volume;
        // Three timed arms plus one profiled call and its independent reference.
        let required_points = expected_points * (3 * args.repeats + 2) as f64;
        let gate_enabled = total_expected_points + required_points <= args.max_expected_points;
        if gate_enabled {
            total_expected_points += required_points;
        }
        eligible_cases += 1;
        nonempty_cases += usize::from(reference.volume > 0.);
        let mut samples = Vec::new();
        for repeat in 0..args.repeats {
            // Alternate order to reduce systematic cache/frequency timing bias.
            let (reference_envelope_cpu, cached_envelope_cpu) = if (case + repeat) % 2 == 0 {
                let t = cpu_seconds();
                let a = Envelope::build(&env, old, new, cfg.endpoint_gate)?;
                let ta = cpu_seconds() - t;
                let t = cpu_seconds();
                let b = cache.build(&env, old, new, cfg.endpoint_gate)?;
                let tb = cpu_seconds() - t;
                ensure!(same_envelope(&a, &b), "warm envelope mismatch");
                (ta, tb)
            } else {
                let t = cpu_seconds();
                let b = cache.build(&env, old, new, cfg.endpoint_gate)?;
                let tb = cpu_seconds() - t;
                let t = cpu_seconds();
                let a = Envelope::build(&env, old, new, cfg.endpoint_gate)?;
                let ta = cpu_seconds() - t;
                ensure!(same_envelope(&a, &b), "warm envelope mismatch");
                (ta, tb)
            };
            total_reference_envelope_cpu += reference_envelope_cpu;
            total_cached_envelope_cpu += cached_envelope_cpu;
            let mut row = json!({"repeat":repeat,"reference_envelope_cpu_seconds":reference_envelope_cpu,
                "cached_envelope_cpu_seconds":cached_envelope_cpu,"envelope_bitwise_equal":true});
            if gate_enabled {
                let seed = args
                    .seed
                    .wrapping_add(0x9e3779b97f4a7c15)
                    .wrapping_add((case * args.repeats + repeat) as u64);
                let mut gates: [Option<GateResult>; 3] = std::array::from_fn(|_| None);
                let mut times = [0.; 3];
                let mut tails = [[0_u64; 8]; 3];
                let order: [usize; 3] = std::array::from_fn(|k| (case + repeat + k) % 3);
                for arm in order {
                    let mut gate_rng = StdRng::seed_from_u64(seed);
                    let t = cpu_seconds();
                    let gate = match arm {
                        0 => depletion::sample(
                            &mut gate_rng,
                            &env,
                            old,
                            new,
                            lambda,
                            cfg.reservoir_density,
                            cfg.endpoint_gate,
                        )?,
                        1 => cache.sample_unclassified(
                            &mut gate_rng,
                            &env,
                            old,
                            new,
                            lambda,
                            cfg.reservoir_density,
                            cfg.endpoint_gate,
                        )?,
                        2 => cache.sample(
                            &mut gate_rng,
                            &env,
                            old,
                            new,
                            lambda,
                            cfg.reservoir_density,
                            cfg.endpoint_gate,
                        )?,
                        _ => unreachable!(),
                    };
                    times[arm] = cpu_seconds() - t;
                    tails[arm] = std::array::from_fn(|_| gate_rng.random());
                    gates[arm] = Some(gate);
                }
                let a = gates[0].as_ref().expect("reference gate");
                ensure!(
                    same_gate(a, gates[1].as_ref().expect("cached gate"))
                        && same_gate(a, gates[2].as_ref().expect("certified gate")),
                    "gate mismatch in case {case}, repeat {repeat}"
                );
                ensure!(
                    tails[0] == tails[1] && tails[0] == tails[2],
                    "RNG continuation mismatch"
                );
                total_reference_gate_cpu += times[0];
                total_cached_gate_cpu += times[1];
                total_certified_gate_cpu += times[2];
                row["reference_gate_cpu_seconds"] = json!(times[0]);
                row["cached_gate_cpu_seconds"] = json!(times[1]);
                row["certified_gate_cpu_seconds"] = json!(times[2]);
                row["gate_arm_order"] = json!(order);
                row["gate"] = json!(a);
                row["gate_seed"] = json!(seed);
                row["gate_bitwise_equal"] = json!(true);
                row["rng_continuation_equal"] = json!(true);
                row["rng_continuation_u64"] = json!(tails[0]);
            } else {
                row["gate_status"] = json!("not_run_expected_point_budget");
            }
            samples.push(row);
        }
        // Counters use a distinct frozen cloud and are outside measured loops.
        let profile = if gate_enabled {
            let seed = args
                .seed
                .wrapping_add(0xd1b54a32d192ed03)
                .wrapping_add(case as u64);
            let mut profile_rng = StdRng::seed_from_u64(seed);
            let (profile_gate, queries) = cache.sample_profiled(
                &mut profile_rng,
                &env,
                old,
                new,
                lambda,
                cfg.reservoir_density,
                cfg.endpoint_gate,
            )?;
            let mut reference_rng = StdRng::seed_from_u64(seed);
            let reference_gate = depletion::sample(
                &mut reference_rng,
                &env,
                old,
                new,
                lambda,
                cfg.reservoir_density,
                cfg.endpoint_gate,
            )?;
            let profile_tail: [u64; 8] = std::array::from_fn(|_| profile_rng.random());
            let reference_tail: [u64; 8] = std::array::from_fn(|_| reference_rng.random());
            ensure!(
                same_gate(&profile_gate, &reference_gate) && profile_tail == reference_tail,
                "profile gate or RNG continuation mismatch in case {case}"
            );
            let counts = [
                queries.body_queries,
                queries.body_skipped,
                queries.old_queries,
                queries.old_skipped,
                queries.new_queries,
                queries.new_skipped,
            ];
            for (total, value) in total_queries.iter_mut().zip(counts) {
                *total += value;
            }
            json!({"seed":seed,"gate":profile_gate,"queries":queries,
                "containment_calls":query_summary(counts),
                "gate_bitwise_equal":true,"rng_continuation_equal":true,
                "rng_continuation_u64":profile_tail})
        } else {
            json!({"status":"not_run_expected_point_budget"})
        };
        records.push(json!({"case":case,"moving_index":i,"old_pose":old,"new_pose":new,"attempts":attempts,
            "status":"measured","endpoint_setup_cpu_seconds":endpoint_setup_cpu,"spectator_labels":env.labels,
            "reference_first_envelope_cpu_seconds":reference_first_cpu,"cold_cache_build_cpu_seconds":cold_cpu,
            "shared_cache_first_visit_cpu_seconds":shared_cache_first_visit_cpu,"cold_cache_nodes":cold_cache.node_count(),
            "shared_cache_nodes":cache.node_count(),"envelope_volume":reference.volume,"retained_cells":reference.cells.len(),
            "created_cells":reference.created,"expected_raw_points_per_gate":expected_points,"samples":samples,
            "profile":profile}));
    }
    let endpoints: Vec<_> = records
        .iter()
        .map(|r| {
            json!({"case":r["case"],"moving_index":r["moving_index"],
        "old_pose":r["old_pose"],"new_pose":r.get("new_pose"),"attempts":r["attempts"]})
        })
        .collect();
    let report = json!({
        "schema":"envelope-cache-fixed-endpoints-v2","scope":"Synthetic local endpoints from a frozen saved configuration, conditioned on hard validity with bounded retries; no Monte Carlo evolution, no acceptance or mixing-speed claim",
        "config_path":config_path,"config_sha256":hash_bytes(&config_bytes),"config":serde_json::from_slice::<Value>(&config_bytes)?,
        "shape_path":shape_path,"shape_sha256":hash_bytes(&shape_bytes),"shape_atoms":tree.shape.atoms.len(),
        "benchmark_source_sha256":hash_bytes(include_bytes!("envelope-cache-benchmark.rs")),
        "benchmark_source":include_str!("envelope-cache-benchmark.rs"),
        "source_bundle_sha256":hash_bytes(include_bytes!(concat!(env!("OUT_DIR"), "/source-bundle.json"))),
        "executable_sha256":hash_bytes(&fs::read(std::env::current_exe()?)?),
        "arguments":{"cases":args.cases,"repeats":args.repeats,"seed":args.seed,"max_attempts":args.max_attempts,"max_expected_points":args.max_expected_points},
        "endpoint_cases_sha256":hash_bytes(&serde_json::to_vec(&endpoints)?),"shape_tree_cpu_seconds":shape_tree_cpu,
        "cache_initialization_cpu_seconds":cache_initialization_cpu,"cache_nodes":cache.node_count(),
        "eligible_cases":eligible_cases,"nonempty_envelope_cases":nonempty_cases,"summed_expected_gate_points":total_expected_points,
        "gate_arms":["reference_uncached","cached_unclassified","cached_certified"],
        "expected_point_budget_includes":"three timed arms per repeat plus one profiled gate and its reference per endpoint",
        "profile_containment_calls":query_summary(total_queries),
        "timing_totals":{"reference_envelope_cpu_seconds":total_reference_envelope_cpu,"cached_envelope_cpu_seconds":total_cached_envelope_cpu,
            "reference_gate_cpu_seconds":total_reference_gate_cpu,"cached_gate_cpu_seconds":total_cached_gate_cpu,
            "certified_gate_cpu_seconds":total_certified_gate_cpu},
        "cases":records,
    });
    if let Some(parent) = args.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    save(&args.out, &report)?;
    println!(
        "{}",
        json!({"out":args.out,"eligible_cases":eligible_cases,"nonempty_envelope_cases":nonempty_cases,
        "timing_totals":report["timing_totals"],"cache_nodes":cache.node_count()})
    );
    Ok(())
}
