//! Fixed-state diagnosis of overlap compensation under production-like GCA axes.
//! No physical moves, component coin draws, or modified bond rules are executed.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use rand::{RngExt, SeedableRng, rngs::StdRng, seq::SliceRandom};
use rand_distr::{Distribution, StandardNormal};
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::PathBuf};
use tetramer_mc::{
    gca_overlap_diagnostic::estimate_pair,
    geometry::{Cell, Placed, Shape, SphereTree},
    math::{Vec3, norm, scale, sub},
    simulation::{Boundary, Config, cpu_seconds, hash_bytes, save},
    spherical::{self, Container, HalfTurn},
};

#[derive(Parser, Serialize)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 32)]
    axes: usize,
    #[arg(long, default_value_t = 32)]
    max_probes_per_stratum: usize,
    #[arg(long, default_value_t = 4096)]
    draws_per_population: usize,
    #[arg(long, default_value_t = 4)]
    populations: usize,
    #[arg(long, default_value_t = 2047)]
    max_cells: usize,
    #[arg(long, default_value_t = 2026092701)]
    seed: u64,
}
fn named_seed(seed: u64, index: usize, label: &str) -> u64 {
    let mut h = Sha256::new();
    h.update(b"tetramer-mc-gca-overlap-diagnostic-v1");
    h.update(seed.to_le_bytes());
    h.update((index as u64).to_le_bytes());
    h.update(label.as_bytes());
    u64::from_le_bytes(h.finalize()[..8].try_into().unwrap())
}
fn world_bounds(body: &Placed, root: Cell, bound: f64) -> Cell {
    let corners: [Vec3; 8] = std::array::from_fn(|mask| {
        body.apply(std::array::from_fn(|k| {
            if mask & (1 << k) == 0 {
                root.lo[k]
            } else {
                root.hi[k]
            }
        }))
    });
    let guard = 1024. * f64::EPSILON * (1. + norm(body.position) + bound);
    Cell {
        lo: std::array::from_fn(|k| {
            corners.iter().map(|p| p[k]).fold(f64::INFINITY, f64::min) - guard
        }),
        hi: std::array::from_fn(|k| {
            corners
                .iter()
                .map(|p| p[k])
                .fold(f64::NEG_INFINITY, f64::max)
                + guard
        }),
    }
}
fn possible(a: &Placed, ab: Cell, b: &Placed, bb: Cell, radius: f64) -> bool {
    let guard = 1024. * f64::EPSILON * (1. + norm(a.position) + norm(b.position) + radius);
    norm(sub(a.position, b.position)) <= 2. * radius + guard
        && (0..3).all(|k| ab.lo[k] < bb.hi[k] && bb.lo[k] < ab.hi[k])
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!((1..=256).contains(&args.axes), "axes must be 1..256");
    ensure!(
        (1..=256).contains(&args.max_probes_per_stratum),
        "probe cap must be 1..256"
    );
    ensure!(
        (1..=1_048_576).contains(&args.draws_per_population)
            && (1..=16).contains(&args.populations),
        "invalid fixed population allocation"
    );
    ensure!(
        (1..=65535).contains(&args.max_cells),
        "invalid traversal budget"
    );
    let census_path = args.out.with_extension("census.json");
    let probe_path = args.out.with_extension("probes.jsonl");
    ensure!(
        [&args.out, &census_path, &probe_path]
            .iter()
            .all(|p| !p.exists()),
        "diagnostic output already exists; choose a fresh path"
    );
    let start = cpu_seconds();
    let config_path = args.config.canonicalize()?;
    let config_raw = fs::read(&config_path)?;
    let config: Config = serde_json::from_slice(&config_raw)?;
    config.validate()?;
    let radius = match config.boundary {
        Boundary::Spherical { radius } => radius,
        _ => anyhow::bail!("this diagnostic requires the current spherical GCA ensemble"),
    };
    ensure!(
        config.fixed_body_indices.is_empty(),
        "diagnostic expects an all-mobile configuration"
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
    let old: Vec<_> = config
        .initial_poses
        .iter()
        .copied()
        .map(Placed::new)
        .collect();
    let bound = tree.bound + config.depletant_radius;
    let root = tree.bounds(config.depletant_radius);
    let old_bounds: Vec<_> = old.iter().map(|p| world_bounds(p, root, bound)).collect();
    let mut pairs = Vec::new();
    let mut pair_records = Vec::new();
    for i in 0..old.len() {
        for j in i + 1..old.len() {
            if possible(&old[i], old_bounds[i], &old[j], old_bounds[j], bound) {
                pairs.push((i, j));
                pair_records.push(
                    json!({"i":i,"j":j,"center_distance":norm(sub(old[i].position,old[j].position)),
                "radii_from_vessel_center":[norm(old[i].position),norm(old[j].position)]}),
                );
            }
        }
    }
    let mut axes = Vec::new();
    let mut shadows = Vec::new();
    let mut cross_cases = Vec::new();
    let mut disjoint_cases = Vec::new();
    let mut total_forced = 0usize;
    for axis_index in 0..args.axes {
        let seed = named_seed(args.seed, axis_index, "axis");
        let mut rng = StdRng::seed_from_u64(seed);
        let _: f64 = rng.random(); // Production consumes the GCA scheduling draw first.
        let v: Vec3 = std::array::from_fn(|_| StandardNormal.sample(&mut rng));
        let axis = scale(v, 1. / norm(v));
        let transform = HalfTurn::new(axis)?;
        let (groups, shadow_poses) =
            spherical::hard_components(&tree, &config.initial_poses, transform)?;
        let shadow: Vec<_> = shadow_poses.iter().copied().map(Placed::new).collect();
        let bounds: Vec<_> = shadow
            .iter()
            .map(|p| world_bounds(p, root, bound))
            .collect();
        let mut labels = vec![0; old.len()];
        for (k, g) in groups.iter().enumerate() {
            for &i in g {
                labels[i] = k;
            }
        }
        let mut records = Vec::new();
        for (pair_index, &(i, j)) in pairs.iter().enumerate() {
            let status = if labels[i] == labels[j] {
                total_forced += 1;
                "hard_connected"
            } else if possible(&old[i], old_bounds[i], &shadow[j], bounds[j], bound)
                || possible(&shadow[i], bounds[i], &old[j], old_bounds[j], bound)
            {
                cross_cases.push((axis_index, pair_index));
                "cross_possible"
            } else {
                disjoint_cases.push((axis_index, pair_index));
                "cross_disjoint"
            };
            records.push(json!({"pair_index":pair_index,"status":status}));
        }
        axes.push(json!({"axis_index":axis_index,"axis":axis,"seed":seed,
            "hard_component_sizes":groups.iter().map(Vec::len).collect::<Vec<_>>(),
            "body_hard_component":labels,"pairs":records}));
        shadows.push(shadow);
    }
    // Freeze sampling without looking at any overlap estimates or native labels.
    let mut selected = Vec::new();
    for (stratum, cases) in [
        ("cross_possible", &cross_cases),
        ("cross_disjoint", &disjoint_cases),
    ] {
        let mut candidates = cases.clone();
        let mut rng = StdRng::seed_from_u64(named_seed(args.seed, 0, stratum));
        candidates.shuffle(&mut rng);
        candidates.truncate(args.max_probes_per_stratum);
        let probability = if cases.is_empty() {
            0.
        } else {
            candidates.len() as f64 / cases.len() as f64
        };
        for (a, p) in candidates {
            selected.push((a, p, stratum, probability));
        }
    }
    selected.sort_by_key(|&(a, p, _, _)| (a, p));
    let mut report = json!({"schema":"gca-overlap-compensation-diagnostic-v1","args":args,
        "scope":"Frozen configuration; independent production-law isotropic centered half-turns, geometric census and stratified fixed-draw volume estimates; no physical moves or changed bonds",
        "config_path":config_path,"config_sha256":hash_bytes(&config_raw),"config":serde_json::from_slice::<serde_json::Value>(&config_raw)?,
        "shape_path":shape_path,"shape_sha256":hash_bytes(&shape_raw),"shape_atoms":tree.shape.atoms.len(),
        "source_bundle_sha256":hash_bytes(include_bytes!(concat!(env!("OUT_DIR"),"/source-bundle.json"))),
        "executable_sha256":hash_bytes(&fs::read(std::env::current_exe()?)?),
        "runner_source":include_str!("gca-overlap-diagnostic.rs"),"estimator_source":include_str!("../gca_overlap_diagnostic.rs"),
        "total_body_pairs":old.len()*(old.len()-1)/2,"old_candidate_pairs":pair_records,"axes":axes,
        "census_counts":{"axis_pair_cases":args.axes*pairs.len(),"hard_connected":total_forced,"cross_possible":cross_cases.len(),"cross_disjoint":disjoint_cases.len()},
        "selection_protocol":"Uniform without replacement over all axis-pair cases within each GLOBAL stratum; independent named seeds; frozen before volume probes; unselected cases explicitly remain unmeasured",
        "selected_cases":selected.iter().map(|&(a,p,s,q)|json!({"axis_index":a,"pair_index":p,"stratum":s,"selection_probability":q})).collect::<Vec<_>>(),
        "cloud_allocation_per_probe":args.populations*args.draws_per_population,
        "screen_cpu_seconds":cpu_seconds()-start,"probes":[],"completed":false});
    if let Some(parent) = args.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    save(&census_path, &report)?;
    let mut probe_file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&probe_path)?;
    eprintln!(
        "Screened {} axes and {} old bounding candidates: {} hard-connected, {} cross-possible, {} cross-disjoint; {} frozen probes",
        args.axes,
        pairs.len(),
        total_forced,
        cross_cases.len(),
        disjoint_cases.len(),
        selected.len()
    );
    let mut probes = Vec::new();
    for (index, &(a, p, stratum, probability)) in selected.iter().enumerate() {
        let (i, j) = pairs[p];
        let seed = named_seed(args.seed, a * pairs.len() + p, "volume");
        let before = cpu_seconds();
        let estimate = estimate_pair(
            &tree,
            &old,
            &shadows[a],
            i,
            j,
            config.depletant_radius,
            args.draws_per_population,
            args.populations,
            seed,
            args.max_cells,
        )?;
        let row = json!({"axis_index":a,"pair_index":p,"i":i,"j":j,"stratum":stratum,"selection_probability":probability,"estimate":estimate,"cpu_seconds":cpu_seconds()-before});
        serde_json::to_writer(&mut probe_file, &row)?;
        probe_file.write_all(b"\n")?;
        probe_file.flush()?;
        probes.push(row);
        if (index + 1) % 16 == 0 {
            eprintln!(
                "Completed {}/{} fixed volume probes",
                index + 1,
                selected.len()
            );
        }
    }
    probe_file.sync_all()?;
    report["probes"] = json!(probes);
    report["completed"] = json!(true);
    report["total_cpu_seconds"] = json!(cpu_seconds() - start);
    save(&args.out, &report)?;
    println!(
        "{}",
        json!({"out":args.out,"census":report["census_counts"],"probes":selected.len(),"cpu_seconds":report["total_cpu_seconds"]})
    );
    Ok(())
}
