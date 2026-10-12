//! Seeded growth with exact morphometric solvation plus a pairwise atom square well
//! (`tetramer_mc::morph_mc`). `calibrate` counts square-well pairs of environments
//! (neighbour poses relative to body 0); `run` samples a fixed seed plus free bodies.
use anyhow::{Result, ensure};
use clap::{Parser, Subcommand};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde_json::{Value, json};
use std::{fs, io::Write, path::PathBuf, time::Instant};
use tetramer_mc::{
    geometry::{Placed, Shape, SphereTree},
    math::{Pose, norm, sub},
    morph_mc::{MorphMc, MoveSettings, SquareWell},
    solvation::{GaussConvention, SolvationModel},
    spherical::Container,
};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Square-well atom pairs between body 0 and each environment's neighbours.
    Calibrate {
        #[arg(long)]
        shape: PathBuf,
        #[arg(long)]
        environments: PathBuf,
        #[arg(long, default_value_t = 1.0)]
        width: f64,
    },
    /// Replica exchange in epsilon: one replica per well depth on a uniform ladder,
    /// neighbour swaps accepted with min(1, exp((eps_hi - eps_lo)(n_lo - n_hi))) where n is
    /// the replica's square-well pair count (the solvation term is the same in both).
    Rex {
        #[arg(long)]
        shape: PathBuf,
        #[arg(long)]
        seed_config: PathBuf,
        #[arg(long, default_value_t = 32)]
        seed_count: usize,
        #[arg(long)]
        free_bodies: usize,
        #[arg(long)]
        concentration_um: f64,
        #[arg(long)]
        eps_min: f64,
        #[arg(long)]
        eps_max: f64,
        #[arg(long)]
        replicas: usize,
        #[arg(long, default_value_t = 1.0)]
        width: f64,
        #[arg(long)]
        sweeps: usize,
        #[arg(long, default_value_t = 10)]
        swap_every: usize,
        #[arg(long, default_value_t = 500)]
        sample_every: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(long)]
        out: PathBuf,
    },
    Run {
        #[arg(long)]
        shape: PathBuf,
        /// Config whose `initial_poses` (first `seed_count`) form the fixed native seed.
        #[arg(long)]
        seed_config: PathBuf,
        #[arg(long, default_value_t = 32)]
        seed_count: usize,
        #[arg(long)]
        free_bodies: usize,
        /// Free-body concentration in micromolar over the vessel volume.
        #[arg(long)]
        concentration_um: f64,
        #[arg(long)]
        epsilon: f64,
        #[arg(long, default_value_t = 1.0)]
        width: f64,
        /// Hard-sphere morphometric solvation on (default) or off (square well only).
        #[arg(long)]
        no_solvation: bool,
        #[arg(long)]
        sweeps: usize,
        #[arg(long, default_value_t = 50)]
        sample_every: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(long)]
        out: PathBuf,
    },
}

fn shape_tree(path: &PathBuf) -> Result<SphereTree> {
    SphereTree::new(serde_json::from_slice::<Shape>(&fs::read(path)?)?)
}

fn hard_sphere(n: usize, on: bool) -> SolvationModel {
    SolvationModel {
        probe: 1.4,
        pressure: if on { 0.2090 } else { 0. },
        coefficients: vec![if on {
            [-0.1564, 0.1362, -0.0369]
        } else {
            [0.; 3]
        }],
        atom_types: vec![0; n],
        gauss_convention: GaussConvention::PatchGaussBonnet,
    }
}

/// Fixed native seed (first `seed_count` poses of the config, recentred) plus free bodies
/// placed uniformly without contact in a vessel sized for the free-body concentration.
fn start_state(
    tree: &SphereTree,
    seed_config: &PathBuf,
    seed_count: usize,
    free_bodies: usize,
    concentration_um: f64,
    seed: u64,
) -> Result<(Vec<Pose>, Vec<bool>, f64)> {
    let cfg: Value = serde_json::from_slice(&fs::read(seed_config)?)?;
    let mut seedp: Vec<Pose> = serde_json::from_value(cfg["initial_poses"].clone())?;
    seedp.truncate(seed_count);
    let c: [f64; 3] = std::array::from_fn(|k| {
        seedp.iter().map(|p| p.position[k]).sum::<f64>() / seedp.len() as f64
    });
    for p in &mut seedp {
        p.position = sub(p.position, c);
    }
    let rho = concentration_um * 1e-6 * 6.02214076e23 / 1e27; // A^-3
    let volume = free_bodies as f64 / rho;
    let radius = (3. * volume / (4. * std::f64::consts::PI)).cbrt();
    let wall = Container::new(radius, tree)?;
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5eed_5eed);
    let mut poses = seedp.clone();
    let mut placed: Vec<Placed> = poses.iter().map(|&p| Placed::new(p)).collect();
    // Optional free bodies at given poses (recentred seed frame), e.g. lattice sites.
    if let Some(v) = cfg.get("placed_free_poses") {
        let given: Vec<Pose> = serde_json::from_value(v.clone())?;
        ensure!(
            given.len() <= free_bodies,
            "more placed free poses than free bodies"
        );
        for p in given {
            let pl = Placed::new(p);
            ensure!(wall.contains(p), "placed free pose outside the wall");
            ensure!(
                !placed.iter().any(|o| tree.overlaps(&pl, o)),
                "placed free pose overlaps"
            );
            poses.push(p);
            placed.push(pl);
        }
    }
    while poses.len() < seed_count + free_bodies {
        let x: [f64; 3] = std::array::from_fn(|_| (2. * rng.random::<f64>() - 1.) * radius);
        let q: [f64; 4] = std::array::from_fn(|_| StandardNormal.sample(&mut rng));
        let qn = q.iter().map(|v| v * v).sum::<f64>().sqrt();
        let p = Pose {
            position: x,
            orientation: q.map(|v| v / qn),
        };
        if norm(x) > radius || !wall.contains(p) {
            continue;
        }
        // Start free bodies out of contact (centres at least 2 bound + 4 A apart).
        if poses
            .iter()
            .any(|o| norm(sub(o.position, x)) < 2. * tree.bound + 4.)
        {
            continue;
        }
        let pl = Placed::new(p);
        if placed.iter().any(|o| tree.overlaps(&pl, o)) {
            continue;
        }
        poses.push(p);
        placed.push(pl);
    }
    let fixed = (0..poses.len()).map(|i| i < seed_count).collect();
    Ok((poses, fixed, radius))
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Calibrate {
            shape,
            environments,
            width,
        } => {
            let tree = shape_tree(&shape)?;
            let n = tree.shape.atoms.len();
            let envs: Vec<Value> = serde_json::from_slice(&fs::read(&environments)?)?;
            let identity = Pose {
                position: [0.; 3],
                orientation: [1., 0., 0., 0.],
            };
            let mut rows = Vec::new();
            for e in envs {
                let nb: Vec<Pose> = serde_json::from_value(e["poses"].clone())?;
                let mut poses = vec![identity];
                poses.extend(&nb);
                let wall = Container::new(1e6, &tree)?;
                let fixed = vec![true; poses.len()];
                let mc = MorphMc::new(
                    tree.clone(),
                    wall,
                    hard_sphere(n, false),
                    SquareWell { epsilon: 1., width },
                    poses,
                    fixed,
                )?;
                rows.push(json!({"label": e["label"], "pairs": mc.well_pairs(0, identity)}));
            }
            println!("{}", serde_json::to_string_pretty(&rows)?);
        }
        Cmd::Rex {
            shape,
            seed_config,
            seed_count,
            free_bodies,
            concentration_um,
            eps_min,
            eps_max,
            replicas,
            width,
            sweeps,
            swap_every,
            sample_every,
            seed,
            out,
        } => {
            ensure!(
                replicas >= 2 && swap_every > 0 && sample_every % swap_every == 0,
                "invalid replica settings"
            );
            fs::create_dir_all(&out)?;
            let tree = shape_tree(&shape)?;
            let n = tree.shape.atoms.len();
            let ladder: Vec<f64> = (0..replicas)
                .map(|k| eps_min + (eps_max - eps_min) * k as f64 / (replicas - 1) as f64)
                .collect();
            let mut mcs = Vec::new();
            let mut radius = 0.;
            for k in 0..replicas {
                let (poses, fixed, r) = start_state(
                    &tree,
                    &seed_config,
                    seed_count,
                    free_bodies,
                    concentration_um,
                    seed * 1000 + k as u64,
                )?;
                radius = r;
                let wall = Container::new(r, &tree)?;
                mcs.push(MorphMc::new(
                    tree.clone(),
                    wall,
                    hard_sphere(n, true),
                    SquareWell {
                        epsilon: ladder[k],
                        width,
                    },
                    poses,
                    fixed,
                )?);
            }
            let mut rngs: Vec<StdRng> = (0..replicas)
                .map(|k| StdRng::seed_from_u64(seed * 7919 + k as u64))
                .collect();
            let mut master = StdRng::seed_from_u64(seed ^ 0xabcdef);
            // level -> replica holding it
            let mut holder: Vec<usize> = (0..replicas).collect();
            let mut swaps = vec![(0u64, 0u64); replicas - 1];
            let meta = json!({"shape": shape, "seed_config": seed_config, "seed_count": seed_count,
                "free_bodies": free_bodies, "concentration_um": concentration_um, "vessel_radius": radius,
                "ladder": ladder, "width": width, "sweeps": sweeps, "swap_every": swap_every,
                "sample_every": sample_every, "seed": seed,
                "moves": "uniform free body; 0.5 tiny (0.1 A, 1 deg) / 0.45 medium (0.5 A, 4 deg) / 0.05 uniform reinsertion",
                "energy": "exact hard-sphere White Bear II morphometry (eta 0.383, R 1.4) + atom square well; exact hard cores and wall"});
            fs::write(
                out.join("config.json"),
                serde_json::to_string_pretty(&meta)?,
            )?;
            let mut trajs = Vec::new();
            for l in 0..replicas {
                let d = out.join(format!("level{l:02}"));
                fs::create_dir_all(&d)?;
                trajs.push(fs::File::create(d.join("trajectory.jsonl"))?);
            }
            let mut log = fs::File::create(out.join("progress.jsonl"))?;
            let t0 = Instant::now();
            let tiny = MoveSettings {
                translation: 0.1,
                rotation: 1f64.to_radians(),
                reinsertion_probability: 0.,
            };
            let medium = MoveSettings {
                translation: 0.5,
                rotation: 4f64.to_radians(),
                reinsertion_probability: 0.,
            };
            let reinsert = MoveSettings {
                translation: 0.,
                rotation: 0.,
                reinsertion_probability: 1.,
            };
            let mut sweep = 0;
            let mut parity = 0;
            loop {
                if sweep % sample_every == 0 {
                    for l in 0..replicas {
                        let r = holder[l];
                        writeln!(
                            trajs[l],
                            "{}",
                            json!({"sweep": sweep, "replica": r, "epsilon": ladder[l],
                            "pairs": mcs[r].pair_count(), "solvation": mcs[r].solvation_energy(), "poses": mcs[r].poses()})
                        )?;
                        trajs[l].flush()?;
                    }
                    writeln!(
                        log,
                        "{}",
                        json!({"sweep": sweep, "seconds": t0.elapsed().as_secs_f64(),
                        "holder": holder, "pairs": holder.iter().map(|&r| mcs[r].pair_count()).collect::<Vec<_>>(),
                        "swap_accepted": swaps.iter().map(|s| s.1).collect::<Vec<_>>(),
                        "swap_attempts": swaps.iter().map(|s| s.0).collect::<Vec<_>>()})
                    )?;
                    log.flush()?;
                }
                if sweep >= sweeps {
                    break;
                }
                std::thread::scope(|sc| -> Result<()> {
                    let handles: Vec<_> = mcs
                        .iter_mut()
                        .zip(rngs.iter_mut())
                        .map(|(mc, rng)| {
                            sc.spawn(move || -> Result<()> {
                                for _ in 0..swap_every * free_bodies {
                                    let u = rng.random::<f64>();
                                    let mv = if u < 0.5 {
                                        tiny
                                    } else if u < 0.95 {
                                        medium
                                    } else {
                                        reinsert
                                    };
                                    mc.step(rng, mv)?;
                                }
                                Ok(())
                            })
                        })
                        .collect();
                    for h in handles {
                        h.join()
                            .map_err(|_| anyhow::anyhow!("replica thread panicked"))??;
                    }
                    Ok(())
                })?;
                sweep += swap_every;
                for l in (parity..replicas - 1).step_by(2) {
                    let (a, b) = (holder[l], holder[l + 1]);
                    let x = (ladder[l + 1] - ladder[l])
                        * (mcs[a].pair_count() - mcs[b].pair_count()) as f64;
                    swaps[l].0 += 1;
                    if x >= 0. || master.random::<f64>() < x.exp() {
                        swaps[l].1 += 1;
                        holder.swap(l, l + 1);
                        mcs[a].set_epsilon(ladder[l + 1]);
                        mcs[b].set_epsilon(ladder[l]);
                    }
                }
                parity = 1 - parity;
            }
            let checks: Vec<Value> = mcs
                .iter()
                .map(|mc| {
                    let (sf, sw) = mc.recompute();
                    json!({"solvation_cached": mc.solvation_energy(), "solvation_recomputed": sf,
                    "well_cached": mc.well_energy(), "well_recomputed": sw})
                })
                .collect();
            fs::write(
                out.join("final_check.json"),
                serde_json::to_string_pretty(&checks)?,
            )?;
        }
        Cmd::Run {
            shape,
            seed_config,
            seed_count,
            free_bodies,
            concentration_um,
            epsilon,
            width,
            no_solvation,
            sweeps,
            sample_every,
            seed,
            out,
        } => {
            fs::create_dir_all(&out)?;
            let tree = shape_tree(&shape)?;
            let n = tree.shape.atoms.len();
            let (poses, fixed, radius) = start_state(
                &tree,
                &seed_config,
                seed_count,
                free_bodies,
                concentration_um,
                seed,
            )?;
            let wall = Container::new(radius, &tree)?;
            let mut rng = StdRng::seed_from_u64(seed);
            let model = hard_sphere(n, !no_solvation);
            let t0 = Instant::now();
            let mut mc = MorphMc::new(
                tree,
                wall,
                model,
                SquareWell { epsilon, width },
                poses,
                fixed,
            )?;
            let meta = json!({"shape": shape, "seed_config": seed_config, "seed_count": seed_count,
                "free_bodies": free_bodies, "concentration_um": concentration_um, "vessel_radius": radius,
                "epsilon": epsilon, "width": width, "solvation": !no_solvation, "sweeps": sweeps,
                "sample_every": sample_every, "seed": seed,
                "moves": "uniform free body; 0.5 tiny (0.1 A, 1 deg) / 0.45 medium (0.5 A, 4 deg) / 0.05 uniform reinsertion",
                "energy": "exact hard-sphere White Bear II morphometry (eta 0.383, R 1.4) + atom square well; exact hard cores and wall"});
            fs::write(
                out.join("config.json"),
                serde_json::to_string_pretty(&meta)?,
            )?;
            let mut traj = fs::File::create(out.join("trajectory.jsonl"))?;
            let mut log = fs::File::create(out.join("progress.jsonl"))?;
            ensure!(free_bodies > 0, "no free bodies");
            let tiny = MoveSettings {
                translation: 0.1,
                rotation: 1f64.to_radians(),
                reinsertion_probability: 0.,
            };
            let medium = MoveSettings {
                translation: 0.5,
                rotation: 4f64.to_radians(),
                reinsertion_probability: 0.,
            };
            let reinsert = MoveSettings {
                translation: 0.,
                rotation: 0.,
                reinsertion_probability: 1.,
            };
            for sweep in 0..=sweeps {
                if sweep % sample_every == 0 {
                    writeln!(traj, "{}", json!({"sweep": sweep, "poses": mc.poses()}))?;
                    let s = mc.stats;
                    writeln!(
                        log,
                        "{}",
                        json!({"sweep": sweep, "seconds": t0.elapsed().as_secs_f64(),
                        "solvation": mc.solvation_energy(), "well": mc.well_energy(),
                        "attempts": s.attempts, "accepted": s.accepted, "hard_rejected": s.hard_rejected,
                        "wall_rejected": s.wall_rejected, "reinsertion_attempts": s.reinsertion_attempts,
                        "reinsertion_accepted": s.reinsertion_accepted})
                    )?;
                    log.flush()?;
                    traj.flush()?;
                }
                if sweep == sweeps {
                    break;
                }
                for _ in 0..free_bodies {
                    let u = rng.random::<f64>();
                    let mv = if u < 0.5 {
                        tiny
                    } else if u < 0.95 {
                        medium
                    } else {
                        reinsert
                    };
                    mc.step(&mut rng, mv)?;
                }
            }
            let (sf, sw) = mc.recompute();
            fs::write(
                out.join("final_check.json"),
                serde_json::to_string_pretty(&json!({
                "solvation_cached": mc.solvation_energy(), "solvation_recomputed": sf,
                "well_cached": mc.well_energy(), "well_recomputed": sw}))?,
            )?;
        }
    }
    Ok(())
}
