//! Seeded growth with exact morphometric solvation plus a pairwise atom square well
//! (`tetramer_mc::morph_mc`). `calibrate` counts square-well pairs of environments
//! (neighbour poses relative to body 0); `run` samples a fixed seed plus free bodies.
#![recursion_limit = "256"]
use anyhow::{Result, ensure};
use clap::{Parser, Subcommand};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde_json::{Value, json};
use std::{fs, io::Write, path::PathBuf, time::Instant};
use tetramer_mc::{
    geometry::{Placed, Shape, SphereTree},
    math::{Pose, cayley, matmul, matvec, norm, quaternion, sub, transpose},
    morph_mc::{
        ChargeGroup, ChemistryModel, ClusterSettings, DockSettings, LikeCharge, MemorySettings,
        MorphMc, MoveSettings, SquareWell,
    },
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
        /// Like-charge penalty (eps-independent, so swaps are unchanged); see `run`.
        #[arg(long)]
        charge_groups: Option<PathBuf>,
        #[arg(long, default_value_t = 0.)]
        like_charge_weight: f64,
        #[arg(long, default_value_t = 5.0)]
        like_charge_cutoff: f64,
        /// Optional SE(3)-invariant interface chemistry residual model JSON (eps-independent).
        #[arg(long)]
        chemistry_model: Option<PathBuf>,
        /// Probability per step of an FK-bonded rigid cluster move (`step_cluster`).
        #[arg(long, default_value_t = 0.)]
        cluster_probability: f64,
        /// Maximum cluster size `k_max` for FK rigid cluster moves (0 disables size cap).
        #[arg(long, default_value_t = 6)]
        cluster_max_size: usize,
        /// Inverse-temperature scale `beta_fk` for FK bond recruitment (`1.0` = exact cancellation, `0.0` = singleton ablation).
        #[arg(long, default_value_t = 1.0)]
        cluster_fk_beta: f64,
        /// Fraction of FK cluster moves that propose rigid cluster docking onto spectator anchors.
        #[arg(long, default_value_t = 0.5)]
        cluster_dock_probability: f64,
        /// Fraction of non-docking FK cluster moves that propose rigid uniform reinsertion.
        #[arg(long, default_value_t = 0.2)]
        cluster_reinsertion_probability: f64,
        /// Base COM translation step size (A) for local FK cluster diffusion.
        #[arg(long, default_value_t = 0.5)]
        cluster_translation: f64,
        /// Base COM rotation step size (degrees) for local FK cluster diffusion.
        #[arg(long, default_value_t = 10.0)]
        cluster_rotation_degrees: f64,
        /// Disable `|C|^-0.5` step-size scaling on local FK cluster diffusion (ablation).
        #[arg(long)]
        cluster_no_size_scaling: bool,
        /// Probability per step of a centered proper half-turn GCA move (`step_gca`).
        #[arg(long, default_value_t = 0.)]
        gca_probability: f64,
        /// Inverse-temperature scale `beta_fk` for half-turn GCA bond recruitment.
        #[arg(long, default_value_t = 1.0)]
        gca_fk_beta: f64,
        /// Probability per sweep of a rejection-free common-translation chord shift (`step_center_shift`).
        #[arg(long, default_value_t = 0.)]
        center_shift_probability: f64,
        /// Docking from the contact library (and memory) on this fraction of the steps; the
        /// docking and memory terms do not depend on eps, so swaps are unchanged.
        #[arg(long, default_value_t = 0.)]
        dock_probability: f64,
        #[arg(long, default_value_t = 2.8)]
        dock_library_gap: f64,
        /// Boltzmann inverse-temperature weight `beta` on spectator contact pair energies in the
        /// contact library (`w_ij = exp(-beta * E_pair(i, j))`; `0.0` = uniform contact library).
        #[arg(long, default_value_t = 0.)]
        dock_library_beta: f64,
        /// Number of monomer-only surface-normal-opposed contact poses to generate as a discovery
        /// prior (`0` = off). Uses only the monomer template and its smooth density-gradient normals.
        #[arg(long, default_value_t = 0)]
        surface_prior_samples: usize,
        /// Boltzmann inverse-temperature weight `beta` used to weight surface-normal-opposed prior
        /// poses by their monomer-monomer pair energy (`exp(-beta * E_pair)`).
        #[arg(long, default_value_t = 0.5)]
        surface_prior_beta: f64,
        #[arg(long, default_value_t = 0.15)]
        dock_sigma_translation: f64,
        #[arg(long, default_value_t = 1.0)]
        dock_sigma_rotation_degrees: f64,
        #[arg(long, default_value_t = 0.5)]
        dock_uniform_weight: f64,
        /// Poisson memory per replica (see `run`); > 0 switches it on (needs docking).
        #[arg(long, default_value_t = 0.)]
        memory_lambda_floor: f64,
        #[arg(long, default_value_t = 0.)]
        memory_lambda: f64,
        #[arg(long, default_value_t = 0.)]
        memory_lambda_prior: f64,
        #[arg(long)]
        memory_prior_model: Option<PathBuf>,
        #[arg(long, default_value_t = 0.005)]
        memory_death: f64,
        #[arg(long, default_value_t = 0.3)]
        memory_sigma_translation: f64,
        #[arg(long, default_value_t = 3.0)]
        memory_sigma_rotation_degrees: f64,
        #[arg(long, default_value_t = 20.0)]
        memory_floor_sigma: f64,
        #[arg(long, default_value_t = 1)]
        memory_refresh_per_sweep: usize,
        /// Propose docking moves from the Poisson memory slots only, omitting raw instantaneous
        /// contact library pairs (prevents Polya-urn self-copying of transient 1-bond contacts).
        #[arg(long)]
        memory_only: bool,
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
        #[arg(long, default_value_t = 0.0)]
        epsilon: f64,
        #[arg(long, default_value_t = 0.0)]
        width: f64,
        /// Hard-sphere morphometric solvation on (default) or off (square well only).
        #[arg(long)]
        no_solvation: bool,
        /// JSON list of relative poses (B relative to A) for docking proposals.
        #[arg(long)]
        dock_components: Option<PathBuf>,
        /// Frozen learned relative-pose model (e.g. a blind contact map) for docking instead.
        #[arg(long)]
        dock_model: Option<PathBuf>,
        /// Probability that a step is a docking Metropolis-Hastings step.
        #[arg(long, default_value_t = 0.)]
        dock_probability: f64,
        #[arg(long, default_value_t = 0.25)]
        dock_sigma_translation: f64,
        #[arg(long, default_value_t = 2.0)]
        dock_sigma_rotation_degrees: f64,
        #[arg(long, default_value_t = 0.5)]
        dock_uniform_weight: f64,
        /// Dock from the current contact library (contacts between the other bodies) instead
        /// of a fixed component list or learned model.
        #[arg(long)]
        dock_library: bool,
        /// Atom-gap threshold of a library contact, A.
        #[arg(long, default_value_t = 2.8)]
        dock_library_gap: f64,
        /// Boltzmann inverse-temperature weight `beta` on spectator contact pair energies in the
        /// contact library (`w_ij = exp(-beta * E_pair(i, j))`; `0.0` = uniform contact library).
        #[arg(long, default_value_t = 0.)]
        dock_library_beta: f64,
        /// Number of monomer-only surface-normal-opposed contact poses to generate as a discovery
        /// prior (`0` = off). Uses only the monomer template and its smooth density-gradient normals.
        #[arg(long, default_value_t = 0)]
        surface_prior_samples: usize,
        /// Boltzmann inverse-temperature weight `beta` used to weight surface-normal-opposed prior
        /// poses by their monomer-monomer pair energy (`exp(-beta * E_pair)`).
        #[arg(long, default_value_t = 0.5)]
        surface_prior_beta: f64,
        /// Poisson memory: intensity per contact entry (0 with --memory-lambda-floor 0: off;
        /// needs --dock-library). Memory slots join the docking components; each contact
        /// formed costs 2 lambda kT in the acceptance (exactly compensated in equilibrium).
        #[arg(long, default_value_t = 0.)]
        memory_lambda: f64,
        /// Floor intensity (expected number of floor slots); > 0 switches memory on.
        #[arg(long, default_value_t = 0.)]
        memory_lambda_floor: f64,
        /// Prior intensity: expected number of slots drawn from the prior centres.
        #[arg(long, default_value_t = 0.)]
        memory_lambda_prior: f64,
        /// Frozen relative-pose model (e.g. a blind contact map) whose component centres form
        /// the memory prior.
        #[arg(long)]
        memory_prior_model: Option<PathBuf>,
        /// Death probability of each slot per refresh (memory lifetime ~ 1 / death refreshes).
        #[arg(long, default_value_t = 0.01)]
        memory_death: f64,
        #[arg(long, default_value_t = 0.3)]
        memory_sigma_translation: f64,
        #[arg(long, default_value_t = 3.0)]
        memory_sigma_rotation_degrees: f64,
        /// Translation sd of the floor density, A.
        #[arg(long, default_value_t = 20.0)]
        memory_floor_sigma: f64,
        /// Memory refreshes per sweep.
        #[arg(long, default_value_t = 1)]
        memory_refresh_per_sweep: usize,
        /// Propose docking moves from the Poisson memory slots only, omitting raw instantaneous
        /// contact library pairs (prevents Polya-urn self-copying of transient 1-bond contacts).
        #[arg(long)]
        memory_only: bool,
        /// JSON list of charged groups {atoms, sign, fraction} for the like-charge penalty.
        #[arg(long)]
        charge_groups: Option<PathBuf>,
        /// kT per pair of full like charges in contact (requires --charge-groups).
        #[arg(long, default_value_t = 0.)]
        like_charge_weight: f64,
        /// Closest atom-centre distance for a like-charge contact, A.
        #[arg(long, default_value_t = 5.0)]
        like_charge_cutoff: f64,
        /// Optional SE(3)-invariant interface chemistry residual model JSON.
        #[arg(long)]
        chemistry_model: Option<PathBuf>,
        /// Probability per step of an FK-bonded rigid cluster move (`step_cluster`).
        #[arg(long, default_value_t = 0.)]
        cluster_probability: f64,
        /// Maximum cluster size `k_max` for FK rigid cluster moves (0 disables size cap).
        #[arg(long, default_value_t = 6)]
        cluster_max_size: usize,
        /// Inverse-temperature scale `beta_fk` for FK bond recruitment (`1.0` = exact cancellation, `0.0` = singleton ablation).
        #[arg(long, default_value_t = 1.0)]
        cluster_fk_beta: f64,
        /// Fraction of FK cluster moves that propose rigid cluster docking onto spectator anchors.
        #[arg(long, default_value_t = 0.5)]
        cluster_dock_probability: f64,
        /// Fraction of non-docking FK cluster moves that propose rigid uniform reinsertion.
        #[arg(long, default_value_t = 0.2)]
        cluster_reinsertion_probability: f64,
        /// Base COM translation step size (A) for local FK cluster diffusion.
        #[arg(long, default_value_t = 0.5)]
        cluster_translation: f64,
        /// Base COM rotation step size (degrees) for local FK cluster diffusion.
        #[arg(long, default_value_t = 10.0)]
        cluster_rotation_degrees: f64,
        /// Disable `|C|^-0.5` step-size scaling on local FK cluster diffusion (ablation).
        #[arg(long)]
        cluster_no_size_scaling: bool,
        /// Probability per step of a centered proper half-turn GCA move (`step_gca`).
        #[arg(long, default_value_t = 0.)]
        gca_probability: f64,
        /// Inverse-temperature scale `beta_fk` for half-turn GCA bond recruitment.
        #[arg(long, default_value_t = 1.0)]
        gca_fk_beta: f64,
        /// Probability per sweep of a rejection-free common-translation chord shift (`step_center_shift`).
        #[arg(long, default_value_t = 0.)]
        center_shift_probability: f64,
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

/// Centres and weights of the (virtual, reciprocal-split) components of a frozen relative-pose
/// model, as a memory prior: t = anchor + mean[0..3], R = cayley(mean[3..6] / L) R_anchor;
/// an inverse branch carries half the component weight at the inverse pose.
fn prior_from_model(model: &PathBuf, shape: &PathBuf, radius: f64) -> Result<Vec<(Pose, f64)>> {
    let raw = fs::read(model)?;
    let sha = tetramer_mc::simulation::hash_bytes(&fs::read(shape)?);
    let frozen = tetramer_mc::proposal::FrozenRelativePoseProposal::from_json_str_open(
        std::str::from_utf8(&raw)?,
        [2. * radius; 3],
        0.1,
        &sha,
    )?;
    let l = frozen.angular_length();
    let reciprocal = frozen.reciprocal_components();
    let mut out = Vec::new();
    for (i, c) in frozen.component_parameters().iter().enumerate() {
        let t: [f64; 3] = std::array::from_fn(|k| c.anchor_position[k] + c.mean[k]);
        let r = matmul(cayley(std::array::from_fn(|k| c.mean[k + 3] / l)), c.anchor_rotation);
        let pose = Pose { position: t, orientation: quaternion(r) };
        if reciprocal[i] {
            out.push((pose, c.weight / 2.));
            let ri = transpose(r);
            out.push((
                Pose { position: matvec(ri, t).map(|x| -x), orientation: quaternion(ri) },
                c.weight / 2.,
            ));
        } else {
            out.push((pose, c.weight));
        }
    }
    Ok(out)
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
            charge_groups,
            like_charge_weight,
            like_charge_cutoff,
            chemistry_model,
            cluster_probability,
            cluster_max_size,
            cluster_fk_beta,
            cluster_dock_probability,
            cluster_reinsertion_probability,
            cluster_translation,
            cluster_rotation_degrees,
            cluster_no_size_scaling,
            gca_probability,
            gca_fk_beta,
            center_shift_probability,
            dock_probability,
            dock_library_gap,
            dock_library_beta,
            surface_prior_samples,
            surface_prior_beta,
            dock_sigma_translation,
            dock_sigma_rotation_degrees,
            dock_uniform_weight,
            memory_lambda_floor,
            memory_lambda,
            memory_lambda_prior,
            memory_prior_model,
            memory_death,
            memory_sigma_translation,
            memory_sigma_rotation_degrees,
            memory_floor_sigma,
            memory_refresh_per_sweep,
            memory_only,
            sample_every,
            seed,
            out,
        } => {
            ensure!(
                replicas >= 2 && swap_every > 0 && sample_every % swap_every == 0,
                "invalid replica settings"
            );
            ensure!(
                like_charge_weight == 0. || charge_groups.is_some(),
                "--like-charge-weight needs --charge-groups"
            );
            let groups: Vec<ChargeGroup> = match &charge_groups {
                Some(f) => serde_json::from_slice(&fs::read(f)?)?,
                None => Vec::new(),
            };
            let chem: ChemistryModel = match &chemistry_model {
                Some(f) => serde_json::from_slice(&fs::read(f)?)?,
                None => ChemistryModel::default(),
            };
            fs::create_dir_all(&out)?;
            let tree = shape_tree(&shape)?;
            let n = tree.shape.atoms.len();
            let ladder: Vec<f64> = (0..replicas)
                .map(|k| eps_min + (eps_max - eps_min) * k as f64 / (replicas - 1) as f64)
                .collect();
            let mut mcs = Vec::new();
            let mut radius = 0.;
            let mut rngs: Vec<StdRng> = (0..replicas)
                .map(|k| StdRng::seed_from_u64(seed * 7919 + k as u64))
                .collect();
            ensure!(
                memory_lambda_floor == 0. || dock_probability > 0.,
                "rex memory needs docking (--dock-probability > 0)"
            );
            ensure!(
                !memory_only || memory_lambda_floor > 0.,
                "--memory-only requires --memory-lambda-floor > 0"
            );
            let mut dock_prior_components = Vec::new();
            let cs = ClusterSettings {
                max_size: cluster_max_size,
                fk_beta: cluster_fk_beta,
                dock_probability: cluster_dock_probability,
                reinsertion_probability: cluster_reinsertion_probability,
                translation: cluster_translation,
                rotation: cluster_rotation_degrees.to_radians(),
                size_scaling: !cluster_no_size_scaling,
            };
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
                )?
                .with_like_charge(LikeCharge {
                    weight: like_charge_weight,
                    cutoff: like_charge_cutoff,
                    groups: groups.clone(),
                })?
                .with_chemistry_model(chem.clone())?);
                if dock_probability > 0. || (cluster_probability > 0. && cluster_dock_probability > 0.) {
                    let mc = mcs.pop().expect("replica");
                    let mut mc = mc.with_contact_library(dock_library_gap)?;
                    let surf_prior = if surface_prior_samples > 0 {
                        let mut prng = StdRng::seed_from_u64(seed ^ 0x50fa_ce00);
                        mc.generate_surface_normal_prior(&mut prng, surface_prior_samples, surface_prior_beta)
                    } else {
                        Vec::new()
                    };
                    if k == 0 && memory_lambda_floor == 0. && !surf_prior.is_empty() {
                        dock_prior_components = surf_prior.iter().map(|&(p, _)| p).collect();
                    }
                    if memory_lambda_floor > 0. {
                        let mut prior = match &memory_prior_model {
                            Some(f) => prior_from_model(f, &shape, r)?,
                            None => Vec::new(),
                        };
                        prior.extend(surf_prior);
                        let eff_lambda_prior = if memory_lambda_prior > 0. {
                            memory_lambda_prior
                        } else if !prior.is_empty() {
                            2.0
                        } else {
                            0.0
                        };
                        mc = mc.with_memory(
                            MemorySettings {
                                lambda_floor: memory_lambda_floor,
                                lambda_prior: eff_lambda_prior,
                                lambda: memory_lambda,
                                death: memory_death,
                                sigma_translation: memory_sigma_translation,
                                sigma_rotation: memory_sigma_rotation_degrees.to_radians(),
                                floor_sigma: memory_floor_sigma,
                            },
                            prior,
                            &mut rngs[k],
                        )?;
                    }
                    mcs.push(mc);
                }
            }
            let dock = DockSettings {
                learned: None,
                library: !memory_only,
                memory_only,
                library_beta: dock_library_beta,
                components: dock_prior_components,
                sigma_translation: dock_sigma_translation,
                sigma_rotation: dock_sigma_rotation_degrees.to_radians(),
                uniform_weight: dock_uniform_weight,
            };
            let mut master = StdRng::seed_from_u64(seed ^ 0xabcdef);
            // level -> replica holding it
            let mut holder: Vec<usize> = (0..replicas).collect();
            let mut swaps = vec![(0u64, 0u64); replicas - 1];
            let meta = json!({"shape": shape, "seed_config": seed_config, "seed_count": seed_count,
                "free_bodies": free_bodies, "concentration_um": concentration_um, "vessel_radius": radius,
                "ladder": ladder, "width": width, "sweeps": sweeps, "swap_every": swap_every,
                "sample_every": sample_every, "seed": seed, "charge_groups": charge_groups,
                "dock_probability": dock_probability, "dock_library_gap": dock_library_gap,
                "dock_sigma_translation": dock_sigma_translation,
                "dock_sigma_rotation_degrees": dock_sigma_rotation_degrees,
                "dock_uniform_weight": dock_uniform_weight, "memory_lambda_floor": memory_lambda_floor,
                "memory_lambda": memory_lambda, "memory_lambda_prior": memory_lambda_prior,
                "memory_prior_model": memory_prior_model, "memory_death": memory_death,
                "memory_sigma_translation": memory_sigma_translation,
                "memory_sigma_rotation_degrees": memory_sigma_rotation_degrees,
                "memory_floor_sigma": memory_floor_sigma, "memory_refresh_per_sweep": memory_refresh_per_sweep,
                "memory_only": memory_only,
                "like_charge_weight": like_charge_weight, "like_charge_cutoff": like_charge_cutoff,
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
                            "pairs": mcs[r].pair_count(), "solvation": mcs[r].solvation_energy(),
                            "coordinate_wall_center": mcs[r].coordinate_wall_center(),
                            "poses": mcs[r].poses()})
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
                let dock = &dock;
                let cs = &cs;
                std::thread::scope(|sc| -> Result<()> {
                    let handles: Vec<_> = mcs
                        .iter_mut()
                        .zip(rngs.iter_mut())
                        .map(|(mc, rng)| {
                            sc.spawn(move || -> Result<()> {
                                for _ in 0..swap_every * free_bodies {
                                    if gca_probability > 0. && rng.random::<f64>() < gca_probability {
                                        mc.step_gca(rng, gca_fk_beta)?;
                                        continue;
                                    }
                                    if cluster_probability > 0. && rng.random::<f64>() < cluster_probability {
                                        mc.step_cluster(rng, cs, Some(dock))?;
                                        continue;
                                    }
                                    if dock_probability > 0. && rng.random::<f64>() < dock_probability {
                                        mc.step_dock(rng, dock)?;
                                        continue;
                                    }
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
                                for _ in 0..swap_every {
                                    if center_shift_probability > 0. && rng.random::<f64>() < center_shift_probability {
                                        mc.step_center_shift(rng)?;
                                    }
                                }
                                for _ in 0..swap_every * memory_refresh_per_sweep {
                                    mc.refresh_memory(rng);
                                }
                                mc.recompute_memory();
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
                    "well_cached": mc.well_energy(), "well_recomputed": sw,
                    "like_charge_cached": mc.like_energy(), "like_charge_recomputed": mc.recompute_like(),
                    "chemistry_cached": mc.chem_energy(), "chemistry_recomputed": mc.recompute_chem(),
                    "memory_sum_relative_error": mc.memory_sum_error()})
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
            dock_components,
            dock_model,
            dock_probability,
            dock_sigma_translation,
            dock_sigma_rotation_degrees,
            dock_uniform_weight,
            dock_library,
            dock_library_gap,
            dock_library_beta,
            surface_prior_samples,
            surface_prior_beta,
            memory_lambda,
            memory_lambda_floor,
            memory_lambda_prior,
            memory_prior_model,
            memory_death,
            memory_sigma_translation,
            memory_sigma_rotation_degrees,
            memory_floor_sigma,
            memory_refresh_per_sweep,
            memory_only,
            charge_groups,
            like_charge_weight,
            like_charge_cutoff,
            chemistry_model,
            cluster_probability,
            cluster_max_size,
            cluster_fk_beta,
            cluster_dock_probability,
            cluster_reinsertion_probability,
            cluster_translation,
            cluster_rotation_degrees,
            cluster_no_size_scaling,
            gca_probability,
            gca_fk_beta,
            center_shift_probability,
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
            let model = hard_sphere(n, !no_solvation);
            let t0 = Instant::now();
            let mc = MorphMc::new(
                tree,
                wall,
                model,
                SquareWell { epsilon, width },
                poses,
                fixed,
            )?;
            ensure!(
                like_charge_weight == 0. || charge_groups.is_some(),
                "--like-charge-weight needs --charge-groups"
            );
            let groups: Vec<ChargeGroup> = match &charge_groups {
                Some(f) => serde_json::from_slice(&fs::read(f)?)?,
                None => Vec::new(),
            };
            let chem: ChemistryModel = match &chemistry_model {
                Some(f) => serde_json::from_slice(&fs::read(f)?)?,
                None => ChemistryModel::default(),
            };
            let mut mc = mc
                .with_like_charge(LikeCharge {
                    weight: like_charge_weight,
                    cutoff: like_charge_cutoff,
                    groups,
                })?
                .with_chemistry_model(chem)?;
            if dock_library || memory_only || memory_lambda_floor > 0. {
                ensure!(
                    dock_model.is_none() && dock_components.is_none(),
                    "--dock-library / --memory-only excludes --dock-model and --dock-components"
                );
                mc = mc.with_contact_library(dock_library_gap)?;
            }
            let mut rng = StdRng::seed_from_u64(seed);
            let surf_prior = if surface_prior_samples > 0 {
                let mut prng = StdRng::seed_from_u64(seed ^ 0x50fa_ce00);
                mc.generate_surface_normal_prior(&mut prng, surface_prior_samples, surface_prior_beta)
            } else {
                Vec::new()
            };
            if memory_lambda_floor > 0. {
                ensure!(
                    dock_library || memory_only,
                    "memory needs --dock-library or --memory-only"
                );
                let mut prior = match &memory_prior_model {
                    Some(f) => prior_from_model(f, &shape, radius)?,
                    None => Vec::new(),
                };
                prior.extend(surf_prior.iter().copied());
                let eff_lambda_prior = if memory_lambda_prior > 0. {
                    memory_lambda_prior
                } else if !prior.is_empty() {
                    2.0
                } else {
                    0.0
                };
                mc = mc.with_memory(
                    MemorySettings {
                        lambda_floor: memory_lambda_floor,
                        lambda_prior: eff_lambda_prior,
                        lambda: memory_lambda,
                        death: memory_death,
                        sigma_translation: memory_sigma_translation,
                        sigma_rotation: memory_sigma_rotation_degrees.to_radians(),
                        floor_sigma: memory_floor_sigma,
                    },
                    prior,
                    &mut rng,
                )?;
            } else {
                ensure!(
                    !memory_only,
                    "--memory-only requires --memory-lambda-floor > 0"
                );
            }
            let meta = json!({"shape": shape, "seed_config": seed_config, "seed_count": seed_count,
                "free_bodies": free_bodies, "concentration_um": concentration_um, "vessel_radius": radius,
                "epsilon": epsilon, "width": width, "solvation": !no_solvation, "sweeps": sweeps,
                "charge_groups": charge_groups, "like_charge_weight": like_charge_weight,
                "like_charge_cutoff": like_charge_cutoff, "chemistry_model": chemistry_model,
                "dock_library_beta": dock_library_beta,
                "surface_prior_samples": surface_prior_samples,
                "surface_prior_beta": surface_prior_beta,
                "surface_prior_generated_poses": surf_prior.len(),
                "cluster_probability": cluster_probability, "cluster_max_size": cluster_max_size,
                "cluster_fk_beta": cluster_fk_beta, "cluster_dock_probability": cluster_dock_probability,
                "cluster_reinsertion_probability": cluster_reinsertion_probability,
                "cluster_translation": cluster_translation,
                "cluster_rotation_degrees": cluster_rotation_degrees,
                "cluster_size_scaling": !cluster_no_size_scaling,
                "gca_probability": gca_probability, "gca_fk_beta": gca_fk_beta,
                "center_shift_probability": center_shift_probability,
                "memory_lambda": memory_lambda,
                "memory_lambda_floor": memory_lambda_floor, "memory_lambda_prior": memory_lambda_prior,
                "memory_prior_model": memory_prior_model, "memory_death": memory_death,
                "memory_sigma_translation": memory_sigma_translation,
                "memory_sigma_rotation_degrees": memory_sigma_rotation_degrees,
                "memory_floor_sigma": memory_floor_sigma,
                "memory_refresh_per_sweep": memory_refresh_per_sweep,
                "memory_only": memory_only,
                "sample_every": sample_every, "seed": seed,
                "moves": "uniform free body; 0.5 tiny (0.1 A, 1 deg) / 0.45 medium (0.5 A, 4 deg) / 0.05 uniform reinsertion; optional FK rigid cluster + half-turn GCA",
                "energy": "exact hard-sphere White Bear II morphometry (eta 0.383, R 1.4) + atom square well + like-charge + chemistry residual; exact hard cores and wall"});
            fs::write(
                out.join("config.json"),
                serde_json::to_string_pretty(&meta)?,
            )?;
            let learned = match &dock_model {
                Some(f) => {
                    let raw = fs::read(f)?;
                    let sha = tetramer_mc::simulation::hash_bytes(&fs::read(&shape)?);
                    let frozen =
                        tetramer_mc::proposal::FrozenRelativePoseProposal::from_json_str_open(
                            std::str::from_utf8(&raw)?,
                            [2. * radius; 3],
                            0.1,
                            &sha,
                        )?;
                    Some(std::sync::Arc::new(
                        tetramer_mc::docking::DockingProposal::new(
                            frozen,
                            tetramer_mc::docking::DockingMethod::PosteriorInvolution,
                            0.9,
                            [0.; 3],
                        )?,
                    ))
                }
                None => None,
            };
            let mut base_components: Vec<Pose> = match &dock_components {
                Some(f) => serde_json::from_slice(&fs::read(f)?)?,
                None => Vec::new(),
            };
            if memory_lambda_floor == 0. && !surf_prior.is_empty() {
                base_components.extend(surf_prior.iter().map(|&(p, _)| p));
            }
            let dock = if (dock_probability > 0. || (cluster_probability > 0. && cluster_dock_probability > 0.))
                && (learned.is_some() || !base_components.is_empty() || dock_library || memory_only)
            {
                Some(DockSettings {
                    learned,
                    library: dock_library && !memory_only,
                    memory_only,
                    library_beta: dock_library_beta,
                    components: base_components,
                    sigma_translation: dock_sigma_translation,
                    sigma_rotation: dock_sigma_rotation_degrees.to_radians(),
                    uniform_weight: dock_uniform_weight,
                })
            } else {
                None
            };
            let cs = ClusterSettings {
                max_size: cluster_max_size,
                fk_beta: cluster_fk_beta,
                dock_probability: cluster_dock_probability,
                reinsertion_probability: cluster_reinsertion_probability,
                translation: cluster_translation,
                rotation: cluster_rotation_degrees.to_radians(),
                size_scaling: !cluster_no_size_scaling,
            };
            fs::write(
                out.join("dock.json"),
                serde_json::to_string_pretty(
                    &json!({"components": dock_components, "model": dock_model, "library": dock_library && !memory_only,
                    "memory_only": memory_only,
                    "library_gap": dock_library_gap, "probability": dock_probability,
                    "sigma_translation": dock_sigma_translation, "sigma_rotation_degrees": dock_sigma_rotation_degrees,
                    "uniform_weight": dock_uniform_weight}),
                )?,
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
                    writeln!(
                        traj,
                        "{}",
                        json!({"sweep": sweep, "coordinate_wall_center": mc.coordinate_wall_center(), "poses": mc.poses(), "memory": mc.memory_slots()})
                    )?;
                    let s = mc.stats;
                    writeln!(
                        log,
                        "{}",
                        json!({"sweep": sweep, "seconds": t0.elapsed().as_secs_f64(),
                        "solvation": mc.solvation_energy(), "well": mc.well_energy(), "like_charge": mc.like_energy(), "chemistry": mc.chem_energy(), "contacts": mc.contact_count(),
                        "attempts": s.attempts, "accepted": s.accepted, "hard_rejected": s.hard_rejected,
                        "wall_rejected": s.wall_rejected, "reinsertion_attempts": s.reinsertion_attempts,
                        "reinsertion_accepted": s.reinsertion_accepted, "dock_attempts": s.dock_attempts, "dock_accepted": s.dock_accepted,
                        "cluster_attempts": s.cluster_attempts, "cluster_accepted": s.cluster_accepted,
                        "cluster_multibody_attempts": s.cluster_multibody_attempts, "cluster_multibody_accepted": s.cluster_multibody_accepted,
                        "cluster_dock_attempts": s.cluster_dock_attempts, "cluster_dock_accepted": s.cluster_dock_accepted,
                        "cluster_reinsertion_attempts": s.cluster_reinsertion_attempts, "cluster_reinsertion_accepted": s.cluster_reinsertion_accepted,
                        "cluster_pinned_rejected": s.cluster_pinned_rejected, "cluster_cap_rejected": s.cluster_cap_rejected,
                        "gca_attempts": s.gca_attempts, "gca_accepted": s.gca_accepted, "gca_moved_bodies": s.gca_moved_bodies,
                        "center_shift_attempts": s.center_shift_attempts, "center_shift_accepted": s.center_shift_accepted})
                    )?;
                    log.flush()?;
                    traj.flush()?;
                }
                if sweep == sweeps {
                    break;
                }
                for _ in 0..free_bodies {
                    if gca_probability > 0. && rng.random::<f64>() < gca_probability {
                        mc.step_gca(&mut rng, gca_fk_beta)?;
                        continue;
                    }
                    if cluster_probability > 0. && rng.random::<f64>() < cluster_probability {
                        mc.step_cluster(&mut rng, &cs, dock.as_ref())?;
                        continue;
                    }
                    if let Some(d) = &dock {
                        if dock_probability > 0. && rng.random::<f64>() < dock_probability {
                            mc.step_dock(&mut rng, d)?;
                            continue;
                        }
                    }
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
                if center_shift_probability > 0. && rng.random::<f64>() < center_shift_probability {
                    mc.step_center_shift(&mut rng)?;
                }
                for _ in 0..memory_refresh_per_sweep {
                    mc.refresh_memory(&mut rng);
                }
                mc.recompute_memory();
            }
            let (sf, sw) = mc.recompute();
            fs::write(
                out.join("final_check.json"),
                serde_json::to_string_pretty(&json!({
                "solvation_cached": mc.solvation_energy(), "solvation_recomputed": sf,
                "well_cached": mc.well_energy(), "well_recomputed": sw,
                "like_charge_cached": mc.like_energy(), "like_charge_recomputed": mc.recompute_like(),
                "chemistry_cached": mc.chem_energy(), "chemistry_recomputed": mc.recompute_chem(),
                "memory_sum_relative_error": mc.memory_sum_error()}))?,
            )?;
        }
    }
    Ok(())
}
