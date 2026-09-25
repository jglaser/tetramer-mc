use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tetramer_mc::initialization::FreeTetramerStart;
use tetramer_mc::simulation::{self, Method, RunOptions, RunOverrides};

#[derive(Parser)]
#[command(
    version,
    about = "Rigid sphere-union MC with reversible learned poses and exact ideal depletion"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// Run all-mobile MC with the configured periodic or spherical boundary.
    /// Resume into a fresh output directory.
    Run {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long, value_enum, default_value = "learned")]
        method: Method,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 400)]
        sweeps: u64,
        #[arg(long, default_value_t = 10)]
        sample_every: u64,
        #[arg(long)]
        resume: Option<PathBuf>,
        /// Preserve labeled seed bodies and generate N free tetramers; size the vessel from the total count.
        #[arg(
            long,
            requires = "tetramer_concentration_um",
            conflicts_with = "resume"
        )]
        free_tetramers: Option<usize>,
        /// Explicitly discard labeled seed bodies when generating the free preparation.
        #[arg(long, requires = "free_tetramers", conflicts_with = "resume")]
        discard_seed: bool,
        /// Total seed-plus-free tetramer concentration in micromolar per full vessel volume.
        #[arg(
            long,
            visible_alias = "concentration-um",
            requires = "free_tetramers",
            conflicts_with = "resume"
        )]
        tetramer_concentration_um: Option<f64>,
        /// Master seed for a generated free start and its MC run (defaults to config seed).
        #[arg(long, requires = "free_tetramers", conflicts_with = "resume")]
        seed: Option<u64>,
        /// Override depletant radius in angstroms; keep the supplied move map.
        #[arg(long, conflicts_with = "resume")]
        depletant_radius: Option<f64>,
        /// Override ideal-depletant activity in A^-3 (zero disables depletion).
        #[arg(long, conflicts_with = "resume")]
        depletant_activity: Option<f64>,
        #[arg(long)]
        no_gsd: bool,
        #[arg(long)]
        no_moves: bool,
    },
}
fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Run {
            config,
            model,
            method,
            out,
            sweeps,
            sample_every,
            resume,
            free_tetramers,
            discard_seed,
            tetramer_concentration_um,
            seed,
            depletant_radius,
            depletant_activity,
            no_gsd,
            no_moves,
        } => {
            let initialization = free_tetramers.map(|count| FreeTetramerStart {
                count,
                concentration_um: tetramer_concentration_um.expect("required by clap"),
                seed,
            });
            let summary = simulation::run_with_overrides(
                RunOptions {
                    config,
                    model,
                    method,
                    out,
                    sweeps,
                    sample_every,
                    resume,
                    write_gsd: !no_gsd,
                    record_moves: !no_moves,
                },
                RunOverrides {
                    initialization,
                    discard_seed,
                    depletant_radius,
                    depletant_activity,
                },
            )?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
    }
    Ok(())
}
