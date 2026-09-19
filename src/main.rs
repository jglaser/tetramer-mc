use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tetramer_mc::simulation::{self, Method, RunOptions};

#[derive(Parser)]
#[command(
    version,
    about = "Rigid sphere-union MC with frozen learned poses and exact ideal depletion"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// Run all-mobile periodic MC. Resume into a fresh output directory.
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
            no_gsd,
            no_moves,
        } => {
            let summary = simulation::run(RunOptions {
                config,
                model,
                method,
                out,
                sweeps,
                sample_every,
                resume,
                write_gsd: !no_gsd,
                record_moves: !no_moves,
            })?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
    }
    Ok(())
}
