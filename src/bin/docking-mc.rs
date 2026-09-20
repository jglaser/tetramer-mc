use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use tetramer_mc::docking::{self, DockingMethod, DockingOptions};

#[derive(Parser)]
#[command(about = "Conditional rigid-body docking with a fixed atlas and exact implicit depletion")]
struct Cli {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 5000)]
    cycles: u64,
    #[arg(long, default_value_t = 1)]
    sample_every: u64,
    #[arg(long, value_enum, default_value = "mixture")]
    method: DockingMethod,
    #[arg(long, default_value_t = 0.)]
    correlation: f64,
    #[arg(long)]
    resume: Option<PathBuf>,
}
fn main() -> Result<()> {
    let c = Cli::parse();
    let summary = docking::run(DockingOptions {
        config: c.config,
        model: c.model,
        out: c.out,
        cycles: c.cycles,
        sample_every: c.sample_every,
        method: c.method,
        correlation: c.correlation,
        resume: c.resume,
    })?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
