use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use tetramer_mc::{
    context_docking::{self, ContextDockingOptions},
    docking::DockingMethod,
};

#[derive(Parser)]
#[command(about = "One-mobile fixed-context sampling in the full atomic spherical wall")]
struct Cli {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    prior: Option<PathBuf>,
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    cycles: u64,
    #[arg(long)]
    sample_every: u64,
    #[arg(long, value_enum)]
    method: DockingMethod,
    #[arg(long)]
    correlation: f64,
    #[arg(long)]
    resume: Option<PathBuf>,
    #[arg(long)]
    certify: Option<PathBuf>,
}
fn main() -> Result<()> {
    let c = Cli::parse();
    let value = context_docking::run(ContextDockingOptions {
        config: c.config,
        model: c.model,
        prior: c.prior,
        out: c.out,
        cycles: c.cycles,
        sample_every: c.sample_every,
        method: c.method,
        correlation: c.correlation,
        resume: c.resume,
        certify: c.certify,
    })?;
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}
