use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use tetramer_mc::latent_region::contact_line::{AuditOptions, run_audit};

#[derive(Parser)]
#[command(
    about = "Exact contact-distance proposal diagnostic; no physical weights or Poisson sampling"
)]
struct Cli {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    region: PathBuf,
    #[arg(long)]
    importance_guide: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 64)]
    samples: u64,
    #[arg(long)]
    seed: u64,
    #[arg(long)]
    probes: Option<PathBuf>,
}
fn main() -> Result<()> {
    let c = Cli::parse();
    println!(
        "{}",
        serde_json::to_string_pretty(&run_audit(AuditOptions {
            config: c.config,
            region: c.region,
            importance_guide: c.importance_guide,
            out: c.out,
            samples: c.samples,
            seed: c.seed,
            probes: c.probes,
        })?)?
    );
    Ok(())
}
