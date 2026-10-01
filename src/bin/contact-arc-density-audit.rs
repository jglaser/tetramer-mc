use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use tetramer_mc::latent_region::{ContactArcAuditOptions, run_contact_arc_audit};

#[derive(Parser)]
#[command(about = "Score frozen hard-free-arc density candidates; no pose or Poisson draws")]
struct Cli {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    region: PathBuf,
    #[arg(long = "importance-guide", required = true)]
    importance_guides: Vec<PathBuf>,
    #[arg(long)]
    probes: PathBuf,
    #[arg(long)]
    minimum_arc_mass: f64,
    #[arg(long)]
    out: PathBuf,
}
fn main() -> Result<()> {
    let c = Cli::parse();
    println!(
        "{}",
        serde_json::to_string_pretty(&run_contact_arc_audit(ContactArcAuditOptions {
            config: c.config,
            region: c.region,
            importance_guides: c.importance_guides,
            probes: c.probes,
            minimum_arc_mass: c.minimum_arc_mass,
            out: c.out
        })?)?
    );
    Ok(())
}
