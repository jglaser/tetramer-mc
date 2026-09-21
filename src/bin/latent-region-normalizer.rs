use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use tetramer_mc::latent_region::{self, LatentRegionOptions};

#[derive(Parser)]
#[command(about = "Direct physical mass of a frozen latent ellipsoid; NOT a global normalizer")]
struct Cli {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    region: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 16384)]
    samples: u64,
    #[arg(long)]
    seed: u64,
    #[arg(long, default_value_t = 2)]
    cloud_replicates: usize,
    #[arg(long, default_value_t = 64.)]
    lambda_ratio: f64,
    /// Frozen defensive proposal in the unchanged whitened region chart.
    #[arg(long)]
    importance_guide: Option<PathBuf>,
}
fn main() -> Result<()> {
    let c = Cli::parse();
    let options = LatentRegionOptions {
        config: c.config,
        region: c.region,
        out: c.out,
        samples: c.samples,
        seed: c.seed,
        cloud_replicates: c.cloud_replicates,
        lambda_ratio: c.lambda_ratio,
    };
    let result = if let Some(guide) = c.importance_guide {
        latent_region::run_with_importance(options, &guide)?
    } else {
        latent_region::run(options)?
    };
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
