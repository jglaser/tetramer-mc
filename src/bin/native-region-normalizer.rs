use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use tetramer_mc::native_region::{self, NativeRegionOptions};

#[derive(Parser)]
#[command(
    about = "Independent native-region normalizer with a complete geometric cover; no learned proposal"
)]
struct Cli {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 262144)]
    samples: u64,
    #[arg(long, default_value_t = 98581010)]
    seed: u64,
    #[arg(long, default_value_t = 2)]
    cloud_replicates: usize,
    #[arg(long)]
    activity: Option<f64>,
    #[arg(long)]
    lambda_ratio: Option<f64>,
    /// Proposal scales only; the original native metric always defines the target.
    #[arg(long, value_delimiter = ',', default_value = "1")]
    cover_scales: Vec<f64>,
    /// Positive mixture weights, normalized internally; equal when omitted.
    #[arg(long, value_delimiter = ',')]
    cover_weights: Vec<f64>,
}
fn main() -> Result<()> {
    let c = Cli::parse();
    println!(
        "{}",
        serde_json::to_string_pretty(&native_region::run(NativeRegionOptions {
            config: c.config,
            out: c.out,
            samples: c.samples,
            seed: c.seed,
            cloud_replicates: c.cloud_replicates,
            activity: c.activity,
            lambda_ratio: c.lambda_ratio,
            cover_scales: c.cover_scales,
            cover_weights: c.cover_weights,
        })?)?
    );
    Ok(())
}
