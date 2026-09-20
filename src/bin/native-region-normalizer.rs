use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use tetramer_mc::native_region::{self, NativeRegionOptions};

#[derive(Parser)]
#[command(
    about = "Independent native-region normalizer with a complete geometric cover and optional frozen Gaussian guide"
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
    /// Optional frozen relative-pose Gaussian atlas, mixed with geometric covers.
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long, default_value_t = 0.75)]
    model_weight: f64,
    #[arg(long, default_value_t = 0.05)]
    model_uniform_probability: f64,
    /// Proposal anchor only; every physical neighbor is retained in the target.
    #[arg(long, default_value_t = 0)]
    model_anchor_index: usize,
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
            model: c.model,
            model_weight: c.model_weight,
            model_uniform_probability: c.model_uniform_probability,
            model_anchor_index: c.model_anchor_index,
        })?)?
    );
    Ok(())
}
