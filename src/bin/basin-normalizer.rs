use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use tetramer_mc::normalizer::{self, NormalizerOptions};

#[derive(Parser)]
#[command(
    about = "Independent positive-Poisson importance estimates of physical contact-region masses"
)]
struct Cli {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 1024)]
    samples: u64,
    #[arg(long, default_value_t = 193840217)]
    seed: u64,
    #[arg(long, default_value_t = 1.)]
    covariance_scale: f64,
    #[arg(long)]
    uniform_probability: Option<f64>,
    #[arg(long, default_value_t = 2)]
    cloud_replicates: usize,
    #[arg(long)]
    activity: Option<f64>,
}
fn main() -> Result<()> {
    let c = Cli::parse();
    let summary = normalizer::run(NormalizerOptions {
        config: c.config,
        model: c.model,
        out: c.out,
        samples: c.samples,
        seed: c.seed,
        covariance_scale: c.covariance_scale,
        uniform_probability: c.uniform_probability,
        cloud_replicates: c.cloud_replicates,
        activity: c.activity,
    })?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
