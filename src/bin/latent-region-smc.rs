use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use tetramer_mc::latent_region::smc::{self, SmcBridge, SmcOptions};

#[derive(Parser)]
#[command(
    about = "Fixed-budget random-potential SMC on the exact original R4; not a full-vessel normalizer"
)]
struct Cli {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    region: PathBuf,
    #[arg(long)]
    initial_reference_region: Option<PathBuf>,
    #[arg(long)]
    initial_current_probability: Option<f64>,
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    seed: u64,
    #[arg(long, value_enum, default_value_t=SmcBridge::PhysicalActivity)]
    bridge: SmcBridge,
    #[arg(long, default_value_t = 262144)]
    initial_draws: usize,
    #[arg(long, default_value_t = 2048)]
    population: usize,
    #[arg(long, default_value_t = 128)]
    stages: usize,
    #[arg(long, default_value_t = 4)]
    sweeps_per_stage: usize,
    #[arg(long, default_value_t = 2)]
    cloud_replicates: usize,
    #[arg(long, default_value_t = 128.)]
    lambda_ratio: f64,
}
fn main() -> Result<()> {
    let cli = Cli::parse();
    anyhow::ensure!(
        cli.stages > 0,
        "At least one fixed annealing stage required"
    );
    let initial_current_probability =
        cli.initial_current_probability
            .unwrap_or(if cli.initial_reference_region.is_some() {
                0.5
            } else {
                1.
            });
    let result = smc::run(SmcOptions {
        config: cli.config,
        region: cli.region,
        initial_reference_region: cli.initial_reference_region,
        initial_current_probability,
        out: cli.out,
        seed: cli.seed,
        bridge: cli.bridge,
        initial_draws: cli.initial_draws,
        population: cli.population,
        schedule: (0..=cli.stages)
            .map(|i| i as f64 / cli.stages as f64)
            .collect(),
        sweeps_per_stage: cli.sweeps_per_stage,
        cloud_replicates: cli.cloud_replicates,
        lambda_ratio: cli.lambda_ratio,
    })?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
