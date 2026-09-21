use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use tetramer_mc::normalizer::{self, NormalizerOptions, NormalizerWall};

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
    /// Use one fixed neighbor as the proposal frame, retaining all physical neighbors.
    #[arg(long)]
    proposal_anchor_index: Option<usize>,
    #[arg(long, default_value_t = 2)]
    cloud_replicates: usize,
    #[arg(long)]
    activity: Option<f64>,
    /// Integrate an atomic protein wall; the ideal bath permeates it. Capture
    /// must enclose every feasible center (radius >= R_wall + shape bound).
    #[arg(long)]
    wall_radius: Option<f64>,
    /// Wall center in laboratory coordinates (default: origin).
    #[arg(
        long,
        num_args = 3,
        allow_hyphen_values = true,
        requires = "wall_radius"
    )]
    wall_center: Option<Vec<f64>>,
}
fn main() -> Result<()> {
    let c = Cli::parse();
    let options = NormalizerOptions {
        config: c.config,
        model: c.model,
        out: c.out,
        samples: c.samples,
        seed: c.seed,
        covariance_scale: c.covariance_scale,
        uniform_probability: c.uniform_probability,
        proposal_anchor_index: c.proposal_anchor_index,
        cloud_replicates: c.cloud_replicates,
        activity: c.activity,
    };
    let summary = if let Some(radius) = c.wall_radius {
        let center = c.wall_center.map_or([0.; 3], |v| [v[0], v[1], v[2]]);
        normalizer::run_with_wall(options, NormalizerWall { center, radius })?
    } else {
        normalizer::run(options)?
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
