//! Isolated diagnostic: call the archived SMC overlap-count implementation unchanged.
mod geometry;
mod single_body_depletion;
mod mirror {
    pub type V = hoomd_vector::Cartesian<3>;
    pub type P = hoomd_microstate::property::OrientedPoint<V, hoomd_vector::Versor>;
}
use anyhow::{Result, ensure};
use hoomd_vector::Quaternion;
use rand::{SeedableRng, rngs::StdRng};
use serde::Deserialize;
use serde_json::json;
use std::{fs, path::PathBuf, time::Instant};

#[derive(Clone, Copy, Deserialize)]
struct Pose { position: [f64; 3], orientation: [f64; 4] }
#[derive(Deserialize)]
struct Case { id: String, pose: Pose, fixed: Vec<Pose>, box_lengths: [f64; 3], seed: u64 }
#[derive(Deserialize)]
struct Input { shape: PathBuf, rd: f64, activity: f64, clouds: usize, intensity: f64, cases: Vec<Case> }
fn pose(p: Pose) -> Result<mirror::P> {
    Ok(mirror::P { position: p.position.into(), orientation: Quaternion::from(p.orientation).to_versor()? })
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(args.len()==3,"input.json output.json");
    let input: Input = serde_json::from_slice(&fs::read(&args[1])?)?;
    ensure!(input.clouds>1 && input.intensity>0.,"need positive exposure and multiple clouds");
    let shape: geometry::Shape = serde_json::from_slice(&fs::read(&input.shape)?)?;
    let mut results = Vec::new();
    for case in input.cases {
        let start = Instant::now();
        let fixed = case.fixed.iter().copied().map(pose).collect::<Result<Vec<_>>>()?;
        let bath = single_body_depletion::SingleBodyDepletion::new(&shape.atoms,&fixed,case.box_lengths,input.rd,input.activity)?;
        let moving = pose(case.pose)?;
        let mut counts = Vec::new();
        let mut seeds = Vec::new();
        let mut stats = single_body_depletion::SingleDepletionStats::default();
        for replicate in 0..input.clouds {
            let seed = case.seed + 1009 * replicate as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            let sample = bath.sample_overlap_count(&moving,input.intensity,&mut rng)?;
            counts.push(sample.count);
            seeds.push(seed);
            stats.add_assign(&sample.stats);
        }
        let total: u64 = counts.iter().sum();
        let mean = total as f64/input.clouds as f64;
        let variance = counts.iter().map(|&k| (k as f64-mean).powi(2)).sum::<f64>()/(input.clouds-1) as f64;
        let exposure = input.clouds as f64 * input.intensity;
        let result=json!({"id":case.id,"clouds":input.clouds,"intensity":input.intensity,
            "total_intensity_exposure":exposure,"counts":counts,"seeds":seeds,"total_count":total,
            "count_mean":mean,"count_sample_variance":variance,"variance_to_mean":variance/mean,
            "C_A3":total as f64/exposure,"C_poisson_standard_error_A3":(total as f64).sqrt()/exposure,
            "C_empirical_standard_error_A3":(variance/input.clouds as f64).sqrt()/input.intensity,
            "statistics":stats,"elapsed_seconds":start.elapsed().as_secs_f64()});
        println!("{result}");
        results.push(result);
    }
    fs::write(&args[2],serde_json::to_vec_pretty(&json!({"results":results}))?)?;
    Ok(())
}
