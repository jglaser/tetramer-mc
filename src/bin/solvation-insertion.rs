//! Exact many-body insertion free energy of one body into an environment of bodies:
//! beta dF_ins = F(env u {0}) - F(env) - F({0}), with the morphometric solvation model
//! (hard-sphere White Bear II or Harano atom types) on the exact union measures. Also
//! the sum of the pair terms of the same neighbours (additivity check) and the
//! inserted overlap volume -dV (ideal depletion gives -z dV).
//! Input: JSON list of {label, poses: [pose of each neighbour relative to body 0]}.
use anyhow::{Result, bail, ensure};
use clap::Parser;
use serde_json::{Value, json};
use std::{fs, path::PathBuf};
use tetramer_mc::{
    geometry::Shape,
    math::Pose,
    solvation::{GaussConvention, SolvationModel, SolvationState},
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    shape: PathBuf,
    #[arg(long)]
    types: Option<PathBuf>,
    #[arg(long)]
    environments: PathBuf,
    #[arg(long, default_value = "hard-sphere")]
    model: String,
    #[arg(long, default_value = "patch")]
    gauss: String,
    #[arg(long, default_value_t = 298.15)]
    temperature: f64,
    /// Skip the pairwise sum (one extra two-body evaluation per neighbour).
    #[arg(long)]
    no_pairs: bool,
}

fn model(a: &Args, n: usize) -> Result<SolvationModel> {
    let mut m = match a.model.as_str() {
        "hard-sphere" => SolvationModel {
            probe: 1.4,
            pressure: 0.2090,
            coefficients: vec![[-0.1564, 0.1362, -0.0369]],
            atom_types: vec![0; n],
            gauss_convention: GaussConvention::PatchGaussBonnet,
        },
        "harano" => {
            let tv: Value = serde_json::from_slice(&fs::read(a.types.as_ref().unwrap())?)?;
            let base: Vec<usize> = tv["types"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| t.as_u64().unwrap() as usize)
                .collect();
            ensure!(
                n % base.len() == 0,
                "shape atoms are not a multiple of the type list"
            );
            SolvationModel::harano_octanol_water(
                base.iter().copied().cycle().take(n).collect(),
                a.temperature,
            )?
        }
        "volume" => SolvationModel {
            probe: 1.4,
            pressure: 1.,
            coefficients: vec![[0., 0., 0.]],
            atom_types: vec![0; n],
            gauss_convention: GaussConvention::PatchGaussBonnet,
        },
        other => bail!("unknown model {other}"),
    };
    m.gauss_convention = match a.gauss.as_str() {
        "patch" => GaussConvention::PatchGaussBonnet,
        "alphamol" => GaussConvention::AlphaMol,
        other => bail!("unknown Gaussian convention {other}"),
    };
    Ok(m)
}

fn main() -> Result<()> {
    let a = Args::parse();
    let shape: Shape = serde_json::from_slice(&fs::read(&a.shape)?)?;
    let centers: Vec<[f64; 3]> = shape.atoms.iter().map(|x| x.center).collect();
    let radii: Vec<f64> = shape.atoms.iter().map(|x| x.radius).collect();
    let n = centers.len();
    let identity = Pose {
        position: [0.; 3],
        orientation: [1., 0., 0., 0.],
    };
    let far = Pose {
        position: [1.0e5, 0., 0.],
        orientation: [1., 0., 0., 0.],
    };
    let envs: Vec<Value> = serde_json::from_slice(&fs::read(&a.environments)?)?;
    let mut rows = Vec::new();
    for e in envs {
        let nb: Vec<Pose> = serde_json::from_value(e["poses"].clone())?;
        let mut poses = vec![far];
        poses.extend(&nb);
        let s = SolvationState::new(model(&a, n)?, &centers, &radii, &poses)?;
        let ins = s.propose(&[(0, identity)])?.delta;
        let vol = SolvationState::new(model_volume(n), &centers, &radii, &poses)?
            .propose(&[(0, identity)])?
            .delta;
        let pairs = if a.no_pairs {
            Value::Null
        } else {
            let mut sum = 0.;
            for &p in &nb {
                let s2 = SolvationState::new(model(&a, n)?, &centers, &radii, &[far, p])?;
                sum += s2.propose(&[(0, identity)])?.delta;
            }
            json!(sum)
        };
        rows.push(
            json!({"label": e["label"], "neighbours": nb.len(), "dF_insertion": ins,
            "dF_pair_sum": pairs, "dV_insertion": vol}),
        );
        eprintln!("{} {:.2}", e["label"], ins);
    }
    println!("{}", serde_json::to_string_pretty(&rows)?);
    Ok(())
}

fn model_volume(n: usize) -> SolvationModel {
    SolvationModel {
        probe: 1.4,
        pressure: 1.,
        coefficients: vec![[0., 0., 0.]],
        atom_types: vec![0; n],
        gauss_convention: GaussConvention::PatchGaussBonnet,
    }
}
