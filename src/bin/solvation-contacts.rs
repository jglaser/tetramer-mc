//! Atom-type morphometric association free energies of pair contacts: for each pose of
//! body B relative to body A, beta dF = F(A u B) - F(A) - F(B) with the exact union
//! measures and the Harano octanol -> water coefficients, plus the per-term split.
use anyhow::Result;
use clap::Parser;
use serde_json::{Value, json};
use std::{fs, path::PathBuf};
use tetramer_mc::{
    geometry::Shape,
    math::Pose,
    solvation::{SolvationModel, SolvationState},
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    shape: PathBuf,
    /// JSON with "types" (index per template atom); repeated if the shape is a multiple.
    #[arg(long)]
    types: PathBuf,
    /// JSON list of {label, pose} with B relative to A.
    #[arg(long)]
    poses: PathBuf,
    #[arg(long, default_value_t = 298.15)]
    temperature: f64,
}

fn main() -> Result<()> {
    let a = Args::parse();
    let shape: Shape = serde_json::from_slice(&fs::read(&a.shape)?)?;
    let tv: Value = serde_json::from_slice(&fs::read(&a.types)?)?;
    let base: Vec<usize> = tv["types"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_u64().unwrap() as usize)
        .collect();
    let n = shape.atoms.len();
    anyhow::ensure!(
        n % base.len() == 0,
        "shape atoms are not a multiple of the type list"
    );
    let types: Vec<usize> = base.iter().copied().cycle().take(n).collect();
    let centers: Vec<[f64; 3]> = shape.atoms.iter().map(|x| x.center).collect();
    let radii: Vec<f64> = shape.atoms.iter().map(|x| x.radius).collect();
    let identity = Pose {
        position: [0.; 3],
        orientation: [1., 0., 0., 0.],
    };
    let far = Pose {
        position: [1.0e4, 0., 0.],
        orientation: [1., 0., 0., 0.],
    };
    let poses: Vec<Value> = serde_json::from_slice(&fs::read(&a.poses)?)?;
    let mut rows = Vec::new();
    for p in poses {
        let pose: Pose = serde_json::from_value(p["pose"].clone())?;
        let mut terms = Vec::new();
        // Per-term split: one coefficient column at a time.
        for col in 0..3 {
            let mut model = SolvationModel::harano_octanol_water(types.clone(), a.temperature)?;
            for c in &mut model.coefficients {
                for (k, x) in c.iter_mut().enumerate() {
                    if k != col {
                        *x = 0.;
                    }
                }
            }
            let s = SolvationState::new(model, &centers, &radii, &[identity, far])?;
            terms.push(s.propose(&[(1, pose)])?.delta);
        }
        let total: f64 = terms.iter().sum();
        rows.push(json!({"label": p["label"], "dF": total, "area": terms[0], "mean": terms[1], "gauss": terms[2]}));
    }
    println!("{}", serde_json::to_string_pretty(&rows)?);
    Ok(())
}
