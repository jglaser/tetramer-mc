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
    solvation::{GaussConvention, SolvationModel, SolvationState},
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
    /// Local Gaussian-curvature share per atom: patch (Gauss-Bonnet), face-edge or face.
    #[arg(long, default_value = "patch")]
    gauss: String,
    /// Coefficients: harano (octanol -> water, typed) or hard-sphere (White Bear II).
    #[arg(long, default_value = "harano")]
    model: String,
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
        // Per-term split: one coefficient column at a time (column 3 = volume term).
        for col in 0..4 {
            let mut model = match a.model.as_str() {
                "harano" => SolvationModel::harano_octanol_water(types.clone(), a.temperature)?,
                // White Bear II hard spheres, eta 0.383, R 1.4 A (kT, A), one type.
                "hard-sphere" => SolvationModel {
                    probe: 1.4,
                    pressure: 0.2090,
                    coefficients: vec![[-0.1564, 0.1362, -0.0369]],
                    atom_types: vec![0; centers.len()],
                    gauss_convention: GaussConvention::PatchGaussBonnet,
                },
                other => anyhow::bail!("unknown model {other}"),
            };
            model.gauss_convention = match a.gauss.as_str() {
                "patch" => GaussConvention::PatchGaussBonnet,
                "alphamol" => GaussConvention::AlphaMol,
                "face-edge" => GaussConvention::FaceEdge,
                "face" => GaussConvention::Face,
                other => anyhow::bail!("unknown Gaussian convention {other}"),
            };
            if col == 3 && model.pressure == 0. {
                terms.push(0.);
                continue;
            }
            for c in &mut model.coefficients {
                for (k, x) in c.iter_mut().enumerate() {
                    if k != col {
                        *x = 0.;
                    }
                }
            }
            if col != 3 {
                model.pressure = 0.;
            }
            let s = SolvationState::new(model, &centers, &radii, &[identity, far])?;
            terms.push(s.propose(&[(1, pose)])?.delta);
        }
        let total: f64 = terms.iter().sum();
        rows.push(
            json!({"label": p["label"], "dF": total, "area": terms[0], "mean": terms[1],
            "gauss": terms[2], "volume": terms[3]}),
        );
    }
    println!("{}", serde_json::to_string_pretty(&rows)?);
    Ok(())
}
