//! Deterministic proposal-density audit: no RNG, particle geometry or MC.
//!
//! The optional map uses one self-pair only to prepare the production chart
//! factors. Its pair-selection law is not the production transport law and is
//! never sampled. Direct and reconstructed-map densities remain separate.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs, io::Write, path::PathBuf};
use tetramer_mc::{
    basin_involution::{BasinPair, FixedBasinInvolution},
    math::{Pose, invert_relative_pose, rotation},
    proposal::{FrozenRelativePoseProposal, RelativePoseBranch},
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    expected_shape_sha256: String,
    /// JSON array of {"label": "...", "pose": {position, orientation}}.
    /// Poses are already relative to a fixed anchor; quaternion order is wxyz.
    #[arg(long)]
    probes: PathBuf,
    #[arg(long)]
    expected_probes: usize,
    #[arg(long)]
    expected_components: usize,
    #[arg(long)]
    expected_branches: usize,
    #[arg(long)]
    out: PathBuf,
    /// Side of the dummy defensive cube. Learned-only scores do not depend on it.
    #[arg(long, default_value_t = 1000.)]
    cube_side: f64,
    #[arg(long, default_value_t = 0.5)]
    uniform_probability: f64,
    #[arg(long, default_value_t = false)]
    transport_factors: bool,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Probe {
    label: String,
    pose: Pose,
}

#[derive(Debug, Serialize)]
struct DensityValue {
    status: &'static str,
    log_density: Option<f64>,
    error: Option<String>,
}

impl DensityValue {
    fn from_result(value: Result<f64>) -> Self {
        match value {
            Ok(v) if v.is_finite() => Self {
                status: "finite",
                log_density: Some(v),
                error: None,
            },
            Ok(v) if v == f64::NEG_INFINITY => Self {
                status: "negative_infinity",
                log_density: None,
                error: None,
            },
            Ok(v) => Self {
                status: "error",
                log_density: None,
                error: Some(format!("Invalid log density: {v}")),
            },
            Err(error) => Self {
                status: "error",
                log_density: None,
                error: Some(format!("{error:#}")),
            },
        }
    }
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn direct_density(model: &FrozenRelativePoseProposal, pose: Pose) -> Result<f64> {
    pose.validate()?;
    model.relative_log_density(pose.position, rotation(pose.orientation))
}

fn density_only_map(model: &FrozenRelativePoseProposal) -> Result<FixedBasinInvolution> {
    // component_parameters reconstructs LL^T exactly as DockingProposal::new.
    // Each duplicate virtual branch would prepare the same base factor. Pair
    // weights and correlation do not enter chart density preparation/scoring.
    FixedBasinInvolution::new(
        model.component_parameters(),
        model.angular_length(),
        0.,
        vec![BasinPair {
            first: 0,
            second: 0,
            weight: 1.,
        }],
    )
}

fn map_density(
    map: &FixedBasinInvolution,
    branches: &[RelativePoseBranch],
    pose: Pose,
) -> Result<f64> {
    pose.validate()?;
    let inverse = invert_relative_pose(pose);
    let mut logs = Vec::with_capacity(branches.len());
    for branch in branches {
        let input = if branch.inverted { inverse } else { pose };
        logs.push(branch.weight.ln() + map.checked_log_density(branch.component_index, input)?);
    }
    let maximum = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if maximum == f64::NEG_INFINITY {
        return Ok(maximum);
    }
    ensure!(maximum.is_finite(), "Invalid maximum chart density");
    Ok(maximum + logs.iter().map(|v| (v - maximum).exp()).sum::<f64>().ln())
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(!args.out.exists(), "Output must be fresh");
    ensure!(
        args.expected_probes > 0 && args.expected_components > 0 && args.expected_branches > 0,
        "Empty allocation"
    );
    let model_bytes = fs::read(&args.model)?;
    let probe_bytes = fs::read(&args.probes)?;
    let model = FrozenRelativePoseProposal::from_json_str_open(
        std::str::from_utf8(&model_bytes)?,
        [args.cube_side; 3],
        args.uniform_probability,
        &args.expected_shape_sha256,
    )?;
    let branches = model.virtual_branches();
    ensure!(
        model.component_count() == args.expected_components
            && branches.len() == args.expected_branches,
        "Changed component/branch inventory"
    );
    let probes: Vec<Probe> = serde_json::from_slice(&probe_bytes)?;
    ensure!(
        probes.len() == args.expected_probes,
        "Changed probe allocation"
    );
    let mut labels = BTreeSet::new();
    ensure!(
        probes
            .iter()
            .all(|p| !p.label.is_empty() && labels.insert(p.label.clone())),
        "Probe labels must be unique and nonempty"
    );
    // Keep a map-preparation failure separate: direct-loader scores still exist.
    let map = args.transport_factors.then(|| density_only_map(&model));
    let mut passed = map.as_ref().is_none_or(|m| m.is_ok());
    let mut rows = Vec::with_capacity(probes.len());
    for probe in probes {
        let direct = DensityValue::from_result(direct_density(&model, probe.pose));
        let transport = match &map {
            None => None,
            Some(Ok(map)) => Some(DensityValue::from_result(map_density(
                map, &branches, probe.pose,
            ))),
            Some(Err(error)) => Some(DensityValue::from_result(Err(anyhow::anyhow!(
                "Map preparation failed: {error:#}"
            )))),
        };
        passed &=
            direct.status != "error" && transport.as_ref().is_none_or(|v| v.status != "error");
        rows.push(json!({"probe":probe,"direct":direct,"transport":transport}));
    }
    ensure!(
        fs::read(&args.model)? == model_bytes && fs::read(&args.probes)? == probe_bytes,
        "Input changed during audit"
    );
    let report = json!({"schema":"proposal-density-probe-v1","complete":true,"passed":passed,
        "model_sha256":sha(&model_bytes),"probes_sha256":sha(&probe_bytes),"shape_sha256":model.shape_sha256(),
        "base_components":model.component_count(),"virtual_branches":branches,
        "reciprocal_components":model.reciprocal_components(),"normalized_component_weights":model.component_weights(),
        "angular_length":model.angular_length(),"cube_side":args.cube_side,"uniform_probability":args.uniform_probability,
        "density_measure":"learned G only with respect to d^3t times normalized Haar; no uniform or physical factor",
        "transport_factor_check":args.transport_factors,"transport_pair_entries":if args.transport_factors {1} else {0},
        "transport_map_preparation_error":map.as_ref().and_then(|m| m.as_ref().err().map(|e| format!("{e:#}"))),
        "probes":rows,"physical_updates":0,"geometry_queries":0,"random_draws":0,
        "scope":"Deterministic density evaluation; single-self-pair map checks chart factors only, never production pair-selection behavior."});
    let mut out = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.out)
        .context("Create fresh audit output")?;
    serde_json::to_writer_pretty(&mut out, &report)?;
    writeln!(out)?;
    out.sync_all()?;
    ensure!(
        passed,
        "One or more density checks failed; complete report retained"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tetramer_mc::{
        docking::{DockingMethod, DockingProposal},
        math::{cayley, quaternion},
    };
    const SHA: &str = "0000000000000000000000000000000000000000000000000000000000000000";

    fn raw() -> serde_json::Value {
        let cov: [[f64; 6]; 6] = std::array::from_fn(|i| {
            std::array::from_fn(|j| if i == j { 0.15 + (i as f64) * 0.03 } else { 0. })
        });
        json!({"schema":"reciprocal-pose-mixture-v1","reciprocal_components":[true,false],
            "base_model":{"coordinate_convention":"anchor-body-relative","shape_sha256":SHA,
            "angular_length":1.7,"weights":[0.4,0.6],"means":vec![[0.1,-0.2,0.3,0.04,-0.01,0.02],[0.;6]],
            "anchors":[{"position":[2.,-0.3,0.7],"rotation":cayley([0.1,-0.2,0.05])},
                       {"position":[-1.,0.6,2.],"rotation":cayley([-0.3,0.15,0.04])}],"covariances":[cov,cov]}})
    }
    fn load(v: &serde_json::Value) -> FrozenRelativePoseProposal {
        FrozenRelativePoseProposal::from_json_str_open(&v.to_string(), [100.; 3], 0.5, SHA).unwrap()
    }
    fn probes() -> [Pose; 3] {
        [[1., 0.5, -0.4], [-1., 0.2, 1.3], [0.7, -2., 0.1]].map(|position| Pose {
            position,
            orientation: quaternion(cayley([0.07, -0.13, 0.19])),
        })
    }
    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 3e-12, "{a} != {b}");
    }

    #[test]
    fn mixed_flags_direct_density_matches_explicit_virtual_mixture() {
        let value = raw();
        let model = load(&value);
        let branches = model.virtual_branches();
        assert_eq!(branches.len(), 3);
        assert_eq!(model.reciprocal_components(), vec![true, false]);
        let single = |k: usize| {
            let mut v = value["base_model"].clone();
            for field in ["anchors", "means", "covariances"] {
                v[field] = json!([v[field][k].clone()]);
            }
            v["weights"] = json!([1.]);
            load(&v)
        };
        let a = single(0);
        let b = single(1);
        for p in probes() {
            let terms = [
                0.2_f64.ln() + direct_density(&a, p).unwrap(),
                0.2_f64.ln() + direct_density(&a, invert_relative_pose(p)).unwrap(),
                0.6_f64.ln() + direct_density(&b, p).unwrap(),
            ];
            let max = terms.into_iter().fold(f64::NEG_INFINITY, f64::max);
            close(
                direct_density(&model, p).unwrap(),
                max + terms.iter().map(|v| (v - max).exp()).sum::<f64>().ln(),
            );
        }
    }

    #[test]
    fn linear_map_probe_matches_full_production_docking_density() {
        let model = load(&raw());
        let map = density_only_map(&model).unwrap();
        let branches = model.virtual_branches();
        let full =
            DockingProposal::new(model, DockingMethod::PosteriorInvolution, 0.9, [0.; 3]).unwrap();
        let identity = Pose {
            position: [0.; 3],
            orientation: [1., 0., 0., 0.],
        };
        assert_eq!(map.chart_count(), 2);
        for p in probes() {
            close(
                map_density(&map, &branches, p).unwrap(),
                full.members_log_density(&[p], &[identity]).unwrap(),
            );
        }
    }

    #[test]
    fn rejects_invalid_pose_and_preserves_negative_infinity_status() {
        let m = load(&raw());
        let bad = Pose {
            position: [0.; 3],
            orientation: [2., 0., 0., 0.],
        };
        assert!(direct_density(&m, bad).is_err());
        assert!(map_density(&density_only_map(&m).unwrap(), &m.virtual_branches(), bad).is_err());
        let value = DensityValue::from_result(Ok(f64::NEG_INFINITY));
        assert_eq!(value.status, "negative_infinity");
        assert_eq!(value.log_density, None);
        assert_eq!(DensityValue::from_result(Ok(f64::NAN)).status, "error");
    }
}
