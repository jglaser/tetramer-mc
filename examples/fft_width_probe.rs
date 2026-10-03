//! Frozen FFT covariance-width passive comparison; six arms share each root cloud.
//! No physical bath or state updates; retain every raw cloud and attempted edge.
use anyhow::{ensure, Context, Result};
use clap::Parser;
use rand::{rngs::StdRng, RngExt, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    path::PathBuf,
};
use tetramer_mc::{
    auxiliary_overlap_threshold::AuxiliaryOverlapThreshold,
    capped_dimer::FixedDimerContext,
    defensive_dimer_proposal::DefensiveDimerProposal,
    dimer_tree_proposal::tree_coordinates,
    docking::{DockingMethod, DockingProposal},
    factorized_dimer::{FactorizedDimerCaps, FactorizedDimerOrder, FactorizedDimerProposal},
    geometry::{Placed, Shape, SphereTree},
    math::Pose,
    proposal::FrozenRelativePoseProposal,
    simulation::{cpu_seconds, hash_bytes, hash_file, save},
    spherical::{validate_state, Container},
};
const BUNDLE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/source-bundle.json"));
const SOURCE: &[u8] = include_bytes!("fft_width_probe.rs");
#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    binding: PathBuf,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BoundFile {
    path: PathBuf,
    sha256: String,
}
impl BoundFile {
    fn read(&self) -> Result<Vec<u8>> {
        let bytes = fs::read(&self.path)?;
        ensure!(
            hash_bytes(&bytes) == self.sha256,
            "changed input {}",
            self.path.display()
        );
        Ok(bytes)
    }
}
#[derive(Deserialize)]
struct Atlas {
    name: String,
    model: BoundFile,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ScaledAtlas {
    name: String,
    tau: f64,
    model: BoundFile,
}
#[derive(Deserialize, Serialize)]
struct Case {
    name: String,
    root: usize,
    child: usize,
    anchor: usize,
}
/// The previous immutable screen configuration supplies exactly the same inputs.
#[derive(Deserialize)]
struct Reference {
    source_config: BoundFile,
    source_frame: BoundFile,
    source_freeze_manifest: BoundFile,
    shape: BoundFile,
    panel: BoundFile,
    atlases: Vec<Atlas>,
    cases: Vec<Case>,
    depletant_radius: f64,
    activity: f64,
    wall_radius: f64,
    uniform_half_width: f64,
    uniform_probability: f64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    schema: String,
    scientific_allocation: BoundFile,
    protocol: BoundFile,
    reference_config: BoundFile,
    master_seed: u64,
    scaled_atlases: Vec<ScaledAtlas>,
    cloud_raw_count: usize,
    attempts_per_context: usize,
    root_cap: usize,
    internal_cap: usize,
    factorized_joint_cap: usize,
    factorized_order: String,
    density_law: String,
    compiled_source_sha256: BTreeMap<String, String>,
    output: PathBuf,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Binding {
    schema: String,
    config_sha256: String,
    protocol_sha256: String,
    example_source_sha256: String,
    compiled_source_bundle_sha256: String,
    executable_sha256: String,
}
fn seed(master: u64, case: usize, attempt: usize, role: &str) -> u64 {
    let h = hash_bytes(format!("fft-width-probe-v1/{master}/{case}/{attempt}/{role}").as_bytes());
    u64::from_str_radix(&h[..16], 16).unwrap()
}
fn method_order(master: u64, case: usize, attempt: usize) -> Vec<(usize, &'static str, u64)> {
    let mut arms: Vec<_> = (0..3)
        .flat_map(|scale| {
            ["unguided", "m4"].map(|method| {
                (
                    scale,
                    method,
                    seed(
                        master,
                        case,
                        attempt,
                        &format!("execution_order/{scale}/{method}"),
                    ),
                )
            })
        })
        .collect();
    arms.sort_by_key(|&(scale, method, key)| (key, scale, method));
    arms
}
/// Validate the complete artifact transformation; only base covariances may change.
fn validate_scaled(original: &Value, scaled: &Value, tau: f64) -> Result<()> {
    ensure!(
        [0.125, 0.25, 0.5].contains(&tau),
        "unknown covariance scale"
    );
    ensure!(
        original["schema"] == "reciprocal-pose-mixture-v1"
            && original["base_model"]["schema"] == "weighted-pose-mixture-v1",
        "unexpected atlas schema"
    );
    let before = original["base_model"]["covariances"]
        .as_array()
        .context("source covariances")?;
    let after = scaled["base_model"]["covariances"]
        .as_array()
        .context("scaled covariances")?;
    ensure!(
        !before.is_empty() && before.len() == after.len(),
        "covariance count differs"
    );
    for (a, b) in before.iter().zip(after) {
        ensure!(
            a.as_array().is_some_and(|v| v.len() == 6)
                && b.as_array().is_some_and(|v| v.len() == 6),
            "covariance dimensions"
        );
        for i in 0..6 {
            ensure!(
                a[i].as_array().is_some_and(|v| v.len() == 6)
                    && b[i].as_array().is_some_and(|v| v.len() == 6),
                "covariance row dimensions"
            );
            for j in 0..6 {
                let x = a[i][j].as_f64().context("covariance number")?;
                let y = b[i][j].as_f64().context("scaled covariance number")?;
                ensure!(
                    x.is_finite() && y.is_finite() && y == x * (tau * tau),
                    "covariance entry scaling differs"
                );
            }
        }
    }
    let mut restored = scaled.clone();
    restored["base_model"]["covariances"] = original["base_model"]["covariances"].clone();
    ensure!(
        restored == *original,
        "non-covariance atlas metadata changed"
    );
    Ok(())
}
fn encode_points(points: &[[f64; 3]]) -> Vec<u8> {
    points
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect()
}
fn cloud_bounds(exclusion: &SphereTree) -> ([f64; 3], [f64; 3]) {
    let low = std::array::from_fn(|k| {
        exclusion
            .shape
            .atoms
            .iter()
            .map(|a| a.center[k] - a.radius)
            .fold(f64::INFINITY, f64::min)
    });
    let high = std::array::from_fn(|k| {
        exclusion
            .shape
            .atoms
            .iter()
            .map(|a| a.center[k] + a.radius)
            .fold(f64::NEG_INFINITY, f64::max)
    });
    (low, high)
}
fn line(out: &mut impl Write, value: &Value) -> Result<()> {
    serde_json::to_writer(&mut *out, value)?;
    out.write_all(b"\n")?;
    out.flush()?;
    Ok(())
}
fn fingerprint(
    tree: &SphereTree,
    state: &[Pose],
    case: &Case,
    poses: [Pose; 2],
) -> Vec<[usize; 2]> {
    let labels = [case.root, case.child];
    let mut result = BTreeSet::new();
    for i in 0..2 {
        let p = Placed::new(poses[i]);
        for (j, &q) in state
            .iter()
            .enumerate()
            .filter(|(j, _)| !labels.contains(j))
        {
            if tree.overlaps(&p, &Placed::new(q)) {
                result.insert([labels[i].min(j), labels[i].max(j)]);
            }
        }
    }
    if tree.overlaps(&Placed::new(poses[0]), &Placed::new(poses[1])) {
        result.insert([labels[0].min(labels[1]), labels[0].max(labels[1])]);
    }
    result.into_iter().collect()
}
fn execute(
    plan: &Plan,
    out: &mut impl Write,
    cloud_meta: &mut impl Write,
    cloud_raw: &mut impl Write,
) -> Result<Value> {
    let allocation: Value = serde_json::from_slice(&plan.scientific_allocation.read()?)?;
    ensure!(
        allocation["schema"] == "fft-width-allocation-v1"
            && allocation["new_outer_attempts"] == 1536
            && allocation["covariance_scales"] == json!([0.125, 0.25, 0.5])
            && allocation["methods"] == json!(["unguided", "m4"]),
        "scientific allocation differs"
    );
    ensure!(
        plan.schema == "fft-width-screen-v1" && plan.density_law == "map-factor-full-mixture-v1",
        "unknown plan/law"
    );
    ensure!(
        plan.root_cap == 32
            && plan.internal_cap == 32
            && plan.factorized_joint_cap == 1
            && plan.factorized_order == "root_first"
            && plan.attempts_per_context == 32
            && plan.master_seed == 6100300401
            && plan.cloud_raw_count == 16384
            && plan
                .scaled_atlases
                .iter()
                .map(|a| a.tau)
                .collect::<Vec<_>>()
                == [0.125, 0.25, 0.5],
        "allocation differs"
    );
    let embedded: Value = serde_json::from_slice(BUNDLE)?;
    let hashes: BTreeMap<String, String> = embedded["files"]
        .as_object()
        .context("source bundle")?
        .iter()
        .map(|(k, v)| {
            Ok((
                k.clone(),
                v["sha256"].as_str().context("source hash")?.into(),
            ))
        })
        .collect::<Result<_>>()?;
    ensure!(
        hashes == plan.compiled_source_sha256,
        "compiled source bindings differ"
    );
    let reference: Reference = serde_json::from_slice(&plan.reference_config.read()?)?;
    ensure!(
        reference.cases.len() == 8
            && reference.atlases.len() == 3
            && reference.depletant_radius == 1.4
            && reference.activity == 0.0275
            && reference.uniform_half_width == 160.
            && reference.uniform_probability == 0.5,
        "reference settings differ"
    );
    let config: Value = serde_json::from_slice(&reference.source_config.read()?)?;
    let frame: Value = serde_json::from_slice(&reference.source_frame.read()?)?;
    let archive: Value = serde_json::from_slice(&reference.source_freeze_manifest.read()?)?;
    let panel: Value = serde_json::from_slice(&reference.panel.read()?)?;
    ensure!(
        archive["frame_sha256"] == reference.source_frame.sha256
            && archive["shape_sha256"] == reference.shape.sha256,
        "archive mismatch"
    );
    ensure!(
        config["initial_poses"] == frame["poses"]
            && panel["cases"] == serde_json::to_value(&reference.cases)?,
        "source/panel differs"
    );
    ensure!(
        config["depletant_radius"] == reference.depletant_radius
            && config["reservoir_density"] == reference.activity
            && config["boundary"]["kind"] == "spherical"
            && config["boundary"]["radius"] == reference.wall_radius,
        "physical settings differ"
    );
    let state: Vec<Pose> = serde_json::from_value(config["initial_poses"].clone())?;
    let shape: Shape = serde_json::from_slice(&reference.shape.read()?)?;
    let core = SphereTree::new(shape.clone())?;
    let mut inflated = shape;
    for atom in &mut inflated.atoms {
        atom.radius += reference.depletant_radius;
    }
    let exclusion = SphereTree::new(inflated)?;
    let wall = Container::new(reference.wall_radius, &core)?;
    validate_state(&core, &wall, &state)?;
    let mut outer = 0usize;
    let mut raw_edges = 0usize;
    let mut candidates = 0usize;
    let mut clouds = 0usize;
    let mut retained_cloud_points = 0usize;
    let (low, high) = cloud_bounds(&exclusion);
    let width: [f64; 3] = std::array::from_fn(|k| high[k] - low[k]);
    ensure!(
        width.iter().all(|v| v.is_finite() && *v > 0.),
        "invalid cloud bounds"
    );
    ensure!(
        reference.atlases[1].name == "blind_fft512slots"
            && reference.atlases[1].model.sha256
                == "c460dc61fb7bf9e76f1d4dca77987b5886ea82cdda5d703ea5de6a25733fcb08",
        "FFT source differs"
    );
    let original: Value = serde_json::from_slice(&reference.atlases[1].model.read()?)?;
    let mut dockings = Vec::new();
    for atlas in &plan.scaled_atlases {
        let bytes = atlas.model.read()?;
        validate_scaled(&original, &serde_json::from_slice(&bytes)?, atlas.tau)?;
        // Fresh construction recomputes every Cholesky factor, inverse and determinant.
        let model = FrozenRelativePoseProposal::from_json_str_open(
            std::str::from_utf8(&bytes)?,
            [2. * (reference.wall_radius + core.bound); 3],
            0.1,
            &reference.shape.sha256,
        )?;
        dockings.push(DockingProposal::new(
            model,
            DockingMethod::PosteriorInvolution,
            0.,
            [0.; 3],
        )?);
    }
    for (ci, case) in reference.cases.iter().enumerate() {
        let context = FixedDimerContext::new(
            &core,
            &exclusion,
            &state,
            [case.root, case.child],
            case.anchor,
            Some(reference.wall_radius),
            [0.; 3],
        )?;
        let old = [state[case.root], state[case.child]];
        let old_contacts = fingerprint(&exclusion, &state, case, old);
        for attempt in 0..plan.attempts_per_context {
            let cloud_id = ci * plan.attempts_per_context + attempt;
            let cloud_seed = seed(plan.master_seed, ci, attempt, "cloud");
            let auxiliary_seed = seed(plan.master_seed, ci, attempt, "threshold");
            let cloud_clock = cpu_seconds();
            let mut cloud_rng = StdRng::seed_from_u64(cloud_seed);
            let uniforms: Vec<[f64; 3]> = (0..plan.cloud_raw_count)
                .map(|_| std::array::from_fn(|_| cloud_rng.random()))
                .collect();
            let points: Vec<[f64; 3]> = uniforms
                .iter()
                .map(|u| std::array::from_fn(|k| low[k] + width[k] * u[k]))
                .collect();
            let kept_indices: Vec<usize> = points
                .iter()
                .enumerate()
                .filter_map(|(i, &p)| exclusion.contains(p, 0.).then_some(i))
                .collect();
            let retained: Vec<[f64; 3]> = kept_indices.iter().map(|&i| points[i]).collect();
            let cloud_cpu = cpu_seconds() - cloud_clock;
            cloud_raw.write_all(&encode_points(&uniforms))?;
            cloud_raw.flush()?;
            let cloud_fingerprint: [u64; 4] = std::array::from_fn(|_| cloud_rng.random());
            line(
                cloud_meta,
                &json!({"cloud_id":cloud_id,"case_index":ci,"attempt":attempt,
                    "seed":cloud_seed,"threshold_seed":auxiliary_seed,"raw_count":plan.cloud_raw_count,
                    "raw_byte_offset":cloud_id*plan.cloud_raw_count*3*8,"raw_byte_length":plan.cloud_raw_count*3*8,
                    "low":low,"high":high,"width":width,"kept_indices":kept_indices,"retained_count":retained.len(),
                    "raw_uniform_sha256":hash_bytes(&encode_points(&uniforms)),"transformed_raw_sha256":hash_bytes(&encode_points(&points)),
                    "retained_points_sha256":hash_bytes(&encode_points(&retained)),"cloud_construction_cpu_seconds":cloud_cpu,
                    "rng_after_fingerprint":cloud_fingerprint}),
            )?;
            clouds += 1;
            retained_cloud_points += retained.len();
            let execution_order = method_order(plan.master_seed, ci, attempt);
            for (position, &(ai, method, order_seed)) in execution_order.iter().enumerate() {
                let atlas = &plan.scaled_atlases[ai];
                let docking = &dockings[ai];
                let stream_seed = seed(plan.master_seed, ci, attempt, "proposal");
                let mut rng = StdRng::seed_from_u64(stream_seed);
                let mut auxiliary_rng = StdRng::seed_from_u64(auxiliary_seed);
                let m = if method == "m4" { 4 } else { 0 };
                let raw = DefensiveDimerProposal::new(
                    docking,
                    reference.uniform_half_width,
                    reference.uniform_probability,
                )?;
                let density_clock = cpu_seconds();
                let old_coordinates = tree_coordinates(context.anchor(), old[0], old[1])?;
                let old_density_result = (|| -> Result<_> {
                    Ok([
                        raw.density(old_coordinates[0])?,
                        raw.density(old_coordinates[1])?,
                    ])
                })();
                let old_density_evaluation_cpu_seconds = cpu_seconds() - density_clock;
                let old_edges = match old_density_result {
                    Ok(value) => value,
                    Err(error) => {
                        line(
                            out,
                            &json!({"status":"fatal_source_density","atlas_index":ai,"tau":atlas.tau,"case_index":ci,"attempt":attempt,"method":method,"seed":stream_seed,"fatal_error":format!("{error:#}")}),
                        )?;
                        return Err(error);
                    }
                };
                let setup_clock = cpu_seconds();
                let guidance_result = if m == 4 {
                    AuxiliaryOverlapThreshold::new(&exclusion, retained.clone(), m).map(Some)
                } else {
                    Ok(None)
                };
                let guidance = match guidance_result {
                    Ok(value) => value,
                    Err(error) => {
                        line(
                            out,
                            &json!({"status":"fatal_guidance_setup","atlas_index":ai,"tau":atlas.tau,"case_index":ci,"attempt":attempt,"method":method,"cloud_id":cloud_id,"seed":stream_seed,"auxiliary_seed":auxiliary_seed,"fatal_error":format!("{error:#}")}),
                        )?;
                        return Err(error);
                    }
                };
                let guidance_setup_cpu_seconds = cpu_seconds() - setup_clock;
                let proposal = FactorizedDimerProposal::new(
                    raw,
                    FactorizedDimerCaps {
                        root: plan.root_cap,
                        internal: plan.internal_cap,
                        joint: plan.factorized_joint_cap,
                    },
                    FactorizedDimerOrder::RootFirst,
                );
                let clock = cpu_seconds();
                let result = if let Some(guidance) = &guidance {
                    proposal.propose_guided(&mut rng, &mut auxiliary_rng, &context, old, guidance)
                } else {
                    proposal.propose(&mut rng, &context, old)
                };
                // The checked accessor guards the mandatory auxiliary term;
                // keep its two components in the outcome as well.
                let complete_correction = result
                    .as_ref()
                    .ok()
                    .and_then(|v| v.candidate.as_ref().map(|_| v.complete_log_correction()))
                    .transpose();
                let proposal_cpu_seconds = cpu_seconds() - clock;
                let result: std::result::Result<Value, Value> =
                    result.map(|v| json!(v)).map_err(|e| json!(e));
                let auxiliary_fingerprint: [u64; 4] =
                    std::array::from_fn(|_| auxiliary_rng.random());
                let rng_fingerprint: [u64; 4] = std::array::from_fn(|_| rng.random());
                let mut row = json!({"atlas":atlas.name,"atlas_index":ai,"scale_index":ai,"source_atlas_index":1,"tau":atlas.tau,"case":case,"case_index":ci,
                        "attempt":attempt,"method":method,"execution_order":execution_order,"execution_position":position,
                        "order_seed":order_seed,"seed":stream_seed,"old":old,"anchor_pose":context.anchor(),
                        "old_contacts":old_contacts,"old_edges":old_edges,"old_coordinates":old_coordinates,"old_density_evaluation_cpu_seconds":old_density_evaluation_cpu_seconds,"proposal_cpu_seconds":proposal_cpu_seconds,"rng_after_fingerprint":rng_fingerprint,
                        "cloud_id":cloud_id,"cloud_seed":cloud_seed,"auxiliary_seed":auxiliary_seed,"m":m,"auxiliary_rng_after_fingerprint":auxiliary_fingerprint,
                        "cloud_construction_cpu_seconds":cloud_cpu,"guidance_setup_cpu_seconds":guidance_setup_cpu_seconds,
                        "standalone_proposal_cpu_seconds":if m == 4 { cloud_cpu+guidance_setup_cpu_seconds+proposal_cpu_seconds } else { proposal_cpu_seconds }});
                outer += 1;
                match complete_correction {
                    Ok(value) => row["complete_log_correction"] = json!(value),
                    Err(error) => {
                        row["status"] = json!("fatal_correction");
                        row["outcome"] = result.unwrap_or_else(|e| e);
                        row["fatal_error"] = json!(format!("{error:#}"));
                        line(out, &row)?;
                        return Err(error);
                    }
                }
                match result {
                    Err(error) => {
                        row["status"] = json!("fatal");
                        row["failure"] = error;
                        line(out, &row)?;
                        anyhow::bail!(
                                "fatal proposal at atlas={ai} case={ci} attempt={attempt} method={method}; partial trace retained"
                            );
                    }
                    Ok(outcome) => {
                        candidates += usize::from(!outcome["candidate"].is_null());
                        row["status"] = json!("completed");
                        row["outcome"] = outcome;
                        let clock = cpu_seconds();
                        let diagnostics = (|| -> Result<(usize, Vec<Option<Vec<[usize; 2]>>>)> {
                            let mut contacts = Vec::new();
                            let count;
                            let attempts = row["outcome"]["attempts"]
                                .as_array()
                                .context("factorized attempts")?;
                            count = attempts
                                .iter()
                                .map(|a| {
                                    Ok(a["root_draws"]
                                        .as_array()
                                        .context("root draw ledger")?
                                        .len()
                                        + a["internal_draws"]
                                            .as_array()
                                            .context("internal draw ledger")?
                                            .len())
                                })
                                .collect::<Result<Vec<usize>>>()?
                                .into_iter()
                                .sum();
                            for a in attempts {
                                let poses: Option<[Pose; 2]> =
                                    serde_json::from_value(a["proposed"].clone())?;
                                contacts
                                    .push(poses.map(|p| fingerprint(&exclusion, &state, case, p)));
                            }

                            Ok((count, contacts))
                        })();
                        match diagnostics {
                            Ok((count, contacts)) => {
                                raw_edges += count;
                                row["raw_edge_draws"] = json!(count);
                                row["raw_contacts"] = json!(contacts);
                            }
                            Err(error) => {
                                row["status"] = json!("fatal_diagnostic");
                                row["fatal_error"] = json!(format!("{error:#}"));
                                line(out, &row)?;
                                return Err(error);
                            }
                        }
                        row["contact_diagnostic_cpu_seconds"] = json!(cpu_seconds() - clock);
                        line(out, &row)?;
                    }
                }
            }
        }
    }
    ensure!(
        outer == 1536 && raw_edges <= 98304 && clouds == 256,
        "incomplete or exceeded allocation"
    );
    Ok(
        json!({"outer_attempts":outer,"raw_edge_draws":raw_edges,"candidates":candidates,"clouds":clouds,"raw_cloud_points":clouds*plan.cloud_raw_count,"retained_cloud_points":retained_cloud_points,"physical_bath_draws":0,"state_updates":0}),
    )
}
fn main() -> Result<()> {
    let clock = cpu_seconds();
    let args = Args::parse();
    let raw = fs::read(&args.config)?;
    let plan: Plan = serde_json::from_slice(&raw)?;
    let binding_raw = fs::read(&args.binding)?;
    let binding: Binding = serde_json::from_slice(&binding_raw)?;
    ensure!(
        binding.schema == "fft-width-probe-binding-v1",
        "binding schema"
    );
    ensure!(
        hash_bytes(&raw) == binding.config_sha256
            && hash_bytes(SOURCE) == binding.example_source_sha256
            && hash_bytes(BUNDLE) == binding.compiled_source_bundle_sha256
            && hash_file(&std::env::current_exe()?)? == binding.executable_sha256,
        "executable/config binding mismatch"
    );
    let protocol = plan.protocol.read()?;
    ensure!(
        hash_bytes(&protocol) == binding.protocol_sha256,
        "protocol mismatch"
    );
    fs::create_dir(&plan.output).context("output exists or parent missing")?;
    fs::write(plan.output.join("config.json"), raw)?;
    fs::write(plan.output.join("binding.json"), binding_raw)?;
    fs::write(plan.output.join("protocol.json"), protocol)?;
    fs::write(plan.output.join("source-bundle.json"), BUNDLE)?;
    fs::write(plan.output.join("example.rs"), SOURCE)?;
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(plan.output.join("attempts.jsonl"))?;
    let mut out = BufWriter::new(file);
    let mut cloud_meta = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(plan.output.join("clouds.jsonl"))?,
    );
    let mut cloud_raw = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(plan.output.join("cloud-uniforms.bin"))?,
    );
    let result = execute(&plan, &mut out, &mut cloud_meta, &mut cloud_raw);
    cloud_meta.flush()?;
    cloud_raw.flush()?;
    out.flush()?;
    let summary = match &result {
        Ok(v) => json!({"complete":true,"result":v}),
        Err(e) => json!({"complete":false,"error":format!("{e:#}")}),
    };
    save(
        &plan.output.join("terminal.json"),
        &json!({"summary":summary,"cpu_seconds":cpu_seconds()-clock,
        "attempts_sha256":hash_file(&plan.output.join("attempts.jsonl"))?,
        "clouds_sha256":hash_file(&plan.output.join("clouds.jsonl"))?,"cloud_uniforms_sha256":hash_file(&plan.output.join("cloud-uniforms.bin"))?}),
    )?;
    result.map(|_| ())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn distinct_domains_shared_arm_prefixes_and_allocation() {
        let mut all = BTreeSet::new();
        let mut first = [0usize; 6];
        for case in 0..8 {
            for attempt in 0..32 {
                for role in ["cloud", "threshold", "proposal"] {
                    assert!(all.insert(seed(6100300401, case, attempt, role)));
                }
                let order = method_order(6100300401, case, attempt);
                assert_eq!(order.len(), 6);
                assert_eq!(
                    order
                        .iter()
                        .map(|&(i, m, _)| (i, m))
                        .collect::<BTreeSet<_>>()
                        .len(),
                    6
                );
                for &(_, _, key) in &order {
                    assert!(all.insert(key));
                }
                first[order[0].0 * 2 + usize::from(order[0].1 == "m4")] += 1;
            }
        }
        assert_eq!(all.len(), 2304);
        assert!(first.iter().all(|&n| n > 20));
        assert_eq!(256 * 16384 * 3 * 8, 100663296);
        assert_eq!(256 * 6, 1536);
    }
    #[test]
    fn transform_preserves_cross_covariance_and_other_fields() {
        let covariance: Vec<Vec<f64>> = (0..6)
            .map(|i| (0..6).map(|j| if i == j { 2. } else { -0.1 }).collect())
            .collect();
        let original = json!({"schema":"reciprocal-pose-mixture-v1","reciprocal_components":[true],"base_model":{"schema":"weighted-pose-mixture-v1","covariances":[covariance],"means":[[1,2,3,4,5,6]],"anchors":["unchanged"],"angular_length":35,"weights":[1]}});
        for tau in [0.125, 0.25, 0.5] {
            let mut scaled = original.clone();
            for row in scaled["base_model"]["covariances"][0]
                .as_array_mut()
                .unwrap()
            {
                for x in row.as_array_mut().unwrap() {
                    *x = json!(x.as_f64().unwrap() * tau * tau);
                }
            }
            validate_scaled(&original, &scaled, tau).unwrap();
            let mut bad = scaled.clone();
            bad["base_model"]["means"][0][0] = json!(2);
            assert!(validate_scaled(&original, &bad, tau).is_err());
            let mut bad = scaled.clone();
            bad["base_model"]["covariances"][0][0][1] = json!(0.);
            assert!(validate_scaled(&original, &bad, tau).is_err());
            let mut bad = scaled.clone();
            bad["reciprocal_components"][0] = json!(false);
            assert!(validate_scaled(&original, &bad, tau).is_err());
        }
    }
    #[test]
    fn little_endian_raw_coordinate_record() {
        let bytes = encode_points(&[[0., 0.5, 1.], [-1., 2., 3.]]);
        assert_eq!(bytes.len(), 48);
        assert_eq!(f64::from_le_bytes(bytes[8..16].try_into().unwrap()), 0.5);
        assert_eq!(f64::from_le_bytes(bytes[24..32].try_into().unwrap()), -1.);
    }
}
