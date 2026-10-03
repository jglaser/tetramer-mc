//! Frozen auxiliary-overlap passive screen with a reused factorized baseline.
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
    docking::{DockingMethod, DockingProposal},
    factorized_dimer::{FactorizedDimerCaps, FactorizedDimerOrder, FactorizedDimerProposal},
    geometry::{Placed, Shape, SphereTree},
    math::Pose,
    proposal::FrozenRelativePoseProposal,
    simulation::{cpu_seconds, hash_bytes, hash_file, save},
    spherical::{validate_state, Container},
};
const BUNDLE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/source-bundle.json"));
const SOURCE: &[u8] = include_bytes!("auxiliary_overlap_probe.rs");
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
    proposal_master_seed: u64,
    auxiliary_master_seed: u64,
    cloud_raw_count: usize,
    baseline: BoundFile,
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
fn seed(master: u64, atlas: usize, case: usize, attempt: usize, role: &str) -> u64 {
    let h = hash_bytes(
        format!("auxiliary-overlap-probe-v1/{master}/{atlas}/{case}/{attempt}/{role}").as_bytes(),
    );
    u64::from_str_radix(&h[..16], 16).unwrap()
}
fn proposal_seed(master: u64, atlas: usize, case: usize, attempt: usize) -> u64 {
    let h = hash_bytes(
        format!("factorized-dimer-probe-v1/{master}/{atlas}/{case}/{attempt}/factorized")
            .as_bytes(),
    );
    u64::from_str_radix(&h[..16], 16).unwrap()
}
fn method_order(master: u64, atlas: usize, case: usize, attempt: usize) -> [&'static str; 2] {
    if seed(master, atlas, case, attempt, "execution_order") & 1 == 0 {
        ["m1", "m4"]
    } else {
        ["m4", "m1"]
    }
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
    ensure!(
        plan.scientific_allocation.sha256
            == "d768b70d07e5837218217d512a3266bb75b221916fb3ee74c7312aba15126b1f",
        "scientific allocation binding differs"
    );
    let _allocation = plan.scientific_allocation.read()?;
    ensure!(
        plan.schema == "auxiliary-overlap-screen-v1"
            && plan.density_law == "map-factor-full-mixture-v1",
        "unknown plan/law"
    );
    ensure!(
        plan.root_cap == 32
            && plan.internal_cap == 32
            && plan.factorized_joint_cap == 1
            && plan.factorized_order == "root_first"
            && plan.attempts_per_context == 32
            && plan.proposal_master_seed == 6100203101
            && plan.auxiliary_master_seed == 6100300101
            && plan.cloud_raw_count == 16384,
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
    let _baseline_bytes = plan.baseline.read()?;
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
    for (ai, atlas) in reference.atlases.iter().enumerate() {
        let model_bytes = atlas.model.read()?;
        let model = FrozenRelativePoseProposal::from_json_str_open(
            std::str::from_utf8(&model_bytes)?,
            [2. * (reference.wall_radius + core.bound); 3],
            0.1,
            &reference.shape.sha256,
        )?;
        // Correlation is irrelevant for independent destination draws.
        let docking = DockingProposal::new(model, DockingMethod::PosteriorInvolution, 0., [0.; 3])?;
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
                let cloud_id =
                    (ai * reference.cases.len() + ci) * plan.attempts_per_context + attempt;
                let cloud_seed = seed(plan.auxiliary_master_seed, ai, ci, attempt, "cloud");
                let auxiliary_seed = seed(plan.auxiliary_master_seed, ai, ci, attempt, "threshold");
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
                    &json!({"cloud_id":cloud_id,"atlas_index":ai,"case_index":ci,"attempt":attempt,
                    "seed":cloud_seed,"threshold_seed":auxiliary_seed,"raw_count":plan.cloud_raw_count,
                    "raw_byte_offset":cloud_id*plan.cloud_raw_count*3*8,"raw_byte_length":plan.cloud_raw_count*3*8,
                    "low":low,"high":high,"width":width,"kept_indices":kept_indices,"retained_count":retained.len(),
                    "raw_uniform_sha256":hash_bytes(&encode_points(&uniforms)),"transformed_raw_sha256":hash_bytes(&encode_points(&points)),
                    "retained_points_sha256":hash_bytes(&encode_points(&retained)),"cloud_construction_cpu_seconds":cloud_cpu,
                    "rng_after_fingerprint":cloud_fingerprint}),
                )?;
                clouds += 1;
                retained_cloud_points += retained.len();
                let execution_order = method_order(plan.auxiliary_master_seed, ai, ci, attempt);
                let order_seed = seed(
                    plan.auxiliary_master_seed,
                    ai,
                    ci,
                    attempt,
                    "execution_order",
                );
                for (position, method) in execution_order.iter().enumerate() {
                    let stream_seed = proposal_seed(plan.proposal_master_seed, ai, ci, attempt);
                    let mut rng = StdRng::seed_from_u64(stream_seed);
                    let mut auxiliary_rng = StdRng::seed_from_u64(auxiliary_seed);
                    let m = if *method == "m1" { 1 } else { 4 };
                    let raw = DefensiveDimerProposal::new(
                        &docking,
                        reference.uniform_half_width,
                        reference.uniform_probability,
                    )?;
                    let setup_clock = cpu_seconds();
                    let guidance = match AuxiliaryOverlapThreshold::new(
                        &exclusion,
                        retained.clone(),
                        m,
                    ) {
                        Ok(value) => value,
                        Err(error) => {
                            line(
                                out,
                                &json!({"status":"fatal_guidance_setup", "atlas_index":ai,
                                "case_index":ci,"attempt":attempt,"method":method,"cloud_id":cloud_id,
                                "seed":stream_seed,"auxiliary_seed":auxiliary_seed,
                                "fatal_error":format!("{error:#}")}),
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
                    let result = proposal.propose_guided(
                        &mut rng,
                        &mut auxiliary_rng,
                        &context,
                        old,
                        &guidance,
                    );
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
                    let mut row = json!({"atlas":atlas.name,"atlas_index":ai,"case":case,"case_index":ci,
                        "attempt":attempt,"method":method,"execution_order":execution_order,"execution_position":position,
                        "order_seed":order_seed,"seed":stream_seed,"old":old,"anchor_pose":context.anchor(),
                        "old_contacts":old_contacts,"proposal_cpu_seconds":proposal_cpu_seconds,"rng_after_fingerprint":rng_fingerprint,
                        "cloud_id":cloud_id,"cloud_seed":cloud_seed,"auxiliary_seed":auxiliary_seed,"m":m,"auxiliary_rng_after_fingerprint":auxiliary_fingerprint,
                        "cloud_construction_cpu_seconds":cloud_cpu,"guidance_setup_cpu_seconds":guidance_setup_cpu_seconds,
                        "standalone_proposal_cpu_seconds":cloud_cpu+guidance_setup_cpu_seconds+proposal_cpu_seconds});
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
                            let diagnostics =
                                (|| -> Result<(usize, Vec<Option<Vec<[usize; 2]>>>)> {
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
                                        contacts.push(
                                            poses.map(|p| fingerprint(&exclusion, &state, case, p)),
                                        );
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
    }
    ensure!(
        outer == 1536 && raw_edges <= 98304 && clouds == 768,
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
        binding.schema == "auxiliary-overlap-probe-binding-v1",
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
    fn domains_and_shared_prefix_seeds() {
        let mut seeds = BTreeSet::new();
        let mut first = [0usize; 2];
        for a in 0..3 {
            for c in 0..8 {
                for i in 0..32 {
                    for role in ["cloud", "threshold", "execution_order"] {
                        assert!(seeds.insert(seed(6100300101, a, c, i, role)));
                    }
                    assert!(seeds.insert(proposal_seed(6100203101, a, c, i)));
                    let order = method_order(6100300101, a, c, i);
                    first[usize::from(order[0] == "m4")] += 1;
                    assert!(order == ["m1", "m4"] || order == ["m4", "m1"]);
                }
            }
        }
        assert_eq!(seeds.len(), 3072);
        assert!(first.iter().all(|&n| n > 250));
        assert_eq!(768 * 16384 * 3 * 8, 301989888);
        assert_eq!(1536 * 64, 98304);
    }
    #[test]
    fn little_endian_raw_coordinate_record() {
        let bytes = encode_points(&[[0., 0.5, 1.], [-1., 2., 3.]]);
        assert_eq!(bytes.len(), 48);
        assert_eq!(f64::from_le_bytes(bytes[8..16].try_into().unwrap()), 0.5);
        assert_eq!(f64::from_le_bytes(bytes[24..32].try_into().unwrap()), -1.);
    }
}
