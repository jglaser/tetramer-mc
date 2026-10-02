//! Frozen paired-cap protein proposal screen; no bath draws or physical updates.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    path::PathBuf,
};
use tetramer_mc::{
    capped_dimer::{CappedDimerProposal, FixedDimerContext},
    defensive_dimer_proposal::DefensiveDimerProposal,
    docking::{DockingMethod, DockingProposal},
    geometry::{Placed, Shape, SphereTree},
    math::Pose,
    proposal::FrozenRelativePoseProposal,
    simulation::{cpu_seconds, hash_bytes, hash_file, save},
    spherical::{Container, validate_state},
};
const BUNDLE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/source-bundle.json"));
const SOURCE: &[u8] = include_bytes!("capped_dimer_probe.rs");
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
    protocol: BoundFile,
    reference_config: BoundFile,
    master_seed: u64,
    attempts_per_context: usize,
    caps: Vec<usize>,
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
fn seed(master: u64, atlas: usize, case: usize, attempt: usize) -> u64 {
    let h =
        hash_bytes(format!("capped-dimer-probe-v1/{master}/{atlas}/{case}/{attempt}").as_bytes());
    u64::from_str_radix(&h[..16], 16).unwrap()
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
fn execute(plan: &Plan, out: &mut impl Write) -> Result<Value> {
    ensure!(
        plan.schema == "capped-dimer-screen-v1" && plan.density_law == "map-factor-full-mixture-v1",
        "unknown plan/law"
    );
    ensure!(
        plan.caps == [1, 8, 32] && plan.attempts_per_context == 32,
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
    let mut raw_trials = 0usize;
    let mut candidates = 0usize;
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
                let stream_seed = seed(plan.master_seed, ai, ci, attempt);
                for &cap in &plan.caps {
                    let mut rng = StdRng::seed_from_u64(stream_seed);
                    let raw = DefensiveDimerProposal::new(
                        &docking,
                        reference.uniform_half_width,
                        reference.uniform_probability,
                    )?;
                    let proposal = CappedDimerProposal::new(raw, cap);
                    let clock = cpu_seconds();
                    let outcome = proposal.propose(&mut rng, &context, old);
                    let proposal_cpu_seconds = cpu_seconds() - clock;
                    let rng_fingerprint: [u64; 4] = std::array::from_fn(|_| rng.random());
                    let mut row = json!({"atlas":atlas.name,"atlas_index":ai,"case":case,"case_index":ci,
                        "attempt":attempt,"cap":cap,"seed":stream_seed,"old":old,"anchor_pose":context.anchor(),
                        "old_contacts":old_contacts,"proposal_cpu_seconds":proposal_cpu_seconds,
                        "rng_after_fingerprint":rng_fingerprint});
                    outer += 1;
                    match outcome {
                        Err(error) => {
                            row["status"] = json!("fatal");
                            row["failure"] = json!(error);
                            line(out, &row)?;
                            anyhow::bail!(
                                "fatal proposal at atlas={ai} case={ci} attempt={attempt} cap={cap}: {error}"
                            );
                        }
                        Ok(outcome) => {
                            raw_trials += outcome.trials.len();
                            candidates += usize::from(outcome.candidate.is_some());
                            let clock = cpu_seconds();
                            let raw_contacts: Result<Vec<_>> = outcome
                                .trials
                                .iter()
                                .map(|trial| {
                                    let candidate = trial
                                        .draw
                                        .as_ref()
                                        .and_then(|d| d.candidate.as_ref())
                                        .context("missing successful raw endpoint")?;
                                    Ok(fingerprint(
                                        &exclusion,
                                        &state,
                                        case,
                                        [candidate.root, candidate.child],
                                    ))
                                })
                                .collect();
                            row["status"] = json!("completed");
                            row["outcome"] = json!(outcome);
                            row["raw_contacts"] = match raw_contacts {
                                Ok(contacts) => json!(contacts),
                                Err(error) => {
                                    row["status"] = json!("fatal_diagnostic");
                                    row["fatal_error"] = json!(format!("{error:#}"));
                                    line(out, &row)?;
                                    return Err(error);
                                }
                            };
                            row["contact_diagnostic_cpu_seconds"] = json!(cpu_seconds() - clock);
                            line(out, &row)?;
                        }
                    }
                }
            }
        }
    }
    ensure!(
        outer == 2304 && raw_trials <= 31488,
        "incomplete or exceeded allocation"
    );
    Ok(
        json!({"outer_attempts":outer,"raw_trials":raw_trials,"candidates":candidates,"physical_draws":0,"state_updates":0}),
    )
}
fn main() -> Result<()> {
    let args = Args::parse();
    let raw = fs::read(&args.config)?;
    let plan: Plan = serde_json::from_slice(&raw)?;
    let binding_raw = fs::read(&args.binding)?;
    let binding: Binding = serde_json::from_slice(&binding_raw)?;
    ensure!(
        binding.schema == "capped-dimer-probe-binding-v1",
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
    let clock = cpu_seconds();
    let result = execute(&plan, &mut out);
    out.flush()?;
    let summary = match &result {
        Ok(v) => json!({"complete":true,"result":v}),
        Err(e) => json!({"complete":false,"error":format!("{e:#}")}),
    };
    save(
        &plan.output.join("terminal.json"),
        &json!({"summary":summary,"cpu_seconds":cpu_seconds()-clock,
        "attempts_sha256":hash_file(&plan.output.join("attempts.jsonl"))?}),
    )?;
    result.map(|_| ())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn capped_streams_share_prefix_without_sharing_attempts() {
        let mut seeds = BTreeSet::new();
        for atlas in 0..3 {
            for case in 0..8 {
                for attempt in 0..32 {
                    let seed = seed(6100202901, atlas, case, attempt);
                    assert!(seeds.insert(seed));
                    let mut a = StdRng::seed_from_u64(seed);
                    let mut b = StdRng::seed_from_u64(seed);
                    for _ in 0..128 {
                        assert_eq!(a.random::<u64>(), b.random::<u64>());
                    }
                }
            }
        }
        assert_eq!(3 * 8 * 32 * (1 + 8 + 32), 31488);
    }
}
