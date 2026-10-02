//! Frozen, reset-to-source protein destination screen. No bath draws or MC updates.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use rand::{SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    path::PathBuf,
};
use tetramer_mc::{
    defensive_dimer_proposal::DefensiveDimerProposal,
    dimer_tree_proposal::{DimerTreeProposal, tree_coordinates},
    docking::{DockingMethod, DockingProposal},
    geometry::{Placed, Shape, SphereTree},
    math::Pose,
    proposal::FrozenRelativePoseProposal,
    simulation::{cpu_seconds, hash_bytes, hash_file, save},
    spherical::{Container, validate_state},
};
const BUNDLE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/source-bundle.json"));
const SOURCE: &[u8] = include_bytes!("dimer_destination_probe.rs");
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
#[serde(deny_unknown_fields)]
struct Atlas {
    name: String,
    model: BoundFile,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Case {
    name: String,
    root: usize,
    child: usize,
    anchor: usize,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Law {
    Tree,
    IndependentLearned,
    Defensive,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    schema: String,
    protocol: BoundFile,
    source_config: BoundFile,
    source_frame: BoundFile,
    source_freeze_manifest: BoundFile,
    shape: BoundFile,
    panel: BoundFile,
    atlases: Vec<Atlas>,
    cases: Vec<Case>,
    laws: Vec<Law>,
    master_seed: u64,
    attempts_per_cell: usize,
    correlation: f64,
    depletant_radius: f64,
    activity: f64,
    wall_radius: f64,
    uniform_half_width: f64,
    uniform_probability: f64,
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
fn validate_case(case: &Case, count: usize) -> Result<()> {
    ensure!(
        case.root < count && case.child < count && case.anchor < count,
        "label out of range"
    );
    ensure!(
        case.root != case.child && case.root != case.anchor && case.child != case.anchor,
        "tree requires distinct fixed labels"
    );
    Ok(())
}
fn seed(master: u64, atlas: usize, case: usize, law: usize) -> u64 {
    let h = hash_bytes(
        format!("protein-dimer-destinations-v1/{master}/{atlas}/{case}/{law}").as_bytes(),
    );
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
    selected: [Pose; 2],
) -> Vec<[usize; 2]> {
    let mut edges = BTreeSet::new();
    let labels = [case.root, case.child];
    for i in 0..2 {
        let p = Placed::new(selected[i]);
        for (j, q) in state.iter().enumerate() {
            if labels.contains(&j) {
                continue;
            }
            if tree.overlaps(&p, &Placed::new(*q)) {
                edges.insert([labels[i].min(j), labels[i].max(j)]);
            }
        }
    }
    if tree.overlaps(&Placed::new(selected[0]), &Placed::new(selected[1])) {
        edges.insert([case.root.min(case.child), case.root.max(case.child)]);
    }
    edges.into_iter().collect()
}
fn execute(plan: &Plan, out: &mut impl Write) -> Result<Value> {
    ensure!(plan.schema == "dimer-destination-screen-v1", "unknown plan");
    ensure!(
        plan.attempts_per_cell > 0
            && !plan.atlases.is_empty()
            && !plan.cases.is_empty()
            && !plan.laws.is_empty(),
        "empty allocation"
    );
    ensure!(
        plan.uniform_probability > 0. && plan.uniform_probability < 1.,
        "defensive probability must be interior"
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
        "source bindings differ"
    );
    let config: Value = serde_json::from_slice(&plan.source_config.read()?)?;
    let frame: Value = serde_json::from_slice(&plan.source_frame.read()?)?;
    let archive: Value = serde_json::from_slice(&plan.source_freeze_manifest.read()?)?;
    ensure!(
        archive["frame_sha256"] == plan.source_frame.sha256
            && archive["shape_sha256"] == plan.shape.sha256,
        "archive mismatch"
    );
    ensure!(
        config["initial_poses"] == frame["poses"],
        "source poses differ"
    );
    ensure!(
        config["depletant_radius"] == plan.depletant_radius
            && config["reservoir_density"] == plan.activity
            && config["boundary"]["kind"] == "spherical"
            && config["boundary"]["radius"] == plan.wall_radius,
        "source physical settings differ"
    );
    let panel: Value = serde_json::from_slice(&plan.panel.read()?)?;
    ensure!(
        panel["cases"] == serde_json::to_value(&plan.cases)?,
        "panel labels differ"
    );
    let state: Vec<Pose> = serde_json::from_value(config["initial_poses"].clone())?;
    let shape: Shape = serde_json::from_slice(&plan.shape.read()?)?;
    let tree = SphereTree::new(shape.clone())?;
    let mut inflated = shape;
    for a in &mut inflated.atoms {
        a.radius += plan.depletant_radius;
    }
    let contacts = SphereTree::new(inflated)?;
    let wall = Container::new(plan.wall_radius, &tree)?;
    validate_state(&tree, &wall, &state)?;
    for case in &plan.cases {
        validate_case(case, state.len())?;
        for edge in tree_coordinates(state[case.anchor], state[case.root], state[case.child])? {
            ensure!(
                edge.position
                    .iter()
                    .all(|x| x.abs() <= plan.uniform_half_width),
                "source edge outside defensive cube"
            );
        }
    }
    let mut expected = 0usize;
    let mut totals =
        json!({"attempts":0,"finite_endpoints":0,"hard_valid":0,"proposal_errors":0,"nulls":0});
    for (ai, atlas) in plan.atlases.iter().enumerate() {
        let bytes = atlas.model.read()?;
        let model = FrozenRelativePoseProposal::from_json_str_open(
            std::str::from_utf8(&bytes)?,
            [2. * (plan.wall_radius + tree.bound); 3],
            0.1,
            &plan.shape.sha256,
        )?;
        let docking = DockingProposal::new(
            model,
            DockingMethod::PosteriorInvolution,
            plan.correlation,
            [0.; 3],
        )?;
        let correlated = DimerTreeProposal::new(&docking)?;
        let learned = DefensiveDimerProposal::new(&docking, plan.uniform_half_width, 0.)?;
        let defensive = DefensiveDimerProposal::new(
            &docking,
            plan.uniform_half_width,
            plan.uniform_probability,
        )?;
        for (ci, case) in plan.cases.iter().enumerate() {
            let old = [state[case.root], state[case.child]];
            let anchor = state[case.anchor];
            let old_edges = tree_coordinates(anchor, old[0], old[1])?;
            let old_contacts = fingerprint(&contacts, &state, case, old);
            let old_densities = [
                defensive.density(old_edges[0])?,
                defensive.density(old_edges[1])?,
            ];
            for (li, law) in plan.laws.iter().enumerate() {
                let cell_seed = seed(plan.master_seed, ai, ci, li);
                let mut rng = StdRng::seed_from_u64(cell_seed);
                for attempt in 0..plan.attempts_per_cell {
                    expected += 1;
                    let start = cpu_seconds();
                    let result: Result<(Value, Option<([Pose; 2], f64)>)> = match law {
                        Law::Tree => {
                            correlated
                                .propose(&mut rng, anchor, old[0], old[1])
                                .map(|o| {
                                    let c = o.candidate.as_ref().map(|c| {
                                        ([c.root, c.child], c.diagnostics.log_reverse_forward)
                                    });
                                    (json!(o), c)
                                })
                        }
                        Law::IndependentLearned | Law::Defensive => {
                            let proposal = if matches!(law, Law::IndependentLearned) {
                                &learned
                            } else {
                                &defensive
                            };
                            proposal.propose(&mut rng, anchor, old[0], old[1]).map(|o| {
                                let c = o.candidate.as_ref().map(|c| {
                                    ([c.root, c.child], c.diagnostics.log_reverse_forward)
                                });
                                (json!(o), c)
                            })
                        }
                    };
                    let mut row = json!({"atlas":atlas.name,"case":case,"law":law,"cell_seed":cell_seed,"attempt":attempt,
                        "old_selected":old,"anchor_pose":anchor,"old_edges":old_edges,"old_densities":old_densities,
                        "old_contacts":old_contacts,"proposal_cpu_seconds":cpu_seconds()-start});
                    totals["attempts"] = json!(expected);
                    match result {
                        Err(e) => {
                            row["status"] = json!("proposal_error");
                            row["error"] = json!(format!("{e:#}"));
                            totals["proposal_errors"] =
                                json!(totals["proposal_errors"].as_u64().unwrap() + 1);
                        }
                        Ok((trace, None)) => {
                            row["status"] = json!("proposal_null");
                            row["outcome"] = trace;
                            totals["nulls"] = json!(totals["nulls"].as_u64().unwrap() + 1);
                        }
                        Ok((trace, Some((proposed, log_ratio)))) => {
                            totals["finite_endpoints"] =
                                json!(totals["finite_endpoints"].as_u64().unwrap() + 1);
                            row["status"] = json!("finite_endpoint");
                            row["outcome"] = trace;
                            row["proposed"] = json!(proposed);
                            row["log_reverse_forward"] = if log_ratio == f64::NEG_INFINITY {
                                json!("-inf")
                            } else {
                                json!(log_ratio)
                            };
                            let clock = cpu_seconds();
                            let hard_edges = fingerprint(&tree, &state, case, proposed);
                            let wall_valid = proposed.iter().all(|p| wall.contains(*p));
                            let hard_valid = hard_edges.is_empty() && wall_valid;
                            row["hard_overlap_edges"] = json!(hard_edges);
                            row["wall_valid"] = json!(wall_valid);
                            row["hard_valid"] = json!(hard_valid);
                            row["hard_cpu_seconds"] = json!(cpu_seconds() - clock);
                            if hard_valid {
                                totals["hard_valid"] =
                                    json!(totals["hard_valid"].as_u64().unwrap() + 1);
                            }
                            let clock = cpu_seconds();
                            row["new_contacts"] =
                                json!(fingerprint(&contacts, &state, case, proposed));
                            row["contact_cpu_seconds"] = json!(cpu_seconds() - clock);
                            let diagnostics = (|| -> Result<Value> {
                                let edges = tree_coordinates(anchor, proposed[0], proposed[1])?;
                                Ok(
                                    json!({"edges":edges,"densities":[defensive.density(edges[0])?,defensive.density(edges[1])?]}),
                                )
                            })();
                            match diagnostics {
                                Ok(d) => {
                                    row["new_edges"] = d["edges"].clone();
                                    row["new_densities"] = d["densities"].clone();
                                }
                                Err(e) => {
                                    row["diagnostic_error"] = json!(format!("{e:#}"));
                                }
                            }
                        }
                    }
                    line(out, &row)?;
                }
            }
        }
    }
    ensure!(
        expected
            == plan.atlases.len() * plan.cases.len() * plan.laws.len() * plan.attempts_per_cell,
        "incomplete allocation"
    );
    totals["physical_draws"] = json!(0);
    totals["state_updates"] = json!(0);
    Ok(totals)
}
fn main() -> Result<()> {
    let args = Args::parse();
    let raw = fs::read(&args.config)?;
    let plan: Plan = serde_json::from_slice(&raw)?;
    let binding_raw = fs::read(&args.binding)?;
    let binding: Binding = serde_json::from_slice(&binding_raw)?;
    ensure!(
        binding.schema == "dimer-destination-binding-v1",
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
    fn fixed_labels_and_stream_partition() {
        assert!(
            validate_case(
                &Case {
                    name: "valid".into(),
                    root: 0,
                    child: 1,
                    anchor: 2
                },
                3
            )
            .is_ok()
        );
        assert!(
            validate_case(
                &Case {
                    name: "duplicate".into(),
                    root: 0,
                    child: 1,
                    anchor: 1
                },
                3
            )
            .is_err()
        );
        assert!(
            validate_case(
                &Case {
                    name: "outside".into(),
                    root: 0,
                    child: 1,
                    anchor: 3
                },
                3
            )
            .is_err()
        );
        let mut seeds = BTreeSet::new();
        for a in 0..3 {
            for c in 0..8 {
                for l in 0..3 {
                    assert!(seeds.insert(seed(123, a, c, l)));
                }
            }
        }
    }
    #[test]
    fn fingerprint_uses_new_internal_pair_and_keeps_all_collisions() {
        use tetramer_mc::geometry::Atom;
        let tree = SphereTree::new(Shape {
            name: "sphere".into(),
            volume: 0.,
            atoms: vec![Atom {
                center: [0.; 3],
                radius: 1.,
            }],
        })
        .unwrap();
        let pose = |x| Pose {
            position: [x, 0., 0.],
            orientation: [1., 0., 0., 0.],
        };
        let state = vec![pose(0.), pose(3.), pose(6.)];
        let case = Case {
            name: "x".into(),
            root: 0,
            child: 1,
            anchor: 2,
        };
        assert!(fingerprint(&tree, &state, &case, [pose(0.), pose(3.)]).is_empty());
        assert_eq!(
            fingerprint(&tree, &state, &case, [pose(4.), pose(5.)]),
            vec![[0, 1], [1, 2]]
        );
    }
}
