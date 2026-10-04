//! Passive construction diagnostics at explicitly bound singleton contexts.
//! No source selection, trajectories, RNG, proposals, bath or native observer.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};
use tetramer_mc::{
    docking::{DockingMethod, DockingProposal},
    geometry::{Shape, SphereTree},
    math::{Pose, norm},
    oligomer_proposal::{FusionBuildDiagnostics, FusionFitPolicy, OligomerConfig, OligomerMixture},
    proposal::FrozenRelativePoseProposal,
    simulation::{Config, cpu_seconds, hash_bytes, hash_file},
    spherical::Container,
};

const SCHEMA: &str = "singleton-fusion-diagnostic-manifest-v1";
const BUNDLE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/source-bundle.json"));
const SOURCE: &[u8] = include_bytes!("singleton-fusion-diagnostic.rs");
const IDENTITY: Pose = Pose {
    position: [0.; 3],
    orientation: [1., 0., 0., 0.],
};

#[derive(Parser)]
#[command(about = "Passive singleton fusion construction; no sampling")]
struct Args {
    #[arg(long)]
    manifest: PathBuf,
    #[arg(long)]
    manifest_sha256: String,
    #[arg(long)]
    out: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BoundFile {
    path: PathBuf,
    sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum StateKind {
    Source,
    Prepared,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct NamedState {
    id: String,
    kind: StateKind,
    file: BoundFile,
    /// A single complete JSON state object, never a trajectory selector.
    poses_pointer: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Construction {
    id: String,
    state: String,
    moving: usize,
    /// Ordered exactly [other mobile, fixed external anchor].
    neighbors: [usize; 2],
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Caps {
    constructions: usize,
    fits: usize,
    center_checks: usize,
    core_overlap_calls: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Witness {
    executable_sha256: String,
    compiled_source_bundle_sha256: String,
    cli_source_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: String,
    context: String,
    coordinate_frame: String,
    wall_center: [f64; 3],
    output: PathBuf,
    config: BoundFile,
    model: BoundFile,
    shape: BoundFile,
    members: [usize; 2],
    anchor: usize,
    states: Vec<NamedState>,
    constructions: Vec<Construction>,
    oligomer: OligomerConfig,
    /// Omitted v1 fields preserve the historical fitter. This is a passive
    /// diagnostic policy, not a change to any production assembly kernel.
    #[serde(default)]
    fit_policy: FusionFitPolicy,
    caps: Caps,
    witness: Witness,
}

fn identity(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}
fn digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn validate_manifest(m: &Manifest) -> Result<()> {
    ensure!(
        m.schema == SCHEMA && identity(&m.context),
        "invalid manifest schema/context"
    );
    ensure!(
        m.coordinate_frame == "sphere_centered" && m.wall_center == [0.; 3],
        "explicit origin-centered sphere coordinates required; no recentering"
    );
    ensure!(m.output.is_absolute(), "output must be absolute");
    ensure!(
        m.members[0] != m.members[1] && !m.members.contains(&m.anchor),
        "invalid context labels"
    );
    ensure!(
        m.oligomer == OligomerConfig::default(),
        "diagnostic must use the frozen singleton fusion configuration"
    );
    m.oligomer.validate()?;
    ensure!(
        (1..=5).contains(&m.states.len())
            && m.states
                .iter()
                .filter(|s| s.kind == StateKind::Source)
                .count()
                == 1,
        "one shared source and at most four prepared states required"
    );
    let mut state_ids = BTreeSet::new();
    for state in &m.states {
        ensure!(
            identity(&state.id) && state_ids.insert(state.id.as_str()),
            "duplicate/invalid state ID"
        );
        ensure!(
            matches!(state.poses_pointer.as_str(), "/poses" | "/initial_poses"),
            "explicit complete-state pointer required"
        );
    }
    ensure!(
        m.constructions.len() == 2 * m.states.len()
            && m.constructions.len() <= m.caps.constructions
            && m.caps.constructions <= 10
            && m.caps.fits <= 40960
            && m.caps.center_checks <= 2560
            && m.caps.core_overlap_calls <= 673280,
        "construction allocation exceeds fixed caps"
    );
    ensure!(
        m.constructions
            .len()
            .checked_mul(m.oligomer.max_candidates)
            .context("fit cap overflow")?
            <= m.caps.fits
            && m.constructions
                .len()
                .checked_mul(m.oligomer.max_hard_checks)
                .context("center cap overflow")?
                <= m.caps.center_checks,
        "declared caps cannot cover the frozen constructor"
    );
    let mut ids = BTreeSet::new();
    let mut pairs = BTreeSet::new();
    for c in &m.constructions {
        ensure!(
            identity(&c.id) && ids.insert(c.id.as_str()) && state_ids.contains(c.state.as_str()),
            "invalid construction ID/state"
        );
        ensure!(
            m.members.contains(&c.moving) && pairs.insert((c.state.as_str(), c.moving)),
            "duplicate/invalid state/member construction"
        );
        let other = m.members[usize::from(c.moving == m.members[0])];
        ensure!(
            c.neighbors == [other, m.anchor],
            "neighbors must be ordered [other mobile, fixed anchor]"
        );
    }
    for reference in [&m.config, &m.model, &m.shape]
        .into_iter()
        .chain(m.states.iter().map(|s| &s.file))
    {
        ensure!(
            reference.path.is_absolute() && digest(&reference.sha256),
            "invalid absolute input binding"
        );
        ensure!(
            !reference.path.starts_with(&m.output),
            "input aliases output directory"
        );
    }
    ensure!(
        digest(&m.witness.executable_sha256)
            && digest(&m.witness.compiled_source_bundle_sha256)
            && digest(&m.witness.cli_source_sha256),
        "invalid executable/source witness"
    );
    Ok(())
}

#[derive(Default)]
struct Inputs {
    files: BTreeMap<PathBuf, String>,
}
impl Inputs {
    fn read(&mut self, reference: &BoundFile) -> Result<Vec<u8>> {
        ensure!(
            fs::canonicalize(&reference.path)? == reference.path,
            "input must be canonical: {}",
            reference.path.display()
        );
        let bytes = fs::read(&reference.path)?;
        ensure!(
            hash_bytes(&bytes) == reference.sha256,
            "changed input {}",
            reference.path.display()
        );
        if let Some(previous) = self
            .files
            .insert(reference.path.clone(), reference.sha256.clone())
        {
            ensure!(previous == reference.sha256, "contradictory input bindings");
        }
        Ok(bytes)
    }
    fn recheck(&self) -> Result<()> {
        for (path, expected) in &self.files {
            ensure!(
                hash_file(path)? == *expected,
                "input changed during diagnostic: {}",
                path.display()
            );
        }
        Ok(())
    }
}

struct Loaded {
    core: SphereTree,
    wall: Container,
    model: DockingProposal,
    states: BTreeMap<String, Vec<Pose>>,
}
fn load(m: &Manifest, inputs: &mut Inputs) -> Result<Loaded> {
    validate_manifest(m)?;
    let cfg: Config = serde_json::from_slice(&inputs.read(&m.config)?)?;
    cfg.validate()?;
    let shape_path = if cfg.shape.is_absolute() {
        cfg.shape.clone()
    } else {
        m.config
            .path
            .parent()
            .context("config parent")?
            .join(&cfg.shape)
    };
    ensure!(
        fs::canonicalize(shape_path)? == m.shape.path,
        "config shape path differs from explicit bound shape"
    );
    let radius = cfg
        .boundary
        .radius()
        .context("spherical source configuration required")?;
    let core = SphereTree::new(serde_json::from_slice::<Shape>(&inputs.read(&m.shape)?)?)?;
    let wall = Container::new(radius, &core)?;
    let bytes = inputs.read(&m.model)?;
    // Exactly the open posterior atlas loading used by evolving_dimer_benchmark.
    let model = DockingProposal::new(
        FrozenRelativePoseProposal::from_json_str_open(
            std::str::from_utf8(&bytes)?,
            [2. * (radius + core.bound); 3],
            0.1,
            &m.shape.sha256,
        )?,
        DockingMethod::PosteriorInvolution,
        0.,
        [0.; 3],
    )?;
    let mut states = BTreeMap::new();
    for state in &m.states {
        let value: Value = serde_json::from_slice(&inputs.read(&state.file)?)?;
        let poses: Vec<Pose> = serde_json::from_value(
            value
                .pointer(&state.poses_pointer)
                .context("missing complete state poses")?
                .clone(),
        )?;
        ensure!(
            poses.len() == cfg.initial_poses.len()
                && m.members
                    .iter()
                    .chain([&m.anchor])
                    .all(|&i| i < poses.len()),
            "state count/context label differs from source"
        );
        for pose in &poses {
            pose.validate()?;
            ensure!(
                (8. * (1. + norm(pose.position) + core.bound))
                    .powi(2)
                    .is_finite(),
                "unsupported geometry arithmetic magnitude"
            );
        }
        if state.kind == StateKind::Source {
            ensure!(
                poses == cfg.initial_poses,
                "shared source differs from bound config initial_poses"
            );
        }
        for (i, (pose, original)) in poses.iter().zip(&cfg.initial_poses).enumerate() {
            if !m.members.contains(&i) {
                ensure!(pose == original, "prepared state changed a fixed spectator");
            }
        }
        states.insert(state.id.clone(), poses);
    }
    let bound = m
        .constructions
        .len()
        .checked_mul(m.oligomer.max_hard_checks)
        .and_then(|v| v.checked_mul(cfg.initial_poses.len() - 1))
        .context("core-overlap cap overflow")?;
    ensure!(
        bound <= m.caps.core_overlap_calls,
        "declared core-overlap budget cannot cover all spectators"
    );
    inputs.recheck()?;
    Ok(Loaded {
        core,
        wall,
        model,
        states,
    })
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}
fn save_new(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    write_new(path, &bytes)
}
fn emit(file: &mut File, value: &Value) -> Result<()> {
    serde_json::to_writer(&mut *file, value)?;
    file.write_all(b"\n")?;
    file.flush()?;
    file.sync_all()?;
    Ok(())
}

#[derive(Default, Serialize)]
struct Totals {
    constructions_begun: usize,
    constructions_completed: usize,
    fits: usize,
    center_checks: usize,
    core_overlap_calls: usize,
}
fn construct(
    loaded: &Loaded,
    m: &Manifest,
    c: &Construction,
    diagnostics: &mut FusionBuildDiagnostics,
) -> Result<()> {
    let poses = &loaded.states[&c.state];
    let spectators: Vec<_> = poses
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != c.moving)
        .map(|(_, &p)| p)
        .collect();
    let pool = c.neighbors.map(|i| poses[i]);
    let _mixture = OligomerMixture::build_with_fit_diagnostics(
        &loaded.model,
        &loaded.core,
        &loaded.wall,
        &[IDENTITY],
        &spectators,
        &pool,
        &m.oligomer,
        m.fit_policy,
        diagnostics,
    )?;
    Ok(())
}

fn execute<F>(
    m: &Manifest,
    loaded: &Loaded,
    journal: &mut File,
    totals: &mut Totals,
    mut build: F,
) -> Result<()>
where
    F: FnMut(&Loaded, &Manifest, &Construction, &mut FusionBuildDiagnostics) -> Result<()>,
{
    for (ordinal, c) in m.constructions.iter().enumerate() {
        let spectator_labels: Vec<_> = (0..loaded.states[&c.state].len())
            .filter(|&i| i != c.moving)
            .collect();
        emit(
            journal,
            &json!({"state":"begin","ordinal":ordinal,"construction":c,
            "members":[IDENTITY],"spectator_labels":spectator_labels,"context":m.context,
            "fit_policy":m.fit_policy}),
        )?;
        totals.constructions_begun += 1;
        let mut diagnostics = FusionBuildDiagnostics::default();
        let started = cpu_seconds();
        let result = build(loaded, m, c, &mut diagnostics);
        totals.fits += diagnostics.fit_attempts;
        totals.center_checks += diagnostics.centers_checked;
        totals.core_overlap_calls += diagnostics.core_overlap_calls;
        let counters_valid = totals.fits <= m.caps.fits
            && totals.center_checks <= m.caps.center_checks
            && totals.core_overlap_calls <= m.caps.core_overlap_calls
            && diagnostics.fit_attempts <= m.oligomer.max_candidates
            && diagnostics.centers_checked <= m.oligomer.max_hard_checks;
        let success = result.is_ok() && counters_valid && diagnostics.complete;
        emit(
            journal,
            &json!({"state":if success {"complete"} else {"failed"},"ordinal":ordinal,"construction":c,
            "fit_policy":m.fit_policy,"diagnostics":diagnostics,"spectator_labels":spectator_labels,"totals":totals,
            "cpu_seconds":cpu_seconds()-started,"error":result.as_ref().err().map(|e|format!("{e:#}")),
            "counter_limits_passed":counters_valid}),
        )?;
        result?;
        ensure!(
            counters_valid,
            "constructor diagnostic counts exceeded fixed caps"
        );
        ensure!(
            diagnostics.complete,
            "successful constructor left incomplete diagnostics"
        );
        totals.constructions_completed += 1;
    }
    Ok(())
}

fn run(args: Args) -> Result<()> {
    ensure!(
        args.manifest.is_absolute() && digest(&args.manifest_sha256),
        "absolute immutable manifest/hash required"
    );
    let mut inputs = Inputs::default();
    let raw = inputs.read(&BoundFile {
        path: args.manifest.clone(),
        sha256: args.manifest_sha256.clone(),
    })?;
    let m: Manifest = serde_json::from_slice(&raw)?;
    validate_manifest(&m)?;
    ensure!(
        args.out == m.output && !args.out.exists(),
        "fresh manifest-bound output required"
    );
    ensure!(
        fs::canonicalize(args.out.parent().context("output parent")?)?
            == args.out.parent().unwrap(),
        "output parent must be canonical"
    );
    fs::create_dir(&args.out)?;
    let journal_path = args.out.join("attempts.jsonl");
    let mut journal = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&journal_path)?;
    let mut totals = Totals::default();
    let started = cpu_seconds();
    let result = (|| -> Result<()> {
        emit(
            &mut journal,
            &json!({"state":"started","schema":SCHEMA,"manifest_sha256":args.manifest_sha256,
            "fit_policy":m.fit_policy}),
        )?;
        ensure!(
            hash_bytes(BUNDLE) == m.witness.compiled_source_bundle_sha256
                && hash_bytes(SOURCE) == m.witness.cli_source_sha256,
            "compiled source witness differs"
        );
        inputs.read(&BoundFile {
            path: fs::canonicalize(std::env::current_exe()?)?,
            sha256: m.witness.executable_sha256.clone(),
        })?;
        fs::create_dir(args.out.join("provenance"))?;
        write_new(&args.out.join("provenance/manifest.json"), &raw)?;
        write_new(&args.out.join("provenance/source-bundle.json"), BUNDLE)?;
        let loaded = load(&m, &mut inputs)?;
        emit(
            &mut journal,
            &json!({"state":"inputs_verified","input_sha256":inputs.files,"pose_count":loaded.states.values().next().unwrap().len(),
            "coordinate_frame":m.coordinate_frame,"wall_center":m.wall_center,"wall_radius":loaded.wall.radius,
            "atlas_loading":{"periodic":false,"uniform_weight":0.1,"correlation":0.},"caps":m.caps,
            "fit_policy":m.fit_policy}),
        )?;
        execute(&m, &loaded, &mut journal, &mut totals, construct)?;
        inputs.recheck()?;
        Ok(())
    })();
    if let Err(error) = result {
        let recheck_error = inputs.recheck().err().map(|e| format!("{e:#}"));
        emit(
            &mut journal,
            &json!({"state":"failed","error":format!("{error:#}"),"totals":totals,"input_recheck_error":recheck_error,
            "fit_policy":m.fit_policy}),
        )?;
        save_new(
            &args.out.join("failure.json"),
            &json!({"schema":"singleton-fusion-diagnostic-failure-v1","complete":false,
            "manifest_sha256":args.manifest_sha256,"fit_policy":m.fit_policy,"error":format!("{error:#}"),"totals":totals,
            "input_sha256":inputs.files,"input_recheck_error":recheck_error,"attempts_sha256":hash_file(&journal_path)?,
            "retries":0,"new_pose_draws":0,"new_Poisson_clouds":0}),
        )?;
        return Err(error);
    }
    emit(
        &mut journal,
        &json!({"state":"finished","fit_policy":m.fit_policy,"totals":totals}),
    )?;
    save_new(
        &args.out.join("summary.json"),
        &json!({"schema":"singleton-fusion-diagnostic-v1","complete":true,"passed":true,
        "context":m.context,"manifest_sha256":args.manifest_sha256,"witness":m.witness,"totals":totals,
        "fit_policy":m.fit_policy,
        "input_sha256":inputs.files,"attempts_sha256":hash_file(&journal_path)?,"cpu_seconds":cpu_seconds()-started,
        "new_pose_draws":0,"new_Poisson_clouds":0,"native_queries":0,"contact_graph_queries":0,"density_queries":0,
        "scope":"Passive exact singleton catalogue construction at explicitly bound states; no sampling or scientific mixing conclusion"}),
    )?;
    Ok(())
}
fn main() -> Result<()> {
    run(Args::parse())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Fixture {
        root: PathBuf,
        manifest: Manifest,
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }
    fn bound(path: PathBuf) -> Result<BoundFile> {
        Ok(BoundFile {
            sha256: hash_file(&path)?,
            path,
        })
    }
    fn fixture() -> Result<Fixture> {
        let root = std::env::temp_dir().join(format!(
            "singleton-fusion-manifest-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root)?;
        let pose = |x| Pose {
            position: [x, 0., 0.],
            orientation: IDENTITY.orientation,
        };
        let poses = vec![pose(0.6), pose(1.2), pose(0.)];
        let shape_path = root.join("shape.json");
        save_new(
            &shape_path,
            &json!({"name":"toy","volume":1.,"atoms":[{"center":[0.,0.,0.],"radius":0.1}]}),
        )?;
        let shape = bound(shape_path)?;
        let config_path = root.join("config.json");
        save_new(
            &config_path,
            &json!({"shape":shape.path,"box_lengths":[100.,100.,100.],"boundary":{"kind":"spherical","radius":50.},
            "initial_poses":poses,"seed":1,"depletant_radius":0.1,"reservoir_density":0.01}),
        )?;
        let config = bound(config_path)?;
        let model_path = root.join("model.json");
        let covariance: [[f64; 6]; 6] =
            std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 0.01 } else { 0. }));
        save_new(
            &model_path,
            &json!({"coordinate_convention":"anchor-body-relative","shape_sha256":shape.sha256,
            "angular_length":1.,"weights":[1.],"anchors":[{"position":[0.6,0.,0.],"rotation":[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]}],
            "means":[[0.,0.,0.,0.,0.,0.]],"covariances":[covariance]}),
        )?;
        let manifest = Manifest {
            schema: SCHEMA.into(),
            context: "toy".into(),
            coordinate_frame: "sphere_centered".into(),
            wall_center: [0.; 3],
            output: root.join("output"),
            config: config.clone(),
            model: bound(model_path)?,
            shape,
            members: [0, 1],
            anchor: 2,
            states: vec![NamedState {
                id: "source".into(),
                kind: StateKind::Source,
                file: config,
                poses_pointer: "/initial_poses".into(),
            }],
            constructions: vec![
                Construction {
                    id: "source-0".into(),
                    state: "source".into(),
                    moving: 0,
                    neighbors: [1, 2],
                },
                Construction {
                    id: "source-1".into(),
                    state: "source".into(),
                    moving: 1,
                    neighbors: [0, 2],
                },
            ],
            oligomer: OligomerConfig::default(),
            fit_policy: FusionFitPolicy::EarlyAbort,
            caps: Caps {
                constructions: 2,
                fits: 8192,
                center_checks: 512,
                core_overlap_calls: 1024,
            },
            witness: Witness {
                executable_sha256: hash_file(&std::env::current_exe()?)?,
                compiled_source_bundle_sha256: hash_bytes(BUNDLE),
                cli_source_sha256: hash_bytes(SOURCE),
            },
        };
        Ok(Fixture { root, manifest })
    }
    #[test]
    fn singleton_fusion_fit_policy_defaults_and_strict_deserialization() -> Result<()> {
        let fixture = fixture()?;
        let mut legacy = serde_json::to_value(&fixture.manifest)?;
        legacy.as_object_mut().unwrap().remove("fit_policy");
        let omitted: Manifest = serde_json::from_value(legacy.clone())?;
        assert_eq!(omitted.fit_policy, FusionFitPolicy::EarlyAbort);
        validate_manifest(&omitted)?;
        for (name, expected) in [
            ("early_abort", FusionFitPolicy::EarlyAbort),
            ("full_iterations", FusionFitPolicy::FullIterations),
        ] {
            let mut explicit = legacy.clone();
            explicit["fit_policy"] = json!(name);
            let parsed: Manifest = serde_json::from_value(explicit.clone())?;
            assert_eq!(parsed.fit_policy, expected);
            validate_manifest(&parsed)?;
            assert_eq!(serde_json::to_value(parsed)?, explicit);
        }
        for invalid in [
            json!(null),
            json!("full"),
            json!("EarlyAbort"),
            json!(1),
            json!({}),
        ] {
            let mut value = legacy.clone();
            value["fit_policy"] = invalid;
            assert!(serde_json::from_value::<Manifest>(value).is_err());
        }
        Ok(())
    }
    #[test]
    fn singleton_fusion_manifest_rejects_frame_neighbor_inventory_and_budget_drift() -> Result<()> {
        let fixture = fixture()?;
        let m = &fixture.manifest;
        validate_manifest(m)?;
        let mut bad = m.clone();
        bad.coordinate_frame = "laboratory".into();
        assert!(validate_manifest(&bad).is_err());
        let mut bad = m.clone();
        bad.wall_center[0] = 1.;
        assert!(validate_manifest(&bad).is_err());
        let mut bad = m.clone();
        bad.constructions[0].neighbors.swap(0, 1);
        assert!(validate_manifest(&bad).is_err());
        let mut bad = m.clone();
        bad.constructions[1] = bad.constructions[0].clone();
        assert!(validate_manifest(&bad).is_err());
        let mut bad = m.clone();
        bad.caps.fits -= 1;
        assert!(validate_manifest(&bad).is_err());
        let mut value = serde_json::to_value(m)?;
        value["select_source"] = json!(true);
        assert!(serde_json::from_value::<Manifest>(value).is_err());
        Ok(())
    }
    #[test]
    fn singleton_fusion_bound_state_hash_and_fixed_spectators_are_checked() -> Result<()> {
        let fixture = fixture()?;
        let mut inputs = Inputs::default();
        let loaded = load(&fixture.manifest, &mut inputs)?;
        assert_eq!(loaded.states["source"].len(), 3);
        let mut m = fixture.manifest.clone();
        let mut value: Value = serde_json::from_slice(&fs::read(&m.config.path)?)?;
        value["initial_poses"][2]["position"][0] = json!(0.2);
        let prepared = fixture.root.join("prepared.json");
        save_new(&prepared, &value)?;
        m.states.push(NamedState {
            id: "prepared".into(),
            kind: StateKind::Prepared,
            file: bound(prepared.clone())?,
            poses_pointer: "/initial_poses".into(),
        });
        for slot in 0..2 {
            m.constructions.push(Construction {
                id: format!("prepared-{slot}"),
                state: "prepared".into(),
                moving: slot,
                neighbors: [1 - slot, 2],
            });
        }
        m.caps = Caps {
            constructions: 4,
            fits: 16384,
            center_checks: 1024,
            core_overlap_calls: 2048,
        };
        assert!(
            format!("{:#}", load(&m, &mut Inputs::default()).err().unwrap())
                .contains("fixed spectator")
        );
        fs::write(prepared, b"{}")?;
        assert!(
            format!("{:#}", load(&m, &mut Inputs::default()).err().unwrap())
                .contains("changed input")
        );
        Ok(())
    }
    #[test]
    fn singleton_fusion_failure_keeps_partial_diagnostics_without_retry() -> Result<()> {
        let fixture = fixture()?;
        let loaded = load(&fixture.manifest, &mut Inputs::default())?;
        let path = fixture.root.join("journal.jsonl");
        let mut journal = File::create(&path)?;
        let mut totals = Totals::default();
        let result = execute(
            &fixture.manifest,
            &loaded,
            &mut journal,
            &mut totals,
            |_, _, _, d| {
                d.fit_attempts = 1;
                d.centers_checked = 1;
                d.core_overlap_calls = 2;
                anyhow::bail!("synthetic construction failure")
            },
        );
        assert!(result.is_err());
        assert_eq!(totals.constructions_begun, 1);
        assert_eq!(totals.constructions_completed, 0);
        let rows: Vec<Value> = fs::read_to_string(path)?
            .lines()
            .map(serde_json::from_str)
            .collect::<std::result::Result<_, _>>()?;
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["state"], "begin");
        assert_eq!(rows[0]["fit_policy"], "early_abort");
        assert_eq!(rows[1]["state"], "failed");
        assert_eq!(rows[1]["fit_policy"], "early_abort");
        assert_eq!(rows[1]["diagnostics"]["core_overlap_calls"], 2);
        assert_eq!(rows[1]["spectator_labels"], json!([1, 2]));
        Ok(())
    }
    #[test]
    fn singleton_fusion_passive_cli_completes_only_declared_synthetic_constructions() -> Result<()>
    {
        for policy in [None, Some("early_abort"), Some("full_iterations")] {
            let fixture = fixture()?;
            let path = fixture.root.join("manifest.json");
            let mut value = serde_json::to_value(&fixture.manifest)?;
            match policy {
                Some(name) => value["fit_policy"] = json!(name),
                None => {
                    value.as_object_mut().unwrap().remove("fit_policy");
                }
            }
            save_new(&path, &value)?;
            let manifest_bytes = fs::read(&path)?;
            run(Args {
                manifest: path.clone(),
                manifest_sha256: hash_file(&path)?,
                out: fixture.manifest.output.clone(),
            })?;
            let summary: Value =
                serde_json::from_slice(&fs::read(fixture.manifest.output.join("summary.json"))?)?;
            let effective = policy.unwrap_or("early_abort");
            assert_eq!(summary["fit_policy"], effective);
            assert_eq!(summary["totals"]["constructions_completed"], 2);
            assert_eq!(summary["new_pose_draws"], 0);
            assert_eq!(summary["native_queries"], 0);
            assert_eq!(summary["density_queries"], 0);
            assert_eq!(fs::read(&path)?, manifest_bytes);
            assert_eq!(
                fs::read(fixture.manifest.output.join("provenance/manifest.json"))?,
                manifest_bytes
            );
            let rows: Vec<Value> =
                fs::read_to_string(fixture.manifest.output.join("attempts.jsonl"))?
                    .lines()
                    .map(serde_json::from_str)
                    .collect::<std::result::Result<_, _>>()?;
            let complete: Vec<_> = rows
                .iter()
                .filter(|row| row["state"] == "complete")
                .collect();
            assert_eq!(complete.len(), 2);
            for row in complete {
                assert_eq!(row["fit_policy"], effective);
                let fits = row["diagnostics"]["fit_records"].as_array().unwrap();
                assert!(!fits.is_empty());
                assert_eq!(
                    fits.len() as u64,
                    row["diagnostics"]["fit_attempts"].as_u64().unwrap()
                );
            }
            assert!(
                run(Args {
                    manifest: path.clone(),
                    manifest_sha256: hash_file(&path)?,
                    out: fixture.manifest.output.clone()
                })
                .is_err()
            );
        }
        Ok(())
    }
}
