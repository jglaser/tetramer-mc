//! Prepared, conditional two-mobile-body trajectories. Frozen spectators, NOT
//! finite-system assembly. No adaptive selection, basin filtering or retries
//! after an outer null/failure. Every elementary attempt and retained block is
//! journaled; independent named streams give deterministic graceful continuation.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};
use tetramer_mc::{
    auxiliary_overlap_threshold::AuxiliaryOverlapThreshold,
    bounded_singleton_path::{Budget, Limits},
    defensive_dimer_proposal::DefensiveDimerProposal,
    docking::{DockingMethod, DockingProposal},
    evolving_dimer::FixedLabelUpdates,
    factorized_dimer::{FactorizedDimerCaps, FactorizedDimerOrder, FactorizedDimerProposal},
    flexible_surrogate_chain::FlexibleSurrogateKernel,
    geometry::{Shape, SphereTree},
    math::{Pose, norm, sub},
    oligomer_proposal::OligomerConfig,
    proposal::FrozenRelativePoseProposal,
    rigid_surrogate_chain::{RigidSurrogateConfig, RigidSurrogateKernel},
    simulation::{cpu_seconds, hash_bytes, hash_file, save},
    spherical::{Container, validate_state},
};
const BUNDLE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/source-bundle.json"));
const SOURCE: &[u8] = include_bytes!("evolving_dimer_benchmark.rs");
#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    binding: PathBuf,
    #[arg(long)]
    mode: String,
    #[arg(long)]
    job: Option<usize>,
    /// Graceful deterministic continuation control; changes no allocation.
    #[arg(long)]
    stop_after_block: Option<usize>,
    #[arg(long)]
    resume: Option<PathBuf>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
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
    fn json(&self) -> Result<Value> {
        Ok(serde_json::from_slice(&self.read()?)?)
    }
    fn make(path: &Path) -> Result<Self> {
        Ok(Self {
            path: fs::canonicalize(path)?,
            sha256: hash_file(path)?,
        })
    }
}
fn bound(value: &Value) -> Result<BoundFile> {
    Ok(serde_json::from_value(value.clone())?)
}
fn last_journal_row(path: &Path) -> Result<Value> {
    let mut file = File::open(path)?;
    let size = file.metadata()?.len();
    ensure!(size > 0, "empty checkpoint journal");
    let start = size.saturating_sub(65536);
    file.seek(SeekFrom::Start(start))?;
    let mut tail = Vec::new();
    file.read_to_end(&mut tail)?;
    ensure!(
        tail.last() == Some(&b'\n'),
        "incomplete checkpoint journal line"
    );
    tail.pop();
    let begin = tail.iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1);
    ensure!(
        start == 0 || begin > 0,
        "checkpoint terminal row exceeds audit buffer"
    );
    Ok(serde_json::from_slice(&tail[begin..])?)
}
fn usize_at(v: &Value, k: &str) -> Result<usize> {
    usize::try_from(
        v[k].as_u64()
            .with_context(|| format!("missing integer {k}"))?,
    )
    .map_err(Into::into)
}
fn num(v: &Value, k: &str) -> Result<f64> {
    v[k].as_f64().with_context(|| format!("missing number {k}"))
}
fn rng(master: u64, context: usize, init: &str, stream: usize, block: usize, role: &str) -> StdRng {
    let digest = Sha256::digest(
        format!("evolving-dimer-v1/{master}/{context}/{init}/{stream}/{block}/{role}").as_bytes(),
    );
    StdRng::from_seed(digest.into())
}
/// Retain m4's common random numbers. Extra root-threshold and root-order
/// choices get separate roles; every existing arm keeps its original bytes.
fn dimer_role(arm: &str, role: &str) -> String {
    let prefix = if (arm == "root_m4" && role != "root_threshold")
        || (arm == "two_root_m4" && role != "root_order")
    {
        "m4"
    } else {
        arm
    };
    format!("{prefix}/{role}")
}
fn two_root_policy(plan: &Value, arm: &str) -> Result<()> {
    if arm != "two_root_m4" {
        ensure!(
            plan.get("two_root_policy").is_none(),
            "two-root policy on another arm"
        );
    } else {
        ensure!(
            plan["two_root_policy"]
                == json!({
                    "schema":"two-root-factorized-m4-policy-v1",
                    "root_probabilities":[0.5,0.5],"selection":"once_per_global_slot",
                    "internal_threshold_m":4
                }),
            "two-root policy differs from the frozen control"
        );
    }
    Ok(())
}
fn root_order(master: u64, context: usize, init: &str, stream: usize, block: usize) -> usize {
    usize::from(
        rng(
            master,
            context,
            init,
            stream,
            block,
            &dimer_role("two_root_m4", "root_order"),
        )
        .random::<bool>(),
    )
}
fn rooted_engine<'a>(
    canonical: &FixedLabelUpdates<'a>,
    root_slot: usize,
    state_len: usize,
) -> Result<FixedLabelUpdates<'a>> {
    ensure!(root_slot < 2, "invalid root slot");
    ensure!(
        canonical.members[0] != canonical.members[1]
            && canonical
                .members
                .iter()
                .all(|&i| i < state_len && i != canonical.anchor)
            && canonical.anchor < state_len,
        "invalid canonical member/anchor labels"
    );
    Ok(FixedLabelUpdates {
        core: canonical.core,
        exclusion: canonical.exclusion,
        wall: canonical.wall,
        radius: canonical.radius,
        members: [
            canonical.members[root_slot],
            canonical.members[1 - root_slot],
        ],
        anchor: canonical.anchor,
        rd: canonical.rd,
        activity: canonical.activity,
        lambda: canonical.lambda,
        envelope: canonical.envelope,
    })
}
/// A state-independent mixture of the two unchanged fixed-label kernels.
/// The selected order is retained across every capped trial and the one bath
/// decision. Library records keep selected-root order; explicit canonical
/// projections and the outer retained state prevent a label-order ambiguity.
fn two_root_dimer(
    canonical: &FixedLabelUpdates<'_>,
    root_slot: usize,
    state: &mut [Pose],
    proposal: &FactorizedDimerProposal<'_>,
    guide: &AuxiliaryOverlapThreshold<'_>,
    proposal_rng: &mut StdRng,
    threshold_rng: &mut StdRng,
    bath_rng: &mut StdRng,
    accept_rng: &mut StdRng,
    budget: &mut Budget,
    record: &mut Value,
) -> Result<()> {
    let mut selected_members = Value::Null;
    let mut canonical_old = Value::Null;
    let result = (|| -> Result<()> {
        let selected = rooted_engine(canonical, root_slot, state.len())?;
        selected_members = json!(selected.members);
        canonical_old = json!(canonical.selected(state));
        selected.dimer(
            state,
            proposal,
            Some(guide),
            proposal_rng,
            threshold_rng,
            bath_rng,
            accept_rng,
            budget,
            record,
        )
    })();
    if record.is_null() {
        *record =
            json!({"kind":"factorized_dimer","status":"failed_before_proposal","accepted":false});
    }
    record["canonical_members"] = json!(canonical.members);
    record["selected_members"] = selected_members;
    record["root_slot"] = json!(root_slot);
    record["root_order_probability"] = json!(0.5);
    record["root_order_log_reverse_forward"] = json!(0.);
    record["root_order_rng_role"] = json!(dimer_role("two_root_m4", "root_order"));
    record["canonical_old"] = canonical_old;
    if let Some(poses) = record.get("proposed").and_then(Value::as_array) {
        // This array is produced by the unchanged successful dimer kernel.
        let ordered = json!([poses[root_slot], poses[1 - root_slot]]);
        record["canonical_proposed"] = ordered;
    }
    result
}
fn two_root_contract(engine: &FixedLabelUpdates<'_>, bank: &Value) -> Value {
    json!({"schema":"evolving-dimer-two-root-m4-v1","canonical_members":engine.members,
        "root_orders":[engine.members,[engine.members[1],engine.members[0]]],
        "root_probabilities":[0.5,0.5],"anchor_label":engine.anchor,
        "selection":"once_per_global_slot_retained_through_all_retries",
        "root_order_rng_role":dimer_role("two_root_m4","root_order"),
        "selection_log_reverse_forward":0.,"cloud":bank,
        "internal":{"m":4,"point_frame":"selected_mobile_root_body",
            "threshold_rng_role":dimer_role("two_root_m4","threshold")},
        "root_guidance":false,"matched_control_arm":"m4",
        "proposal_rng_role":dimer_role("two_root_m4","proposal"),
        "bath_rng_role":dimer_role("two_root_m4","bath"),
        "accept_rng_role":dimer_role("two_root_m4","accept"),
        "local_schedule":"canonical members [0,1,0,1]; shared unchanged local RNG roles",
        "retained_and_checkpoint_order":"canonical_members",
        "proposal_record_order":"selected_members","physical_decisions_per_candidate":1})
}
fn singleton_arm(arm: &str) -> bool {
    matches!(
        arm,
        "singleton_two_neighbor" | "singleton_two_neighbor_unfused"
    )
}
fn surrogate_arm(arm: &str) -> bool {
    matches!(
        arm,
        "rigid_surrogate_1" | "rigid_surrogate_8" | "rigid_surrogate_flat8"
    )
}
fn flexible_surrogate_arm(arm: &str) -> bool {
    matches!(arm, "flexible_m1" | "flexible_m8" | "flexible_flat8")
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "source", rename_all = "snake_case", deny_unknown_fields)]
enum SurrogateScales {
    Local {},
    FrozenOverride {
        #[serde(rename = "translation_std_A")]
        translation_std_a: f64,
        rotation_std_degrees: f64,
    },
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SurrogatePolicy {
    schema: String,
    proposal_scales: SurrogateScales,
}
impl SurrogatePolicy {
    fn config(&self, plan: &Value, arm: &str) -> Result<RigidSurrogateConfig> {
        let (inner_steps, guidance_strength) = match arm {
            "rigid_surrogate_1" | "flexible_m1" => (1, 1.),
            "rigid_surrogate_8" | "flexible_m8" => (8, 1.),
            "rigid_surrogate_flat8" | "flexible_flat8" => (8, 0.),
            _ => anyhow::bail!("surrogate policy on another arm"),
        };
        let (translation_std, rotation_std_degrees) = match self.proposal_scales {
            SurrogateScales::Local {} => (
                num(&plan["local"], "translation_std_A")?,
                num(&plan["local"], "rotation_std_degrees")?,
            ),
            SurrogateScales::FrozenOverride {
                translation_std_a,
                rotation_std_degrees,
            } => (translation_std_a, rotation_std_degrees),
        };
        let config = RigidSurrogateConfig {
            inner_steps,
            translation_std,
            rotation_std_degrees,
            guidance_strength,
        };
        config.validate()?;
        Ok(config)
    }
}
/// One frozen scale policy is shared by all three controls. The arm name alone
/// fixes the horizon and strength; neither is tuned from a retained trajectory.
fn surrogate_policy(plan: &Value, arm: &str) -> Result<Option<SurrogatePolicy>> {
    if !surrogate_arm(arm) {
        ensure!(
            plan.get("surrogate_policy").is_none(),
            "surrogate policy on another arm"
        );
        return Ok(None);
    }
    let policy: SurrogatePolicy = serde_json::from_value(
        plan.get("surrogate_policy")
            .context("surrogate arm requires explicit surrogate_policy")?
            .clone(),
    )?;
    ensure!(
        policy.schema == "rigid-surrogate-policy-v1",
        "unknown surrogate policy"
    );
    policy.config(plan, arm)?;
    Ok(Some(policy))
}
fn surrogate_role(role: &str) -> String {
    format!("rigid_surrogate/{role}")
}
fn flexible_surrogate_policy(plan: &Value, arm: &str) -> Result<Option<SurrogatePolicy>> {
    if !flexible_surrogate_arm(arm) {
        ensure!(
            plan.get("flexible_surrogate_policy").is_none(),
            "flexible surrogate policy on another arm"
        );
        return Ok(None);
    }
    let policy: SurrogatePolicy = serde_json::from_value(
        plan.get("flexible_surrogate_policy")
            .context("flexible arm requires explicit flexible_surrogate_policy")?
            .clone(),
    )?;
    ensure!(
        policy.schema == "flexible-surrogate-policy-v1",
        "unknown flexible surrogate policy"
    );
    policy.config(plan, arm)?;
    Ok(Some(policy))
}
fn flexible_surrogate_role(role: &str) -> String {
    format!("flexible_surrogate/{role}")
}
fn rows_per_block(arm: &str) -> u64 {
    if arm == "local" {
        5
    } else if surrogate_arm(arm) || flexible_surrogate_arm(arm) {
        7 // Four locals, durable begun/outcome rows, retained block.
    } else {
        6
    }
}
fn surrogate_point_volume(meta: &Value, expected_count: usize, raw: &[u8]) -> Result<f64> {
    let count = usize_at(meta, "raw_count")?;
    ensure!(
        count > 0 && count == expected_count,
        "surrogate raw cloud count differs"
    );
    ensure!(
        count.checked_mul(24) == Some(raw.len()),
        "surrogate raw cloud length differs"
    );
    let low: [f64; 3] = serde_json::from_value(meta["low"].clone())?;
    let high: [f64; 3] = serde_json::from_value(meta["high"].clone())?;
    ensure!(
        (0..3).all(|k| low[k].is_finite() && high[k].is_finite() && high[k] > low[k]),
        "invalid surrogate quadrature box"
    );
    let indices: Vec<usize> = serde_json::from_value(meta["kept_indices"].clone())?;
    ensure!(
        indices.iter().all(|&i| i < count) && indices.windows(2).all(|w| w[0] < w[1]),
        "invalid frozen surrogate kept indices"
    );
    for bytes in raw.chunks_exact(8) {
        let u = f64::from_le_bytes(bytes.try_into().unwrap());
        ensure!(
            u.is_finite() && (0. ..1.).contains(&u),
            "invalid raw surrogate cloud point"
        );
    }
    let point_volume = (0..3).map(|k| high[k] - low[k]).product::<f64>() / count as f64;
    ensure!(
        point_volume.is_finite() && point_volume > 0.,
        "invalid surrogate point volume"
    );
    Ok(point_volume)
}
fn surrogate_contract(
    engine: &FixedLabelUpdates<'_>,
    policy: &SurrogatePolicy,
    config: RigidSurrogateConfig,
    bank: &Value,
    meta: &Value,
    point_count: usize,
    point_volume: f64,
) -> Value {
    json!({"schema":"evolving-dimer-rigid-surrogate-v1","policy":policy,"effective_config":config,
        "members":engine.members,"handle":engine.members[0],"fixed_spectators":"all other labels",
        "cloud":bank,"cloud_reuse":"identical frozen body-frame points for both members; no extra draws",
        "raw_count":meta["raw_count"],"points_per_body":point_count,"point_volume":point_volume,
        "point_weight":"raw_box_volume/raw_count","wall_center":[0.,0.,0.],
        "proposal_rng_role":surrogate_role("proposal"),
        "inner_accept_rng_role":surrogate_role("inner_accept"),
        "bath_rng_role":surrogate_role("bath"),"accept_rng_role":surrogate_role("accept"),
        "local_schedule":"canonical members [0,1,0,1]; shared unchanged local RNG roles",
        "collective_slots_per_block":1,"physical_decisions_per_nonidentity_endpoint":1,
        "identity_endpoint":"retained self-loop without bath",
        "inner_rejections":"consume a step and retain current state",
        "outer_correction":"S(old)-S(new)","journal_rows_per_block":7})
}
fn flexible_surrogate_contract(
    engine: &FixedLabelUpdates<'_>,
    policy: &SurrogatePolicy,
    config: RigidSurrogateConfig,
    bank: &Value,
    meta: &Value,
    point_count: usize,
    point_volume: f64,
) -> Value {
    json!({"schema":"evolving-dimer-flexible-surrogate-v1","policy":policy,"effective_config":config,
        "members":engine.members,"fixed_spectators":"all other labels",
        "selection_probabilities":[0.5,0.5],
        "inner_selection":"independent fair random scan each step",
        "physical_path":"fair-order two-singleton path; intermediate is not hard-filtered",
        "cloud":bank,"cloud_reuse":"identical frozen body-frame points for both members; no extra draws",
        "raw_count":meta["raw_count"],"points_per_body":point_count,"point_volume":point_volume,
        "point_weight":"raw_box_volume/raw_count","wall_center":[0.,0.,0.],
        "proposal_rng_role":flexible_surrogate_role("proposal"),
        "inner_accept_rng_role":flexible_surrogate_role("inner_accept"),
        "bath_rng_role":flexible_surrogate_role("bath"),"accept_rng_role":flexible_surrogate_role("accept"),
        "local_schedule":"canonical members [0,1,0,1]; shared unchanged local RNG roles",
        "collective_slots_per_block":1,"physical_decisions_per_nonidentity_endpoint":1,
        "identity_endpoint":"retained self-loop without bath",
        "inner_rejections":"consume a step and retain current state",
        "outer_correction":"S(old)-S(new)","journal_rows_per_block":7})
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SingletonPolicy {
    schema: String,
    uniform_half_width: f64,
    uniform_probability: f64,
    trial_cap: usize,
    member_schedule: String,
    oligomer: OligomerConfig,
}
impl SingletonPolicy {
    fn oligomer_for(&self, arm: &str) -> OligomerConfig {
        let mut config = self.oligomer.clone();
        if arm == "singleton_two_neighbor_unfused" {
            config.multi_contact_mass = 0.;
        }
        config
    }
}
/// Existing plans omit this field entirely. The two new arms declare the
/// frozen common policy; only their deterministic fused mass differs.
fn singleton_policy(plan: &Value, arm: &str) -> Result<Option<SingletonPolicy>> {
    if !singleton_arm(arm) {
        ensure!(
            plan.get("singleton_policy").is_none(),
            "singleton policy on an existing arm"
        );
        return Ok(None);
    }
    let policy: SingletonPolicy = serde_json::from_value(
        plan.get("singleton_policy")
            .context("new singleton arm requires singleton_policy")?
            .clone(),
    )?;
    ensure!(
        policy.schema == "two-neighbor-singleton-policy-v1"
            && policy.uniform_half_width == 160.
            && policy.uniform_probability == 0.5
            && policy.trial_cap == 32
            && policy.member_schedule == "alternating_0_first"
            && policy.oligomer == OligomerConfig::default(),
        "singleton policy differs from the frozen matched control"
    );
    Ok(Some(policy))
}
fn extra_role(arm: &str, role: &str) -> String {
    if singleton_arm(arm) {
        format!("singleton_two_neighbor/{role}")
    } else {
        dimer_role(arm, role)
    }
}
fn singleton_contract(
    engine: &FixedLabelUpdates<'_>,
    policy: &SingletonPolicy,
    arm: &str,
) -> Value {
    json!({
        "schema":"evolving-dimer-two-neighbor-singleton-v1",
        "policy":policy,"effective_oligomer":policy.oligomer_for(arm),
        "member_labels":engine.members,"anchor_label":engine.anchor,
        "member_slot_schedule":"(block-1)%2","neighbors":"[other_mobile,fixed_anchor]",
        "uniform_frame":"fixed_anchor_body","catalogue_rebuild":"every elementary attempt",
        "proposal_rng_role":extra_role(arm,"proposal"),
        "bath_rng_role":extra_role(arm,"bath"),"accept_rng_role":extra_role(arm,"accept"),
        "local_rng_roles":"unchanged shared local/{attempt}/{proposal,bath,accept}",
        "physical_decisions_per_candidate":1,"guidance_cloud_used":false
    })
}
fn root_guidance_contract(engine: &FixedLabelUpdates<'_>, bank: &Value) -> Value {
    json!({
        "schema":"evolving-dimer-root-guidance-v1",
        "cloud":bank,
        "cloud_reuse":"identical frozen body-frame points; no additional cloud draws",
        "root":{"m":4,"point_frame":"fixed_anchor_body","anchor_label":engine.anchor,
            "mobile_label":engine.members[0],"threshold_rng_role":dimer_role("root_m4","root_threshold")},
        "internal":{"m":4,"point_frame":"mobile_root_body","root_label":engine.members[0],
            "child_label":engine.members[1],"threshold_rng_role":dimer_role("root_m4","threshold")},
        "matched_control_arm":"m4",
        "proposal_rng_role":dimer_role("root_m4","proposal"),
        "bath_rng_role":dimer_role("root_m4","bath"),
        "accept_rng_role":dimer_role("root_m4","accept"),
        "local_rng_roles":"unchanged shared local/{attempt}/{proposal,bath,accept}",
        "physical_decisions_per_candidate":1
    })
}
fn far_enough(old: [Pose; 2], new: [Pose; 2], distance: f64, angle: f64) -> bool {
    (0..2).any(|i| {
        let a = old[i].orientation;
        let b = new[i].orientation;
        let dot = a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        norm(sub(old[i].position, new[i].position)) >= distance
            || 2. * (dot / (na * nb)).abs().min(1.).acos().to_degrees() >= angle
    })
}
struct Journal {
    out: BufWriter<File>,
    digest: Sha256,
    bytes: u64,
    rows: u64,
}
impl Journal {
    fn new(path: &Path) -> Result<Self> {
        Ok(Self {
            out: BufWriter::new(OpenOptions::new().write(true).create_new(true).open(path)?),
            digest: Sha256::new(),
            bytes: 0,
            rows: 0,
        })
    }
    fn resume(path: &Path, bytes: u64, rows: u64, sha: &str) -> Result<Self> {
        let mut file = File::open(path)?;
        ensure!(
            file.metadata()?.len() == bytes,
            "journal has an uncheckpointed tail; preserve and audit it before recovery"
        );
        let mut digest = Sha256::new();
        let mut buffer = [0; 65536];
        loop {
            let n = file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            digest.update(&buffer[..n]);
        }
        ensure!(
            format!("{:x}", digest.clone().finalize()) == sha,
            "checkpoint journal hash mismatch"
        );
        Ok(Self {
            out: BufWriter::new(OpenOptions::new().append(true).open(path)?),
            digest,
            bytes,
            rows,
        })
    }
    fn line(&mut self, value: &Value) -> Result<()> {
        let mut bytes = serde_json::to_vec(value)?;
        bytes.push(b'\n');
        self.out.write_all(&bytes)?;
        self.out.flush()?;
        self.digest.update(&bytes);
        self.bytes += bytes.len() as u64;
        self.rows += 1;
        Ok(())
    }
    fn durable_line(&mut self, value: &Value) -> Result<()> {
        self.line(value)?;
        self.out.get_ref().sync_data()?;
        Ok(())
    }
    fn sha(&self) -> String {
        format!("{:x}", self.digest.clone().finalize())
    }
}
struct Geometry {
    core: SphereTree,
    exclusion: SphereTree,
    wall: Container,
    state: Vec<Pose>,
    model: DockingProposal,
    load_cpu_seconds: f64,
}
impl Geometry {
    fn load(plan: &Value) -> Result<Self> {
        let started = cpu_seconds();
        let source = bound(&plan["source_config"])?.json()?;
        let frame = bound(&plan["source_frame"])?.json()?;
        ensure!(
            source["initial_poses"] == frame["poses"],
            "source/frame state mismatch"
        );
        let state: Vec<Pose> = serde_json::from_value(source["initial_poses"].clone())?;
        let shape_file = bound(&plan["shape"])?;
        let shape: Shape = serde_json::from_slice(&shape_file.read()?)?;
        let core = SphereTree::new(shape.clone())?;
        let mut inflated = shape;
        for atom in &mut inflated.atoms {
            atom.radius += num(&plan["physical"], "depletant_radius")?;
        }
        let exclusion = SphereTree::new(inflated)?;
        let radius = num(&plan["physical"], "wall_radius")?;
        let wall = Container::new(radius, &core)?;
        validate_state(&core, &wall, &state)?;
        let bytes = bound(&plan["atlas"])?.read()?;
        let model = DockingProposal::new(
            FrozenRelativePoseProposal::from_json_str_open(
                std::str::from_utf8(&bytes)?,
                [2. * (radius + core.bound); 3],
                0.1,
                &shape_file.sha256,
            )?,
            DockingMethod::PosteriorInvolution,
            0.,
            [0.; 3],
        )?;
        Ok(Self {
            core,
            exclusion,
            wall,
            state,
            model,
            load_cpu_seconds: cpu_seconds() - started,
        })
    }
    fn engine(&self, plan: &Value, ci: usize) -> Result<FixedLabelUpdates<'_>> {
        let case = &plan["contexts"][ci];
        let physical = &plan["physical"];
        let activity = num(physical, "activity")?;
        Ok(FixedLabelUpdates {
            core: &self.core,
            exclusion: &self.exclusion,
            wall: &self.wall,
            radius: num(physical, "wall_radius")?,
            members: [usize_at(case, "root")?, usize_at(case, "child")?],
            anchor: usize_at(case, "anchor")?,
            rd: num(physical, "depletant_radius")?,
            activity,
            lambda: if activity == 0. {
                1.
            } else {
                activity * num(physical, "lambda_ratio")?
            },
            envelope: serde_json::from_value(plan["envelope"].clone())?,
        })
    }
    fn proposal(&self, plan: &Value) -> Result<FactorizedDimerProposal<'_>> {
        let f = &plan["factorized"];
        ensure!(f["order"] == "root_first", "unsupported stage order");
        Ok(FactorizedDimerProposal::new(
            DefensiveDimerProposal::new(
                &self.model,
                num(f, "uniform_half_width")?,
                num(f, "uniform_probability")?,
            )?,
            FactorizedDimerCaps {
                root: usize_at(f, "root_cap")?,
                internal: usize_at(f, "internal_cap")?,
                joint: usize_at(f, "joint_cap")?,
            },
            FactorizedDimerOrder::RootFirst,
        ))
    }
}
fn validate_binding(args: &Args, plan: &Value) -> Result<Value> {
    let binding: Value = serde_json::from_slice(&fs::read(&args.binding)?)?;
    ensure!(
        binding["config_sha256"] == hash_file(&args.config)?,
        "config binding differs"
    );
    let pre = if args.mode == "run" {
        let file = bound(&binding["prelaunch_binding"])?;
        file.json()?
    } else {
        binding.clone()
    };
    ensure!(
        pre["config_sha256"] == hash_file(&args.config)?
            && pre["protocol_sha256"] == bound(&plan["protocol"])?.sha256
            && pre["example_source_sha256"] == hash_bytes(SOURCE)
            && pre["compiled_source_bundle_sha256"] == hash_bytes(BUNDLE)
            && pre["executable_sha256"] == hash_file(&std::env::current_exe()?)?,
        "execution closure differs"
    );
    for key in [
        "protocol",
        "scientific_allocation",
        "source_config",
        "source_frame",
        "shape",
        "panel",
        "atlas",
        "patch_map",
    ] {
        bound(&plan[key])?.read()?;
    }
    for key in [
        "reference_config",
        "source_freeze_manifest",
        "original_atlas",
    ] {
        if !plan[key].is_null() {
            bound(&plan[key])?.read()?;
        }
    }
    ensure!(
        plan["schema"] == "evolving-dimer-benchmark-v1",
        "unexpected plan schema"
    );
    ensure!(
        plan["allocation"]["local_member_order"] == json!([0, 1, 0, 1]),
        "local schedule differs"
    );
    ensure!(
        plan.get("singleton_policy").is_none() || args.mode == "run",
        "singleton arms reuse frozen preparation; new preparation is forbidden"
    );
    ensure!(
        plan.get("surrogate_policy").is_none() || args.mode == "run",
        "surrogate arms reuse frozen preparation; new preparation is forbidden"
    );
    ensure!(
        plan.get("flexible_surrogate_policy").is_none() || args.mode == "run",
        "flexible arms reuse frozen preparation; new preparation is forbidden"
    );
    for job in plan["jobs"].as_array().context("jobs")? {
        let arm = job["arm"].as_str().context("arm")?;
        if singleton_policy(plan, arm)?.is_some() {
            ensure!(
                args.mode == "run",
                "singleton arms reuse frozen preparation; new preparation is forbidden"
            );
        }
        if surrogate_policy(plan, arm)?.is_some() {
            ensure!(
                args.mode == "run",
                "surrogate arms reuse frozen preparation; new preparation is forbidden"
            );
        }
        if flexible_surrogate_policy(plan, arm)?.is_some() {
            ensure!(
                args.mode == "run",
                "flexible arms reuse frozen preparation; new preparation is forbidden"
            );
        }
    }
    Ok(binding)
}
fn prepare(args: &Args, plan: &Value, geometry: &Geometry) -> Result<()> {
    let output = PathBuf::from(
        plan["preparation_output"]
            .as_str()
            .context("preparation_output")?,
    );
    fs::create_dir(&output)?;
    let result = prepare_inner(args, plan, geometry, &output);
    if let Err(error) = &result {
        save(
            &output.join("failure.json"),
            &json!({"complete":false,"error":format!("{error:#}")}),
        )?;
    }
    result
}
fn prepare_inner(args: &Args, plan: &Value, geometry: &Geometry, output: &Path) -> Result<()> {
    let preparation_started = cpu_seconds();
    let master = plan["master_seed"].as_u64().context("master_seed")?;
    let contexts = plan["contexts"].as_array().context("contexts")?;
    let streams = usize_at(&plan["allocation"], "streams")?;
    let mut files = Vec::<BoundFile>::new();
    let mut banks = vec![];
    let mut starts = vec![];
    let bounds = geometry.exclusion.bounds(0.);
    let raw_count = usize_at(&plan["cloud"], "raw_count")?;
    // Complete independent bank generated BEFORE inspecting any prepared start.
    for ci in 0..contexts.len() {
        for init in ["source", "proposal_prepared"] {
            for stream in 0..streams {
                let before = cpu_seconds();
                let mut cloud_rng = rng(master, ci, init, stream, 0, "cloud");
                let mut raw = Vec::with_capacity(raw_count * 24);
                let mut kept = vec![];
                for index in 0..raw_count {
                    let u: [f64; 3] = std::array::from_fn(|_| cloud_rng.random());
                    raw.extend(u.into_iter().flat_map(f64::to_le_bytes));
                    let p = std::array::from_fn(|k| {
                        bounds.lo[k] + (bounds.hi[k] - bounds.lo[k]) * u[k]
                    });
                    if geometry.exclusion.contains(p, 0.) {
                        kept.push(index);
                    }
                }
                let raw_path = output.join(format!("cloud-{ci}-{init}-{stream}.bin"));
                fs::write(&raw_path, raw)?;
                let meta_path = output.join(format!("cloud-{ci}-{init}-{stream}.json"));
                save(
                    &meta_path,
                    &json!({"raw_count":raw_count,"low":bounds.lo,"high":bounds.hi,
            "kept_indices":kept,"cpu_seconds":cpu_seconds()-before}),
                )?;
                let raw_file = BoundFile::make(&raw_path)?;
                let meta_file = BoundFile::make(&meta_path)?;
                banks.push(json!({"context_index":ci,"initialization":init,"stream":stream,"raw":raw_file,"metadata":meta_file}));
                files.extend([raw_file, meta_file]);
            }
        }
    }
    let proposal = geometry.proposal(plan)?;
    for ci in 0..contexts.len() {
        for stream in 0..streams {
            let before = cpu_seconds();
            let engine = geometry.engine(plan, ci)?;
            let context = engine.context(&geometry.state)?;
            let old = engine.selected(&geometry.state);
            let ledger_path = output.join(format!("start-{ci}-{stream}-attempts.jsonl"));
            let mut ledger = Journal::new(&ledger_path)?;
            let mut chosen = None;
            for attempt in 1..=usize_at(&plan["allocation"], "prep_outer_cap")? {
                let mut prng = rng(
                    master,
                    ci,
                    "proposal_prepared",
                    stream,
                    attempt,
                    "preparation",
                );
                let outcome = match proposal.propose(&mut prng, &context, old) {
                    Ok(v) => v,
                    Err(e) => {
                        ledger.line(&json!({"attempt":attempt,"failure":e}))?;
                        anyhow::bail!("preparation failed");
                    }
                };
                let selected = outcome
                    .candidate
                    .as_ref()
                    .map(|c| [c.root, c.child])
                    .filter(|p| {
                        far_enough(
                            old,
                            *p,
                            num(&plan["preparation"], "minimum_max_center_displacement_A").unwrap(),
                            num(&plan["preparation"], "minimum_max_body_orientation_degrees")
                                .unwrap(),
                        )
                    });
                ledger.line(
                    &json!({"attempt":attempt,"outcome":outcome,"selected":selected.is_some()}),
                )?;
                if let Some(p) = selected {
                    chosen = Some((attempt, p));
                    break;
                }
            }
            let Some((attempts, selected)) = chosen else {
                anyhow::bail!(
                    "preparation exhausted for context {ci} stream {stream}; no replacement/fallback"
                );
            };
            let record_path = output.join(format!("start-{ci}-{stream}.json"));
            save(
                &record_path,
                &json!({"context_index":ci,"stream":stream,"source":old,"selected":selected,
            "attempts":attempts,"status":"prepared","is_equilibrium_sample":false,
            "preparation_cpu_seconds":cpu_seconds()-before}),
            )?;
            let record = BoundFile::make(&record_path)?;
            let ledger_file = BoundFile::make(&ledger_path)?;
            starts.push(json!({"context_index":ci,"stream":stream,"status":"prepared","attempts":attempts,"record":record,"ledger":ledger_file}));
            files.extend([record, ledger_file]);
        }
    }
    save(
        &output.join("manifest.json"),
        &json!({"schema":"evolving-dimer-prepared-starts-v1","complete":true,"passed":true,
        "config_sha256":hash_file(&args.config)?,"binding_sha256":hash_file(&args.binding)?,
        "geometry_load_cpu_seconds":geometry.load_cpu_seconds,
        "preparation_cpu_seconds":cpu_seconds()-preparation_started,
        "all_attempts_retained":true,"files":files,"cloud_banks":banks,"alternative_starts":starts}),
    )?;
    Ok(())
}
fn checkpoint(
    path: &Path,
    config_hash: &str,
    binding_hash: &str,
    job: &Value,
    block: usize,
    selected: [Pose; 2],
    journal: &Journal,
    budget: &Budget,
    counts: &Value,
) -> Result<()> {
    save(
        path,
        &json!({"schema":"evolving-dimer-checkpoint-v1","config_sha256":config_hash,
        "binding_sha256":binding_hash,"job":job,"block":block,"selected":selected,
        "journal_bytes":journal.bytes,"journal_rows":journal.rows,"journal_sha256":journal.sha(),
        "cpu_seconds":cpu_seconds()-budget.started,"raw":budget.raw,"retained":budget.retained,"counts":counts}),
    )
}
fn run(args: &Args, plan: &Value, binding: &Value, geometry: &Geometry) -> Result<()> {
    let job_id = args.job.context("--job required")?;
    let job = plan["jobs"]
        .as_array()
        .context("jobs")?
        .iter()
        .find(|j| j["id"].as_u64() == Some(job_id as u64))
        .context("unknown job")?;
    let ci = usize_at(job, "context_index")?;
    let stream = usize_at(job, "stream")?;
    let init = job["initialization"].as_str().context("initialization")?;
    let arm = job["arm"].as_str().context("arm")?;
    ensure!(
        ["local", "unguided", "m4", "root_m4", "two_root_m4"].contains(&arm)
            || singleton_arm(arm)
            || surrogate_arm(arm)
            || flexible_surrogate_arm(arm),
        "unknown arm"
    );
    singleton_policy(plan, arm)?;
    two_root_policy(plan, arm)?;
    surrogate_policy(plan, arm)?;
    flexible_surrogate_policy(plan, arm)?;
    let parent = PathBuf::from(plan["output"].as_str().context("output")?);
    fs::create_dir_all(&parent)?;
    let output = parent.join(format!("job-{job_id:03}"));
    if args.resume.is_none() {
        fs::create_dir(&output)?;
    } else {
        ensure!(output.is_dir(), "missing continuation directory");
        ensure!(
            !output.join("terminal.json").exists(),
            "completed run cannot be resumed"
        );
        ensure!(
            !output.join("failure.json").exists(),
            "fatal run receipt must be preserved; cannot resume"
        );
    }
    let result = run_inner(
        args, plan, binding, geometry, job, &output, ci, stream, init, arm,
    );
    if let Err(error) = &result {
        save(
            &output.join("failure.json"),
            &json!({"job":job,"complete":false,"error":format!("{error:#}")}),
        )?;
    }
    result
}
fn run_inner(
    args: &Args,
    plan: &Value,
    binding: &Value,
    geometry: &Geometry,
    job: &Value,
    output: &Path,
    ci: usize,
    stream: usize,
    init: &str,
    arm: &str,
) -> Result<()> {
    let singleton = singleton_policy(plan, arm)?;
    two_root_policy(plan, arm)?;
    let surrogate = surrogate_policy(plan, arm)?;
    let flexible = flexible_surrogate_policy(plan, arm)?;
    let surrogate_config = surrogate
        .as_ref()
        .or(flexible.as_ref())
        .map(|p| p.config(plan, arm))
        .transpose()?;
    let config_hash = hash_file(&args.config)?;
    let binding_hash = hash_file(&args.binding)?;
    let started = cpu_seconds();
    let prepared = bound(&binding["prepared_manifest"])?.json()?;
    ensure!(
        prepared["complete"] == true && prepared["passed"] == true,
        "incomplete preparation"
    );
    let (bank, meta, guide, surrogate_cloud) = if singleton.is_some() {
        // The bound manifest attests the original preparation. New singleton
        // arms authenticate only their reused start below, and never load or
        // thin any of the archived guidance clouds.
        (None, None, None, None)
    } else {
        for file in prepared["files"].as_array().context("prepared files")? {
            bound(file)?.read()?;
        }
        let bank = prepared["cloud_banks"]
            .as_array()
            .context("cloud_banks")?
            .iter()
            .find(|b| {
                b["context_index"].as_u64() == Some(ci as u64)
                    && b["initialization"] == init
                    && b["stream"].as_u64() == Some(stream as u64)
            })
            .context("missing cloud bank")?;
        let raw = bound(&bank["raw"])?.read()?;
        let meta = bound(&bank["metadata"])?.json()?;
        let low: [f64; 3] = serde_json::from_value(meta["low"].clone())?;
        let high: [f64; 3] = serde_json::from_value(meta["high"].clone())?;
        let point_volume = if surrogate.is_some() || flexible.is_some() {
            Some(surrogate_point_volume(
                &meta,
                usize_at(&plan["cloud"], "raw_count")?,
                &raw,
            )?)
        } else {
            None
        };
        ensure!(
            raw.len() == 24 * usize_at(&plan["cloud"], "raw_count")?,
            "cloud length differs"
        );
        let indices: Vec<usize> = serde_json::from_value(meta["kept_indices"].clone())?;
        let points: Vec<[f64; 3]> = indices
            .iter()
            .map(|&i| {
                std::array::from_fn(|k| {
                    let u = f64::from_le_bytes(
                        raw[i * 24 + k * 8..i * 24 + k * 8 + 8].try_into().unwrap(),
                    );
                    low[k] + (high[k] - low[k]) * u
                })
            })
            .collect();
        let (guide, surrogate_cloud) = if matches!(arm, "m4" | "root_m4" | "two_root_m4") {
            (
                Some(AuxiliaryOverlapThreshold::new(
                    &geometry.exclusion,
                    points,
                    4,
                )?),
                None,
            )
        } else {
            (None, point_volume.map(|volume| (points, volume)))
        };
        (Some(bank), Some(meta), guide, surrogate_cloud)
    };
    let mut state = geometry.state.clone();
    let engine = geometry.engine(plan, ci)?;
    let mut reused_start = Value::Null;
    if init == "proposal_prepared" {
        let start = prepared["alternative_starts"]
            .as_array()
            .context("starts")?
            .iter()
            .find(|s| {
                s["context_index"].as_u64() == Some(ci as u64)
                    && s["stream"].as_u64() == Some(stream as u64)
            })
            .context("missing prepared start")?;
        let record = bound(&start["record"])?.json()?;
        if singleton.is_some() || surrogate.is_some() || flexible.is_some() {
            if let Some(ledger) = start.get("ledger") {
                bound(ledger)?.read()?;
            }
            reused_start = start.clone();
        }
        let poses: [Pose; 2] = serde_json::from_value(record["selected"].clone())?;
        for i in 0..2 {
            state[engine.members[i]] = poses[i];
        }
    } else {
        ensure!(init == "source", "unknown initialization");
    }
    let limits: Limits = serde_json::from_value(plan["limits"].clone())?;
    let mut budget = Budget::new(limits)?;
    budget.started = started;
    let mut counts = if singleton.is_some() {
        json!({"local_attempted":0u64,"local_accepted":0u64,"singleton_attempted":0u64,"singleton_accepted":0u64,"singleton_self_loop":0u64})
    } else {
        json!({"local_attempted":0u64,"local_accepted":0u64,"dimer_attempted":0u64,"dimer_accepted":0u64,"dimer_self_loop":0u64})
    };
    let mut completed = 0usize;
    let journal_path = output.join("trajectory.jsonl");
    let mut journal = if let Some(path) = &args.resume {
        ensure!(
            !output.join("failure.json").exists(),
            "fatal runs cannot be resumed as ordinary rejections"
        );
        let cp: Value = serde_json::from_slice(&fs::read(path)?)?;
        ensure!(
            cp["schema"] == "evolving-dimer-checkpoint-v1"
                && cp["config_sha256"] == config_hash
                && cp["binding_sha256"] == binding_hash
                && cp["job"] == *job,
            "checkpoint provenance differs"
        );
        completed = usize_at(&cp, "block")?;
        ensure!(
            cp["journal_rows"].as_u64() == Some(1 + (completed as u64) * rows_per_block(arm)),
            "checkpoint row count differs from attempted schedule"
        );
        let last = last_journal_row(&journal_path)?;
        ensure!(
            last["kind"] == "retained_block"
                && last["block"] == cp["block"]
                && last["selected"] == cp["selected"]
                && last["raw"] == cp["raw"]
                && last["retained"] == cp["retained"]
                && last["counts"] == cp["counts"]
                && num(&cp, "cpu_seconds")? >= num(&last, "sampler_cpu_seconds")?,
            "checkpoint disagrees with bound retained trajectory"
        );
        let selected: [Pose; 2] = serde_json::from_value(cp["selected"].clone())?;
        for i in 0..2 {
            state[engine.members[i]] = selected[i];
        }
        budget.raw = cp["raw"].as_u64().context("raw counter")?;
        budget.retained = cp["retained"].as_u64().context("retained counter")?;
        ensure!(
            budget.raw <= limits.raw_campaign && budget.retained <= limits.retained_campaign,
            "checkpoint exceeds completed bath allocation"
        );
        budget.started -= num(&cp, "cpu_seconds")?;
        counts = cp["counts"].clone();
        Journal::resume(
            &journal_path,
            cp["journal_bytes"].as_u64().context("bytes")?,
            cp["journal_rows"].as_u64().context("rows")?,
            cp["journal_sha256"].as_str().context("hash")?,
        )?
    } else {
        let mut journal = Journal::new(&journal_path)?;
        let mut initial = if let Some(policy) = &singleton {
            json!({"kind":"initial","block":0,"job":job,"selected":engine.selected(&state),
                "fixed_source":plan["source_frame"],"prepared_manifest":binding["prepared_manifest"],
                "prepared_start":reused_start,"singleton_contract":singleton_contract(&engine,policy,arm),
                "geometry_load_cpu_seconds":geometry.load_cpu_seconds,
                "conditional_target":true,"sampler_cpu_seconds":cpu_seconds()-budget.started})
        } else {
            json!({"kind":"initial","block":0,"job":job,"selected":engine.selected(&state),
            "fixed_source":plan["source_frame"],"cloud":bank.unwrap(),"cloud_cpu_seconds":meta.as_ref().unwrap()["cpu_seconds"],
            "geometry_load_cpu_seconds":geometry.load_cpu_seconds,
            "conditional_target":true,"sampler_cpu_seconds":cpu_seconds()-budget.started})
        };
        if arm == "root_m4" {
            initial["guidance_contract"] = root_guidance_contract(&engine, bank.unwrap());
        }
        if arm == "two_root_m4" {
            initial["root_order_contract"] = two_root_contract(&engine, bank.unwrap());
        }
        if let Some(policy) = &surrogate {
            let (points, point_volume) = surrogate_cloud.as_ref().unwrap();
            initial["prepared_manifest"] = binding["prepared_manifest"].clone();
            initial["prepared_start"] = reused_start.clone();
            initial["surrogate_contract"] = surrogate_contract(
                &engine,
                policy,
                surrogate_config.unwrap(),
                bank.unwrap(),
                meta.as_ref().unwrap(),
                points.len(),
                *point_volume,
            );
        }
        if let Some(policy) = &flexible {
            let (points, point_volume) = surrogate_cloud.as_ref().unwrap();
            initial["prepared_manifest"] = binding["prepared_manifest"].clone();
            initial["prepared_start"] = reused_start;
            initial["flexible_surrogate_contract"] = flexible_surrogate_contract(
                &engine,
                policy,
                surrogate_config.unwrap(),
                bank.unwrap(),
                meta.as_ref().unwrap(),
                points.len(),
                *point_volume,
            );
        }
        journal.line(&initial)?;
        journal
    };
    ensure!(
        engine
            .context(&state)?
            .evaluate(engine.selected(&state))?
            .hard_valid(),
        "invalid evolving start"
    );
    let proposal = if singleton.is_none() && surrogate.is_none() && flexible.is_none() {
        Some(geometry.proposal(plan)?)
    } else {
        None
    };
    let warmup = usize_at(&plan["allocation"], "warmup_blocks")?;
    let total = warmup + usize_at(&plan["allocation"], "production_blocks")?;
    ensure!(completed <= total, "checkpoint block exceeds allocation");
    let end = args.stop_after_block.unwrap_or(total).min(total);
    ensure!(end >= completed, "stop precedes checkpoint");
    let master = plan["master_seed"].as_u64().context("master_seed")?;
    for block in completed + 1..=end {
        for (attempt, slot) in [0, 1, 0, 1].into_iter().enumerate() {
            let mut record = Value::Null;
            let mut prng = rng(
                master,
                ci,
                init,
                stream,
                block,
                &format!("local/{attempt}/proposal"),
            );
            let mut brng = rng(
                master,
                ci,
                init,
                stream,
                block,
                &format!("local/{attempt}/bath"),
            );
            let mut arng = rng(
                master,
                ci,
                init,
                stream,
                block,
                &format!("local/{attempt}/accept"),
            );
            let result = engine.local(
                &mut state,
                slot,
                num(&plan["local"], "translation_std_A")?,
                num(&plan["local"], "rotation_std_degrees")?,
                &mut prng,
                &mut brng,
                &mut arng,
                &mut budget,
                &mut record,
            );
            if record.is_null() {
                record = json!({"kind":"local","status":"failed_before_proposal"});
            }
            record["block"] = json!(block);
            record["attempt"] = json!(attempt);
            record["retained"] = json!(engine.selected(&state));
            record["sampler_cpu_seconds"] = json!(cpu_seconds() - budget.started);
            if let Err(error) = &result {
                record["fatal_error"] = json!(format!("{error:#}"));
            }
            journal.line(&record)?;
            result?;
            increment(&mut counts, "local_attempted");
            if record["accepted"] == true {
                increment(&mut counts, "local_accepted");
            }
        }
        if let Some(policy) = &singleton {
            let slot = (block - 1) % 2;
            let mut record = Value::Null;
            let mut prng = rng(
                master,
                ci,
                init,
                stream,
                block,
                &extra_role(arm, "proposal"),
            );
            let mut brng = rng(master, ci, init, stream, block, &extra_role(arm, "bath"));
            let mut arng = rng(master, ci, init, stream, block, &extra_role(arm, "accept"));
            let result = engine.two_neighbor_singleton(
                &mut state,
                slot,
                &geometry.model,
                &policy.oligomer_for(arm),
                policy.uniform_half_width,
                policy.trial_cap,
                &mut prng,
                &mut brng,
                &mut arng,
                &mut budget,
                &mut record,
            );
            if record.is_null() {
                record = json!({"kind":"two_neighbor_singleton","status":"failed_before_proposal"});
            }
            record["block"] = json!(block);
            record["member_slot"] = json!(slot);
            record["retained"] = json!(engine.selected(&state));
            record["sampler_cpu_seconds"] = json!(cpu_seconds() - budget.started);
            if let Err(error) = &result {
                record["fatal_error"] = json!(format!("{error:#}"));
            }
            journal.line(&record)?;
            result?;
            increment(&mut counts, "singleton_attempted");
            if record["accepted"] == true {
                increment(&mut counts, "singleton_accepted");
            }
            if record["status"] == "proposal_self_loop" {
                increment(&mut counts, "singleton_self_loop");
            }
        } else if let Some(config) = surrogate_config {
            let (points, point_volume) = surrogate_cloud.as_ref().unwrap();
            let kernel = RigidSurrogateKernel {
                core: engine.core,
                exclusion: engine.exclusion,
                wall: Some(engine.wall),
                wall_center: [0.; 3],
                members: engine.members,
                handle: engine.members[0],
                rd: engine.rd,
                activity: engine.activity,
                lambda: engine.lambda,
                envelope: engine.envelope,
                body_points: points,
                point_volume: *point_volume,
                config,
            };
            let role = |name| {
                if flexible.is_some() {
                    flexible_surrogate_role(name)
                } else {
                    surrogate_role(name)
                }
            };
            let mut prng = rng(master, ci, init, stream, block, &role("proposal"));
            let mut irng = rng(master, ci, init, stream, block, &role("inner_accept"));
            let mut brng = rng(master, ci, init, stream, block, &role("bath"));
            let mut arng = rng(master, ci, init, stream, block, &role("accept"));
            // A begun row survives a fatal/resource interruption. An incomplete
            // block is an auditable tail, never an ordinary rejected slot to retry.
            let begun = if flexible.is_some() {
                json!({"kind":"flexible_surrogate_attempt_begun",
                    "status":"begun","block":block,"members":engine.members,
                    "config":config,"old":engine.selected(&state),
                    "raw":budget.raw,"bath_retained":budget.retained,
                    "sampler_cpu_seconds":cpu_seconds()-budget.started})
            } else {
                json!({"kind":"rigid_surrogate_attempt_begun",
                "status":"begun","block":block,"members":engine.members,
                "handle":engine.members[0],"config":config,"old":engine.selected(&state),
                "raw":budget.raw,"bath_retained":budget.retained,
                "sampler_cpu_seconds":cpu_seconds()-budget.started})
            };
            journal.durable_line(&begun)?;
            let mut record = Value::Null;
            let result = if flexible.is_some() {
                FlexibleSurrogateKernel {
                    core: engine.core,
                    exclusion: engine.exclusion,
                    wall: Some(engine.wall),
                    wall_center: [0.; 3],
                    members: engine.members,
                    rd: engine.rd,
                    activity: engine.activity,
                    lambda: engine.lambda,
                    envelope: engine.envelope,
                    body_points: points,
                    point_volume: *point_volume,
                    config,
                }
                .step(
                    &mut state,
                    &mut prng,
                    &mut irng,
                    &mut brng,
                    &mut arng,
                    &mut budget,
                    &mut record,
                )
            } else {
                kernel.step(
                    &mut state,
                    &mut prng,
                    &mut irng,
                    &mut brng,
                    &mut arng,
                    &mut budget,
                    &mut record,
                )
            };
            record["block"] = json!(block);
            record["retained"] = json!(engine.selected(&state));
            record["sampler_cpu_seconds"] = json!(cpu_seconds() - budget.started);
            if let Err(error) = &result {
                record["fatal_error"] = json!(format!("{error:#}"));
            }
            journal.durable_line(&record)?;
            result?;
            increment(&mut counts, "dimer_attempted");
            if record["accepted"] == true {
                increment(&mut counts, "dimer_accepted");
            }
            if record["status"] == "identity_self_loop" {
                increment(&mut counts, "dimer_self_loop");
            }
        } else if arm != "local" {
            let mut record = Value::Null;
            let mut prng = rng(
                master,
                ci,
                init,
                stream,
                block,
                &dimer_role(arm, "proposal"),
            );
            let mut trng = rng(
                master,
                ci,
                init,
                stream,
                block,
                &dimer_role(arm, "threshold"),
            );
            let mut brng = rng(master, ci, init, stream, block, &dimer_role(arm, "bath"));
            let mut arng = rng(master, ci, init, stream, block, &dimer_role(arm, "accept"));
            let result = if arm == "two_root_m4" {
                // A separate role is consumed exactly once, before the entire
                // selected kernel, including cap failures and physical rejects.
                let selected_root = root_order(master, ci, init, stream, block);
                two_root_dimer(
                    &engine,
                    selected_root,
                    &mut state,
                    proposal.as_ref().unwrap(),
                    guide.as_ref().unwrap(),
                    &mut prng,
                    &mut trng,
                    &mut brng,
                    &mut arng,
                    &mut budget,
                    &mut record,
                )
            } else if arm == "root_m4" {
                let mut root_rng = rng(
                    master,
                    ci,
                    init,
                    stream,
                    block,
                    &dimer_role(arm, "root_threshold"),
                );
                engine.dimer_with_guides(
                    &mut state,
                    proposal.as_ref().unwrap(),
                    guide.as_ref(),
                    guide.as_ref(),
                    &mut prng,
                    Some(&mut root_rng),
                    Some(&mut trng),
                    &mut brng,
                    &mut arng,
                    &mut budget,
                    &mut record,
                )
            } else {
                engine.dimer(
                    &mut state,
                    proposal.as_ref().unwrap(),
                    guide.as_ref(),
                    &mut prng,
                    &mut trng,
                    &mut brng,
                    &mut arng,
                    &mut budget,
                    &mut record,
                )
            };
            if record.is_null() {
                record = json!({"kind":"factorized_dimer","status":"failed_before_proposal"});
            }
            record["block"] = json!(block);
            record["retained"] = json!(engine.selected(&state));
            record["sampler_cpu_seconds"] = json!(cpu_seconds() - budget.started);
            if let Err(error) = &result {
                record["fatal_error"] = json!(format!("{error:#}"));
            }
            journal.line(&record)?;
            result?;
            increment(&mut counts, "dimer_attempted");
            if record["accepted"] == true {
                increment(&mut counts, "dimer_accepted");
            }
            if record["status"] == "proposal_self_loop" {
                increment(&mut counts, "dimer_self_loop");
            }
        }
        journal.line(
            &json!({"kind":"retained_block","block":block,"production":block>warmup,
            "selected":engine.selected(&state),"sampler_cpu_seconds":cpu_seconds()-budget.started,
            "raw":budget.raw,"retained":budget.retained,"counts":counts}),
        )?;
        checkpoint(
            &output.join("checkpoint.json"),
            &config_hash,
            &binding_hash,
            job,
            block,
            engine.selected(&state),
            &journal,
            &budget,
            &counts,
        )?;
    }
    save(
        &output.join(if end == total {
            "terminal.json"
        } else {
            "paused.json"
        }),
        &json!({
        "complete":end==total,"job":job,"blocks":end,"counts":counts,"raw":budget.raw,"retained":budget.retained,
        "cpu_seconds":cpu_seconds()-budget.started,"trajectory":BoundFile::make(&journal_path)?,
        "geometry_load_cpu_seconds":geometry.load_cpu_seconds,
        "config_sha256":config_hash,"binding_sha256":binding_hash,"conditional_target":true}),
    )?;
    Ok(())
}
fn increment(counts: &mut Value, key: &str) {
    counts[key] = json!(counts[key].as_u64().unwrap() + 1);
}
fn main() -> Result<()> {
    let args = Args::parse();
    let plan: Value = serde_json::from_slice(&fs::read(&args.config)?)?;
    let binding = validate_binding(&args, &plan)?;
    let geometry = Geometry::load(&plan)?;
    match args.mode.as_str() {
        "prepare-starts" => prepare(&args, &plan, &geometry),
        "run" => run(&args, &plan, &binding, &geometry),
        _ => anyhow::bail!("mode must be prepare-starts or run"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn synthetic_flexible_surrogate_policy() -> Value {
        json!({"schema":"flexible-surrogate-policy-v1","proposal_scales":{"source":"local"}})
    }
    #[test]
    fn flexible_surrogate_policy_is_explicit_and_does_not_change_existing_arms() -> Result<()> {
        let policy = synthetic_flexible_surrogate_policy();
        let local = json!({"translation_std_A":0.2,"rotation_std_degrees":1.});
        for (arm, steps, strength) in [
            ("flexible_m1", 1, 1.),
            ("flexible_m8", 8, 1.),
            ("flexible_flat8", 8, 0.),
        ] {
            assert!(flexible_surrogate_policy(&json!({"local":local}), arm).is_err());
            let plan = json!({"local":local,"flexible_surrogate_policy":policy});
            let config = flexible_surrogate_policy(&plan, arm)?
                .unwrap()
                .config(&plan, arm)?;
            assert_eq!(config.inner_steps, steps);
            assert_eq!(config.guidance_strength, strength);
            assert_eq!(config.translation_std, 0.2);
            assert_eq!(config.rotation_std_degrees, 1.);
            assert_eq!(rows_per_block(arm), 7);
            assert!(surrogate_policy(&plan, arm)?.is_none());
            for bad in [
                Value::Null,
                json!({}),
                synthetic_surrogate_policy(),
                json!({"schema":"flexible-surrogate-policy-v1","proposal_scales":{"source":"local","translation_std_A":0.1}}),
                json!({"schema":"flexible-surrogate-policy-v1","proposal_scales":{"source":"local"},"inner_steps":2}),
            ] {
                assert!(
                    flexible_surrogate_policy(
                        &json!({"local":local,"flexible_surrogate_policy":bad}),
                        arm
                    )
                    .is_err()
                );
            }
            let mut overridden = plan.clone();
            overridden["flexible_surrogate_policy"]["proposal_scales"] = json!({"source":"frozen_override","translation_std_A":0.5,"rotation_std_degrees":6.});
            let config = flexible_surrogate_policy(&overridden, arm)?
                .unwrap()
                .config(&overridden, arm)?;
            assert_eq!(
                (config.translation_std, config.rotation_std_degrees),
                (0.5, 6.)
            );
            overridden["flexible_surrogate_policy"]["proposal_scales"]["translation_std_A"] =
                json!(-1.);
            assert!(flexible_surrogate_policy(&overridden, arm).is_err());
            let mut mixed = plan;
            mixed["surrogate_policy"] = synthetic_surrogate_policy();
            assert!(surrogate_policy(&mixed, arm).is_err());
        }
        for arm in [
            "local",
            "unguided",
            "m4",
            "root_m4",
            "two_root_m4",
            "singleton_two_neighbor",
            "singleton_two_neighbor_unfused",
            "rigid_surrogate_1",
            "rigid_surrogate_8",
            "rigid_surrogate_flat8",
        ] {
            assert!(flexible_surrogate_policy(&json!({}), arm)?.is_none());
            assert!(
                flexible_surrogate_policy(&json!({"flexible_surrogate_policy":policy}), arm)
                    .is_err()
            );
            assert!(
                flexible_surrogate_policy(&json!({"flexible_surrogate_policy":null}), arm).is_err()
            );
        }
        Ok(())
    }
    #[test]
    fn flexible_surrogate_roles_are_shared_only_within_the_new_family() {
        for init in ["source", "proposal_prepared"] {
            for block in 1..=8 {
                let values: Vec<_> = ["proposal", "inner_accept", "bath", "accept"]
                    .into_iter()
                    .map(|role| {
                        rng(77, 0, init, 0, block, &flexible_surrogate_role(role)).random::<u64>()
                    })
                    .collect();
                for i in 0..values.len() {
                    for j in 0..i {
                        assert_ne!(values[i], values[j]);
                    }
                }
                for role in ["proposal", "inner_accept", "bath", "accept"] {
                    assert_eq!(
                        flexible_surrogate_role(role),
                        format!("flexible_surrogate/{role}")
                    );
                    for prior in [surrogate_role(role), format!("local/0/{role}")] {
                        assert_ne!(
                            rng(77, 0, init, 0, block, &flexible_surrogate_role(role))
                                .random::<u64>(),
                            rng(77, 0, init, 0, block, &prior).random::<u64>()
                        );
                    }
                }
            }
        }
    }
    fn synthetic_surrogate_policy() -> Value {
        json!({"schema":"rigid-surrogate-policy-v1","proposal_scales":{"source":"local"}})
    }
    #[test]
    fn surrogate_policy_requires_explicit_matched_scales_and_fixed_arm_controls() -> Result<()> {
        let policy = synthetic_surrogate_policy();
        let local = json!({"translation_std_A":0.125,"rotation_std_degrees":3.});
        for (arm, steps, strength) in [
            ("rigid_surrogate_1", 1, 1.),
            ("rigid_surrogate_8", 8, 1.),
            ("rigid_surrogate_flat8", 8, 0.),
        ] {
            assert!(surrogate_policy(&json!({"local":local}), arm).is_err());
            let plan = json!({"surrogate_policy":policy,"local":local});
            let config = surrogate_policy(&plan, arm)?.unwrap().config(&plan, arm)?;
            assert_eq!(config.inner_steps, steps);
            assert_eq!(config.guidance_strength, strength);
            assert_eq!(config.translation_std, 0.125);
            assert_eq!(config.rotation_std_degrees, 3.);
            assert_eq!(rows_per_block(arm), 7);
            for bad in [
                Value::Null,
                json!({}),
                json!({"schema":"other"}),
                json!({"schema":"rigid-surrogate-policy-v1","proposal_scales":{"source":"local","translation_std_A":1.}}),
                json!({"schema":"rigid-surrogate-policy-v1","proposal_scales":{"source":"local"},"inner_steps":2}),
            ] {
                assert!(
                    surrogate_policy(&json!({"surrogate_policy":bad,"local":local}), arm).is_err()
                );
            }
            let plan = json!({"local":local,"surrogate_policy":{
                "schema":"rigid-surrogate-policy-v1","proposal_scales":{
                    "source":"frozen_override","translation_std_A":0.5,"rotation_std_degrees":6.}}});
            let config = surrogate_policy(&plan, arm)?.unwrap().config(&plan, arm)?;
            assert_eq!(config.translation_std, 0.5);
            assert_eq!(config.rotation_std_degrees, 6.);
            let mut bad = plan;
            bad["surrogate_policy"]["proposal_scales"]["translation_std_A"] = json!(-0.1);
            assert!(surrogate_policy(&bad, arm).is_err());
        }
        for arm in [
            "local",
            "unguided",
            "m4",
            "root_m4",
            "two_root_m4",
            "singleton_two_neighbor",
            "singleton_two_neighbor_unfused",
        ] {
            assert!(surrogate_policy(&json!({}), arm)?.is_none());
            assert!(surrogate_policy(&json!({"surrogate_policy":policy}), arm).is_err());
            assert!(surrogate_policy(&json!({"surrogate_policy":null}), arm).is_err());
            assert_eq!(rows_per_block(arm), if arm == "local" { 5 } else { 6 });
        }
        Ok(())
    }
    #[test]
    fn surrogate_streams_share_arm_controls_and_separate_acceptance_stages() {
        for init in ["source", "proposal_prepared"] {
            for block in 1..=8 {
                let values: Vec<_> = ["proposal", "inner_accept", "bath", "accept"]
                    .into_iter()
                    .map(|role| rng(77, 0, init, 0, block, &surrogate_role(role)).random::<u64>())
                    .collect();
                for i in 0..values.len() {
                    for j in 0..i {
                        assert_ne!(values[i], values[j]);
                    }
                }
                for role in ["proposal", "inner_accept", "bath", "accept"] {
                    assert_eq!(surrogate_role(role), format!("rigid_surrogate/{role}"));
                    assert_ne!(
                        rng(77, 0, init, 0, block, &surrogate_role(role)).random::<u64>(),
                        rng(77, 0, init, 0, block, &format!("local/0/{role}")).random::<u64>()
                    );
                }
            }
        }
    }
    #[test]
    fn surrogate_cloud_weight_uses_raw_count_and_rejects_corrupted_metadata() -> Result<()> {
        let raw: Vec<_> = [0.55f64, 0.5, 0.5, 0., 0., 0.]
            .into_iter()
            .flat_map(f64::to_le_bytes)
            .collect();
        let meta = json!({"raw_count":2,"low":[-1.,-1.,-1.],"high":[1.,1.,1.],"kept_indices":[0]});
        assert_eq!(surrogate_point_volume(&meta, 2, &raw)?, 4.);
        let mut empty = meta.clone();
        empty["kept_indices"] = json!([]);
        assert_eq!(surrogate_point_volume(&empty, 2, &raw)?, 4.);
        assert!(surrogate_point_volume(&meta, 1, &raw).is_err());
        assert!(surrogate_point_volume(&meta, 2, &raw[..24]).is_err());
        for indices in [json!([2]), json!([0, 0]), json!([1, 0])] {
            let mut bad = meta.clone();
            bad["kept_indices"] = indices;
            assert!(surrogate_point_volume(&bad, 2, &raw).is_err());
        }
        let mut bad = meta;
        bad["raw_count"] = json!(0);
        assert!(surrogate_point_volume(&bad, 0, &[]).is_err());
        Ok(())
    }
    fn synthetic_two_root_policy() -> Value {
        json!({"schema":"two-root-factorized-m4-policy-v1",
            "root_probabilities":[0.5,0.5],"selection":"once_per_global_slot","internal_threshold_m":4})
    }
    #[test]
    fn two_root_policy_and_separate_root_role_preserve_control_streams() -> Result<()> {
        let policy = synthetic_two_root_policy();
        two_root_policy(&json!({"two_root_policy":policy}), "two_root_m4")?;
        for arm in [
            "local",
            "unguided",
            "m4",
            "root_m4",
            "singleton_two_neighbor",
            "singleton_two_neighbor_unfused",
        ] {
            two_root_policy(&json!({}), arm)?;
            assert!(two_root_policy(&json!({"two_root_policy":policy}), arm).is_err());
        }
        for bad in [Value::Null, json!({}), json!({"schema":"other"})] {
            assert!(two_root_policy(&json!({"two_root_policy":bad}), "two_root_m4").is_err());
        }
        for key in ["root_probabilities", "selection", "internal_threshold_m"] {
            let mut bad = policy.clone();
            bad[key] = Value::Null;
            assert!(two_root_policy(&json!({"two_root_policy":bad}), "two_root_m4").is_err());
        }
        let choices: Vec<_> = (1..=64)
            .map(|block| root_order(77, 0, "source", 2, block))
            .collect();
        assert!(choices.contains(&0) && choices.contains(&1));
        assert_eq!(
            choices,
            (1..=19)
                .chain(20..=64)
                .map(|block| root_order(77, 0, "source", 2, block))
                .collect::<Vec<_>>()
        );
        for block in 1..=8 {
            for role in ["proposal", "threshold", "bath", "accept"] {
                assert_eq!(dimer_role("two_root_m4", role), format!("m4/{role}"));
                assert_eq!(
                    rng(77, 0, "source", 2, block, &dimer_role("two_root_m4", role))
                        .random::<u64>(),
                    rng(77, 0, "source", 2, block, &dimer_role("m4", role)).random::<u64>()
                );
            }
            assert_eq!(
                dimer_role("two_root_m4", "root_order"),
                "two_root_m4/root_order"
            );
            assert_ne!(
                rng(77, 0, "source", 2, block, "two_root_m4/root_order").random::<u64>(),
                rng(77, 0, "source", 2, block, "m4/proposal").random::<u64>()
            );
        }
        Ok(())
    }
    fn two_root_toy_plan() -> Value {
        json!({"contexts":[{"root":0,"child":1,"anchor":2}],
            "physical":{"wall_radius":50.,"depletant_radius":0.8,"activity":0.05,"lambda_ratio":64.},
            "envelope":{"max_cells":63,"max_depth":6,"min_width":0.},
            "factorized":{"order":"root_first","uniform_half_width":4.,"uniform_probability":0.5,
                "root_cap":32,"internal_cap":32,"joint_cap":1}})
    }
    fn two_root_budget() -> Result<Budget> {
        Budget::new(Limits {
            raw_per_leg: 1_000_000,
            raw_per_outer: 2_000_000,
            raw_campaign: 20_000_000,
            retained_per_leg: 1_000_000,
            retained_per_outer: 2_000_000,
            retained_campaign: 20_000_000,
            cpu_seconds: 60.,
        })
    }
    #[test]
    fn two_root_forced_orders_match_fixed_label_kernels_and_canonical_mapping() -> Result<()> {
        let geometry = synthetic_geometry()?;
        let plan = two_root_toy_plan();
        let engine = geometry.engine(&plan, 0)?;
        let proposal = geometry.proposal(&plan)?;
        let guide = AuxiliaryOverlapThreshold::new(&geometry.exclusion, vec![[0.8, 0., 0.]], 4)?;
        for root in [0, 1] {
            let reference_engine = rooted_engine(&engine, root, geometry.state.len())?;
            for seed in 1..=4 {
                let mut actual = geometry.state.clone();
                let mut expected = actual.clone();
                let mut a = Value::Null;
                let mut b = Value::Null;
                let streams = || {
                    (
                        StdRng::seed_from_u64(seed),
                        StdRng::seed_from_u64(seed + 100),
                        StdRng::seed_from_u64(seed + 200),
                        StdRng::seed_from_u64(seed + 300),
                    )
                };
                let (mut ap, mut at, mut ab, mut aa) = streams();
                let (mut bp, mut bt, mut bb, mut ba) = streams();
                let mut budget_a = two_root_budget()?;
                let mut budget_b = two_root_budget()?;
                two_root_dimer(
                    &engine,
                    root,
                    &mut actual,
                    &proposal,
                    &guide,
                    &mut ap,
                    &mut at,
                    &mut ab,
                    &mut aa,
                    &mut budget_a,
                    &mut a,
                )?;
                reference_engine.dimer(
                    &mut expected,
                    &proposal,
                    Some(&guide),
                    &mut bp,
                    &mut bt,
                    &mut bb,
                    &mut ba,
                    &mut budget_b,
                    &mut b,
                )?;
                assert_eq!(json!(actual), json!(expected));
                assert_eq!(actual[2].position, geometry.state[2].position);
                assert_eq!(a["canonical_members"], json!([0, 1]));
                assert_eq!(a["selected_members"], json!([root, 1 - root]));
                assert_eq!(a["members"], a["selected_members"]);
                assert_eq!(a["canonical_old"], json!(engine.selected(&geometry.state)));
                assert_eq!(a["root_order_log_reverse_forward"], 0.);
                assert_eq!(
                    a["proposal"]["guidance"]["old_count"],
                    if root == 0 { 1 } else { 0 }
                );
                if a["proposed"].is_array() {
                    assert_eq!(
                        a["canonical_proposed"],
                        json!([a["proposed"][root], a["proposed"][1 - root]])
                    );
                }
                for field in [
                    "canonical_members",
                    "selected_members",
                    "root_slot",
                    "root_order_probability",
                    "root_order_log_reverse_forward",
                    "root_order_rng_role",
                    "canonical_old",
                    "canonical_proposed",
                ] {
                    a.as_object_mut().unwrap().remove(field);
                }
                assert_eq!(a, b);
                assert_eq!(
                    (budget_a.raw, budget_a.retained),
                    (budget_b.raw, budget_b.retained)
                );
                assert_eq!(
                    (
                        ap.random::<u64>(),
                        at.random::<u64>(),
                        ab.random::<u64>(),
                        aa.random::<u64>()
                    ),
                    (
                        bp.random::<u64>(),
                        bt.random::<u64>(),
                        bb.random::<u64>(),
                        ba.random::<u64>()
                    )
                );
            }
        }
        Ok(())
    }
    #[test]
    fn two_root_null_fatal_and_invalid_labels_retain_selection_without_mutation() -> Result<()> {
        let geometry = synthetic_geometry()?;
        let mut plan = two_root_toy_plan();
        plan["factorized"]["root_cap"] = json!(0);
        let engine = geometry.engine(&plan, 0)?;
        let proposal = geometry.proposal(&plan)?;
        let guide = AuxiliaryOverlapThreshold::new(&geometry.exclusion, vec![[0.8, 0., 0.]], 4)?;
        for root in [0, 1] {
            for fatal in [false, true] {
                let mut state = geometry.state.clone();
                let mut record = Value::Null;
                let mut budget = two_root_budget()?;
                if fatal {
                    budget.started = cpu_seconds() - 120.;
                }
                let mut a = StdRng::seed_from_u64(1);
                let mut b = StdRng::seed_from_u64(2);
                let mut c = StdRng::seed_from_u64(3);
                let mut d = StdRng::seed_from_u64(4);
                let result = two_root_dimer(
                    &engine,
                    root,
                    &mut state,
                    &proposal,
                    &guide,
                    &mut a,
                    &mut b,
                    &mut c,
                    &mut d,
                    &mut budget,
                    &mut record,
                );
                assert_eq!(result.is_err(), fatal);
                assert_eq!(json!(state), json!(geometry.state));
                assert_eq!(record["root_slot"], root);
                assert_eq!(record["canonical_members"], json!([0, 1]));
                assert_eq!(record["selected_members"], json!([root, 1 - root]));
                assert_eq!(
                    record["status"],
                    if fatal {
                        "failed_before_proposal"
                    } else {
                        "proposal_self_loop"
                    }
                );
                assert!(record.get("bath").is_none());
                assert!(record.get("log_u").is_none());
                assert_eq!(a.random::<u64>(), StdRng::seed_from_u64(1).random::<u64>());
                assert_eq!(b.random::<u64>(), StdRng::seed_from_u64(2).random::<u64>());
            }
        }
        assert!(rooted_engine(&engine, 2, geometry.state.len()).is_err());
        assert!(rooted_engine(&engine, 0, 1).is_err());
        let mut bad = rooted_engine(&engine, 0, geometry.state.len())?;
        bad.members = [0, 0];
        assert!(rooted_engine(&bad, 0, geometry.state.len()).is_err());
        bad.members = [0, 2];
        assert!(rooted_engine(&bad, 0, geometry.state.len()).is_err());
        bad.members = [2, 0];
        bad.anchor = 1;
        let reordered = rooted_engine(&bad, 1, geometry.state.len())?;
        assert_eq!(reordered.members, [0, 2]);
        assert_eq!(reordered.anchor, 1);
        assert_eq!(
            json!(reordered.selected(&geometry.state)),
            json!([geometry.state[0], geometry.state[2]])
        );
        Ok(())
    }
    #[test]
    fn two_root_prior_mixture_preserves_discrete_balance() {
        // Exact integer flows: each matrix has denominator12, pi∝[1,2,3].
        // Their fair mixture has denominator24, with every self-loop retained.
        let pi = [1, 2, 3];
        let a = [[6, 6, 0], [3, 9, 0], [0, 0, 12]];
        let b = [[12, 0, 0], [0, 6, 6], [0, 4, 8]];
        for i in 0..3 {
            assert_eq!((0..3).map(|j| a[i][j] + b[i][j]).sum::<i32>(), 24);
            for j in 0..3 {
                assert_eq!(pi[i] * a[i][j], pi[j] * a[j][i]);
                assert_eq!(pi[i] * b[i][j], pi[j] * b[j][i]);
                assert_eq!(pi[i] * (a[i][j] + b[i][j]), pi[j] * (a[j][i] + b[j][i]));
            }
        }
        // A source-dependent root preference does not inherit cancellation.
        assert_ne!(pi[0] * (a[0][1] + b[0][1]), pi[1] * (2 * b[1][0]));
    }
    #[test]
    fn preparation_uses_quaternion_sign_invariant_distance() {
        let a = Pose {
            position: [0.; 3],
            orientation: [1., 0., 0., 0.],
        };
        let minus = Pose {
            orientation: [-1., 0., 0., 0.],
            ..a
        };
        assert!(!far_enough([a, a], [minus, minus], 5., 10.));
        let b = Pose {
            position: [5., 0., 0.],
            ..a
        };
        assert!(far_enough([a, a], [b, a], 5., 10.));
    }
    #[test]
    fn named_stream_continuation_and_role_separation() {
        let a = (1..=8)
            .map(|b| rng(77, 1, "source", 2, b, "local/0/proposal").random::<u64>())
            .collect::<Vec<_>>();
        let b = (1..=3)
            .chain(4..=8)
            .map(|b| rng(77, 1, "source", 2, b, "local/0/proposal").random::<u64>())
            .collect::<Vec<_>>();
        assert_eq!(a, b);
        assert_ne!(
            a[0],
            rng(77, 1, "source", 2, 1, "m4/proposal").random::<u64>()
        );
    }
    #[test]
    fn root_arm_shares_control_streams_but_separates_root_threshold() {
        for arm in ["unguided", "m4"] {
            for role in ["proposal", "threshold", "bath", "accept"] {
                assert_eq!(dimer_role(arm, role), format!("{arm}/{role}"));
            }
        }
        for block in 1..=8 {
            for role in ["proposal", "threshold", "bath", "accept"] {
                assert_eq!(dimer_role("root_m4", role), format!("m4/{role}"));
                assert_eq!(
                    rng(77, 1, "source", 2, block, &dimer_role("root_m4", role)).random::<u64>(),
                    rng(77, 1, "source", 2, block, &dimer_role("m4", role)).random::<u64>(),
                );
            }
            assert_ne!(
                rng(
                    77,
                    1,
                    "source",
                    2,
                    block,
                    &dimer_role("root_m4", "root_threshold")
                )
                .random::<u64>(),
                rng(
                    77,
                    1,
                    "source",
                    2,
                    block,
                    &dimer_role("root_m4", "threshold")
                )
                .random::<u64>(),
            );
        }
        assert_eq!(
            dimer_role("root_m4", "root_threshold"),
            "root_m4/root_threshold"
        );
    }

    fn synthetic_geometry() -> Result<Geometry> {
        use tetramer_mc::geometry::Atom;
        let sphere = |radius: f64| -> Result<SphereTree> {
            SphereTree::new(Shape {
                name: "synthetic sphere".into(),
                volume: 4. * std::f64::consts::PI * radius.powi(3) / 3.,
                atoms: vec![Atom {
                    center: [0.; 3],
                    radius,
                }],
            })
        };
        let core = sphere(0.2)?;
        let exclusion = sphere(1.)?;
        let wall = Container::new(50., &core)?;
        let pose = |x| Pose {
            position: [x, 0., 0.],
            orientation: [1., 0., 0., 0.],
        };
        let sha = "0".repeat(64);
        let cov: [[f64; 6]; 6] =
            std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 0.01 } else { 0. }));
        let model_json = json!({"coordinate_convention":"anchor-body-relative","shape_sha256":sha,
            "angular_length":1.,"weights":[1.],"anchors":[{"position":[0.6,0.,0.],
            "rotation":[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]}],
            "means":[[0.,0.,0.,0.,0.,0.]],"covariances":[cov]})
        .to_string();
        let model = DockingProposal::new(
            FrozenRelativePoseProposal::from_json_str_open(&model_json, [100.; 3], 0.1, &sha)?,
            DockingMethod::PosteriorInvolution,
            0.,
            [0.; 3],
        )?;
        Ok(Geometry {
            core,
            exclusion,
            wall,
            state: vec![pose(0.6), pose(1.2), pose(0.)],
            model,
            load_cpu_seconds: 0.,
        })
    }
    #[test]
    fn root_arm_frozen_cloud_and_prepared_starts_replay_across_disk_checkpoint() -> Result<()> {
        root_arm_restart("root_m4")
    }
    #[test]
    fn two_root_frozen_cloud_and_prepared_starts_replay_across_disk_checkpoint() -> Result<()> {
        root_arm_restart("two_root_m4")
    }
    #[test]
    fn surrogate_all_arms_and_frozen_starts_replay_across_disk_checkpoint() -> Result<()> {
        for arm in [
            "rigid_surrogate_1",
            "rigid_surrogate_8",
            "rigid_surrogate_flat8",
        ] {
            root_arm_restart(arm)?;
        }
        Ok(())
    }
    #[test]
    fn flexible_surrogate_all_arms_and_starts_replay_checkpoint_and_preserve_fatal_tails()
    -> Result<()> {
        for arm in ["flexible_m1", "flexible_m8", "flexible_flat8"] {
            root_arm_restart(arm)?;
        }
        Ok(())
    }
    fn root_arm_restart(arm: &str) -> Result<()> {
        let flexible = flexible_surrogate_arm(arm);
        let any_surrogate = surrogate_arm(arm) || flexible;
        let policy_key = if flexible {
            "flexible_surrogate_policy"
        } else {
            "surrogate_policy"
        };
        let begun_kind = if flexible {
            "flexible_surrogate_attempt_begun"
        } else {
            "rigid_surrogate_attempt_begun"
        };
        let outcome_kind = if flexible {
            "flexible_surrogate_chain"
        } else {
            "rigid_surrogate_chain"
        };
        let dir = std::env::temp_dir().join(format!("{arm}-restart-{}", std::process::id()));
        fs::create_dir(&dir)?;
        let geometry = synthetic_geometry()?;
        let pose = |x| Pose {
            position: [x, 0., 0.],
            orientation: [1., 0., 0., 0.],
        };
        let raw_path = dir.join("frozen-cloud.bin");
        let mut raw = [0.55f64, 0.5, 0.5]
            .into_iter()
            .flat_map(f64::to_le_bytes)
            .collect::<Vec<_>>();
        let raw_count = if any_surrogate { 2 } else { 1 };
        if any_surrogate {
            raw.extend([0.0f64, 0., 0.].into_iter().flat_map(f64::to_le_bytes));
        }
        fs::write(&raw_path, &raw)?;
        let meta_path = dir.join("frozen-cloud.json");
        save(
            &meta_path,
            &json!({"raw_count":raw_count,"low":[-1.,-1.,-1.],"high":[1.,1.,1.],
            "kept_indices":[0],"cpu_seconds":0.}),
        )?;
        let prepared_path = dir.join("frozen-start.json");
        save(&prepared_path, &json!({"selected":[pose(0.62),pose(1.23)]}))?;
        let raw_file = BoundFile::make(&raw_path)?;
        let meta_file = BoundFile::make(&meta_path)?;
        let start_file = BoundFile::make(&prepared_path)?;
        let banks: Vec<Value> = ["source", "proposal_prepared"]
            .into_iter()
            .map(|init| {
                json!({"context_index":0,"initialization":init,"stream":0,
                "raw":raw_file,"metadata":meta_file})
            })
            .collect();
        let manifest_path = dir.join("original-prepared-manifest.json");
        save(
            &manifest_path,
            &json!({"complete":true,"passed":true,
            "files":[raw_file,meta_file,start_file],"cloud_banks":banks,
            "alternative_starts":[{"context_index":0,"stream":0,"record":start_file}]}),
        )?;
        let binding = json!({"prepared_manifest":BoundFile::make(&manifest_path)?});
        let binding_path = dir.join("binding.json");
        save(&binding_path, &binding)?;
        let frozen: Vec<_> = [&raw_path, &meta_path, &prepared_path, &manifest_path]
            .into_iter()
            .map(|p| (p.clone(), hash_file(p).unwrap()))
            .collect();
        for init in ["source", "proposal_prepared"] {
            let plan_for = |output: PathBuf| {
                let mut value = json!({
                    "master_seed":77,"output":output,"contexts":[{"root":0,"child":1,"anchor":2}],
                    "source_frame":{"synthetic":"frozen_source"},
                    "physical":{"wall_radius":50.,"depletant_radius":0.8,"activity":0.05,"lambda_ratio":64.},
                    "factorized":{"order":"root_first","uniform_half_width":4.,"uniform_probability":0.,
                        "root_cap":4,"internal_cap":4,"joint_cap":2},
                    "cloud":{"raw_count":raw_count},"local":{"translation_std_A":0.02,"rotation_std_degrees":1.},
                    "envelope":{"max_cells":63,"max_depth":6,"min_width":0.},
                    "limits":{"raw_per_leg":1000000,"raw_per_outer":2000000,"raw_campaign":20000000,
                        "retained_per_leg":1000000,"retained_per_outer":2000000,"retained_campaign":20000000,
                        "cpu_seconds":60.},
                    "allocation":{"warmup_blocks":2,"production_blocks":6},
                    "jobs":[{"id":0,"context_index":0,"stream":0,"initialization":init,"arm":arm}]
                });
                if arm == "two_root_m4" {
                    value["two_root_policy"] = synthetic_two_root_policy();
                }
                if surrogate_arm(arm) {
                    value["surrogate_policy"] = synthetic_surrogate_policy();
                }
                if flexible {
                    value["flexible_surrogate_policy"] = synthetic_flexible_surrogate_policy();
                }
                value
            };
            let full_plan = plan_for(dir.join(format!("{init}-full")));
            let split_plan = plan_for(dir.join(format!("{init}-split")));
            let full_config = dir.join(format!("{init}-full.json"));
            let split_config = dir.join(format!("{init}-split.json"));
            save(&full_config, &full_plan)?;
            save(&split_config, &split_plan)?;
            let make_args = |config, stop, resume| Args {
                config,
                binding: binding_path.clone(),
                mode: "run".into(),
                job: Some(0),
                stop_after_block: stop,
                resume,
            };
            run(
                &make_args(full_config, None, None),
                &full_plan,
                &binding,
                &geometry,
            )?;
            run(
                &make_args(split_config.clone(), Some(3), None),
                &split_plan,
                &binding,
                &geometry,
            )?;
            let split_out = dir.join(format!("{init}-split/job-000"));
            let checkpoint_path = split_out.join("checkpoint.json");
            let checkpoint: Value = serde_json::from_slice(&fs::read(&checkpoint_path)?)?;
            assert_eq!(checkpoint["block"], 3);
            assert_eq!(checkpoint["journal_rows"], 1 + 3 * rows_per_block(arm));
            run(
                &make_args(split_config, None, Some(checkpoint_path)),
                &split_plan,
                &binding,
                &geometry,
            )?;
            let full_out = dir.join(format!("{init}-full/job-000"));
            let rows = |path: &Path| -> Result<Vec<Value>> {
                fs::read_to_string(path)?
                    .lines()
                    .map(|line| {
                        let mut value: Value = serde_json::from_str(line)?;
                        value.as_object_mut().unwrap().remove("sampler_cpu_seconds");
                        Ok(value)
                    })
                    .collect()
            };
            let actual = rows(&split_out.join("trajectory.jsonl"))?;
            assert_eq!(actual, rows(&full_out.join("trajectory.jsonl"))?);
            assert_eq!(actual.len() as u64, 1 + 8 * rows_per_block(arm));
            if any_surrogate {
                let contract = &actual[0][if flexible {
                    "flexible_surrogate_contract"
                } else {
                    "surrogate_contract"
                }];
                let policy = if flexible {
                    flexible_surrogate_policy(&full_plan, arm)?
                } else {
                    surrogate_policy(&full_plan, arm)?
                };
                let config = policy.unwrap().config(&full_plan, arm)?;
                assert_eq!(contract["members"], json!([0, 1]));
                if flexible {
                    assert!(contract.get("handle").is_none());
                    assert_eq!(contract["selection_probabilities"], json!([0.5, 0.5]));
                    assert_eq!(contract["schema"], "evolving-dimer-flexible-surrogate-v1");
                    assert_eq!(contract["proposal_rng_role"], "flexible_surrogate/proposal");
                    assert!(actual[0].get("surrogate_contract").is_none());
                } else {
                    assert_eq!(contract["handle"], 0);
                }
                assert_eq!(contract["cloud"], actual[0]["cloud"]);
                assert_eq!(contract["raw_count"], 2);
                assert_eq!(contract["points_per_body"], 1);
                assert_eq!(contract["point_volume"], 4.);
                assert_eq!(contract["effective_config"], json!(config));
                assert_eq!(actual[0]["prepared_start"].is_null(), init == "source");
                for block in 1..=8 {
                    let base = 1 + (block - 1) * 7;
                    for (attempt, slot) in [0, 1, 0, 1].into_iter().enumerate() {
                        assert_eq!(actual[base + attempt]["kind"], "local");
                        assert_eq!(actual[base + attempt]["member"], slot);
                    }
                    let begun = &actual[base + 4];
                    let row = &actual[base + 5];
                    let retained = &actual[base + 6];
                    assert_eq!(begun["kind"], begun_kind);
                    assert_eq!(begun["old"], row["old"]);
                    assert_eq!(row["kind"], outcome_kind);
                    assert_eq!(row["members"], json!([0, 1]));
                    if flexible {
                        assert!(row.get("handle").is_none());
                        assert_eq!(row["selection_probabilities"], json!([0.5, 0.5]));
                        assert_eq!(row["budget_before"]["raw"], begun["raw"]);
                        assert_eq!(row["budget_before"]["retained"], begun["bath_retained"]);
                        assert_eq!(row["budget_after"]["raw"], retained["raw"]);
                        assert_eq!(row["budget_after"]["retained"], retained["retained"]);
                        if row["status"] == "completed" {
                            assert_eq!(row["bath"]["legs"].as_array().unwrap().len(), 2);
                            let order = row["bath"]["ordered_members"].as_array().unwrap();
                            let first = order[0].as_u64().unwrap() as usize;
                            assert_eq!(order[1], 1 - first);
                            assert_eq!(
                                row["bath"]["intermediate_selected"][first],
                                row["proposed"][first]
                            );
                            assert_eq!(
                                row["bath"]["intermediate_selected"][1 - first],
                                row["old"][1 - first]
                            );
                            let bath = row["bath"]["aggregate"]["log_weight"].as_f64().unwrap();
                            let correction = row["complete_log_correction"].as_f64().unwrap();
                            assert_eq!(
                                row["log_acceptance_ratio"].as_f64().unwrap(),
                                bath + correction
                            );
                        }
                    } else {
                        assert_eq!(row["handle"], 0);
                    }
                    assert_eq!(row["config"], json!(config));
                    assert_eq!(row["steps"].as_array().unwrap().len(), config.inner_steps);
                    assert_eq!(row["retained"], retained["selected"]);
                    assert_eq!(retained["counts"]["local_attempted"], block * 4);
                    assert_eq!(retained["counts"]["dimer_attempted"], block);
                    assert_eq!(
                        row["physical_decisions"],
                        if row["status"] == "identity_self_loop" {
                            0
                        } else {
                            1
                        }
                    );
                    if row["accepted"] == false {
                        assert_eq!(row["retained"], row["old"]);
                    }
                    for step in row["steps"].as_array().unwrap() {
                        assert!(step["retained"].is_array());
                        if flexible {
                            let slot = step["selected_slot"].as_u64().unwrap() as usize;
                            assert!(slot < 2);
                            assert_eq!(step["selected_label"], slot);
                            assert_eq!(step["proposed"][1 - slot], step["old"][1 - slot]);
                            if step["accepted"] == false {
                                assert_eq!(step["retained"], step["old"]);
                            }
                        }
                    }
                    if arm == "rigid_surrogate_flat8" || arm == "flexible_flat8" {
                        assert_eq!(row["complete_log_correction"], 0.);
                        assert_eq!(row["old_score"]["log_surrogate"], 0.);
                    }
                }
            } else if arm == "two_root_m4" {
                let contract = &actual[0]["root_order_contract"];
                assert_eq!(contract["canonical_members"], json!([0, 1]));
                assert_eq!(contract["root_order_rng_role"], "two_root_m4/root_order");
                assert_eq!(
                    contract["internal"]["point_frame"],
                    "selected_mobile_root_body"
                );
                assert_eq!(contract["cloud"], actual[0]["cloud"]);
                for block in 1..=8 {
                    let base = 1 + (block - 1) * 6;
                    for (attempt, slot) in [0, 1, 0, 1].into_iter().enumerate() {
                        assert_eq!(actual[base + attempt]["member"], slot);
                    }
                    let row = &actual[base + 4];
                    let root = root_order(77, 0, init, 0, block);
                    assert_eq!(row["root_slot"], root);
                    assert_eq!(row["members"], json!([root, 1 - root]));
                    assert_eq!(row["retained"], actual[base + 5]["selected"]);
                    assert_eq!(row["proposal"]["root_guidance"], Value::Null);
                    assert_eq!(row["proposal"]["guidance"]["m"], 4);
                }
            } else {
                let contract = &actual[0]["guidance_contract"];
                assert_eq!(
                    contract["root"]["threshold_rng_role"],
                    "root_m4/root_threshold"
                );
                assert_eq!(contract["internal"]["threshold_rng_role"], "m4/threshold");
                assert_eq!(contract["root"]["point_frame"], "fixed_anchor_body");
                assert_eq!(contract["internal"]["point_frame"], "mobile_root_body");
                assert_eq!(contract["cloud"], actual[0]["cloud"]);
                assert!(
                    actual.iter().any(
                        |row| row["kind"] == "factorized_dimer" && row["status"] == "completed"
                    )
                );
                for row in actual
                    .iter()
                    .filter(|row| row["kind"] == "factorized_dimer")
                {
                    assert_eq!(row["proposal"]["root_guidance"]["point_count"], 1);
                    assert_eq!(row["proposal"]["guidance"]["point_count"], 1);
                    assert_eq!(row["proposal"]["root_guidance"]["m"], 4);
                    assert_eq!(row["proposal"]["guidance"]["m"], 4);
                }
            }
            let terminal =
                |path: &Path| -> Result<Value> { Ok(serde_json::from_slice(&fs::read(path)?)?) };
            let a = terminal(&full_out.join("terminal.json"))?;
            let b = terminal(&split_out.join("terminal.json"))?;
            for key in ["counts", "raw", "retained", "blocks", "complete", "job"] {
                assert_eq!(a[key], b[key]);
            }
            if any_surrogate {
                // At the same initial state the shared four local updates have
                // exactly the old local-only control's records and RNG draws.
                let mut control_plan = plan_for(dir.join(format!("{init}-local-control")));
                control_plan.as_object_mut().unwrap().remove(policy_key);
                control_plan["jobs"][0]["arm"] = json!("local");
                control_plan["allocation"]["warmup_blocks"] = json!(0);
                control_plan["allocation"]["production_blocks"] = json!(1);
                let control_path = dir.join(format!("{init}-local-control.json"));
                save(&control_path, &control_plan)?;
                run(
                    &make_args(control_path, None, None),
                    &control_plan,
                    &binding,
                    &geometry,
                )?;
                let control =
                    rows(&dir.join(format!("{init}-local-control/job-000/trajectory.jsonl")))?;
                assert_eq!(&actual[1..5], &control[1..5]);
            }
            if (arm == "rigid_surrogate_flat8" || arm == "flexible_flat8") && init == "source" {
                let tail_plan = plan_for(dir.join("begun-tail"));
                let tail_config = dir.join("begun-tail.json");
                save(&tail_config, &tail_plan)?;
                run(
                    &make_args(tail_config.clone(), Some(1), None),
                    &tail_plan,
                    &binding,
                    &geometry,
                )?;
                let tail_output = dir.join("begun-tail/job-000");
                let tail_checkpoint = tail_output.join("checkpoint.json");
                let tail_journal = tail_output.join("trajectory.jsonl");
                let cp: Value = serde_json::from_slice(&fs::read(&tail_checkpoint)?)?;
                let cp_hash = hash_file(&tail_checkpoint)?;
                let mut journal = Journal::resume(
                    &tail_journal,
                    cp["journal_bytes"].as_u64().unwrap(),
                    cp["journal_rows"].as_u64().unwrap(),
                    cp["journal_sha256"].as_str().unwrap(),
                )?;
                journal.durable_line(&json!({"kind":begun_kind,
                    "block":2,"status":"begun","old":cp["selected"]}))?;
                drop(journal);
                let tail_hash = hash_file(&tail_journal)?;
                assert!(
                    run(
                        &make_args(tail_config, None, Some(tail_checkpoint.clone())),
                        &tail_plan,
                        &binding,
                        &geometry
                    )
                    .is_err()
                );
                assert_eq!(hash_file(&tail_checkpoint)?, cp_hash);
                assert_eq!(hash_file(&tail_journal)?, tail_hash);
                assert!(tail_output.join("failure.json").exists());
                assert!(!tail_output.join("terminal.json").exists());
                for scenario in ["hard_rejected", "identity", "fatal_bath"] {
                    let control_output = dir.join(scenario);
                    let mut control_plan = plan_for(control_output.clone());
                    control_plan["local"] =
                        json!({"translation_std_A":0.,"rotation_std_degrees":0.});
                    control_plan[policy_key]["proposal_scales"] = json!({
                        "source":"frozen_override","rotation_std_degrees":0.,
                        "translation_std_A":match scenario {"hard_rejected"=>1e6,"identity"=>0.,_=>0.1}});
                    control_plan["allocation"]["warmup_blocks"] = json!(0);
                    control_plan["allocation"]["production_blocks"] = json!(1);
                    if scenario == "fatal_bath" {
                        control_plan["physical"]["activity"] = json!(100.);
                        for key in [
                            "raw_per_leg",
                            "raw_per_outer",
                            "raw_campaign",
                            "retained_per_leg",
                            "retained_per_outer",
                            "retained_campaign",
                        ] {
                            control_plan["limits"][key] = json!(0);
                        }
                    }
                    let config_path = dir.join(format!("{scenario}.json"));
                    save(&config_path, &control_plan)?;
                    let result = run(
                        &make_args(config_path.clone(), None, None),
                        &control_plan,
                        &binding,
                        &geometry,
                    );
                    let output = control_output.join("job-000");
                    let records = rows(&output.join("trajectory.jsonl"))?;
                    assert_eq!(records[5]["kind"], begun_kind);
                    let outcome = &records[6];
                    assert_eq!(outcome["kind"], outcome_kind);
                    assert_eq!(outcome["old"], records[5]["old"]);
                    assert_eq!(outcome["retained"], outcome["old"]);
                    assert_eq!(outcome["accepted"], false);
                    assert_eq!(outcome["physical_decisions"], 0);
                    assert_eq!(outcome["steps"].as_array().unwrap().len(), 8);
                    if scenario == "fatal_bath" {
                        assert!(result.is_err());
                        assert_eq!(records.len(), 7);
                        assert_eq!(outcome["status"], "fatal");
                        assert!(outcome["fatal_error"].is_string());
                        assert!(
                            outcome["bath_failure"]["failed_progress"]["gate"]["raw_points"]
                                .as_u64()
                                .unwrap()
                                > 0
                        );
                        assert!(!output.join("checkpoint.json").exists());
                        assert!(!output.join("terminal.json").exists());
                        let failure_hash = hash_file(&output.join("failure.json"))?;
                        assert!(
                            run(
                                &make_args(config_path, None, Some(output.join("checkpoint.json"))),
                                &control_plan,
                                &binding,
                                &geometry
                            )
                            .is_err()
                        );
                        assert_eq!(hash_file(&output.join("failure.json"))?, failure_hash);
                    } else {
                        result?;
                        assert_eq!(records.len(), 8);
                        assert_eq!(outcome["status"], "identity_self_loop");
                        assert_eq!(records[7]["counts"]["dimer_attempted"], 1);
                        assert_eq!(records[7]["counts"]["dimer_self_loop"], 1);
                        for step in outcome["steps"].as_array().unwrap() {
                            assert_eq!(step["retained"], outcome["old"]);
                            if scenario == "hard_rejected" {
                                assert_eq!(step["status"], "hard_rejected");
                                assert_eq!(step["accepted"], false);
                            } else {
                                assert_eq!(step["status"], "completed");
                                assert_eq!(step["accepted"], true);
                            }
                        }
                    }
                }
            }
        }
        for (path, hash) in frozen {
            assert_eq!(hash_file(&path)?, hash);
        }
        fs::remove_dir_all(dir)?;
        Ok(())
    }
    #[test]
    fn singleton_policy_is_explicit_frozen_and_absent_on_existing_arms() -> Result<()> {
        let policy = synthetic_singleton_policy();
        for arm in ["singleton_two_neighbor", "singleton_two_neighbor_unfused"] {
            assert!(singleton_policy(&json!({}), arm).is_err());
            let accepted = singleton_policy(&json!({"singleton_policy":policy}), arm)?.unwrap();
            assert_eq!(
                accepted.oligomer_for(arm).multi_contact_mass,
                if arm == "singleton_two_neighbor" {
                    0.8
                } else {
                    0.
                }
            );
            for (field, value) in [
                ("trial_cap", json!(31)),
                ("uniform_probability", json!(0.4)),
                ("uniform_half_width", json!(159.)),
                ("member_schedule", json!("random")),
                ("schema", json!("other")),
                ("extra_field", json!(true)),
            ] {
                let mut changed = policy.clone();
                changed[field] = value;
                assert!(singleton_policy(&json!({"singleton_policy":changed}), arm).is_err());
            }
            let mut changed = policy.clone();
            changed["oligomer"]["max_components"] = json!(33);
            assert!(singleton_policy(&json!({"singleton_policy":changed}), arm).is_err());
            let mut changed = policy.clone();
            changed["oligomer"]["multi_contact_mass"] = json!(0.);
            assert!(singleton_policy(&json!({"singleton_policy":changed}), arm).is_err());
        }
        for arm in ["local", "unguided", "m4", "root_m4"] {
            assert!(singleton_policy(&json!({}), arm)?.is_none());
            assert!(singleton_policy(&json!({"singleton_policy":null}), arm).is_err());
            assert!(singleton_policy(&json!({"singleton_policy":policy}), arm).is_err());
            for role in ["proposal", "bath", "accept", "threshold", "root_threshold"] {
                assert_eq!(extra_role(arm, role), dimer_role(arm, role));
            }
        }
        Ok(())
    }
    fn synthetic_singleton_policy() -> Value {
        json!({"schema":"two-neighbor-singleton-policy-v1","uniform_half_width":160.,
            "uniform_probability":0.5,"trial_cap":32,"member_schedule":"alternating_0_first",
            "oligomer":OligomerConfig::default()})
    }
    #[test]
    fn singleton_arms_share_named_streams_and_alternate_from_first_member() {
        for init in ["source", "proposal_prepared"] {
            for block in 1..=8 {
                assert_eq!((block - 1) % 2, usize::from(block % 2 == 0));
                for role in ["proposal", "bath", "accept"] {
                    let a = extra_role("singleton_two_neighbor", role);
                    let b = extra_role("singleton_two_neighbor_unfused", role);
                    assert_eq!(a, format!("singleton_two_neighbor/{role}"));
                    assert_eq!(a, b);
                    assert_eq!(
                        rng(77, 0, init, 0, block, &a).random::<u64>(),
                        rng(77, 0, init, 0, block, &b).random::<u64>()
                    );
                    assert_ne!(
                        rng(77, 0, init, 0, block, &a).random::<u64>(),
                        rng(77, 0, init, 0, block, &format!("local/0/{role}")).random::<u64>()
                    );
                }
            }
        }
    }
    #[test]
    fn singleton_both_arms_and_starts_resume_without_loading_guidance_clouds() -> Result<()> {
        let dir =
            std::env::temp_dir().join(format!("singleton-arms-restart-{}", std::process::id()));
        fs::create_dir(&dir)?;
        let geometry = synthetic_geometry()?;
        let pose = |x| Pose {
            position: [x, 0., 0.],
            orientation: [1., 0., 0., 0.],
        };
        let start_path = dir.join("frozen-start.json");
        save(&start_path, &json!({"selected":[pose(0.62),pose(1.23)]}))?;
        let ledger_path = dir.join("frozen-start-attempts.jsonl");
        fs::write(&ledger_path, b"{\"synthetic_preparation\":true}\n")?;
        let manifest_path = dir.join("original-prepared-manifest.json");
        let unreadable_cloud =
            json!({"path":dir.join("cloud-must-not-be-opened"),"sha256":"0".repeat(64)});
        save(
            &manifest_path,
            &json!({"complete":true,"passed":true,
            "files":[unreadable_cloud],"cloud_banks":"must-not-be-inspected",
            "alternative_starts":[{"context_index":0,"stream":0,"record":BoundFile::make(&start_path)?,
                "ledger":BoundFile::make(&ledger_path)?}]}),
        )?;
        let binding = json!({"prepared_manifest":BoundFile::make(&manifest_path)?});
        let binding_path = dir.join("binding.json");
        save(&binding_path, &binding)?;
        let frozen: Vec<_> = [&start_path, &ledger_path, &manifest_path]
            .into_iter()
            .map(|p| (p.clone(), hash_file(p).unwrap()))
            .collect();
        for arm in ["singleton_two_neighbor", "singleton_two_neighbor_unfused"] {
            for init in ["source", "proposal_prepared"] {
                let label = format!("{arm}-{init}");
                let plan_for = |output: PathBuf| {
                    json!({
                    "master_seed":77,"output":output,"contexts":[{"root":0,"child":1,"anchor":2}],
                    "source_frame":{"synthetic":"frozen_source"},
                    "physical":{"wall_radius":50.,"depletant_radius":0.8,"activity":0.05,"lambda_ratio":64.},
                    "singleton_policy":synthetic_singleton_policy(),
                    "local":{"translation_std_A":0.02,"rotation_std_degrees":1.},
                    "envelope":{"max_cells":63,"max_depth":6,"min_width":0.},
                    "limits":{"raw_per_leg":1000000,"raw_per_outer":2000000,"raw_campaign":20000000,
                        "retained_per_leg":1000000,"retained_per_outer":2000000,"retained_campaign":20000000,
                        "cpu_seconds":60.},
                    "allocation":{"warmup_blocks":2,"production_blocks":6},
                    "jobs":[{"id":0,"context_index":0,"stream":0,"initialization":init,"arm":arm}]})
                };
                let full_plan = plan_for(dir.join(format!("{label}-full")));
                let split_plan = plan_for(dir.join(format!("{label}-split")));
                let full_config = dir.join(format!("{label}-full.json"));
                let split_config = dir.join(format!("{label}-split.json"));
                save(&full_config, &full_plan)?;
                save(&split_config, &split_plan)?;
                let args = |config, stop_after_block, resume| Args {
                    config,
                    binding: binding_path.clone(),
                    mode: "run".into(),
                    job: Some(0),
                    stop_after_block,
                    resume,
                };
                run(
                    &args(full_config, None, None),
                    &full_plan,
                    &binding,
                    &geometry,
                )?;
                run(
                    &args(split_config.clone(), Some(3), None),
                    &split_plan,
                    &binding,
                    &geometry,
                )?;
                let split_out = dir.join(format!("{label}-split/job-000"));
                let checkpoint_path = split_out.join("checkpoint.json");
                let checkpoint: Value = serde_json::from_slice(&fs::read(&checkpoint_path)?)?;
                assert_eq!(checkpoint["block"], 3);
                assert_eq!(checkpoint["journal_rows"], 19);
                run(
                    &args(split_config, None, Some(checkpoint_path)),
                    &split_plan,
                    &binding,
                    &geometry,
                )?;
                let full_out = dir.join(format!("{label}-full/job-000"));
                let rows = |output: &Path| -> Result<Vec<Value>> {
                    fs::read_to_string(output.join("trajectory.jsonl"))?
                        .lines()
                        .map(|line| {
                            let mut row: Value = serde_json::from_str(line)?;
                            row.as_object_mut().unwrap().remove("sampler_cpu_seconds");
                            Ok(row)
                        })
                        .collect()
                };
                let actual = rows(&split_out)?;
                assert_eq!(actual, rows(&full_out)?);
                assert_eq!(actual.len(), 49);
                let initial = &actual[0];
                assert!(
                    initial.get("cloud").is_none() && initial.get("guidance_contract").is_none()
                );
                assert_eq!(initial["singleton_contract"]["guidance_cloud_used"], false);
                assert_eq!(
                    initial["singleton_contract"]["effective_oligomer"]["multi_contact_mass"],
                    if arm == "singleton_two_neighbor" {
                        json!(0.8)
                    } else {
                        json!(0.)
                    }
                );
                assert_eq!(initial["prepared_start"].is_null(), init == "source");
                assert_eq!(
                    initial["selected"],
                    if init == "source" {
                        json!([pose(0.6), pose(1.2)])
                    } else {
                        json!([pose(0.62), pose(1.23)])
                    }
                );
                for block in 1..=8 {
                    let base = 1 + (block - 1) * 6;
                    for (attempt, slot) in [0, 1, 0, 1].into_iter().enumerate() {
                        assert_eq!(actual[base + attempt]["kind"], "local");
                        assert_eq!(actual[base + attempt]["member"], slot);
                    }
                    let extra = &actual[base + 4];
                    assert_eq!(extra["kind"], "two_neighbor_singleton");
                    assert_eq!(extra["member_slot"], (block - 1) % 2);
                    assert_eq!(extra["member"], (block - 1) % 2);
                    assert_eq!(extra["neighbors"], json!([1 - (block - 1) % 2, 2]));
                    assert_eq!(actual[base + 5]["counts"]["local_attempted"], block * 4);
                    assert_eq!(actual[base + 5]["counts"]["singleton_attempted"], block);
                    assert!(actual[base + 5]["counts"].get("dimer_attempted").is_none());
                }
                let a: Value = serde_json::from_slice(&fs::read(full_out.join("terminal.json"))?)?;
                let b: Value = serde_json::from_slice(&fs::read(split_out.join("terminal.json"))?)?;
                for key in ["counts", "raw", "retained", "blocks", "complete", "job"] {
                    assert_eq!(a[key], b[key]);
                }
            }
        }
        for (path, hash) in frozen {
            assert_eq!(hash_file(&path)?, hash);
        }
        fs::remove_dir_all(dir)?;
        Ok(())
    }
    #[test]
    fn journal_rejects_uncheckpointed_tail() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("evolving-journal-{}", std::process::id()));
        fs::create_dir_all(&dir)?;
        let path = dir.join("journal.jsonl");
        let mut j = Journal::new(&path)?;
        j.line(&json!({"x":1}))?;
        let (bytes, rows, sha) = (j.bytes, j.rows, j.sha());
        drop(j);
        let mut j = Journal::resume(&path, bytes, rows, &sha)?;
        j.line(&json!({"x":2}))?;
        drop(j);
        assert!(Journal::resume(&path, bytes, rows, &sha).is_err());
        fs::remove_dir_all(dir)?;
        Ok(())
    }
}
