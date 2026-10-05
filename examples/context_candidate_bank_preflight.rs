//! Saved rho=0 proposal-bank preflight. No RNG, clouds, proposals, or MC.
//! Density is with respect to translation volume times normalized rotational Haar.
#[path = "support/context_bank_geometry.rs"]
mod patch_geometry;
use anyhow::{Context, Result, ensure};
use clap::Parser;
use patch_geometry::{Limits, PatchBvh, Token, Work, World, near};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};
use tetramer_mc::{
    context_docking::{ContextBody, ContextDockingConfig, uniform_half_width},
    depletion::GateOptions,
    docking::{self, DockingMethod, DockingProposal},
    geometry::{Environment, Placed, Shape, SphereTree},
    math::*,
    overlap_weight::OverlapEnvelope,
    proposal::{FrozenRelativePoseProposal, GaussianComponentParameters, RelativePoseBranch},
    simulation::cpu_seconds,
    spherical::Container,
};
const SCHEMA: &str = "context-candidate-bank-preflight-v1";
#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    bank: PathBuf,
    #[arg(long)]
    out: PathBuf,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Asset {
    path: PathBuf,
    sha256: String,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Regions {
    a_neighbors: Vec<usize>,
    b_neighbors: Vec<usize>,
    secondary_label: usize,
    source_secondary_tokens: Vec<Token>,
    inclusion_boundaries: Vec<f64>,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Config {
    schema: String,
    invocation_config: Asset,
    model: Asset,
    prior: Option<Asset>,
    patch_map: Asset,
    bank_sha256: String,
    identity: Value,
    expected_records: usize,
    expected_virtual_branches: usize,
    regions: Regions,
    envelope: GateOptions,
    limits: Limits,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BankRow {
    ordinal: usize,
    cycle: u64,
    slot: usize,
    attempt_index: u64,
    event_index: u64,
    identity: Value,
    proposed_pose: Pose,
    branch: String,
    status: String,
    wall_valid: bool,
    core_valid: Option<bool>,
    saved_log_g: Option<f64>,
    production: bool,
    accepted: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FixedContext {
    schema: String,
    anchor_label: usize,
    excluded_moving_labels: Vec<usize>,
    bodies: Vec<ContextBody>,
}
#[derive(Deserialize)]
struct SourceState {
    schema: String,
    moving_label: usize,
    pose: Pose,
    anchor_label: usize,
    anchor_pose: Pose,
    boundary: String,
    spherical_wall_radius: f64,
    coordinate_frame: String,
    endpoint_sha256: String,
}
#[derive(Deserialize)]
struct PatchMap {
    schema: String,
    shape_sha256: String,
    atom_patch_ids: Vec<String>,
}
#[derive(Deserialize)]
struct PriorBranch {
    virtual_label: usize,
    component_index: usize,
    inverted: bool,
    original_probability: f64,
    original_log_probability: f64,
    eligible: bool,
    log_probability: f64,
    probability: f64,
}
#[derive(Deserialize)]
struct PriorAsset {
    schema: String,
    complete: bool,
    passed: bool,
    method: String,
    moving_labels: Vec<usize>,
    anchor_label: usize,
    floor_probability: f64,
    virtual_branch_count: usize,
    branches: Vec<PriorBranch>,
    log_prior: Vec<f64>,
    base_log_prior: Vec<f64>,
    eligibility: Vec<bool>,
    input_sha256: BTreeMap<String, String>,
    fixed_context_sha256: String,
    source_state_sha256: String,
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn checked_bytes(path: &Path, expected: &str) -> Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    ensure!(sha(&bytes) == expected, "Hash mismatch: {}", path.display());
    Ok(bytes)
}
fn asset(a: &Asset) -> Result<Vec<u8>> {
    ensure!(a.path.is_absolute(), "Asset paths must be absolute");
    checked_bytes(&a.path, &a.sha256)
}
fn json_file(path: &Path, value: &Value) -> Result<()> {
    let mut f = OpenOptions::new().write(true).create_new(true).open(path)?;
    serde_json::to_writer_pretty(&mut f, value)?;
    f.write_all(b"\n")?;
    f.flush()?;
    Ok(())
}
struct Journal {
    file: BufWriter<File>,
    next: u64,
}
impl Journal {
    fn new(path: &Path) -> Result<Self> {
        Ok(Self {
            file: BufWriter::new(OpenOptions::new().write(true).create_new(true).open(path)?),
            next: 0,
        })
    }
    fn emit(&mut self, mut v: Value) -> Result<()> {
        v["event_index"] = json!(self.next);
        serde_json::to_writer(&mut self.file, &v)?;
        self.file.write_all(b"\n")?;
        self.file.flush()?;
        self.next += 1;
        Ok(())
    }
}
fn close(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= 2e-12 * (1. + b.abs())
}
fn prior_validate(
    a: &PriorAsset,
    b: &[RelativePoseBranch],
    cfg: &ContextDockingConfig,
    c: &FixedContext,
    s: &SourceState,
) -> Result<()> {
    ensure!(
        a.schema == "fixed-context-virtual-branch-prior-v1"
            && a.complete
            && a.passed
            && a.method == "posterior_involution",
        "Invalid prior schema/status"
    );
    ensure!(
        a.moving_labels == c.excluded_moving_labels
            && a.anchor_label == c.anchor_label
            && a.fixed_context_sha256 == cfg.expected_sha256.fixed_context
            && a.source_state_sha256 == cfg.expected_sha256.source_state,
        "Prior context mismatch"
    );
    ensure!(
        a.input_sha256.get("model") == Some(&cfg.expected_sha256.model)
            && a.input_sha256.get("shape") == Some(&cfg.expected_sha256.shape)
            && a.input_sha256.get("endpoint") == Some(&s.endpoint_sha256),
        "Prior input mismatch"
    );
    let n = b.len();
    ensure!(
        a.virtual_branch_count == n
            && a.branches.len() == n
            && a.log_prior.len() == n
            && a.base_log_prior.len() == n
            && a.eligibility.len() == n,
        "Prior inventory mismatch"
    );
    for (i, (x, y)) in b.iter().zip(&a.branches).enumerate() {
        ensure!(
            y.virtual_label == i
                && y.component_index == x.component_index
                && y.inverted == x.inverted
                && close(y.original_probability, x.weight)
                && close(y.original_log_probability, x.weight.ln())
                && close(a.base_log_prior[i], x.weight.ln())
                && y.eligible == a.eligibility[i]
                && close(y.log_probability, a.log_prior[i])
                && close(y.probability, a.log_prior[i].exp()),
            "Prior record mismatch {i}"
        );
    }
    let r = docking::defensive_virtual_branch_log_prior(
        &a.base_log_prior,
        &a.eligibility,
        a.floor_probability,
    )?;
    ensure!(
        r.iter().zip(&a.log_prior).all(|(x, y)| close(*x, *y)),
        "Prior rule mismatch"
    );
    Ok(())
}
fn relative(pose: Pose, anchor: Pose) -> Pose {
    let rt = transpose(rotation(anchor.orientation));
    Pose {
        position: matvec(rt, sub(pose.position, anchor.position)),
        orientation: quaternion(matmul(rt, rotation(pose.orientation))),
    }
}
fn log_add(a: f64, b: f64) -> Result<f64> {
    ensure!(
        (a.is_finite() || a == f64::NEG_INFINITY) && (b.is_finite() || b == f64::NEG_INFINITY),
        "Invalid log density"
    );
    let m = a.max(b);
    Ok(if m == f64::NEG_INFINITY {
        m
    } else {
        m + ((a - m).exp() + (b - m).exp()).ln()
    })
}
fn learned(proposal: &DockingProposal, pose: Pose, anchor: Pose) -> Result<f64> {
    pose.validate()?;
    let r = relative(pose, anchor);
    let (map, flags, prior) = proposal.member_chart_parts();
    let mut g = f64::NEG_INFINITY;
    for i in 0..flags.len() {
        let p = if flags[i] { invert_relative_pose(r) } else { r };
        g = log_add(g, prior[i] + map.checked_log_density(i, p)?)?;
    }
    Ok(g)
}
fn density(
    proposal: &DockingProposal,
    pose: Pose,
    anchor: Pose,
    half: f64,
    saved: Option<f64>,
) -> Result<Value> {
    let g = learned(proposal, pose, anchor)?;
    let inside = pose.position.iter().all(|x| x.abs() <= half);
    let u = if inside {
        -3. * (2. * half).ln()
    } else {
        f64::NEG_INFINITY
    };
    let q = log_add(g, u)? - 2f64.ln();
    ensure!(
        q.is_finite(),
        "Saved generated candidate has zero/nonfinite full Q"
    );
    let delta = if let Some(v) = saved {
        ensure!(
            v.is_finite()
                && g.is_finite()
                && (v - g).abs() <= 2e-8 + 2e-10 * (1. + v.abs() + g.abs()),
            "Saved compiled-map log G mismatch"
        );
        Some(g - v)
    } else {
        None
    };
    Ok(
        json!({"log_g_status":if g.is_finite(){"finite"}else{"negative_infinity"},"log_g":g.is_finite().then_some(g),"log_u":u.is_finite().then_some(u),"uniform_contains":inside,"log_q":q,"saved_log_g_delta":delta}),
    )
}
/// Manifest-only arithmetic reconstruction; density above always uses the actual compiled map.
fn reconstructed_map_lower(p: &GaussianComponentParameters) -> Result<[[f64; 6]; 6]> {
    let mut l = [[0.; 6]; 6];
    for i in 0..6 {
        for j in 0..=i {
            let residual = 0.5 * (p.covariance[i][j] + p.covariance[j][i])
                - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            l[i][j] = if i == j {
                ensure!(
                    residual > 0. && residual.is_finite(),
                    "Manifest covariance failure"
                );
                residual.sqrt()
            } else {
                residual / l[j][j]
            };
        }
    }
    Ok(l)
}
fn validate_row(row: &BankRow, ordinal: usize, previous: Option<u64>, cfg: &Config) -> Result<()> {
    ensure!(
        row.ordinal == ordinal
            && row.cycle == ordinal as u64 + 1
            && row.slot == 4
            && row.attempt_index == 5 * (row.cycle - 1) + 4
            && row.identity == cfg.identity
            && row.production == (row.cycle > 256),
        "Candidate inventory mismatch at {ordinal}"
    );
    ensure!(
        row.event_index == 17 + 16 * ordinal as u64 && previous.is_none_or(|v| row.event_index > v),
        "Event ordering mismatch"
    );
    row.proposed_pose.validate()?;
    ensure!(
        matches!(row.branch.as_str(), "uniform" | "involution")
            && (row.branch == "uniform") == row.saved_log_g.is_none(),
        "Proposal branch/log G mismatch"
    );
    ensure!(
        row.wall_valid == row.core_valid.is_some(),
        "Saved early-exit verdict mismatch"
    );
    ensure!(
        matches!(
            row.status.as_str(),
            "accepted" | "bath_rejected" | "wall_rejected" | "core_rejected"
        ),
        "Unknown candidate disposition"
    );
    ensure!(
        row.accepted == (row.status == "accepted")
            && (row.status == "wall_rejected") == !row.wall_valid
            && (row.status == "core_rejected") == (row.core_valid == Some(false)),
        "Inconsistent disposition"
    );
    Ok(())
}
struct Geometry {
    tree: SphereTree,
    contact: SphereTree,
    wall: Container,
    patch: PatchBvh,
    worlds: Vec<World>,
    fixed: Vec<Placed>,
    context: FixedContext,
    source: SourceState,
    rd: f64,
}
impl Geometry {
    fn validity(&self, pose: Pose, work: &Work) -> Result<(bool, Option<bool>)> {
        work.check()?;
        let wall = self.wall.contains(pose);
        if !wall {
            return Ok((false, None));
        }
        let moving = Placed::new(pose);
        for (i, p) in self.fixed.iter().enumerate() {
            if i % 32 == 0 {
                work.check()?;
            }
            if self.tree.overlaps(&moving, p) {
                return Ok((true, Some(false)));
            }
        }
        Ok((true, Some(true)))
    }
    fn patches(
        &self,
        pose: Pose,
        work: &mut Work,
    ) -> Result<(BTreeSet<Token>, Vec<usize>, Vec<usize>)> {
        let world = self.patch.placed(pose)?;
        let placed = Placed::new(pose);
        let mut tokens = BTreeSet::new();
        let mut neighbors = Vec::new();
        let mut near_indices = Vec::new();
        for (i, b) in self.context.bodies.iter().enumerate() {
            work.check()?;
            if !near(pose, b.pose, self.tree.bound, self.rd) {
                continue;
            }
            near_indices.push(i);
            let found = if b.label < self.source.moving_label {
                self.patch.contacts(
                    &self.worlds[i],
                    &world,
                    b.label,
                    self.source.moving_label,
                    self.rd,
                    work,
                )?
            } else {
                self.patch.contacts(
                    &world,
                    &self.worlds[i],
                    self.source.moving_label,
                    b.label,
                    self.rd,
                    work,
                )?
            };
            let exact = self.contact.overlaps(&placed, &self.fixed[i]);
            ensure!(
                exact == !found.is_empty(),
                "Patch BVH/production inflated-tree disagreement at label {}",
                b.label
            );
            if exact {
                neighbors.push(b.label);
            }
            tokens.extend(found);
        }
        Ok((tokens, neighbors, near_indices))
    }
}
fn region_data(
    tokens: &BTreeSet<Token>,
    neighbors: &[usize],
    r: &Regions,
) -> Result<(String, Value)> {
    let source: BTreeSet<_> = r.source_secondary_tokens.iter().cloned().collect();
    let secondary: BTreeSet<_> = tokens
        .iter()
        .filter(|t| t.0 == r.secondary_label || t.1 == r.secondary_label)
        .cloned()
        .collect();
    let n = source.intersection(&secondary).count();
    let union = source.union(&secondary).count();
    let fraction = n as f64 / source.len() as f64;
    let name = if neighbors == r.a_neighbors {
        if n == source.len() {
            "A_patch_complete"
        } else if fraction < 0.25 {
            "A_patch_0_0.25"
        } else if fraction < 0.5 {
            "A_patch_0.25_0.5"
        } else if fraction < 0.75 {
            "A_patch_0.5_0.75"
        } else {
            "A_patch_0.75_1"
        }
    } else if neighbors == r.b_neighbors {
        "B"
    } else if neighbors.is_empty() {
        "unbound"
    } else {
        "other_contact"
    };
    let fingerprint = sha(&serde_json::to_vec(tokens)?);
    Ok((
        name.into(),
        json!({"tokens":tokens,"neighbor_labels":neighbors,"fingerprint":fingerprint,"secondary_tokens":secondary,"source_intersection":n,"source_union":union,"source_fraction":fraction,"jaccard":n as f64/union as f64}),
    ))
}
fn run(
    args: &Args,
    events: &mut Journal,
    rows: &mut BufWriter<File>,
    completed: &mut usize,
    work: &mut Work,
) -> Result<Value> {
    let cfg_bytes = fs::read(&args.config)?;
    let cfg: Config = serde_json::from_slice(&cfg_bytes)?;
    ensure!(
        cfg.schema == SCHEMA
            && cfg.expected_records == 2304
            && cfg.expected_virtual_branches == 2048,
        "Wrong preflight allocation"
    );
    ensure!(
        matches!(cfg.identity["arm"].as_str(), Some("original" | "context"))
            && cfg.prior.is_some() == (cfg.identity["arm"] == "context"),
        "Prior presence differs from arm"
    );
    cfg.envelope.validate()?;
    ensure!(
        cfg.envelope.max_cells == 255
            && cfg.envelope.max_depth == 8
            && cfg.envelope.min_width == 0.,
        "Envelope settings differ from frozen plan"
    );
    ensure!(
        cfg.regions.a_neighbors == [16, 217]
            && cfg.regions.b_neighbors == [16, 56]
            && cfg.regions.secondary_label == 217
            && cfg.regions.inclusion_boundaries == [0., 0.25, 0.5, 0.75, 1.],
        "Region definition differs"
    );
    let ts: BTreeSet<_> = cfg
        .regions
        .source_secondary_tokens
        .iter()
        .cloned()
        .collect();
    ensure!(
        ts.len() == 16
            && ts.len() == cfg.regions.source_secondary_tokens.len()
            && ts.iter().all(|t| t.0 == 77 && t.1 == 217),
        "Source patch reference differs"
    );
    let invocation_bytes = asset(&cfg.invocation_config)?;
    let invocation: ContextDockingConfig = serde_json::from_slice(&invocation_bytes)?;
    ensure!(
        invocation.identity == cfg.identity
            && invocation.uniform_probability == 0.5
            && invocation.depletant_radius == 1.5
            && invocation.reservoir_density == 0.035
            && invocation.poisson_lambda_ratio == 64.
            && invocation.warmup_cycles == 256
            && invocation.local_attempts_per_cycle == 4,
        "Physical/identity schedule differs"
    );
    ensure!(
        cfg.model.sha256 == invocation.expected_sha256.model,
        "Model binding mismatch"
    );
    let base = cfg
        .invocation_config
        .path
        .parent()
        .context("Invocation parent absent")?;
    let resolve = |p: &Path| {
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            base.join(p)
        }
    };
    let shape_bytes = checked_bytes(
        &resolve(&invocation.shape),
        &invocation.expected_sha256.shape,
    )?;
    let shape: Shape = serde_json::from_slice(&shape_bytes)?;
    ensure!(shape.atoms.len() == 4004, "Frozen atomic inventory differs");
    let context_bytes = checked_bytes(
        &resolve(&invocation.fixed_context),
        &invocation.expected_sha256.fixed_context,
    )?;
    let context: FixedContext = serde_json::from_slice(&context_bytes)?;
    let source_bytes = checked_bytes(
        &resolve(&invocation.source_state),
        &invocation.expected_sha256.source_state,
    )?;
    let source: SourceState = serde_json::from_slice(&source_bytes)?;
    ensure!(
        context.schema == "fixed-outside-context-v1"
            && context.anchor_label == 16
            && context.excluded_moving_labels == [77]
            && context.bodies.len() == 263
            && invocation.expected_fixed_body_count == 263,
        "Wrong fixed context"
    );
    ensure!(
        context
            .bodies
            .iter()
            .map(|b| b.label)
            .eq((0..264).filter(|i| *i != 77)),
        "Fixed labels/order mismatch"
    );
    for b in &context.bodies {
        b.pose.validate()?;
    }
    ensure!(
        source.schema == "saved-source-state-metadata-v1"
            && source.moving_label == 77
            && source.anchor_label == 16
            && source.boundary == "spherical"
            && source.coordinate_frame
                == "Saved spherical-center frame; no display offset, wrapping or pose transform.",
        "Source frame mismatch"
    );
    source.pose.validate()?;
    source.anchor_pose.validate()?;
    let anchor_index = context
        .bodies
        .iter()
        .position(|b| b.label == 16)
        .context("Missing anchor")?;
    let anchor = context.bodies[anchor_index].pose;
    ensure!(anchor == source.anchor_pose, "Source anchor differs");
    let patch_bytes = asset(&cfg.patch_map)?;
    let patch_map: PatchMap = serde_json::from_slice(&patch_bytes)?;
    ensure!(
        patch_map.schema == "body-frame-atom-patch-map-v1"
            && patch_map.shape_sha256 == invocation.expected_sha256.shape,
        "Patch binding mismatch"
    );
    let tree = SphereTree::new(shape.clone())?;
    let wall = Container::new(source.spherical_wall_radius, &tree)?;
    let half = uniform_half_width(source.spherical_wall_radius, tree.bound)?;
    let model_bytes = asset(&cfg.model)?;
    let model = FrozenRelativePoseProposal::from_json_str_open(
        std::str::from_utf8(&model_bytes)?,
        [2. * half; 3],
        0.5,
        &invocation.expected_sha256.shape,
    )?;
    let branches = model.virtual_branches();
    let parameters = model.component_parameters();
    ensure!(
        branches.len() == cfg.expected_virtual_branches,
        "Virtual branch inventory mismatch"
    );
    let mut proposer =
        DockingProposal::new(model, DockingMethod::PosteriorInvolution, 0., [0.; 3])?
            .with_anchor_index(Some(anchor_index));
    if let Some(p) = &cfg.prior {
        let a: PriorAsset = serde_json::from_slice(&asset(p)?)?;
        prior_validate(&a, &branches, &invocation, &context, &source)?;
        proposer = proposer.with_virtual_branch_log_prior(&a.log_prior)?;
    }
    let (_, flags, priors) = proposer.member_chart_parts();
    ensure!(
        flags
            .iter()
            .copied()
            .eq(branches.iter().map(|b| b.inverted)),
        "Virtual reciprocal order mismatch"
    );
    let charts:Vec<_>=branches.iter().enumerate().map(|(i,b)|{let mut p=parameters[b.component_index].clone();p.weight=b.weight;Ok(json!({"virtual_label":i,"component_index":b.component_index,"inverted":b.inverted,"original_probability":b.weight,"effective_log_prior":priors[i],"parameters":p,"reconstructed_map_lower":reconstructed_map_lower(&p)?}))}).collect::<Result<_>>()?;
    json_file(
        &args.out.join("chart-manifest.json"),
        &json!({"schema":"context-candidate-chart-manifest-v1","angular_length":proposer.angular_length(),"uniform_half_width":half,"uniform_probability":0.5,"density_measure":"translation volume times normalized rotational Haar","anchor_pose":anchor,"charts":charts,"lower_scope":"Arithmetic reconstruction of private compiled map factors; every density uses actual checked_log_density, independently audited from model/source."}),
    )?;
    let patch = PatchBvh::new(&shape, &patch_map.atom_patch_ids)?;
    let worlds = context
        .bodies
        .iter()
        .map(|b| patch.placed(b.pose))
        .collect::<Result<Vec<_>>>()?;
    let mut inflated = shape;
    for a in &mut inflated.atoms {
        a.radius += invocation.depletant_radius;
    }
    let contact = SphereTree::new(inflated)?;
    let fixed = context.bodies.iter().map(|b| Placed::new(b.pose)).collect();
    let geo = Geometry {
        tree,
        contact,
        wall,
        patch,
        worlds,
        fixed,
        context,
        source,
        rd: invocation.depletant_radius,
    };
    let bank = checked_bytes(&args.bank, &cfg.bank_sha256)?;
    ensure!(bank.last() == Some(&b'\n'), "Bank must end with newline");
    json_file(
        &args.out.join("protocol.json"),
        &json!({"schema":SCHEMA,"config":cfg,"config_sha256":sha(&cfg_bytes),"source_bundle_sha256":sha(include_str!(concat!(env!("OUT_DIR"), "/source-bundle.json")).as_bytes()),"source_pose":geo.source.pose,"fixed_body_count":263,"patch_node_count":geo.patch.node_count(),"no_poisson_clouds":true,"density_saved_tolerance":{"absolute":2e-8,"relative_scale":2e-10},"scope":"Retrospective candidate geometry and exact normalized proposal-density audit; no physical weights or coverage guarantee."}),
    )?;
    events.emit(json!({"kind":"source_begun"}))?;
    work.begin()?;
    ensure!(
        geo.validity(geo.source.pose, work)? == (true, Some(true)),
        "Certified original source invalid"
    );
    let (tokens, neighbors, _) = geo.patches(geo.source.pose, work)?;
    ensure!(
        neighbors == [16, 217],
        "Original source neighbor reference mismatch"
    );
    let secondary: BTreeSet<_> = tokens
        .iter()
        .filter(|t| t.0 == 217 || t.1 == 217)
        .cloned()
        .collect();
    ensure!(
        secondary == ts,
        "Independent source patch reference mismatch"
    );
    events.emit(json!({"kind":"source_complete","tokens":tokens,"neighbor_labels":neighbors,"patch_node_visits":work.nodes,"patch_leaf_tests":work.leaves}))?;
    let mut last = None;
    let mut valid_count = 0usize;
    let mut regions: BTreeMap<String, usize> = [
        "A_patch_0_0.25",
        "A_patch_0.25_0.5",
        "A_patch_0.5_0.75",
        "A_patch_0.75_1",
        "A_patch_complete",
        "B",
        "other_contact",
        "unbound",
    ]
    .into_iter()
    .map(|s| (s.to_owned(), 0))
    .collect();
    let mut uncertain_sum = 0.;
    for (line_index, line) in BufReader::new(bank.as_slice()).lines().enumerate() {
        let line = line?;
        ensure!(line_index < cfg.expected_records, "Extra bank row");
        let input: BankRow = serde_json::from_str(&line)?;
        events.emit(json!({"kind":"candidate_begun","ordinal":line_index,"input":input}))?;
        validate_row(&input, line_index, last, &cfg)?;
        last = Some(input.event_index);
        work.begin()?;
        let den = density(
            &proposer,
            input.proposed_pose,
            anchor,
            half,
            input.saved_log_g,
        )?;
        events.emit(json!({"kind":"density_complete","ordinal":line_index,"density":den}))?;
        let (wall, core) = geo.validity(input.proposed_pose, work)?;
        ensure!(
            wall == input.wall_valid && core == input.core_valid,
            "Saved hard verdict mismatch at {line_index}"
        );
        let valid = wall && core == Some(true);
        events.emit(json!({"kind":"geometry_complete","ordinal":line_index,"wall_valid":wall,"core_valid":core}))?;
        let mut patch_result = Value::Null;
        let mut region = Value::Null;
        let mut envelope = Value::Null;
        if valid {
            valid_count += 1;
            let (tokens, neighbors, indices) = geo.patches(input.proposed_pose, work)?;
            let (name, data) = region_data(&tokens, &neighbors, &cfg.regions)?;
            *regions.entry(name.clone()).or_default() += 1;
            region = json!(name);
            patch_result = data;
            events.emit(json!({"kind":"patches_complete","ordinal":line_index,"patches":patch_result,"region":region}))?;
            let labels: Vec<_> = indices
                .iter()
                .map(|&i| geo.context.bodies[i].label)
                .collect();
            let env = Environment {
                tree: &geo.tree,
                fixed: indices.iter().map(|&i| geo.fixed[i]).collect(),
                labels: labels.iter().map(|&i| (i, [0; 3])).collect(),
                rd: geo.rd,
            };
            work.check()?;
            let e = OverlapEnvelope::build(&env, input.proposed_pose, cfg.envelope)?;
            work.check()?;
            uncertain_sum += e.uncertain_volume;
            ensure!(uncertain_sum.is_finite(), "Envelope total overflow");
            envelope = json!({"lower_volume":e.lower_volume,"uncertain_volume":e.uncertain_volume,"upper_volume":e.upper_volume(),"retained_cells":e.cells.len(),"created_cells":e.created,"certified_cells":e.certified_cells,"fixed_labels":labels});
            events.emit(
                json!({"kind":"envelope_complete","ordinal":line_index,"envelope":envelope}),
            )?;
        }
        let output = json!({"kind":"candidate","complete":true,"input":input,"density":den,"actual":{"wall_valid":wall,"core_valid":core,"physical_valid":valid,"verdict_matches":true},"patches":patch_result,"region":region,"envelope":envelope,"work":{"patch_node_visits":work.nodes,"patch_leaf_tests":work.leaves},"physical_zero":!valid});
        serde_json::to_writer(&mut *rows, &output)?;
        rows.write_all(b"\n")?;
        rows.flush()?;
        *completed += 1;
        events.emit(json!({"kind":"candidate_complete","ordinal":line_index}))?;
    }
    ensure!(*completed == cfg.expected_records, "Missing bank rows");
    Ok(
        json!({"schema":SCHEMA,"complete":true,"passed":true,"attempted_records":*completed,"completed_records":*completed,"physical_valid_records":valid_count,"physical_zero_records":*completed-valid_count,"regions":regions,"uncertain_volume_sum":uncertain_sum,"prospective_two_cloud_expected_raw_points":4.48*uncertain_sum,"clouds_generated":0,"new_poses_generated":0,"source_reference_checked":true,"physical_weight_status":"not_estimated","coverage_scope":"Region hit inventory is diagnostic; zero hits do not establish a physical upper bound."}),
    )
}
fn main() -> Result<()> {
    let args = Args::parse();
    fs::create_dir(&args.out).context("Output must be fresh")?;
    let cfg: Config = serde_json::from_slice(&fs::read(&args.config)?)?;
    let mut work = Work::new(cfg.limits.clone())?;
    let mut events = Journal::new(&args.out.join("events.jsonl"))?;
    let mut rows = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(args.out.join("rows.jsonl"))?,
    );
    let mut completed = 0;
    let result = run(&args, &mut events, &mut rows, &mut completed, &mut work);
    let mut summary = match &result {
        Ok(v) => v.clone(),
        Err(e) => {
            let _ = events.emit(
                json!({"kind":"fatal","reason":format!("{e:#}"),"completed_records":completed}),
            );
            json!({"schema":SCHEMA,"complete":false,"passed":false,"completed_records":completed,"error":format!("{e:#}"),"clouds_generated":0,"new_poses_generated":0})
        }
    };
    summary["cpu_seconds"] = json!(cpu_seconds() - work.started_cpu);
    summary["wall_seconds"] = json!(work.started_wall.elapsed().as_secs_f64());
    summary["patch_node_visits"] = json!(work.total_nodes);
    summary["patch_leaf_tests"] = json!(work.total_leaves);
    summary["journal_events"] = json!(events.next);
    json_file(&args.out.join("summary.json"), &summary)?;
    result.map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn id() -> Pose {
        Pose {
            position: [0.; 3],
            orientation: [1., 0., 0., 0.],
        }
    }
    fn config() -> Config {
        Config {
            schema: SCHEMA.into(),
            invocation_config: Asset {
                path: "/toy".into(),
                sha256: "x".into(),
            },
            model: Asset {
                path: "/toy".into(),
                sha256: "x".into(),
            },
            prior: None,
            patch_map: Asset {
                path: "/toy".into(),
                sha256: "x".into(),
            },
            bank_sha256: "x".into(),
            identity: json!({"test":true}),
            expected_records: 2304,
            expected_virtual_branches: 2048,
            regions: Regions {
                a_neighbors: vec![16, 217],
                b_neighbors: vec![16, 56],
                secondary_label: 217,
                source_secondary_tokens: (0..16)
                    .map(|i| (77, 217, format!("p{i:02}"), "q".into()))
                    .collect(),
                inclusion_boundaries: vec![0., 0.25, 0.5, 0.75, 1.],
            },
            envelope: GateOptions {
                max_cells: 255,
                max_depth: 8,
                min_width: 0.,
            },
            limits: Limits {
                cpu_seconds: 30.,
                wall_seconds: 60.,
                patch_node_visits_per_candidate: 100,
                patch_leaf_tests_per_candidate: 100,
                patch_node_visits_total: 1000,
                patch_leaf_tests_total: 1000,
            },
        }
    }
    fn model() -> FrozenRelativePoseProposal {
        let covariance: Vec<Vec<f64>> = (0..6)
            .map(|i| (0..6).map(|j| if i == j { 1. } else { 0. }).collect())
            .collect();
        let value = json!({"schema":"reciprocal-pose-mixture-v1","base_model":{"angular_length":2.,"anchors":[{"position":[0.,0.,0.],"rotation":IDENTITY}],"means":[[0.,0.,0.,0.,0.,0.]],"covariances":[covariance],"weights":[1.],"shape_sha256":"0000000000000000000000000000000000000000000000000000000000000000","coordinate_convention":"anchor-body-relative"},"reciprocal_components":[true]});
        FrozenRelativePoseProposal::from_json_str_open(
            &value.to_string(),
            [20.; 3],
            0.5,
            "0000000000000000000000000000000000000000000000000000000000000000",
        )
        .unwrap()
    }
    #[test]
    fn all_bank_ordinals_and_hard_invalid_zeros_remain() {
        let cfg = config();
        let mut previous = None;
        let mut zero = 0;
        for i in 0..2304 {
            let row = BankRow {
                ordinal: i,
                cycle: i as u64 + 1,
                slot: 4,
                attempt_index: 5 * i as u64 + 4,
                event_index: 17 + 16 * i as u64,
                identity: cfg.identity.clone(),
                proposed_pose: id(),
                branch: "uniform".into(),
                status: "core_rejected".into(),
                wall_valid: true,
                core_valid: Some(false),
                saved_log_g: None,
                production: i >= 256,
                accepted: false,
            };
            validate_row(&row, i, previous, &cfg).unwrap();
            previous = Some(row.event_index);
            zero += usize::from(row.core_valid == Some(false));
        }
        assert_eq!(zero, 2304);
    }
    #[test]
    fn bank_rejects_missing_duplicate_or_wrong_warmup() {
        let cfg = config();
        let mut row = BankRow {
            ordinal: 0,
            cycle: 1,
            slot: 4,
            attempt_index: 4,
            event_index: 17,
            identity: cfg.identity.clone(),
            proposed_pose: id(),
            branch: "involution".into(),
            status: "wall_rejected".into(),
            wall_valid: false,
            core_valid: None,
            saved_log_g: Some(0.),
            production: false,
            accepted: false,
        };
        validate_row(&row, 0, None, &cfg).unwrap();
        assert!(validate_row(&row, 1, None, &cfg).is_err());
        assert!(validate_row(&row, 0, Some(17), &cfg).is_err());
        row.production = true;
        assert!(validate_row(&row, 0, None, &cfg).is_err());
    }
    #[test]
    fn source_patch_quarter_bins_and_full_complement() {
        let cfg = config();
        for (n, name) in [
            (0, "A_patch_0_0.25"),
            (3, "A_patch_0_0.25"),
            (4, "A_patch_0.25_0.5"),
            (8, "A_patch_0.5_0.75"),
            (12, "A_patch_0.75_1"),
            (16, "A_patch_complete"),
        ] {
            let tokens = cfg.regions.source_secondary_tokens[..n]
                .iter()
                .cloned()
                .collect();
            let (name_actual, d) = region_data(&tokens, &[16, 217], &cfg.regions).unwrap();
            assert_eq!(name_actual, name);
            assert_eq!(d["source_intersection"], n);
            assert_eq!(d["source_union"], 16);
        }
        for (labels, name) in [
            (vec![16, 56], "B"),
            (vec![16], "other_contact"),
            (vec![], "unbound"),
        ] {
            assert_eq!(
                region_data(&BTreeSet::new(), &labels, &cfg.regions)
                    .unwrap()
                    .0,
                name
            );
        }
    }
    #[test]
    fn exact_full_q_uniform_cube_and_cayley_seam() {
        let proposal =
            DockingProposal::new(model(), DockingMethod::PosteriorInvolution, 0., [0.; 3]).unwrap();
        let g = learned(&proposal, id(), id()).unwrap();
        let q = density(&proposal, id(), id(), 10., Some(g)).unwrap();
        let expected = (0.5 * g.exp() + 0.5 / 8000.).ln();
        assert!((q["log_q"].as_f64().unwrap() - expected).abs() < 1e-13);
        let seam = Pose {
            orientation: [0., 1., 0., 0.],
            ..id()
        };
        let q = density(&proposal, seam, id(), 10., None).unwrap();
        assert_eq!(q["log_g_status"], "negative_infinity");
        assert!((q["log_q"].as_f64().unwrap() + 16000f64.ln()).abs() < 1e-13);
        let edge = Pose {
            position: [10., 0., 0.],
            ..id()
        };
        assert_eq!(
            density(&proposal, edge, id(), 10., None).unwrap()["uniform_contains"],
            true
        );
        let outside = Pose {
            position: [10. + 1e-12, 0., 0.],
            ..id()
        };
        assert_eq!(
            density(&proposal, outside, id(), 10., None).unwrap()["uniform_contains"],
            false
        );
    }
    #[test]
    fn normalization_haar_jacobian_and_manifest_factor() {
        let m = model();
        let p = m.component_parameters()[0].clone();
        let l = reconstructed_map_lower(&p).unwrap();
        let proposal =
            DockingProposal::new(m, DockingMethod::PosteriorInvolution, 0., [0.; 3]).unwrap();
        let (map, _, _) = proposal.member_chart_parts();
        let z = [0.4, -0.2, 0.8, 0.3, -0.7, 0.1];
        let decoded = map.decode(0, z).unwrap();
        let encoded = map.encode(0, decoded).unwrap();
        assert!(encoded.iter().zip(z).all(|(x, y)| (x - y).abs() < 1e-13));
        let u: Vec3 = std::array::from_fn(|i| z[i + 3] / 2.);
        let log_j = -3. * 2f64.ln() - 2. * std::f64::consts::PI.ln() - 2. * (1. + dot(u, u)).ln();
        let gaussian =
            -3. * (2. * std::f64::consts::PI).ln() - 0.5 * z.iter().map(|x| x * x).sum::<f64>();
        assert!((map.checked_log_density(0, decoded).unwrap() + log_j - gaussian).abs() < 1e-12);
        for i in 0..6 {
            for j in 0..6 {
                let c = (0..6).map(|k| l[i][k] * l[j][k]).sum::<f64>();
                assert!((c - p.covariance[i][j]).abs() < 1e-13);
            }
        }
    }
    #[test]
    fn noncommuting_anchor_relative_roundtrip() {
        let anchor = Pose {
            position: [2., -3., 1.],
            orientation: quaternion(cayley([0.1, 0.4, -0.2])),
        };
        let local = Pose {
            position: [-0.7, 0.3, 0.9],
            orientation: quaternion(cayley([-0.3, 0.2, 0.7])),
        };
        let global = Pose {
            position: anchor.apply(local.position),
            orientation: quaternion(matmul(
                rotation(anchor.orientation),
                rotation(local.orientation),
            )),
        };
        let recovered = relative(global, anchor);
        assert!(norm(sub(recovered.position, local.position)) < 1e-13);
        for (i, row) in rotation(recovered.orientation).iter().enumerate() {
            for (j, x) in row.iter().enumerate() {
                assert!((*x - rotation(local.orientation)[i][j]).abs() < 1e-13);
            }
        }
    }
    #[test]
    fn journal_flush_preserves_failure_prefix() {
        let dir = std::env::temp_dir().join(format!("bank-prefix-{}", std::process::id()));
        fs::create_dir(&dir).unwrap();
        let result = (|| -> Result<()> {
            let mut j = Journal::new(&dir.join("events.jsonl"))?;
            j.emit(json!({"kind":"candidate_begun","ordinal":0}))?;
            j.emit(json!({"kind":"density_complete","ordinal":0}))?;
            let failure = log_add(f64::NAN, 0.);
            assert!(failure.is_err());
            j.emit(json!({"kind":"fatal","completed_records":0}))?;
            let lines = fs::read_to_string(dir.join("events.jsonl"))?;
            let v: Vec<Value> = lines
                .lines()
                .map(serde_json::from_str)
                .collect::<std::result::Result<_, _>>()?;
            assert_eq!(v.len(), 3);
            assert_eq!(v[0]["event_index"], 0);
            assert_eq!(v[2]["kind"], "fatal");
            Ok(())
        })();
        fs::remove_dir_all(&dir).unwrap();
        result.unwrap();
    }
}
