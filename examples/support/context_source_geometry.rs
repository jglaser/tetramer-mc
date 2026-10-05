//! Source-guide asset/geometry bridge, forked from the frozen bank preflight
//! (8268c202fc9835309708aa2fa3a97e02ff236d4eb2cf8c0aaaba7ae297a87c58).
//! Its saved-data implementation remains unchanged. No optimizer or native classifier.
#[path = "context_bank_geometry.rs"]
pub mod patch_geometry;
use anyhow::{Context, Result, ensure};
use patch_geometry::{PatchBvh, Token, Work, World, near};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};
use tetramer_mc::{
    basin_involution::{BasinPair, FixedBasinInvolution},
    context_docking::{ContextBody, ContextDockingConfig, uniform_half_width},
    docking,
    geometry::{Placed, Shape, SphereTree},
    math::*,
    proposal::{FrozenRelativePoseProposal, GaussianComponentParameters, RelativePoseBranch},
    spherical::Container,
};
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Asset {
    pub path: PathBuf,
    pub sha256: String,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Regions {
    pub a_neighbors: Vec<usize>,
    pub b_neighbors: Vec<usize>,
    pub secondary_label: usize,
    pub source_secondary_tokens: Vec<Token>,
    pub inclusion_boundaries: Vec<f64>,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Inputs {
    pub invocation_config: Asset,
    pub model: Asset,
    pub prior: Asset,
    pub patch_map: Asset,
    pub expected_virtual_branches: usize,
    pub regions: Regions,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixedContext {
    pub schema: String,
    pub anchor_label: usize,
    pub excluded_moving_labels: Vec<usize>,
    pub bodies: Vec<ContextBody>,
}
#[derive(Deserialize)]
pub struct SourceState {
    pub schema: String,
    pub moving_label: usize,
    pub pose: Pose,
    pub anchor_label: usize,
    pub anchor_pose: Pose,
    pub boundary: String,
    pub spherical_wall_radius: f64,
    pub coordinate_frame: String,
    pub endpoint_sha256: String,
}
#[derive(Deserialize)]
pub struct PatchMap {
    pub schema: String,
    pub shape_sha256: String,
    pub atom_patch_ids: Vec<String>,
}
#[derive(Deserialize)]
pub struct PriorBranch {
    pub virtual_label: usize,
    pub component_index: usize,
    pub inverted: bool,
    pub original_probability: f64,
    pub original_log_probability: f64,
    pub eligible: bool,
    pub log_probability: f64,
    pub probability: f64,
}
#[derive(Deserialize)]
pub struct PriorAsset {
    pub schema: String,
    pub complete: bool,
    pub passed: bool,
    pub method: String,
    pub moving_labels: Vec<usize>,
    pub anchor_label: usize,
    pub floor_probability: f64,
    pub virtual_branch_count: usize,
    pub branches: Vec<PriorBranch>,
    pub log_prior: Vec<f64>,
    pub base_log_prior: Vec<f64>,
    pub eligibility: Vec<bool>,
    pub input_sha256: BTreeMap<String, String>,
    pub fixed_context_sha256: String,
    pub source_state_sha256: String,
}
pub fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub fn checked_bytes(path: &Path, expected: &str) -> Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    ensure!(sha(&bytes) == expected, "Hash mismatch: {}", path.display());
    Ok(bytes)
}
pub fn asset(a: &Asset) -> Result<Vec<u8>> {
    ensure!(a.path.is_absolute(), "Asset paths must be absolute");
    checked_bytes(&a.path, &a.sha256)
}
pub fn close(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= 2e-12 * (1. + b.abs())
}
pub fn prior_validate(
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
pub fn relative(pose: Pose, anchor: Pose) -> Pose {
    let rt = transpose(rotation(anchor.orientation));
    Pose {
        position: matvec(rt, sub(pose.position, anchor.position)),
        orientation: quaternion(matmul(rt, rotation(pose.orientation))),
    }
}
pub fn log_add(a: f64, b: f64) -> Result<f64> {
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
/// Manifest-only arithmetic reconstruction; density above always uses the actual compiled map.
pub fn reconstructed_map_lower(p: &GaussianComponentParameters) -> Result<[[f64; 6]; 6]> {
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
pub struct Geometry {
    pub tree: SphereTree,
    pub contact: SphereTree,
    pub wall: Container,
    pub patch: PatchBvh,
    pub worlds: Vec<World>,
    pub fixed: Vec<Placed>,
    pub context: FixedContext,
    pub source: SourceState,
    pub rd: f64,
}
impl Geometry {
    pub fn validity(&self, pose: Pose, work: &Work) -> Result<(bool, Option<bool>)> {
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
    pub fn patches(
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
pub fn region_data(
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

pub struct Loaded {
    pub geometry: Geometry,
    pub atlas: FixedBasinInvolution,
    pub flags: Vec<bool>,
    pub log_prior: Vec<f64>,
    pub half: f64,
    pub angular_length: f64,
    pub manifest: Value,
    pub input_hashes: BTreeMap<String, String>,
    pub z: f64,
    pub lambda: f64,
}

pub fn load(cfg: &Inputs) -> Result<Loaded> {
    ensure!(
        cfg.expected_virtual_branches == 2048,
        "Wrong atlas allocation"
    );
    ensure!(
        cfg.regions.a_neighbors == [16, 217]
            && cfg.regions.b_neighbors == [16, 56]
            && cfg.regions.secondary_label == 217
            && cfg.regions.inclusion_boundaries == [0., 0.25, 0.5, 0.75, 1.],
        "Changed regions"
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
        "Changed sourceT reference"
    );
    let invocation: ContextDockingConfig = serde_json::from_slice(&asset(&cfg.invocation_config)?)?;
    ensure!(
        invocation.depletant_radius == 1.5
            && invocation.reservoir_density == 0.035
            && invocation.poisson_lambda_ratio == 64.,
        "Changed physical law"
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

    let a: PriorAsset = serde_json::from_slice(&asset(&cfg.prior)?)?;
    prior_validate(&a, &branches, &invocation, &context, &source)?;
    let expanded: Vec<_> = branches
        .iter()
        .map(|b| {
            let mut p = parameters[b.component_index].clone();
            p.weight = b.weight;
            p
        })
        .collect();
    let angular_length = model.angular_length();
    // Pair support is unused: this tool only calls decode/checked_log_density.
    // The exact same exported parameters and map Cholesky are used as production.
    let atlas = FixedBasinInvolution::new(
        expanded.clone(),
        angular_length,
        0.,
        vec![BasinPair {
            first: 0,
            second: 0,
            weight: 1.,
        }],
    )?;
    let flags: Vec<_> = branches.iter().map(|b| b.inverted).collect();
    let priors = a.log_prior;
    let charts:Vec<_>=branches.iter().enumerate().map(|(i,b)|Ok(json!({
  "virtual_label":i,"component_index":b.component_index,"inverted":b.inverted,
  "original_probability":b.weight,"effective_log_prior":priors[i],
  "parameters":expanded[i],"reconstructed_map_lower":reconstructed_map_lower(&expanded[i])?
 }))).collect::<Result<_>>()?;
    let manifest = json!({"schema":"context-source-guide-charts-v1","angular_length":angular_length,
  "uniform_half_width":half,"mixture":[0.5,0.25,0.25],
  "density_measure":"translation volume times normalized rotational Haar",
  "anchor_pose":anchor,"charts":charts,"map_pair_support":"one unused selfpair; decode and density only"});
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

    let mut input_hashes = BTreeMap::new();
    for a in [
        &cfg.invocation_config,
        &cfg.model,
        &cfg.prior,
        &cfg.patch_map,
    ] {
        input_hashes.insert(
            a.path.canonicalize()?.display().to_string(),
            a.sha256.clone(),
        );
    }
    for (path, digest) in [
        (
            resolve(&invocation.shape),
            &invocation.expected_sha256.shape,
        ),
        (
            resolve(&invocation.fixed_context),
            &invocation.expected_sha256.fixed_context,
        ),
        (
            resolve(&invocation.source_state),
            &invocation.expected_sha256.source_state,
        ),
    ] {
        input_hashes.insert(path.canonicalize()?.display().to_string(), digest.clone());
    }
    Ok(Loaded {
        geometry: geo,
        atlas,
        flags,
        log_prior: priors,
        half,
        angular_length,
        manifest,
        input_hashes,
        z: invocation.reservoir_density,
        lambda: invocation.reservoir_density * invocation.poisson_lambda_ratio,
    })
}
